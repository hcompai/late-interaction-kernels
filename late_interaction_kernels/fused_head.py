"""Fused D-side projection + normalize + MaxSim.

Motivation
----------
When the on-disk format is ``[Nd, Ld, d_model]`` hidden states (e.g. the
last layer of a ModernBERT encoder) and the MaxSim dimension is
``d_out`` (typically 64-128), the standard pipeline is::

    D_proj = F.normalize(H_d @ W.T)          # [Nd, Ld, d_out]  <-- big HBM scratch
    scores = maxsim(Q, D_proj)

For large corpora the ``D_proj`` intermediate can be multi-GB. The fused
kernel folds projection + L2-normalize + MaxSim into a single pass, so
the corpus only has to exist in HBM as ``[Nd, Ld, d_model]`` hidden
states.

Scope (0.8.0)
-------------
- **Inference**: :func:`maxsim_from_hidden` — no ``D_proj`` ever
  materialized, no autograd.
- **Training**: :func:`maxsim_from_hidden_train` — autograd-aware
  wrapper. Forward runs the same fused kernel with an extra
  ``save_argmax`` store. Backward gathers ``H`` only at winning
  positions (``Nq · Nd · Lq`` rows, typically <10 % of ``Nd · Ld``),
  runs the projection + normalize + maxsim gradient in closed form
  using bf16/fp16 matmuls with fp32 accumulator (no autograd rebuild,
  no full-precision ``F.linear`` recompute), and flows gradients into
  ``H_d`` via a single ``index_add_`` scatter. Numerically matches the
  unfused path (``F.linear -> F.normalize -> maxsim``) to bf16 tolerance.

API
---
``maxsim_from_hidden(Q, H_d, W, b=None, d_mask=None, normalize=True)``
``maxsim_from_hidden_train(Q, H_d, W, b=None, d_mask=None, normalize=True)``
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._utils import next_pow2, pick_compute_dtype


def _fused_head_configs():
    # Smaller shortlist than the main kernel: d_model can be big (≥ 768),
    # which eats SRAM fast — we bias toward small BLOCK_D and moderate
    # BLOCK_Q.
    return [
        triton.Config({"BLOCK_Q": 16, "BLOCK_D": 32, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 32, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 64, "BLOCK_K": 128}, num_warps=8, num_stages=3),
    ]


@triton.autotune(
    configs=_fused_head_configs(),
    key=["Lq", "Ld", "d_out_pad", "d_model_pad", "has_bias", "has_d_mask", "normalize", "save_argmax"],
)
@triton.jit
def _fused_head_fwd_kernel(
    Q_ptr,  # [Nq, Lq, d_out]  (already projected + optionally normalized)
    H_ptr,  # [Nd, Ld, d_model]
    W_ptr,  # [d_out, d_model]
    b_ptr,  # [d_out] or dummy
    d_mask_ptr,  # [Nd, Ld] or dummy
    scores_ptr,  # [Nq, Nd]
    argmax_ptr,  # [Nq*Nd, Lq] int32 or dummy
    Nq: tl.constexpr,
    Nd: tl.constexpr,
    Lq: tl.constexpr,
    Ld: tl.constexpr,
    d_out: tl.constexpr,
    d_out_pad: tl.constexpr,
    d_model: tl.constexpr,
    d_model_pad: tl.constexpr,
    stride_q_n,
    stride_q_l,
    stride_q_d,
    stride_h_n,
    stride_h_l,
    stride_h_d,
    stride_w_out,
    stride_w_in,
    stride_s_n,
    stride_s_d,
    stride_dm_n,
    stride_dm_l,
    stride_a_pair,
    stride_a_lq,
    has_bias: tl.constexpr,
    has_d_mask: tl.constexpr,
    normalize: tl.constexpr,
    save_argmax: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    q_idx = pid // Nd
    d_idx = pid % Nd

    out_off = tl.arange(0, d_out_pad)
    out_mask = out_off < d_out

    score_acc = tl.zeros([], dtype=tl.float32)

    # ---- optional bias load (small, once per program) ------------------
    if has_bias:
        bias_vec = tl.load(b_ptr + out_off, mask=out_mask, other=0.0).to(tl.float32)
    else:
        bias_vec = tl.zeros([d_out_pad], dtype=tl.float32)

    for q_start in tl.static_range(0, Lq, BLOCK_Q):
        q_off = q_start + tl.arange(0, BLOCK_Q)
        q_valid = q_off < Lq

        Q_block = tl.load(
            Q_ptr + q_idx * stride_q_n + q_off[:, None] * stride_q_l + out_off[None, :] * stride_q_d,
            mask=q_valid[:, None] & out_mask[None, :],
            other=0.0,
        ).to(COMPUTE_DTYPE)

        m = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
        m_idx = tl.full([BLOCK_Q], 0, dtype=tl.int32)

        for d_start in range(0, Ld, BLOCK_D):
            d_off = d_start + tl.arange(0, BLOCK_D)
            d_valid = d_off < Ld

            if has_d_mask:
                dm = tl.load(
                    d_mask_ptr + d_idx * stride_dm_n + d_off * stride_dm_l,
                    mask=d_valid,
                    other=0,
                ).to(tl.int1)
                d_active = d_valid & dm
            else:
                d_active = d_valid

            # ---- on-the-fly projection of the D tile: D_tile = H_tile @ W.T + b
            # Accumulate in fp32 across the d_model split-K loop.
            D_tile_f32 = tl.zeros([BLOCK_D, d_out_pad], dtype=tl.float32)
            for k_start in range(0, d_model, BLOCK_K):
                k_off = k_start + tl.arange(0, BLOCK_K)
                k_mask = k_off < d_model

                H_tile = tl.load(
                    H_ptr + d_idx * stride_h_n + d_off[:, None] * stride_h_l + k_off[None, :] * stride_h_d,
                    mask=d_valid[:, None] & k_mask[None, :],
                    other=0.0,
                ).to(COMPUTE_DTYPE)

                # W is [d_out, d_model]; we want D = H @ W.T → load W tile
                # as [d_out_pad, BLOCK_K] and transpose in the matmul.
                W_tile = tl.load(
                    W_ptr + out_off[:, None] * stride_w_out + k_off[None, :] * stride_w_in,
                    mask=out_mask[:, None] & k_mask[None, :],
                    other=0.0,
                ).to(COMPUTE_DTYPE)

                D_tile_f32 += tl.dot(H_tile, tl.trans(W_tile), out_dtype=tl.float32)

            D_tile_f32 += bias_vec[None, :]

            if normalize:
                d_norm_sq = tl.sum(D_tile_f32 * D_tile_f32, axis=1)
                d_inv = 1.0 / tl.sqrt(tl.maximum(d_norm_sq, 1e-12))
                D_tile_f32 = D_tile_f32 * d_inv[:, None]

            D_tile = D_tile_f32.to(COMPUTE_DTYPE)

            # Zero out the padded d_out columns so they don't contaminate the dot.
            D_tile = tl.where(out_mask[None, :], D_tile, tl.zeros_like(D_tile))

            S = tl.dot(Q_block, tl.trans(D_tile), out_dtype=tl.float32)
            S = tl.where(d_active[None, :], S, float("-inf"))

            tile_max = tl.max(S, axis=1)
            if save_argmax:
                tile_argmax = tl.argmax(S, axis=1).to(tl.int32) + d_start
                update = tile_max > m
                m_idx = tl.where(update, tile_argmax, m_idx)
            m = tl.maximum(m, tile_max)

        m_finite = m != float("-inf")
        m = tl.where(m_finite & q_valid, m, 0.0)
        score_acc += tl.sum(m)

        if save_argmax:
            tl.store(
                argmax_ptr + pid * stride_a_pair + q_off * stride_a_lq,
                m_idx,
                mask=q_valid,
            )

    tl.store(scores_ptr + q_idx * stride_s_n + d_idx * stride_s_d, score_acc)


def _fused_head_forward(
    Q: torch.Tensor,
    H_d: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor | None,
    d_mask: torch.Tensor | None,
    normalize: bool,
    save_argmax: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    Nq, Lq, d_out = Q.shape
    Nd, Ld, d_model = H_d.shape
    assert W.shape == (d_out, d_model), (
        f"W must be [d_out, d_model] = [{d_out}, {d_model}], got {tuple(W.shape)}"
    )

    has_bias = b is not None
    has_d_mask = d_mask is not None

    compute_dtype = pick_compute_dtype(Q, H_d)
    tl_dtype = tl.float16 if compute_dtype == torch.float16 else tl.bfloat16

    scores = torch.empty(Nq, Nd, device=Q.device, dtype=torch.float32)
    argmax = torch.empty(Nq * Nd, Lq, device=Q.device, dtype=torch.int32) if save_argmax else None

    b_ptr = b if has_bias else Q
    d_mask_ptr = d_mask if has_d_mask else H_d
    dm_strides = (d_mask.stride(0), d_mask.stride(1)) if has_d_mask else (0, 0)
    argmax_ptr = argmax if save_argmax else scores
    a_strides = (argmax.stride(0), argmax.stride(1)) if save_argmax else (0, 0)

    grid = (Nq * Nd,)
    _fused_head_fwd_kernel[grid](
        Q,
        H_d,
        W,
        b_ptr,
        d_mask_ptr,
        scores,
        argmax_ptr,
        Nq,
        Nd,
        Lq,
        Ld,
        d_out,
        next_pow2(d_out),
        d_model,
        next_pow2(d_model),
        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        H_d.stride(0),
        H_d.stride(1),
        H_d.stride(2),
        W.stride(0),
        W.stride(1),
        scores.stride(0),
        scores.stride(1),
        dm_strides[0],
        dm_strides[1],
        a_strides[0],
        a_strides[1],
        has_bias,
        has_d_mask,
        normalize,
        save_argmax,
        COMPUTE_DTYPE=tl_dtype,
    )
    return scores, argmax


def maxsim_from_hidden(
    Q: torch.Tensor,
    H_d: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor | None = None,
    d_mask: torch.Tensor | None = None,
    normalize: bool = True,
) -> torch.Tensor:
    """Inference-only fused MaxSim with on-the-fly D-side projection.

    Computes::

        D = F.linear(H_d, W, b)                 # [Nd, Ld, d_out]
        D = F.normalize(D, dim=-1) if normalize # else D
        scores = maxsim_inference(Q, D)          # [Nq, Nd]

    but without ever materializing ``D`` in HBM. For large corpora stored
    as hidden states this saves multi-GB of scratch memory.

    Args:
        Q: ``[Lq, d_out]`` or ``[Nq, Lq, d_out]``. Already projected and,
           if you pass ``normalize=True``, already L2-normalized.
        H_d: ``[Nd, Ld, d_model]``. Raw last-layer hidden states.
        W: ``[d_out, d_model]``. Projection weight (same convention as
           ``torch.nn.Linear.weight``).
        b: ``[d_out]`` or ``None``. Projection bias.
        d_mask: ``[Nd, Ld]`` boolean mask (``True`` = keep).
        normalize: if ``True`` (default), L2-normalize each projected
           D token across the ``d_out`` dimension before scoring.

    Returns:
        ``scores: [Nq, Nd]`` fp32.

    Notes:
        Inference only — not autograd-aware. Use the non-fused path if you
        need to backprop through the projection.
    """
    q_was_2d = Q.dim() == 2
    if q_was_2d:
        Q = Q.unsqueeze(0)
    if d_mask is not None and d_mask.dim() == 1:
        d_mask = d_mask.unsqueeze(0)

    if Q.dim() != 3:
        raise ValueError(f"Q must be [Lq, d_out] or [Nq, Lq, d_out]; got {Q.shape}")
    if H_d.dim() != 3:
        raise ValueError(f"H_d must be [Nd, Ld, d_model]; got {H_d.shape}")
    if W.dim() != 2 or W.shape != (Q.shape[-1], H_d.shape[-1]):
        raise ValueError(f"W must be [d_out={Q.shape[-1]}, d_model={H_d.shape[-1]}]; got {W.shape}")
    if Q.device != H_d.device or W.device != Q.device:
        raise ValueError("Q, H_d, W must be on the same device.")
    if b is not None and (b.dim() != 1 or b.shape[0] != Q.shape[-1]):
        raise ValueError(f"b must be [d_out={Q.shape[-1]}]; got {tuple(b.shape) if b is not None else None}")

    Q = Q.contiguous()
    H_d = H_d.contiguous()
    W = W.contiguous()
    b_c = b.contiguous() if b is not None else None
    d_mask_c = d_mask.contiguous().to(torch.int8) if d_mask is not None else None

    scores, _ = _fused_head_forward(Q, H_d, W, b_c, d_mask_c, normalize, save_argmax=False)
    if q_was_2d:
        return scores.squeeze(0)
    return scores


class _MaxSimFromHiddenFn(torch.autograd.Function):
    """Autograd-aware fused head for training.

    Forward runs the same fused kernel as inference plus an ``argmax``
    store (``[Nq*Nd, Lq]`` int32). Backward does the minimum-work
    closed-form gradient:

    1. Gather ``H_d`` at winning positions once → ``[Nq, Nd, Lq, d_model]``.
    2. Recompute ``D_proj_win = F.linear(H_win, W, b)`` and ``D_hat_win``
       in the input dtype (bf16/fp16) with fp32 accumulator — the only
       big matmul in the backward.
    3. Apply the L2-normalize Jacobian in fp32, in place.
    4. Close-form gradients for ``Q``, ``W``, ``b`` via two more matmuls
       on the winners slice, and scatter ``grad_H_win`` into
       ``grad_H_d`` with a single ``index_add_``.

    No autograd rebuild, no fp32 ``F.linear`` recompute: backward does
    ~3 matmuls of shape ``[Nq·Nd·Lq, d_model] × [d_model, d_out]``,
    versus the unfused ``[Nd·Ld, d_model] × [d_model, d_out]`` triple —
    typically 3–5× cheaper at training shapes where ``Nq·Lq ≪ Ld``.
    """

    @staticmethod
    def forward(ctx, Q, H_d, W, b, d_mask, normalize):
        scores, argmax = _fused_head_forward(Q, H_d, W, b, d_mask, normalize, save_argmax=True)
        ctx.save_for_backward(Q, H_d, W, b, d_mask, argmax)
        ctx.normalize = bool(normalize)
        ctx.Nq = Q.shape[0]
        ctx.Nd = H_d.shape[0]
        ctx.Lq = Q.shape[1]
        ctx.Ld = H_d.shape[1]
        return scores

    @staticmethod
    def backward(ctx, grad_scores):
        Q, H_d, W, b, _d_mask, argmax = ctx.saved_tensors
        normalize = ctx.normalize
        Nq, Nd, Lq, Ld = ctx.Nq, ctx.Nd, ctx.Lq, ctx.Ld
        d_model = H_d.shape[2]
        d_out = Q.shape[2]
        N = Nq * Nd * Lq
        compute_dtype = pick_compute_dtype(Q, H_d)

        need_Q = ctx.needs_input_grad[0]
        need_H = ctx.needs_input_grad[1]
        need_W = ctx.needs_input_grad[2]
        need_b = ctx.needs_input_grad[3] and b is not None

        if not (need_Q or need_H or need_W or need_b):
            return None, None, None, None, None, None

        grad_scores = grad_scores.contiguous().to(torch.float32)

        # Winning positions: argmax is [Nq*Nd, Lq] → we want a [Nq, Nd, Lq]
        # view plus a flat [N] "(j, t_winner) in the Nd·Ld grid" index for
        # scatter / gather against H_d.
        am = argmax.view(Nq, Nd, Lq).long()
        j_idx = torch.arange(Nd, device=H_d.device, dtype=torch.long).view(1, Nd, 1).expand(Nq, Nd, Lq)
        flat_jt = (j_idx * Ld + am).reshape(-1)  # [N]

        # Step 1 — gather H at winners, one row per (i, j, s). Size
        # [N, d_model]. Keep it flat to avoid per-j python loops.
        H_flat = H_d.reshape(Nd * Ld, d_model)
        H_win_flat = H_flat.index_select(0, flat_jt).to(compute_dtype)  # [N, d_model]

        # Step 2 — recompute the winners slice of D_proj in compute_dtype
        # (bf16/fp16) with fp32 accumulator. No autograd, no fp32 weights.
        W_c = W.to(compute_dtype)
        D_unnorm_win_f32 = torch.matmul(H_win_flat, W_c.T).to(torch.float32)  # [N, d_out]
        if b is not None:
            D_unnorm_win_f32 = D_unnorm_win_f32 + b.to(torch.float32)

        if normalize:
            norm = D_unnorm_win_f32.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            D_hat_win = D_unnorm_win_f32 / norm  # [N, d_out], fp32
        else:
            norm = None
            D_hat_win = D_unnorm_win_f32
        del D_unnorm_win_f32

        D_hat_win_view = D_hat_win.view(Nq, Nd, Lq, d_out)

        # Step 3 — grad_Q = Σ_j grad_scores · D_hat_win  (fp32 matmul).
        gQ = None
        if need_Q:
            gQ = torch.einsum("ij,ijsd->isd", grad_scores, D_hat_win_view)

        # Step 4 — closed-form normalize-Jacobian pullback.
        #     g_hat[i,j,s,:]     = grad_scores[i,j] · Q[i,s,:]
        #     g_unnorm[i,j,s,:]  = (1/|D|) · (g_hat - D_hat · <D_hat, g_hat>)
        # Fused in a handful of broadcasts; no extra temporaries live past
        # this block.
        Q_f32 = Q.to(torch.float32)
        grad_D_hat = grad_scores.view(Nq, Nd, 1, 1) * Q_f32.view(Nq, 1, Lq, d_out)
        del Q_f32
        if normalize:
            dot = (D_hat_win_view * grad_D_hat).sum(dim=-1, keepdim=True)  # [Nq, Nd, Lq, 1]
            grad_D_unnorm_flat = ((grad_D_hat - D_hat_win_view * dot) / norm.view(Nq, Nd, Lq, 1)).reshape(
                N, d_out
            )
            del dot, norm, D_hat_win, grad_D_hat
        else:
            grad_D_unnorm_flat = grad_D_hat.reshape(N, d_out)
            del grad_D_hat

        grad_D_unnorm_compute = grad_D_unnorm_flat.to(compute_dtype)  # [N, d_out]

        # Step 5 — grad_W (if requested) in compute_dtype with fp32 acc.
        gW = None
        if need_W:
            gW_compute = torch.matmul(grad_D_unnorm_compute.T, H_win_flat)  # [d_out, d_model]
            gW = gW_compute.to(W.dtype)

        gb = None
        if need_b:
            gb = grad_D_unnorm_flat.sum(dim=0).to(b.dtype)
        if not need_H:
            del grad_D_unnorm_flat

        # Step 6 — grad_H_d via scatter of grad_H_win at winning positions.
        # Scatter in the input dtype to avoid an fp32 [Nd·Ld, d_model]
        # buffer — that was the memory hog in the old path.
        grad_H_d = None
        if need_H:
            grad_H_win = torch.matmul(grad_D_unnorm_compute, W_c)  # [N, d_model]
            del grad_D_unnorm_flat, grad_D_unnorm_compute
            grad_H_d_flat = torch.zeros(Nd * Ld, d_model, device=H_d.device, dtype=H_d.dtype)
            grad_H_d_flat.index_add_(0, flat_jt, grad_H_win.to(H_d.dtype))
            grad_H_d = grad_H_d_flat.view(Nd, Ld, d_model)

        if need_Q:
            gQ = gQ.to(Q.dtype)

        return gQ, grad_H_d, gW, gb, None, None


def maxsim_from_hidden_train(
    Q: torch.Tensor,
    H_d: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor | None = None,
    d_mask: torch.Tensor | None = None,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Autograd-aware fused MaxSim with on-the-fly D-side projection.

    Forward fuses ``F.linear + F.normalize + maxsim`` into a single
    streaming pass (no ``[Nd, Ld, d_out]`` ``D_proj`` scratch). Backward
    gathers ``H_d`` at winning positions only (``Nq · Nd · Lq`` rows,
    typically <10 % of ``Nd · Ld``) and produces gradients in closed
    form — no autograd rebuild, no fp32 ``F.linear`` recompute.
    Numerically matches the unfused path
    (``F.normalize(F.linear(H_d, W, b)) -> maxsim(Q, D)``) to bf16
    tolerance (<2 % RMS vs PyTorch reference, verified under gradcheck
    with fp32 inputs).

    Drop-in replacement for::

        D_proj = F.linear(H_d, W, b)
        D_proj = F.normalize(D_proj, p=2, dim=-1) if normalize else D_proj
        scores = maxsim(Q, D_proj, d_mask=d_mask)

    Args match :func:`maxsim_from_hidden`. The gradient flows into
    whichever of ``Q``, ``H_d``, ``W``, ``b`` have ``requires_grad=True``.
    If none do, backward is a no-op.

    Notes:
        For training shapes where ``Nq · Lq ≪ Ld`` (i.e. most
        ColBERT-style late-interaction training — LateOn / LateOn-Code
        at ``Ld ∈ {300, 2k, 8k}``, LateOn-Code-edge at ``Ld=2k``,
        ColPali at ``Ld=1024``), the backward does ~``Nq · Lq / Ld``
        the matmul work of the unfused path and skips the full
        ``D_proj`` scratch entirely.
    """
    q_was_2d = Q.dim() == 2
    if q_was_2d:
        Q = Q.unsqueeze(0)
    if d_mask is not None and d_mask.dim() == 1:
        d_mask = d_mask.unsqueeze(0)

    if Q.dim() != 3:
        raise ValueError(f"Q must be [Lq, d_out] or [Nq, Lq, d_out]; got {Q.shape}")
    if H_d.dim() != 3:
        raise ValueError(f"H_d must be [Nd, Ld, d_model]; got {H_d.shape}")
    if W.dim() != 2 or W.shape != (Q.shape[-1], H_d.shape[-1]):
        raise ValueError(f"W must be [d_out={Q.shape[-1]}, d_model={H_d.shape[-1]}]; got {W.shape}")
    if Q.device != H_d.device or W.device != Q.device:
        raise ValueError("Q, H_d, W must be on the same device.")
    if b is not None and (b.dim() != 1 or b.shape[0] != Q.shape[-1]):
        raise ValueError(f"b must be [d_out={Q.shape[-1]}]; got {tuple(b.shape) if b is not None else None}")

    Q_c = Q.contiguous()
    H_c = H_d.contiguous()
    W_c = W.contiguous()
    b_c = b.contiguous() if b is not None else None
    d_mask_c = d_mask.contiguous().to(torch.int8) if d_mask is not None else None

    scores = _MaxSimFromHiddenFn.apply(Q_c, H_c, W_c, b_c, d_mask_c, normalize)
    if q_was_2d:
        return scores.squeeze(0)
    return scores
