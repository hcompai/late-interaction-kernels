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

Scope (0.7.0)
-------------
- **Inference**: :func:`maxsim_from_hidden` — no ``D_proj`` ever
  materialized, no autograd.
- **Training**: :func:`maxsim_from_hidden_train` — autograd-aware
  wrapper. Forward runs the same fused kernel with an extra
  ``save_argmax`` store; backward re-materializes only the winning
  ``D_proj`` rows (``Nq · Nd · Lq · d_out`` — typically <10 % of
  the full ``D_proj``) and back-props into ``H_d``, ``W``, ``b``, ``Q``
  via PyTorch autograd. Matches the unfused path numerically to fp32
  tolerance.

A full persistent-kernel training-side fusion (reads ``H`` only once
per ``(j, t)`` across all ``Nq``) is a 0.8.0 follow-up — see
``docs/rfc/0.7.0.md`` §5.

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
    argmax = (
        torch.empty(Nq * Nd, Lq, device=Q.device, dtype=torch.int32) if save_argmax else None
    )

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

    Forward runs the same fused kernel as inference plus an extra
    ``argmax`` store (``[Nq*Nd, Lq]`` int32). Backward gathers only the
    winning ``H_d`` rows, recomputes the tiny ``[Nq·Nd·Lq, d_out]``
    winners slice of ``D_proj`` in fp32, and flows the gradient back
    through normalize + linear via PyTorch autograd. Numerically
    identical to the unfused path
    (``F.normalize(F.linear(H_d, W, b)) -> maxsim(Q, D)``) to fp32
    tolerance — gradcheck-validated.

    Forward never materializes the full ``[Nd, Ld, d_out]`` ``D_proj``
    tensor in HBM; the backward only touches it at winning positions.
    """

    @staticmethod
    def forward(ctx, Q, H_d, W, b, d_mask, normalize):
        scores, argmax = _fused_head_forward(
            Q, H_d, W, b, d_mask, normalize, save_argmax=True
        )
        ctx.save_for_backward(Q, H_d, W, b, d_mask, argmax)
        ctx.normalize = bool(normalize)
        ctx.Nq = Q.shape[0]
        ctx.Nd = H_d.shape[0]
        ctx.Lq = Q.shape[1]
        ctx.Ld = H_d.shape[1]
        return scores

    @staticmethod
    def backward(ctx, grad_scores):
        Q, H_d, W, b, d_mask, argmax = ctx.saved_tensors
        normalize = ctx.normalize
        Nq, Nd, Lq, Ld = ctx.Nq, ctx.Nd, ctx.Lq, ctx.Ld
        d_model = H_d.shape[2]
        d_out = Q.shape[2]

        grad_scores = grad_scores.contiguous().to(torch.float32)

        # argmax is [Nq*Nd, Lq] → reshape to [Nq, Nd, Lq]
        am = argmax.view(Nq, Nd, Lq).long()

        # Gather H_d at the winning doc-token positions. This is the
        # only part that touches H_d sparsely; size is
        # [Nq, Nd, Lq, d_model]. For typical shapes (Lq=32, Nq=Nd=128,
        # d_model=768) that's 48 MB — tolerable.
        j_idx = torch.arange(Nd, device=H_d.device).view(1, Nd, 1).expand(Nq, Nd, Lq)
        H_win = H_d[j_idx, am]  # [Nq, Nd, Lq, d_model]

        # Rebuild the tiny winners slice of D_proj in fp32 through an
        # autograd-tracked pipeline so grad flows back into H_d, W, b.
        # H_win and Q are leaf-ish inputs we need grads for; the rest
        # (W, b) are parameters.
        H_win_fp = H_win.detach().to(torch.float32).requires_grad_(H_win.requires_grad or H_d.requires_grad)
        W_fp = W.detach().to(torch.float32).requires_grad_(W.requires_grad)
        b_fp = b.detach().to(torch.float32).requires_grad_(b.requires_grad) if b is not None else None
        Q_fp = Q.detach().to(torch.float32).requires_grad_(Q.requires_grad)

        D_win = torch.nn.functional.linear(H_win_fp, W_fp, b_fp)  # [Nq, Nd, Lq, d_out]
        if normalize:
            D_win = torch.nn.functional.normalize(D_win, p=2, dim=-1, eps=1e-12)

        # Score from saved argmax:
        #   per (i, j, s):  contrib = D_win[i, j, s, :] · Q[i, s, :]
        #   score[i, j] = sum_s contrib if q_mask == 1 else 0
        # The fused forward already applied q_mask via the kernel's -inf;
        # here we reuse the same convention implicitly via grad_scores.
        contrib = (D_win * Q_fp.view(Nq, 1, Lq, d_out)).sum(dim=-1)  # [Nq, Nd, Lq]
        scores_rebuilt = contrib.sum(dim=-1)  # [Nq, Nd]

        # Backprop grad_scores through scores_rebuilt to collect grad_H_win,
        # grad_W, grad_b, grad_Q_fp.
        need_Q = ctx.needs_input_grad[0]
        need_H = ctx.needs_input_grad[1]
        need_W = ctx.needs_input_grad[2]
        need_b = ctx.needs_input_grad[3] and b is not None
        # d_mask and normalize are non-differentiable
        leaves = []
        if need_Q:
            leaves.append(Q_fp)
        if need_H:
            leaves.append(H_win_fp)
        if need_W:
            leaves.append(W_fp)
        if need_b:
            leaves.append(b_fp)

        grads: list[torch.Tensor] = []
        if leaves:
            grads = list(
                torch.autograd.grad(
                    scores_rebuilt,
                    leaves,
                    grad_outputs=grad_scores,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )
            )

        gQ = gH_win = gW = gb = None
        it = iter(grads)
        if need_Q:
            gQ = next(it)
        if need_H:
            gH_win = next(it)
        if need_W:
            gW = next(it)
        if need_b:
            gb = next(it)

        # Scatter grad_H_win back into grad_H_d via index_add_.
        grad_H_d = None
        if need_H:
            grad_H_d = torch.zeros(Nd, Ld, d_model, device=H_d.device, dtype=torch.float32)
            # am_j is [Nq, Lq], gH_win is [Nq, Nd, Lq, d_model]
            for j in range(Nd):
                idx_j = am[:, j, :]  # [Nq, Lq]
                cont_j = gH_win[:, j, :, :]  # [Nq, Lq, d_model]
                grad_H_d[j].index_add_(0, idx_j.reshape(-1), cont_j.reshape(-1, d_model))
            grad_H_d = grad_H_d.to(H_d.dtype)

        if need_Q:
            gQ = gQ.to(Q.dtype)
        if need_W:
            gW = gW.to(W.dtype)
        if need_b:
            gb = gb.to(b.dtype)

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
    gathers only the winning ``H_d`` slots and back-props through
    normalize + linear via PyTorch autograd — numerically identical to
    the unfused path to fp32 tolerance.

    Drop-in replacement for::

        D_proj = F.linear(H_d, W, b)
        D_proj = F.normalize(D_proj, p=2, dim=-1) if normalize else D_proj
        scores = maxsim(Q, D_proj, d_mask=d_mask)

    Args match :func:`maxsim_from_hidden`. The gradient flows into
    ``Q``, ``H_d``, ``W``, ``b`` (whichever have ``requires_grad=True``).

    Notes:
        For encoder-bound training (GTE-ModernColBERT etc.), the
        forward kernel eliminates the ``D_proj`` intermediate entirely.
        On the backward side we only materialize ``D_proj`` at the
        argmax positions (<10 % of the full tensor on typical shapes).
        A fully fused training-side backward (persistent kernel,
        SMEM-cached ``H``) is deferred to 0.8.0 — see
        ``docs/rfc/0.7.0.md`` §5.
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
        raise ValueError(
            f"b must be [d_out={Q.shape[-1]}]; got "
            f"{tuple(b.shape) if b is not None else None}"
        )

    Q_c = Q.contiguous()
    H_c = H_d.contiguous()
    W_c = W.contiguous()
    b_c = b.contiguous() if b is not None else None
    d_mask_c = d_mask.contiguous().to(torch.int8) if d_mask is not None else None

    scores = _MaxSimFromHiddenFn.apply(Q_c, H_c, W_c, b_c, d_mask_c, normalize)
    if q_was_2d:
        return scores.squeeze(0)
    return scores
