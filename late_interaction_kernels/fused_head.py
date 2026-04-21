"""Fused D-side projection + normalize + MaxSim (inference).

Motivation
----------
When the on-disk format is ``[Nd, Ld, d_model]`` hidden states (e.g. the
last layer of a ModernBERT encoder) and the MaxSim dimension is
``d_out`` (typically 64-128), the standard pipeline is::

    D_proj = F.normalize(H_d @ W.T)          # [Nd, Ld, d_out]  <-- big HBM scratch
    scores = maxsim(Q, D_proj)

For large corpora the ``D_proj`` intermediate can be multi-GB. This kernel
folds projection + L2-normalize + MaxSim into a single pass, so the
corpus only has to exist in HBM as ``[Nd, Ld, d_model]`` hidden states.

Scope
-----
- **Inference only** (no backward). The symmetric training-side fused
  head reads each ``H`` row ``Nq·Nd`` times, which is 4-6× more HBM
  traffic than reading the already-projected ``Q`` — a net loss. The
  training-side variant requires a persistent kernel with per-SM SMEM
  caching and is deferred to 0.7.0. See `docs/rfc/0.6.0.md` for the
  HBM accounting.
- ``Q`` is expected to be already projected (cheap on the query side —
  one small matmul per query).
- D-side bias and ``W`` weight are loaded once per tile via Triton's
  standard tiling; no special persistent layout.

API
---
``maxsim_from_hidden(Q, H_d, W, b=None, d_mask=None, normalize=True)``
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
    key=["Lq", "Ld", "d_out_pad", "d_model_pad", "has_bias", "has_d_mask", "normalize"],
)
@triton.jit
def _fused_head_fwd_kernel(
    Q_ptr,  # [Nq, Lq, d_out]  (already projected + optionally normalized)
    H_ptr,  # [Nd, Ld, d_model]
    W_ptr,  # [d_out, d_model]
    b_ptr,  # [d_out] or dummy
    d_mask_ptr,  # [Nd, Ld] or dummy
    scores_ptr,  # [Nq, Nd]
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
    has_bias: tl.constexpr,
    has_d_mask: tl.constexpr,
    normalize: tl.constexpr,
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
            m = tl.maximum(m, tile_max)

        m_finite = m != float("-inf")
        m = tl.where(m_finite & q_valid, m, 0.0)
        score_acc += tl.sum(m)

    tl.store(scores_ptr + q_idx * stride_s_n + d_idx * stride_s_d, score_acc)


def _fused_head_forward(
    Q: torch.Tensor,
    H_d: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor | None,
    d_mask: torch.Tensor | None,
    normalize: bool,
) -> torch.Tensor:
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

    b_ptr = b if has_bias else Q
    d_mask_ptr = d_mask if has_d_mask else H_d
    dm_strides = (d_mask.stride(0), d_mask.stride(1)) if has_d_mask else (0, 0)

    grid = (Nq * Nd,)
    _fused_head_fwd_kernel[grid](
        Q,
        H_d,
        W,
        b_ptr,
        d_mask_ptr,
        scores,
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
        has_bias,
        has_d_mask,
        normalize,
        COMPUTE_DTYPE=tl_dtype,
    )
    return scores


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

    scores = _fused_head_forward(Q, H_d, W, b_c, d_mask_c, normalize)
    if q_was_2d:
        return scores.squeeze(0)
    return scores
