"""Matryoshka MaxSim — score at multiple embedding dimensions in one call.

Matryoshka Representation Learning (MRL) trains embeddings so that any prefix
``emb[:, :, :d_i]`` for a nested set of dimensions ``d_1 < d_2 < ... < d_K`` is
itself a valid embedding. This is useful for adaptive retrieval: cheap low-dim
scoring first, then rerank with the full dim.

Implementation
--------------
We launch a 3-D grid ``(Nq, Nd, K)``. Each program computes one MaxSim score
at one cutoff, reading only the first ``dims[k]`` features from both Q and D.
The kernel is nearly identical to the main MaxSim forward but parameterized
by ``d_active`` (the current cutoff).

Compute cost is ``O(K * Nq * Nd * Lq * Ld * d_max)`` which is the honest cost
of computing K different MaxSim scores; we do it in one launch with shared
autotuning and no Python loop. Memory for the output is ``[K, Nq, Nd]`` fp32.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._utils import ensure_contiguous_last, next_pow2, pick_compute_dtype


@triton.jit
def _matryoshka_fwd_kernel(
    Q_ptr,
    D_ptr,
    q_mask_ptr,
    d_mask_ptr,
    dims_ptr,
    scores_ptr,
    Nq: tl.constexpr,
    Nd: tl.constexpr,
    Lq: tl.constexpr,
    Ld: tl.constexpr,
    d: tl.constexpr,
    d_pad: tl.constexpr,
    K: tl.constexpr,
    stride_q_n,
    stride_q_l,
    stride_q_d,
    stride_d_n,
    stride_d_l,
    stride_d_d,
    stride_s_k,
    stride_s_n,
    stride_s_d,
    stride_qm_n,
    stride_qm_l,
    stride_dm_n,
    stride_dm_l,
    has_q_mask: tl.constexpr,
    has_d_mask: tl.constexpr,
    normalize: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    k_idx = pid % K
    nd_idx = (pid // K) % Nd
    nq_idx = pid // (K * Nd)

    # This program's active feature cutoff.
    cutoff = tl.load(dims_ptr + k_idx)

    k_off = tl.arange(0, d_pad)
    k_mask_full = k_off < d  # valid features for load
    k_mask_cut = k_off < cutoff  # valid features for this cutoff

    score_acc = tl.zeros([], dtype=tl.float32)

    for q_start in tl.static_range(0, Lq, BLOCK_Q):
        q_off = q_start + tl.arange(0, BLOCK_Q)
        q_valid = q_off < Lq

        if has_q_mask:
            qm = tl.load(
                q_mask_ptr + nq_idx * stride_qm_n + q_off * stride_qm_l,
                mask=q_valid,
                other=0,
            ).to(tl.int1)
            q_active = q_valid & qm
        else:
            q_active = q_valid

        Qf = tl.load(
            Q_ptr + nq_idx * stride_q_n + q_off[:, None] * stride_q_l + k_off[None, :] * stride_q_d,
            mask=q_valid[:, None] & k_mask_full[None, :],
            other=0.0,
        ).to(tl.float32)

        if normalize:
            # Normalization uses the FULL d (MRL convention).
            qn = tl.sum(Qf * Qf, axis=1)
            q_inv = 1.0 / tl.sqrt(tl.maximum(qn, 1e-12))
            Qf = Qf * q_inv[:, None]

        # Zero out features beyond this cutoff so the dot stays static-shape.
        Qf = Qf * k_mask_cut[None, :].to(tl.float32)
        Q_block = Qf.to(COMPUTE_DTYPE)

        m = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)

        for d_start in range(0, Ld, BLOCK_D):
            d_off = d_start + tl.arange(0, BLOCK_D)
            d_valid = d_off < Ld

            if has_d_mask:
                dm = tl.load(
                    d_mask_ptr + nd_idx * stride_dm_n + d_off * stride_dm_l,
                    mask=d_valid,
                    other=0,
                ).to(tl.int1)
                d_active = d_valid & dm
            else:
                d_active = d_valid

            Df = tl.load(
                D_ptr + nd_idx * stride_d_n + d_off[:, None] * stride_d_l + k_off[None, :] * stride_d_d,
                mask=d_valid[:, None] & k_mask_full[None, :],
                other=0.0,
            ).to(tl.float32)
            if normalize:
                dn = tl.sum(Df * Df, axis=1)
                d_inv = 1.0 / tl.sqrt(tl.maximum(dn, 1e-12))
                Df = Df * d_inv[:, None]
            Df = Df * k_mask_cut[None, :].to(tl.float32)
            D_block = Df.to(COMPUTE_DTYPE)

            S = tl.dot(Q_block, tl.trans(D_block), out_dtype=tl.float32)
            S = tl.where(d_active[None, :], S, float("-inf"))
            m = tl.maximum(m, tl.max(S, axis=1))

        m_finite = m != float("-inf")
        m = tl.where(m_finite & q_active, m, 0.0)
        score_acc += tl.sum(m)

    tl.store(
        scores_ptr + k_idx * stride_s_k + nq_idx * stride_s_n + nd_idx * stride_s_d,
        score_acc,
    )


def maxsim_matryoshka(
    Q: torch.Tensor,
    D: torch.Tensor,
    dims: list[int] | tuple[int, ...] | torch.Tensor,
    q_mask: torch.Tensor | None = None,
    d_mask: torch.Tensor | None = None,
    *,
    normalize: bool = False,
) -> torch.Tensor:
    """Multi-dimensional MaxSim scoring in one kernel launch.

    Args:
        Q: ``[Nq, Lq, d]`` or ``[Lq, d]``.
        D: ``[Nd, Ld, d]`` or ``[Ld, d]``.
        dims: sequence of feature-dim cutoffs, each in ``(0, d]``.
        q_mask, d_mask: optional boolean masks.
        normalize: L2-normalize Q and D at the FULL dim d (MRL convention).

    Returns:
        scores: ``[K, Nq, Nd]`` fp32 where ``K = len(dims)``. Squeezed on the
        matching trailing dim(s) if the inputs were 2-D.
    """
    q_was_2d = Q.dim() == 2
    d_was_2d = D.dim() == 2
    if q_was_2d:
        Q = Q.unsqueeze(0)
    if d_was_2d:
        D = D.unsqueeze(0)
    if q_mask is not None and q_mask.dim() == 1:
        q_mask = q_mask.unsqueeze(0)
    if d_mask is not None and d_mask.dim() == 1:
        d_mask = d_mask.unsqueeze(0)

    Q = ensure_contiguous_last(Q)
    D = ensure_contiguous_last(D)
    Nq, Lq, d = Q.shape
    Nd, Ld, d2 = D.shape
    if d != d2:
        raise ValueError(f"embedding dims don't match: Q has d={d}, D has d={d2}")

    if isinstance(dims, (list, tuple)):
        dims_t = torch.tensor(list(dims), dtype=torch.int32, device=Q.device)
    else:
        dims_t = dims.to(dtype=torch.int32, device=Q.device).contiguous()
    K = int(dims_t.numel())
    if K == 0:
        raise ValueError("`dims` must have at least one entry")
    dmax = int(dims_t.max().item())
    dmin = int(dims_t.min().item())
    if dmax > d or dmin <= 0:
        raise ValueError(f"all dims must be in (0, {d}], got {dims_t.tolist()}")

    d_pad = next_pow2(d)
    compute_dtype = pick_compute_dtype(Q, D)
    tl_dtype = tl.float16 if compute_dtype == torch.float16 else tl.bfloat16

    scores = torch.empty(K, Nq, Nd, device=Q.device, dtype=torch.float32)

    has_q_mask = q_mask is not None
    has_d_mask = d_mask is not None
    q_mask_i8 = q_mask.contiguous().to(torch.int8) if has_q_mask else None
    d_mask_i8 = d_mask.contiguous().to(torch.int8) if has_d_mask else None
    q_mask_ptr = q_mask_i8 if has_q_mask else Q
    d_mask_ptr = d_mask_i8 if has_d_mask else D
    qm_strides = (q_mask_i8.stride(0), q_mask_i8.stride(1)) if has_q_mask else (0, 0)
    dm_strides = (d_mask_i8.stride(0), d_mask_i8.stride(1)) if has_d_mask else (0, 0)

    BLOCK_Q = 32 if Lq >= 32 else max(16, next_pow2(Lq))
    BLOCK_D = 64 if Ld >= 64 else max(16, next_pow2(Ld))

    grid = (Nq * Nd * K,)
    _matryoshka_fwd_kernel[grid](
        Q,
        D,
        q_mask_ptr,
        d_mask_ptr,
        dims_t,
        scores,
        Nq,
        Nd,
        Lq,
        Ld,
        d,
        d_pad,
        K,
        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        D.stride(0),
        D.stride(1),
        D.stride(2),
        scores.stride(0),
        scores.stride(1),
        scores.stride(2),
        qm_strides[0],
        qm_strides[1],
        dm_strides[0],
        dm_strides[1],
        has_q_mask,
        has_d_mask,
        normalize,
        BLOCK_Q=BLOCK_Q,
        BLOCK_D=BLOCK_D,
        COMPUTE_DTYPE=tl_dtype,
        num_warps=4,
        num_stages=2,
    )

    if q_was_2d:
        scores = scores.squeeze(1)
    if d_was_2d:
        scores = scores.squeeze(-1)
    return scores


__all__ = ["maxsim_matryoshka"]
