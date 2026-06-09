"""Variable-length (packed) MaxSim kernel — forward + fused backward.

Queries and docs arrive as ``[sum(L), d]`` tensors with ``cu_seqlens``
offsets (FlashAttention varlen convention). When either input requires
grad, the forward saves a ``[Nq, Nd, max_lq]`` argmax buffer and a fused
backward produces ``grad_Q`` (row-owned) and ``grad_D`` (atomic scatter
into the packed grad tensor) directly on the packed layout.
"""

import torch
import triton
import triton.language as tl

from late_interaction_kernels._autotune import autotune_kwargs, forward_configs, prune_forward
from late_interaction_kernels._utils import (
    assert_max_seqlen_covers,
    bucket_seqlen,
    next_pow2,
    pick_compute_dtype,
)


@triton.autotune(
    configs=forward_configs(),
    key=["max_lq", "max_ld", "d_pad"],
    prune_configs_by={"early_config_prune": prune_forward},
    **autotune_kwargs(),
)
@triton.jit
def _varlen_fwd_kernel(
    Q_ptr,
    D_ptr,
    cu_q_ptr,
    cu_d_ptr,
    scores_ptr,
    argmax_ptr,
    Nq: tl.constexpr,
    Nd: tl.constexpr,
    max_lq: tl.constexpr,
    max_ld: tl.constexpr,
    d: tl.constexpr,
    d_pad: tl.constexpr,
    stride_q_t,
    stride_q_k,
    stride_d_t,
    stride_d_k,
    stride_s_n,
    stride_s_d,
    stride_am_n,
    stride_am_d,
    stride_am_l,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    SAVE_ARGMAX: tl.constexpr,
):
    pid = tl.program_id(0)
    q_idx = pid // Nd
    d_idx = pid % Nd

    q_lo = tl.load(cu_q_ptr + q_idx).to(tl.int32)
    q_hi = tl.load(cu_q_ptr + q_idx + 1).to(tl.int32)
    d_lo = tl.load(cu_d_ptr + d_idx).to(tl.int32)
    d_hi = tl.load(cu_d_ptr + d_idx + 1).to(tl.int32)

    lq = q_hi - q_lo
    ld = d_hi - d_lo

    emb_off = tl.arange(0, d_pad)
    emb_mask = emb_off < d
    score_acc = tl.zeros([], dtype=tl.float32)

    # Empty sequences contribute zero.
    if lq == 0 or ld == 0:
        tl.store(scores_ptr + q_idx * stride_s_n + d_idx * stride_s_d, score_acc)
        if SAVE_ARGMAX:
            # Fill the argmax row with -1 so the backward skips it cleanly.
            for q_start in range(0, max_lq, BLOCK_Q):
                q_off = q_start + tl.arange(0, BLOCK_Q)
                tl.store(
                    argmax_ptr + q_idx * stride_am_n + d_idx * stride_am_d + q_off * stride_am_l,
                    tl.full([BLOCK_Q], -1, dtype=tl.int32),
                    mask=q_off < max_lq,
                )
        return

    for q_start in range(0, max_lq, BLOCK_Q):
        q_off = q_start + tl.arange(0, BLOCK_Q)
        q_valid = q_off < lq

        Q_block = tl.load(
            Q_ptr + (q_lo + q_off)[:, None] * stride_q_t + emb_off[None, :] * stride_q_k,
            mask=q_valid[:, None] & emb_mask[None, :],
            other=0.0,
        ).to(COMPUTE_DTYPE)

        m = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
        m_idx = tl.zeros([BLOCK_Q], dtype=tl.int32)

        for d_start in range(0, max_ld, BLOCK_D):
            d_off = d_start + tl.arange(0, BLOCK_D)
            d_valid = d_off < ld

            D_block = tl.load(
                D_ptr + (d_lo + d_off)[:, None] * stride_d_t + emb_off[None, :] * stride_d_k,
                mask=d_valid[:, None] & emb_mask[None, :],
                other=0.0,
            ).to(COMPUTE_DTYPE)

            S = tl.dot(Q_block, tl.trans(D_block), out_dtype=tl.float32)
            S = tl.where(d_valid[None, :], S, float("-inf"))
            tile_max = tl.max(S, axis=1)
            tile_arg = tl.argmax(S, axis=1).to(tl.int32) + d_start
            update = tile_max > m
            m = tl.where(update, tile_max, m)
            m_idx = tl.where(update, tile_arg, m_idx)

        m = tl.where(q_valid & (m != float("-inf")), m, 0.0)
        score_acc += tl.sum(m)

        if SAVE_ARGMAX:
            # Store -1 for padding query positions; the bwd uses it as "skip".
            m_idx_out = tl.where(q_valid, m_idx, -1)
            tl.store(
                argmax_ptr + q_idx * stride_am_n + d_idx * stride_am_d + q_off * stride_am_l,
                m_idx_out,
                mask=q_off < max_lq,
            )

    tl.store(scores_ptr + q_idx * stride_s_n + d_idx * stride_s_d, score_acc)


# -----------------------------------------------------------------------------
# Backward: grad_Q — one program per (q_batch, q_token_abs)
# -----------------------------------------------------------------------------


@triton.jit
def _varlen_bwd_dQ_kernel(
    D_ptr,  # [sum_Ld, d]
    cu_q_ptr,  # [Nq+1]
    cu_d_ptr,  # [Nd+1]
    argmax_ptr,  # [Nq, Nd, max_lq]
    grad_s_ptr,  # [Nq, Nd] fp32
    grad_Q_ptr,  # [sum_Lq, d] fp32
    Nq: tl.constexpr,
    Nd: tl.constexpr,
    max_lq: tl.constexpr,
    d: tl.constexpr,
    d_pad: tl.constexpr,
    stride_d_t,
    stride_d_k,
    stride_am_n,
    stride_am_d,
    stride_am_l,
    stride_gs_n,
    stride_gs_d,
    stride_gq_t,
    stride_gq_k,
):
    pid = tl.program_id(0)
    q_idx = pid // max_lq
    s = pid % max_lq

    q_lo = tl.load(cu_q_ptr + q_idx).to(tl.int32)
    q_hi = tl.load(cu_q_ptr + q_idx + 1).to(tl.int32)
    lq = q_hi - q_lo

    if s >= lq:
        return

    emb_off = tl.arange(0, d_pad)
    emb_mask = emb_off < d
    acc = tl.zeros([d_pad], dtype=tl.float32)

    for d_idx in range(0, Nd):
        d_lo = tl.load(cu_d_ptr + d_idx).to(tl.int32)
        d_hi = tl.load(cu_d_ptr + d_idx + 1).to(tl.int32)
        ld = d_hi - d_lo

        t = tl.load(argmax_ptr + q_idx * stride_am_n + d_idx * stride_am_d + s * stride_am_l).to(tl.int32)
        # t == -1 on empty docs or invalid positions; skip those safely.
        valid = (t >= 0) & (ld > 0)
        if valid:
            gs = tl.load(grad_s_ptr + q_idx * stride_gs_n + d_idx * stride_gs_d).to(tl.float32)
            dv = tl.load(
                D_ptr + (d_lo + t) * stride_d_t + emb_off * stride_d_k,
                mask=emb_mask,
                other=0.0,
            ).to(tl.float32)
            acc += gs * dv

    tl.store(
        grad_Q_ptr + (q_lo + s) * stride_gq_t + emb_off * stride_gq_k,
        acc,
        mask=emb_mask,
    )


# -----------------------------------------------------------------------------
# Backward: grad_D — atomic scatter. One program per (q_batch, d_batch).
# -----------------------------------------------------------------------------


@triton.jit
def _varlen_bwd_dD_kernel(
    Q_ptr,  # [sum_Lq, d]
    cu_q_ptr,
    cu_d_ptr,
    argmax_ptr,
    grad_s_ptr,
    grad_D_ptr,  # [sum_Ld, d] fp32
    Nd: tl.constexpr,
    max_lq: tl.constexpr,
    d: tl.constexpr,
    d_pad: tl.constexpr,
    stride_q_t,
    stride_q_k,
    stride_am_n,
    stride_am_d,
    stride_am_l,
    stride_gs_n,
    stride_gs_d,
    stride_gd_t,
    stride_gd_k,
):
    pid = tl.program_id(0)
    q_idx = pid // Nd
    d_idx = pid % Nd

    q_lo = tl.load(cu_q_ptr + q_idx).to(tl.int32)
    q_hi = tl.load(cu_q_ptr + q_idx + 1).to(tl.int32)
    lq = q_hi - q_lo
    d_lo = tl.load(cu_d_ptr + d_idx).to(tl.int32)

    if lq == 0:
        return

    gs = tl.load(grad_s_ptr + q_idx * stride_gs_n + d_idx * stride_gs_d).to(tl.float32)

    emb_off = tl.arange(0, d_pad)
    emb_mask = emb_off < d

    for s in range(0, max_lq):
        if s < lq:
            t = tl.load(argmax_ptr + q_idx * stride_am_n + d_idx * stride_am_d + s * stride_am_l).to(tl.int32)
            if t >= 0:
                qv = tl.load(
                    Q_ptr + (q_lo + s) * stride_q_t + emb_off * stride_q_k,
                    mask=emb_mask,
                    other=0.0,
                ).to(tl.float32)
                tl.atomic_add(
                    grad_D_ptr + (d_lo + t) * stride_gd_t + emb_off * stride_gd_k,
                    gs * qv,
                    mask=emb_mask,
                )


# -----------------------------------------------------------------------------
# Python launcher + autograd wrapper
# -----------------------------------------------------------------------------


def _varlen_forward(
    Q_packed: torch.Tensor,
    D_packed: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_d: torch.Tensor,
    max_seqlen_q: int | None,
    max_seqlen_d: int | None,
    save_argmax: bool,
):
    if Q_packed.dim() != 2:
        raise ValueError(
            f"Q_packed must be 2-D [sum(Lq_i), d]; got shape {tuple(Q_packed.shape)} "
            f"(ndim={Q_packed.dim()}). For padded inputs use `maxsim(Q, D, ...)` instead."
        )
    if D_packed.dim() != 2:
        raise ValueError(
            f"D_packed must be 2-D [sum(Ld_j), d]; got shape {tuple(D_packed.shape)} "
            f"(ndim={D_packed.dim()}). For padded inputs use `maxsim(Q, D, ...)` instead."
        )
    d = Q_packed.shape[1]
    if D_packed.shape[1] != d:
        raise ValueError(
            f"Q_packed and D_packed must share the embedding dim; got "
            f"Q_packed.shape[1]={Q_packed.shape[1]} vs D_packed.shape[1]={D_packed.shape[1]}."
        )

    cu_seqlens_q = cu_seqlens_q.to(torch.int32).contiguous()
    cu_seqlens_d = cu_seqlens_d.to(torch.int32).contiguous()
    Nq = cu_seqlens_q.numel() - 1
    Nd = cu_seqlens_d.numel() - 1

    if max_seqlen_q is None:
        max_seqlen_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()) if Nq else 0
    else:
        assert_max_seqlen_covers(cu_seqlens_q, int(max_seqlen_q), "max_seqlen_q")
    if max_seqlen_d is None:
        max_seqlen_d = int((cu_seqlens_d[1:] - cu_seqlens_d[:-1]).max().item()) if Nd else 0
    else:
        assert_max_seqlen_covers(cu_seqlens_d, int(max_seqlen_d), "max_seqlen_d")

    # max_lq / max_ld are constexpr loop bounds AND autotune keys, so without
    # bucketing every distinct (max_lq, max_ld) pair re-triggers the full
    # autotune sweep. The kernel masks on the actual cu_seqlens bounds, so a
    # larger loop bound only adds fully-masked iterations. The argmax buffer
    # is sized on the bucketed value (its rows are -1-padded; backward skips).
    max_seqlen_q = bucket_seqlen(int(max_seqlen_q))
    max_seqlen_d = bucket_seqlen(int(max_seqlen_d))

    d_pad = next_pow2(d)
    compute_dtype = pick_compute_dtype(Q_packed, D_packed)
    tl_dtype = tl.float16 if compute_dtype == torch.float16 else tl.bfloat16

    scores = torch.zeros(Nq, Nd, device=Q_packed.device, dtype=torch.float32)

    Q_packed = Q_packed.contiguous()
    D_packed = D_packed.contiguous()

    if save_argmax and max_seqlen_q > 0:
        argmax = torch.full((Nq, Nd, max_seqlen_q), -1, device=Q_packed.device, dtype=torch.int32)
        am_strides = (argmax.stride(0), argmax.stride(1), argmax.stride(2))
    else:
        argmax = torch.empty(1, device=Q_packed.device, dtype=torch.int32)
        am_strides = (0, 0, 0)

    _varlen_fwd_kernel[(Nq * Nd,)](
        Q_packed,
        D_packed,
        cu_seqlens_q,
        cu_seqlens_d,
        scores,
        argmax,
        Nq,
        Nd,
        max_seqlen_q,
        max_seqlen_d,
        d,
        d_pad,
        Q_packed.stride(0),
        Q_packed.stride(1),
        D_packed.stride(0),
        D_packed.stride(1),
        scores.stride(0),
        scores.stride(1),
        am_strides[0],
        am_strides[1],
        am_strides[2],
        COMPUTE_DTYPE=tl_dtype,
        SAVE_ARGMAX=save_argmax,
    )
    return scores, (argmax if save_argmax else None), max_seqlen_q, max_seqlen_d


class _MaxSimVarlenFn(torch.autograd.Function):
    """Autograd-aware varlen MaxSim."""

    @staticmethod
    def forward(ctx, Q, D, cu_q, cu_d, max_q, max_d):
        scores, argmax, max_q, max_d = _varlen_forward(Q, D, cu_q, cu_d, max_q, max_d, save_argmax=True)
        ctx.save_for_backward(Q, D, cu_q.to(torch.int32), cu_d.to(torch.int32), argmax)
        ctx.max_q = max_q
        ctx.max_d = max_d
        return scores

    @staticmethod
    def backward(ctx, grad_scores):
        Q, D, cu_q, cu_d, argmax = ctx.saved_tensors
        max_q = ctx.max_q
        grad_scores = grad_scores.contiguous().to(torch.float32)

        _, d = Q.shape
        Nq = cu_q.numel() - 1
        Nd = cu_d.numel() - 1
        d_pad = next_pow2(d)

        grad_Q = torch.zeros_like(Q, dtype=torch.float32)
        grad_D = torch.zeros_like(D, dtype=torch.float32)

        if max_q > 0:
            _varlen_bwd_dQ_kernel[(Nq * max_q,)](
                D,
                cu_q,
                cu_d,
                argmax,
                grad_scores,
                grad_Q,
                Nq,
                Nd,
                max_q,
                d,
                d_pad,
                D.stride(0),
                D.stride(1),
                argmax.stride(0),
                argmax.stride(1),
                argmax.stride(2),
                grad_scores.stride(0),
                grad_scores.stride(1),
                grad_Q.stride(0),
                grad_Q.stride(1),
                num_warps=4,
                num_stages=2,
            )

            _varlen_bwd_dD_kernel[(Nq * Nd,)](
                Q,
                cu_q,
                cu_d,
                argmax,
                grad_scores,
                grad_D,
                Nd,
                max_q,
                d,
                d_pad,
                Q.stride(0),
                Q.stride(1),
                argmax.stride(0),
                argmax.stride(1),
                argmax.stride(2),
                grad_scores.stride(0),
                grad_scores.stride(1),
                grad_D.stride(0),
                grad_D.stride(1),
                num_warps=4,
                num_stages=2,
            )

        return (
            grad_Q.to(Q.dtype),
            grad_D.to(D.dtype),
            None,  # cu_q
            None,  # cu_d
            None,  # max_q
            None,  # max_d
        )


def maxsim_varlen(
    Q_packed: torch.Tensor,
    D_packed: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_d: torch.Tensor,
    max_seqlen_q: int | None = None,
    max_seqlen_d: int | None = None,
) -> torch.Tensor:
    """MaxSim on packed (no-padding) inputs. Autograd-aware.

    Args:
        Q_packed: ``[sum(Lq_i), d]`` fp16 / bf16 / fp32.
        D_packed: ``[sum(Ld_j), d]``.
        cu_seqlens_q: ``[Nq + 1]`` int32 cumulative offsets.
        cu_seqlens_d: ``[Nd + 1]`` int32 cumulative offsets.
        max_seqlen_q, max_seqlen_d: hard kernel loop bounds, NOT hints — a
            value smaller than the longest sequence would silently drop
            tokens, so it is rejected by an on-device assert (no D2H sync).
            Inferred from ``cu_seqlens`` (one D2H sync) if omitted. Both are
            rounded up to the next power of two internally so the autotune
            cache is reused across batches with different maxima.

    Returns:
        scores: ``[Nq, Nd]`` fp32.

    The argmax save and fused backward kernels run only when either input
    has ``requires_grad=True``; for pure reranking the inference path is
    automatic.
    """
    if Q_packed.requires_grad or D_packed.requires_grad:
        return _MaxSimVarlenFn.apply(
            Q_packed,
            D_packed,
            cu_seqlens_q.to(torch.int32),
            cu_seqlens_d.to(torch.int32),
            max_seqlen_q,
            max_seqlen_d,
        )
    scores, _, _, _ = _varlen_forward(
        Q_packed, D_packed, cu_seqlens_q, cu_seqlens_d, max_seqlen_q, max_seqlen_d, False
    )
    return scores
