"""Pair-list MaxSim — score arbitrary ``(query, doc)`` pairs from packed batches.

Use case: a single forward pass projects every query and every doc into one
flat buffer, then a scheduler asks for an arbitrary subset of
``(query_index, doc_index)`` pairs to be scored (typical vLLM / LM-server
reranker workload). The full ``[Nq, Nd]`` matrix is wasteful when the pair
list is sparse — this kernel produces a ``[num_pairs]`` vector directly.

Inputs are packed (``cu_seqlens``) like ``maxsim_varlen``; the only addition
is the ``pair_q_idx`` / ``pair_d_idx`` index pair. The forward saves a
``[num_pairs, max_lq]`` argmax buffer when either input has
``requires_grad=True`` and two fused backward kernels produce ``grad_Q`` /
``grad_D`` directly on the packed layout (atomic-add scatter on both sides:
multiple pairs may share ``q_idx`` or ``d_idx``).
"""

import torch
import triton
import triton.language as tl

from late_interaction_kernels._autotune import autotune_kwargs, forward_configs, prune_forward
from late_interaction_kernels._utils import assert_max_seqlen_covers, next_pow2, pick_compute_dtype


@triton.autotune(
    configs=forward_configs(),
    key=["d_pad"],
    prune_configs_by={"early_config_prune": prune_forward},
    **autotune_kwargs(),
)
@triton.jit
def _scatter_fwd_kernel(
    Q_ptr,  # [sum_Lq, d]
    D_ptr,  # [sum_Ld, d]
    cu_q_ptr,  # [Nq + 1]
    cu_d_ptr,  # [Nd + 1]
    pair_q_ptr,  # [num_pairs] int32
    pair_d_ptr,  # [num_pairs] int32
    out_ptr,  # [num_pairs] fp32
    argmax_ptr,  # [num_pairs, max_lq] int32 (unused when SAVE_ARGMAX=False)
    num_pairs,
    max_lq,
    max_ld,
    d: tl.constexpr,
    d_pad: tl.constexpr,
    stride_q_t,
    stride_q_k,
    stride_d_t,
    stride_d_k,
    stride_am_pair,
    stride_am_lq,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    SAVE_ARGMAX: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_pairs:
        return

    q_idx = tl.load(pair_q_ptr + pid).to(tl.int32)
    d_idx = tl.load(pair_d_ptr + pid).to(tl.int32)

    q_lo = tl.load(cu_q_ptr + q_idx).to(tl.int32)
    q_hi = tl.load(cu_q_ptr + q_idx + 1).to(tl.int32)
    d_lo = tl.load(cu_d_ptr + d_idx).to(tl.int32)
    d_hi = tl.load(cu_d_ptr + d_idx + 1).to(tl.int32)

    lq = q_hi - q_lo
    ld = d_hi - d_lo

    score_acc = tl.zeros([], dtype=tl.float32)

    if lq == 0 or ld == 0:
        tl.store(out_ptr + pid, score_acc)
        if SAVE_ARGMAX:
            for q_start in range(0, max_lq, BLOCK_Q):
                q_off = q_start + tl.arange(0, BLOCK_Q)
                tl.store(
                    argmax_ptr + pid * stride_am_pair + q_off * stride_am_lq,
                    tl.full([BLOCK_Q], -1, dtype=tl.int32),
                    mask=q_off < max_lq,
                )
        return

    emb_off = tl.arange(0, d_pad)
    emb_mask = emb_off < d

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
            m_idx_out = tl.where(q_valid, m_idx, -1)
            tl.store(
                argmax_ptr + pid * stride_am_pair + q_off * stride_am_lq,
                m_idx_out,
                mask=q_off < max_lq,
            )

    tl.store(out_ptr + pid, score_acc)


# -----------------------------------------------------------------------------
# Backward: grad_Q — one program per (pair, q_token slot).
#
# Multiple pairs may share q_idx, so writes into grad_Q are atomic. We slice
# by slot rather than by pair to keep each program's working set to one
# embedding row, which matches varlen's `_varlen_bwd_dQ_kernel` shape.
# -----------------------------------------------------------------------------


@triton.jit
def _scatter_bwd_dQ_kernel(
    D_ptr,
    cu_q_ptr,
    cu_d_ptr,
    pair_q_ptr,
    pair_d_ptr,
    argmax_ptr,
    grad_s_ptr,
    grad_Q_ptr,
    num_pairs,
    max_lq,
    d: tl.constexpr,
    d_pad: tl.constexpr,
    stride_d_t,
    stride_d_k,
    stride_am_pair,
    stride_am_lq,
    stride_gq_t,
    stride_gq_k,
):
    pid = tl.program_id(0)
    pair = pid // max_lq
    s = pid % max_lq

    if pair >= num_pairs:
        return

    q_idx = tl.load(pair_q_ptr + pair).to(tl.int32)
    d_idx = tl.load(pair_d_ptr + pair).to(tl.int32)

    q_lo = tl.load(cu_q_ptr + q_idx).to(tl.int32)
    q_hi = tl.load(cu_q_ptr + q_idx + 1).to(tl.int32)
    d_lo = tl.load(cu_d_ptr + d_idx).to(tl.int32)

    if s >= q_hi - q_lo:
        return

    t = tl.load(argmax_ptr + pair * stride_am_pair + s * stride_am_lq).to(tl.int32)
    if t < 0:
        return

    gs = tl.load(grad_s_ptr + pair).to(tl.float32)

    emb_off = tl.arange(0, d_pad)
    emb_mask = emb_off < d
    dv = tl.load(
        D_ptr + (d_lo + t) * stride_d_t + emb_off * stride_d_k,
        mask=emb_mask,
        other=0.0,
    ).to(tl.float32)

    tl.atomic_add(
        grad_Q_ptr + (q_lo + s) * stride_gq_t + emb_off * stride_gq_k,
        gs * dv,
        mask=emb_mask,
    )


# -----------------------------------------------------------------------------
# Backward: grad_D — one program per (pair, q_token slot).
#
# Multiple pairs may share d_idx, and within a single pair multiple slots
# may resolve to the same winning doc-token: atomic-add on both axes.
# -----------------------------------------------------------------------------


@triton.jit
def _scatter_bwd_dD_kernel(
    Q_ptr,
    cu_q_ptr,
    cu_d_ptr,
    pair_q_ptr,
    pair_d_ptr,
    argmax_ptr,
    grad_s_ptr,
    grad_D_ptr,
    num_pairs,
    max_lq,
    d: tl.constexpr,
    d_pad: tl.constexpr,
    stride_q_t,
    stride_q_k,
    stride_am_pair,
    stride_am_lq,
    stride_gd_t,
    stride_gd_k,
):
    pid = tl.program_id(0)
    pair = pid // max_lq
    s = pid % max_lq

    if pair >= num_pairs:
        return

    q_idx = tl.load(pair_q_ptr + pair).to(tl.int32)
    d_idx = tl.load(pair_d_ptr + pair).to(tl.int32)

    q_lo = tl.load(cu_q_ptr + q_idx).to(tl.int32)
    q_hi = tl.load(cu_q_ptr + q_idx + 1).to(tl.int32)
    d_lo = tl.load(cu_d_ptr + d_idx).to(tl.int32)

    if s >= q_hi - q_lo:
        return

    t = tl.load(argmax_ptr + pair * stride_am_pair + s * stride_am_lq).to(tl.int32)
    if t < 0:
        return

    gs = tl.load(grad_s_ptr + pair).to(tl.float32)

    emb_off = tl.arange(0, d_pad)
    emb_mask = emb_off < d
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
# Python-side launcher + autograd wrapper
# -----------------------------------------------------------------------------


def _scatter_forward(
    Q_packed: torch.Tensor,
    D_packed: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_d: torch.Tensor,
    pair_q: torch.Tensor,
    pair_d: torch.Tensor,
    max_seqlen_q: int | None,
    max_seqlen_d: int | None,
    save_argmax: bool,
):
    num_pairs = pair_q.numel()
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

    d = Q_packed.shape[1]
    d_pad = next_pow2(d)
    compute_dtype = pick_compute_dtype(Q_packed, D_packed)
    tl_dtype = tl.float16 if compute_dtype == torch.float16 else tl.bfloat16

    out = torch.empty(num_pairs, device=Q_packed.device, dtype=torch.float32)

    if save_argmax and max_seqlen_q > 0 and num_pairs > 0:
        argmax = torch.full((num_pairs, max_seqlen_q), -1, device=Q_packed.device, dtype=torch.int32)
        am_strides = (argmax.stride(0), argmax.stride(1))
    else:
        argmax = torch.empty(1, device=Q_packed.device, dtype=torch.int32)
        am_strides = (0, 0)

    _scatter_fwd_kernel[(num_pairs,)](
        Q_packed,
        D_packed,
        cu_seqlens_q,
        cu_seqlens_d,
        pair_q,
        pair_d,
        out,
        argmax,
        num_pairs,
        max_seqlen_q,
        max_seqlen_d,
        d,
        d_pad,
        Q_packed.stride(0),
        Q_packed.stride(1),
        D_packed.stride(0),
        D_packed.stride(1),
        am_strides[0],
        am_strides[1],
        COMPUTE_DTYPE=tl_dtype,
        SAVE_ARGMAX=save_argmax,
    )
    return out, (argmax if save_argmax else None), max_seqlen_q, max_seqlen_d


class _MaxSimScorePairsFn(torch.autograd.Function):
    """Autograd-aware pair-list MaxSim on packed batches."""

    @staticmethod
    def forward(ctx, Q, D, cu_q, cu_d, pair_q, pair_d, max_q, max_d):
        out, argmax, max_q, _ = _scatter_forward(
            Q, D, cu_q, cu_d, pair_q, pair_d, max_q, max_d, save_argmax=True
        )
        ctx.save_for_backward(Q, D, cu_q, cu_d, pair_q, pair_d, argmax)
        ctx.max_q = max_q
        return out

    @staticmethod
    def backward(ctx, grad_out):
        Q, D, cu_q, cu_d, pair_q, pair_d, argmax = ctx.saved_tensors
        max_q = ctx.max_q
        grad_out = grad_out.contiguous().to(torch.float32)

        num_pairs = pair_q.numel()
        d = Q.shape[1]
        d_pad = next_pow2(d)

        grad_Q = torch.zeros_like(Q, dtype=torch.float32)
        grad_D = torch.zeros_like(D, dtype=torch.float32)

        if num_pairs > 0 and max_q > 0:
            _scatter_bwd_dQ_kernel[(num_pairs * max_q,)](
                D,
                cu_q,
                cu_d,
                pair_q,
                pair_d,
                argmax,
                grad_out,
                grad_Q,
                num_pairs,
                max_q,
                d,
                d_pad,
                D.stride(0),
                D.stride(1),
                argmax.stride(0),
                argmax.stride(1),
                grad_Q.stride(0),
                grad_Q.stride(1),
                num_warps=4,
                num_stages=2,
            )
            _scatter_bwd_dD_kernel[(num_pairs * max_q,)](
                Q,
                cu_q,
                cu_d,
                pair_q,
                pair_d,
                argmax,
                grad_out,
                grad_D,
                num_pairs,
                max_q,
                d,
                d_pad,
                Q.stride(0),
                Q.stride(1),
                argmax.stride(0),
                argmax.stride(1),
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
            None,  # pair_q
            None,  # pair_d
            None,  # max_q
            None,  # max_d
        )


def score_pairs_packed(
    Q_packed: torch.Tensor,
    D_packed: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_d: torch.Tensor,
    pair_q_idx: torch.Tensor,
    pair_d_idx: torch.Tensor,
    *,
    max_seqlen_q: int | None = None,
    max_seqlen_d: int | None = None,
) -> torch.Tensor:
    """Score arbitrary ``(query, doc)`` pairs from packed batches. Autograd-aware.

    Args:
        Q_packed: ``[sum(Lq_i), d]`` query tokens, concatenated.
        D_packed: ``[sum(Ld_j), d]`` doc tokens, concatenated.
        cu_seqlens_q: ``[Nq + 1]`` int32 cumulative offsets into ``Q_packed``.
        cu_seqlens_d: ``[Nd + 1]`` int32 cumulative offsets into ``D_packed``.
        pair_q_idx: ``[num_pairs]`` int32 query indices.
        pair_d_idx: ``[num_pairs]`` int32 doc indices.
        max_seqlen_q / max_seqlen_d: hard kernel loop bounds, NOT hints — a
            value smaller than the longest sequence would silently drop
            tokens, so it is rejected by an on-device assert (no D2H sync).
            Computed from ``cu_seqlens`` (one D2H sync) if omitted.

    Returns:
        scores: ``[num_pairs]`` fp32. ``scores[k]`` is the MaxSim of
        ``Q_packed[cu_seqlens_q[pair_q_idx[k]]:...]`` against
        ``D_packed[cu_seqlens_d[pair_d_idx[k]]:...]``.

    Notes:
        Skips the ``[Nq, Nd]`` allocation. Use this when the pair list is
        sparse relative to ``Nq * Nd`` (typical reranker scheduling).
        For full pairwise scoring, ``maxsim_varlen`` is faster.

        The argmax save and backward kernels run only when either input has
        ``requires_grad=True``; pure inference pays no overhead.
    """
    if Q_packed.dim() != 2 or D_packed.dim() != 2:
        raise ValueError(
            "Q_packed / D_packed must be 2-D [sum(L), d]; "
            f"got Q={tuple(Q_packed.shape)}, D={tuple(D_packed.shape)}."
        )
    d = Q_packed.shape[1]
    if D_packed.shape[1] != d:
        raise ValueError(f"Q / D embedding dims mismatch: {Q_packed.shape[1]} vs {D_packed.shape[1]}.")
    if pair_q_idx.shape != pair_d_idx.shape or pair_q_idx.dim() != 1:
        raise ValueError(
            f"pair_q_idx / pair_d_idx must be matching 1-D tensors; "
            f"got {tuple(pair_q_idx.shape)} vs {tuple(pair_d_idx.shape)}."
        )

    cu_seqlens_q = cu_seqlens_q.to(torch.int32).contiguous()
    cu_seqlens_d = cu_seqlens_d.to(torch.int32).contiguous()
    pair_q = pair_q_idx.to(torch.int32).contiguous()
    pair_d = pair_d_idx.to(torch.int32).contiguous()
    Q_packed = Q_packed.contiguous()
    D_packed = D_packed.contiguous()

    if Q_packed.requires_grad or D_packed.requires_grad:
        return _MaxSimScorePairsFn.apply(
            Q_packed,
            D_packed,
            cu_seqlens_q,
            cu_seqlens_d,
            pair_q,
            pair_d,
            max_seqlen_q,
            max_seqlen_d,
        )

    if pair_q.numel() == 0:
        return torch.empty(0, device=Q_packed.device, dtype=torch.float32)

    out, _, _, _ = _scatter_forward(
        Q_packed,
        D_packed,
        cu_seqlens_q,
        cu_seqlens_d,
        pair_q,
        pair_d,
        max_seqlen_q,
        max_seqlen_d,
        save_argmax=False,
    )
    return out


__all__ = ["score_pairs_packed"]
