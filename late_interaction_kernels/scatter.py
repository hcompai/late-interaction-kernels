"""Pair-list MaxSim — score arbitrary ``(query, doc)`` pairs from packed batches.

Use case: a single forward pass projects every query and every doc into one
flat buffer, then a scheduler asks for an arbitrary subset of
``(query_index, doc_index)`` pairs to be scored (typical vLLM / LM-server
reranker workload). The full ``[Nq, Nd]`` matrix is wasteful when the pair
list is sparse — this kernel produces a ``[num_pairs]`` vector directly.

Inputs are packed (``cu_seqlens``) like ``maxsim_varlen``; the only addition
is the ``pair_q_idx`` / ``pair_d_idx`` index pair. Forward only — for
gradient flow on packed batches use ``maxsim_varlen``.
"""

import torch
import triton
import triton.language as tl

from ._autotune import forward_configs, prune_forward
from ._utils import next_pow2, pick_compute_dtype


@triton.autotune(
    configs=forward_configs(),
    key=["max_lq", "max_ld", "d_pad"],
    prune_configs_by={"early_config_prune": prune_forward},
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
    num_pairs: tl.constexpr,
    max_lq: tl.constexpr,
    max_ld: tl.constexpr,
    d: tl.constexpr,
    d_pad: tl.constexpr,
    stride_q_t,
    stride_q_k,
    stride_d_t,
    stride_d_k,
    Lq,  # kernel uses max_lq/max_ld; these args are kept only for call-site compat.
    Ld,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
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
        return

    k_off = tl.arange(0, d_pad)
    k_mask = k_off < d

    for q_start in range(0, max_lq, BLOCK_Q):
        q_off = q_start + tl.arange(0, BLOCK_Q)
        q_valid = q_off < lq

        Q_block = tl.load(
            Q_ptr + (q_lo + q_off)[:, None] * stride_q_t + k_off[None, :] * stride_q_k,
            mask=q_valid[:, None] & k_mask[None, :],
            other=0.0,
        ).to(COMPUTE_DTYPE)

        m = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)

        for d_start in range(0, max_ld, BLOCK_D):
            d_off = d_start + tl.arange(0, BLOCK_D)
            d_valid = d_off < ld

            D_block = tl.load(
                D_ptr + (d_lo + d_off)[:, None] * stride_d_t + k_off[None, :] * stride_d_k,
                mask=d_valid[:, None] & k_mask[None, :],
                other=0.0,
            ).to(COMPUTE_DTYPE)

            S = tl.dot(Q_block, tl.trans(D_block), out_dtype=tl.float32)
            S = tl.where(d_valid[None, :], S, float("-inf"))
            m = tl.maximum(m, tl.max(S, axis=1))

        m = tl.where(q_valid & (m != float("-inf")), m, 0.0)
        score_acc += tl.sum(m)

    tl.store(out_ptr + pid, score_acc)


def maxsim_inference_scatter(
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
    """Score arbitrary ``(query, doc)`` pairs from packed batches. Inference only.

    Args:
        Q_packed: ``[sum(Lq_i), d]`` query tokens, concatenated.
        D_packed: ``[sum(Ld_j), d]`` doc tokens, concatenated.
        cu_seqlens_q: ``[Nq + 1]`` int32 cumulative offsets into ``Q_packed``.
        cu_seqlens_d: ``[Nd + 1]`` int32 cumulative offsets into ``D_packed``.
        pair_q_idx: ``[num_pairs]`` int32 query indices.
        pair_d_idx: ``[num_pairs]`` int32 doc indices.
        max_seqlen_q / max_seqlen_d: hints; computed from ``cu_seqlens`` if
            omitted (one D2H sync per call).

    Returns:
        scores: ``[num_pairs]`` fp32. ``scores[k]`` is the MaxSim of
        ``Q_packed[cu_seqlens_q[pair_q_idx[k]]:...]`` against
        ``D_packed[cu_seqlens_d[pair_d_idx[k]]:...]``.

    Notes:
        Skips the ``[Nq, Nd]`` allocation. Use this when the pair list is
        sparse relative to ``Nq * Nd`` (typical reranker scheduling).
        For full pairwise scoring, ``maxsim_varlen`` is faster.
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
    num_pairs = pair_q.numel()

    if num_pairs == 0:
        return torch.empty(0, device=Q_packed.device, dtype=torch.float32)

    if max_seqlen_q is None:
        max_seqlen_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item())
    if max_seqlen_d is None:
        max_seqlen_d = int((cu_seqlens_d[1:] - cu_seqlens_d[:-1]).max().item())

    d_pad = next_pow2(d)
    compute_dtype = pick_compute_dtype(Q_packed, D_packed)
    tl_dtype = tl.float16 if compute_dtype == torch.float16 else tl.bfloat16

    Q_packed = Q_packed.contiguous()
    D_packed = D_packed.contiguous()
    out = torch.empty(num_pairs, device=Q_packed.device, dtype=torch.float32)

    _scatter_fwd_kernel[(num_pairs,)](
        Q_packed,
        D_packed,
        cu_seqlens_q,
        cu_seqlens_d,
        pair_q,
        pair_d,
        out,
        num_pairs,
        max_seqlen_q,
        max_seqlen_d,
        d,
        d_pad,
        Q_packed.stride(0),
        Q_packed.stride(1),
        D_packed.stride(0),
        D_packed.stride(1),
        max_seqlen_q,
        max_seqlen_d,
        COMPUTE_DTYPE=tl_dtype,
    )
    return out


__all__ = ["maxsim_inference_scatter"]
