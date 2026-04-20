"""Variable-length (packed) MaxSim kernel.

This is the **zero-padding** path: queries and docs arrive as `[total_tokens, d]`
tensors, and per-sequence lengths come via `cu_seqlens` (CSR-style offsets),
exactly like FlashAttention's varlen mode.

Why it matters for ColBERT / ColPali
------------------------------------
PyLate's `rerank()` does `torch.nn.utils.rnn.pad_sequence` on documents before
scoring. That's a deep-copy of every doc up to Ld_max, with ~50 % waste on
realistic distributions (some docs 32 tokens, some 512). Varlen skips padding
entirely.

API
---
    scores = maxsim_varlen(
        Q_packed,            # [sum(Lq_i), d]
        D_packed,            # [sum(Ld_j), d]
        cu_seqlens_q,        # [Nq + 1] int32, cumulative sum
        cu_seqlens_d,        # [Nd + 1] int32, cumulative sum
        max_seqlen_q,        # int
        max_seqlen_d,        # int
    )                        # -> [Nq, Nd] fp32

Internally we dispatch one Triton program per (q_batch, d_batch) pair. Each
program reads its own `cu_seqlens` entries, bounds-checks loads, and runs the
same inner loop as the padded kernel.
"""

from __future__ import annotations

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
def _varlen_fwd_kernel(
    Q_ptr,
    D_ptr,
    cu_q_ptr,
    cu_d_ptr,
    scores_ptr,
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
    Lq: tl.constexpr,
    Ld: tl.constexpr,  # unused, kept for autotune key compat
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
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

    k_off = tl.arange(0, d_pad)
    k_mask = k_off < d
    score_acc = tl.zeros([], dtype=tl.float32)

    # Empty sequences contribute zero.
    if lq == 0 or ld == 0:
        tl.store(scores_ptr + q_idx * stride_s_n + d_idx * stride_s_d, score_acc)
        return

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
        # Short-circuit: if this q-tile is entirely past the end of the query,
        # all further tiles are too — we can break out. Triton lacks a `break`
        # statement in for loops, so we rely on the tl.where zeroing it out.

    tl.store(scores_ptr + q_idx * stride_s_n + d_idx * stride_s_d, score_acc)


def maxsim_varlen(
    Q_packed: torch.Tensor,
    D_packed: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_d: torch.Tensor,
    max_seqlen_q: int | None = None,
    max_seqlen_d: int | None = None,
) -> torch.Tensor:
    """Run MaxSim on packed (no-padding) inputs.

    Args:
        Q_packed: [sum(Lq_i), d] fp16/bf16/fp32.
        D_packed: [sum(Ld_j), d].
        cu_seqlens_q: [Nq+1] int32, cumulative sums so seq i is
            `Q_packed[cu[i] : cu[i+1]]`.
        cu_seqlens_d: [Nd+1] int32, same for docs.
        max_seqlen_q / max_seqlen_d: hints for tile counts. Computed from
            cu_seqlens if not passed.

    Returns:
        scores: [Nq, Nd] fp32.
    """
    assert Q_packed.dim() == 2 and D_packed.dim() == 2
    d = Q_packed.shape[1]
    assert D_packed.shape[1] == d

    cu_seqlens_q = cu_seqlens_q.to(torch.int32).contiguous()
    cu_seqlens_d = cu_seqlens_d.to(torch.int32).contiguous()
    Nq = cu_seqlens_q.numel() - 1
    Nd = cu_seqlens_d.numel() - 1

    if max_seqlen_q is None:
        max_seqlen_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item())
    if max_seqlen_d is None:
        max_seqlen_d = int((cu_seqlens_d[1:] - cu_seqlens_d[:-1]).max().item())

    d_pad = next_pow2(d)
    compute_dtype = pick_compute_dtype(Q_packed, D_packed)
    tl_dtype = tl.float16 if compute_dtype == torch.float16 else tl.bfloat16

    scores = torch.zeros(Nq, Nd, device=Q_packed.device, dtype=torch.float32)

    Q_packed = Q_packed.contiguous()
    D_packed = D_packed.contiguous()

    _varlen_fwd_kernel[(Nq * Nd,)](
        Q_packed,
        D_packed,
        cu_seqlens_q,
        cu_seqlens_d,
        scores,
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
        max_seqlen_q,
        max_seqlen_d,  # Lq, Ld placeholders
        COMPUTE_DTYPE=tl_dtype,
    )
    return scores
