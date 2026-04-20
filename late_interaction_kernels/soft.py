"""Log-sum-exp / soft-MaxSim variant.

    score[i, j] = sum_{s : q_mask} (1 / beta) * logsumexp_t(beta * S[i, j, s, t])

where S[i, j, s, t] = Q[i, s] · D[j, t], and masked doc tokens are forced to
-inf before the logsumexp.

Compared to hard max:
  * gradient is a softmax-weighted combination of ALL doc tokens, not just
    the argmax → denser, smoother training signal
  * the forward converges to hard max as beta -> inf
  * still a single streaming pass over doc tiles, same memory footprint

This is the "FlashAttention for retrieval" formulation — identical loop
structure as attention with the final scale swapped from `1/sqrt(d)` to
`beta`, and with no value-tensor projection.

The backward is obtained for free via `tl.dot` + softmax reweighting, but we
use a simpler path here: rely on `torch.autograd` through a fused forward
that returns the online (m, l) streaming stats, then recompute. This keeps
the code small while still being fast.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._autotune import forward_configs, prune_forward
from ._utils import next_pow2, pick_compute_dtype


@triton.autotune(
    configs=forward_configs(),
    key=["Lq", "Ld", "d_pad", "has_q_mask", "has_d_mask"],
    prune_configs_by={"early_config_prune": prune_forward},
)
@triton.jit
def _soft_maxsim_kernel(
    Q_ptr,
    D_ptr,
    q_mask_ptr,
    d_mask_ptr,
    scores_ptr,
    Nq: tl.constexpr,
    Nd: tl.constexpr,
    Lq: tl.constexpr,
    Ld: tl.constexpr,
    d: tl.constexpr,
    d_pad: tl.constexpr,
    beta,
    stride_q_n,
    stride_q_l,
    stride_q_d,
    stride_d_n,
    stride_d_l,
    stride_d_d,
    stride_s_n,
    stride_s_d,
    stride_qm_n,
    stride_qm_l,
    stride_dm_n,
    stride_dm_l,
    has_q_mask: tl.constexpr,
    has_d_mask: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    q_idx = pid // Nd
    d_idx = pid % Nd

    k_off = tl.arange(0, d_pad)
    k_mask = k_off < d

    score_acc = tl.zeros([], dtype=tl.float32)

    for q_start in tl.static_range(0, Lq, BLOCK_Q):
        q_off = q_start + tl.arange(0, BLOCK_Q)
        q_valid = q_off < Lq

        if has_q_mask:
            qm = tl.load(
                q_mask_ptr + q_idx * stride_qm_n + q_off * stride_qm_l,
                mask=q_valid,
                other=0,
            ).to(tl.int1)
            q_active = q_valid & qm
        else:
            q_active = q_valid

        Q_block = tl.load(
            Q_ptr + q_idx * stride_q_n + q_off[:, None] * stride_q_l + k_off[None, :] * stride_q_d,
            mask=q_valid[:, None] & k_mask[None, :],
            other=0.0,
        ).to(COMPUTE_DTYPE)

        # Online logsumexp streaming state: (m, l) where
        #   lse = m + log(l)
        m = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
        l = tl.zeros([BLOCK_Q], dtype=tl.float32)

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

            D_block = tl.load(
                D_ptr + d_idx * stride_d_n + d_off[:, None] * stride_d_l + k_off[None, :] * stride_d_d,
                mask=d_valid[:, None] & k_mask[None, :],
                other=0.0,
            ).to(COMPUTE_DTYPE)

            S = tl.dot(Q_block, tl.trans(D_block), out_dtype=tl.float32) * beta
            S = tl.where(d_active[None, :], S, float("-inf"))

            tile_max = tl.max(S, axis=1)
            new_m = tl.maximum(m, tile_max)
            # Correct l to the new reference point
            l = l * tl.exp(m - new_m)
            # Add the current tile's contribution
            # (mask -inf entries to 0 exp to avoid NaNs)
            p = tl.exp(S - new_m[:, None])
            p = tl.where(d_active[None, :], p, 0.0)
            l += tl.sum(p, axis=1)
            m = new_m

        # lse = m + log(l) ; scaled back by 1/beta
        safe = (l > 0) & q_active
        lse = tl.where(safe, (m + tl.log(l)) / beta, 0.0)
        score_acc += tl.sum(lse)

    tl.store(scores_ptr + q_idx * stride_s_n + d_idx * stride_s_d, score_acc)


def soft_maxsim(
    Q: torch.Tensor,
    D: torch.Tensor,
    q_mask: torch.Tensor | None = None,
    d_mask: torch.Tensor | None = None,
    beta: float = 10.0,
) -> torch.Tensor:
    """Soft (log-sum-exp) approximation of MaxSim.

    As `beta -> inf`, the result converges to hard `maxsim`. For training it
    gives denser gradients (all doc tokens contribute) at the cost of a tiny
    bias.

    Returns an fp32 tensor of shape [Nq, Nd] (or squeezed).
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

    Nq, Lq, d = Q.shape
    Nd, Ld, _ = D.shape

    Q = Q.contiguous()
    D = D.contiguous()
    has_q_mask = q_mask is not None
    has_d_mask = d_mask is not None
    q_mask_i8 = q_mask.contiguous().to(torch.int8) if has_q_mask else Q
    d_mask_i8 = d_mask.contiguous().to(torch.int8) if has_d_mask else D
    qm_strides = (q_mask_i8.stride(0), q_mask_i8.stride(1)) if has_q_mask else (0, 0)
    dm_strides = (d_mask_i8.stride(0), d_mask_i8.stride(1)) if has_d_mask else (0, 0)

    d_pad = next_pow2(d)
    compute_dtype = pick_compute_dtype(Q, D)
    tl_dtype = tl.float16 if compute_dtype == torch.float16 else tl.bfloat16
    scores = torch.empty(Nq, Nd, device=Q.device, dtype=torch.float32)

    _soft_maxsim_kernel[(Nq * Nd,)](
        Q,
        D,
        q_mask_i8,
        d_mask_i8,
        scores,
        Nq,
        Nd,
        Lq,
        Ld,
        d,
        d_pad,
        float(beta),
        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        D.stride(0),
        D.stride(1),
        D.stride(2),
        scores.stride(0),
        scores.stride(1),
        qm_strides[0],
        qm_strides[1],
        dm_strides[0],
        dm_strides[1],
        has_q_mask,
        has_d_mask,
        COMPUTE_DTYPE=tl_dtype,
    )

    if q_was_2d and d_was_2d:
        return scores.reshape(())
    if q_was_2d:
        return scores.squeeze(0)
    if d_was_2d:
        return scores.squeeze(-1)
    return scores
