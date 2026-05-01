"""Log-sum-exp (soft) MaxSim variant.

    score[i, j] = sum_{s : q_mask} (1 / beta) · logsumexp_t(beta · ⟨Q[i,s], D[j,t]⟩)

``beta -> inf`` recovers hard MaxSim. The backward is a softmax-weighted
combination of all doc tokens (denser gradient than the argmax-only hard
backward).
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


def _soft_maxsim_reference(
    Q: torch.Tensor,
    D: torch.Tensor,
    q_mask: torch.Tensor | None,
    d_mask: torch.Tensor | None,
    beta: float,
) -> torch.Tensor:
    """Pure-PyTorch reference implementation of soft-MaxSim.

    Used for:
      * fp64 inputs (Triton tensor-core dot doesn't support fp64)
      * CPU inputs (no CUDA required)
      * the autograd backward path
    """
    # Always accumulate in at least fp32 to match the Triton path's output
    # dtype and numerical behavior. fp64 inputs stay in fp64 for gradcheck.
    acc_dtype = Q.dtype if Q.dtype == torch.float64 else torch.float32
    S = torch.einsum("ild,jtd->ijlt", Q.to(acc_dtype), D.to(acc_dtype)) * beta
    if d_mask is not None:
        S = S.masked_fill(~d_mask.bool()[None, :, None, :], float("-inf"))
    lse = torch.logsumexp(S, dim=-1) / beta  # [Nq, Nd, Lq]
    if q_mask is not None:
        lse = lse.masked_fill(~q_mask.bool()[:, None, :], 0.0)
    return lse.sum(dim=-1)


def _soft_maxsim_triton(
    Q: torch.Tensor,
    D: torch.Tensor,
    q_mask: torch.Tensor | None,
    d_mask: torch.Tensor | None,
    beta: float,
) -> torch.Tensor:
    Nq, Lq, d = Q.shape
    Nd, Ld, _ = D.shape

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
    return scores


class _SoftMaxSimFn(torch.autograd.Function):
    """Autograd wrapper: Triton forward (fast) + PyTorch reference backward.

    The backward recomputes the per-pair softmax weights in fp32 on the fly.
    This is O(Nq·Nd·Lq·Ld·d) FLOPs but avoids materializing the [Nq,Nd,Lq,Ld]
    attention tensor in the forward — we only keep it alive briefly during
    the backward. Good enough for PyLate training regimes where Ld ≤ 512.
    """

    @staticmethod
    def forward(ctx, Q, D, q_mask, d_mask, beta):
        # Only use the Triton kernel for fp16 / bf16 — the tensor-core dot
        # uses an fp16 accumulator for fp32 inputs (see `pick_compute_dtype`),
        # which would make the forward inconsistent with the fp32 backward.
        # For fp32 / fp64 / CPU, fall back to the pure-PyTorch reference so
        # `torch.autograd.gradcheck` sees matching precision on both sides.
        use_triton = Q.is_cuda and Q.dtype in (torch.float16, torch.bfloat16) and D.dtype == Q.dtype
        if use_triton:
            scores = _soft_maxsim_triton(Q, D, q_mask, d_mask, beta)
        else:
            scores = _soft_maxsim_reference(Q, D, q_mask, d_mask, beta)
        ctx.save_for_backward(Q, D, q_mask, d_mask)
        ctx.beta = float(beta)
        return scores

    @staticmethod
    def backward(ctx, grad_scores):
        Q, D, q_mask, d_mask = ctx.saved_tensors
        beta = ctx.beta

        # Recompute in at least fp32 for a stable, deterministic backward —
        # the soft path is smooth in every (s, t) pair so the grad is dense
        # and we can afford one reference pass. fp64 inputs stay in fp64
        # so `torch.autograd.gradcheck` sees a matching precision on both
        # sides of the finite-difference comparison.
        acc_dtype = Q.dtype if Q.dtype == torch.float64 else torch.float32
        Qf = Q.to(acc_dtype)
        Df = D.to(acc_dtype)

        S = torch.einsum("ild,jtd->ijlt", Qf, Df) * beta  # [Nq, Nd, Lq, Ld]
        if d_mask is not None:
            S = S.masked_fill(~d_mask.bool()[None, :, None, :], float("-inf"))

        # softmax over doc-token dimension — these are the weights that
        # each (query_token, doc_token) pair contributes to the score.
        W = torch.softmax(S, dim=-1)  # [Nq, Nd, Lq, Ld]
        if q_mask is not None:
            W = W * q_mask.bool()[:, None, :, None].to(W.dtype)

        # Distribute grad_scores ∈ [Nq, Nd] back through the sum over Lq
        # and the softmax over Ld. Grad w.r.t. (query_token s, dim d) is
        #   Σ_{j, t} grad_scores[i, j] · W[i, j, s, t] · D[j, t, d].
        g = grad_scores.to(acc_dtype)  # [Nq, Nd]
        # [Nq, Nd, Lq, Ld] * [Nq, Nd, 1, 1]
        Wg = W * g[:, :, None, None]
        grad_Q = torch.einsum("ijlt,jtd->ild", Wg, Df)
        grad_D = torch.einsum("ijlt,ild->jtd", Wg, Qf)

        return grad_Q.to(Q.dtype), grad_D.to(D.dtype), None, None, None


def soft_maxsim(
    Q: torch.Tensor,
    D: torch.Tensor,
    q_mask: torch.Tensor | None = None,
    d_mask: torch.Tensor | None = None,
    beta: float = 10.0,
) -> torch.Tensor:
    """Soft (log-sum-exp) approximation of MaxSim, autograd-aware.

    As ``beta -> inf``, the result converges to hard :func:`maxsim`. During
    training this gives denser gradients (every doc token contributes
    through the softmax) at the cost of a tiny forward bias.

    Forward uses the Triton kernel when possible; backward is a stable
    fp32 PyTorch recomputation of the softmax weights — deterministic and
    autograd-checked.

    Returns an fp32 tensor of shape ``[Nq, Nd]`` (squeezed to match 2-D
    inputs, like :func:`maxsim`).
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

    if Q.shape[-1] != D.shape[-1]:
        raise ValueError(
            f"Q and D must share the embedding dim; got Q.shape[-1]={Q.shape[-1]} "
            f"vs D.shape[-1]={D.shape[-1]}."
        )
    if Q.device != D.device:
        raise ValueError(
            f"Q and D must be on the same device; got Q.device={Q.device} vs D.device={D.device}."
        )

    Q = Q.contiguous()
    D = D.contiguous()
    q_mask_c = q_mask.contiguous().to(torch.bool) if q_mask is not None else None
    d_mask_c = d_mask.contiguous().to(torch.bool) if d_mask is not None else None

    scores = _SoftMaxSimFn.apply(Q, D, q_mask_c, d_mask_c, beta)

    if q_was_2d and d_was_2d:
        return scores.reshape(())
    if q_was_2d:
        return scores.squeeze(0)
    if d_was_2d:
        return scores.squeeze(-1)
    return scores
