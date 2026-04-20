"""Pure-PyTorch reference implementations of MaxSim.

These are the ground truth against which all Triton kernels are validated.
They are slow and memory-hungry by design — they are not meant to be fast.

Shapes (following PyLate conventions):
    Q: [Nq, Lq, d]   query-side token embeddings
    D: [Nd, Ld, d]   document-side token embeddings
    q_mask: [Nq, Lq]  bool — False positions are ignored on the SUM side
    d_mask: [Nd, Ld]  bool — False positions are ignored on the MAX side
                       (they get -inf before the max reduction)

The scalar score formula is:

    score[i, j] = sum_{s : q_mask[i, s]} max_{t : d_mask[j, t]} Q[i, s] @ D[j, t]

For symmetry with PyLate's `colbert_scores` we return a [Nq, Nd] matrix.
If either tensor is 2-D we reshape to 3-D with a leading 1 and squeeze back.
"""

from __future__ import annotations

import torch

NEG_INF = float("-inf")


def maxsim_reference(
    Q: torch.Tensor,
    D: torch.Tensor,
    q_mask: torch.Tensor | None = None,
    d_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense reference MaxSim with full mask support.

    Fully materializes the [Nq, Nd, Lq, Ld] similarity tensor. Correct, but slow.

    Args:
        Q: [Nq, Lq, d] or [Lq, d] query embeddings.
        D: [Nd, Ld, d] or [Ld, d] document embeddings.
        q_mask: [Nq, Lq] or [Lq] boolean mask. True = keep, False = drop from sum.
        d_mask: [Nd, Ld] or [Ld] boolean mask. True = keep, False = mask out of max.

    Returns:
        scores: [Nq, Nd] float tensor. If inputs were 2-D the matching dim is
                squeezed out.
    """
    q_squeeze = Q.dim() == 2
    d_squeeze = D.dim() == 2
    if q_squeeze:
        Q = Q.unsqueeze(0)
    if d_squeeze:
        D = D.unsqueeze(0)

    Nq, Lq, d = Q.shape
    Nd, Ld, d2 = D.shape
    if d != d2:
        raise ValueError(f"embedding dims don't match: Q has d={d}, D has d={d2}")

    S = torch.einsum("ild,jtd->ijlt", Q.float(), D.float())  # [Nq, Nd, Lq, Ld]

    if d_mask is not None:
        if d_mask.dim() == 1:
            d_mask = d_mask.unsqueeze(0)
        # -inf on masked doc tokens so they lose the max
        S = S.masked_fill(~d_mask.bool()[None, :, None, :], NEG_INF)

    # If an entire row is -inf (whole doc masked) max() would give -inf; clamp to 0.
    row_max = S.max(dim=-1).values  # [Nq, Nd, Lq]
    row_max = torch.where(
        torch.isfinite(row_max), row_max, torch.zeros_like(row_max)
    )

    if q_mask is not None:
        if q_mask.dim() == 1:
            q_mask = q_mask.unsqueeze(0)
        row_max = row_max * q_mask.to(row_max.dtype)[:, None, :]

    scores = row_max.sum(dim=-1)  # [Nq, Nd]

    if q_squeeze:
        scores = scores.squeeze(0)
    if d_squeeze:
        scores = scores.squeeze(-1)
    return scores


def maxsim_reference_soft(
    Q: torch.Tensor,
    D: torch.Tensor,
    q_mask: torch.Tensor | None = None,
    d_mask: torch.Tensor | None = None,
    beta: float = 10.0,
) -> torch.Tensor:
    """Log-sum-exp relaxation of MaxSim: smooth approximation of max.

        softmax_t(beta * S) -> max when beta -> inf.

    Score: sum_s (1/beta) * logsumexp_t(beta * S[s, t]).

    This variant is differentiable through ALL query-document pairs (not just
    the argmax), giving denser gradients during training. At eval time you can
    either pick a large beta (scores approach hard max) or swap back to
    `maxsim_reference`.
    """
    q_squeeze = Q.dim() == 2
    d_squeeze = D.dim() == 2
    if q_squeeze:
        Q = Q.unsqueeze(0)
    if d_squeeze:
        D = D.unsqueeze(0)

    S = torch.einsum("ild,jtd->ijlt", Q.float(), D.float())

    if d_mask is not None:
        if d_mask.dim() == 1:
            d_mask = d_mask.unsqueeze(0)
        S = S.masked_fill(~d_mask.bool()[None, :, None, :], NEG_INF)

    # (1/beta) * logsumexp(beta * S, dim=-1)
    row_soft = (1.0 / beta) * torch.logsumexp(beta * S, dim=-1)  # [Nq, Nd, Lq]
    row_soft = torch.where(
        torch.isfinite(row_soft), row_soft, torch.zeros_like(row_soft)
    )

    if q_mask is not None:
        if q_mask.dim() == 1:
            q_mask = q_mask.unsqueeze(0)
        row_soft = row_soft * q_mask.to(row_soft.dtype)[:, None, :]

    scores = row_soft.sum(dim=-1)
    if q_squeeze:
        scores = scores.squeeze(0)
    if d_squeeze:
        scores = scores.squeeze(-1)
    return scores


def maxsim_reference_varlen(
    Q: torch.Tensor,
    D: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_d: torch.Tensor,
) -> torch.Tensor:
    """Packed/varlen reference: Q/D are 2-D tensors of stacked token embeddings.

    Args:
        Q: [total_q_tokens, d]
        D: [total_d_tokens, d]
        cu_seqlens_q: [Nq + 1] cumulative query token counts (int32).
        cu_seqlens_d: [Nd + 1] cumulative doc   token counts (int32).

    Returns:
        scores: [Nq, Nd]
    """
    Nq = cu_seqlens_q.numel() - 1
    Nd = cu_seqlens_d.numel() - 1
    out = torch.empty(Nq, Nd, device=Q.device, dtype=torch.float32)
    for i in range(Nq):
        qi = Q[cu_seqlens_q[i].item() : cu_seqlens_q[i + 1].item()].float()
        if qi.shape[0] == 0:
            out[i].zero_()
            continue
        for j in range(Nd):
            dj = D[cu_seqlens_d[j].item() : cu_seqlens_d[j + 1].item()].float()
            if dj.shape[0] == 0:
                out[i, j] = 0.0
                continue
            S = qi @ dj.T  # [lq, ld]
            out[i, j] = S.max(dim=-1).values.sum()
    return out
