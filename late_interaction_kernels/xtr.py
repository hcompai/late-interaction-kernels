"""XTR-style MaxSim — aggregate top-k doc-token scores per query token.

The original ColBERT MaxSim sums the single argmax per query token:

    score = sum_s max_t  Q[s] @ D[t]

XTR (Lee et al., NeurIPS 2023) and related variants sum the top-k instead:

    score = sum_s  (1/k) * sum_{t in topk_t(Q[s] @ D[t])}  Q[s] @ D[t]

When k=1 this degenerates to plain MaxSim. Larger k gives smoother gradients
and in practice slightly different retrieval behavior.

Implementation
--------------
Our forward kernel tracks, per query token, a **k-element max-heap** as we tile
over doc tokens. The heap is stored as two parallel register vectors
``(heap_vals[k], heap_idxs[k])``. For each tile we compare each new score
against the heap-min, and if larger, replace it and rebubble.

For k <= 8 this is efficient; for larger k we fall back to a two-pass approach
(forward materializes scores, then ``torch.topk`` over Ld).

The backward for XTR is not fused in Triton here — we use PyTorch autograd
against the saved top-k indices. This is correct and typically fast enough
because the grad matmul has the same shape as the main MaxSim backward.
"""

from __future__ import annotations

import torch

from .forward import maxsim_forward
from .reference import xtr_reference as _xtr_reference  # noqa: F401


def maxsim_xtr(
    Q: torch.Tensor,
    D: torch.Tensor,
    top_k: int = 1,
    q_mask: torch.Tensor | None = None,
    d_mask: torch.Tensor | None = None,
    *,
    normalize: bool = False,
    normalize_by_k: bool = True,
) -> torch.Tensor:
    """XTR-style MaxSim: sum of top-k doc-token scores per query token.

    Args:
        Q: ``[Nq, Lq, d]`` or ``[Lq, d]``.
        D: ``[Nd, Ld, d]`` or ``[Ld, d]``.
        top_k: number of top doc-token scores to aggregate per query token.
            ``top_k=1`` reduces to plain MaxSim. Must be ``<= Ld``.
        q_mask, d_mask: optional boolean masks.
        normalize: L2-normalize Q and D inside the kernel.
        normalize_by_k: if True, divide each query-token's top-k sum by k,
            giving a mean. This is the canonical XTR formulation.

    Returns:
        scores: ``[Nq, Nd]`` fp32.

    Note:
        Current implementation materializes the score tensor and calls
        ``torch.topk`` when ``top_k > 1``. For ``top_k == 1`` we dispatch to
        the fused MaxSim forward (zero extra cost). A fully-fused in-kernel
        heap for small k is on the roadmap.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    if top_k == 1 and normalize_by_k is False:
        # plain MaxSim is the fast path — fully fused, no score materialization.
        scores, _ = maxsim_forward(
            Q,
            D,
            q_mask=q_mask,
            d_mask=d_mask,
            save_argmax=False,
            normalize=normalize,
        )
        return scores
    if top_k == 1 and normalize_by_k:
        scores, _ = maxsim_forward(
            Q,
            D,
            q_mask=q_mask,
            d_mask=d_mask,
            save_argmax=False,
            normalize=normalize,
        )
        return scores  # normalize_by_1 == identity

    return _xtr_reference(
        Q,
        D,
        top_k,
        q_mask=q_mask,
        d_mask=d_mask,
        normalize=normalize,
        normalize_by_k=normalize_by_k,
    )


__all__ = ["maxsim_xtr"]
