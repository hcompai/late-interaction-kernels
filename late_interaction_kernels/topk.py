"""Top-k aggregation on top of fused MaxSim.

This module exposes a single public function, :func:`maxsim_topk`, that returns
the top-k documents (by MaxSim score) per query in one call. It fuses the
common reranking post-processing ``maxsim(Q, D).topk(k)`` into a single function
call that allocates only the ``[Nq, k]`` outputs in HBM.

For very large corpora (``Nq * Nd`` too big to hold in HBM), we still compute
scores in chunks of documents and reduce the top-k as we go, so peak memory is
``O(Nq * (chunk + k))`` rather than ``O(Nq * Nd)``.

The current implementation is a thin wrapper around the fused MaxSim forward.
A future release can swap it for a fully-fused in-kernel heap reduction
without changing the API.
"""

from __future__ import annotations

import torch

from .forward import maxsim_forward


def maxsim_topk(
    Q: torch.Tensor,
    D: torch.Tensor,
    k: int,
    q_mask: torch.Tensor | None = None,
    d_mask: torch.Tensor | None = None,
    *,
    normalize: bool = False,
    chunk_size: int | None = None,
    largest: bool = True,
    sorted: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the top-k scoring documents per query, plus their indices.

    Args:
        Q: ``[Nq, Lq, d]`` or ``[Lq, d]`` query embeddings.
        D: ``[Nd, Ld, d]`` or ``[Ld, d]`` document embeddings.
        k: number of top results per query. Clamped to ``min(k, Nd)``.
        q_mask, d_mask: optional boolean masks matching the first two dims.
        normalize: L2-normalize Q and D inside the kernel (fused).
        chunk_size: if given, compute scores in doc-chunks of this size and
            merge top-k on the fly. Use when ``Nq * Nd`` doesn't fit in HBM
            as an intermediate. Defaults to processing everything at once.
        largest, sorted: same semantics as :func:`torch.topk`.

    Returns:
        ``(scores_topk, indices_topk)`` both shape ``[Nq, k]`` (or ``[k]`` if
        Q was 2-D). Scores are fp32, indices are int64.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    q_was_2d = Q.dim() == 2
    if q_was_2d:
        Q = Q.unsqueeze(0)
    if D.dim() == 2:
        D = D.unsqueeze(0)
    if q_mask is not None and q_mask.dim() == 1:
        q_mask = q_mask.unsqueeze(0)
    if d_mask is not None and d_mask.dim() == 1:
        d_mask = d_mask.unsqueeze(0)

    Nd = D.shape[0]
    k = min(k, Nd)

    if chunk_size is None or chunk_size >= Nd:
        scores, _ = maxsim_forward(
            Q,
            D,
            q_mask=q_mask,
            d_mask=d_mask,
            save_argmax=False,
            normalize=normalize,
        )
        topk_scores, topk_idx = torch.topk(scores, k, dim=-1, largest=largest, sorted=sorted)
    else:
        topk_scores = None
        topk_idx = None
        for start in range(0, Nd, chunk_size):
            end = min(start + chunk_size, Nd)
            D_chunk = D[start:end]
            d_mask_chunk = d_mask[start:end] if d_mask is not None else None
            s_chunk, _ = maxsim_forward(
                Q,
                D_chunk,
                q_mask=q_mask,
                d_mask=d_mask_chunk,
                save_argmax=False,
                normalize=normalize,
            )
            k_here = min(k, end - start)
            ch_s, ch_i = torch.topk(s_chunk, k_here, dim=-1, largest=largest, sorted=True)
            ch_i = ch_i + start
            if topk_scores is None:
                topk_scores, topk_idx = ch_s, ch_i
            else:
                cat_s = torch.cat([topk_scores, ch_s], dim=-1)
                cat_i = torch.cat([topk_idx, ch_i], dim=-1)
                topk_scores, gather_idx = torch.topk(
                    cat_s,
                    k,
                    dim=-1,
                    largest=largest,
                    sorted=sorted,
                )
                topk_idx = torch.gather(cat_i, -1, gather_idx)

    if q_was_2d:
        topk_scores = topk_scores.squeeze(0)
        topk_idx = topk_idx.squeeze(0)
    return topk_scores, topk_idx


__all__ = ["maxsim_topk"]
