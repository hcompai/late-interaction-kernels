"""Top-k MaxSim. Used by :func:`late_interaction_kernels.retrieve`.

Wraps :func:`maxsim_forward` + :func:`torch.topk`, optionally chunked
over docs so peak HBM stays at ``O(Nq · (chunk + k))``.
"""

import torch

from late_interaction_kernels.forward import maxsim_forward


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
    """Top-``k`` documents per query. Used by :func:`retrieve`."""
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
