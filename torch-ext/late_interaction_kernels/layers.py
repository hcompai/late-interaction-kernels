"""Stateless ``nn.Module`` layers for HF Kernels ``LayerRepository`` / ``kernelize``.

Layers must satisfy the HF Kernels contract: pure ``nn.Module`` subclasses, no
``__init__``, only class-level type annotations, and (optionally)
``has_backward`` / ``can_torch_compile`` class attributes.
"""

from __future__ import annotations

import torch
from torch import nn

from .autograd import maxsim_inference


class MaxSim(nn.Module):
    """Forward-only late-interaction MaxSim scoring.

    Scores a batch of query token embeddings against a batch of document token
    embeddings. ColBERT / ColPali / LateOn all use MaxSim with L2-normalized
    embeddings, so ``normalize=True`` is fused into the kernel and is the
    recommended path.

    Inputs:
        Q: ``[Nq, Lq, d]`` or ``[Lq, d]`` — query token embeddings.
        D: ``[Nd, Ld, d]`` or ``[Ld, d]`` — document token embeddings.
        q_mask: optional bool tensor matching Q's first two dims.
        d_mask: optional bool tensor matching D's first two dims.

    Returns:
        scores: ``[Nq, Nd]`` fp32.
    """

    has_backward: bool = False
    can_torch_compile: bool = True

    def forward(
        self,
        Q: torch.Tensor,
        D: torch.Tensor,
        q_mask: torch.Tensor | None = None,
        d_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return maxsim_inference(Q, D, q_mask=q_mask, d_mask=d_mask, normalize=True)


__all__ = ["MaxSim"]
