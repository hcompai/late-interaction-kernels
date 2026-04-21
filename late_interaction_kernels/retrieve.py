"""High-level retrieval API — :class:`MaxSimScorer` and :func:`retrieve`.

These are the batteries-included entry points for users who want to use
late-interaction-kernels without knowing anything about the low-level
primitives. If you only remember two symbols from this package, these
are the ones::

    from late_interaction_kernels import MaxSimScorer, retrieve

Both default to ``normalize=True`` because ColBERT / ColPali / LateOn
always score L2-normalized token embeddings — matching vanilla PyLate /
FastPlaid out of the box. Override with ``normalize=False`` if you're
working on unnormalized inputs (e.g., DPR-style single-vector scores).
"""

from __future__ import annotations

from typing import Literal

import torch

from .autograd import _VALID_METHODS, maxsim, maxsim_inference
from .topk import maxsim_topk

BackwardMethod = Literal["auto", "unified", "csr", "atomic"]


class MaxSimScorer(torch.nn.Module):
    """Late-interaction (MaxSim) scorer as an ``nn.Module``.

    Wraps :func:`maxsim` / :func:`maxsim_inference` with the usual
    ergonomic defaults:

    * ``normalize=True`` — match ColBERT / ColPali / LateOn behavior.
    * ``backward="auto"`` — let the library pick the best ``grad_D``
      path per call.
    * device / dtype / mask checks live here so they're not re-implemented
      at every call site.

    Example::

        scorer = MaxSimScorer()                       # normalize=True, autograd on
        scores = scorer(Q, D, q_mask=q_mask, d_mask=d_mask)  # [Nq, Nd]
        scores.mean().backward()                      # gradients flow into Q, D

    For reranking (no grad), wrap in ``torch.no_grad()`` or use
    :meth:`score` explicitly — both skip the saved argmax::

        with torch.no_grad():
            scores = scorer.score(Q_enc, D_enc)       # fast inference path

    The module has no learnable parameters; it's a stateless scorer that
    composes cleanly with any encoder ``nn.Module``.

    Args:
        normalize: L2-normalize Q and D per-token inside the kernel.
            Default ``True``. Set ``False`` only if you are scoring
            already-normalized or intentionally unnormalized embeddings
            (and don't want the one-time "looks unnormalized" warning).
        backward: per-call ``grad_D`` selector —
            ``"auto" | "unified" | "csr" | "atomic"``.
        mask_pad_token: if given, :meth:`forward` can derive the mask from
            a token-id tensor by comparing against this id. Convenience
            for users who pass ``input_ids`` instead of a precomputed
            bool mask. Use :meth:`forward_with_ids` instead.
    """

    def __init__(
        self,
        *,
        normalize: bool = True,
        backward: BackwardMethod = "auto",
        mask_pad_token: int | None = None,
    ) -> None:
        super().__init__()
        if backward not in _VALID_METHODS:
            raise ValueError(f"backward= must be one of {_VALID_METHODS}, got {backward!r}")
        self.normalize = normalize
        self.backward = backward
        self.mask_pad_token = mask_pad_token

    def extra_repr(self) -> str:
        return f"normalize={self.normalize}, backward={self.backward!r}"

    def forward(
        self,
        Q: torch.Tensor,
        D: torch.Tensor,
        q_mask: torch.Tensor | None = None,
        d_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute MaxSim scores. Autograd-aware.

        Args:
            Q: ``[Nq, Lq, d]`` or ``[Lq, d]``.
            D: ``[Nd, Ld, d]`` or ``[Ld, d]``.
            q_mask, d_mask: optional bool masks. ``True`` = valid token.

        Returns:
            ``[Nq, Nd]`` fp32 scores (squeezed to match 2-D inputs).
        """
        return maxsim(
            Q,
            D,
            q_mask=q_mask,
            d_mask=d_mask,
            normalize=self.normalize,
            backward=self.backward,
        )

    def score(
        self,
        Q: torch.Tensor,
        D: torch.Tensor,
        q_mask: torch.Tensor | None = None,
        d_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Inference-only MaxSim (no saved argmax). Does **not** participate in autograd."""
        with torch.no_grad():
            return maxsim_inference(
                Q,
                D,
                q_mask=q_mask,
                d_mask=d_mask,
                normalize=self.normalize,
            )

    def retrieve(
        self,
        Q: torch.Tensor,
        D: torch.Tensor,
        top_k: int,
        q_mask: torch.Tensor | None = None,
        d_mask: torch.Tensor | None = None,
        *,
        chunk: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(top_k_scores, top_k_indices)`` per query. Inference-only.

        See :func:`retrieve` for argument semantics. This method is a
        thin wrapper that threads the scorer's ``normalize`` through.
        """
        return retrieve(
            Q,
            D,
            top_k=top_k,
            q_mask=q_mask,
            d_mask=d_mask,
            normalize=self.normalize,
            chunk=chunk,
        )

    def forward_with_ids(
        self,
        Q: torch.Tensor,
        D: torch.Tensor,
        q_ids: torch.Tensor,
        d_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Like :meth:`forward`, but derive masks from token-id tensors.

        Requires ``mask_pad_token`` to be set on the scorer. Valid tokens
        are those != ``mask_pad_token``.
        """
        if self.mask_pad_token is None:
            raise ValueError(
                "`forward_with_ids` requires `mask_pad_token` to be set on the "
                "scorer (e.g. `MaxSimScorer(mask_pad_token=tokenizer.pad_token_id)`)."
            )
        q_mask = q_ids != self.mask_pad_token
        d_mask = d_ids != self.mask_pad_token
        return self.forward(Q, D, q_mask=q_mask, d_mask=d_mask)


def retrieve(
    Q: torch.Tensor,
    D: torch.Tensor,
    top_k: int,
    q_mask: torch.Tensor | None = None,
    d_mask: torch.Tensor | None = None,
    *,
    normalize: bool = True,
    chunk: int | None = None,
    largest: bool = True,
    sorted: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-``top_k`` late-interaction retrieval — "how do I search 100k docs" in one call.

    ::

        from late_interaction_kernels import retrieve
        scores, indices = retrieve(Q, D, top_k=100)

    Semantically equivalent to ``maxsim(Q, D).topk(top_k)`` but:

    1. Fuses the forward into a single Triton kernel (no materialized
       ``[Nq, Nd]`` tensor beyond the top-k slice when chunked).
    2. Supports ``chunk=`` to cap HBM usage at ``Nq · (chunk + top_k)``
       instead of ``Nq · Nd`` when ``Nd`` is very large.
    3. Defaults to ``normalize=True`` to match ColBERT / ColPali / LateOn
       scoring semantics.

    Args:
        Q: query embeddings ``[Nq, Lq, d]`` or ``[Lq, d]``.
        D: document embeddings ``[Nd, Ld, d]`` or ``[Ld, d]``.
        top_k: number of results per query. Clamped to ``min(top_k, Nd)``.
        q_mask, d_mask: optional bool masks. ``True`` = valid token.
        normalize: L2-normalize Q / D inside the kernel. Default ``True``.
        chunk: if given, score the corpus in doc-chunks of this size and
            merge top-k on the fly. Use when ``Nq · Nd`` would OOM.
        largest, sorted: same semantics as :func:`torch.topk`.

    Returns:
        ``(top_k_scores, top_k_indices)`` both shape ``[Nq, top_k]``
        (or ``[top_k]`` if Q was 2-D). Scores fp32, indices int64.

    Notes:
        * This entry point is inference-only — it does not save argmax
          and is not autograd-aware. Use :class:`MaxSimScorer` + manual
          topk if you need gradients on the retrieval path.
        * For varlen / packed corpora, materialize via
          :func:`late_interaction_kernels.maxsim_varlen` directly; a
          varlen-aware ``retrieve`` entry point is tracked for a future
          release.
    """
    return maxsim_topk(
        Q,
        D,
        k=top_k,
        q_mask=q_mask,
        d_mask=d_mask,
        normalize=normalize,
        chunk_size=chunk,
        largest=largest,
        sorted=sorted,
    )


__all__ = ["MaxSimScorer", "retrieve"]
