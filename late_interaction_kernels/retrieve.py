"""High-level entry points: :class:`MaxSimScorer` and :func:`retrieve`.

Both default to ``normalize=True`` (ColBERT / ColPali / LateOn convention)
and dispatch by tensor device:

* CUDA + Triton → fused Triton kernels (:mod:`late_interaction_kernels.forward`),
* MPS (Apple Silicon) → ``torch.compile``-fused reference,
* CPU / anything else → eager PyTorch reference.

Every backend is autograd-aware, so training and retrieval code is
unit-testable on macOS / Windows before renting a CUDA box.
"""

from typing import Literal

import torch

BackwardMethod = Literal["auto", "unified", "csr", "atomic"]

_VALID_METHODS: tuple[str, ...] = ("auto", "unified", "csr", "atomic")

try:
    import triton  # noqa: F401

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False


def _score(
    Q: torch.Tensor,
    D: torch.Tensor,
    q_mask: torch.Tensor | None,
    d_mask: torch.Tensor | None,
    *,
    normalize: bool,
    backward: str,
    inference: bool,
) -> torch.Tensor:
    """Dispatch by device: Triton on CUDA, ``torch.compile`` on MPS, eager elsewhere.

    ``Q`` and ``D`` must live on the same device. Mixed-device pairs
    (e.g. ``Q`` on ``mps:0`` and ``D`` on ``cpu``) would otherwise drop
    through to the eager reference and surface a confusing internal
    ``RuntimeError`` from ``torch.matmul``; we raise an explicit
    ``ValueError`` up-front instead — same contract as :func:`retrieve`.
    """
    if Q.device != D.device:
        raise ValueError(
            f"Q and D must be on the same device; got Q.device={Q.device} vs D.device={D.device}."
        )

    if _HAS_TRITON and Q.is_cuda and D.is_cuda:
        from late_interaction_kernels.autograd import maxsim, maxsim_inference

        if inference:
            return maxsim_inference(Q, D, q_mask=q_mask, d_mask=d_mask, normalize=normalize)
        return maxsim(Q, D, q_mask=q_mask, d_mask=d_mask, normalize=normalize, backward=backward)

    if Q.device.type == "mps":
        from late_interaction_kernels.mps import maxsim_inference_mps, maxsim_mps

        if inference:
            return maxsim_inference_mps(Q, D, q_mask=q_mask, d_mask=d_mask, normalize=normalize)
        return maxsim_mps(Q, D, q_mask=q_mask, d_mask=d_mask, normalize=normalize)

    from late_interaction_kernels.reference import maxsim_reference

    with torch.no_grad() if inference else torch.enable_grad():
        return maxsim_reference(Q, D, q_mask=q_mask, d_mask=d_mask, normalize=normalize)


class MaxSimScorer(torch.nn.Module):
    """Stateless ``nn.Module`` for late-interaction (MaxSim) scoring.

    Defaults match ColBERT / ColPali / LateOn (``normalize=True``,
    ``backward="auto"``). The module has no learnable parameters; it
    composes cleanly with any encoder.

    Example::

        scorer = MaxSimScorer()
        scores = scorer(Q, D, q_mask=q_mask, d_mask=d_mask)  # [Nq, Nd]
        scores.mean().backward()                             # gradients flow

    For inference, use :meth:`score` (or wrap the call in ``torch.no_grad()``)
    to skip the saved argmax buffer.

    Args:
        normalize: L2-normalize Q and D per-token inside the kernel.
        backward: per-call ``grad_D`` strategy
            (``"auto" | "unified" | "csr" | "atomic"``).
        mask_pad_token: optional pad-token id; enables
            :meth:`forward_with_ids` to derive masks from token-id tensors.
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
        return (
            f"normalize={self.normalize}, backward={self.backward!r}, mask_pad_token={self.mask_pad_token!r}"
        )

    def forward(
        self,
        Q: torch.Tensor,
        D: torch.Tensor,
        q_mask: torch.Tensor | None = None,
        d_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute MaxSim scores ``[Nq, Nd]``. Autograd-aware."""
        return _score(
            Q,
            D,
            q_mask,
            d_mask,
            normalize=self.normalize,
            backward=self.backward,
            inference=False,
        )

    def score(
        self,
        Q: torch.Tensor,
        D: torch.Tensor,
        q_mask: torch.Tensor | None = None,
        d_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Inference-only MaxSim. Does not participate in autograd."""
        return _score(
            Q,
            D,
            q_mask,
            d_mask,
            normalize=self.normalize,
            backward=self.backward,
            inference=True,
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
        """``(top_k_scores, top_k_indices)`` per query. See :func:`retrieve`."""
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
        """Like :meth:`forward`, but derive masks from ``q_ids != mask_pad_token``."""
        if self.mask_pad_token is None:
            raise ValueError("`forward_with_ids` requires `mask_pad_token` set on the scorer.")
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
    """Top-``top_k`` late-interaction retrieval.

    ::

        scores, indices = retrieve(Q, D, top_k=100, chunk=4096)

    Equivalent to ``maxsim(Q, D).topk(top_k)``, plus a ``chunk=`` option to
    keep peak HBM at ``Nq · (chunk + top_k)`` when scoring large corpora.

    Args:
        Q: query embeddings ``[Nq, Lq, d]`` or ``[Lq, d]``.
        D: document embeddings ``[Nd, Ld, d]`` or ``[Ld, d]``.
        top_k: number of results per query, clamped to ``min(top_k, Nd)``.
        q_mask, d_mask: optional bool masks (``True`` = valid token).
        normalize: L2-normalize Q / D inside the kernel.
        chunk: if set, score in doc-chunks and merge top-k on the fly.
        largest, sorted: as in :func:`torch.topk`.

    Returns:
        ``(top_k_scores, top_k_indices)`` of shape ``[Nq, top_k]`` (or
        ``[top_k]`` if Q was 2-D).

    Notes:
        Inference only — no saved argmax, no autograd. For training-time
        retrieval, call :class:`MaxSimScorer` and ``torch.topk`` separately.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if Q.device != D.device:
        raise ValueError(
            f"Q and D must be on the same device; got Q.device={Q.device} vs D.device={D.device}."
        )

    if _HAS_TRITON and Q.is_cuda and D.is_cuda:
        from late_interaction_kernels.topk import maxsim_topk

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
    k = min(top_k, Nd)

    # Pick the per-device scoring kernel. MPS goes through ``torch.compile``
    # to fuse einsum+max+sum; CPU stays in eager (no compile dependency).
    if Q.device.type == "mps":
        from late_interaction_kernels.mps import maxsim_inference_mps as _score_kernel
    else:
        from late_interaction_kernels.reference import maxsim_reference as _score_kernel

    def _score_chunk(Q_, D_, qm_, dm_):
        with torch.no_grad():
            return _score_kernel(Q_, D_, q_mask=qm_, d_mask=dm_, normalize=normalize)

    if chunk is None or chunk >= Nd:
        scores = _score_chunk(Q, D, q_mask, d_mask)
        topk_s, topk_i = torch.topk(scores, k, dim=-1, largest=largest, sorted=sorted)
    else:
        topk_s = None
        topk_i = None
        for start in range(0, Nd, chunk):
            end = min(start + chunk, Nd)
            D_chunk = D[start:end]
            d_mask_chunk = d_mask[start:end] if d_mask is not None else None
            s_chunk = _score_chunk(Q, D_chunk, q_mask, d_mask_chunk)
            k_here = min(k, end - start)
            ch_s, ch_i = torch.topk(s_chunk, k_here, dim=-1, largest=largest, sorted=True)
            ch_i = ch_i + start
            if topk_s is None:
                topk_s, topk_i = ch_s, ch_i
            else:
                cat_s = torch.cat([topk_s, ch_s], dim=-1)
                cat_i = torch.cat([topk_i, ch_i], dim=-1)
                topk_s, gather_idx = torch.topk(cat_s, k, dim=-1, largest=largest, sorted=sorted)
                topk_i = torch.gather(cat_i, -1, gather_idx)

    if q_was_2d:
        topk_s = topk_s.squeeze(0)
        topk_i = topk_i.squeeze(0)
    return topk_s, topk_i


__all__ = ["MaxSimScorer", "retrieve"]
