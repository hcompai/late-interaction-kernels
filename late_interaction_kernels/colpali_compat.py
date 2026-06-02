"""colpali_engine drop-in: route MaxSim through the fused kernel.

::

    from late_interaction_kernels import patch_colpali_engine, unpatch_colpali_engine
    patch_colpali_engine()        # colpali_engine training + scoring uses our kernel
    # ... train / score ...
    unpatch_colpali_engine()      # restore the originals

Patches the entry points where colpali_engine materializes the
``[B, C, Lq, Ld]`` similarity tensor with an unfused ``torch.einsum``:

* :meth:`colpali_engine.utils.processing_utils.BaseVisualRetrieverProcessor.score_multi_vector`
  — the inference / evaluation scoring helper used by ``vidore`` and ad-hoc
  scoring code.
* :meth:`colpali_engine.loss.late_interaction_losses.ColbertLoss.forward`
* :meth:`...ColbertPairwiseCELoss.forward`
* :meth:`...ColbertSigmoidLoss.forward`
  — the three in-batch loss heads.
* :meth:`...ColbertNegativeCELoss.forward`
* :meth:`...ColbertPairwiseNegativeCELoss.forward`
  — the explicit-hard-negative heads: positives route through
  :func:`maxsim_pairs`, per-query negatives through 4-D :func:`maxsim` (KD
  layout), and the in-batch term reuses the already-patched ``inner_loss`` /
  ``inner_pairwise``. Pos/neg fusion is CUDA-only; elsewhere those terms fall
  back to the einsum while the in-batch term still accelerates.

Dispatch rules match :func:`patch_pylate`: CUDA (Ampere+) → fused Triton
kernel; MPS → ``torch.compile``-fused reference; CPU / sub-Ampere /
``LIK_DISABLE=1`` / ``use_smooth_max=True`` / shape edge cases fall
through to colpali_engine's original implementation.
"""

import os

import torch

# Bookkeeping: ``patch_colpali_engine()`` stashes the original callables
# here before swapping in ours; ``unpatch_colpali_engine()`` reads them
# back. Non-empty == patched.
_ORIGINAL = {}


def _device_path(q: torch.Tensor, d: torch.Tensor) -> str | None:
    """Pick the dispatch path or ``None`` when we should fall back."""
    if os.environ.get("LIK_DISABLE", "0") == "1":
        return None
    if q.device != d.device:
        return None
    if q.shape[-1] < 8:
        return None
    if q.is_cuda and d.is_cuda:
        try:
            cap = torch.cuda.get_device_capability(q.device)
        except Exception:
            return None
        if cap[0] < 8:  # need Ampere or newer for bf16 + modern tensor cores
            return None
        return "cuda"
    if q.device.type == "mps" and d.device.type == "mps":
        return "mps"
    return None


def _dispatch_maxsim(Q: torch.Tensor, D: torch.Tensor, path: str) -> torch.Tensor:
    """Compute MaxSim through the fused kernel for ``path``.

    colpali_engine doesn't pass masks into the in-batch einsum (padding is
    handled by the embeddings themselves — pad tokens are zero), so we
    don't either.
    """
    if path == "cuda":
        from late_interaction_kernels.autograd import maxsim

        return maxsim(Q, D)
    if path == "mps":
        from late_interaction_kernels.mps import maxsim_mps

        return maxsim_mps(Q, D, normalize=False)
    raise ValueError(f"unknown dispatch path {path!r}")


def patched_score_multi_vector(qs, ps, batch_size: int = 128, device=None):
    """Drop-in replacement for ``BaseVisualRetrieverProcessor.score_multi_vector``.

    Mirrors the original's chunking + ``pad_sequence`` logic, but the inner
    ``einsum("bnd,csd->bcns").max(dim=3)[0].sum(dim=2)`` collapses into one
    fused-kernel call per (qs_chunk, ps_chunk) tile.
    """
    from colpali_engine.utils.torch_utils import get_torch_device  # type: ignore

    device = device or get_torch_device("auto")

    if len(qs) == 0:
        raise ValueError("No queries provided")
    if len(ps) == 0:
        raise ValueError("No passages provided")

    scores_list: list[torch.Tensor] = []

    for i in range(0, len(qs), batch_size):
        scores_batch = []
        qs_batch = torch.nn.utils.rnn.pad_sequence(
            qs[i : i + batch_size], batch_first=True, padding_value=0
        ).to(device)
        for j in range(0, len(ps), batch_size):
            ps_batch = torch.nn.utils.rnn.pad_sequence(
                ps[j : j + batch_size], batch_first=True, padding_value=0
            ).to(device)
            path = _device_path(qs_batch, ps_batch)
            if path is None:
                tile = torch.einsum("bnd,csd->bcns", qs_batch, ps_batch).max(dim=3)[0].sum(dim=2)
            else:
                tile = _dispatch_maxsim(qs_batch, ps_batch, path)
            scores_batch.append(tile)
        scores_list.append(torch.cat(scores_batch, dim=1).cpu())

    scores = torch.cat(scores_list, dim=0).to(torch.float32)
    assert scores.shape[0] == len(qs), f"Expected {len(qs)} scores, got {scores.shape[0]}"
    return scores


def _patched_inbatch_scores(self, Q: torch.Tensor, D: torch.Tensor) -> torch.Tensor | None:
    """Replacement for the ``einsum + amax + sum + normalize`` sequence shared
    by every ``ColbertModule`` in-batch forward.

    Returns ``None`` when we can't accelerate (``use_smooth_max``,
    ``LIK_DISABLE``, mixed devices, …); the caller must then fall through to
    the original ``forward``.
    """
    if self.use_smooth_max:
        return None
    path = _device_path(Q, D)
    if path is None:
        return None
    lengths = (Q[:, :, 0] != 0).sum(dim=1)
    scores = _dispatch_maxsim(Q, D, path)
    if self.normalize_scores:
        scores = self._apply_normalization(scores, lengths)
    return scores


def patched_colbert_loss_forward(self, query_embeddings, doc_embeddings, offset: int = 0):
    """Drop-in for :meth:`ColbertLoss.forward`."""
    scores = _patched_inbatch_scores(self, query_embeddings, doc_embeddings)
    if scores is None:
        return _ORIGINAL["ColbertLoss.forward"](self, query_embeddings, doc_embeddings, offset)
    batch_size = scores.size(0)
    _idx, pos_idx = self._get_idx(batch_size, offset, scores.device)
    if self.pos_aware_negative_filtering:
        self._filter_high_negatives(scores, pos_idx)
    return self.ce_loss(scores / self.temperature, pos_idx)


def patched_colbert_pairwise_ce_forward(self, query_embeddings, doc_embeddings, offset: int = 0):
    """Drop-in for :meth:`ColbertPairwiseCELoss.forward`."""
    import torch.nn.functional as F  # local — keeps the module light

    scores = _patched_inbatch_scores(self, query_embeddings, doc_embeddings)
    if scores is None:
        return _ORIGINAL["ColbertPairwiseCELoss.forward"](self, query_embeddings, doc_embeddings, offset)
    batch_size = scores.size(0)
    _idx, pos_idx = self._get_idx(batch_size, offset, scores.device)
    if self.pos_aware_negative_filtering:
        self._filter_high_negatives(scores, pos_idx)
    pos_scores = scores.diagonal(offset=offset)
    top2 = scores.topk(2, dim=1).values
    neg_scores = torch.where(top2[:, 0] == pos_scores, top2[:, 1], top2[:, 0])
    return F.softplus((neg_scores - pos_scores) / self.temperature).mean()


def patched_colbert_sigmoid_forward(self, query_embeddings, doc_embeddings, offset: int = 0):
    """Drop-in for :meth:`ColbertSigmoidLoss.forward`."""
    import torch.nn.functional as F

    scores = _patched_inbatch_scores(self, query_embeddings, doc_embeddings)
    if scores is None:
        return _ORIGINAL["ColbertSigmoidLoss.forward"](self, query_embeddings, doc_embeddings, offset)
    batch_size = scores.size(0)
    _idx, pos_idx = self._get_idx(batch_size, offset, scores.device)
    if self.pos_aware_negative_filtering:
        self._filter_high_negatives(scores, pos_idx)

    flat_pos = pos_idx * (batch_size + 1)
    pos_mask = -torch.ones(batch_size * batch_size, device=scores.device)
    pos_mask[flat_pos] = 1.0
    scores = scores.view(-1) / self.temperature
    return F.softplus(-scores * pos_mask).mean()


def _patched_negative_scores(self, Q, pos_D, neg_D, offset: int):
    """Pos (``maxsim_pairs``) / neg (4-D ``maxsim``, KD layout) MaxSim for the
    explicit-hard-negative heads, replacing the original ``einsum -> amax ->
    sum``.

    Returns ``(pos_scores[B], neg_scores[B, n_neg])``, or ``None`` to tell the
    caller to fall through to the original forward (``use_smooth_max``, non-CUDA
    device, mixed devices, narrow ``d`` — these kernels are the Triton path).
    """
    if self.use_smooth_max:
        return None
    if _device_path(Q, pos_D) != "cuda":
        return None
    if neg_D.device != Q.device or neg_D.shape[-1] < 8:
        return None

    from late_interaction_kernels.autograd import maxsim, maxsim_pairs

    lengths = (Q[:, :, 0] != 0).sum(dim=1)
    pos_D = pos_D[offset : offset + Q.size(0)]
    pos_scores = maxsim_pairs(Q, pos_D)  # [B]
    neg_scores = maxsim(Q, neg_D)  # [B, n_neg] via the 4-D KD dispatch
    if self.normalize_scores:
        pos_scores = self._apply_normalization(pos_scores, lengths)
        neg_scores = self._apply_normalization(neg_scores, lengths)
    return pos_scores, neg_scores


def _negative_ce_loss(self, Q, pos_D, neg_D, offset, original_key, inner_attr):
    """Shared body for the two explicit-negative heads.

    Identical softplus(neg - pos) objective; they differ only in which inner
    head supplies the in-batch term (``inner_loss`` vs ``inner_pairwise``) and
    which original to restore on fall-through.
    """
    import torch.nn.functional as F

    scored = _patched_negative_scores(self, Q, pos_D, neg_D, offset)
    if scored is None:
        return _ORIGINAL[original_key](self, Q, pos_D, neg_D, offset)
    pos_scores, neg_scores = scored
    loss = F.softplus((neg_scores - pos_scores.unsqueeze(1)) / self.temperature).mean()
    if self.in_batch_term_weight > 0:
        # The inner head's forward is patched at class level, so its in-batch
        # term is already fused — no double work, no extra dispatch here.
        loss_ib = getattr(self, inner_attr)(Q, pos_D, offset)
        loss = loss * (1 - self.in_batch_term_weight) + loss_ib * self.in_batch_term_weight
    return loss


def patched_colbert_negative_ce_forward(
    self, query_embeddings, doc_embeddings, neg_doc_embeddings, offset: int = 0
):
    """Drop-in for :meth:`ColbertNegativeCELoss.forward`."""
    return _negative_ce_loss(
        self,
        query_embeddings,
        doc_embeddings,
        neg_doc_embeddings,
        offset,
        "ColbertNegativeCELoss.forward",
        "inner_loss",
    )


def patched_colbert_pairwise_negative_ce_forward(
    self, query_embeddings, doc_embeddings, neg_doc_embeddings, offset: int = 0
):
    """Drop-in for :meth:`ColbertPairwiseNegativeCELoss.forward`."""
    return _negative_ce_loss(
        self,
        query_embeddings,
        doc_embeddings,
        neg_doc_embeddings,
        offset,
        "ColbertPairwiseNegativeCELoss.forward",
        "inner_pairwise",
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def patch_colpali_engine():
    """Install the fused kernel across colpali_engine's MaxSim entry points."""
    import warnings

    if _ORIGINAL:
        return  # already patched

    missed: list[str] = []

    # Inference / evaluation scoring helper.
    try:
        from colpali_engine.utils.processing_utils import (  # type: ignore
            BaseVisualRetrieverProcessor,
        )

        _ORIGINAL["BaseVisualRetrieverProcessor.score_multi_vector"] = (
            BaseVisualRetrieverProcessor.score_multi_vector
        )
        BaseVisualRetrieverProcessor.score_multi_vector = staticmethod(patched_score_multi_vector)
    except ImportError:
        missed.append("colpali_engine.utils.processing_utils")

    # Loss heads. Each one shadows its own forward() so subclasses inherit.
    losses_module = "colpali_engine.loss.late_interaction_losses"
    try:
        import importlib

        losses = importlib.import_module(losses_module)
    except ImportError:
        losses = None
        missed.append(losses_module)

    if losses is not None:
        for cls_name, replacement in (
            ("ColbertLoss", patched_colbert_loss_forward),
            ("ColbertPairwiseCELoss", patched_colbert_pairwise_ce_forward),
            ("ColbertSigmoidLoss", patched_colbert_sigmoid_forward),
            ("ColbertNegativeCELoss", patched_colbert_negative_ce_forward),
            ("ColbertPairwiseNegativeCELoss", patched_colbert_pairwise_negative_ce_forward),
        ):
            cls = getattr(losses, cls_name, None)
            if cls is None:
                missed.append(f"{losses_module}.{cls_name}")
                continue
            _ORIGINAL[f"{cls_name}.forward"] = cls.forward
            cls.forward = replacement

    if not _ORIGINAL:
        raise ImportError(
            "late-interaction-kernels: `patch_colpali_engine()` couldn't reach any "
            "expected colpali_engine entry point. Install with "
            "`pip install colpali-engine` and check the version is >=0.3.10."
        )

    if missed:
        warnings.warn(
            "late-interaction-kernels: `patch_colpali_engine()` could not reach "
            + ", ".join(missed)
            + ". The reachable entry points are still hooked; any unreachable one "
            "keeps using vanilla colpali_engine. This usually means colpali_engine "
            "was refactored — check the installed version.",
            RuntimeWarning,
            stacklevel=2,
        )


def unpatch_colpali_engine():
    """Restore colpali_engine's original MaxSim implementations."""
    if not _ORIGINAL:
        return

    key = "BaseVisualRetrieverProcessor.score_multi_vector"
    if key in _ORIGINAL:
        from colpali_engine.utils.processing_utils import (  # type: ignore
            BaseVisualRetrieverProcessor,
        )

        # Re-wrap as staticmethod: descriptor access during patching unwrapped
        # the original to a plain function; restoring without staticmethod()
        # would turn it into a bound instance method.
        BaseVisualRetrieverProcessor.score_multi_vector = staticmethod(_ORIGINAL[key])

    import importlib

    for cls_name in (
        "ColbertLoss",
        "ColbertPairwiseCELoss",
        "ColbertSigmoidLoss",
        "ColbertNegativeCELoss",
        "ColbertPairwiseNegativeCELoss",
    ):
        attr = f"{cls_name}.forward"
        if attr not in _ORIGINAL:
            continue
        losses = importlib.import_module("colpali_engine.loss.late_interaction_losses")
        cls = getattr(losses, cls_name)
        cls.forward = _ORIGINAL[attr]

    _ORIGINAL.clear()
