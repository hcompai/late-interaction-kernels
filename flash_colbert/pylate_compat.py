"""PyLate drop-in: monkey-patch `pylate.scores.colbert_scores` with our kernel.

Usage
-----
    from flash_colbert.pylate_compat import patch_pylate
    patch_pylate()             # after this, PyLate training & rerank use flash-colbert

    # to revert:
    from flash_colbert.pylate_compat import unpatch_pylate
    unpatch_pylate()

The replacement honors PyLate's exact signature:

    colbert_scores(
        queries_embeddings,        # [Nq, Lq, d]
        documents_embeddings,      # [Nd, Ld, d]
        queries_mask=None,         # [Nq, Lq] float/bool
        documents_mask=None,       # [Nd, Ld] float/bool
    ) -> [Nq, Nd]

We fall back to the original PyTorch implementation when:
  * tensors are on CPU,
  * CUDA device isn't Ampere-or-newer,
  * `d` (embedding dim) is ridiculously small (< 8) — naive is fine,
  * the environment variable `FLASH_COLBERT_DISABLE=1` is set.
"""

from __future__ import annotations

import os

import torch

from .autograd import maxsim
from .varlen import maxsim_varlen  # noqa: F401 (re-export convenience)

_ORIGINAL = {}


def _bool_mask(m):
    if m is None:
        return None
    return m.bool() if m.dtype != torch.bool else m


def _should_fallback(q: torch.Tensor, d: torch.Tensor) -> bool:
    if os.environ.get("FLASH_COLBERT_DISABLE", "0") == "1":
        return True
    if not q.is_cuda or not d.is_cuda:
        return True
    if q.device != d.device:
        return True
    try:
        cap = torch.cuda.get_device_capability(q.device)
    except Exception:
        return True
    if cap[0] < 8:  # need Ampere or newer for bf16 + modern tensor cores
        return True
    if q.shape[-1] < 8:
        return True
    return False


def _mask_as_bool(m):
    """PyLate masks can be float (0/1), bool, or None."""
    if m is None:
        return None
    if m.dtype == torch.bool:
        return m
    return m != 0


def flash_colbert_scores(queries_embeddings, documents_embeddings, mask=None):
    """Drop-in replacement for `pylate.scores.colbert_scores` (pylate 1.2).

    PyLate signature: `colbert_scores(Q, D, mask=None)` where `mask` is the
    DOCUMENT mask (shape [Nd, Ld]) that gets multiplied into the similarity
    tensor. We translate it to our `d_mask` argument.
    """
    from pylate.utils.tensor import convert_to_tensor  # type: ignore

    Q = convert_to_tensor(queries_embeddings)
    D = convert_to_tensor(documents_embeddings)

    if _should_fallback(Q, D):
        return _ORIGINAL["colbert_scores"](Q, D, mask)

    return maxsim(Q, D, d_mask=_mask_as_bool(mask))


def flash_colbert_kd_scores(queries_embeddings, documents_embeddings, mask=None):
    """Drop-in replacement for `pylate.scores.colbert_kd_scores`.

    PyLate shape convention: `documents_embeddings` is `[Nq, Nd, Ld, d]` —
    each query has its own candidate list — and `mask` is `[Nq, Nd, Ld]`.
    """
    from pylate.utils.tensor import convert_to_tensor  # type: ignore

    Q = convert_to_tensor(queries_embeddings)
    D = convert_to_tensor(documents_embeddings)

    if _should_fallback(Q, D):
        return _ORIGINAL["colbert_kd_scores"](Q, D, mask)

    if D.dim() != 4:
        raise ValueError(f"colbert_kd_scores expects D.dim()==4, got {D.dim()}")

    d_mask = _mask_as_bool(mask)
    Nq, _Lq, _d = Q.shape
    _, Nd, _Ld, _ = D.shape
    out = torch.empty(Nq, Nd, device=Q.device, dtype=torch.float32)
    # Each query has its own doc set, so we can't batch across Nq naively.
    # Per-query dispatch is OK here — KD typically uses small Nd (<= 16 negatives).
    for i in range(Nq):
        out[i] = maxsim(
            Q[i].unsqueeze(0),
            D[i],
            d_mask=d_mask[i] if d_mask is not None else None,
        ).squeeze(0)
    return out


def patch_pylate():
    """Install flash-colbert as the default MaxSim inside `pylate.scores`."""
    import pylate.scores as api  # type: ignore
    import pylate.scores.scores as s  # type: ignore

    if "colbert_scores" in _ORIGINAL:
        return  # already patched

    _ORIGINAL["colbert_scores"] = s.colbert_scores
    _ORIGINAL["colbert_kd_scores"] = s.colbert_kd_scores

    s.colbert_scores = flash_colbert_scores
    s.colbert_kd_scores = flash_colbert_kd_scores
    api.colbert_scores = flash_colbert_scores
    api.colbert_kd_scores = flash_colbert_kd_scores

    # Losses hold a direct reference captured at import time — patch those too.
    try:
        import pylate.losses.contrastive as c  # type: ignore

        c.colbert_scores = flash_colbert_scores
    except Exception:
        pass
    try:
        import pylate.losses.cached_contrastive as cc  # type: ignore

        cc.colbert_scores = flash_colbert_scores
    except Exception:
        pass


def unpatch_pylate():
    """Restore PyLate's original MaxSim implementation."""
    if not _ORIGINAL:
        return

    import pylate.scores as api  # type: ignore
    import pylate.scores.scores as s  # type: ignore

    s.colbert_scores = _ORIGINAL["colbert_scores"]
    s.colbert_kd_scores = _ORIGINAL["colbert_kd_scores"]
    api.colbert_scores = _ORIGINAL["colbert_scores"]
    api.colbert_kd_scores = _ORIGINAL["colbert_kd_scores"]

    try:
        import pylate.losses.contrastive as c  # type: ignore

        c.colbert_scores = _ORIGINAL["colbert_scores"]
    except Exception:
        pass
    try:
        import pylate.losses.cached_contrastive as cc  # type: ignore

        cc.colbert_scores = _ORIGINAL["colbert_scores"]
    except Exception:
        pass

    _ORIGINAL.clear()
