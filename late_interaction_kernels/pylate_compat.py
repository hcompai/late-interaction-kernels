"""PyLate drop-in: monkey-patch `pylate.scores.colbert_scores` with our kernel.

Usage
-----
    from late_interaction_kernels.pylate_compat import patch_pylate
    patch_pylate()             # after this, PyLate training & rerank use late-interaction-kernels

    # to revert:
    from late_interaction_kernels.pylate_compat import unpatch_pylate
    unpatch_pylate()

The replacement honors PyLate's exact signature:

    colbert_scores(
        queries_embeddings,        # [Nq, Lq, d]
        documents_embeddings,      # [Nd, Ld, d]
        queries_mask=None,         # [Nq, Lq] float/bool
        documents_mask=None,       # [Nd, Ld] float/bool
    ) -> [Nq, Nd]

As of PyLate 1.2.x the loss modules (``Contrastive``, ``CachedContrastive``,
``Distillation``) call into ``colbert_scores`` with keyword arguments
``queries_mask=`` and ``documents_mask=``. For backward compatibility we
also accept the legacy single ``mask=`` kwarg (older PyLate versions had
that signature) and a third positional / keyword alias that some forks
use. In every case the document mask is the primary mask applied to the
``max`` reduction; the query mask, when present, is applied to query
token rows before the scatter.

We fall back to the original PyTorch implementation when:
  * tensors are on CPU,
  * CUDA device isn't Ampere-or-newer,
  * `d` (embedding dim) is ridiculously small (< 8) — naive is fine,
  * the environment variable `LIK_DISABLE=1` is set.
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
    if os.environ.get("LIK_DISABLE", "0") == "1":
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


def _resolve_masks(queries_mask, documents_mask, legacy_mask):
    """PyLate's signature has evolved. Accept all known shapes:

    * current (>= 1.1):  ``queries_mask=``, ``documents_mask=``
    * legacy (< 1.1):    ``mask=`` (document mask only)
    * positional legacy: 3rd positional == document mask

    Whichever arrived, hand back ``(q_mask, d_mask)`` as bool tensors.
    """
    if documents_mask is None and legacy_mask is not None:
        documents_mask = legacy_mask
    return _mask_as_bool(queries_mask), _mask_as_bool(documents_mask)


def patched_colbert_scores(
    queries_embeddings,
    documents_embeddings,
    queries_mask=None,
    documents_mask=None,
    *,
    mask=None,  # legacy PyLate kwarg — accepted and mapped to documents_mask
):
    """Drop-in replacement for :func:`pylate.scores.colbert_scores`.

    PyLate current signature (>= 1.1)::

        colbert_scores(Q, D, queries_mask=None, documents_mask=None)

    ``Contrastive`` / ``CachedContrastive`` call this with **keyword** args
    ``queries_mask=`` and ``documents_mask=``. We also accept the legacy
    single ``mask=`` kwarg to stay compatible with older PyLate releases.

    PyLate implements masks by multiplication (``scores * mask``); we
    preserve the same algebraic result by masking-out tokens (``-inf``) so
    they can't win the max, which matches PyLate whenever real scores are
    non-negative (the common case after L2-normalization).
    """
    from pylate.utils.tensor import convert_to_tensor  # type: ignore

    Q = convert_to_tensor(queries_embeddings)
    D = convert_to_tensor(documents_embeddings)
    q_mask, d_mask = _resolve_masks(queries_mask, documents_mask, mask)

    if _should_fallback(Q, D):
        return _ORIGINAL["colbert_scores"](Q, D, queries_mask=queries_mask, documents_mask=documents_mask)

    return maxsim(Q, D, q_mask=q_mask, d_mask=d_mask)


def patched_colbert_kd_scores(
    queries_embeddings,
    documents_embeddings,
    queries_mask=None,
    documents_mask=None,
    *,
    mask=None,  # legacy
):
    """Drop-in replacement for :func:`pylate.scores.colbert_kd_scores`.

    PyLate current signature (>= 1.1)::

        colbert_kd_scores(Q, D, queries_mask=None, documents_mask=None)

    Shape convention: ``D`` is ``[Nq, Nd, Ld, d]`` — each query has its
    own candidate list — and ``documents_mask`` is ``[Nq, Nd, Ld]``.
    ``queries_mask``, when supplied, is ``[Nq, Lq]``.
    """
    from pylate.utils.tensor import convert_to_tensor  # type: ignore

    Q = convert_to_tensor(queries_embeddings)
    D = convert_to_tensor(documents_embeddings)
    q_mask, d_mask = _resolve_masks(queries_mask, documents_mask, mask)

    if _should_fallback(Q, D):
        return _ORIGINAL["colbert_kd_scores"](Q, D, queries_mask=queries_mask, documents_mask=documents_mask)

    if D.dim() != 4:
        raise ValueError(f"colbert_kd_scores expects D.dim()==4, got {D.dim()}")

    Nq, _Lq, _d = Q.shape
    _, Nd, _Ld, _ = D.shape
    out = torch.empty(Nq, Nd, device=Q.device, dtype=torch.float32)
    for i in range(Nq):
        out[i] = maxsim(
            Q[i].unsqueeze(0),
            D[i],
            q_mask=q_mask[i : i + 1] if q_mask is not None else None,
            d_mask=d_mask[i] if d_mask is not None else None,
        ).squeeze(0)
    return out


def patch_pylate():
    """Install late-interaction-kernels as the default MaxSim inside `pylate.scores`."""
    import pylate.scores as api  # type: ignore
    import pylate.scores.scores as s  # type: ignore

    if "colbert_scores" in _ORIGINAL:
        return  # already patched

    _ORIGINAL["colbert_scores"] = s.colbert_scores
    _ORIGINAL["colbert_kd_scores"] = s.colbert_kd_scores

    s.colbert_scores = patched_colbert_scores
    s.colbert_kd_scores = patched_colbert_kd_scores
    api.colbert_scores = patched_colbert_scores
    api.colbert_kd_scores = patched_colbert_kd_scores

    # Losses hold a direct reference captured at import time — patch those too.
    for mod_name, attr in (
        ("pylate.losses.contrastive", "colbert_scores"),
        ("pylate.losses.cached_contrastive", "colbert_scores"),
        ("pylate.losses.distillation", "colbert_kd_scores"),
    ):
        try:
            import importlib

            mod = importlib.import_module(mod_name)
            setattr(
                mod, attr, patched_colbert_scores if attr == "colbert_scores" else patched_colbert_kd_scores
            )
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

    for mod_name, attr, orig_key in (
        ("pylate.losses.contrastive", "colbert_scores", "colbert_scores"),
        ("pylate.losses.cached_contrastive", "colbert_scores", "colbert_scores"),
        ("pylate.losses.distillation", "colbert_kd_scores", "colbert_kd_scores"),
    ):
        try:
            import importlib

            mod = importlib.import_module(mod_name)
            setattr(mod, attr, _ORIGINAL[orig_key])
        except Exception:
            pass

    _ORIGINAL.clear()
