"""PyLate drop-in: replace ``pylate.scores.colbert_scores`` with our kernel.

::

    from late_interaction_kernels import patch_pylate, unpatch_pylate
    patch_pylate()                # PyLate trainers + rerank now use the fused kernel
    # ... train / rerank ...
    unpatch_pylate()              # restore the original

Dispatch:

* CUDA (Ampere+) — fused Triton kernel via :func:`maxsim`.
* MPS (Apple Silicon) — :func:`maxsim_mps`, the ``torch.compile``-fused
  reference. Autograd-aware, so PyLate's training backward graph keeps
  flowing.
* CPU / sub-Ampere CUDA / ``d < 8`` / ``LIK_DISABLE=1`` — fall through
  to PyLate's original implementation.
"""

import os

import torch

# ``maxsim`` lives in the Triton-backed autograd module and ``maxsim_varlen``
# in the Triton varlen module; both are imported lazily so this module
# stays importable on machines without Triton (e.g. macOS), where only
# the MPS / CPU paths are reachable anyway.

# Bookkeeping for monkey-patching: ``patch_pylate()`` stashes the original
# ``pylate.scores.colbert_*`` callables here before swapping in our fused
# replacements; ``unpatch_pylate()`` reads them back to restore PyLate.
# The dict doubles as the "patched?" flag — non-empty == patched.
_ORIGINAL = {}


def _bool_mask(m):
    if m is None:
        return None
    return m.bool() if m.dtype != torch.bool else m


def _device_path(q: torch.Tensor, d: torch.Tensor) -> str | None:
    """Pick the dispatch path or ``None`` when we should fall back.

    Returns ``"cuda"`` / ``"mps"`` for the fused paths, ``None`` to defer
    to PyLate's original implementation.
    """
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


def _dispatch_maxsim(Q, D, q_mask, d_mask, path: str) -> torch.Tensor:
    """Route to the fused implementation for ``path``.

    PyLate's ``colbert_scores`` does not L2-normalize internally — the
    encoder is expected to emit unit vectors — so we keep ``normalize=False``
    on every backend.
    """
    if path == "cuda":
        from .autograd import maxsim

        return maxsim(Q, D, q_mask=q_mask, d_mask=d_mask)
    if path == "mps":
        from .mps import maxsim_mps

        return maxsim_mps(Q, D, q_mask=q_mask, d_mask=d_mask, normalize=False)
    raise ValueError(f"unknown dispatch path {path!r}")


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
    mask=None,  # legacy single-mask alias, mapped to documents_mask
):
    """Drop-in replacement for :func:`pylate.scores.colbert_scores`."""
    from pylate.utils.tensor import convert_to_tensor  # type: ignore

    Q = convert_to_tensor(queries_embeddings)
    D = convert_to_tensor(documents_embeddings)
    q_mask, d_mask = _resolve_masks(queries_mask, documents_mask, mask)

    path = _device_path(Q, D)
    if path is None:
        return _ORIGINAL["colbert_scores"](Q, D, queries_mask=queries_mask, documents_mask=documents_mask)

    return _dispatch_maxsim(Q, D, q_mask, d_mask, path)


def patched_colbert_kd_scores(
    queries_embeddings,
    documents_embeddings,
    queries_mask=None,
    documents_mask=None,
    *,
    mask=None,  # legacy
):
    """Drop-in replacement for :func:`pylate.scores.colbert_kd_scores`.

    KD shape: ``D`` is ``[Nq, Nd, Ld, d]`` (each query has its own
    candidate list), ``documents_mask`` ``[Nq, Nd, Ld]``.
    """
    from pylate.utils.tensor import convert_to_tensor  # type: ignore

    Q = convert_to_tensor(queries_embeddings)
    D = convert_to_tensor(documents_embeddings)
    q_mask, d_mask = _resolve_masks(queries_mask, documents_mask, mask)

    path = _device_path(Q, D)
    if path is None:
        return _ORIGINAL["colbert_kd_scores"](Q, D, queries_mask=queries_mask, documents_mask=documents_mask)

    if D.dim() != 4:
        raise ValueError(f"colbert_kd_scores expects D.dim()==4, got {D.dim()}")

    Nq, _Lq, _d = Q.shape
    _, Nd, _Ld, _ = D.shape
    out = torch.empty(Nq, Nd, device=Q.device, dtype=torch.float32)
    for i in range(Nq):
        out[i] = _dispatch_maxsim(
            Q[i].unsqueeze(0),
            D[i],
            q_mask[i : i + 1] if q_mask is not None else None,
            d_mask[i] if d_mask is not None else None,
            path,
        ).squeeze(0)
    return out


def patch_pylate():
    """Install the fused kernel as the default MaxSim across ``pylate.scores`` and PyLate's loss modules."""
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

    # PyLate loss modules capture `colbert_scores` at import time; patch
    # those references too. Warn if any are unreachable so users notice
    # silent fallbacks after a future PyLate refactor.
    import importlib
    import warnings

    missed: list[str] = []
    for mod_name, attr in (
        ("pylate.losses.contrastive", "colbert_scores"),
        ("pylate.losses.cached_contrastive", "colbert_scores"),
        ("pylate.losses.distillation", "colbert_kd_scores"),
    ):
        try:
            mod = importlib.import_module(mod_name)
            if not hasattr(mod, attr):
                missed.append(f"{mod_name}.{attr}")
                continue
            setattr(
                mod, attr, patched_colbert_scores if attr == "colbert_scores" else patched_colbert_kd_scores
            )
        except ImportError:
            # Loss module doesn't exist in this PyLate version — not a bug.
            continue
        except Exception as exc:  # noqa: BLE001 — surface the failure verbatim
            missed.append(f"{mod_name}.{attr} ({type(exc).__name__}: {exc})")

    if missed:
        warnings.warn(
            "late-interaction-kernels: `patch_pylate()` could not reach "
            + ", ".join(missed)
            + ". `pylate.scores.colbert_scores` is hooked, but any loss module that "
            "captured the un-patched symbol at import time will keep using vanilla "
            "PyLate. This usually means PyLate was refactored — check you're on a "
            "supported version (`pylate>=1.3.3`).",
            RuntimeWarning,
            stacklevel=2,
        )


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
