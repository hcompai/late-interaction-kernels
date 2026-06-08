"""PyLate drop-in: replace ``pylate.scores.colbert_scores`` with our kernel.

For PyLate without a native LIK backend. Recent PyLate ships its own
(``pip install "pylate[lik]"``, selected via ``auto`` or
``PYLATE_SCORES_BACKEND``); there ``patch_pylate()`` is a deprecated no-op
(see ``_PYLATE_NATIVE_MIN`` for the cutoff).

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

from late_interaction_kernels._utils import package_at_least

# ``maxsim`` lives in the Triton-backed autograd module and ``maxsim_varlen``
# in the Triton varlen module; both are imported lazily so this module
# stays importable on machines without Triton (e.g. macOS), where only
# the MPS / CPU paths are reachable anyway.

# Bookkeeping for monkey-patching: ``patch_pylate()`` stashes the original
# ``pylate.scores.colbert_*`` callables here before swapping in our fused
# replacements; ``unpatch_pylate()`` reads them back to restore PyLate.
# The dict doubles as the "patched?" flag — non-empty == patched.
_ORIGINAL = {}

# First PyLate with native LIK support (lightonai/pylate#222): ColBERTScores
# forwards a ``backend=`` kwarg our drop-in doesn't accept, so we step aside.
_PYLATE_NATIVE_MIN = "1.5.1"

# One-time guard so the native-support notice isn't re-emitted on every
# ``patch_pylate()`` call (e.g. per-process training setup).
_NATIVE_NOTICE_SHOWN = False


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
        from late_interaction_kernels.autograd import maxsim

        return maxsim(Q, D, q_mask=q_mask, d_mask=d_mask)
    if path == "mps":
        from late_interaction_kernels.mps import maxsim_mps

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

    # Single fused launch over all Nq*K pairs via the 4-D dispatch in
    # ``maxsim`` (kd_layout=True), avoiding the per-query Python loop that
    # caused the regression in pylate#224 §2/§4.
    return _dispatch_maxsim(Q, D, q_mask, d_mask, path)


def patched_colbert_scores_pairwise(
    queries_embeddings,
    documents_embeddings,
    queries_mask=None,
    documents_mask=None,
    *,
    mask=None,  # legacy
):
    """Drop-in replacement for :func:`pylate.scores.colbert_scores_pairwise`.

    Diagonal pairwise scoring ``[B, Lq, d] x [B, Ld, d] -> [B]``. Routes
    through :func:`maxsim_pairs`, which is the ``K=1`` case of the KD
    layout — same fast kernel as in-batch, never materialises the
    ``[B, B]`` cross-product that vanilla ``maxsim_varlen`` would build.
    Closes the pylate#224 §5 regression for free.
    """
    from pylate.utils.tensor import convert_to_tensor  # type: ignore

    Q = convert_to_tensor(queries_embeddings)
    D = convert_to_tensor(documents_embeddings)
    q_mask, d_mask = _resolve_masks(queries_mask, documents_mask, mask)

    path = _device_path(Q, D)
    # CUDA gets the fused K=1 KD launch; everything else defers to PyLate's
    # original pairwise scorer (MPS doesn't have a 4-D fused path yet).
    if path != "cuda":
        return _ORIGINAL["colbert_scores_pairwise"](
            Q, D, queries_mask=queries_mask, documents_mask=documents_mask
        )

    from late_interaction_kernels.autograd import maxsim_pairs

    return maxsim_pairs(Q, D, q_mask=q_mask, d_mask=d_mask)


def _scores_defining_module():
    """The module where ``colbert_scores`` is defined — PyLate 1.5 renamed it
    (``scores.scores`` → ``scores.colbert``) and rerouted the contrastive losses
    through ``ColBERTScores``, which resolves the symbol from these module
    globals at call time — so on the new layout, patching here covers the loss
    path and only ``Distillation`` keeps an import-time capture."""
    try:
        import pylate.scores.scores as defining  # type: ignore  # PyLate <= 1.4

        return defining, False
    except ImportError:
        import pylate.scores.colbert as defining  # type: ignore  # PyLate >= 1.5

        return defining, True


def _loss_capture_targets(new_layout: bool) -> tuple[tuple[str, str], ...]:
    """Loss modules that captured a scoring symbol at import time, per layout."""
    if new_layout:
        return (("pylate.losses.distillation", "colbert_kd_scores"),)
    return (
        ("pylate.losses.contrastive", "colbert_scores"),
        ("pylate.losses.cached_contrastive", "colbert_scores"),
        ("pylate.losses.distillation", "colbert_kd_scores"),
    )


def patch_pylate():
    """Install the fused kernel as the default MaxSim across ``pylate.scores`` and PyLate's loss modules.

    Deprecated no-op on PyLate versions that ship native LIK support (see
    ``_PYLATE_NATIVE_MIN``): LIK is selected automatically when installed, so
    there is nothing to patch.
    """
    import warnings

    if package_at_least("pylate", _PYLATE_NATIVE_MIN):
        global _NATIVE_NOTICE_SHOWN
        if not _NATIVE_NOTICE_SHOWN:
            _NATIVE_NOTICE_SHOWN = True
            warnings.warn(
                f"late-interaction-kernels: PyLate >= {_PYLATE_NATIVE_MIN} ships native LIK "
                "support, so `patch_pylate()` is now a no-op. LIK is used automatically when "
                'installed (`pip install "pylate[lik]"`); force it over flash/torch with '
                '`PYLATE_SCORES_BACKEND=lik` (or `backend="lik"`).',
                DeprecationWarning,
                stacklevel=2,
            )
        return  # leave PyLate's own dispatch alone; `_ORIGINAL` stays empty so unpatch is a no-op too

    import pylate.scores as api  # type: ignore

    s, new_layout = _scores_defining_module()

    if "colbert_scores" in _ORIGINAL:
        return  # already patched

    _ORIGINAL["colbert_scores"] = s.colbert_scores
    _ORIGINAL["colbert_kd_scores"] = s.colbert_kd_scores
    # ``colbert_scores_pairwise`` only exists in recent PyLate; fall back
    # cleanly when the attribute is missing instead of breaking the patch.
    has_pairwise = hasattr(s, "colbert_scores_pairwise")
    if has_pairwise:
        _ORIGINAL["colbert_scores_pairwise"] = s.colbert_scores_pairwise

    s.colbert_scores = patched_colbert_scores
    s.colbert_kd_scores = patched_colbert_kd_scores
    api.colbert_scores = patched_colbert_scores
    api.colbert_kd_scores = patched_colbert_kd_scores
    if has_pairwise:
        s.colbert_scores_pairwise = patched_colbert_scores_pairwise
        api.colbert_scores_pairwise = patched_colbert_scores_pairwise

    # Some PyLate loss modules capture a scoring symbol at import time; patch
    # those references too. Warn if any are unreachable so users notice
    # silent fallbacks after a future PyLate refactor.
    import importlib

    missed: list[str] = []
    for mod_name, attr in _loss_capture_targets(new_layout):
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

    s, new_layout = _scores_defining_module()

    s.colbert_scores = _ORIGINAL["colbert_scores"]
    s.colbert_kd_scores = _ORIGINAL["colbert_kd_scores"]
    api.colbert_scores = _ORIGINAL["colbert_scores"]
    api.colbert_kd_scores = _ORIGINAL["colbert_kd_scores"]
    if "colbert_scores_pairwise" in _ORIGINAL:
        s.colbert_scores_pairwise = _ORIGINAL["colbert_scores_pairwise"]
        api.colbert_scores_pairwise = _ORIGINAL["colbert_scores_pairwise"]

    for mod_name, attr in _loss_capture_targets(new_layout):
        try:
            import importlib

            mod = importlib.import_module(mod_name)
            setattr(mod, attr, _ORIGINAL[attr])
        except Exception:
            pass

    _ORIGINAL.clear()
