"""CPU-safe unit tests for `patch_pylate()`'s loss-module warning path.

The end-to-end PyLate test in ``test_pylate_compat.py`` needs CUDA and a
live PyLate install. This module tests just the **warning contract**
added in 0.9.0: when PyLate ships a loss submodule that doesn't expose
``colbert_scores`` / ``colbert_kd_scores`` (e.g. a future rename), the
user must see a ``RuntimeWarning`` rather than silently running an
unpatched loss.

We simulate that by injecting a minimal fake PyLate into ``sys.modules``
before importing ``pylate_compat``. No GPU, no real PyLate required.
"""

from __future__ import annotations

import importlib
import sys
import types
import warnings

import pytest

# ``pylate_compat`` pulls in ``autograd`` which transitively imports Triton.
# Triton ships with torch on Linux (including CPU-only CI) but not on macOS.
pytest.importorskip("triton")


def _install_fake_pylate(*, include_attr: bool) -> list[str]:
    """Build a minimal fake ``pylate`` hierarchy.

    If ``include_attr=False``, the loss submodules are missing the
    expected ``colbert_scores`` attribute — the exact condition that
    must trigger the RuntimeWarning in ``patch_pylate()``.

    Returns the list of module names we inserted, so the caller can
    tear them down.
    """

    def _noop(*_a, **_kw):  # stand-in for pylate's own colbert_scores
        raise RuntimeError("fake — should be overwritten by patch_pylate()")

    installed: list[str] = []

    def _mk(name: str, **attrs) -> types.ModuleType:
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        installed.append(name)
        return m

    pylate = _mk("pylate")
    scores_pkg = _mk("pylate.scores", colbert_scores=_noop, colbert_kd_scores=_noop)
    pylate.scores = scores_pkg  # type: ignore[attr-defined]
    _mk("pylate.scores.scores", colbert_scores=_noop, colbert_kd_scores=_noop)

    loss_attrs = {"colbert_scores": _noop} if include_attr else {}
    kd_attrs = {"colbert_kd_scores": _noop} if include_attr else {}
    losses_pkg = _mk("pylate.losses")
    pylate.losses = losses_pkg  # type: ignore[attr-defined]
    _mk("pylate.losses.contrastive", **loss_attrs)
    _mk("pylate.losses.cached_contrastive", **loss_attrs)
    _mk("pylate.losses.distillation", **kd_attrs)

    return installed


def _uninstall(modules: list[str]) -> None:
    for m in modules:
        sys.modules.pop(m, None)
    # Force `pylate_compat` to re-read whatever pylate is on sys.modules
    # the next time it's imported by another test.
    sys.modules.pop("late_interaction_kernels.pylate_compat", None)


def test_patch_pylate_warns_when_loss_submodule_symbol_missing():
    """Missing ``colbert_scores`` attr on a loss submodule ⇒ RuntimeWarning.

    This guards the 0.9.0 addition: previously the patch silently left
    the loss module un-patched. Now the user sees a warning that lists
    every symbol the patch couldn't reach.
    """
    installed = _install_fake_pylate(include_attr=False)
    try:
        # Reload so `patch_pylate()` sees our fakes (not any cached pylate).
        sys.modules.pop("late_interaction_kernels.pylate_compat", None)
        pc = importlib.import_module("late_interaction_kernels.pylate_compat")

        # Keep the test hermetic — reset the internal "already patched" flag.
        pc._ORIGINAL.clear()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pc.patch_pylate()

        runtime = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert len(runtime) == 1, [str(x.message) for x in w]
        msg = str(runtime[0].message)
        assert "pylate.losses.contrastive.colbert_scores" in msg
        assert "pylate.losses.cached_contrastive.colbert_scores" in msg
        assert "pylate.losses.distillation.colbert_kd_scores" in msg
        # Must point the user at the likely cause.
        assert "refactor" in msg.lower()
        pc.unpatch_pylate()
    finally:
        _uninstall(installed)


def test_patch_pylate_silent_when_all_loss_symbols_present():
    """Happy path: all loss submodules expose the expected symbol ⇒ no warning."""
    installed = _install_fake_pylate(include_attr=True)
    try:
        sys.modules.pop("late_interaction_kernels.pylate_compat", None)
        pc = importlib.import_module("late_interaction_kernels.pylate_compat")
        pc._ORIGINAL.clear()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pc.patch_pylate()

        lik_runtime = [
            x
            for x in w
            if issubclass(x.category, RuntimeWarning) and "late-interaction-kernels" in str(x.message)
        ]
        assert lik_runtime == [], [str(x.message) for x in lik_runtime]
        pc.unpatch_pylate()
    finally:
        _uninstall(installed)


def test_patch_pylate_is_idempotent():
    """Calling ``patch_pylate()`` twice must be a no-op on the second call.

    Previously the second call could double-patch (the installed hook
    would then delegate to itself). The early-return on ``_ORIGINAL``
    guards that; this test pins the contract.
    """
    installed = _install_fake_pylate(include_attr=True)
    try:
        sys.modules.pop("late_interaction_kernels.pylate_compat", None)
        pc = importlib.import_module("late_interaction_kernels.pylate_compat")
        pc._ORIGINAL.clear()

        pc.patch_pylate()
        hook = sys.modules["pylate.scores.scores"].colbert_scores
        pc.patch_pylate()  # second call: must not re-wrap the already-patched hook
        assert sys.modules["pylate.scores.scores"].colbert_scores is hook
        pc.unpatch_pylate()
    finally:
        _uninstall(installed)


def test_patch_pylate_unpatch_restores_originals():
    """``unpatch_pylate()`` must leave the module state byte-identical to pre-patch."""
    installed = _install_fake_pylate(include_attr=True)
    try:
        sys.modules.pop("late_interaction_kernels.pylate_compat", None)
        pc = importlib.import_module("late_interaction_kernels.pylate_compat")
        pc._ORIGINAL.clear()

        scores_mod = sys.modules["pylate.scores.scores"]
        api_mod = sys.modules["pylate.scores"]
        before_scores = scores_mod.colbert_scores
        before_kd = scores_mod.colbert_kd_scores

        pc.patch_pylate()
        assert scores_mod.colbert_scores is not before_scores  # patched
        pc.unpatch_pylate()
        assert scores_mod.colbert_scores is before_scores
        assert scores_mod.colbert_kd_scores is before_kd
        assert api_mod.colbert_scores is before_scores
    finally:
        _uninstall(installed)


if __name__ == "__main__":  # pragma: no cover — run directly for quick checks
    pytest.main([__file__, "-v"])
