"""``autotune_kwargs()`` is the single source of truth for shared
``triton.autotune`` keyword arguments. These tests run on CPU — they just
check the feature-detect logic for Triton's persistent on-disk autotune
cache (``cache_results``, added in Triton 3.4).
"""

from __future__ import annotations

import inspect

import pytest

triton = pytest.importorskip("triton")


def test_autotune_kwargs_matches_triton_signature():
    """We inject ``cache_results=True`` iff Triton's ``autotune`` accepts it.

    On Triton 3.0 - 3.3 the parameter doesn't exist and passing it would
    raise ``TypeError`` from the decorator. On Triton 3.4+ it's supported
    and writes the best-config to ``$TRITON_CACHE_DIR`` so subsequent
    processes skip the benchmark sweep.
    """
    from late_interaction_kernels._autotune import autotune_kwargs

    params = inspect.signature(triton.autotune).parameters
    expected = {"cache_results": True} if "cache_results" in params else {}
    assert autotune_kwargs() == expected, (
        f"autotune_kwargs() drifted from triton.autotune's signature. "
        f"Triton has params={sorted(params)}, autotune_kwargs returned={autotune_kwargs()}"
    )


def test_autotune_kwargs_safe_to_unpack_into_decorator():
    """Whatever ``autotune_kwargs()`` returns must compose with our other
    autotune args — i.e. the call site

        @triton.autotune(configs=..., key=..., **autotune_kwargs())

    must not collide with the args we already pass explicitly.
    """
    from late_interaction_kernels._autotune import autotune_kwargs

    explicit = {"configs", "key", "prune_configs_by", "reset_to_zero", "restore_value"}
    overlap = explicit & autotune_kwargs().keys()
    assert not overlap, (
        f"autotune_kwargs() must only contain keys we don't already set explicitly; colliding keys: {overlap}"
    )
