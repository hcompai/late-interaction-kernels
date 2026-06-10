"""``autotune_kwargs()`` is the single source of truth for shared
``triton.autotune`` keyword arguments. These tests run on CPU — they just
check the feature-detect logic for Triton's persistent on-disk autotune
cache (``cache_results``, added in Triton 3.4).
"""

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


def test_prune_forward_fallback_returns_smallest_footprint():
    """When every config overflows the SMEM budget, the fallback must hand
    back the smallest-footprint candidates, not the first two list entries."""
    from late_interaction_kernels._autotune import _smem_bytes, forward_configs, prune_forward

    configs = forward_configs()
    # d_pad large enough that no config fits any family's budget.
    named_args = {"Lq": 32, "d_pad": 1 << 20}
    kept = prune_forward(configs, named_args)

    assert len(kept) == 2
    d_pad: int = named_args["d_pad"]
    smallest = sorted(_smem_bytes(cfg, d_pad) for cfg in configs)[:2]
    assert sorted(_smem_bytes(cfg, d_pad) for cfg in kept) == smallest


def test_prune_forward_reads_varlen_max_lq_spelling():
    """The varlen kernels spell their query loop bound ``max_lq``; the
    oversized-BLOCK_Q rule must apply to it like to the dense ``Lq``."""
    from late_interaction_kernels._autotune import forward_configs, prune_forward

    kept = prune_forward(forward_configs(), {"max_lq": 16, "d_pad": 64})
    assert kept
    assert all(cfg.kwargs["BLOCK_Q"] <= 32 for cfg in kept)


def test_prune_forward_zero_lq_is_not_treated_as_missing():
    """A legit ``max_lq == 0`` must not fall through to the default 32: every
    BLOCK_Q is oversized for it, so the smallest-footprint fallback applies."""
    from late_interaction_kernels._autotune import _smem_bytes, forward_configs, prune_forward

    configs = forward_configs()
    kept = prune_forward(configs, {"max_lq": 0, "d_pad": 128})

    assert len(kept) == 2
    smallest = sorted(_smem_bytes(cfg, 128) for cfg in configs)[:2]
    assert sorted(_smem_bytes(cfg, 128) for cfg in kept) == smallest


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
