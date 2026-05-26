"""Autotune-cache regressions across kernels.

Pins the invariants for what stays in / out of each kernel's autotune
key:

* Dense forward: ``Ld`` is OUT of the key — distinct doc lengths share
  one cache entry. ``Lq`` is IN the key (drives ``tl.static_range``
  unrolling).
* Scatter pair-list kernel: ``max_lq`` and ``max_ld`` are OUT of the
  key — the kernel keys only on ``d_pad``, so variable-seqlen pair
  lists share a single autotune entry.
"""

import pytest
import torch

pytestmark = pytest.mark.cuda


def test_forward_kernel_compiles_once_for_varying_ld():
    """Many distinct ``Ld`` values share one autotune-cache entry."""
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.forward import _maxsim_fwd_kernel

    _maxsim_fwd_kernel.cache.clear()
    Q = torch.randn(2, 32, 128, device="cuda", dtype=torch.float16)
    for ld in (192, 256, 384, 512, 768, 1024, 1100, 1280):
        D = torch.randn(4, ld, 128, device="cuda", dtype=torch.float16)
        _ = maxsim(Q, D)

    cache = _maxsim_fwd_kernel.cache
    assert len(cache) == 1, f"forward autotune cache exploded across Ld: {len(cache)} entries (expected 1)"


def test_forward_kernel_keys_on_lq_bucket():
    """Distinct ``Lq`` buckets get distinct entries (Lq is still in the key)."""
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.forward import _maxsim_fwd_kernel

    _maxsim_fwd_kernel.cache.clear()
    for lq in (32, 128):  # both are exact powers of two → distinct buckets
        Q = torch.randn(2, lq, 128, device="cuda", dtype=torch.float16)
        D = torch.randn(4, 256, 128, device="cuda", dtype=torch.float16)
        _ = maxsim(Q, D)

    cache = _maxsim_fwd_kernel.cache
    assert len(cache) == 2, (
        f"Lq bucket must stay in the autotune key; got {len(cache)} entries for 2 distinct buckets"
    )


def test_forward_kernel_buckets_lq_to_pow2():
    """Many non-pow2 ``Lq`` values inside one bucket share a single entry.

    Real-world ColBERT/ColPali training has Lq varying with the tokenizer
    output. Without bucketing each new Lq re-triggered the full autotune
    sweep. Bucketing to power-of-two collapses them to one entry per bucket.
    """
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.forward import _maxsim_fwd_kernel

    _maxsim_fwd_kernel.cache.clear()
    # All of these fall in the bucket=32 slot.
    for lq in (17, 19, 23, 25, 29, 31, 32):
        Q = torch.randn(2, lq, 128, device="cuda", dtype=torch.float16)
        D = torch.randn(4, 256, 128, device="cuda", dtype=torch.float16)
        _ = maxsim(Q, D)

    cache = _maxsim_fwd_kernel.cache
    assert len(cache) == 1, f"Lq=17..32 should bucket to a single autotune entry; got {len(cache)}"


def test_scatter_kernel_compiles_once_for_varying_max_ld():
    """Many distinct ``max_ld`` values share one autotune-cache entry."""
    from late_interaction_kernels.score_pairs import _scatter_fwd_kernel, score_pairs_packed

    _scatter_fwd_kernel.cache.clear()

    d_emb = 64
    lq = 8
    Q = torch.randn(lq, d_emb, device="cuda", dtype=torch.float16)
    cu_q = torch.tensor([0, lq], device="cuda", dtype=torch.int32)
    pair_q = torch.zeros(1, device="cuda", dtype=torch.int32)
    pair_d = torch.zeros(1, device="cuda", dtype=torch.int32)

    for max_ld in (64, 128, 256, 512, 1024):
        D = torch.randn(max_ld, d_emb, device="cuda", dtype=torch.float16)
        cu_d = torch.tensor([0, max_ld], device="cuda", dtype=torch.int32)
        _ = score_pairs_packed(Q, D, cu_q, cu_d, pair_q, pair_d, max_seqlen_q=lq, max_seqlen_d=max_ld)

    cache = _scatter_fwd_kernel.cache
    assert len(cache) == 1, (
        f"scatter autotune cache exploded across max_ld: {len(cache)} entries (expected 1)"
    )


def test_forward_normalize_shares_autotune_entry():
    """Toggling ``normalize`` must NOT spawn a new autotune entry.

    ``normalize`` is a tl.constexpr knob that adds ~3 ops to the inner Ld
    loop (an L2-norm + multiply). It doesn't shift register pressure enough
    to change the winning ``(BLOCK_Q, BLOCK_D, num_warps, num_stages)``
    config, so keeping it in the autotune key would just double the cache
    cardinality for free. This test is the canary if anyone re-adds it.
    """
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.forward import _maxsim_fwd_kernel

    _maxsim_fwd_kernel.cache.clear()
    Q = torch.randn(2, 32, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(4, 256, 128, device="cuda", dtype=torch.float16)
    _ = maxsim(Q, D, normalize=False)
    _ = maxsim(Q, D, normalize=True)

    cache = _maxsim_fwd_kernel.cache
    assert len(cache) == 1, (
        f"normalize=True/False must share one autotune entry; got {len(cache)}. "
        "If autotune behaviour changed and normalize genuinely shifts the optimum, "
        "re-add it to the key in forward.py and update this test."
    )


def test_forward_small_input_bypasses_autotune():
    """Small inference shapes route through the no-autotune fast path.

    ``_run_forward`` checks ``_should_bypass_autotune(...)`` and, when true,
    calls ``_maxsim_fwd_kernel.fn[grid](...)`` directly with a fixed config.
    The autotune cache must stay empty in that case — that's the entire
    point of the bypass (no benchmark, no first-call stall).
    """
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.forward import _maxsim_fwd_kernel

    _maxsim_fwd_kernel.cache.clear()
    # Nq * Nd = 1 * 100 = 100 ≤ 500, d = 128 ≤ 256 → bypass triggers.
    Q = torch.randn(1, 32, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(100, 180, 128, device="cuda", dtype=torch.float16)
    out = maxsim(Q, D)

    assert out.shape == (1, 100)
    assert len(_maxsim_fwd_kernel.cache) == 0, (
        f"small-input call must bypass the autotuner; cache has {len(_maxsim_fwd_kernel.cache)} entries"
    )


def test_forward_large_input_goes_through_autotune():
    """Above the bypass threshold the autotune path runs and caches a winner."""
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.forward import _maxsim_fwd_kernel

    _maxsim_fwd_kernel.cache.clear()
    # Nq * Nd = 32 * 32 = 1024 > 500 → autotune.
    Q = torch.randn(32, 32, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(32, 180, 128, device="cuda", dtype=torch.float16)
    _ = maxsim(Q, D)

    assert len(_maxsim_fwd_kernel.cache) == 1, (
        f"large-input call must populate the autotune cache; got {len(_maxsim_fwd_kernel.cache)} entries"
    )


def test_forward_compute_bound_shape_skips_bypass():
    """ColPali-style compute-bound shapes must NOT hit the bypass.

    Grid is small enough (Nq*Nd ≤ 500) that the original ``_SMALL_BYPASS_NQND``
    gate would accept, but Lq*Ld = 1M makes the kernel compute-bound. The
    fixed bypass tile ``(BLOCK_Q=32, BLOCK_D=64, warps=4)`` loses ~2.4× to
    the autotuned Hopper compute winner ``(128, 128, warps=8)`` on this
    shape, so we want the autotuner to run.
    """
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.forward import _maxsim_fwd_kernel

    _maxsim_fwd_kernel.cache.clear()
    # Nq*Nd = 16*16 = 256 ≤ 500 ✓, but Lq*Ld = 1024*1024 = 1M > 200_000 ✗
    # → autotune.
    Q = torch.randn(16, 1024, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(16, 1024, 128, device="cuda", dtype=torch.float16)
    _ = maxsim(Q, D)

    assert len(_maxsim_fwd_kernel.cache) == 1, (
        f"compute-bound shape must populate the autotune cache; got {len(_maxsim_fwd_kernel.cache)} entries"
    )


def test_forward_bypass_matches_autotune_path():
    """Bypass kernel and autotuned kernel produce numerically equivalent scores.

    Both call the same Triton kernel body — only the ``(BLOCK_Q, BLOCK_D,
    num_warps, num_stages)`` tuple differs. Within fp32-accumulator
    nondeterminism (reduction order can shift across configs), the scores
    must agree to fp16-input slack.
    """
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.reference import maxsim_reference

    torch.manual_seed(0)
    # Small enough to bypass (Nq*Nd=200 ≤ 500).
    Q = torch.randn(2, 32, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(100, 180, 128, device="cuda", dtype=torch.float16)

    fast = maxsim(Q, D).float()
    ref = maxsim_reference(Q.float(), D.float())
    torch.testing.assert_close(fast, ref, rtol=5e-3, atol=5e-3)


def test_scatter_kernel_compiles_once_for_varying_max_lq():
    """Many distinct ``max_lq`` values share one autotune-cache entry."""
    from late_interaction_kernels.score_pairs import _scatter_fwd_kernel, score_pairs_packed

    _scatter_fwd_kernel.cache.clear()

    d_emb = 64
    ld = 64
    D = torch.randn(ld, d_emb, device="cuda", dtype=torch.float16)
    cu_d = torch.tensor([0, ld], device="cuda", dtype=torch.int32)
    pair_q = torch.zeros(1, device="cuda", dtype=torch.int32)
    pair_d = torch.zeros(1, device="cuda", dtype=torch.int32)

    for lq in (8, 16, 32, 64, 128):
        Q = torch.randn(lq, d_emb, device="cuda", dtype=torch.float16)
        cu_q = torch.tensor([0, lq], device="cuda", dtype=torch.int32)
        _ = score_pairs_packed(Q, D, cu_q, cu_d, pair_q, pair_d, max_seqlen_q=lq, max_seqlen_d=ld)

    cache = _scatter_fwd_kernel.cache
    assert len(cache) == 1, (
        f"scatter autotune cache exploded across max_lq: {len(cache)} entries (expected 1)"
    )
