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
    # requires_grad=True routes through autograd so save_argmax=True, which
    # defeats the small-input bypass and forces the autotune path.
    Q = torch.randn(2, 32, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    for ld in (192, 256, 384, 512, 768, 1024, 1100, 1280):
        D = torch.randn(4, ld, 128, device="cuda", dtype=torch.float16, requires_grad=True)
        _ = maxsim(Q, D)

    cache = _maxsim_fwd_kernel.cache
    assert len(cache) == 1, f"forward autotune cache exploded across Ld: {len(cache)} entries (expected 1)"


def test_forward_kernel_keys_on_lq_bucket():
    """Distinct ``Lq`` buckets get distinct entries (Lq is still in the key)."""
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.forward import _maxsim_fwd_kernel

    _maxsim_fwd_kernel.cache.clear()
    for lq in (32, 128):  # both are exact powers of two → distinct buckets
        Q = torch.randn(2, lq, 128, device="cuda", dtype=torch.float16, requires_grad=True)
        D = torch.randn(4, 256, 128, device="cuda", dtype=torch.float16, requires_grad=True)
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
    # All of these fall in the bucket=32 slot. We deliberately exclude
    # lq=32 from the sweep: `_bucket_lq` only creates a q_mask when Lq
    # < bucket (the padded values need a mask), so lq=32 would land with
    # has_q_mask=False while the others have has_q_mask=True, and the
    # autotune key (which includes has_q_mask) would split the cache.
    for lq in (17, 19, 23, 25, 29, 31):
        Q = torch.randn(2, lq, 128, device="cuda", dtype=torch.float16, requires_grad=True)
        D = torch.randn(4, 256, 128, device="cuda", dtype=torch.float16, requires_grad=True)
        _ = maxsim(Q, D)

    cache = _maxsim_fwd_kernel.cache
    assert len(cache) == 1, f"Lq=17..31 should bucket to a single autotune entry; got {len(cache)}"


def test_large_lq_chunks_to_single_autotune_entry():
    """Long queries collapse onto the Lq=128 autotune entry via chunking.

    Without chunking, Lq ∈ {1024, 2048, 4096} would each get its own bucket →
    3 autotune sweeps (and an unbounded set as ColPali patch counts vary).
    Query-token chunking rewrites every long Lq to the 128-token chunk, so
    they share one entry — the *same* entry a native Lq=128 call uses.
    """
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.forward import _maxsim_fwd_kernel

    _maxsim_fwd_kernel.cache.clear()
    # requires_grad routes through autograd (save_argmax=True) so the
    # small-input bypass never fires and the autotuner always runs. All Lq are
    # exact multiples of 128 (no tail pad → has_q_mask stays False, matching
    # the native Lq=128 key).
    D = torch.randn(4, 256, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    for lq in (128, 1024, 2048, 4096):
        Q = torch.randn(2, lq, 128, device="cuda", dtype=torch.float16, requires_grad=True)
        _ = maxsim(Q, D)

    cache = _maxsim_fwd_kernel.cache
    assert len(cache) == 1, (
        f"chunking must collapse every long Lq onto the Lq=128 entry; got {len(cache)} entries"
    )


def test_chunked_tail_padding_bounds_cache_at_two_entries():
    """Tail-padded long Lq adds at most one extra entry (has_q_mask=True).

    Exact multiples of 128 chunk with no tail → has_q_mask=False; a non-multiple
    (e.g. 1030) synthesizes a q_mask → has_q_mask=True. The two land on distinct
    autotune keys, so a workload mixing both is bounded at exactly 2 — still a
    tiny constant vs the per-Lq sweep, but not literally one.
    """
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.forward import _maxsim_fwd_kernel

    _maxsim_fwd_kernel.cache.clear()
    D = torch.randn(4, 256, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    for lq in (1024, 2048):  # exact multiples → one shared (has_q_mask=False) entry
        Q = torch.randn(2, lq, 128, device="cuda", dtype=torch.float16, requires_grad=True)
        _ = maxsim(Q, D)
    for lq in (1030, 1670):  # tail-padded → one shared (has_q_mask=True) entry
        Q = torch.randn(2, lq, 128, device="cuda", dtype=torch.float16, requires_grad=True)
        _ = maxsim(Q, D)

    cache = _maxsim_fwd_kernel.cache
    assert len(cache) == 2, (
        f"mixed exact-multiple and tail-padded long Lq must bound the cache at 2; got {len(cache)}"
    )


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
    Q = torch.randn(2, 32, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    D = torch.randn(4, 256, 128, device="cuda", dtype=torch.float16, requires_grad=True)
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


def test_backward_unified_autotune_cache_bounded():
    """The unified backward autotunes once per Lq regime, not per batch shape.

    grad_Q/grad_D launch params are autotuned, but the key deliberately
    excludes Nd and Ld (like the forward excludes Ld) so distinct batch sizes
    and doc lengths reuse one cached config. Without that, every new training
    shape would re-trigger the launch-param sweep.
    """
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.backward.unified import _bwd_unified_kernel

    _bwd_unified_kernel.cache.clear()
    # Nq small so the "auto" selector stays on the unified path (high-contention
    # squares now route to lowmem). Lq fixed → single autotune key across Nd, Ld.
    for nd in (16, 32, 64):
        for ld in (180, 256, 512):
            Q = torch.randn(8, 32, 128, device="cuda", dtype=torch.float16, requires_grad=True)
            D = torch.randn(nd, ld, 128, device="cuda", dtype=torch.float16, requires_grad=True)
            maxsim(Q, D).sum().backward()

    assert len(_bwd_unified_kernel.cache) == 1, (
        f"backward autotune cache must stay at 1 across Nd/Ld; got {len(_bwd_unified_kernel.cache)}"
    )


def test_backward_lowmem_autotune_caches_bounded():
    """The lowmem grad_Q and destination-owned grad_D kernels also stay at one
    entry across Nd/Ld: both keys exclude the corpus/doc-length dims.
    """
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.backward.lowmem import _bwd_dD_owned_kernel, _bwd_dQ_kernel

    for k in (_bwd_dQ_kernel, _bwd_dD_owned_kernel):
        k.cache.clear()

    for nd in (16, 32, 64):
        for ld in (180, 256, 512):
            Q = torch.randn(8, 32, 128, device="cuda", dtype=torch.float16, requires_grad=True)
            D = torch.randn(nd, ld, 128, device="cuda", dtype=torch.float16, requires_grad=True)
            maxsim(Q, D, backward="lowmem").sum().backward()
    assert len(_bwd_dD_owned_kernel.cache) == 1, (
        f"lowmem grad_D cache must stay at 1 across Nd/Ld; got {len(_bwd_dD_owned_kernel.cache)}"
    )
    assert len(_bwd_dQ_kernel.cache) == 1, (
        f"grad_Q cache must stay at 1 across Nd/Ld; got {len(_bwd_dQ_kernel.cache)}"
    )


def test_backward_kd_layout_gets_distinct_autotune_entry():
    """The layout flag stays in each backward key so KD vs cross-product don't
    collide. ``unified`` keys on ``kd_layout``; ``lowmem`` keys on ``cross``
    (KD auto-routes to lowmem, so force lowmem on both layouts here)."""
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.backward.lowmem import _bwd_dD_owned_kernel
    from late_interaction_kernels.backward.unified import _bwd_unified_kernel

    _bwd_unified_kernel.cache.clear()
    Q = torch.randn(8, 32, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    D = torch.randn(16, 180, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    maxsim(Q, D).sum().backward()  # cross-product → unified, kd_layout=False
    assert len(_bwd_unified_kernel.cache) == 1

    _bwd_dD_owned_kernel.cache.clear()
    D_kd = torch.randn(8, 4, 180, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    maxsim(Q, D, backward="lowmem").sum().backward()  # lowmem cross → cross=True
    maxsim(Q, D_kd, backward="lowmem").sum().backward()  # lowmem KD → cross=False
    assert len(_bwd_dD_owned_kernel.cache) == 2, (
        f"layout must produce a distinct lowmem backward entry; got {len(_bwd_dD_owned_kernel.cache)}"
    )
