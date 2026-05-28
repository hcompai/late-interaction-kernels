"""MPS Metal kernel tests.

These exercise the ``simdgroup_matrix``-based forward kernel in
:mod:`late_interaction_kernels.metal` directly, plus the
:mod:`late_interaction_kernels.mps.compile_dispatch` dispatch heuristic that routes
inference between the Metal kernel and the compile path.

Skips on machines without an MPS-capable PyTorch build or without
``torch.mps.compile_shader`` (PyTorch < 2.10).
"""

import pytest
import torch

mps = pytest.importorskip(
    "torch.backends.mps",
    reason="MPS not available — these tests need an Apple-Silicon PyTorch build.",
)
if not torch.backends.mps.is_available():
    pytest.skip("MPS device not available", allow_module_level=True)

from late_interaction_kernels.mps import metal as _metal  # noqa: E402

if not _metal.is_available():
    pytest.skip(
        "torch.mps.compile_shader not available (needs PyTorch ≥ 2.10)",
        allow_module_level=True,
    )

from late_interaction_kernels.reference import maxsim_reference  # noqa: E402


def _ref(Q, D, q_mask=None, d_mask=None, normalize=True):
    return maxsim_reference(
        Q.cpu().float(),
        D.cpu().float(),
        q_mask=None if q_mask is None else q_mask.cpu(),
        d_mask=None if d_mask is None else d_mask.cpu(),
        normalize=normalize,
    )


def _kd_ref(Q, D, q_mask=None, d_mask=None, normalize=True):
    """Dense KD reference: ``Q[Nq, Lq, d] × D[Nq, K, Ld, d] -> [Nq, K]``."""
    import torch.nn.functional as F

    from late_interaction_kernels.reference import NEG_INF

    Qf = Q.cpu().float()
    Df = D.cpu().float()
    if normalize:
        Qf = F.normalize(Qf, p=2, dim=-1, eps=1e-12)
        Df = F.normalize(Df, p=2, dim=-1, eps=1e-12)
    S = torch.einsum("ild,iktd->iklt", Qf, Df)  # [Nq, K, Lq, Ld]
    if d_mask is not None:
        S = S.masked_fill(~d_mask.cpu().bool().unsqueeze(2), NEG_INF)
    row_max = S.max(dim=-1).values
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    if q_mask is not None:
        row_max = row_max * q_mask.cpu().to(row_max.dtype).unsqueeze(1)
    return row_max.sum(dim=-1)


def _rel(out: torch.Tensor, ref: torch.Tensor) -> float:
    return (out - ref).abs().max().item() / max(1e-6, ref.abs().max().item())


# --------------------------------------------------------------------------- #
# Forward parity                                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 8, 8, 8),
        (1, 1, 32, 32, 64),
        (1, 1, 32, 32, 128),
        (2, 3, 32, 128, 128),
        (1, 100, 32, 1024, 128),
        (1, 1000, 32, 304, 128),  # Ld not multiple of 32
        (4, 8, 32, 200, 128),
        (1, 4, 16, 32, 48),  # d=48
        (1, 4, 16, 32, 96),  # d=96
        (1, 4, 16, 32, 128),  # d=128 (max)
        (1, 4, 64, 128, 128),  # Lq > BLOCK_Q
        (1, 4, 200, 128, 128),  # Lq not multiple of BLOCK_Q
        (1, 4, 256, 128, 128),
        (1, 4, 1024, 128, 128),  # Lq much > BLOCK_Q
    ],
    ids=lambda s: f"Nq{s[0]}_Nd{s[1]}_Lq{s[2]}_Ld{s[3]}_d{s[4]}",
)
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize(
    "dtype,tol",
    [(torch.float16, 5e-3), (torch.bfloat16, 3e-2)],
    ids=["fp16", "bf16"],
)
def test_metal_kernel_matches_reference(shape, normalize, dtype, tol):
    Nq, Nd, Lq, Ld, d = shape
    torch.manual_seed(0)
    Q = torch.randn(Nq, Lq, d, dtype=dtype, device="mps")
    D = torch.randn(Nd, Ld, d, dtype=dtype, device="mps")
    out = _metal.maxsim_inference_metal(Q, D, normalize=normalize)
    assert out.shape == (Nq, Nd)
    assert out.dtype == torch.float32
    rel = _rel(out.cpu(), _ref(Q, D, normalize=normalize))
    assert rel < tol, f"rel err {rel:.2e} exceeds {tol}"


def test_metal_kernel_q_mask_matches_reference():
    torch.manual_seed(0)
    Q = torch.randn(2, 16, 64, dtype=torch.float16, device="mps")
    D = torch.randn(4, 32, 64, dtype=torch.float16, device="mps")
    qm = torch.ones(2, 16, dtype=torch.bool, device="mps")
    qm[:, -3:] = False
    out = _metal.maxsim_inference_metal(Q, D, q_mask=qm, normalize=True)
    rel = _rel(out.cpu(), _ref(Q, D, q_mask=qm, normalize=True))
    assert rel < 5e-3


def test_metal_kernel_d_mask_matches_reference():
    torch.manual_seed(0)
    Q = torch.randn(2, 16, 64, dtype=torch.float16, device="mps")
    D = torch.randn(4, 32, 64, dtype=torch.float16, device="mps")
    dm = torch.ones(4, 32, dtype=torch.bool, device="mps")
    dm[:, -7:] = False
    out = _metal.maxsim_inference_metal(Q, D, d_mask=dm, normalize=True)
    rel = _rel(out.cpu(), _ref(Q, D, d_mask=dm, normalize=True))
    assert rel < 5e-3


def test_metal_kernel_both_masks_matches_reference():
    torch.manual_seed(0)
    Q = torch.randn(3, 20, 96, dtype=torch.float16, device="mps")
    D = torch.randn(5, 40, 96, dtype=torch.float16, device="mps")
    qm = torch.ones(3, 20, dtype=torch.bool, device="mps")
    qm[:, -2:] = False
    dm = torch.ones(5, 40, dtype=torch.bool, device="mps")
    dm[:, -8:] = False
    out = _metal.maxsim_inference_metal(Q, D, q_mask=qm, d_mask=dm, normalize=True)
    rel = _rel(out.cpu(), _ref(Q, D, q_mask=qm, d_mask=dm, normalize=True))
    assert rel < 5e-3


def test_metal_kernel_handles_2d_inputs():
    """``[Lq, d]`` / ``[Ld, d]`` inputs squeeze into a scalar."""
    Q = torch.randn(8, 64, dtype=torch.float16, device="mps")
    D = torch.randn(16, 64, dtype=torch.float16, device="mps")
    out = _metal.maxsim_inference_metal(Q, D, normalize=True)
    assert out.shape == ()
    rel = _rel(out.cpu().reshape(1, 1), _ref(Q, D, normalize=True))
    assert rel < 5e-3


def test_metal_kernel_full_q_mask_is_zero():
    """An all-False q_mask zeros every score (no -inf bleed-through)."""
    Q = torch.randn(2, 8, 64, dtype=torch.float16, device="mps")
    D = torch.randn(3, 16, 64, dtype=torch.float16, device="mps")
    qm = torch.zeros(2, 8, dtype=torch.bool, device="mps")
    out = _metal.maxsim_inference_metal(Q, D, q_mask=qm, normalize=True)
    assert torch.all(out == 0)


def test_metal_kernel_full_d_mask_is_zero():
    """An all-False d_mask zeros every score: every Q row sees only -inf,
    which the kernel clamps to 0 to match the reference contract."""
    Q = torch.randn(2, 8, 64, dtype=torch.float16, device="mps")
    D = torch.randn(3, 16, 64, dtype=torch.float16, device="mps")
    dm = torch.zeros(3, 16, dtype=torch.bool, device="mps")
    out = _metal.maxsim_inference_metal(Q, D, d_mask=dm, normalize=True)
    assert torch.all(out == 0)


# --------------------------------------------------------------------------- #
# KD / pairs layout (4-D D)                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "shape",
    [
        # (Nq, K, Lq, Ld, d)
        (1, 1, 32, 256, 128),
        (1, 10, 32, 300, 128),
        (4, 8, 32, 256, 128),
        (8, 32, 32, 200, 128),
        (2, 4, 32, 64, 48),
        (1, 16, 16, 512, 96),
        (1, 32, 32, 1024, 128),
        (2, 3, 12, 23, 64),
    ],
    ids=lambda s: f"Nq{s[0]}_K{s[1]}_Lq{s[2]}_Ld{s[3]}_d{s[4]}",
)
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize(
    "dtype,tol",
    [(torch.float16, 5e-3), (torch.bfloat16, 3e-2)],
    ids=["fp16", "bf16"],
)
def test_metal_kd_matches_reference(shape, normalize, dtype, tol):
    """KD layout (D 4-D) matches the dense einsum reference within fp16/bf16 tolerance."""
    Nq, K, Lq, Ld, d = shape
    torch.manual_seed(0)
    Q = torch.randn(Nq, Lq, d, dtype=dtype, device="mps")
    D = torch.randn(Nq, K, Ld, d, dtype=dtype, device="mps")
    out = _metal.maxsim_inference_metal(Q, D, normalize=normalize)
    assert out.shape == (Nq, K)
    assert out.dtype == torch.float32
    rel = _rel(out.cpu(), _kd_ref(Q, D, normalize=normalize))
    assert rel < tol, f"rel err {rel:.2e} exceeds {tol}"


def test_metal_kd_with_d_mask_matches_reference():
    """KD layout with a 3-D ``d_mask`` of shape ``[Nq, K, Ld]``."""
    torch.manual_seed(0)
    Nq, K, Lq, Ld, d = 2, 4, 16, 32, 64
    Q = torch.randn(Nq, Lq, d, dtype=torch.float16, device="mps")
    D = torch.randn(Nq, K, Ld, d, dtype=torch.float16, device="mps")
    dm = torch.ones(Nq, K, Ld, dtype=torch.bool, device="mps")
    dm[:, :, -5:] = False  # last 5 doc tokens masked across every slab
    out = _metal.maxsim_inference_metal(Q, D, d_mask=dm, normalize=True)
    rel = _rel(out.cpu(), _kd_ref(Q, D, d_mask=dm, normalize=True))
    assert rel < 5e-3


def test_metal_kd_with_q_mask_matches_reference():
    """KD layout with a 2-D ``q_mask`` ``[Nq, Lq]`` (one mask per query)."""
    torch.manual_seed(0)
    Nq, K, Lq, Ld, d = 3, 5, 12, 24, 64
    Q = torch.randn(Nq, Lq, d, dtype=torch.float16, device="mps")
    D = torch.randn(Nq, K, Ld, d, dtype=torch.float16, device="mps")
    qm = torch.ones(Nq, Lq, dtype=torch.bool, device="mps")
    qm[:, -3:] = False
    out = _metal.maxsim_inference_metal(Q, D, q_mask=qm, normalize=True)
    rel = _rel(out.cpu(), _kd_ref(Q, D, q_mask=qm, normalize=True))
    assert rel < 5e-3


def test_metal_kd_matches_cross_product_diagonal():
    """K=1 KD layout equals cross-product diagonal — same kernel, same answer."""
    torch.manual_seed(0)
    Nq, Lq, Ld, d = 4, 8, 16, 64
    Q = torch.randn(Nq, Lq, d, dtype=torch.float16, device="mps")
    D_pairs = torch.randn(Nq, Ld, d, dtype=torch.float16, device="mps")
    # KD with K=1
    kd_out = _metal.maxsim_inference_metal(Q, D_pairs.unsqueeze(1), normalize=True)
    # Cross-product, take the diagonal
    xp_out = _metal.maxsim_inference_metal(Q, D_pairs, normalize=True)
    assert kd_out.shape == (Nq, 1)
    assert torch.allclose(kd_out.cpu().squeeze(-1), xp_out.cpu().diagonal(), atol=1e-3)


def test_supports_accepts_4d_kd_shape():
    Q = torch.randn(4, 32, 128, dtype=torch.float16, device="mps")
    D = torch.randn(4, 16, 256, 128, dtype=torch.float16, device="mps")
    assert _metal.supports(Q, D)


def test_supports_rejects_4d_mismatched_batch():
    """KD layout requires ``Q.shape[0] == D.shape[0]`` (one slab per query)."""
    Q = torch.randn(4, 32, 128, dtype=torch.float16, device="mps")
    D = torch.randn(3, 16, 256, 128, dtype=torch.float16, device="mps")
    assert not _metal.supports(Q, D)


def test_metal_kd_rejects_wrong_q_dim():
    """KD layout needs Q to be 3-D; reject 2-D Q (ambiguous Nq)."""
    Q = torch.randn(32, 128, dtype=torch.float16, device="mps")
    D = torch.randn(1, 8, 256, 128, dtype=torch.float16, device="mps")
    with pytest.raises(ValueError, match="KD layout"):
        _metal.maxsim_inference_metal(Q, D, normalize=True)


def test_metal_kd_rejects_d_mask_wrong_shape():
    """KD ``d_mask`` must be ``[Nq, K, Ld]`` — flat 2-D mask is rejected."""
    Q = torch.randn(2, 8, 64, dtype=torch.float16, device="mps")
    D = torch.randn(2, 3, 16, 64, dtype=torch.float16, device="mps")
    dm = torch.ones(6, 16, dtype=torch.bool, device="mps")  # flat [Nq*K, Ld]
    with pytest.raises(ValueError, match="d_mask"):
        _metal.maxsim_inference_metal(Q, D, d_mask=dm, normalize=True)


# --------------------------------------------------------------------------- #
# KD dispatch (4-D D routes through `maxsim_inference_mps`)                   #
# --------------------------------------------------------------------------- #


def test_dispatch_routes_4d_d_through_metal_when_worthwhile():
    """A typical KD inference shape (Nq=4, K=32, Ld=300) goes to Metal."""
    from late_interaction_kernels.mps import compile_dispatch as _mps_mod
    from late_interaction_kernels.mps.compile_dispatch import maxsim_inference_mps

    _mps_mod._compiled_cache.clear()
    Q = torch.randn(4, 32, 128, dtype=torch.float16, device="mps")
    D = torch.randn(4, 32, 300, 128, dtype=torch.float16, device="mps")
    out = maxsim_inference_mps(Q, D, normalize=True)
    assert out.shape == (4, 32)
    # Metal path doesn't touch the compile cache.
    assert len(_mps_mod._compiled_cache) == 0
    rel = _rel(out.cpu(), _kd_ref(Q, D, normalize=True))
    assert rel < 5e-3


def test_dispatch_falls_back_to_kd_reference_for_fp32(monkeypatch):
    """fp32 KD inputs route to the dense KD reference, not the Metal path."""
    from late_interaction_kernels.mps.compile_dispatch import maxsim_inference_mps

    Q = torch.randn(2, 16, 64, dtype=torch.float32, device="mps")
    D = torch.randn(2, 4, 32, 64, dtype=torch.float32, device="mps")
    out = maxsim_inference_mps(Q, D, normalize=True)
    assert out.shape == (2, 4)
    rel = _rel(out.cpu(), _kd_ref(Q, D, normalize=True))
    assert rel < 1e-4


# --------------------------------------------------------------------------- #
# `_pack_params` cache                                                        #
# --------------------------------------------------------------------------- #


def test_pack_params_cache_returns_same_tensor_for_identical_key():
    """Same (Nq, Nd, Lq, Ld, d, flags) -> exact same device tensor."""
    _metal._params_cache.clear()
    p1 = _metal._pack_params(1, 8, 32, 200, 128, 5)
    p2 = _metal._pack_params(1, 8, 32, 200, 128, 5)
    assert p1.data_ptr() == p2.data_ptr()


def test_pack_params_cache_distinguishes_keys():
    """Different flags = different cache entry."""
    _metal._params_cache.clear()
    p_norm = _metal._pack_params(1, 8, 32, 200, 128, _metal._FLAG_NORMALIZE)
    p_no_norm = _metal._pack_params(1, 8, 32, 200, 128, 0)
    assert p_norm.data_ptr() != p_no_norm.data_ptr()


def test_kd_q_mask_wrong_shape_raises():
    """KD layout must reject a q_mask that doesn't match (Nq, Lq)."""
    Q = torch.randn(2, 8, 32, device="mps", dtype=torch.float16)
    D = torch.randn(2, 3, 16, 32, device="mps", dtype=torch.float16)
    q_mask = torch.ones(8, dtype=torch.bool, device="mps")
    with pytest.raises(ValueError, match="q_mask"):
        _metal.maxsim_inference_metal(Q, D, q_mask=q_mask, normalize=True)


def test_cross_product_q_dim_4_raises():
    """4-D Q with 3-D D must fail clearly, not crash on shape unpack."""
    Q = torch.randn(1, 2, 8, 32, device="mps", dtype=torch.float16)
    D = torch.randn(2, 16, 32, device="mps", dtype=torch.float16)
    with pytest.raises(ValueError, match=r"Q\.dim\(\)"):
        _metal.maxsim_inference_metal(Q, D, normalize=True)


def test_supports_rejects_1d_d():
    """`supports()` must return False for 1-D D (no Ld axis)."""
    Q = torch.randn(8, 32, device="mps", dtype=torch.float16)
    D = torch.randn(32, device="mps", dtype=torch.float16)
    assert _metal.supports(Q, D) is False


def test_pack_params_cache_lru_eviction():
    """LRU: at capacity, the least-recently-used entry gets evicted; size stays
    pinned at ``_PARAMS_CACHE_MAX`` instead of thrashing to 1 on every overflow."""
    _metal._params_cache.clear()
    for nd in range(_metal._PARAMS_CACHE_MAX):
        _metal._pack_params(1, nd, 32, 200, 128, 0)
    assert len(_metal._params_cache) == _metal._PARAMS_CACHE_MAX

    # Touch key Nd=0 so it's now the most-recently-used; Nd=1 is now the LRU.
    _metal._pack_params(1, 0, 32, 200, 128, 0)

    _metal._pack_params(1, _metal._PARAMS_CACHE_MAX, 32, 200, 128, 0)
    assert len(_metal._params_cache) == _metal._PARAMS_CACHE_MAX
    assert (1, 0, 32, 200, 128, 0) in _metal._params_cache  # touched -> kept
    assert (1, 1, 32, 200, 128, 0) not in _metal._params_cache  # LRU -> evicted


def test_pack_params_bytes_match_struct_pack():
    """Cached tensor must be byte-identical to a fresh ``struct.pack``."""
    _metal._params_cache.clear()
    import struct

    args = (3, 12, 32, 256, 96, 7)
    raw = struct.pack(_metal._PARAMS_FORMAT, *args)
    expected = torch.frombuffer(bytearray(raw), dtype=torch.int32)
    got = _metal._pack_params(*args).cpu()
    assert torch.equal(got, expected)


def test_dispatch_force_reference_handles_4d_d(monkeypatch):
    """``LIK_FORCE_MPS_BACKEND=reference`` must accept 4-D D."""
    monkeypatch.setenv("LIK_FORCE_MPS_BACKEND", "reference")
    from late_interaction_kernels.mps.compile_dispatch import maxsim_inference_mps, maxsim_mps

    Q = torch.randn(2, 8, 64, dtype=torch.float16, device="mps")
    D = torch.randn(2, 3, 16, 64, dtype=torch.float16, device="mps")
    out = maxsim_inference_mps(Q, D, normalize=True)
    assert out.shape == (2, 3)
    # Training-aware path (autograd) must also work on 4-D D.
    out2 = maxsim_mps(Q, D, normalize=True)
    assert torch.allclose(out, out2, atol=1e-4)


# --------------------------------------------------------------------------- #
# supports() / fallback contract                                              #
# --------------------------------------------------------------------------- #


def test_supports_rejects_fp32():
    Q = torch.randn(1, 8, 128, dtype=torch.float32, device="mps")
    D = torch.randn(1, 16, 128, dtype=torch.float32, device="mps")
    assert not _metal.supports(Q, D)


def test_supports_rejects_d_too_large():
    Q = torch.randn(1, 8, 192, dtype=torch.float16, device="mps")
    D = torch.randn(1, 16, 192, dtype=torch.float16, device="mps")
    assert not _metal.supports(Q, D)


def test_supports_rejects_d_not_multiple_of_8():
    Q = torch.randn(1, 8, 60, dtype=torch.float16, device="mps")
    D = torch.randn(1, 16, 60, dtype=torch.float16, device="mps")
    assert not _metal.supports(Q, D)


def test_supports_rejects_dtype_mismatch():
    Q = torch.randn(1, 8, 128, dtype=torch.float16, device="mps")
    D = torch.randn(1, 16, 128, dtype=torch.bfloat16, device="mps")
    assert not _metal.supports(Q, D)


def test_supports_accepts_typical_inference_shape():
    Q = torch.randn(1, 32, 128, dtype=torch.float16, device="mps")
    D = torch.randn(100, 256, 128, dtype=torch.float16, device="mps")
    assert _metal.supports(Q, D)


# --------------------------------------------------------------------------- #
# Dispatch                                                                    #
# --------------------------------------------------------------------------- #


def test_dispatch_falls_back_to_compile_for_fp32():
    """fp32 inputs must use the compile path, not crash on Metal."""
    from late_interaction_kernels import MaxSimScorer
    from late_interaction_kernels.mps import compile_dispatch as _mps_mod

    _mps_mod._compiled_cache.clear()
    Q = torch.randn(2, 32, 128, dtype=torch.float32, device="mps")
    D = torch.randn(100, 256, 128, dtype=torch.float32, device="mps")
    out = MaxSimScorer(normalize=True).score(Q, D)
    assert out.shape == (2, 100)
    assert any(key[0] == torch.float32 for key in _mps_mod._compiled_cache)


def test_dispatch_falls_back_to_compile_for_unsupported_d():
    """d > 128 routes to compile (Metal threadgroup-memory + Q-cache cap)."""
    from late_interaction_kernels import MaxSimScorer
    from late_interaction_kernels.mps import compile_dispatch as _mps_mod

    _mps_mod._compiled_cache.clear()
    Q = torch.randn(1, 32, 192, dtype=torch.float16, device="mps")
    D = torch.randn(50, 200, 192, dtype=torch.float16, device="mps")
    out = MaxSimScorer(normalize=True).score(Q, D)
    assert out.shape == (1, 50)
    assert len(_mps_mod._compiled_cache) == 1


def test_dispatch_uses_compile_for_small_batch(monkeypatch):
    """Small Nq*Nd shapes go to compile (lower launch overhead)."""
    monkeypatch.setenv("LIK_MPS_METAL_MIN_BATCH", "10000000")
    from late_interaction_kernels import MaxSimScorer
    from late_interaction_kernels.mps import compile_dispatch as _mps_mod

    _mps_mod._compiled_cache.clear()
    Q = torch.randn(1, 32, 128, dtype=torch.float16, device="mps")
    D = torch.randn(10, 256, 128, dtype=torch.float16, device="mps")
    MaxSimScorer(normalize=True).score(Q, D)
    assert len(_mps_mod._compiled_cache) == 1


def test_force_metal_via_env(monkeypatch):
    """``LIK_FORCE_MPS_BACKEND=metal`` bypasses the heuristic at inference."""
    monkeypatch.setenv("LIK_FORCE_MPS_BACKEND", "metal")
    from late_interaction_kernels import MaxSimScorer
    from late_interaction_kernels.mps import compile_dispatch as _mps_mod

    _mps_mod._compiled_cache.clear()
    # Tiny shape the heuristic would otherwise route to compile.
    Q = torch.randn(1, 32, 128, dtype=torch.float16, device="mps")
    D = torch.randn(2, 64, 128, dtype=torch.float16, device="mps")
    out = MaxSimScorer(normalize=True).score(Q, D)
    rel = _rel(out.cpu(), _ref(Q, D, normalize=True))
    assert rel < 5e-3
    assert len(_mps_mod._compiled_cache) == 0  # compile path untouched


def test_force_compile_via_env(monkeypatch):
    """``LIK_FORCE_MPS_BACKEND=compile`` skips Metal even on a winning shape."""
    monkeypatch.setenv("LIK_FORCE_MPS_BACKEND", "compile")
    from late_interaction_kernels import MaxSimScorer
    from late_interaction_kernels.mps import compile_dispatch as _mps_mod

    _mps_mod._compiled_cache.clear()
    Q = torch.randn(1, 32, 128, dtype=torch.float16, device="mps")
    D = torch.randn(200, 1024, 128, dtype=torch.float16, device="mps")
    MaxSimScorer(normalize=True).score(Q, D)
    assert len(_mps_mod._compiled_cache) == 1


def test_force_reference_via_env(monkeypatch):
    """``LIK_FORCE_MPS_BACKEND=reference`` runs eager — no compile, no Metal."""
    monkeypatch.setenv("LIK_FORCE_MPS_BACKEND", "reference")
    from late_interaction_kernels import MaxSimScorer
    from late_interaction_kernels.mps import compile_dispatch as _mps_mod

    _mps_mod._compiled_cache.clear()
    Q = torch.randn(1, 16, 64, dtype=torch.float16, device="mps")
    D = torch.randn(20, 32, 64, dtype=torch.float16, device="mps")
    out = MaxSimScorer(normalize=True).score(Q, D)
    assert out.shape == (1, 20)
    assert len(_mps_mod._compiled_cache) == 0


def test_metal_path_is_used_for_inference_winning_shape(monkeypatch):
    """A canonical inference shape skips the compile cache → Metal path."""
    from late_interaction_kernels import MaxSimScorer
    from late_interaction_kernels.mps import compile_dispatch as _mps_mod

    monkeypatch.delenv("LIK_FORCE_MPS_BACKEND", raising=False)
    monkeypatch.delenv("LIK_DISABLE_COMPILE", raising=False)
    _mps_mod._compiled_cache.clear()

    Q = torch.randn(1, 32, 128, dtype=torch.float16, device="mps")
    D = torch.randn(200, 1024, 128, dtype=torch.float16, device="mps")
    MaxSimScorer(normalize=True).score(Q, D)
    assert len(_mps_mod._compiled_cache) == 0


def test_train_time_call_uses_compile_path():
    """Autograd-tracking calls must use compile (Metal is forward-only)."""
    from late_interaction_kernels import MaxSimScorer
    from late_interaction_kernels.mps import compile_dispatch as _mps_mod

    _mps_mod._compiled_cache.clear()
    Q = torch.randn(2, 32, 128, dtype=torch.float32, device="mps", requires_grad=True)
    D = torch.randn(200, 1024, 128, dtype=torch.float32, device="mps", requires_grad=True)
    out = MaxSimScorer(normalize=True)(Q, D)  # forward = autograd path
    out.sum().backward()
    assert Q.grad is not None
    assert len(_mps_mod._compiled_cache) >= 1


def test_retrieve_uses_metal_path_when_eligible(monkeypatch):
    """``retrieve()`` is inference-only → it should pick Metal on the right shapes."""
    from late_interaction_kernels import retrieve
    from late_interaction_kernels.mps import compile_dispatch as _mps_mod

    monkeypatch.delenv("LIK_FORCE_MPS_BACKEND", raising=False)
    _mps_mod._compiled_cache.clear()
    Q = torch.randn(1, 32, 128, dtype=torch.float16, device="mps")
    D = torch.randn(200, 1024, 128, dtype=torch.float16, device="mps")
    s, _ = retrieve(Q, D, top_k=10, normalize=True)
    assert s.shape == (1, 10)
    assert len(_mps_mod._compiled_cache) == 0


# --------------------------------------------------------------------------- #
# Numerical edge cases                                                        #
# --------------------------------------------------------------------------- #


def test_metal_kernel_handles_extreme_values():
    """Large-magnitude inputs don't NaN out via the L2-norm clamp."""
    Q = torch.full((1, 8, 64), 1e3, dtype=torch.float16, device="mps")
    D = torch.full((1, 16, 64), 1e3, dtype=torch.float16, device="mps")
    out = _metal.maxsim_inference_metal(Q, D, normalize=True)
    assert torch.isfinite(out).all()


def test_metal_kernel_handles_zero_rows():
    """A query row of all zeros produces a finite (zeroed) score after norm."""
    Q = torch.randn(1, 8, 64, dtype=torch.float16, device="mps")
    Q[0, 3] = 0
    D = torch.randn(2, 16, 64, dtype=torch.float16, device="mps")
    out = _metal.maxsim_inference_metal(Q, D, normalize=True)
    assert torch.isfinite(out).all()


def test_metal_kernel_idempotent_on_repeated_calls():
    """Calling the kernel twice with the same inputs gives the same scores."""
    Q = torch.randn(2, 32, 128, dtype=torch.float16, device="mps")
    D = torch.randn(50, 256, 128, dtype=torch.float16, device="mps")
    a = _metal.maxsim_inference_metal(Q, D, normalize=True)
    b = _metal.maxsim_inference_metal(Q, D, normalize=True)
    assert torch.equal(a, b)
