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


def test_fp32_train_time_call_uses_compile_path():
    """fp32 autograd calls fall back to compile (Metal kernels are fp16/bf16 only)."""
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


# ============================================================
# Backward kernel
# ============================================================


def _dense_argmax(Q_hat, D_hat):
    """fp32 reference argmax with the same shape as the kernel buffer."""
    Nq, Lq, _ = Q_hat.shape
    Nd, Ld, _ = D_hat.shape
    S = torch.einsum("ild,jtd->ijlt", Q_hat.float(), D_hat.float())  # [Nq, Nd, Lq, Ld]
    return S.argmax(dim=-1).reshape(Nq * Nd, Lq).to(torch.int32)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize(
    "shape",
    [
        (2, 4, 16, 32, 64),
        (4, 8, 16, 32, 64),
        (8, 16, 32, 64, 128),
        (4, 8, 32, 256, 128),
    ],
)
def test_metal_train_argmax_matches_fp32(dtype, normalize, shape):
    """Kernel argmax agrees with the fp32 reference (per (i, s, j))."""
    Nq, Nd, Lq, Ld, d = shape
    torch.manual_seed(0)
    Q = torch.randn(Nq, Lq, d, device="mps", dtype=dtype)
    D = torch.randn(Nd, Ld, d, device="mps", dtype=dtype)
    _, argmax, _ = _metal.maxsim_train_metal(Q, D, normalize=normalize)

    import torch.nn.functional as F

    Qf = Q.float()
    Df = D.float()
    if normalize:
        Qf = F.normalize(Qf, p=2, dim=-1, eps=1e-12)
        Df = F.normalize(Df, p=2, dim=-1, eps=1e-12)
    # Cast back to inference dtype so close ties resolve the same way the
    # kernel resolves them (this is the apples-to-apples comparison).
    ref_argmax = _dense_argmax(Qf.to(dtype), Df.to(dtype))
    disagree = (argmax.cpu() != ref_argmax.cpu()).sum().item()
    # Allow a tiny number of near-tie disagreements at low precision.
    assert disagree / argmax.numel() < 0.02, (
        f"{disagree}/{argmax.numel()} argmax disagreements (dtype={dtype}, normalize={normalize}, shape={shape})"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape",
    [(2, 4, 16, 32, 64), (4, 8, 32, 64, 128), (4, 16, 32, 256, 128)],
)
def test_metal_backward_grad_Q_matches_argmax_reference(dtype, shape):
    """grad_Q = sum_j gs[i,j] * D[d_global, argmax[i*Nd+j, s], :].

    Uses the kernel's own ``argmax`` so the comparison isolates the
    backward kernel from fp16 argmax noise.
    """
    Nq, Nd, Lq, Ld, d = shape
    torch.manual_seed(0)
    Q = torch.randn(Nq, Lq, d, device="mps", dtype=dtype)
    D = torch.randn(Nd, Ld, d, device="mps", dtype=dtype)
    _, argmax, _ = _metal.maxsim_train_metal(Q, D, normalize=False)
    gs = torch.randn(Nq, Nd, device="mps", dtype=torch.float32)

    gQ, gD = _metal.maxsim_backward_metal(gs, Q, D, argmax)

    am = argmax.view(Nq, Nd, Lq).long()
    j_idx = torch.arange(Nd, device="mps").view(1, Nd, 1).expand(Nq, Nd, Lq)
    D_win = D.float()[j_idx, am]  # [Nq, Nd, Lq, d]
    ref_gQ = (gs.view(Nq, Nd, 1, 1) * D_win).sum(dim=1)  # [Nq, Lq, d]

    diff = (gQ.float() - ref_gQ).abs().max().item()
    scale = ref_gQ.abs().max().item()
    tol = 1e-3 if dtype == torch.float16 else 5e-3
    assert diff / max(scale, 1e-6) < tol, f"gQ rel diff {diff}/{scale}"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(2, 4, 16, 32, 64), (4, 8, 32, 64, 128)])
def test_metal_backward_grad_D_matches_argmax_reference(dtype, shape):
    """grad_D scatters gs[i,j] * Q[i, s, :] into D[d_global, argmax, :]."""
    Nq, Nd, Lq, Ld, d = shape
    torch.manual_seed(0)
    Q = torch.randn(Nq, Lq, d, device="mps", dtype=dtype)
    D = torch.randn(Nd, Ld, d, device="mps", dtype=dtype)
    _, argmax, _ = _metal.maxsim_train_metal(Q, D, normalize=False)
    gs = torch.randn(Nq, Nd, device="mps", dtype=torch.float32)

    _, gD = _metal.maxsim_backward_metal(gs, Q, D, argmax)

    am = argmax.view(Nq, Nd, Lq).long()
    ref_gD = torch.zeros(Nd, Ld, d, device="mps", dtype=torch.float32)
    contrib = gs.view(Nq, Nd, 1, 1) * Q.float().view(Nq, 1, Lq, d)  # [Nq, Nd, Lq, d]
    for j in range(Nd):
        ref_gD[j].index_add_(0, am[:, j, :].reshape(-1), contrib[:, j, :, :].reshape(-1, d))

    diff = (gD.float() - ref_gD).abs().max().item()
    scale = ref_gD.abs().max().item()
    tol = 5e-3 if dtype == torch.float16 else 1e-2
    assert diff / max(scale, 1e-6) < tol, f"gD rel diff {diff}/{scale}"


def test_metal_backward_zeroes_grad_Q_on_masked_rows():
    """grad_Q must be 0 wherever ``q_mask`` is False."""
    Nq, Nd, Lq, Ld, d = 2, 4, 16, 32, 64
    torch.manual_seed(0)
    Q = torch.randn(Nq, Lq, d, device="mps", dtype=torch.float16)
    D = torch.randn(Nd, Ld, d, device="mps", dtype=torch.float16)
    q_mask = torch.ones(Nq, Lq, device="mps", dtype=torch.bool)
    q_mask[:, Lq // 2 :] = False

    _, argmax, _ = _metal.maxsim_train_metal(Q, D, q_mask=q_mask, normalize=False)
    gs = torch.randn(Nq, Nd, device="mps", dtype=torch.float32)
    gQ, _ = _metal.maxsim_backward_metal(gs, Q, D, argmax, q_mask=q_mask.to(torch.int8))

    assert torch.all(gQ[:, Lq // 2 :, :] == 0), "masked rows should have zero gradient"
    assert torch.any(gQ[:, : Lq // 2, :].abs() > 0), "unmasked rows should be nonzero"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(4, 8, 16, 32, 64), (8, 16, 32, 128, 128), (4, 8, 32, 64, 48)])
def test_metal_backward_normalize_fused_matches_host_jacobian(dtype, shape):
    """The in-kernel normalize Jacobian must match the host-side unwind
    (norm + project) byte-equivalent up to fp16 rounding."""
    Nq, Nd, Lq, Ld, d = shape
    torch.manual_seed(0)
    Q = torch.randn(Nq, Lq, d, device="mps", dtype=dtype)
    D = torch.randn(Nd, Ld, d, device="mps", dtype=dtype)
    _, argmax, _ = _metal.maxsim_train_metal(Q, D, normalize=True)
    gs = torch.randn(Nq, Nd, device="mps", dtype=torch.float32)

    gQ_fused, gD_fused = _metal.maxsim_backward_metal(gs, Q, D, argmax, normalize=True)

    # Host-side unwind, same shape as the old _MaxSimFnMetal.backward.
    q_norm = torch.linalg.vector_norm(Q.float(), dim=-1, keepdim=True).clamp_min(1e-6)
    d_norm = torch.linalg.vector_norm(D.float(), dim=-1, keepdim=True).clamp_min(1e-6)
    Q_hat = (Q.float() / q_norm).to(dtype)
    D_hat = (D.float() / d_norm).to(dtype)
    gQh, gDh = _metal.maxsim_backward_metal(gs, Q_hat, D_hat, argmax, normalize=False)
    gQh = gQh.float()
    gDh = gDh.float()
    Qh = Q_hat.float()
    Dh = D_hat.float()
    gQ_host = (gQh - (gQh * Qh).sum(-1, keepdim=True) * Qh) / q_norm
    gD_host = (gDh - (gDh * Dh).sum(-1, keepdim=True) * Dh) / d_norm

    # Atomic-add ordering is nondeterministic so allow a small absolute
    # diff scaled by the magnitude of the gradient.
    gQ_err = (gQ_fused.float() - gQ_host).norm() / gQ_host.norm().clamp_min(1e-6)
    gD_err = (gD_fused.float() - gD_host).norm() / gD_host.norm().clamp_min(1e-6)
    assert gQ_err < 5e-3, f"gQ rel diff {gQ_err}"
    assert gD_err < 5e-3, f"gD rel diff {gD_err}"


def test_metal_backward_KD_layout_matches_reference():
    """4-D ``D`` (KD): backward parity vs the per-pair atomic reference."""
    Nq, K, Lq, Ld, d = 4, 8, 16, 32, 64
    torch.manual_seed(0)
    Q = torch.randn(Nq, Lq, d, device="mps", dtype=torch.float16)
    D4 = torch.randn(Nq, K, Ld, d, device="mps", dtype=torch.float16)
    _, argmax, fwd_ctx = _metal.maxsim_train_metal(Q, D4, normalize=False)
    # fwd_ctx.D is the flattened [Nq*K, Ld, d] view the kernel saw
    D_flat = fwd_ctx.D
    gs = torch.randn(Nq, K, device="mps", dtype=torch.float32)
    gQ, gD = _metal.maxsim_backward_metal(gs, Q, D_flat, argmax, kd_layout=True)

    am = argmax.view(Nq, K, Lq).long()
    # grad_Q[i, s, :] = sum_k gs[i, k] * D[i, k, argmax[i, k, s], :]
    i_idx = torch.arange(Nq, device="mps").view(Nq, 1, 1).expand(Nq, K, Lq)
    k_idx = torch.arange(K, device="mps").view(1, K, 1).expand(Nq, K, Lq)
    D_win = D4.float()[i_idx, k_idx, am]  # [Nq, K, Lq, d]
    ref_gQ = (gs.view(Nq, K, 1, 1) * D_win).sum(dim=1)

    ref_gD = torch.zeros(Nq, K, Ld, d, device="mps", dtype=torch.float32)
    contrib = gs.view(Nq, K, 1, 1) * Q.float().view(Nq, 1, Lq, d)  # [Nq, K, Lq, d]
    for i in range(Nq):
        for k in range(K):
            ref_gD[i, k].index_add_(0, am[i, k], contrib[i, k])

    assert (gQ.float() - ref_gQ).abs().max().item() / max(ref_gQ.abs().max().item(), 1e-6) < 2e-3
    assert (gD.view(Nq, K, Ld, d).float() - ref_gD).abs().max().item() / max(
        ref_gD.abs().max().item(), 1e-6
    ) < 5e-3


def test_metal_backward_accepts_4d_D_directly():
    """`maxsim_backward_metal` accepts the natural 4-D D the train kernel saw,
    mirroring `maxsim_train_metal`. grad_D comes back in the same 4-D shape."""
    Nq, K, Lq, Ld, d = 2, 4, 8, 16, 32
    torch.manual_seed(0)
    Q = torch.randn(Nq, Lq, d, device="mps", dtype=torch.float16)
    D4 = torch.randn(Nq, K, Ld, d, device="mps", dtype=torch.float16)
    _, argmax, _ = _metal.maxsim_train_metal(Q, D4, normalize=False)
    gs = torch.randn(Nq, K, device="mps", dtype=torch.float32)

    # 4-D path (auto-detects kd_layout).
    gQ_4d, gD_4d = _metal.maxsim_backward_metal(gs, Q, D4, argmax)
    assert gD_4d.shape == D4.shape

    # 3-D (flat) path: same kernel, same numbers.
    D_flat = D4.contiguous().view(Nq * K, Ld, d)
    gQ_3d, gD_3d = _metal.maxsim_backward_metal(gs, Q, D_flat, argmax, kd_layout=True)
    assert torch.equal(gQ_4d, gQ_3d)
    assert torch.equal(gD_4d.view(Nq * K, Ld, d), gD_3d)


def test_metal_backward_argmax_sentinel_skips_fully_masked_rows():
    """A fully-d-masked (i, s, j) row writes argmax=-1; the backward must skip it
    (no spurious atomic_add into D[d_global, 0, :])."""
    Nq, Nd, Lq, Ld, d = 1, 1, 4, 8, 32
    torch.manual_seed(0)
    Q = torch.randn(Nq, Lq, d, device="mps", dtype=torch.float16)
    D = torch.randn(Nd, Ld, d, device="mps", dtype=torch.float16)
    d_mask = torch.zeros(Nd, Ld, dtype=torch.bool, device="mps")  # everything masked

    _, argmax, _ = _metal.maxsim_train_metal(Q, D, d_mask=d_mask, normalize=False)
    # Every save slot keeps the -1 sentinel (no valid argmax was ever produced).
    assert (argmax == -1).all()

    gs = torch.randn(Nq, Nd, device="mps", dtype=torch.float32)
    _, gD = _metal.maxsim_backward_metal(gs, Q, D, argmax)
    # No scatter happened -> grad_D is all zeros.
    assert torch.equal(gD, torch.zeros_like(gD))


# ============================================================
# Autograd integration
# ============================================================


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize(
    "shape",
    [(4, 8, 16, 32, 64), (8, 16, 32, 64, 128), (8, 32, 32, 256, 128)],
)
def test_maxsim_mps_autograd_routes_through_metal(monkeypatch, normalize, shape):
    """``maxsim_mps`` end-to-end: scores + grads match the dense reference."""
    from late_interaction_kernels.mps.compile_dispatch import maxsim_mps

    monkeypatch.setenv("LIK_FORCE_MPS_BACKEND", "metal")

    Nq, Nd, Lq, Ld, d = shape
    torch.manual_seed(0)
    Q = torch.randn(Nq, Lq, d, device="mps", dtype=torch.float16, requires_grad=True)
    D = torch.randn(Nd, Ld, d, device="mps", dtype=torch.float16, requires_grad=True)
    Qr = Q.detach().float().requires_grad_(True)
    Dr = D.detach().float().requires_grad_(True)

    scores = maxsim_mps(Q, D, normalize=normalize)
    gs = torch.randn_like(scores)
    scores.backward(gs)

    ref_scores = maxsim_reference(Qr, Dr, normalize=normalize)
    ref_scores.backward(gs)

    # Scores: kernel is fp32 accumulator on fp16 inputs.
    assert (scores - ref_scores).abs().max().item() < 5e-3
    # Gradients have a long-tail of large element-wise diffs whenever an
    # fp16 vs fp32 argmax tie flips (typical ≲ 0.1% of positions). The
    # ``ratio of frobenius norms`` is the right global measure here.
    gQ_err = (Q.grad.float() - Qr.grad).norm().item() / max(Qr.grad.norm().item(), 1e-6)
    gD_err = (D.grad.float() - Dr.grad).norm().item() / max(Dr.grad.norm().item(), 1e-6)
    assert gQ_err < 5e-2, f"grad_Q ‖·‖₂ rel diff {gQ_err}"
    assert gD_err < 5e-2, f"grad_D ‖·‖₂ rel diff {gD_err}"


def test_maxsim_mps_no_grad_uses_compile_path(monkeypatch):
    """When neither Q nor D requires grad, ``maxsim_mps`` skips the Metal autograd path."""
    from late_interaction_kernels.mps import compile_dispatch as cd

    monkeypatch.delenv("LIK_FORCE_MPS_BACKEND", raising=False)

    Q = torch.randn(4, 32, 128, device="mps", dtype=torch.float16)
    D = torch.randn(16, 256, 128, device="mps", dtype=torch.float16)
    # Should run through _compile_path; we just verify it doesn't error
    # and produces finite scores.
    out = cd.maxsim_mps(Q, D, normalize=True)
    assert out.shape == (4, 16)
    assert torch.isfinite(out).all()


def test_maxsim_mps_falls_back_to_compile_for_unsupported_dtype(monkeypatch):
    """fp32 inputs aren't handled by the Metal kernel — must go to compile."""
    from late_interaction_kernels.mps.compile_dispatch import maxsim_mps

    monkeypatch.delenv("LIK_FORCE_MPS_BACKEND", raising=False)

    Q = torch.randn(4, 32, 128, device="mps", dtype=torch.float32, requires_grad=True)
    D = torch.randn(16, 256, 128, device="mps", dtype=torch.float32, requires_grad=True)
    scores = maxsim_mps(Q, D, normalize=True)
    gs = torch.randn_like(scores)
    scores.backward(gs)
    assert Q.grad is not None and D.grad is not None
    assert torch.isfinite(Q.grad).all() and torch.isfinite(D.grad).all()
