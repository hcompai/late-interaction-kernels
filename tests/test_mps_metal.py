"""MPS Metal kernel tests.

These exercise the ``simdgroup_matrix``-based forward kernel in
:mod:`late_interaction_kernels.metal` directly, plus the
:mod:`late_interaction_kernels._mps` dispatch heuristic that routes
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

from late_interaction_kernels import metal as _metal  # noqa: E402

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
    from late_interaction_kernels import _mps as _mps_mod

    _mps_mod._compiled_cache.clear()
    Q = torch.randn(2, 32, 128, dtype=torch.float32, device="mps")
    D = torch.randn(100, 256, 128, dtype=torch.float32, device="mps")
    out = MaxSimScorer(normalize=True).score(Q, D)
    assert out.shape == (2, 100)
    assert any(key[0] == torch.float32 for key in _mps_mod._compiled_cache)


def test_dispatch_falls_back_to_compile_for_unsupported_d():
    """d > 128 routes to compile (Metal threadgroup-memory + Q-cache cap)."""
    from late_interaction_kernels import MaxSimScorer
    from late_interaction_kernels import _mps as _mps_mod

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
    from late_interaction_kernels import _mps as _mps_mod

    _mps_mod._compiled_cache.clear()
    Q = torch.randn(1, 32, 128, dtype=torch.float16, device="mps")
    D = torch.randn(10, 256, 128, dtype=torch.float16, device="mps")
    MaxSimScorer(normalize=True).score(Q, D)
    assert len(_mps_mod._compiled_cache) == 1


def test_force_metal_via_env(monkeypatch):
    """``LIK_FORCE_MPS_BACKEND=metal`` bypasses the heuristic at inference."""
    monkeypatch.setenv("LIK_FORCE_MPS_BACKEND", "metal")
    from late_interaction_kernels import MaxSimScorer
    from late_interaction_kernels import _mps as _mps_mod

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
    from late_interaction_kernels import _mps as _mps_mod

    _mps_mod._compiled_cache.clear()
    Q = torch.randn(1, 32, 128, dtype=torch.float16, device="mps")
    D = torch.randn(200, 1024, 128, dtype=torch.float16, device="mps")
    MaxSimScorer(normalize=True).score(Q, D)
    assert len(_mps_mod._compiled_cache) == 1


def test_force_reference_via_env(monkeypatch):
    """``LIK_FORCE_MPS_BACKEND=reference`` runs eager — no compile, no Metal."""
    monkeypatch.setenv("LIK_FORCE_MPS_BACKEND", "reference")
    from late_interaction_kernels import MaxSimScorer
    from late_interaction_kernels import _mps as _mps_mod

    _mps_mod._compiled_cache.clear()
    Q = torch.randn(1, 16, 64, dtype=torch.float16, device="mps")
    D = torch.randn(20, 32, 64, dtype=torch.float16, device="mps")
    out = MaxSimScorer(normalize=True).score(Q, D)
    assert out.shape == (1, 20)
    assert len(_mps_mod._compiled_cache) == 0


def test_metal_path_is_used_for_inference_winning_shape(monkeypatch):
    """A canonical inference shape skips the compile cache → Metal path."""
    from late_interaction_kernels import MaxSimScorer
    from late_interaction_kernels import _mps as _mps_mod

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
    from late_interaction_kernels import _mps as _mps_mod

    _mps_mod._compiled_cache.clear()
    Q = torch.randn(2, 32, 128, dtype=torch.float32, device="mps", requires_grad=True)
    D = torch.randn(200, 1024, 128, dtype=torch.float32, device="mps", requires_grad=True)
    out = MaxSimScorer(normalize=True)(Q, D)  # forward = autograd path
    out.sum().backward()
    assert Q.grad is not None
    assert len(_mps_mod._compiled_cache) >= 1


def test_retrieve_uses_metal_path_when_eligible(monkeypatch):
    """``retrieve()`` is inference-only → it should pick Metal on the right shapes."""
    from late_interaction_kernels import _mps as _mps_mod
    from late_interaction_kernels import retrieve

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
