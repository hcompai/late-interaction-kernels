"""PyLate monkey-patch on Apple Silicon.

Mirrors the CUDA suite in ``tests/test_pylate_compat.py`` but exercises
the MPS dispatch path: ``patched_colbert_scores`` should route ``mps``
tensors through :func:`late_interaction_kernels.mps.compile_dispatch.maxsim_mps` (the
``torch.compile``-fused, autograd-aware path) instead of falling
through to PyLate's reference.

Auto-skips on machines without an MPS-capable PyTorch build, just like
``tests/test_mps.py``.
"""

import inspect

import pytest
import torch

mps = pytest.importorskip(
    "torch.backends.mps",
    reason="MPS not available — these tests need an Apple-Silicon PyTorch build.",
)
if not torch.backends.mps.is_available():
    pytest.skip("MPS device not available", allow_module_level=True)

pylate = pytest.importorskip("pylate")

_params = inspect.signature(
    __import__("pylate.scores", fromlist=["colbert_scores"]).colbert_scores
).parameters
if "documents_mask" not in _params or "queries_mask" not in _params:
    pytest.skip(
        "Installed PyLate predates the (queries_mask, documents_mask) signature; "
        "upgrade with `pip install -U 'pylate>=1.3.3'`.",
        allow_module_level=True,
    )


@pytest.fixture(autouse=True)
def _clear_compile_cache():
    """Reset the per-process ``torch.compile`` cache between tests."""
    from late_interaction_kernels.mps import compile_dispatch as _mps_mod

    _mps_mod._compiled_cache.clear()
    yield
    _mps_mod._compiled_cache.clear()


def _l2(x):
    return torch.nn.functional.normalize(x, p=2, dim=-1)


def _make_qd(Nq=2, Nd=4, Lq=16, Ld=64, d=128, dtype=torch.float16):
    Q = _l2(torch.randn(Nq, Lq, d, device="mps", dtype=dtype))
    D = _l2(torch.randn(Nd, Ld, d, device="mps", dtype=dtype))
    return Q, D


def test_patched_routes_mps_to_dispatch_not_fallback():
    """The patched scorer must not silently fall back on MPS tensors."""
    from late_interaction_kernels.pylate_compat import _device_path

    Q, D = _make_qd()
    assert _device_path(Q, D) == "mps"


def test_patched_matches_pylate_reference_on_mps():
    """Patched ``colbert_scores`` on MPS matches PyLate's own implementation."""
    from pylate.scores import colbert_scores as original_fn

    from late_interaction_kernels.pylate_compat import patch_pylate, unpatch_pylate

    Q, D = _make_qd()
    ref = original_fn(Q, D).float()

    patch_pylate()
    try:
        from pylate.scores import colbert_scores as patched_fn

        out = patched_fn(Q, D).float()
        err = (out - ref).abs().max().item()
        denom = max(1.0, ref.abs().max().item())
        assert err / denom < 5e-3, f"err={err}, denom={denom}"
    finally:
        unpatch_pylate()


def test_patched_with_documents_mask_on_mps():
    from late_interaction_kernels.pylate_compat import patch_pylate, unpatch_pylate
    from late_interaction_kernels.reference import maxsim_reference

    Q, D = _make_qd()
    d_mask = (torch.rand(D.shape[0], D.shape[1], device="mps") > 0.2).float()
    d_mask[:, 0] = 1
    ref = maxsim_reference(Q.cpu().float(), D.cpu().float(), d_mask=d_mask.cpu().bool())

    patch_pylate()
    try:
        from pylate.scores import colbert_scores as patched_fn

        out = patched_fn(Q, D, documents_mask=d_mask).cpu().float()
        err = (out - ref).abs().max().item()
        denom = max(1.0, ref.abs().max().item())
        assert err / denom < 5e-3, f"err={err}, denom={denom}"
    finally:
        unpatch_pylate()


def test_patched_with_both_masks_on_mps():
    from pylate.scores import colbert_scores as original_fn

    from late_interaction_kernels.pylate_compat import patch_pylate, unpatch_pylate

    Q, D = _make_qd()
    q_mask = (torch.rand(Q.shape[0], Q.shape[1], device="mps") > 0.2).float()
    q_mask[:, 0] = 1
    d_mask = (torch.rand(D.shape[0], D.shape[1], device="mps") > 0.2).float()
    d_mask[:, 0] = 1
    ref = original_fn(Q, D, queries_mask=q_mask, documents_mask=d_mask).float()

    patch_pylate()
    try:
        from pylate.scores import colbert_scores as patched_fn

        out = patched_fn(Q, D, queries_mask=q_mask, documents_mask=d_mask).float()
        err = (out - ref).abs().max().item()
        denom = max(1.0, ref.abs().max().item())
        assert err / denom < 5e-3, f"err={err}, denom={denom}"
    finally:
        unpatch_pylate()


def test_patched_kd_on_mps():
    """4-D KD layout (each query has its own [Nd, Ld, d] block)."""
    from pylate.scores import colbert_kd_scores as original_fn

    from late_interaction_kernels.pylate_compat import patch_pylate, unpatch_pylate

    Nq, Nd, Lq, Ld, d = 3, 2, 16, 48, 128
    Q = _l2(torch.randn(Nq, Lq, d, device="mps", dtype=torch.float16))
    Dkd = _l2(torch.randn(Nq, Nd, Ld, d, device="mps", dtype=torch.float16))
    q_mask = (torch.rand(Nq, Lq, device="mps") > 0.2).float()
    q_mask[:, 0] = 1
    d_mask = (torch.rand(Nq, Nd, Ld, device="mps") > 0.2).float()
    d_mask[..., 0] = 1
    ref = original_fn(Q, Dkd, queries_mask=q_mask, documents_mask=d_mask).float()

    patch_pylate()
    try:
        from pylate.scores import colbert_kd_scores as patched_fn

        out = patched_fn(Q, Dkd, queries_mask=q_mask, documents_mask=d_mask).float()
        err = (out - ref).abs().max().item()
        denom = max(1.0, ref.abs().max().item())
        assert err / denom < 5e-3, f"err={err}, denom={denom}"
    finally:
        unpatch_pylate()


def test_backward_flows_through_patched_mps_scores():
    """Training scenario: gradients must reach Q and D."""
    from late_interaction_kernels.pylate_compat import patch_pylate, unpatch_pylate

    Q = torch.randn(2, 16, 64, device="mps", dtype=torch.float32, requires_grad=True)
    D = torch.randn(4, 32, 64, device="mps", dtype=torch.float32, requires_grad=True)
    Qn = _l2(Q)
    Dn = _l2(D)

    patch_pylate()
    try:
        from pylate.scores import colbert_scores as patched_fn

        scores = patched_fn(Qn, Dn)
        scores.mean().backward()
    finally:
        unpatch_pylate()

    assert Q.grad is not None and torch.isfinite(Q.grad).all() and Q.grad.abs().sum() > 0
    assert D.grad is not None and torch.isfinite(D.grad).all() and D.grad.abs().sum() > 0


def test_lik_disable_env_falls_back_on_mps(monkeypatch):
    """``LIK_DISABLE=1`` short-circuits the MPS dispatch too."""
    from pylate.scores import colbert_scores as original_fn

    from late_interaction_kernels.pylate_compat import patch_pylate, unpatch_pylate

    Q, D = _make_qd()
    ref = original_fn(Q, D).float()

    monkeypatch.setenv("LIK_DISABLE", "1")
    patch_pylate()
    try:
        from pylate.scores import colbert_scores as patched_fn

        out = patched_fn(Q, D).float()
        assert torch.allclose(out, ref, atol=1e-4)
    finally:
        unpatch_pylate()


def test_device_mismatch_falls_back():
    """MPS Q + CPU D (or vice-versa) must not pick the MPS path."""
    from late_interaction_kernels.pylate_compat import _device_path

    Q_mps = torch.randn(1, 8, 64, device="mps", dtype=torch.float16)
    D_cpu = torch.randn(2, 16, 64, dtype=torch.float16)
    assert _device_path(Q_mps, D_cpu) is None
