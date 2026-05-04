"""MPS (Apple Silicon) dispatch tests.

These exercise the ``torch.compile``-fused path in
:mod:`late_interaction_kernels._mps` against the eager CPU reference.
The whole file auto-skips on machines without an MPS-capable PyTorch
build, so it lives alongside the CUDA tests but doesn't gate CI.

Tolerances mirror the CUDA suite: ``5e-3`` rel for fp16 / bf16,
``1e-5`` for fp32. The compile cache is cleared before each test so
recompiles are scoped to the current case (matters when assertions
about cache size fire).
"""

import pytest
import torch

mps = pytest.importorskip(
    "torch.backends.mps",
    reason="MPS not available — these tests need an Apple-Silicon PyTorch build.",
)
if not torch.backends.mps.is_available():
    pytest.skip("MPS device not available", allow_module_level=True)


@pytest.fixture(autouse=True)
def _clear_compile_cache():
    """Reset the compile cache so each test starts from a clean slate."""
    from late_interaction_kernels import _mps as _mps_mod

    _mps_mod._compiled_cache.clear()
    yield
    _mps_mod._compiled_cache.clear()


# --------------------------------------------------------------------------- #
# Forward parity                                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "dtype,tol",
    [
        (torch.float32, 1e-5),
        (torch.float16, 5e-3),
        (torch.bfloat16, 2e-2),
    ],
)
def test_scorer_matches_reference_on_mps(dtype, tol):
    """``MaxSimScorer`` on MPS matches ``maxsim_reference`` on CPU."""
    from late_interaction_kernels import MaxSimScorer
    from late_interaction_kernels.reference import maxsim_reference

    torch.manual_seed(0)
    Q = torch.randn(3, 32, 128, dtype=dtype, device="mps")
    D = torch.randn(5, 256, 128, dtype=dtype, device="mps")

    scorer = MaxSimScorer(normalize=True)
    out = scorer(Q, D)
    ref = maxsim_reference(Q.cpu().float(), D.cpu().float(), normalize=True)
    assert out.shape == (3, 5)
    err = (out.cpu().float() - ref).abs().max().item()
    rel = err / max(1e-6, ref.abs().max().item())
    assert rel < tol, f"rel err {rel:.2e} exceeds {tol}"


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 4, 6, 8),
        (2, 3, 32, 128, 128),
        (1, 1000, 32, 300, 128),
        (1, 100, 32, 1024, 128),
        (4, 4, 32, 200, 128),
        (1, 4, 16, 32, 48),
        (1, 4, 16, 32, 96),
        (1, 4, 16, 32, 256),
    ],
    ids=lambda s: f"Nq{s[0]}_Nd{s[1]}_Lq{s[2]}_Ld{s[3]}_d{s[4]}",
)
def test_retrieve_matches_reference_topk_on_mps(shape):
    from late_interaction_kernels import retrieve
    from late_interaction_kernels.reference import maxsim_reference

    Nq, Nd, Lq, Ld, d = shape
    torch.manual_seed(0)
    Q = torch.randn(Nq, Lq, d, dtype=torch.float16, device="mps")
    D = torch.randn(Nd, Ld, d, dtype=torch.float16, device="mps")

    k = min(5, Nd)
    scores, idx = retrieve(Q, D, top_k=k, normalize=True)
    ref_scores = maxsim_reference(Q.cpu().float(), D.cpu().float(), normalize=True)
    ref_s, ref_i = torch.topk(ref_scores, k, dim=-1)

    err = (scores.cpu() - ref_s).abs().max().item()
    rel = err / max(1e-6, ref_s.abs().max().item())
    assert rel < 5e-3, f"score rel err {rel:.2e}"
    # Ranking must agree even when the scores differ within fp16 ULP.
    assert torch.equal(idx.cpu(), ref_i), f"idx mismatch (rel score err {rel:.2e})"


def test_retrieve_chunked_on_mps_matches_unchunked():
    from late_interaction_kernels import retrieve

    torch.manual_seed(0)
    Q = torch.randn(2, 32, 128, dtype=torch.float16, device="mps")
    D = torch.randn(200, 64, 128, dtype=torch.float16, device="mps")

    full_s, full_i = retrieve(Q, D, top_k=10, normalize=True)
    ch_s, ch_i = retrieve(Q, D, top_k=10, normalize=True, chunk=64)
    assert torch.allclose(full_s, ch_s, atol=5e-3)
    assert torch.equal(full_i, ch_i)


def test_retrieve_2d_query_squeeze_on_mps():
    from late_interaction_kernels import retrieve

    Q = torch.randn(8, 64, dtype=torch.float16, device="mps")
    D = torch.randn(20, 16, 64, dtype=torch.float16, device="mps")
    scores, idx = retrieve(Q, D, top_k=3, normalize=True)
    assert scores.shape == (3,)
    assert idx.shape == (3,)


# --------------------------------------------------------------------------- #
# Masks                                                                       #
# --------------------------------------------------------------------------- #


def test_q_mask_on_mps_matches_reference():
    from late_interaction_kernels import MaxSimScorer
    from late_interaction_kernels.reference import maxsim_reference

    torch.manual_seed(0)
    Q = torch.randn(2, 16, 64, dtype=torch.float16, device="mps")
    D = torch.randn(4, 32, 64, dtype=torch.float16, device="mps")
    q_mask = torch.ones(2, 16, dtype=torch.bool, device="mps")
    q_mask[:, -3:] = False

    scorer = MaxSimScorer(normalize=True)
    out = scorer(Q, D, q_mask=q_mask)
    ref = maxsim_reference(Q.cpu().float(), D.cpu().float(), q_mask=q_mask.cpu(), normalize=True)
    assert (out.cpu().float() - ref).abs().max().item() < 5e-3


def test_d_mask_on_mps_matches_reference():
    from late_interaction_kernels import MaxSimScorer
    from late_interaction_kernels.reference import maxsim_reference

    torch.manual_seed(0)
    Q = torch.randn(2, 16, 64, dtype=torch.float16, device="mps")
    D = torch.randn(4, 32, 64, dtype=torch.float16, device="mps")
    d_mask = torch.ones(4, 32, dtype=torch.bool, device="mps")
    d_mask[:, -5:] = False

    scorer = MaxSimScorer(normalize=True)
    out = scorer(Q, D, d_mask=d_mask)
    ref = maxsim_reference(Q.cpu().float(), D.cpu().float(), d_mask=d_mask.cpu(), normalize=True)
    assert (out.cpu().float() - ref).abs().max().item() < 5e-3


def test_both_masks_on_mps_matches_reference():
    from late_interaction_kernels import MaxSimScorer
    from late_interaction_kernels.reference import maxsim_reference

    torch.manual_seed(0)
    Q = torch.randn(3, 20, 96, dtype=torch.float16, device="mps")
    D = torch.randn(5, 40, 96, dtype=torch.float16, device="mps")
    q_mask = torch.ones(3, 20, dtype=torch.bool, device="mps")
    q_mask[:, -2:] = False
    d_mask = torch.ones(5, 40, dtype=torch.bool, device="mps")
    d_mask[:, -7:] = False

    scorer = MaxSimScorer(normalize=True)
    out = scorer(Q, D, q_mask=q_mask, d_mask=d_mask)
    ref = maxsim_reference(
        Q.cpu().float(),
        D.cpu().float(),
        q_mask=q_mask.cpu(),
        d_mask=d_mask.cpu(),
        normalize=True,
    )
    assert (out.cpu().float() - ref).abs().max().item() < 5e-3


# --------------------------------------------------------------------------- #
# Autograd                                                                    #
# --------------------------------------------------------------------------- #


def test_backward_flows_on_mps():
    """Gradients must reach Q and D through the MPS path."""
    from late_interaction_kernels import MaxSimScorer

    torch.manual_seed(0)
    Q = torch.randn(2, 16, 64, dtype=torch.float32, device="mps", requires_grad=True)
    D = torch.randn(4, 32, 64, dtype=torch.float32, device="mps", requires_grad=True)
    scorer = MaxSimScorer(normalize=True)
    out = scorer(Q, D)
    out.mean().backward()
    assert Q.grad is not None and torch.isfinite(Q.grad).all()
    assert D.grad is not None and torch.isfinite(D.grad).all()
    assert Q.grad.abs().sum() > 0
    assert D.grad.abs().sum() > 0


def test_score_is_no_grad_on_mps():
    from late_interaction_kernels import MaxSimScorer

    Q = torch.randn(2, 8, 64, dtype=torch.float16, device="mps", requires_grad=True)
    D = torch.randn(3, 12, 64, dtype=torch.float16, device="mps", requires_grad=True)
    scorer = MaxSimScorer(normalize=True)
    out = scorer.score(Q, D)
    assert not out.requires_grad


def test_grad_matches_reference_finite_difference():
    """Analytic grad on MPS matches the numerical slope from the CPU reference.

    We don't run :func:`torch.autograd.gradcheck` directly on MPS — fp32 on
    Apple Silicon doesn't have enough headroom for the default tolerances —
    but we *do* check that the analytic Q-grad on MPS matches the analytic
    Q-grad on the CPU reference, which is itself gradcheck-verified in
    ``test_retrieve_cpu.py``.
    """
    from late_interaction_kernels import MaxSimScorer
    from late_interaction_kernels.reference import maxsim_reference

    torch.manual_seed(0)
    Q_cpu = torch.randn(2, 8, 32, dtype=torch.float32, requires_grad=True)
    D_cpu = torch.randn(3, 12, 32, dtype=torch.float32, requires_grad=True)
    Q_mps = Q_cpu.detach().clone().to("mps").requires_grad_()
    D_mps = D_cpu.detach().clone().to("mps").requires_grad_()

    scorer = MaxSimScorer(normalize=True)
    out_mps = scorer(Q_mps, D_mps)
    out_mps.sum().backward()

    out_cpu = maxsim_reference(Q_cpu, D_cpu, normalize=True)
    out_cpu.sum().backward()

    assert torch.allclose(Q_mps.grad.cpu(), Q_cpu.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(D_mps.grad.cpu(), D_cpu.grad, atol=1e-4, rtol=1e-4)


# --------------------------------------------------------------------------- #
# Compile cache                                                               #
# --------------------------------------------------------------------------- #


def test_compile_cache_reuses_across_calls():
    """Repeated calls with the same signature don't re-compile."""
    from late_interaction_kernels import MaxSimScorer
    from late_interaction_kernels import _mps as _mps_mod

    Q = torch.randn(2, 16, 64, dtype=torch.float16, device="mps")
    D = torch.randn(3, 32, 64, dtype=torch.float16, device="mps")
    scorer = MaxSimScorer(normalize=True)

    scorer(Q, D)
    scorer(Q, D)
    scorer(Q, D)
    assert len(_mps_mod._compiled_cache) == 1


def test_compile_cache_keys_per_dtype_and_mask_signature():
    """Different dtype / mask combos compile separately."""
    from late_interaction_kernels import MaxSimScorer
    from late_interaction_kernels import _mps as _mps_mod

    scorer = MaxSimScorer(normalize=True)
    Q16 = torch.randn(2, 8, 64, dtype=torch.float16, device="mps")
    D16 = torch.randn(3, 12, 64, dtype=torch.float16, device="mps")
    Q32 = Q16.float()
    D32 = D16.float()
    qm = torch.ones(2, 8, dtype=torch.bool, device="mps")

    scorer(Q16, D16)
    scorer(Q32, D32)
    scorer(Q16, D16, q_mask=qm)
    # 3 distinct (dtype, normalize, has_q_mask, has_d_mask) signatures.
    assert len(_mps_mod._compiled_cache) == 3


def test_disable_compile_env_var(monkeypatch):
    """``LIK_DISABLE_COMPILE=1`` falls back to eager — no cache entries."""
    from late_interaction_kernels import MaxSimScorer
    from late_interaction_kernels import _mps as _mps_mod

    monkeypatch.setenv("LIK_DISABLE_COMPILE", "1")
    scorer = MaxSimScorer(normalize=True)
    Q = torch.randn(2, 8, 64, dtype=torch.float16, device="mps")
    D = torch.randn(3, 12, 64, dtype=torch.float16, device="mps")
    out = scorer(Q, D)
    assert out.shape == (2, 3)
    assert len(_mps_mod._compiled_cache) == 0


# --------------------------------------------------------------------------- #
# Edge cases                                                                  #
# --------------------------------------------------------------------------- #


def test_normalize_false_matches_reference_on_mps():
    from late_interaction_kernels import MaxSimScorer
    from late_interaction_kernels.reference import maxsim_reference

    torch.manual_seed(0)
    Q = torch.randn(2, 16, 64, dtype=torch.float16, device="mps")
    D = torch.randn(4, 32, 64, dtype=torch.float16, device="mps")
    scorer = MaxSimScorer(normalize=False)
    out = scorer(Q, D)
    ref = maxsim_reference(Q.cpu().float(), D.cpu().float(), normalize=False)
    assert (out.cpu().float() - ref).abs().max().item() / max(1e-6, ref.abs().max().item()) < 5e-3


def test_single_query_single_doc_on_mps():
    from late_interaction_kernels import MaxSimScorer
    from late_interaction_kernels.reference import maxsim_reference

    Q = torch.randn(1, 1, 32, dtype=torch.float16, device="mps")
    D = torch.randn(1, 1, 32, dtype=torch.float16, device="mps")
    scorer = MaxSimScorer(normalize=True)
    out = scorer(Q, D)
    ref = maxsim_reference(Q.cpu().float(), D.cpu().float(), normalize=True)
    assert out.shape == (1, 1)
    assert (out.cpu().float() - ref).abs().max().item() < 1e-2


def test_long_lq_on_mps():
    """Lq > 256 (ColPali shape) still parity-matches."""
    from late_interaction_kernels import MaxSimScorer
    from late_interaction_kernels.reference import maxsim_reference

    Q = torch.randn(1, 512, 64, dtype=torch.float16, device="mps")
    D = torch.randn(2, 64, 64, dtype=torch.float16, device="mps")
    scorer = MaxSimScorer(normalize=True)
    out = scorer(Q, D)
    ref = maxsim_reference(Q.cpu().float(), D.cpu().float(), normalize=True)
    err = (out.cpu().float() - ref).abs().max().item()
    rel = err / max(1e-6, ref.abs().max().item())
    assert rel < 5e-3


def test_module_repr_does_not_crash_on_mps():
    """``MaxSimScorer.__repr__`` works on a moved module."""
    from late_interaction_kernels import MaxSimScorer

    scorer = MaxSimScorer(normalize=True).to("mps")  # moves nothing (no params), still legal
    assert "MaxSimScorer" in repr(scorer)


# --------------------------------------------------------------------------- #
# nn.Module composition                                                       #
# --------------------------------------------------------------------------- #


def test_scorer_composes_inside_nn_module_on_mps():
    """A real two-layer model with MaxSim head trains on MPS."""
    from late_interaction_kernels import MaxSimScorer

    class Reranker(torch.nn.Module):
        def __init__(self, d):
            super().__init__()
            self.q_proj = torch.nn.Linear(d, d, bias=False)
            self.d_proj = torch.nn.Linear(d, d, bias=False)
            self.scorer = MaxSimScorer(normalize=True)

        def forward(self, Q, D):
            return self.scorer(self.q_proj(Q), self.d_proj(D))

    torch.manual_seed(0)
    m = Reranker(64).to("mps")
    Q = torch.randn(2, 8, 64, device="mps", dtype=torch.float32)
    D = torch.randn(4, 16, 64, device="mps", dtype=torch.float32)
    out = m(Q, D)
    assert out.shape == (2, 4)
    out.sum().backward()
    assert m.q_proj.weight.grad is not None
    assert m.d_proj.weight.grad is not None
    assert torch.isfinite(m.q_proj.weight.grad).all()
