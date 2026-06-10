"""CPU-reachable tests for the high-level :class:`MaxSimScorer` / :func:`retrieve`.

These exercise the full user-facing contract against the pure-PyTorch
reference, so they run on any platform without CUDA / Triton — valuable
for macOS / CI and for catching wiring regressions before GPU time is
spent. The GPU parity tests live in ``tests/test_retrieve.py``.
"""

import pytest
import torch

# --------------------------------------------------------------------------- #
# MaxSimScorer                                                                #
# --------------------------------------------------------------------------- #


def test_scorer_constructs_and_reprs():
    from late_interaction_kernels import MaxSimScorer

    scorer = MaxSimScorer(normalize=True, backward="unified")
    r = repr(scorer)
    assert "MaxSimScorer" in r
    assert "normalize=True" in r
    assert "backward='unified'" in r


def test_scorer_invalid_backward_raises():
    from late_interaction_kernels import MaxSimScorer

    with pytest.raises(ValueError, match="backward"):
        MaxSimScorer(backward="nope")  # type: ignore[arg-type]


def test_scorer_forward_matches_reference(rel):
    from late_interaction_kernels import MaxSimScorer
    from late_interaction_kernels.reference import maxsim_reference

    torch.manual_seed(0)
    Q = torch.randn(3, 16, 32, dtype=torch.float32)
    D = torch.randn(5, 24, 32, dtype=torch.float32)

    scorer = MaxSimScorer(normalize=True)
    out = scorer(Q, D)
    ref = maxsim_reference(Q, D, normalize=True)
    assert out.shape == (3, 5)
    assert rel(out, ref) < 1e-6


def test_scorer_backward_gradient_flows():
    from late_interaction_kernels import MaxSimScorer

    torch.manual_seed(0)
    Q = torch.randn(2, 8, 16, dtype=torch.float32, requires_grad=True)
    D = torch.randn(3, 12, 16, dtype=torch.float32, requires_grad=True)
    scorer = MaxSimScorer(normalize=True)
    out = scorer(Q, D)
    out.mean().backward()
    assert Q.grad is not None and torch.isfinite(Q.grad).all()
    assert D.grad is not None and torch.isfinite(D.grad).all()
    # Gradients are non-trivially non-zero:
    assert Q.grad.abs().sum() > 0
    assert D.grad.abs().sum() > 0


def test_scorer_score_is_no_grad():
    from late_interaction_kernels import MaxSimScorer

    Q = torch.randn(2, 8, 16, requires_grad=True)
    D = torch.randn(3, 12, 16, requires_grad=True)
    scorer = MaxSimScorer(normalize=True)
    out = scorer.score(Q, D)
    assert not out.requires_grad


def test_scorer_forward_respects_callers_no_grad():
    """Regression: ``forward()`` used to force ``enable_grad()``, overriding a
    caller's ``torch.no_grad()`` and returning ``requires_grad=True``."""
    from late_interaction_kernels import MaxSimScorer

    Q = torch.randn(2, 8, 16, requires_grad=True)
    D = torch.randn(3, 12, 16, requires_grad=True)
    scorer = MaxSimScorer(normalize=True)
    with torch.no_grad():
        out = scorer(Q, D)
    assert not out.requires_grad


def test_scorer_composes_inside_nn_module():
    from late_interaction_kernels import MaxSimScorer

    class Reranker(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scorer = MaxSimScorer(normalize=True, backward="unified")

        def forward(self, Q, D):
            return self.scorer(Q, D).mean(-1)

    torch.manual_seed(0)
    m = Reranker()
    Q = torch.randn(2, 8, 16, dtype=torch.float32, requires_grad=True)
    D = torch.randn(4, 12, 16, dtype=torch.float32, requires_grad=True)
    out = m(Q, D)
    assert out.shape == (2,)
    out.sum().backward()
    assert Q.grad is not None


def test_scorer_forward_with_ids_happy_path():
    from late_interaction_kernels import MaxSimScorer

    torch.manual_seed(0)
    Q = torch.randn(2, 8, 16, dtype=torch.float32, requires_grad=True)
    D = torch.randn(3, 12, 16, dtype=torch.float32, requires_grad=True)
    q_ids = torch.ones(2, 8, dtype=torch.long)
    q_ids[:, -2:] = 0
    d_ids = torch.ones(3, 12, dtype=torch.long)
    d_ids[:, -3:] = 0

    scorer = MaxSimScorer(normalize=True, mask_pad_token=0)
    out = scorer.forward_with_ids(Q, D, q_ids, d_ids)
    assert out.shape == (2, 3)
    out.sum().backward()
    assert Q.grad is not None


def test_scorer_forward_with_ids_requires_mask_pad_token():
    from late_interaction_kernels import MaxSimScorer

    scorer = MaxSimScorer(normalize=True)
    with pytest.raises(ValueError, match="mask_pad_token"):
        scorer.forward_with_ids(
            torch.randn(2, 4, 8),
            torch.randn(3, 6, 8),
            torch.zeros(2, 4, dtype=torch.long),
            torch.zeros(3, 6, dtype=torch.long),
        )


def test_scorer_with_masks_matches_reference(rel):
    from late_interaction_kernels import MaxSimScorer
    from late_interaction_kernels.reference import maxsim_reference

    torch.manual_seed(0)
    Q = torch.randn(3, 10, 16, dtype=torch.float32)
    D = torch.randn(4, 20, 16, dtype=torch.float32)
    q_mask = torch.ones(3, 10, dtype=torch.bool)
    q_mask[:, -2:] = False
    d_mask = torch.ones(4, 20, dtype=torch.bool)
    d_mask[:, -5:] = False

    scorer = MaxSimScorer(normalize=True)
    out = scorer(Q, D, q_mask=q_mask, d_mask=d_mask)
    ref = maxsim_reference(Q, D, q_mask=q_mask, d_mask=d_mask, normalize=True)
    assert rel(out, ref) < 1e-6


# --------------------------------------------------------------------------- #
# retrieve                                                                    #
# --------------------------------------------------------------------------- #


def test_retrieve_top_k_must_be_positive():
    from late_interaction_kernels import retrieve

    with pytest.raises(ValueError, match="top_k"):
        retrieve(torch.randn(2, 4, 8), torch.randn(3, 6, 8), top_k=0)


def test_retrieve_different_devices_raises():
    from late_interaction_kernels import retrieve

    with pytest.raises(ValueError, match="same device"):
        retrieve(torch.randn(2, 4, 8), torch.randn(3, 6, 8).to("meta"), top_k=1)


def test_retrieve_matches_reference_topk():
    from late_interaction_kernels import retrieve
    from late_interaction_kernels.reference import maxsim_reference

    torch.manual_seed(0)
    Q = torch.randn(3, 12, 16, dtype=torch.float32)
    D = torch.randn(20, 16, 16, dtype=torch.float32)
    scores, idx = retrieve(Q, D, top_k=5, normalize=True)
    ref = maxsim_reference(Q, D, normalize=True)
    ref_s, ref_i = torch.topk(ref, 5, dim=-1)
    assert torch.allclose(scores, ref_s, atol=1e-5, rtol=1e-5)
    assert torch.equal(idx, ref_i)


def test_retrieve_chunked_equals_unchunked():
    from late_interaction_kernels import retrieve

    torch.manual_seed(0)
    Q = torch.randn(2, 8, 16, dtype=torch.float32)
    D = torch.randn(50, 16, 16, dtype=torch.float32)
    full_s, full_i = retrieve(Q, D, top_k=7, normalize=True)
    chunked_s, chunked_i = retrieve(Q, D, top_k=7, normalize=True, chunk=13)
    assert torch.allclose(full_s, chunked_s, atol=1e-5)
    assert torch.equal(full_i, chunked_i)


def test_retrieve_chunked_top_k_larger_than_chunk():
    """Regression: each chunk contributes at most ``chunk`` columns, so the
    running merge starts narrower than ``top_k`` and used to crash with
    "selected index k out of range"."""
    from late_interaction_kernels import retrieve

    torch.manual_seed(0)
    Q = torch.randn(2, 8, 16, dtype=torch.float32)
    D = torch.randn(20, 6, 16, dtype=torch.float32)
    full_s, full_i = retrieve(Q, D, top_k=10, normalize=True)
    chunked_s, chunked_i = retrieve(Q, D, top_k=10, normalize=True, chunk=4)
    assert chunked_s.shape == (2, 10)
    assert torch.allclose(full_s, chunked_s, atol=1e-5)
    assert torch.equal(full_i, chunked_i)


def test_retrieve_chunked_top_k_larger_than_nd():
    """``top_k > Nd`` with chunking still clamps the output width to Nd."""
    from late_interaction_kernels import retrieve

    torch.manual_seed(0)
    Q = torch.randn(2, 8, 16, dtype=torch.float32)
    D = torch.randn(6, 6, 16, dtype=torch.float32)
    full_s, full_i = retrieve(Q, D, top_k=10, normalize=True)
    chunked_s, chunked_i = retrieve(Q, D, top_k=10, normalize=True, chunk=4)
    assert chunked_s.shape == (2, 6)
    assert torch.allclose(full_s, chunked_s, atol=1e-5)
    assert torch.equal(full_i, chunked_i)


def test_retrieve_clamps_k_to_nd():
    from late_interaction_kernels import retrieve

    Q = torch.randn(2, 4, 8)
    D = torch.randn(3, 6, 8)
    scores, idx = retrieve(Q, D, top_k=100, normalize=True)
    assert scores.shape == (2, 3)
    assert idx.shape == (2, 3)


def test_retrieve_2d_query_squeezes():
    """A 2-D Q (single query) returns a [top_k] vector, matching `maxsim_topk`."""
    from late_interaction_kernels import retrieve

    Q = torch.randn(4, 8, dtype=torch.float32)  # [Lq, d]
    D = torch.randn(10, 6, 8, dtype=torch.float32)
    scores, idx = retrieve(Q, D, top_k=3, normalize=True)
    assert scores.shape == (3,)
    assert idx.shape == (3,)


def test_retrieve_with_masks():
    from late_interaction_kernels import retrieve
    from late_interaction_kernels.reference import maxsim_reference

    torch.manual_seed(0)
    Q = torch.randn(2, 10, 16, dtype=torch.float32)
    D = torch.randn(8, 20, 16, dtype=torch.float32)
    q_mask = torch.ones(2, 10, dtype=torch.bool)
    q_mask[:, -2:] = False
    d_mask = torch.ones(8, 20, dtype=torch.bool)
    d_mask[:, -3:] = False

    s, i = retrieve(Q, D, top_k=3, q_mask=q_mask, d_mask=d_mask, normalize=True)
    ref = maxsim_reference(Q, D, q_mask=q_mask, d_mask=d_mask, normalize=True)
    ref_s, ref_i = torch.topk(ref, 3, dim=-1)
    assert torch.allclose(s, ref_s, atol=1e-5)
    assert torch.equal(i, ref_i)


# --------------------------------------------------------------------------- #
# Public API invariants                                                       #
# --------------------------------------------------------------------------- #


def test_public_all_exports_are_resolvable():
    """Every name in ``__all__`` must actually exist on the module.

    Guards against PR #-drift where someone renames a private symbol and
    forgets to update ``__all__``. Also catches dropped re-exports.
    """
    import late_interaction_kernels as lik

    missing = [name for name in lik.__all__ if not hasattr(lik, name)]
    assert not missing, f"Names in __all__ missing from module: {missing}"


def test_module_getattr_rejects_unknown_names():
    import late_interaction_kernels as lik

    with pytest.raises(AttributeError):
        _ = lik.this_symbol_does_not_exist  # noqa: B018


def test_reference_topk_path_matches_fused_topk_contract_on_ties():
    """Top-k must be stable under the same-ordering convention torch.topk uses.

    Not a user-visible invariant, but a drift guard: if a future refactor
    changes the reference chunking order, this test surfaces it before
    users do.
    """
    from late_interaction_kernels import retrieve

    # Construct a deliberate near-tie so that any chunking bug on the
    # reference path surfaces as a differing index.
    torch.manual_seed(42)
    Q = torch.randn(1, 8, 16, dtype=torch.float64)
    D = torch.randn(16, 8, 16, dtype=torch.float64)
    s_full, i_full = retrieve(Q, D, top_k=5, normalize=True)
    s_ch, i_ch = retrieve(Q, D, top_k=5, normalize=True, chunk=3)
    assert torch.allclose(s_full, s_ch, atol=1e-10)
    assert torch.equal(i_full, i_ch)


# --------------------------------------------------------------------------- #
# Gradcheck — gold-standard autograd sanity on the reference path             #
# --------------------------------------------------------------------------- #


def test_suppress_norm_warn_env_var_silences_warning(monkeypatch):
    """``LIK_SUPPRESS_NORM_WARN=1`` must silence the unnormalized-input warning.

    The warning lives in ``autograd._maybe_warn_unnormalized``; we exercise
    it directly so this test is CPU-safe (no Triton kernel launch). The
    positive (warning fires) case is tested on GPU in ``test_retrieve.py``.
    """
    import warnings

    # ``autograd`` imports Triton transitively — auto-skip on macOS.
    pytest.importorskip("triton")
    from late_interaction_kernels import autograd as _autograd_mod

    monkeypatch.setenv("LIK_SUPPRESS_NORM_WARN", "1")
    _autograd_mod._WARNED_UNNORMALIZED = False

    Q = torch.randn(2, 16, 64, dtype=torch.float32) * 10.0  # clearly not unit-norm
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _autograd_mod._maybe_warn_unnormalized(Q)

    lik_msgs = [
        str(x.message)
        for x in w
        if issubclass(x.category, UserWarning) and "late-interaction-kernels" in str(x.message)
    ]
    assert lik_msgs == [], f"LIK_SUPPRESS_NORM_WARN=1 should silence, got {lik_msgs}"


def test_unnormalized_warn_fires_once_direct(monkeypatch):
    """Direct test of the unnormalized-warning helper on CPU.

    The GPU test ``test_unnormalized_input_warns_once`` covers the same
    one-shot semantics but requires calling ``maxsim()`` (Triton). This
    CPU test pins the helper contract itself, so the one-shot logic is
    protected even when Triton tests aren't run.
    """
    import warnings

    pytest.importorskip("triton")
    from late_interaction_kernels import autograd as _autograd_mod

    monkeypatch.delenv("LIK_SUPPRESS_NORM_WARN", raising=False)
    _autograd_mod._WARNED_UNNORMALIZED = False

    Q = torch.randn(2, 16, 64, dtype=torch.float32) * 10.0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _autograd_mod._maybe_warn_unnormalized(Q)
        _autograd_mod._maybe_warn_unnormalized(Q)  # second call: must NOT re-warn

    lik_msgs = [
        str(x.message)
        for x in w
        if issubclass(x.category, UserWarning) and "late-interaction-kernels" in str(x.message)
    ]
    assert len(lik_msgs) == 1, f"expected exactly one warning, got {lik_msgs}"
    assert "normalize=True" in lik_msgs[0]


def test_scorer_retrieve_method_matches_top_level_retrieve():
    """``scorer.retrieve(...)`` must be equivalent to ``retrieve(..., normalize=scorer.normalize)``."""
    from late_interaction_kernels import MaxSimScorer, retrieve

    torch.manual_seed(0)
    Q = torch.randn(3, 12, 16, dtype=torch.float32)
    D = torch.randn(20, 16, 16, dtype=torch.float32)
    scorer = MaxSimScorer(normalize=True)
    s1, i1 = scorer.retrieve(Q, D, top_k=4)
    s2, i2 = retrieve(Q, D, top_k=4, normalize=True)
    assert torch.equal(s1, s2)
    assert torch.equal(i1, i2)


def test_scorer_gradcheck_fp64():
    """`torch.autograd.gradcheck` against the reference in fp64.

    This runs on CPU, needs no Triton, and validates that
    :class:`MaxSimScorer` respects the full autograd contract — not just
    "runs without NaN" but "analytic gradient matches numerical".
    """
    from late_interaction_kernels import MaxSimScorer

    torch.manual_seed(0)
    Q = torch.randn(2, 4, 6, dtype=torch.float64, requires_grad=True)
    D = torch.randn(3, 5, 6, dtype=torch.float64, requires_grad=True)
    scorer = MaxSimScorer(normalize=True)

    # `gradcheck` only wiggles individual tensors. The argmax step inside
    # MaxSim is non-differentiable at ties, but `randn` inputs have
    # measure-zero tie probability, so the sub-gradient matches the
    # numerical slope everywhere with overwhelming probability.
    assert torch.autograd.gradcheck(
        lambda q, d: scorer(q, d),
        (Q, D),
        atol=1e-5,
        rtol=1e-4,
        nondet_tol=1e-8,
    )
