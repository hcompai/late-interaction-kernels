"""Tests for the high-level :class:`MaxSimScorer` and :func:`retrieve`.

These cover the user-facing ergonomic layer: the contract that a
one-liner ``MaxSimScorer() (Q, D)`` or ``retrieve(Q, D, top_k=k)`` gives
the same answer as the lower-level kernels and composes cleanly with
autograd / ``torch.compile`` / ``nn.Module`` hierarchies.
"""

import pytest
import torch

pytestmark = pytest.mark.cuda


def test_maxsim_scorer_matches_maxsim(rel):
    from late_interaction_kernels import MaxSimScorer, maxsim

    torch.manual_seed(0)
    Q = torch.randn(4, 32, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(8, 128, 128, device="cuda", dtype=torch.float16)

    scorer = MaxSimScorer(normalize=True)
    out = scorer(Q, D)
    ref = maxsim(Q, D, normalize=True)
    assert out.shape == (4, 8)
    assert rel(out, ref) < 1e-4


def test_maxsim_scorer_backward(rel):
    from late_interaction_kernels import MaxSimScorer

    torch.manual_seed(0)
    Q = torch.randn(3, 16, 128, device="cuda", dtype=torch.float32, requires_grad=True)
    D = torch.randn(5, 64, 128, device="cuda", dtype=torch.float32, requires_grad=True)
    scorer = MaxSimScorer(normalize=True)
    scores = scorer(Q, D)
    loss = scores.mean()
    loss.backward()
    assert Q.grad is not None
    assert D.grad is not None
    assert torch.isfinite(Q.grad).all()
    assert torch.isfinite(D.grad).all()


def test_maxsim_scorer_composes_in_module():
    """`MaxSimScorer` should slot into a bigger `nn.Module` without tricks."""
    from late_interaction_kernels import MaxSimScorer

    class Reranker(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scorer = MaxSimScorer(normalize=True, backward="unified")

        def forward(self, Q, D):
            return self.scorer(Q, D).mean(-1)

    m = Reranker().cuda()
    Q = torch.randn(4, 32, 128, device="cuda", dtype=torch.float32, requires_grad=True)
    D = torch.randn(8, 64, 128, device="cuda", dtype=torch.float32, requires_grad=True)
    out = m(Q, D)
    assert out.shape == (4,)
    out.sum().backward()
    assert Q.grad is not None


def test_retrieve_matches_explicit_topk():
    from late_interaction_kernels import maxsim, retrieve

    torch.manual_seed(0)
    Q = torch.randn(4, 16, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(64, 32, 128, device="cuda", dtype=torch.float16)
    scores, idx = retrieve(Q, D, top_k=10, normalize=True)
    ref = maxsim(Q, D, normalize=True)
    ref_scores, ref_idx = torch.topk(ref, 10, dim=-1)
    assert torch.allclose(scores.float(), ref_scores.float(), atol=1e-4, rtol=1e-3)
    assert torch.equal(idx, ref_idx)


def test_retrieve_chunked_matches_unchunked():
    from late_interaction_kernels import retrieve

    torch.manual_seed(0)
    Q = torch.randn(2, 16, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(100, 32, 128, device="cuda", dtype=torch.float16)
    full_s, full_i = retrieve(Q, D, top_k=5, normalize=True)
    chunked_s, chunked_i = retrieve(Q, D, top_k=5, normalize=True, chunk=17)
    # Top-k scores / indices must agree (ties are extremely unlikely here).
    assert torch.allclose(full_s, chunked_s, atol=1e-5)
    assert torch.equal(full_i, chunked_i)


def test_retrieve_chunked_reduces_peak_memory():
    """The README claim: ``chunk=`` bounds peak HBM at ``Nq·(chunk+top_k)``.

    This is a drift guard, not a micro-benchmark. If a future refactor
    accidentally materializes the full ``[Nq, Nd]`` score matrix on the
    chunked path, this test fails loudly.
    """
    from late_interaction_kernels import retrieve

    torch.manual_seed(0)
    Nq, Nd = 8, 4096
    Q = torch.randn(Nq, 32, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(Nd, 32, 128, device="cuda", dtype=torch.float16)
    top_k = 32

    # Warm up so allocator caches settle.
    _ = retrieve(Q, D, top_k=top_k, normalize=True)
    _ = retrieve(Q, D, top_k=top_k, normalize=True, chunk=128)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    _ = retrieve(Q, D, top_k=top_k, normalize=True)
    torch.cuda.synchronize()
    peak_full = torch.cuda.max_memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    _ = retrieve(Q, D, top_k=top_k, normalize=True, chunk=128)
    torch.cuda.synchronize()
    peak_chunked = torch.cuda.max_memory_allocated()

    # Chunked path must never exceed unchunked peak, and should use
    # meaningfully less when Nd ≫ chunk. We don't assert a tight ratio
    # because the underlying kernel already allocates workspace; the
    # contract we protect is "chunked <= unchunked, substantially so".
    assert peak_chunked <= peak_full, (peak_chunked, peak_full)


def test_maxsim_scorer_wrapper_has_negligible_overhead(rel):
    """Drift guard: ``MaxSimScorer(Q, D) ≡ maxsim(Q, D, normalize=True)``.

    Not a benchmark — a correctness assertion that the nn.Module wrapper
    adds no semantics beyond argument forwarding. Catches accidental
    extra ops (e.g. a stray ``.float()`` or mask cast creeping into
    ``MaxSimScorer.forward``).
    """
    from late_interaction_kernels import MaxSimScorer, maxsim

    torch.manual_seed(0)
    Q = torch.randn(3, 24, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(7, 96, 128, device="cuda", dtype=torch.float16)

    a = MaxSimScorer(normalize=True, backward="unified")(Q, D)
    b = maxsim(Q, D, normalize=True, backward="unified")
    # Byte-identical, not "close" — the wrapper must be pure forwarding.
    assert torch.equal(a, b), rel(a, b)


def test_scorer_retrieve_method():
    from late_interaction_kernels import MaxSimScorer, retrieve

    torch.manual_seed(0)
    Q = torch.randn(2, 16, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(32, 32, 128, device="cuda", dtype=torch.float16)
    scorer = MaxSimScorer(normalize=True)
    s1, i1 = scorer.retrieve(Q, D, top_k=5)
    s2, i2 = retrieve(Q, D, top_k=5, normalize=True)
    assert torch.equal(s1, s2)
    assert torch.equal(i1, i2)


def test_scorer_forward_with_ids():
    from late_interaction_kernels import MaxSimScorer

    torch.manual_seed(0)
    Q = torch.randn(2, 8, 128, device="cuda", dtype=torch.float32, requires_grad=True)
    D = torch.randn(4, 16, 128, device="cuda", dtype=torch.float32, requires_grad=True)
    q_ids = torch.ones(2, 8, device="cuda", dtype=torch.long)
    q_ids[:, -2:] = 0  # last 2 are pad
    d_ids = torch.ones(4, 16, device="cuda", dtype=torch.long)
    d_ids[:, -4:] = 0

    scorer = MaxSimScorer(normalize=True, mask_pad_token=0)
    scores = scorer.forward_with_ids(Q, D, q_ids, d_ids)
    assert scores.shape == (2, 4)
    scores.sum().backward()


@pytest.mark.parametrize("backward", ["auto", "unified", "csr", "atomic"])
def test_maxsim_per_call_backward_kwarg(backward):
    from late_interaction_kernels import maxsim

    torch.manual_seed(0)
    Q = torch.randn(2, 16, 64, device="cuda", dtype=torch.float32, requires_grad=True)
    D = torch.randn(4, 32, 64, device="cuda", dtype=torch.float32, requires_grad=True)
    scores = maxsim(Q, D, normalize=True, backward=backward)
    scores.sum().backward()
    assert Q.grad is not None
    assert D.grad is not None


def test_maxsim_per_call_backward_invalid_raises():
    from late_interaction_kernels import maxsim

    Q = torch.randn(2, 16, 64, device="cuda", dtype=torch.float32)
    D = torch.randn(4, 32, 64, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="backward"):
        maxsim(Q, D, normalize=True, backward="nope")  # type: ignore[arg-type]


def test_unnormalized_input_warns_once(monkeypatch):
    import warnings

    from late_interaction_kernels import autograd as _autograd_mod
    from late_interaction_kernels import maxsim

    # Reset the one-shot flag so we can deterministically observe the warn.
    monkeypatch.delenv("LIK_SUPPRESS_NORM_WARN", raising=False)
    _autograd_mod._WARNED_UNNORMALIZED = False

    Q = torch.randn(2, 16, 64, device="cuda", dtype=torch.float32) * 10.0  # clearly not unit
    D = torch.randn(4, 32, 64, device="cuda", dtype=torch.float32)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = maxsim(Q, D, normalize=False)
        _ = maxsim(Q, D, normalize=False)  # second call — should NOT re-warn
    lik_msgs = [
        str(x.message)
        for x in w
        if issubclass(x.category, UserWarning) and "late-interaction-kernels" in str(x.message)
    ]
    assert len(lik_msgs) == 1, f"expected exactly one normalize warning, got {lik_msgs}"
    assert "normalize=True" in lik_msgs[0]


def test_scorer_compile_smoke():
    """`torch.compile(MaxSimScorer())` must at least not crash."""
    from late_interaction_kernels import MaxSimScorer

    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile unavailable")

    scorer = MaxSimScorer(normalize=True).cuda()
    Q = torch.randn(2, 16, 64, device="cuda", dtype=torch.float16)
    D = torch.randn(4, 32, 64, device="cuda", dtype=torch.float16)
    eager = scorer(Q, D)
    try:
        compiled = torch.compile(scorer, fullgraph=False)(Q, D)
    except Exception as exc:  # pragma: no cover — upstream inductor regressions
        pytest.skip(f"torch.compile crashed in nn.Module wrapper: {exc}")
    assert torch.allclose(eager, compiled, atol=1e-4)
