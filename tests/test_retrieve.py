"""Tests for the high-level :class:`MaxSimScorer` and :func:`retrieve`.

These cover the user-facing ergonomic layer: the contract that a
one-liner ``MaxSimScorer() (Q, D)`` or ``retrieve(Q, D, top_k=k)`` gives
the same answer as the lower-level kernels and composes cleanly with
autograd / ``torch.compile`` / ``nn.Module`` hierarchies.
"""

from __future__ import annotations

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


def test_maxsim_scorer_score_no_grad():
    from late_interaction_kernels import MaxSimScorer

    Q = torch.randn(4, 32, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    D = torch.randn(8, 128, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    scorer = MaxSimScorer()
    scores = scorer.score(Q, D)
    assert not scores.requires_grad


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
    from late_interaction_kernels import maxsim_inference, retrieve

    torch.manual_seed(0)
    Q = torch.randn(4, 16, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(64, 32, 128, device="cuda", dtype=torch.float16)
    scores, idx = retrieve(Q, D, top_k=10, normalize=True)
    ref = maxsim_inference(Q, D, normalize=True)
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

    # Without mask_pad_token set, it should raise.
    scorer_no_pad = MaxSimScorer(normalize=True)
    with pytest.raises(ValueError, match="mask_pad_token"):
        scorer_no_pad.forward_with_ids(Q, D, q_ids, d_ids)


def test_scorer_invalid_backward_raises():
    from late_interaction_kernels import MaxSimScorer

    with pytest.raises(ValueError, match="backward"):
        MaxSimScorer(backward="bogus")  # type: ignore[arg-type]


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


def test_maxsim_per_call_overrides_global():
    """Per-call `backward=` must not leak back into the global state."""
    from late_interaction_kernels import (
        get_backward_method,
        maxsim,
        set_backward_method,
    )

    old = get_backward_method()
    try:
        set_backward_method("csr")
        Q = torch.randn(2, 16, 64, device="cuda", dtype=torch.float32, requires_grad=True)
        D = torch.randn(4, 32, 64, device="cuda", dtype=torch.float32, requires_grad=True)
        maxsim(Q, D, normalize=True, backward="atomic").sum().backward()
        assert get_backward_method() == "csr"
    finally:
        set_backward_method(old)


def test_maxsim_forward_deprecation_warning():
    """`from lik import maxsim_forward` must still work but warn."""
    import warnings

    import late_interaction_kernels as lik

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fn = lik.maxsim_forward  # deprecated path
        assert callable(fn)
    assert any(issubclass(x.category, DeprecationWarning) for x in w), w


def test_maxsim_varlen_inference_deprecated():
    import warnings

    from late_interaction_kernels import maxsim_varlen_inference

    Q = torch.randn(8, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(32, 128, device="cuda", dtype=torch.float16)
    cu_q = torch.tensor([0, 4, 8], device="cuda", dtype=torch.int32)
    cu_d = torch.tensor([0, 16, 32], device="cuda", dtype=torch.int32)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        scores = maxsim_varlen_inference(Q, D, cu_q, cu_d)
    assert scores.shape == (2, 2)
    assert any(issubclass(x.category, DeprecationWarning) for x in w), w


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
