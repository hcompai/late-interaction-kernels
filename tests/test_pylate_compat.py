"""End-to-end PyLate monkey-patch test.

Targets PyLate >= 1.3, which uses the
``colbert_scores(Q, D, queries_mask=None, documents_mask=None)``
signature. Older PyLate (1.2.x, single ``mask=`` kwarg) is not supported.
"""

import inspect

import pytest
import torch

pytestmark = pytest.mark.cuda

pylate = pytest.importorskip("pylate")

# Fail fast (with a clear message) if a pinned-old PyLate got in.
_params = inspect.signature(
    __import__("pylate.scores", fromlist=["colbert_scores"]).colbert_scores
).parameters
if "documents_mask" not in _params or "queries_mask" not in _params:
    pytest.skip(
        f"Installed PyLate predates the (queries_mask, documents_mask) signature "
        f"required by late-interaction-kernels. Got params: "
        f"{list(_params)}. Upgrade with `pip install -U 'pylate>=1.3.3'`.",
        allow_module_level=True,
    )


def _l2(x):
    return torch.nn.functional.normalize(x, p=2, dim=-1)


def _make_qd(Nq=4, Nd=8, Lq=32, Ld=128, d=128):
    Q = _l2(torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float32))
    D = _l2(torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float32))
    return Q, D


def test_patched_matches_pylate_reference_no_mask():
    """Bitwise close to PyLate's own implementation when no mask is used."""
    from pylate.scores import colbert_scores as original_fn

    from late_interaction_kernels.pylate_compat import patch_pylate, unpatch_pylate

    Q, D = _make_qd()
    ref = original_fn(Q, D)

    patch_pylate()
    try:
        from pylate.scores import colbert_scores as patched_fn

        out = patched_fn(Q, D)
        err = (out.float() - ref.float()).abs().max().item()
        denom = max(1.0, ref.abs().max().item())
        assert err / denom < 5e-3, f"err={err}, denom={denom}"
    finally:
        unpatch_pylate()


def test_patched_with_documents_mask():
    """With only documents_mask supplied (the `rerank` + `Contrastive` call
    site when `do_query_expansion=True`), match our reference. PyLate's
    mask-by-multiplication and our -inf mask agree on L2-normalized inputs
    because real sims are ≥ -1 and the padded region contributes 0 to both."""
    from late_interaction_kernels.pylate_compat import patch_pylate, unpatch_pylate
    from late_interaction_kernels.reference import maxsim_reference

    Q, D = _make_qd()
    d_mask = (torch.rand(D.shape[0], D.shape[1], device="cuda") > 0.2).float()
    d_mask[:, 0] = 1  # at least one real token per doc
    ref = maxsim_reference(Q, D, d_mask=d_mask.bool()).float()

    patch_pylate()
    try:
        from pylate.scores import colbert_scores as patched_fn

        out = patched_fn(Q, D, documents_mask=d_mask).float()
        err = (out - ref).abs().max().item()
        denom = max(1.0, ref.abs().max().item())
        assert err / denom < 5e-3, f"err={err}, denom={denom}"
    finally:
        unpatch_pylate()


def test_patched_with_both_masks_matches_pylate():
    """The `Contrastive` / `CachedContrastive` call path passes *both*
    masks. Compare directly against PyLate's own implementation
    (mask-by-multiplication) instead of our reference, because with
    `queries_mask * documents_mask` multiplied in before the max, the
    two semantics agree exactly on L2-normalized inputs whenever at
    least one token of each doc / query is unmasked."""
    from pylate.scores import colbert_scores as original_fn

    from late_interaction_kernels.pylate_compat import patch_pylate, unpatch_pylate

    Q, D = _make_qd()
    q_mask = (torch.rand(Q.shape[0], Q.shape[1], device="cuda") > 0.2).float()
    q_mask[:, 0] = 1
    d_mask = (torch.rand(D.shape[0], D.shape[1], device="cuda") > 0.2).float()
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


def test_patched_kd_matches_pylate():
    """Knowledge-distillation scoring, [Nq, Nd, Ld, d] documents."""
    from pylate.scores import colbert_kd_scores as original_fn

    from late_interaction_kernels.pylate_compat import patch_pylate, unpatch_pylate

    Nq, Nd, Lq, Ld, d = 4, 3, 32, 96, 128
    Q = _l2(torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float32))
    Dkd = _l2(torch.randn(Nq, Nd, Ld, d, device="cuda", dtype=torch.float32))
    q_mask = (torch.rand(Nq, Lq, device="cuda") > 0.2).float()
    q_mask[:, 0] = 1
    d_mask = (torch.rand(Nq, Nd, Ld, device="cuda") > 0.2).float()
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


def test_patched_falls_back_on_cpu():
    """CPU inputs should silently fall through to PyLate's own impl."""
    from pylate.scores import colbert_scores as original_fn

    from late_interaction_kernels.pylate_compat import patch_pylate, unpatch_pylate

    Nq, Nd, Lq, Ld, d = 2, 3, 16, 32, 64
    Q = _l2(torch.randn(Nq, Lq, d, dtype=torch.float32))  # CPU
    D = _l2(torch.randn(Nd, Ld, d, dtype=torch.float32))

    ref = original_fn(Q, D)

    patch_pylate()
    try:
        from pylate.scores import colbert_scores as patched_fn

        out = patched_fn(Q, D)
        assert torch.allclose(out, ref, atol=1e-5)
    finally:
        unpatch_pylate()


def test_contrastive_loss_uses_patched_scores():
    """Verify `patch_pylate()` reaches the path `Contrastive` scores through.

    PyLate <= 1.4 losses capture `colbert_scores` at import time; 1.5 losses
    route through `ColBERTScores`, which resolves the symbol from the defining
    module's globals at call time. Assert whichever reference this version uses.
    """
    from late_interaction_kernels.pylate_compat import (
        _scores_defining_module,
        patch_pylate,
        patched_colbert_scores,
        unpatch_pylate,
    )

    patch_pylate()
    try:
        import pylate.losses.contrastive as c

        defining, new_layout = _scores_defining_module()
        if new_layout:
            assert defining.colbert_scores is patched_colbert_scores
        else:
            assert c.colbert_scores is patched_colbert_scores
    finally:
        unpatch_pylate()


def test_cached_contrastive_loss_uses_patched_scores():
    """`CachedContrastive` is the LightOn Reason-ModernColBERT training recipe —
    it chunks MaxSim via the `score_metric` parameter. Make sure our patch
    intercepts the reference this PyLate version's loss actually calls."""
    from late_interaction_kernels.pylate_compat import (
        _scores_defining_module,
        patch_pylate,
        patched_colbert_scores,
        unpatch_pylate,
    )

    patch_pylate()
    try:
        import pylate.losses.cached_contrastive as cc

        defining, new_layout = _scores_defining_module()
        if new_layout:
            assert defining.colbert_scores is patched_colbert_scores
        else:
            assert cc.colbert_scores is patched_colbert_scores
    finally:
        unpatch_pylate()


def test_distillation_loss_uses_patched_scores():
    """`Distillation` captures `colbert_kd_scores` at import time too."""
    from late_interaction_kernels.pylate_compat import (
        patch_pylate,
        patched_colbert_kd_scores,
        unpatch_pylate,
    )

    patch_pylate()
    try:
        import pylate.losses.distillation as d

        assert d.colbert_kd_scores is patched_colbert_kd_scores
    finally:
        unpatch_pylate()


def test_unpatch_restores_original():
    """`unpatch_pylate()` must restore PyLate's original refs everywhere."""
    from pylate.scores import colbert_scores as original_fn

    from late_interaction_kernels.pylate_compat import (
        _scores_defining_module,
        patch_pylate,
        unpatch_pylate,
    )

    patch_pylate()
    unpatch_pylate()

    import pylate.losses.distillation as dd
    from pylate.scores import colbert_scores as restored_fn

    assert restored_fn is original_fn
    defining, new_layout = _scores_defining_module()
    assert defining.colbert_scores is original_fn
    if not new_layout:
        import pylate.losses.cached_contrastive as cc
        import pylate.losses.contrastive as c

        assert c.colbert_scores is original_fn
        assert cc.colbert_scores is original_fn
    # distillation captures colbert_kd_scores; make sure that's restored too.
    from pylate.scores import colbert_kd_scores as original_kd_fn

    assert dd.colbert_kd_scores is original_kd_fn


def test_lik_disable_env_falls_back(monkeypatch):
    """`LIK_DISABLE=1` must force the fallback path even with CUDA tensors."""
    from pylate.scores import colbert_scores as original_fn

    from late_interaction_kernels.pylate_compat import patch_pylate, unpatch_pylate

    Q, D = _make_qd()
    ref = original_fn(Q, D)

    monkeypatch.setenv("LIK_DISABLE", "1")
    patch_pylate()
    try:
        from pylate.scores import colbert_scores as patched_fn

        out = patched_fn(Q, D)
        assert torch.allclose(out, ref, atol=1e-5)
    finally:
        unpatch_pylate()
