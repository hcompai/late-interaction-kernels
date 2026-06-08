"""End-to-end colpali_engine monkey-patch test.

Runs against the *real* colpali_engine (``colpali`` extra), mirroring
:mod:`tests.test_pylate_compat`: import-or-skip plus a signature guard, so
upstream drift fails loudly instead of silently passing a stub. The loss heads
take only synthetic embedding tensors — no model, processor, or images.

The ``colpali`` extra is CPU-only (its torchvision tree conflicts with the
cu124 torch wheel), so the CUDA parity cases run out-of-band on a GPU host via
``scripts/sky_colpali_compat_test.yaml``.
"""

import inspect

import pytest
import torch

pytest.importorskip("colpali_engine")

from colpali_engine.loss import late_interaction_losses as cel  # noqa: E402
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor  # noqa: E402

# patch_colpali_engine() swaps these forwards by name; pin the contract so an
# upstream rename / re-signature skips loudly instead of testing nothing.
# Mirrors the inspect.signature guard in test_pylate_compat.py.
_EXPECTED_FORWARD_PARAMS = {
    "ColbertLoss": ["query_embeddings", "doc_embeddings", "offset"],
    "ColbertPairwiseCELoss": ["query_embeddings", "doc_embeddings", "offset"],
    "ColbertSigmoidLoss": ["query_embeddings", "doc_embeddings", "offset"],
    "ColbertNegativeCELoss": ["query_embeddings", "doc_embeddings", "neg_doc_embeddings", "offset"],
    "ColbertPairwiseNegativeCELoss": ["query_embeddings", "doc_embeddings", "neg_doc_embeddings", "offset"],
}

_missing = [name for name in _EXPECTED_FORWARD_PARAMS if not hasattr(cel, name)]
if _missing:
    pytest.skip(
        f"Installed colpali_engine is missing {_missing}. Upgrade with "
        f"`pip install -U 'colpali-engine>=0.3.10'`.",
        allow_module_level=True,
    )

for _name, _params in _EXPECTED_FORWARD_PARAMS.items():
    _got = list(inspect.signature(getattr(cel, _name).forward).parameters)[1:]
    if _got[: len(_params)] != _params:
        pytest.skip(
            f"colpali_engine.{_name}.forward signature drifted: expected {_params}, got {_got}. "
            f"patch_colpali_engine() needs updating before these tests are meaningful.",
            allow_module_level=True,
        )

NEGATIVE_HEADS = ["ColbertNegativeCELoss", "ColbertPairwiseNegativeCELoss"]
INBATCH_HEADS = ["ColbertLoss", "ColbertPairwiseCELoss", "ColbertSigmoidLoss"]


def _l2(x):
    return torch.nn.functional.normalize(x, p=2, dim=-1)


@pytest.fixture(autouse=True)
def _always_unpatch():
    """Safety net: restore the originals after every test, even on failure."""
    yield
    from late_interaction_kernels.colpali_compat import unpatch_colpali_engine

    unpatch_colpali_engine()


# ---------------------------------------------------------------------------
# Mechanics (CPU): patching + restoration swap the right symbols.
# ---------------------------------------------------------------------------


def test_patch_and_unpatch_round_trip():
    """Patching swaps the scorer + five forwards; unpatching restores them exactly."""
    from late_interaction_kernels.colpali_compat import (
        patch_colpali_engine,
        unpatch_colpali_engine,
    )

    original_score = BaseVisualRetrieverProcessor.score_multi_vector
    original_forwards = {name: getattr(cel, name).forward for name in _EXPECTED_FORWARD_PARAMS}

    patch_colpali_engine()
    try:
        assert BaseVisualRetrieverProcessor.score_multi_vector is not original_score
        for name, forward in original_forwards.items():
            assert getattr(cel, name).forward is not forward
    finally:
        unpatch_colpali_engine()

    assert BaseVisualRetrieverProcessor.score_multi_vector is original_score
    for name, forward in original_forwards.items():
        assert getattr(cel, name).forward is forward


def test_double_patch_is_idempotent():
    """A second patch_colpali_engine() must not stash the already-patched forwards."""
    from late_interaction_kernels.colpali_compat import (
        _ORIGINAL,
        patch_colpali_engine,
        unpatch_colpali_engine,
    )

    patch_colpali_engine()
    try:
        snapshot = dict(_ORIGINAL)
        patch_colpali_engine()
        assert _ORIGINAL == snapshot
    finally:
        unpatch_colpali_engine()


def test_lik_disable_falls_through(monkeypatch):
    """With ``LIK_DISABLE=1`` the patched scorer defers to the original einsum path bit-for-bit."""
    from late_interaction_kernels.colpali_compat import (
        patch_colpali_engine,
        patched_score_multi_vector,
        unpatch_colpali_engine,
    )

    monkeypatch.setenv("LIK_DISABLE", "1")

    torch.manual_seed(0)
    qs = [_l2(torch.randn(13, 64)), _l2(torch.randn(9, 64))]
    ps = [_l2(torch.randn(40, 64)) for _ in range(4)]

    ref = BaseVisualRetrieverProcessor.score_multi_vector(qs, ps, batch_size=3, device="cpu")
    patch_colpali_engine()
    try:
        out = patched_score_multi_vector(qs, ps, batch_size=3, device="cpu")
    finally:
        unpatch_colpali_engine()
    assert torch.allclose(out, ref)


@pytest.mark.parametrize("cls_name", INBATCH_HEADS)
def test_smooth_max_falls_through(cls_name):
    """``use_smooth_max=True`` must run through the original forward unchanged."""
    from late_interaction_kernels.colpali_compat import (
        patch_colpali_engine,
        unpatch_colpali_engine,
    )

    cls = getattr(cel, cls_name)
    torch.manual_seed(0)
    Q = _l2(torch.randn(4, 8, 64))
    D = _l2(torch.randn(4, 12, 64))

    head = cls(use_smooth_max=True, normalize_scores=False)
    ref = head(Q, D)

    patch_colpali_engine()
    try:
        out = head(Q, D)
    finally:
        unpatch_colpali_engine()

    assert torch.equal(out, ref), "use_smooth_max must fall through unchanged"


@pytest.mark.parametrize("cls_name", NEGATIVE_HEADS)
def test_negative_smooth_max_falls_through(cls_name):
    """``use_smooth_max=True`` on the explicit-negative heads must fall through unchanged."""
    from late_interaction_kernels.colpali_compat import (
        patch_colpali_engine,
        unpatch_colpali_engine,
    )

    cls = getattr(cel, cls_name)
    torch.manual_seed(0)
    Q = _l2(torch.randn(5, 16, 32))
    pos_D = _l2(torch.randn(5, 40, 32))
    neg_D = _l2(torch.randn(5, 3, 28, 32))

    head = cls(use_smooth_max=True, normalize_scores=False)
    ref = head(Q, pos_D, neg_D)

    patch_colpali_engine()
    try:
        out = head(Q, pos_D, neg_D)
    finally:
        unpatch_colpali_engine()

    assert torch.equal(out, ref), "use_smooth_max must fall through unchanged"


@pytest.mark.parametrize("cls_name", NEGATIVE_HEADS)
def test_patched_negative_losses_fall_through_on_cpu(cls_name):
    """On CPU the pos/neg terms aren't fused, so the patched forward equals vanilla bit-for-bit."""
    from late_interaction_kernels.colpali_compat import (
        patch_colpali_engine,
        unpatch_colpali_engine,
    )

    cls = getattr(cel, cls_name)
    torch.manual_seed(0)
    Q = _l2(torch.randn(5, 16, 32))
    pos_D = _l2(torch.randn(5, 40, 32))
    neg_D = _l2(torch.randn(5, 3, 28, 32))

    head = cls(normalize_scores=True)
    ref = head(Q, pos_D, neg_D)

    patch_colpali_engine()
    try:
        out = head(Q, pos_D, neg_D)
    finally:
        unpatch_colpali_engine()

    assert torch.equal(out, ref)


# ---------------------------------------------------------------------------
# Correctness (CUDA): patched output matches the original einsum forward.
# ---------------------------------------------------------------------------


@pytest.mark.cuda
def test_patched_score_multi_vector_matches_original_cuda():
    """On CUDA the fused scorer must agree with the einsum reference within fp16 ULPs."""
    from late_interaction_kernels.colpali_compat import (
        patch_colpali_engine,
        patched_score_multi_vector,
        unpatch_colpali_engine,
    )

    torch.manual_seed(0)
    qs = [_l2(torch.randn(13, 64, device="cuda", dtype=torch.float16)) for _ in range(3)]
    ps = [_l2(torch.randn(40, 64, device="cuda", dtype=torch.float16)) for _ in range(7)]

    ref = BaseVisualRetrieverProcessor.score_multi_vector(qs, ps, batch_size=2, device="cuda")
    patch_colpali_engine()
    try:
        out = patched_score_multi_vector(qs, ps, batch_size=2, device="cuda")
    finally:
        unpatch_colpali_engine()

    err = (out - ref).abs().max().item()
    denom = max(1.0, ref.abs().max().item())
    assert err / denom < 5e-3, f"err={err}, denom={denom}"


@pytest.mark.cuda
@pytest.mark.parametrize("cls_name", INBATCH_HEADS)
def test_patched_loss_forwards_match_original_cuda(cls_name):
    """Each patched in-batch loss head's output must match the original einsum forward on CUDA."""
    from late_interaction_kernels.colpali_compat import (
        patch_colpali_engine,
        unpatch_colpali_engine,
    )

    cls = getattr(cel, cls_name)
    torch.manual_seed(0)
    Q = _l2(torch.randn(8, 32, 128, device="cuda", dtype=torch.float16))
    D = _l2(torch.randn(8, 96, 128, device="cuda", dtype=torch.float16))

    # `normalize_scores=False` keeps the comparison free of the small
    # downstream div-by-lengths divisor that masks per-token drift.
    head = cls(normalize_scores=False)
    ref = head(Q, D)

    patch_colpali_engine()
    try:
        out = head(Q, D)
    finally:
        unpatch_colpali_engine()

    err = (out - ref).abs().max().item()
    denom = max(1e-6, ref.abs().max().item())
    assert err / denom < 5e-3, f"err={err}, denom={denom}"


@pytest.mark.cuda
@pytest.mark.parametrize("cls_name", NEGATIVE_HEADS)
@pytest.mark.parametrize("normalize_scores", [False, True])
@pytest.mark.parametrize("in_batch_term_weight", [0.0, 0.5])
def test_patched_negative_losses_match_original_cuda(cls_name, normalize_scores, in_batch_term_weight):
    """The fused pos (maxsim_pairs) + neg (4-D maxsim) heads match the einsum forward on CUDA.

    ``in_batch_term_weight`` 0 isolates the pos/neg path; 0.5 adds the in-batch
    term. ``normalize_scores=True`` covers the fused ``_apply_normalization`` on
    the 1-D positives + 2-D negatives — the config real training uses.
    """
    from late_interaction_kernels.colpali_compat import (
        patch_colpali_engine,
        unpatch_colpali_engine,
    )

    cls = getattr(cel, cls_name)
    torch.manual_seed(0)
    Q = _l2(torch.randn(6, 24, 128, device="cuda", dtype=torch.float16))
    pos_D = _l2(torch.randn(6, 80, 128, device="cuda", dtype=torch.float16))
    neg_D = _l2(torch.randn(6, 4, 64, 128, device="cuda", dtype=torch.float16))

    head = cls(normalize_scores=normalize_scores, in_batch_term_weight=in_batch_term_weight)
    ref = head(Q, pos_D, neg_D)

    patch_colpali_engine()
    try:
        out = head(Q, pos_D, neg_D)
    finally:
        unpatch_colpali_engine()

    err = (out - ref).abs().max().item()
    denom = max(1e-6, ref.abs().max().item())
    assert err / denom < 5e-3, f"err={err}, denom={denom}"


@pytest.mark.cuda
@pytest.mark.parametrize("cls_name", NEGATIVE_HEADS)
# fp16 checks the pos/neg fusion (this PR) in its native dtype with the
# in-batch term off; fp32 checks the full mix (weight=0.5) because the in-batch
# softmax-CE term amplifies fp16 argmax-tie noise into ~20% grad drift — fp32
# isolates graph correctness from that noise (a sweep collapsed every case to
# <2e-3).
@pytest.mark.parametrize("dtype,weight", [(torch.float16, 0.0), (torch.float32, 0.5)])
def test_patched_negative_losses_backward_matches_original_cuda(cls_name, dtype, weight):
    """Patched negative-loss backward reproduces the original autograd grads on Q / pos / neg."""
    from late_interaction_kernels.colpali_compat import (
        patch_colpali_engine,
        unpatch_colpali_engine,
    )

    cls = getattr(cel, cls_name)
    torch.manual_seed(0)
    Q0 = _l2(torch.randn(6, 24, 128, device="cuda", dtype=dtype))
    P0 = _l2(torch.randn(6, 80, 128, device="cuda", dtype=dtype))
    N0 = _l2(torch.randn(6, 4, 64, 128, device="cuda", dtype=dtype))

    head = cls(normalize_scores=False, in_batch_term_weight=weight).to("cuda")

    def _grads(patched: bool):
        Q = Q0.clone().requires_grad_(True)
        P = P0.clone().requires_grad_(True)
        N = N0.clone().requires_grad_(True)
        if patched:
            patch_colpali_engine()
        try:
            head(Q, P, N).backward()
        finally:
            if patched:
                unpatch_colpali_engine()
        return Q.grad, P.grad, N.grad

    ref = _grads(patched=False)
    got = _grads(patched=True)

    def _rel(a, b):
        return (a.float() - b.float()).abs().max() / max(1e-6, b.float().abs().max().item())

    tol = 3e-2 if dtype == torch.float16 else 5e-3
    for g_got, g_ref in zip(got, ref, strict=True):
        assert _rel(g_got, g_ref) < tol


@pytest.mark.cuda
def test_patched_colbert_loss_backward_matches_original():
    """Patched ``ColbertLoss.backward`` must reproduce the original autograd grads."""
    from late_interaction_kernels.colpali_compat import (
        patch_colpali_engine,
        unpatch_colpali_engine,
    )

    torch.manual_seed(0)
    # fp16 matches the kernel's compute dtype: fp32 inputs would run the
    # reference in fp32 while the kernel quantizes to fp16, flipping the inner
    # argmax on near-tied candidates and blowing up the grad diff. fp16 beats
    # bf16 here (10 mantissa bits vs 7 → fewer ties).
    Q0 = _l2(torch.randn(8, 32, 128, device="cuda", dtype=torch.float16))
    D0 = _l2(torch.randn(8, 96, 128, device="cuda", dtype=torch.float16))

    head = cel.ColbertLoss(normalize_scores=False).to("cuda")

    Q_ref = Q0.clone().requires_grad_(True)
    D_ref = D0.clone().requires_grad_(True)
    head(Q_ref, D_ref).backward()

    Q_fused = Q0.clone().requires_grad_(True)
    D_fused = D0.clone().requires_grad_(True)
    patch_colpali_engine()
    try:
        head(Q_fused, D_fused).backward()
    finally:
        unpatch_colpali_engine()

    def _rel(a, b):
        diff = (a.float() - b.float()).abs().max()
        return diff / max(1e-6, b.float().abs().max().item())

    # 3e-2 on max-abs relative diff: covers the observed ~2.6% argmax-tie drift
    # (kernel reduces in fp32, einsum quantizes to fp16 first) while still
    # flagging real regressions.
    assert _rel(Q_fused.grad, Q_ref.grad) < 3e-2
    assert _rel(D_fused.grad, D_ref.grad) < 3e-2


def test_patch_colpali_engine_is_noop_on_native_colpali(monkeypatch):
    """On colpali-engine >= 0.3.17 (native LIK), `patch_colpali_engine()` must
    detect it by version, leave colpali-engine's own dispatch untouched, and
    warn — never swap class methods that the native build already routes."""
    import warnings

    import late_interaction_kernels.colpali_compat as compat

    monkeypatch.setattr(compat, "package_at_least", lambda name, minimum: True)
    monkeypatch.setattr(compat, "_NATIVE_NOTICE_SHOWN", False)
    compat._ORIGINAL.clear()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compat.patch_colpali_engine()

    assert compat._ORIGINAL == {}, "native colpali-engine must not be patched"
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    compat.unpatch_colpali_engine()  # must be a safe no-op
