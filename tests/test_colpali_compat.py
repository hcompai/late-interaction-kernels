"""End-to-end colpali_engine monkey-patch test.

colpali_engine pulls in transformers / accelerate / torch / PIL, which is a
heavy install for what we need to verify (the patcher swaps two symbol sets
and the fused kernel reproduces the einsum-amax-sum math). We stub the
expected attribute graph in :func:`_fake_colpali_engine` and exercise the
patcher against that stub on CPU (for mechanics) and CUDA (for parity vs
the original einsum implementation).
"""

import sys
import types

import pytest
import torch


def _fake_colpali_engine():
    """Build the minimal :mod:`colpali_engine` surface that ``patch_colpali_engine`` patches.

    Mirrors the upstream signatures so the patcher's ``import`` /
    ``getattr`` lookups succeed and the original behaviour (einsum-amax-
    sum) is faithfully reproduced. Returns the top-level ``colpali_engine``
    module; submodules are wired through ``sys.modules`` so
    ``importlib.import_module(...)`` works.
    """
    top = types.ModuleType("colpali_engine")

    utils = types.ModuleType("colpali_engine.utils")
    processing_utils = types.ModuleType("colpali_engine.utils.processing_utils")
    torch_utils = types.ModuleType("colpali_engine.utils.torch_utils")

    def get_torch_device(_kind: str) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    torch_utils.get_torch_device = get_torch_device

    class BaseVisualRetrieverProcessor:
        """Upstream-shaped stub: only the static ``score_multi_vector`` is used."""

        @staticmethod
        def score_multi_vector(qs, ps, batch_size: int = 128, device=None):
            device = device or get_torch_device("auto")
            scores_list = []
            for i in range(0, len(qs), batch_size):
                qs_batch = torch.nn.utils.rnn.pad_sequence(
                    qs[i : i + batch_size], batch_first=True, padding_value=0
                ).to(device)
                scores_batch = []
                for j in range(0, len(ps), batch_size):
                    ps_batch = torch.nn.utils.rnn.pad_sequence(
                        ps[j : j + batch_size], batch_first=True, padding_value=0
                    ).to(device)
                    scores_batch.append(
                        torch.einsum("bnd,csd->bcns", qs_batch, ps_batch).max(dim=3)[0].sum(dim=2)
                    )
                scores_list.append(torch.cat(scores_batch, dim=1).cpu())
            return torch.cat(scores_list, dim=0).to(torch.float32)

    processing_utils.BaseVisualRetrieverProcessor = BaseVisualRetrieverProcessor

    loss = types.ModuleType("colpali_engine.loss")
    late_interaction_losses = types.ModuleType("colpali_engine.loss.late_interaction_losses")

    class _BaseColbert(torch.nn.Module):
        """Replica of the shared :class:`ColbertModule` helpers used by every patched forward."""

        def __init__(
            self,
            *,
            temperature: float = 0.02,
            normalize_scores: bool = True,
            use_smooth_max: bool = False,
            pos_aware_negative_filtering: bool = False,
            max_batch_size: int = 1024,
        ):
            super().__init__()
            self.temperature = temperature
            self.normalize_scores = normalize_scores
            self.use_smooth_max = use_smooth_max
            self.pos_aware_negative_filtering = pos_aware_negative_filtering
            self.register_buffer("idx_buffer", torch.arange(max_batch_size), persistent=False)
            self.ce_loss = torch.nn.CrossEntropyLoss()

        def _get_idx(self, batch_size: int, offset: int, device: torch.device):
            idx = self.idx_buffer[:batch_size].to(device)
            return idx, idx + offset

        def _apply_normalization(self, scores: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
            return scores / lengths.unsqueeze(1) if scores.ndim == 2 else scores / lengths

        def _filter_high_negatives(self, scores: torch.Tensor, pos_idx: torch.Tensor) -> None:
            batch_size = scores.size(0)
            idx = self.idx_buffer[:batch_size].to(scores.device)
            pos_scores = scores[idx, pos_idx]
            mask = scores > 0.95 * pos_scores.unsqueeze(1)
            mask[idx, pos_idx] = False
            scores[mask] *= 0.5

    class ColbertLoss(_BaseColbert):
        def forward(self, query_embeddings, doc_embeddings, offset: int = 0):
            lengths = (query_embeddings[:, :, 0] != 0).sum(dim=1)
            raw = torch.einsum("bnd,csd->bcns", query_embeddings, doc_embeddings)
            scores = raw.amax(dim=3).sum(dim=2)
            if self.normalize_scores:
                scores = self._apply_normalization(scores, lengths)
            batch_size = scores.size(0)
            _idx, pos_idx = self._get_idx(batch_size, offset, scores.device)
            if self.pos_aware_negative_filtering:
                self._filter_high_negatives(scores, pos_idx)
            return self.ce_loss(scores / self.temperature, pos_idx)

    class ColbertPairwiseCELoss(_BaseColbert):
        def __init__(self, **kwargs):
            kwargs.setdefault("temperature", 1.0)
            super().__init__(**kwargs)

        def forward(self, query_embeddings, doc_embeddings, offset: int = 0):
            import torch.nn.functional as F

            lengths = (query_embeddings[:, :, 0] != 0).sum(dim=1)
            raw = torch.einsum("bnd,csd->bcns", query_embeddings, doc_embeddings)
            scores = raw.amax(dim=3).sum(dim=2)
            if self.normalize_scores:
                scores = self._apply_normalization(scores, lengths)
            batch_size = scores.size(0)
            _idx, pos_idx = self._get_idx(batch_size, offset, scores.device)
            pos_scores = scores.diagonal(offset=offset)
            top2 = scores.topk(2, dim=1).values
            neg_scores = torch.where(top2[:, 0] == pos_scores, top2[:, 1], top2[:, 0])
            return F.softplus((neg_scores - pos_scores) / self.temperature).mean()

    class ColbertSigmoidLoss(_BaseColbert):
        def forward(self, query_embeddings, doc_embeddings, offset: int = 0):
            import torch.nn.functional as F

            lengths = (query_embeddings[:, :, 0] != 0).sum(dim=1)
            raw = torch.einsum("bnd,csd->bcns", query_embeddings, doc_embeddings)
            scores = raw.amax(dim=3).sum(dim=2)
            if self.normalize_scores:
                scores = self._apply_normalization(scores, lengths)
            batch_size = scores.size(0)
            flat_pos = (self.idx_buffer[:batch_size].to(scores.device) + offset) * (batch_size + 1)
            pos_mask = -torch.ones(batch_size * batch_size, device=scores.device)
            pos_mask[flat_pos] = 1.0
            scores = scores.view(-1) / self.temperature
            return F.softplus(-scores * pos_mask).mean()

    late_interaction_losses.ColbertLoss = ColbertLoss
    late_interaction_losses.ColbertPairwiseCELoss = ColbertPairwiseCELoss
    late_interaction_losses.ColbertSigmoidLoss = ColbertSigmoidLoss

    utils.processing_utils = processing_utils
    utils.torch_utils = torch_utils
    loss.late_interaction_losses = late_interaction_losses
    top.utils = utils
    top.loss = loss

    return {
        "colpali_engine": top,
        "colpali_engine.utils": utils,
        "colpali_engine.utils.processing_utils": processing_utils,
        "colpali_engine.utils.torch_utils": torch_utils,
        "colpali_engine.loss": loss,
        "colpali_engine.loss.late_interaction_losses": late_interaction_losses,
    }


@pytest.fixture
def fake_colpali():
    """Install a minimal :mod:`colpali_engine` stub and tear it down on exit."""
    modules = _fake_colpali_engine()
    saved = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    yield modules
    # Always run unpatch first — the test may have left a patched class behind.
    from late_interaction_kernels.colpali_compat import unpatch_colpali_engine

    unpatch_colpali_engine()
    for name, prev in saved.items():
        if prev is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prev


def _l2(x):
    return torch.nn.functional.normalize(x, p=2, dim=-1)


# ---------------------------------------------------------------------------
# Mechanics (CPU): patching + restoration swap the right symbols.
# ---------------------------------------------------------------------------


def test_patch_and_unpatch_round_trip(fake_colpali):
    """Patching swaps four entry points; unpatching restores them exactly."""
    from late_interaction_kernels.colpali_compat import (
        patch_colpali_engine,
        unpatch_colpali_engine,
    )

    base = fake_colpali["colpali_engine.utils.processing_utils"].BaseVisualRetrieverProcessor
    losses = fake_colpali["colpali_engine.loss.late_interaction_losses"]

    original_score = base.score_multi_vector
    original_forwards = {
        cls_name: getattr(losses, cls_name).forward
        for cls_name in ("ColbertLoss", "ColbertPairwiseCELoss", "ColbertSigmoidLoss")
    }

    patch_colpali_engine()
    try:
        assert base.score_multi_vector is not original_score
        for cls_name, forward in original_forwards.items():
            assert getattr(losses, cls_name).forward is not forward
    finally:
        unpatch_colpali_engine()

    assert base.score_multi_vector is original_score
    for cls_name, forward in original_forwards.items():
        assert getattr(losses, cls_name).forward is forward


def test_double_patch_is_idempotent(fake_colpali):
    """A second :func:`patch_colpali_engine` must not stash the already-patched forwards."""
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


def test_lik_disable_falls_through(fake_colpali, monkeypatch):
    """With ``LIK_DISABLE=1`` the patched score_multi_vector defers to the original einsum path bit-for-bit."""
    from late_interaction_kernels.colpali_compat import (
        patch_colpali_engine,
        patched_score_multi_vector,
        unpatch_colpali_engine,
    )

    monkeypatch.setenv("LIK_DISABLE", "1")

    torch.manual_seed(0)
    qs = [_l2(torch.randn(13, 64)), _l2(torch.randn(9, 64))]
    ps = [_l2(torch.randn(40, 64)) for _ in range(4)]

    # The patched callable's CPU branch is the original einsum-amax-sum, so this
    # must match the stub's reference exactly.
    base = fake_colpali["colpali_engine.utils.processing_utils"].BaseVisualRetrieverProcessor
    ref = base.score_multi_vector(qs, ps, batch_size=3, device="cpu")
    patch_colpali_engine()
    try:
        out = patched_score_multi_vector(qs, ps, batch_size=3, device="cpu")
    finally:
        unpatch_colpali_engine()
    assert torch.allclose(out, ref)


def test_smooth_max_falls_through(fake_colpali):
    """``use_smooth_max=True`` must run through the original forward unchanged."""
    from late_interaction_kernels.colpali_compat import (
        patch_colpali_engine,
        unpatch_colpali_engine,
    )

    losses = fake_colpali["colpali_engine.loss.late_interaction_losses"]
    ColbertLoss = losses.ColbertLoss  # noqa: N806 — upstream symbol

    torch.manual_seed(0)
    Q = _l2(torch.randn(4, 8, 64))
    D = _l2(torch.randn(4, 12, 64))

    plain = ColbertLoss(use_smooth_max=False, normalize_scores=False)
    smooth = ColbertLoss(use_smooth_max=True, normalize_scores=False)
    # Smooth-max takes a tau attribute on the upstream module; stub it on
    # the fly so the original branch runs cleanly.
    smooth.tau = 0.1
    smooth._smooth_max = lambda s, dim: smooth.tau * torch.logsumexp(s / smooth.tau, dim=dim)
    ref = smooth(Q, D)
    plain_ref = plain(Q, D)

    patch_colpali_engine()
    try:
        out_smooth = smooth(Q, D)
        out_plain = plain(Q, D)
    finally:
        unpatch_colpali_engine()

    assert torch.equal(out_smooth, ref), "use_smooth_max must fall through unchanged"
    # On CPU the fused path doesn't run (no CUDA/MPS), so plain also falls
    # through and stays bit-equal.
    assert torch.equal(out_plain, plain_ref)


# ---------------------------------------------------------------------------
# Correctness (CUDA): patched output matches the original einsum path.
# ---------------------------------------------------------------------------


@pytest.mark.cuda
def test_patched_score_multi_vector_matches_original_cuda(fake_colpali):
    """On CUDA the fused path must agree with the einsum reference within fp16 ULPs."""
    from late_interaction_kernels.colpali_compat import (
        patch_colpali_engine,
        patched_score_multi_vector,
        unpatch_colpali_engine,
    )

    torch.manual_seed(0)
    qs = [_l2(torch.randn(13, 64, device="cuda", dtype=torch.float16)) for _ in range(3)]
    ps = [_l2(torch.randn(40, 64, device="cuda", dtype=torch.float16)) for _ in range(7)]

    base = fake_colpali["colpali_engine.utils.processing_utils"].BaseVisualRetrieverProcessor
    ref = base.score_multi_vector(qs, ps, batch_size=2, device="cuda")
    patch_colpali_engine()
    try:
        out = patched_score_multi_vector(qs, ps, batch_size=2, device="cuda")
    finally:
        unpatch_colpali_engine()

    err = (out - ref).abs().max().item()
    denom = max(1.0, ref.abs().max().item())
    assert err / denom < 5e-3, f"err={err}, denom={denom}"


@pytest.mark.cuda
@pytest.mark.parametrize("cls_name", ["ColbertLoss", "ColbertPairwiseCELoss", "ColbertSigmoidLoss"])
def test_patched_loss_forwards_match_original_cuda(fake_colpali, cls_name):
    """Each patched loss head's output must match the original einsum forward on CUDA."""
    from late_interaction_kernels.colpali_compat import (
        patch_colpali_engine,
        unpatch_colpali_engine,
    )

    losses = fake_colpali["colpali_engine.loss.late_interaction_losses"]
    cls = getattr(losses, cls_name)

    torch.manual_seed(0)
    Q = _l2(torch.randn(8, 32, 128, device="cuda", dtype=torch.float16))
    D = _l2(torch.randn(8, 96, 128, device="cuda", dtype=torch.float16))

    # `normalize_scores=False` keeps the comparison free of the small
    # downstream div-by-lengths divisor that masks per-token drift.
    head_args: dict[str, object] = {"normalize_scores": False}
    if cls_name == "ColbertPairwiseCELoss":
        head_args["temperature"] = 1.0

    ref_head = cls(**head_args).to("cuda")
    fused_head = cls(**head_args).to("cuda")
    ref = ref_head(Q, D)

    patch_colpali_engine()
    try:
        out = fused_head(Q, D)
    finally:
        unpatch_colpali_engine()

    err = (out - ref).abs().max().item()
    denom = max(1e-6, ref.abs().max().item())
    assert err / denom < 5e-3, f"err={err}, denom={denom}"


@pytest.mark.cuda
def test_patched_colbert_loss_backward_matches_original(fake_colpali):
    """Patched ``ColbertLoss.backward`` must reproduce the original autograd grads."""
    from late_interaction_kernels.colpali_compat import (
        patch_colpali_engine,
        unpatch_colpali_engine,
    )

    losses = fake_colpali["colpali_engine.loss.late_interaction_losses"]
    ColbertLoss = losses.ColbertLoss  # noqa: N806

    torch.manual_seed(0)
    # fp16 matches the kernel's internal compute dtype (`pick_compute_dtype`
    # picks fp16 unless either input is bf16). With fp32 inputs the autograd
    # reference runs in fp32 while the kernel quantizes to fp16 — on
    # L2-normalized tokens that's enough to flip the inner argmax on
    # near-tied max candidates and blow up the gradient diff. fp16 (over
    # bf16) gives tighter argmax agreement here because it has 10 mantissa
    # bits vs bf16's 7, so fewer ties at the bottom of the dot-product
    # distribution. Mirrors `test_patched_loss_forwards_match_original_cuda`.
    Q0 = _l2(torch.randn(8, 32, 128, device="cuda", dtype=torch.float16))
    D0 = _l2(torch.randn(8, 96, 128, device="cuda", dtype=torch.float16))

    head = ColbertLoss(normalize_scores=False).to("cuda")

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

    assert _rel(Q_fused.grad, Q_ref.grad) < 5e-3
    assert _rel(D_fused.grad, D_ref.grad) < 5e-3
