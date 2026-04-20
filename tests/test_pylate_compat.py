"""End-to-end PyLate monkey-patch test."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.cuda

pylate = pytest.importorskip("pylate")


def test_pylate_colbert_scores_patched_no_mask():
    """With no mask (or an all-ones mask) the patched path must match PyLate
    bitwise-close. Mask semantics differ (PyLate multiplies by 0, we use
    -inf) but without a mask both reduce to plain MaxSim."""
    from pylate.scores import colbert_scores as original_fn

    from late_interaction_kernels.pylate_compat import patch_pylate, unpatch_pylate

    Q = torch.nn.functional.normalize(
        torch.randn(4, 32, 128, device="cuda", dtype=torch.float32),
        p=2,
        dim=-1,
    )
    D = torch.nn.functional.normalize(
        torch.randn(8, 128, 128, device="cuda", dtype=torch.float32),
        p=2,
        dim=-1,
    )
    d_mask = torch.ones(8, 128, device="cuda")  # no real masking

    ref = original_fn(Q, D, d_mask)
    patch_pylate()
    try:
        from pylate.scores import colbert_scores as patched_fn

        out = patched_fn(Q, D, d_mask)
        err = (out.float() - ref.float()).abs().max().item()
        denom = max(1.0, ref.abs().max().item())
        assert err / denom < 5e-3, f"err={err}, denom={denom}"
    finally:
        unpatch_pylate()


def test_pylate_colbert_scores_matches_our_reference():
    """With a non-trivial doc mask, the patched function must match our own
    reference (same -inf mask semantics). PyLate's mask-by-multiplication
    is equivalent whenever real scores are ≥ 0 (the normalized case)."""
    from late_interaction_kernels.pylate_compat import patch_pylate, unpatch_pylate
    from late_interaction_kernels.reference import maxsim_reference

    Q = torch.nn.functional.normalize(
        torch.randn(4, 32, 128, device="cuda", dtype=torch.float32),
        p=2,
        dim=-1,
    )
    D = torch.nn.functional.normalize(
        torch.randn(8, 128, 128, device="cuda", dtype=torch.float32),
        p=2,
        dim=-1,
    )
    d_mask = (torch.rand(8, 128, device="cuda") > 0.2).float()
    d_mask[:, 0] = 1

    ref = maxsim_reference(Q, D, d_mask=d_mask.bool()).float()

    patch_pylate()
    try:
        from pylate.scores import colbert_scores as patched_fn

        # PyLate signature: (Q, D, mask) where mask is the doc mask.
        out = patched_fn(Q, D, d_mask).float()
        err = (out - ref).abs().max().item()
        denom = max(1.0, ref.abs().max().item())
        assert err / denom < 5e-3, f"err={err}, denom={denom}"
    finally:
        unpatch_pylate()


def test_pylate_contrastive_loss_uses_flash():
    """Verify the monkey-patch propagates into `pylate.losses.contrastive`."""
    from late_interaction_kernels.pylate_compat import patch_pylate, patched_colbert_scores, unpatch_pylate

    patch_pylate()
    try:
        import pylate.losses.contrastive as c

        assert c.colbert_scores is patched_colbert_scores
    finally:
        unpatch_pylate()


def test_pylate_cached_contrastive_loss_uses_flash():
    """`CachedContrastive` is the LightOn Reason-ModernColBERT training recipe —
    it chunks MaxSim via the `score_metric` parameter (default `colbert_scores`).
    Make sure our patch intercepts that import too."""
    from late_interaction_kernels.pylate_compat import patch_pylate, patched_colbert_scores, unpatch_pylate

    patch_pylate()
    try:
        import pylate.losses.cached_contrastive as cc

        assert cc.colbert_scores is patched_colbert_scores
    finally:
        unpatch_pylate()
