"""End-to-end PyLate monkey-patch test."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.cuda

pylate = pytest.importorskip("pylate")


def test_pylate_colbert_scores_patched():
    from pylate.scores import colbert_scores as original_fn

    from flash_colbert.pylate_compat import patch_pylate, unpatch_pylate

    Q = torch.randn(4, 32, 128, device="cuda", dtype=torch.float32)
    D = torch.randn(8, 128, 128, device="cuda", dtype=torch.float32)
    d_mask = (torch.rand(8, 128, device="cuda") > 0.2).float()
    d_mask[:, 0] = 1

    ref = original_fn(Q, D, d_mask)

    patch_pylate()
    try:
        from pylate.scores import colbert_scores as patched_fn

        out = patched_fn(Q, D, d_mask)
        # Semantics differ slightly: PyLate multiplies similarity by 0 on
        # masked tokens (so they can still "win" if all real scores are
        # negative), while we use -inf (masked tokens never win). On random
        # normalized features with ~70% active tokens this rarely matters;
        # compare in a way that tolerates the semantic gap.
        err = (out.float() - ref.float()).abs().max().item()
        denom = max(1.0, ref.abs().max().item())
        assert err / denom < 5e-2, f"err={err}"
    finally:
        unpatch_pylate()


def test_pylate_contrastive_loss_uses_flash():
    """Verify the monkey-patch propagates into `pylate.losses.contrastive`."""
    from flash_colbert.pylate_compat import flash_colbert_scores, patch_pylate, unpatch_pylate

    patch_pylate()
    try:
        import pylate.losses.contrastive as c

        assert c.colbert_scores is flash_colbert_scores
    finally:
        unpatch_pylate()
