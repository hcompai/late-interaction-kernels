"""Parity tests for `maxsim_xtr` (top-k aggregated MaxSim)."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.cuda


def test_xtr_degenerates_to_maxsim_at_k1():
    from late_interaction_kernels import maxsim, maxsim_xtr

    Q = torch.randn(2, 32, 128, device="cuda", dtype=torch.bfloat16)
    D = torch.randn(4, 128, 128, device="cuda", dtype=torch.bfloat16)

    xtr = maxsim_xtr(Q, D, top_k=1, normalize_by_k=False)
    ref = maxsim(Q, D)
    assert (xtr.float() - ref.float()).abs().max().item() < 1e-3


@pytest.mark.parametrize("top_k", [2, 5, 10])
def test_xtr_matches_reference(top_k):
    from late_interaction_kernels import maxsim_xtr
    from late_interaction_kernels.xtr import _xtr_reference

    Q = torch.randn(2, 32, 128, device="cuda", dtype=torch.bfloat16)
    D = torch.randn(4, 128, 128, device="cuda", dtype=torch.bfloat16)

    fast = maxsim_xtr(Q, D, top_k=top_k).float()
    ref = _xtr_reference(Q, D, top_k=top_k)
    assert (fast - ref).abs().max().item() < 5e-3


def test_xtr_with_masks():
    from late_interaction_kernels import maxsim_xtr
    from late_interaction_kernels.xtr import _xtr_reference

    Q = torch.randn(2, 32, 128, device="cuda", dtype=torch.bfloat16)
    D = torch.randn(4, 128, 128, device="cuda", dtype=torch.bfloat16)
    qm = torch.ones(2, 32, device="cuda", dtype=torch.bool)
    qm[:, 24:] = False
    dm = torch.ones(4, 128, device="cuda", dtype=torch.bool)
    dm[:, 80:] = False

    fast = maxsim_xtr(Q, D, top_k=3, q_mask=qm, d_mask=dm)
    ref = _xtr_reference(Q, D, top_k=3, q_mask=qm, d_mask=dm)
    assert (fast.float() - ref.float()).abs().max().item() < 5e-3


def test_xtr_normalize_by_k_scaling():
    from late_interaction_kernels import maxsim_xtr

    Q = torch.randn(1, 32, 128, device="cuda", dtype=torch.float32)
    D = torch.randn(2, 128, 128, device="cuda", dtype=torch.float32)

    with_k = maxsim_xtr(Q, D, top_k=5, normalize_by_k=True)
    without_k = maxsim_xtr(Q, D, top_k=5, normalize_by_k=False)
    assert torch.allclose(with_k * 5, without_k, atol=1e-3)
