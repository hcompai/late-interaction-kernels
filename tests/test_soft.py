"""Soft-maxsim (log-sum-exp) tests."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.cuda


@pytest.mark.parametrize("beta", [1.0, 5.0, 25.0])
def test_soft_parity(beta):
    from flash_colbert import soft_maxsim
    from flash_colbert.reference import maxsim_reference_soft

    Nq, Nd, Lq, Ld, d = 4, 4, 32, 128, 128
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float16)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float16)
    q_mask = torch.rand(Nq, Lq, device="cuda") > 0.3
    d_mask = torch.rand(Nd, Ld, device="cuda") > 0.2
    q_mask[:, 0] = True
    d_mask[:, 0] = True

    fast = soft_maxsim(Q, D, q_mask=q_mask, d_mask=d_mask, beta=beta).float()
    ref = maxsim_reference_soft(Q.float(), D.float(), q_mask, d_mask, beta=beta)

    denom = max(1.0, ref.abs().max().item())
    err = (fast - ref).abs().max().item() / denom
    assert err < 1e-2, f"err={err}"


def test_soft_approaches_max_as_beta_grows():
    from flash_colbert import maxsim, soft_maxsim

    Nq, Nd, Lq, Ld, d = 2, 2, 16, 64, 128
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float32)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float32)

    hard = maxsim(Q, D).float()
    soft_high = soft_maxsim(Q, D, beta=100.0).float()
    err = (hard - soft_high).abs().max().item() / max(1.0, hard.abs().max().item())
    # beta=100 should put us within ~1% of the hard max
    assert err < 2e-2, f"err={err}"
