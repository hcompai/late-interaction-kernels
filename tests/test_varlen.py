"""Varlen (packed) MaxSim parity tests."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.cuda


def _build_varlen(seqlens, d, dtype=torch.float16, device="cuda"):
    total = sum(seqlens)
    data = torch.randn(total, d, device=device, dtype=dtype)
    cu = torch.zeros(len(seqlens) + 1, device=device, dtype=torch.int32)
    cu[1:] = torch.tensor(seqlens, device=device, dtype=torch.int32).cumsum(0)
    return data, cu


@pytest.mark.parametrize(
    "q_lens,d_lens,d",
    [
        ([32, 32], [64, 128, 32, 200], 128),
        ([5, 10, 1], [50, 60, 1], 128),
        ([32], [300] * 4, 128),
        ([16, 48, 32, 8], [128, 256, 512, 32, 200], 256),
    ],
)
def test_varlen_parity(q_lens, d_lens, d):
    from flash_colbert import maxsim_varlen
    from flash_colbert.reference import maxsim_reference_varlen

    Qp, cu_q = _build_varlen(q_lens, d)
    Dp, cu_d = _build_varlen(d_lens, d)

    fast = maxsim_varlen(Qp, Dp, cu_q, cu_d).float()
    ref = maxsim_reference_varlen(Qp.float(), Dp.float(), cu_q, cu_d)

    err = (fast - ref).abs().max().item()
    denom = max(1.0, ref.abs().max().item())
    assert err / denom < 5e-3, f"err={err} denom={denom}"


def test_varlen_empty_sequence_is_zero():
    from flash_colbert import maxsim_varlen

    Qp, cu_q = _build_varlen([5, 0, 3], 128)  # middle query has 0 tokens
    Dp, cu_d = _build_varlen([8, 16], 128)
    scores = maxsim_varlen(Qp, Dp, cu_q, cu_d)
    assert torch.allclose(scores[1], torch.zeros(2, device="cuda"))


def test_varlen_matches_padded_path():
    """Pack the docs of `maxsim` into cu_seqlens and check scores match."""
    from flash_colbert import maxsim, maxsim_varlen

    Nq, Nd, Lq, Ld, d = 4, 8, 32, 256, 128
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float16)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float16)

    Qp = Q.reshape(-1, d)
    Dp = D.reshape(-1, d)
    cu_q = torch.arange(0, (Nq + 1) * Lq, Lq, device="cuda", dtype=torch.int32)
    cu_d = torch.arange(0, (Nd + 1) * Ld, Ld, device="cuda", dtype=torch.int32)

    a = maxsim(Q, D).float()
    b = maxsim_varlen(Qp, Dp, cu_q, cu_d)
    assert (a - b).abs().max().item() < 1e-3
