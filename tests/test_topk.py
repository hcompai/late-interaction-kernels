"""Parity tests for `maxsim_topk`."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.cuda


@pytest.mark.parametrize("shape", [(1, 100, 32, 200, 128), (4, 50, 32, 128, 128)])
@pytest.mark.parametrize("k", [1, 5, 10])
def test_topk_parity(shape, k, rel):
    from late_interaction_kernels import maxsim, maxsim_topk

    Nq, Nd, Lq, Ld, d = shape
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.bfloat16)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.bfloat16)

    scores, idx = maxsim_topk(Q, D, k)
    ref_scores = maxsim(Q, D).float()
    ref_topk_s, ref_topk_i = torch.topk(ref_scores, k, dim=-1)

    assert rel(scores.float(), ref_topk_s) < 1e-5
    assert torch.equal(idx, ref_topk_i)


@pytest.mark.parametrize("chunk", [50, 100])
def test_topk_chunked_matches_unchunked(chunk):
    from late_interaction_kernels import maxsim_topk

    Q = torch.randn(2, 32, 128, device="cuda", dtype=torch.bfloat16)
    D = torch.randn(300, 128, 128, device="cuda", dtype=torch.bfloat16)

    s_full, i_full = maxsim_topk(Q, D, 10)
    s_chunk, i_chunk = maxsim_topk(Q, D, 10, chunk_size=chunk)

    # Scores should match exactly at fp32 (same underlying kernel, different
    # reduction order only).
    assert (s_full.float() - s_chunk.float()).abs().max().item() < 1e-4
    # Indices should match when scores are distinct (random → safe).
    assert torch.equal(i_full, i_chunk)


def test_topk_with_masks():
    from late_interaction_kernels import maxsim, maxsim_topk

    Q = torch.randn(2, 32, 128, device="cuda", dtype=torch.bfloat16)
    D = torch.randn(50, 128, 128, device="cuda", dtype=torch.bfloat16)
    qm = torch.ones(2, 32, device="cuda", dtype=torch.bool)
    qm[:, 16:] = False
    dm = torch.ones(50, 128, device="cuda", dtype=torch.bool)
    dm[:, 64:] = False

    s, idx = maxsim_topk(Q, D, 5, q_mask=qm, d_mask=dm)
    ref = maxsim(Q, D, q_mask=qm, d_mask=dm)
    ref_s, ref_i = torch.topk(ref, 5, dim=-1)

    assert (s.float() - ref_s.float()).abs().max().item() < 1e-4
    assert torch.equal(idx, ref_i)


def test_topk_2d_query():
    from late_interaction_kernels import maxsim_topk

    Q = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
    D = torch.randn(20, 128, 128, device="cuda", dtype=torch.bfloat16)

    s, idx = maxsim_topk(Q, D, 3)
    assert s.shape == (3,)
    assert idx.shape == (3,)
    # Scores should be sorted descending.
    assert (s[:-1] >= s[1:]).all()
