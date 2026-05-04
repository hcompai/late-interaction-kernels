"""Parity tests for `maxsim_inference_scatter`."""

import pytest
import torch

pytestmark = pytest.mark.cuda


def _pack(seqs):
    cu = torch.zeros(len(seqs) + 1, dtype=torch.int32)
    cu[1:] = torch.tensor([s.shape[0] for s in seqs], dtype=torch.int32).cumsum(0)
    return torch.cat(seqs, dim=0), cu


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scatter_matches_reference(dtype):
    from late_interaction_kernels import maxsim_inference_scatter
    from late_interaction_kernels.reference import maxsim_reference_scatter

    torch.manual_seed(0)
    d = 64
    Lq_list = [16, 24, 8, 32]
    Ld_list = [200, 50, 300, 128, 75]

    Qs = [torch.randn(L, d, dtype=dtype) for L in Lq_list]
    Ds = [torch.randn(L, d, dtype=dtype) for L in Ld_list]
    Qp, cu_q = _pack(Qs)
    Dp, cu_d = _pack(Ds)
    Qp_g, Dp_g, cu_q_g, cu_d_g = (t.cuda() for t in (Qp, Dp, cu_q, cu_d))

    pair_q = torch.tensor([0, 1, 1, 2, 3, 0, 2], dtype=torch.int32, device="cuda")
    pair_d = torch.tensor([4, 0, 2, 1, 3, 2, 0], dtype=torch.int32, device="cuda")

    got = maxsim_inference_scatter(Qp_g, Dp_g, cu_q_g, cu_d_g, pair_q, pair_d)
    ref = maxsim_reference_scatter(Qp_g.float(), Dp_g.float(), cu_q_g, cu_d_g, pair_q, pair_d)

    rel = (got - ref).abs().max().item() / max(1e-6, ref.abs().max().item())
    tol = 5e-3 if dtype == torch.float16 else 2e-2
    assert rel < tol, f"max rel err {rel} >= {tol}"


def test_scatter_handles_empty_pairs_and_empty_seqs():
    from late_interaction_kernels import maxsim_inference_scatter

    d = 32
    Qp = torch.randn(8, d, device="cuda", dtype=torch.float16)
    Dp = torch.randn(0, d, device="cuda", dtype=torch.float16)  # one empty doc
    cu_q = torch.tensor([0, 4, 8], dtype=torch.int32, device="cuda")
    cu_d = torch.tensor([0, 0], dtype=torch.int32, device="cuda")

    pair_q = torch.tensor([0], dtype=torch.int32, device="cuda")
    pair_d = torch.tensor([0], dtype=torch.int32, device="cuda")
    out = maxsim_inference_scatter(Qp, Dp, cu_q, cu_d, pair_q, pair_d)
    assert out.shape == (1,) and out.item() == 0.0

    empty = maxsim_inference_scatter(
        Qp,
        torch.randn(4, d, device="cuda", dtype=torch.float16),
        cu_q,
        torch.tensor([0, 4], dtype=torch.int32, device="cuda"),
        torch.tensor([], dtype=torch.int32, device="cuda"),
        torch.tensor([], dtype=torch.int32, device="cuda"),
    )
    assert empty.shape == (0,)


def test_scatter_matches_varlen_full_grid():
    """When the pair list covers every (i, j), scatter and varlen agree."""
    from late_interaction_kernels import maxsim_inference_scatter, maxsim_varlen

    torch.manual_seed(1)
    d = 48
    Qs = [torch.randn(L, d, dtype=torch.bfloat16) for L in (10, 12, 6)]
    Ds = [torch.randn(L, d, dtype=torch.bfloat16) for L in (40, 18, 30, 25)]
    Qp, cu_q = _pack(Qs)
    Dp, cu_d = _pack(Ds)
    Qp, Dp, cu_q, cu_d = (t.cuda() for t in (Qp, Dp, cu_q, cu_d))

    Nq, Nd = len(Qs), len(Ds)
    pair_q = torch.arange(Nq, device="cuda", dtype=torch.int32).repeat_interleave(Nd)
    pair_d = torch.arange(Nd, device="cuda", dtype=torch.int32).repeat(Nq)

    got = maxsim_inference_scatter(Qp, Dp, cu_q, cu_d, pair_q, pair_d).view(Nq, Nd)
    full = maxsim_varlen(Qp, Dp, cu_q, cu_d)

    rel = (got - full).abs().max().item() / max(1e-6, full.abs().max().item())
    assert rel < 5e-3
