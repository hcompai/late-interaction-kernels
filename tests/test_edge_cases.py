"""Edge cases and robustness tests."""

import pytest
import torch

from tests.conftest import needs_large_smem

pytestmark = pytest.mark.cuda


def test_single_token_doc():
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.reference import maxsim_reference

    Q = torch.randn(2, 8, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(3, 1, 128, device="cuda", dtype=torch.float16)
    a = maxsim(Q, D).float()
    b = maxsim_reference(Q.float(), D.float())
    assert (a - b).abs().max().item() < 1e-2


def test_single_query_token():
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.reference import maxsim_reference

    Q = torch.randn(2, 1, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(3, 64, 128, device="cuda", dtype=torch.float16)
    a = maxsim(Q, D).float()
    b = maxsim_reference(Q.float(), D.float())
    assert (a - b).abs().max().item() < 1e-2


@pytest.mark.parametrize(
    "d",
    [
        16,
        33,
        37,
        63,
        96,
        111,
        250,
        pytest.param(
            513,
            marks=pytest.mark.skipif(
                needs_large_smem(513),
                reason="d=513 overflows sm_75 shared memory; runs on sm_80+",
            ),
        ),
    ],
)
def test_non_power_of_two_embedding_dim(d):
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.reference import maxsim_reference

    Q = torch.randn(2, 16, d, device="cuda", dtype=torch.float16)
    D = torch.randn(3, 32, d, device="cuda", dtype=torch.float16)
    a = maxsim(Q, D).float()
    b = maxsim_reference(Q.float(), D.float())
    denom = max(1.0, b.abs().max().item())
    assert (a - b).abs().max().item() / denom < 5e-3


def test_non_contiguous_inputs():
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.reference import maxsim_reference

    Q_big = torch.randn(4, 64, 128, device="cuda", dtype=torch.float16)
    Q = Q_big[:, ::2, :]  # stride in L dimension
    D = torch.randn(3, 64, 128, device="cuda", dtype=torch.float16)
    a = maxsim(Q, D).float()
    b = maxsim_reference(Q.float(), D.float())
    assert (a - b).abs().max().item() < 1e-2


def test_large_batch_many_docs():
    """Ensures we don't OOM or mis-compute at 'real reranking' scales."""
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.reference import maxsim_reference

    Nq, Nd, Lq, Ld, d = 1, 1000, 32, 300, 128
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float16)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float16)

    fast = maxsim(Q, D).float()
    # Sample a few docs for the reference (full reference would OOM / be slow)
    idx = torch.randperm(Nd, device="cuda")[:10]
    ref = maxsim_reference(Q.float(), D[idx].float())
    err = (fast[:, idx] - ref).abs().max().item()
    assert err < 5e-3 * max(1.0, ref.abs().max().item())
