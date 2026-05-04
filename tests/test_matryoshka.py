"""Parity tests for `maxsim_matryoshka` (multi-dim scoring)."""

import pytest
import torch

pytestmark = pytest.mark.cuda


@pytest.mark.parametrize("dims", [[64], [32, 64, 128], [16, 32]])
@pytest.mark.parametrize("normalize", [False, True])
def test_matryoshka_parity(dims, normalize, rel):
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.experimental import maxsim_matryoshka

    Nq, Nd, Lq, Ld, d = 2, 4, 32, 128, 128
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.bfloat16)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.bfloat16)

    fast = maxsim_matryoshka(Q, D, dims, normalize=normalize).float()

    for i, dim in enumerate(dims):
        if normalize:
            # Matryoshka normalizes at the FULL dim, then truncates.
            Qn = torch.nn.functional.normalize(Q.float(), p=2, dim=-1).to(torch.bfloat16)
            Dn = torch.nn.functional.normalize(D.float(), p=2, dim=-1).to(torch.bfloat16)
            ref = maxsim(Qn[..., :dim].contiguous(), Dn[..., :dim].contiguous()).float()
        else:
            ref = maxsim(Q[..., :dim].contiguous(), D[..., :dim].contiguous()).float()
        assert rel(fast[i], ref) < 5e-2, f"dim={dim}: max-rel-err={rel(fast[i], ref):.2e}"


def test_matryoshka_ordering_invariant():
    """Passing dims in different orders yields the same per-dim score."""
    from late_interaction_kernels.experimental import maxsim_matryoshka

    Q = torch.randn(2, 32, 128, device="cuda", dtype=torch.bfloat16)
    D = torch.randn(4, 128, 128, device="cuda", dtype=torch.bfloat16)

    a = maxsim_matryoshka(Q, D, [32, 64, 128])
    b = maxsim_matryoshka(Q, D, [128, 32, 64])

    assert torch.allclose(a[0], b[1], atol=1e-4)  # 32
    assert torch.allclose(a[1], b[2], atol=1e-4)  # 64
    assert torch.allclose(a[2], b[0], atol=1e-4)  # 128


def test_matryoshka_full_dim_matches_plain_maxsim():
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.experimental import maxsim_matryoshka

    Q = torch.randn(2, 32, 128, device="cuda", dtype=torch.bfloat16)
    D = torch.randn(4, 128, 128, device="cuda", dtype=torch.bfloat16)

    mm = maxsim_matryoshka(Q, D, [128]).squeeze(0)
    ref = maxsim(Q, D)
    assert (mm.float() - ref.float()).abs().max().item() < 1e-2


def test_matryoshka_with_masks():
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.experimental import maxsim_matryoshka

    Q = torch.randn(2, 32, 128, device="cuda", dtype=torch.bfloat16)
    D = torch.randn(4, 128, 128, device="cuda", dtype=torch.bfloat16)
    qm = torch.ones(2, 32, device="cuda", dtype=torch.bool)
    qm[:, 24:] = False
    dm = torch.ones(4, 128, device="cuda", dtype=torch.bool)
    dm[:, 64:] = False

    mm = maxsim_matryoshka(Q, D, [64, 128], q_mask=qm, d_mask=dm)
    ref64 = maxsim(Q[..., :64].contiguous(), D[..., :64].contiguous(), q_mask=qm, d_mask=dm)
    ref128 = maxsim(Q, D, q_mask=qm, d_mask=dm)
    assert (mm[0].float() - ref64.float()).abs().max().item() < 5e-2
    assert (mm[1].float() - ref128.float()).abs().max().item() < 5e-2
