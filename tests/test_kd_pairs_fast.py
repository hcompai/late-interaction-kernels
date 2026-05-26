"""Forward + backward parity for the KD / pairs fast dispatch.

These tests cover the new ``kd_layout=True`` path in ``_maxsim_fwd_kernel``
and the matching backward kernels (unified + atomic-dQ/dD). Reference
gradients come from PyTorch's autograd through the same einsum the
kernels are meant to fuse.
"""

from __future__ import annotations

import pytest
import torch

triton = pytest.importorskip("triton")

from late_interaction_kernels import maxsim, maxsim_pairs  # noqa: E402


def _ref_inbatch(Q: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    return torch.einsum("nqd,mkd->nmqk", Q, D).max(dim=-1).values.sum(dim=-1)


def _ref_kd(Q: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    return torch.einsum("nqd,nkmd->nkqm", Q, D).max(dim=-1).values.sum(dim=-1)


def _ref_pairs(Q: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bqd,bkd->bqk", Q, D).max(dim=-1).values.sum(dim=-1)


def _cuda_available() -> bool:
    return torch.cuda.is_available()


pytestmark = pytest.mark.skipif(not _cuda_available(), reason="needs CUDA")


# --------------------------------------------------------------------------- #
# Forward parity                                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("Nq", "K", "Lq", "Ld", "d"),
    [
        (4, 1, 32, 180, 128),  # pairwise via K=1
        (4, 8, 32, 180, 128),  # small KD
        (8, 16, 64, 128, 128),
        (4, 4, 128, 512, 128),  # long-doc KD
    ],
)
def test_kd_forward_matches_einsum(dtype, Nq, K, Lq, Ld, d):
    torch.manual_seed(0)
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=dtype)
    D = torch.randn(Nq, K, Ld, d, device="cuda", dtype=dtype)

    got = maxsim(Q, D).float()
    ref = _ref_kd(Q.float(), D.float())

    assert got.shape == (Nq, K)
    # bf16/fp16 inputs against fp32-acc kernel and fp32-cast einsum — same
    # numeric contract Raphael's table uses (max rel err ~1e-6 for the
    # kernels; absolute is bounded by Lq*Ld accumulation).
    torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("B", "Lq", "Ld", "d"),
    [
        (4, 32, 180, 128),
        (256, 32, 180, 128),  # the §5 regression shape (small end)
        (1000, 16, 256, 64),  # bigger pair count, irregular d
    ],
)
def test_pairs_forward_matches_einsum(dtype, B, Lq, Ld, d):
    torch.manual_seed(0)
    Q = torch.randn(B, Lq, d, device="cuda", dtype=dtype)
    D = torch.randn(B, Ld, d, device="cuda", dtype=dtype)

    got = maxsim_pairs(Q, D).float()
    ref = _ref_pairs(Q.float(), D.float())

    assert got.shape == (B,)
    torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)


def test_kd_matches_old_padded_path():
    """The fast KD path must agree with ``maxsim_padded`` (the prior route).

    This is the strongest check: we're not just close to einsum, we're close
    to the prior LIK implementation it replaces. The two paths use the same
    fp32 accumulator and the same ``tl.argmax`` lowest-index tie rule.
    """
    from late_interaction_kernels.padded import maxsim_padded

    torch.manual_seed(0)
    Nq, K, Lq, Ld, d = 4, 8, 32, 128, 128
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float16)
    D = torch.randn(Nq, K, Ld, d, device="cuda", dtype=torch.float16)
    qlen = torch.full((Nq,), Lq, device="cuda", dtype=torch.int32)
    dlen = torch.full((Nq, K), Ld, device="cuda", dtype=torch.int32)

    new_path = maxsim(Q, D).float()
    old_path = maxsim_padded(Q, D, qlen, dlen).float()

    # Same kernel family, same accumulator dtype: should match to fp16 noise.
    torch.testing.assert_close(new_path, old_path, atol=5e-3, rtol=5e-3)


def test_in_batch_cross_product_still_works():
    """Regression: adding ``kd_layout`` must not perturb the existing 3-D path."""
    torch.manual_seed(0)
    Nq, Nd, Lq, Ld, d = 4, 6, 32, 128, 128
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float16)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float16)

    got = maxsim(Q, D).float()
    ref = _ref_inbatch(Q.float(), D.float())

    assert got.shape == (Nq, Nd)
    torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)


# --------------------------------------------------------------------------- #
# Backward parity                                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_kd_backward_matches_autograd(dtype):
    torch.manual_seed(0)
    Nq, K, Lq, Ld, d = 4, 8, 32, 128, 128
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=dtype)
    D = torch.randn(Nq, K, Ld, d, device="cuda", dtype=dtype)

    # Kernel path
    Q1 = Q.detach().clone().requires_grad_(True)
    D1 = D.detach().clone().requires_grad_(True)
    out_kernel = maxsim(Q1, D1)
    out_kernel.sum().backward()

    # Reference path — fp32-cast einsum, then cast grads back so the
    # comparison is symmetric.
    Q2 = Q.detach().clone().float().requires_grad_(True)
    D2 = D.detach().clone().float().requires_grad_(True)
    out_ref = _ref_kd(Q2, D2)
    out_ref.sum().backward()

    grad_Q_kernel = Q1.grad.float()
    grad_D_kernel = D1.grad.float()
    grad_Q_ref = Q2.grad
    grad_D_ref = D2.grad

    # Same parity bar as in-batch backward tests in tests/test_forward.py.
    torch.testing.assert_close(grad_Q_kernel, grad_Q_ref, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(grad_D_kernel, grad_D_ref, atol=5e-2, rtol=5e-2)


def test_pairs_backward_matches_autograd():
    torch.manual_seed(0)
    B, Lq, Ld, d = 8, 32, 128, 128
    dtype = torch.float16
    Q = torch.randn(B, Lq, d, device="cuda", dtype=dtype)
    D = torch.randn(B, Ld, d, device="cuda", dtype=dtype)

    Q1 = Q.detach().clone().requires_grad_(True)
    D1 = D.detach().clone().requires_grad_(True)
    maxsim_pairs(Q1, D1).sum().backward()

    Q2 = Q.detach().clone().float().requires_grad_(True)
    D2 = D.detach().clone().float().requires_grad_(True)
    _ref_pairs(Q2, D2).sum().backward()

    torch.testing.assert_close(Q1.grad.float(), Q2.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(D1.grad.float(), D2.grad, atol=5e-2, rtol=5e-2)


# --------------------------------------------------------------------------- #
# Mask handling                                                                #
# --------------------------------------------------------------------------- #


def test_kd_with_uniform_q_mask():
    """A ``q_mask`` that's contiguous-True over the full Lq should match
    the unmasked path (i.e. mask is a no-op semantically).
    """
    torch.manual_seed(0)
    Nq, K, Lq, Ld, d = 4, 4, 32, 128, 128
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float16)
    D = torch.randn(Nq, K, Ld, d, device="cuda", dtype=torch.float16)
    q_mask = torch.ones(Nq, Lq, device="cuda", dtype=torch.bool)

    s_no_mask = maxsim(Q, D).float()
    s_with_mask = maxsim(Q, D, q_mask=q_mask).float()
    torch.testing.assert_close(s_no_mask, s_with_mask)


def test_kd_with_partial_q_mask_zeros_padded_rows():
    """Padded query rows must not contribute to the score."""
    torch.manual_seed(0)
    Nq, K, Lq, Ld, d = 4, 4, 32, 128, 128
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float16)
    D = torch.randn(Nq, K, Ld, d, device="cuda", dtype=torch.float16)

    # Half the query is padding.
    q_mask = torch.zeros(Nq, Lq, device="cuda", dtype=torch.bool)
    q_mask[:, : Lq // 2] = True

    masked = maxsim(Q, D, q_mask=q_mask).float()
    # Compare to manually-shortened reference.
    Q_short = Q[:, : Lq // 2].float()
    ref = _ref_kd(Q_short, D.float())
    torch.testing.assert_close(masked, ref, atol=2e-2, rtol=2e-2)


def test_kd_with_partial_d_mask():
    """Padded doc rows must not contribute either."""
    torch.manual_seed(0)
    Nq, K, Lq, Ld, d = 2, 4, 32, 128, 128
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float16)
    D = torch.randn(Nq, K, Ld, d, device="cuda", dtype=torch.float16)

    d_mask = torch.zeros(Nq, K, Ld, device="cuda", dtype=torch.bool)
    d_mask[..., : Ld // 2] = True

    masked = maxsim(Q, D, d_mask=d_mask).float()
    D_short = D[:, :, : Ld // 2].float()
    ref = _ref_kd(Q.float(), D_short)
    torch.testing.assert_close(masked, ref, atol=2e-2, rtol=2e-2)


# --------------------------------------------------------------------------- #
# API hygiene                                                                  #
# --------------------------------------------------------------------------- #


def test_maxsim_pairs_rejects_mismatched_batch():
    Q = torch.randn(4, 32, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(5, 32, 128, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="Q.shape\\[0\\] == D.shape\\[0\\]"):
        maxsim_pairs(Q, D)


def test_maxsim_pairs_rejects_wrong_ndim():
    Q = torch.randn(4, 32, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(4, 32, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match=r"\[B, L, d\]"):
        maxsim_pairs(Q, D)


def test_kd_rejects_wrong_nq():
    Q = torch.randn(4, 32, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(5, 4, 128, 128, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="D.shape\\[0\\] == Q.shape\\[0\\]"):
        maxsim(Q, D)
