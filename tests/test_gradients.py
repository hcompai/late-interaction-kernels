"""Backward parity against PyTorch autograd."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.cuda


@pytest.mark.parametrize(
    "Nq,Nd,Lq,Ld,d",
    [
        (2, 4, 16, 32, 128),
        (4, 8, 32, 128, 128),
        (2, 2, 32, 300, 128),
        (1, 1, 8, 16, 256),
    ],
)
def test_grad_parity_no_mask(Nq, Nd, Lq, Ld, d):
    from flash_colbert import maxsim
    from flash_colbert.reference import maxsim_reference

    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float32, requires_grad=True)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float32, requires_grad=True)

    # Random upstream grad
    grad_out = torch.randn(Nq, Nd, device="cuda", dtype=torch.float32)

    # Fast path (fp16 compute)
    scores_fast = maxsim(Q.half(), D.half())
    # We need gradients on the fp16 inputs then map back — but we want to
    # compare gradient of the fp32 maxsim (reference) with our kernel's grads
    # on the fp16 inputs. Instead, run reference in fp32 and compare coarse-grained.
    scores_fast.backward(grad_out)
    grad_Q_fast = Q.grad.detach().clone() if Q.grad is not None else None
    grad_D_fast = D.grad.detach().clone() if D.grad is not None else None

    # reset and compute reference grads
    Q.grad = None
    D.grad = None
    scores_ref = maxsim_reference(Q, D)
    scores_ref.backward(grad_out)
    grad_Q_ref = Q.grad.detach().clone()
    grad_D_ref = D.grad.detach().clone()

    # fp16-induced tolerance: scale by tensor magnitude.
    def rel_err(a, b):
        denom = max(1e-6, b.abs().max().item())
        return (a - b).abs().max().item() / denom

    assert grad_Q_fast is not None and grad_D_fast is not None
    errQ = rel_err(grad_Q_fast, grad_Q_ref)
    errD = rel_err(grad_D_fast, grad_D_ref)
    # fp16 compute + fp32 accumulate — tolerate ~1% relative error.
    assert errQ < 1e-2, f"errQ={errQ}"
    assert errD < 1e-2, f"errD={errD}"


def test_grad_parity_with_mask():
    from flash_colbert import maxsim
    from flash_colbert.reference import maxsim_reference

    Nq, Nd, Lq, Ld, d = 4, 4, 32, 128, 128
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float32, requires_grad=True)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float32, requires_grad=True)
    q_mask = torch.rand(Nq, Lq, device="cuda") > 0.2
    d_mask = torch.rand(Nd, Ld, device="cuda") > 0.2
    q_mask[:, 0] = True
    d_mask[:, 0] = True
    grad_out = torch.randn(Nq, Nd, device="cuda", dtype=torch.float32)

    s_fast = maxsim(Q, D, q_mask=q_mask, d_mask=d_mask)
    s_fast.backward(grad_out)
    gQf, gDf = Q.grad.clone(), D.grad.clone()

    Q.grad = None
    D.grad = None
    s_ref = maxsim_reference(Q, D, q_mask, d_mask)
    s_ref.backward(grad_out)
    gQr, gDr = Q.grad.clone(), D.grad.clone()

    def rel_err(a, b):
        return (a - b).abs().max().item() / max(1e-6, b.abs().max().item())

    assert rel_err(gQf, gQr) < 5e-2, rel_err(gQf, gQr)
    assert rel_err(gDf, gDr) < 5e-2, rel_err(gDf, gDr)


def test_masked_query_gets_zero_grad():
    from flash_colbert import maxsim

    Q = torch.randn(2, 8, 128, device="cuda", requires_grad=True)
    D = torch.randn(3, 16, 128, device="cuda", requires_grad=True)
    q_mask = torch.ones(2, 8, device="cuda", dtype=torch.bool)
    q_mask[0, 3] = False

    s = maxsim(Q, D, q_mask=q_mask)
    s.sum().backward()
    assert Q.grad[0, 3].abs().max().item() == 0.0
