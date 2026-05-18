"""Numerical parity for the unified-backward reference implementation.

Pins the reference-vs-torch autograd parity of
``maxsim_backward_unified_reference`` so that the Triton kernel's
output must match this reference to fp32 tolerance.
"""

import pytest
import torch

from late_interaction_kernels.backward import (
    maxsim_backward_unified,
    maxsim_backward_unified_reference,
)
from late_interaction_kernels.reference import maxsim_reference


def _autograd_gradients(Q, D, q_mask=None):
    """Ground truth — let PyTorch's autograd compute grad_Q, grad_D."""
    Q = Q.detach().clone().requires_grad_(True)
    D = D.detach().clone().requires_grad_(True)
    scores = maxsim_reference(Q, D, q_mask=q_mask)
    scores.sum().backward()
    return Q.grad.detach(), D.grad.detach()


def _make_argmax(Q, D):
    """Compute the argmax buffer the forward would save: [Nq*Nd, Lq]."""
    Nq, Lq, _ = Q.shape
    Nd, Ld, _ = D.shape
    S = torch.einsum("ild,jtd->ijlt", Q.float(), D.float())  # [Nq, Nd, Lq, Ld]
    arg = S.argmax(dim=-1).to(torch.int32)  # [Nq, Nd, Lq]
    return arg.reshape(Nq * Nd, Lq)


def test_unified_reference_matches_autograd_no_mask():
    torch.manual_seed(0)
    Nq, Nd, Lq, Ld, d = 3, 4, 8, 16, 12
    Q = torch.randn(Nq, Lq, d, dtype=torch.float32)
    D = torch.randn(Nd, Ld, d, dtype=torch.float32)
    argmax = _make_argmax(Q, D)
    grad_scores = torch.ones(Nq, Nd, dtype=torch.float32)

    gq_ref, gd_ref = _autograd_gradients(Q, D)
    gq, gd = maxsim_backward_unified_reference(grad_scores, Q, D, argmax, q_mask=None)

    torch.testing.assert_close(gq, gq_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(gd, gd_ref, atol=1e-5, rtol=1e-5)


def test_unified_reference_matches_autograd_with_q_mask():
    torch.manual_seed(1)
    Nq, Nd, Lq, Ld, d = 2, 3, 10, 15, 8
    Q = torch.randn(Nq, Lq, d, dtype=torch.float32)
    D = torch.randn(Nd, Ld, d, dtype=torch.float32)
    q_mask = torch.ones(Nq, Lq, dtype=torch.bool)
    q_mask[0, 6:] = False  # first query has 6 real tokens
    q_mask[1, 8:] = False
    argmax = _make_argmax(Q, D)
    grad_scores = torch.randn(Nq, Nd, dtype=torch.float32)

    gq_ref, gd_ref = _autograd_gradients(Q, D, q_mask=q_mask)
    # autograd with q_mask sums only active query tokens — match by
    # pre-masking grad_scores' broadcast pattern.
    gq, gd = maxsim_backward_unified_reference(grad_scores, Q, D, argmax, q_mask=q_mask)

    # Our reference uses explicit grad_scores; recompute autograd with
    # the same grad_scores for a matching comparison.
    Q2 = Q.detach().clone().requires_grad_(True)
    D2 = D.detach().clone().requires_grad_(True)
    scores = maxsim_reference(Q2, D2, q_mask=q_mask)
    scores.backward(grad_scores)
    torch.testing.assert_close(gq, Q2.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(gd, D2.grad, atol=1e-5, rtol=1e-5)


def test_unified_reference_dtype_roundtrip():
    """Reference preserves the dtype of the inputs on the output grads."""
    Q = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    D = torch.randn(3, 6, 8, dtype=torch.bfloat16)
    arg = _make_argmax(Q, D)
    g = torch.ones(2, 3, dtype=torch.float32)

    gq, gd = maxsim_backward_unified_reference(g, Q, D, arg)
    assert gq.dtype == torch.bfloat16
    assert gd.dtype == torch.bfloat16


def test_unified_kernel_rejects_unknown_method():
    with pytest.raises(ValueError, match="atomic"):
        maxsim_backward_unified(
            torch.zeros(1, 1),
            torch.zeros(1, 1, 1),
            torch.zeros(1, 1, 1),
            torch.zeros(1, 1, dtype=torch.int32),
            method="nope",
        )


# --------------------------------------------------------------------------- #
# CUDA parity: unified Triton kernel == two-pass Triton kernel                #
# --------------------------------------------------------------------------- #

pytestmark_cuda = pytest.mark.cuda


PARITY_SHAPES = [
    # (Nq, Nd, Lq, Ld, d)
    (1, 4, 32, 64, 128),
    (4, 8, 32, 128, 128),
    (16, 16, 32, 200, 128),
    (32, 32, 32, 200, 128),  # PyLate in-batch-negatives, B=32
    (64, 64, 32, 200, 128),  # B=64
    (8, 8, 128, 1024, 128),  # long doc
    (2, 2, 1024, 1024, 128),  # ColPali-ish
    (4, 4, 32, 256, 48),  # LateOn-Code-edge
    (4, 4, 32, 256, 256),  # fatter d
]
PARITY_IDS = [f"Nq{s[0]}_Nd{s[1]}_Lq{s[2]}_Ld{s[3]}_d{s[4]}" for s in PARITY_SHAPES]


@pytest.mark.cuda
@pytest.mark.parametrize("shape", PARITY_SHAPES, ids=PARITY_IDS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_unified_kernel_matches_two_pass(shape, dtype, rel):
    """Unified Triton kernel == two-pass Triton kernel, bitwise parity.

    Both paths use fp32 atomics into grad_D, so the summation order can
    differ and introduce tiny fp32 drift. Tight tolerance is fine: the
    per-element contributions are identical.
    """
    from late_interaction_kernels.backward import maxsim_backward
    from late_interaction_kernels.forward import _run_forward

    Nq, Nd, Lq, Ld, d = shape
    torch.manual_seed(0)
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=dtype)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=dtype)
    grad_s = torch.randn(Nq, Nd, device="cuda", dtype=torch.float32)

    _, argmax = _run_forward(Q, D, q_mask=None, d_mask=None, save_argmax=True)

    gQ_atom, gD_atom = maxsim_backward(grad_s, Q, D, argmax, None, None, method="atomic")
    gQ_uni, gD_uni = maxsim_backward_unified(grad_s, Q, D, argmax, q_mask=None)

    # grad_Q path is structurally identical to the two-pass dQ kernel —
    # must match exactly.
    torch.testing.assert_close(gQ_uni.float(), gQ_atom.float(), atol=1e-5, rtol=1e-5)
    # grad_D uses fp32 atomics in both variants; the order of atomic_adds
    # can differ, giving tiny fp32 non-associativity drift. 3e-3 matches
    # the convention used in test_backward.py for atomic-vs-reference.
    assert rel(gD_uni.float(), gD_atom.float()) < 3e-3


@pytest.mark.cuda
@pytest.mark.parametrize("shape", [(8, 16, 32, 128, 128), (32, 32, 32, 200, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_unified_kernel_matches_two_pass_with_qmask(shape, dtype, rel):
    from late_interaction_kernels.backward import maxsim_backward
    from late_interaction_kernels.forward import _run_forward

    Nq, Nd, Lq, Ld, d = shape
    torch.manual_seed(1)
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=dtype)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=dtype)
    grad_s = torch.randn(Nq, Nd, device="cuda", dtype=torch.float32)
    q_mask = torch.rand(Nq, Lq, device="cuda") > 0.3  # ~70% kept
    q_mask_i8 = q_mask.to(torch.int8)

    _, argmax = _run_forward(Q, D, q_mask=q_mask_i8, d_mask=None, save_argmax=True)

    gQ_atom, gD_atom = maxsim_backward(grad_s, Q, D, argmax, q_mask_i8, None, method="atomic")
    gQ_uni, gD_uni = maxsim_backward_unified(grad_s, Q, D, argmax, q_mask=q_mask_i8)

    torch.testing.assert_close(gQ_uni.float(), gQ_atom.float(), atol=1e-5, rtol=1e-5)
    assert rel(gD_uni.float(), gD_atom.float()) < 3e-3


@pytest.mark.cuda
def test_unified_matches_csr_deterministic(rel):
    """Check unified atomic vs CSR-deterministic: same math, different order.

    CSR is deterministic and reference-accurate; the unified-atomic kernel
    should match it to fp32 atomic-reordering tolerance.
    """
    from late_interaction_kernels.backward import maxsim_backward
    from late_interaction_kernels.forward import _run_forward

    Nq, Nd, Lq, Ld, d = 16, 16, 32, 200, 128
    torch.manual_seed(3)
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.bfloat16)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.bfloat16)
    grad_s = torch.randn(Nq, Nd, device="cuda", dtype=torch.float32)
    _, argmax = _run_forward(Q, D, q_mask=None, d_mask=None, save_argmax=True)

    gQ_csr, gD_csr = maxsim_backward(grad_s, Q, D, argmax, None, None, method="csr")
    gQ_uni, gD_uni = maxsim_backward_unified(grad_s, Q, D, argmax, q_mask=None)

    torch.testing.assert_close(gQ_uni.float(), gQ_csr.float(), atol=1e-5, rtol=1e-5)
    assert rel(gD_uni.float(), gD_csr.float()) < 3e-3


@pytest.mark.cuda
def test_unified_end_to_end_autograd():
    """``maxsim(...)`` with method='unified' must train identically to 'atomic'."""
    from late_interaction_kernels import maxsim, set_backward_method

    torch.manual_seed(2)
    Q_ref = torch.randn(8, 32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    D_ref = torch.randn(8, 200, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    Q_uni = Q_ref.detach().clone().requires_grad_(True)
    D_uni = D_ref.detach().clone().requires_grad_(True)

    set_backward_method("atomic")
    maxsim(Q_ref, D_ref).sum().backward()
    set_backward_method("unified")
    maxsim(Q_uni, D_uni).sum().backward()
    set_backward_method("auto")  # reset

    rel_q = (Q_uni.grad.float() - Q_ref.grad.float()).abs().max() / max(1e-6, Q_ref.grad.float().abs().max())
    rel_d = (D_uni.grad.float() - D_ref.grad.float()).abs().max() / max(1e-6, D_ref.grad.float().abs().max())
    assert rel_q < 1e-4, f"grad_Q drift = {rel_q:.2e}"
    assert rel_d < 3e-3, f"grad_D drift = {rel_d:.2e}"
