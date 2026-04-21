"""Numerical parity for the unified-backward reference implementation.

We cannot test the Triton unified kernel here because it is not yet
implemented (lands in 0.6.1). But we can pin the reference-vs-torch
autograd parity of ``maxsim_backward_unified_reference`` so that, when
the Triton kernel lands, its output must match this reference to fp32
tolerance.
"""

from __future__ import annotations

import pytest
import torch

from late_interaction_kernels.backward_unified import (
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


def test_unified_kernel_raises_not_implemented():
    """The Triton kernel is a 0.6.1 deliverable; for now the entry-point raises."""
    with pytest.raises(NotImplementedError, match="0.6.1"):
        maxsim_backward_unified()
