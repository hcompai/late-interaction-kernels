"""Forward-pass numerical parity against the pure-PyTorch reference.

These are the first-line correctness tests: they pin the kernel's output to
within the expected tensor-core ULP drift of the reference implementation
across the realistic shape grid for ColBERT (text), ColPali (visual), and
ModernColBERT (long document) workloads.
"""

import pytest
import torch

from tests.conftest import needs_large_smem

pytestmark = pytest.mark.cuda


# (Nq, Nd, Lq, Ld, d) — representative ColBERT / ColPali / ModernColBERT shapes.
SHAPES = [
    (1, 4, 32, 64, 128),
    (8, 16, 32, 128, 128),
    (16, 8, 32, 300, 128),
    (4, 4, 128, 1024, 128),  # long doc
    (2, 2, 256, 256, 128),  # long-seq parity (shrunk from 1024² for CI speed)
    (4, 8, 32, 200, 256),
    (2, 4, 32, 128, 512),
    (2, 2, 32, 128, 1024),  # large d
    (1, 1, 3, 5, 128),  # tiny edge
]
SHAPE_IDS = [f"Nq{s[0]}_Nd{s[1]}_Lq{s[2]}_Ld{s[3]}_d{s[4]}" for s in SHAPES]


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_forward_parity_no_mask(shape, dtype, rel):
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.reference import maxsim_reference

    Nq, Nd, Lq, Ld, d = shape
    if needs_large_smem(d):
        pytest.skip(f"d={d} overflows sm_75 shared memory; runs on sm_80+")
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=dtype)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=dtype)

    fast = maxsim(Q, D).float()
    ref = maxsim_reference(Q.float(), D.float())

    tol = 5e-3 if dtype == torch.float16 else 2e-2
    assert rel(fast, ref) < tol


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_forward_parity_with_masks(shape, rel):
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.reference import maxsim_reference

    Nq, Nd, Lq, Ld, d = shape
    if needs_large_smem(d):
        pytest.skip(f"d={d} overflows sm_75 shared memory; runs on sm_80+")
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float16)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float16)
    q_mask = torch.rand(Nq, Lq, device="cuda") > 0.2
    d_mask = torch.rand(Nd, Ld, device="cuda") > 0.2
    q_mask[:, 0] = True  # guarantee at least one active token per row
    d_mask[:, 0] = True

    fast = maxsim(Q, D, q_mask=q_mask, d_mask=d_mask).float()
    ref = maxsim_reference(Q.float(), D.float(), q_mask, d_mask)

    assert rel(fast, ref) < 5e-3


def test_fully_masked_doc_scores_zero():
    """A doc whose mask is all-False must score 0 against every query (semantic
    contract: masked tokens are -inf, their max is -inf → clamped to 0)."""
    from late_interaction_kernels import maxsim

    Q = torch.randn(2, 8, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(3, 16, 128, device="cuda", dtype=torch.float16)
    d_mask = torch.ones(3, 16, device="cuda", dtype=torch.bool)
    d_mask[1] = False

    scores = maxsim(Q, D, d_mask=d_mask)
    assert torch.equal(scores[:, 1], torch.zeros(2, device="cuda"))


def test_fully_masked_query_scores_zero():
    """Dual of the above: an all-False query mask gives 0 across the corpus."""
    from late_interaction_kernels import maxsim

    Q = torch.randn(3, 8, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(2, 16, 128, device="cuda", dtype=torch.float16)
    q_mask = torch.ones(3, 8, device="cuda", dtype=torch.bool)
    q_mask[1] = False

    scores = maxsim(Q, D, q_mask=q_mask)
    assert torch.equal(scores[1, :], torch.zeros(2, device="cuda"))


def test_no_grad_dispatch_bit_equals_autograd_path():
    """``maxsim`` must produce a bit-identical forward whether the autograd
    path is taken or not. Pins the ``requires_grad``-driven dispatch added
    when ``maxsim_inference`` was folded into ``maxsim``."""
    from late_interaction_kernels import maxsim

    Nq, Nd, Lq, Ld, d = 4, 8, 32, 128, 128
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float16)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float16)

    no_grad = maxsim(Q, D)
    with_grad = maxsim(Q.clone().requires_grad_(True), D.clone().requires_grad_(True)).detach()
    torch.testing.assert_close(no_grad, with_grad, atol=0, rtol=0)


def test_2d_inputs_return_scalar():
    """Single-query, single-doc call returns a scalar, not a [1, 1] tensor."""
    from late_interaction_kernels import maxsim

    Q = torch.randn(8, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(16, 128, device="cuda", dtype=torch.float16)
    s = maxsim(Q, D)
    assert s.dim() == 0, f"expected scalar, got {s.shape}"


@pytest.mark.parametrize("lq", [3, 9, 17, 33, 63, 127])
def test_non_pow2_lq_parity(lq):
    """Bucketing Lq to the next power of two must not change scores.

    Variable-length training has ``Lq`` floating around with the tokenizer
    output. The internal bucket-pad-mask must be transparent to the caller.
    """
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.reference import maxsim_reference

    Nq, Nd, Ld, d = 2, 3, 64, 128
    Q = torch.randn(Nq, lq, d, device="cuda", dtype=torch.float16)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float16)

    fast = maxsim(Q, D).float()
    ref = maxsim_reference(Q.float(), D.float())

    assert fast.shape == (Nq, Nd), f"output shape leaked the bucketed Lq: {fast.shape}"
    torch.testing.assert_close(fast, ref, rtol=5e-3, atol=5e-3)


def test_non_pow2_lq_gradient_shape():
    """``Q.grad`` must come back at the user's Lq, not the bucketed one.

    ``F.pad`` is autograd-aware, so the grad scattered onto the padded Q
    gets sliced back to the original shape on the way out. Pinning this
    so a future refactor doesn't accidentally break the contract.
    """
    from late_interaction_kernels import maxsim

    Q = torch.randn(2, 17, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    D = torch.randn(3, 64, 128, device="cuda", dtype=torch.float16, requires_grad=True)

    maxsim(Q, D).sum().backward()

    assert Q.grad is not None and Q.grad.shape == Q.shape
    assert D.grad is not None and D.grad.shape == D.shape
    assert torch.all(torch.isfinite(Q.grad))
    assert torch.all(torch.isfinite(D.grad))
