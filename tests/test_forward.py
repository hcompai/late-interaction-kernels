"""Forward-pass numerical parity against the pure-PyTorch reference.

These are the first-line correctness tests: they pin the kernel's output to
within the expected tensor-core ULP drift of the reference implementation
across the realistic shape grid for ColBERT (text), ColPali (visual), and
ModernColBERT (long document) workloads.
"""

import pytest
import torch

pytestmark = pytest.mark.cuda


# (Nq, Nd, Lq, Ld, d) — representative ColBERT / ColPali / ModernColBERT shapes.
SHAPES = [
    (1, 4, 32, 64, 128),
    (8, 16, 32, 128, 128),
    (16, 8, 32, 300, 128),
    (4, 4, 128, 1024, 128),  # long doc
    (2, 2, 1024, 1024, 128),  # ColPali-ish
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


def test_2d_inputs_return_scalar():
    """Single-query, single-doc call returns a scalar, not a [1, 1] tensor."""
    from late_interaction_kernels import maxsim

    Q = torch.randn(8, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(16, 128, device="cuda", dtype=torch.float16)
    s = maxsim(Q, D)
    assert s.dim() == 0, f"expected scalar, got {s.shape}"
