"""Forward numerical parity against the pure-PyTorch reference.

Parametrized across realistic ColBERT / ColPali shapes:

  * Text: d=128, Lq=32, Ld ∈ {32, 128, 256}, batch ∈ {1, 8, 32}
  * Visual / ColPali: d=128, Lq=1024, Ld=1024
  * Large embeddings: d ∈ {256, 512, 1024}
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.cuda


SHAPES = [
    # (Nq, Nd, Lq, Ld, d)
    (1, 4, 32, 64, 128),
    (8, 16, 32, 128, 128),
    (16, 8, 32, 300, 128),
    (4, 4, 128, 1024, 128),       # long doc / long query
    (2, 2, 1024, 1024, 128),      # ColPali-ish
    (4, 8, 32, 200, 256),
    (2, 4, 32, 128, 512),
    (2, 2, 32, 128, 1024),
    (1, 1, 3, 5, 128),            # tiny edge case
]


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"Nq{s[0]}_Nd{s[1]}_Lq{s[2]}_Ld{s[3]}_d{s[4]}")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_forward_parity_no_mask(shape, dtype):
    from flash_colbert import maxsim
    from flash_colbert.reference import maxsim_reference

    Nq, Nd, Lq, Ld, d = shape
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=dtype)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=dtype)

    fast = maxsim(Q, D).float()
    ref = maxsim_reference(Q.float(), D.float())

    tol = 5e-3 if dtype == torch.float16 else 2e-2
    err = (fast - ref).abs().max().item()
    assert err < tol * max(1.0, ref.abs().max().item()), f"max err {err} ref_max {ref.abs().max().item()}"


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"Nq{s[0]}_Nd{s[1]}_Lq{s[2]}_Ld{s[3]}_d{s[4]}")
def test_forward_parity_with_masks(shape):
    from flash_colbert import maxsim
    from flash_colbert.reference import maxsim_reference

    Nq, Nd, Lq, Ld, d = shape
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float16)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float16)

    q_mask = torch.rand(Nq, Lq, device="cuda") > 0.2
    d_mask = torch.rand(Nd, Ld, device="cuda") > 0.2
    # force at least one True per row
    q_mask[:, 0] = True
    d_mask[:, 0] = True

    fast = maxsim(Q, D, q_mask=q_mask, d_mask=d_mask).float()
    ref = maxsim_reference(Q.float(), D.float(), q_mask, d_mask)

    err = (fast - ref).abs().max().item()
    assert err < 5e-3 * max(1.0, ref.abs().max().item()), f"err={err}"


def test_empty_doc_mask_returns_zero():
    """A document whose mask is all-False should score 0 against every query."""
    from flash_colbert import maxsim

    Q = torch.randn(2, 8, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(3, 16, 128, device="cuda", dtype=torch.float16)
    d_mask = torch.ones(3, 16, device="cuda", dtype=torch.bool)
    d_mask[1] = False  # mask out doc #1 completely

    scores = maxsim(Q, D, d_mask=d_mask)
    assert torch.allclose(scores[:, 1], torch.zeros(2, device="cuda"))


def test_empty_query_mask_returns_zero():
    from flash_colbert import maxsim

    Q = torch.randn(3, 8, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(2, 16, 128, device="cuda", dtype=torch.float16)
    q_mask = torch.ones(3, 8, device="cuda", dtype=torch.bool)
    q_mask[1] = False

    scores = maxsim(Q, D, q_mask=q_mask)
    assert torch.allclose(scores[1, :], torch.zeros(2, device="cuda"))


def test_2d_inputs_return_scalar():
    from flash_colbert import maxsim

    Q = torch.randn(8, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(16, 128, device="cuda", dtype=torch.float16)
    s = maxsim(Q, D)
    assert s.dim() == 0, f"expected scalar, got {s.shape}"
