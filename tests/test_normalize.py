"""Parity tests for fused L2-normalize + MaxSim.

The kernel-native `normalize=True` path must match `maxsim(F.normalize(Q), F.normalize(D))`
within tensor-core ULP drift for realistic ColBERT / ColPali / ModernColBERT shapes.
"""

import pytest
import torch

pytestmark = pytest.mark.cuda


SHAPES = [
    (1, 4, 32, 64, 128),
    (8, 16, 32, 128, 128),
    (4, 4, 128, 1024, 128),
    (2, 2, 256, 256, 128),  # long-seq parity (shrunk from 1024² for CI speed)
    (2, 4, 32, 128, 512),
]
SHAPE_IDS = [f"Nq{s[0]}_Nd{s[1]}_Lq{s[2]}_Ld{s[3]}_d{s[4]}" for s in SHAPES]


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_normalize_forward_parity(shape, dtype, rel):
    """Fused normalize forward == explicit F.normalize + maxsim."""
    from late_interaction_kernels import maxsim

    Nq, Nd, Lq, Ld, d = shape
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=dtype)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=dtype)

    fused = maxsim(Q, D, normalize=True).float()
    Qn = torch.nn.functional.normalize(Q.float(), p=2, dim=-1, eps=1e-12).to(dtype)
    Dn = torch.nn.functional.normalize(D.float(), p=2, dim=-1, eps=1e-12).to(dtype)
    ref = maxsim(Qn, Dn).float()

    tol = 5e-3 if dtype == torch.float16 else 2e-2
    assert rel(fused, ref) < tol


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_normalize_forward_vs_reference(shape, rel):
    """Fused normalize forward == reference maxsim(normalize=True)."""
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.reference import maxsim_reference

    Nq, Nd, Lq, Ld, d = shape
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float32)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float32)

    fast = maxsim(Q.to(torch.bfloat16), D.to(torch.bfloat16), normalize=True).float()
    ref = maxsim_reference(Q, D, normalize=True)
    assert rel(fast, ref) < 2e-2


@pytest.mark.parametrize("shape", [(2, 3, 32, 128, 128), (1, 2, 16, 300, 256)])
def test_normalize_backward_parity(shape):
    """Gradients with fused normalize match autograd-through-F.normalize."""
    from late_interaction_kernels import maxsim

    Nq, Nd, Lq, Ld, d = shape
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float32, requires_grad=True)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float32, requires_grad=True)

    # Fused path.
    scores_fused = maxsim(Q, D, normalize=True)
    loss_fused = scores_fused.sum()
    gQ_fused, gD_fused = torch.autograd.grad(loss_fused, (Q, D), retain_graph=False)

    # Reference: F.normalize + maxsim.
    Q.grad = None
    D.grad = None
    Qn = torch.nn.functional.normalize(Q, p=2, dim=-1, eps=1e-12)
    Dn = torch.nn.functional.normalize(D, p=2, dim=-1, eps=1e-12)
    scores_ref = maxsim(Qn, Dn)
    loss_ref = scores_ref.sum()
    gQ_ref, gD_ref = torch.autograd.grad(loss_ref, (Q, D), retain_graph=False)

    tolQ = 5e-3 * max(1.0, gQ_ref.abs().max().item())
    tolD = 5e-3 * max(1.0, gD_ref.abs().max().item())
    assert (gQ_fused - gQ_ref).abs().max().item() < tolQ
    assert (gD_fused - gD_ref).abs().max().item() < tolD


def test_normalize_matches_pylate_formula():
    """Sanity: the fused kernel returns exactly what PyLate computes
    (F.normalize in python) up to ULP drift.
    """
    from late_interaction_kernels import maxsim

    torch.manual_seed(0)
    Q = torch.randn(4, 32, 128, device="cuda", dtype=torch.bfloat16)
    D = torch.randn(8, 300, 128, device="cuda", dtype=torch.bfloat16)

    fused = maxsim(Q, D, normalize=True).float()
    # What PyLate does today: normalize outside, call maxsim.
    Qn = torch.nn.functional.normalize(Q.float(), p=2, dim=-1).to(torch.bfloat16)
    Dn = torch.nn.functional.normalize(D.float(), p=2, dim=-1).to(torch.bfloat16)
    py_like = maxsim(Qn, Dn).float()

    assert (fused - py_like).abs().max().item() / max(1e-6, py_like.abs().max().item()) < 2e-2


def test_inference_normalize():
    """`maxsim_inference(normalize=True)` matches reference."""
    from late_interaction_kernels import maxsim_inference
    from late_interaction_kernels.reference import maxsim_reference

    Q = torch.randn(2, 32, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(4, 200, 128, device="cuda", dtype=torch.float16)

    fast = maxsim_inference(Q, D, normalize=True).float()
    ref = maxsim_reference(Q.float(), D.float(), normalize=True)
    assert (fast - ref).abs().max().item() / max(1e-6, ref.abs().max().item()) < 5e-3
