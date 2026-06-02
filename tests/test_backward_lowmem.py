"""Parity + memory for the low-memory (bf16, atomic-free) backward.

The lowmem path produces grads directly in the input dtype via fp32-register
accumulation in destination-owned kernels (one-hot matmul for both layouts).
It must match the unified (saved-argmax, fp32-then-cast) backend to bf16
rounding, and use strictly less training memory on the n_neg-inflated KD case.
"""

import pytest
import torch


@pytest.mark.cuda
@pytest.mark.parametrize(
    "shape",
    [
        (4, 8, 32, 128, 128),
        (16, 16, 32, 200, 128),
        (32, 32, 32, 300, 128),
        (8, 8, 128, 1024, 128),
        (4, 4, 32, 256, 48),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_lowmem_matches_unified_cross(shape, dtype, rel):
    from late_interaction_kernels.backward import maxsim_backward_lowmem, maxsim_backward_unified
    from late_interaction_kernels.forward import _run_forward

    Nq, Nd, Lq, Ld, d = shape
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=dtype)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=dtype)
    grad_s = torch.randn(Nq, Nd, device="cuda", dtype=torch.float32)

    _, argmax = _run_forward(Q, D, q_mask=None, d_mask=None, save_argmax=True)
    gQ_uni, gD_uni = maxsim_backward_unified(grad_s, Q, D, argmax, q_mask=None)
    gQ_lm, gD_lm = maxsim_backward_lowmem(grad_s, Q, D, argmax, None)

    assert gQ_lm.dtype == dtype and gD_lm.dtype == dtype
    assert rel(gQ_lm.float(), gQ_uni.float()) < 4e-3
    assert rel(gD_lm.float(), gD_uni.float()) < 1e-2  # bf16/fp16 store rounding


@pytest.mark.cuda
@pytest.mark.parametrize("shape", [(4, 8, 32, 128, 128), (8, 4, 32, 256, 64), (16, 8, 32, 1024, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_lowmem_matches_unified_kd(shape, dtype, rel):
    from late_interaction_kernels.backward import maxsim_backward_lowmem, maxsim_backward_unified
    from late_interaction_kernels.forward import _run_forward

    Nq, K, Lq, Ld, d = shape
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=dtype)
    D = torch.randn(Nq * K, Ld, d, device="cuda", dtype=dtype)
    grad_s = torch.randn(Nq, K, device="cuda", dtype=torch.float32)

    _, argmax = _run_forward(Q, D, q_mask=None, d_mask=None, save_argmax=True, kd_layout=True)
    gQ_uni, gD_uni = maxsim_backward_unified(grad_s, Q, D, argmax, q_mask=None, kd_layout=True)
    gQ_lm, gD_lm = maxsim_backward_lowmem(grad_s, Q, D, argmax, None, kd_layout=True)

    assert rel(gQ_lm.float(), gQ_uni.float()) < 4e-3
    assert rel(gD_lm.float(), gD_uni.float()) < 1e-2


@pytest.mark.cuda
@pytest.mark.parametrize("kd", [False, True])
def test_lowmem_matches_unified_qmask(kd, rel):
    from late_interaction_kernels.backward import maxsim_backward_lowmem, maxsim_backward_unified
    from late_interaction_kernels.forward import _run_forward

    Nq, K, Lq, Ld, d = 8, 8, 32, 256, 128
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.bfloat16)
    D = torch.randn(Nq * K if kd else K, Ld, d, device="cuda", dtype=torch.bfloat16)
    grad_s = torch.randn(Nq, K, device="cuda", dtype=torch.float32)
    q_mask_i8 = (torch.rand(Nq, Lq, device="cuda") > 0.3).to(torch.int8)

    _, argmax = _run_forward(Q, D, q_mask=q_mask_i8, d_mask=None, save_argmax=True, kd_layout=kd)
    gQ_uni, gD_uni = maxsim_backward_unified(grad_s, Q, D, argmax, q_mask=q_mask_i8, kd_layout=kd)
    gQ_lm, gD_lm = maxsim_backward_lowmem(grad_s, Q, D, argmax, q_mask_i8, kd_layout=kd)

    assert rel(gQ_lm.float(), gQ_uni.float()) < 4e-3
    assert rel(gD_lm.float(), gD_uni.float()) < 1e-2


@pytest.mark.cuda
def test_lowmem_end_to_end_autograd(rel):
    from late_interaction_kernels import maxsim

    Q_ref = torch.randn(8, 32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    D_ref = torch.randn(8, 200, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    Q_lm = Q_ref.detach().clone().requires_grad_(True)
    D_lm = D_ref.detach().clone().requires_grad_(True)

    maxsim(Q_ref, D_ref, backward="unified").sum().backward()
    maxsim(Q_lm, D_lm, backward="lowmem").sum().backward()

    assert rel(Q_lm.grad.float(), Q_ref.grad.float()) < 4e-3
    assert rel(D_lm.grad.float(), D_ref.grad.float()) < 1e-2


@pytest.mark.cuda
def test_lowmem_saves_memory_negatives():
    """4-D hard-negatives: lowmem (bf16 grads) << unified (fp32 grad_D + cast)."""
    from late_interaction_kernels import maxsim

    B, nn, Lq, Ld, d = 128, 16, 32, 512, 128

    def peak(method):
        Q = torch.randn(B, Lq, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        D = torch.randn(B, nn, Ld, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(2):
            Q.grad = None
            D.grad = None
            maxsim(Q, D, backward=method).sum().backward()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        Q.grad = None
        D.grad = None
        maxsim(Q, D, backward=method).sum().backward()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - base

    p_uni = peak("unified")
    p_lm = peak("lowmem")
    # grad_D fp32 buffer + bf16 transient is ~1.5 GB here; lowmem should cut
    # peak by a wide margin.
    assert p_lm < 0.8 * p_uni, f"lowmem peak {p_lm / 1e6:.0f}MB not < 0.8 * unified {p_uni / 1e6:.0f}MB"
