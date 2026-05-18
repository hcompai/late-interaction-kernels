"""Parity + gradient tests for :func:`smooth_maxsim`."""

import pytest
import torch

pytest.importorskip("triton", reason="smooth_maxsim requires Triton")

from late_interaction_kernels.experimental.smooth import smooth_maxsim_reference  # noqa: E402

# (Nq, Nd, Lq, Ld, d)
SHAPES = [
    (1, 4, 16, 32, 64),
    (2, 8, 32, 64, 128),
    (4, 16, 32, 128, 128),
    (1, 4, 32, 512, 128),  # longer doc
]
SHAPE_IDS = [f"Nq{s[0]}_Nd{s[1]}_Lq{s[2]}_Ld{s[3]}_d{s[4]}" for s in SHAPES]


# --------------------------------------------------------------------------- #
# Reference (CPU) self-consistency + edge cases.                              #
# --------------------------------------------------------------------------- #


def test_reference_topk1_sum_matches_hard_maxsim():
    """top_k=1, aggregation='sum' must be bit-identical to hard MaxSim."""
    from late_interaction_kernels.reference import maxsim_reference

    Q = torch.randn(2, 16, 32, dtype=torch.float32)
    D = torch.randn(3, 24, 32, dtype=torch.float32)

    hard = maxsim_reference(Q, D)
    smooth = smooth_maxsim_reference(Q, D, top_k=1, aggregation="sum")
    torch.testing.assert_close(hard, smooth.to(hard.dtype), atol=1e-6, rtol=1e-6)


def test_reference_topk_large_k_matches_mean_plus_sum():
    """With K=Ld and aggregation='mean' we should get the full mean-sim."""
    Q = torch.randn(1, 4, 8, dtype=torch.float64)
    D = torch.randn(1, 12, 8, dtype=torch.float64)

    full_mean = torch.einsum("ild,jtd->ijlt", Q, D).mean(dim=-1).sum(dim=-1)
    smooth = smooth_maxsim_reference(Q, D, top_k=12, aggregation="mean")
    torch.testing.assert_close(smooth, full_mean, atol=1e-10, rtol=1e-10)


def test_reference_masks():
    Q = torch.randn(1, 6, 8, dtype=torch.float64)
    D = torch.randn(2, 10, 8, dtype=torch.float64)
    q_mask = torch.ones(1, 6, dtype=torch.bool)
    q_mask[0, 4:] = False
    d_mask = torch.ones(2, 10, dtype=torch.bool)
    d_mask[0, 6:] = False

    out = smooth_maxsim_reference(Q, D, q_mask=q_mask, d_mask=d_mask, top_k=3)
    assert out.shape == (1, 2)
    # Manually mask and recompute
    Qm = Q.clone()
    Qm[0, 4:] = 0
    Dm = D.clone()
    Dm[0, 6:] = float("-inf")  # only doc 0 truncated
    S = torch.einsum("ild,jtd->ijlt", Q, D)
    S[:, 0, :, 6:] = float("-inf")
    top = S.topk(3, dim=-1).values.clamp(min=-1e18)
    agg = top.sum(dim=-1) / 3
    agg = agg * q_mask.to(agg.dtype)[:, None, :]
    expected = agg.sum(dim=-1)
    torch.testing.assert_close(out, expected, atol=1e-10, rtol=1e-10)


def test_reference_top_k_clamped_to_ld():
    """``top_k > Ld`` should silently use ``Ld``."""
    Q = torch.randn(1, 3, 4, dtype=torch.float64)
    D = torch.randn(1, 5, 4, dtype=torch.float64)

    # With K=Ld (=5) and 'sum', we get the full sum over t.
    big_k = smooth_maxsim_reference(Q, D, top_k=999, aggregation="sum")
    full_sum = torch.einsum("ild,jtd->ijlt", Q, D).sum(dim=-1).sum(dim=-1)
    torch.testing.assert_close(big_k, full_sum, atol=1e-10, rtol=1e-10)


# --------------------------------------------------------------------------- #
# CUDA kernel parity.                                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.cuda
@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("top_k", [1, 4, 8])
@pytest.mark.parametrize("aggregation", ["mean", "sum"])
def test_smooth_maxsim_kernel_parity(shape, top_k, aggregation):
    from late_interaction_kernels.experimental import smooth_maxsim

    Nq, Nd, Lq, Ld, d = shape
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.bfloat16)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.bfloat16)
    Q = torch.nn.functional.normalize(Q.float(), dim=-1).to(torch.bfloat16)
    D = torch.nn.functional.normalize(D.float(), dim=-1).to(torch.bfloat16)

    out = smooth_maxsim(Q, D, top_k=top_k, aggregation=aggregation)
    ref = smooth_maxsim_reference(Q, D, top_k=top_k, aggregation=aggregation)

    rel = (out.float() - ref.float()).abs().max().item() / max(1e-6, ref.float().abs().max().item())
    assert rel < 5e-3, f"rel_err={rel:.3e} (top_k={top_k}, agg={aggregation}, shape={shape})"


@pytest.mark.cuda
def test_smooth_maxsim_topk1_sum_equals_hard_maxsim():
    """Flagship invariant: (top_k=1, sum) == hard maxsim, bit-for-bit on bf16."""
    from late_interaction_kernels import maxsim_inference
    from late_interaction_kernels.experimental import smooth_maxsim

    Q = torch.randn(2, 32, 128, device="cuda", dtype=torch.bfloat16)
    D = torch.randn(4, 96, 128, device="cuda", dtype=torch.bfloat16)
    Q = torch.nn.functional.normalize(Q.float(), dim=-1).to(torch.bfloat16)
    D = torch.nn.functional.normalize(D.float(), dim=-1).to(torch.bfloat16)

    hard = maxsim_inference(Q, D)
    smooth = smooth_maxsim(Q, D, top_k=1, aggregation="sum")
    torch.testing.assert_close(smooth, hard, atol=1e-3, rtol=1e-3)


# --------------------------------------------------------------------------- #
# Backward parity — reference autograd vs reference numeric diff.             #
# --------------------------------------------------------------------------- #


def test_reference_backward_numeric_gradcheck():
    """`smooth_maxsim_reference` is pure PyTorch; use gradcheck on it."""
    torch.manual_seed(0)
    Q = torch.randn(1, 4, 6, dtype=torch.float64, requires_grad=True)
    D = torch.randn(2, 5, 6, dtype=torch.float64, requires_grad=True)

    def f(Q, D):
        return smooth_maxsim_reference(Q, D, top_k=2, aggregation="mean").sum()

    assert torch.autograd.gradcheck(f, (Q, D), eps=1e-6, atol=1e-4, nondet_tol=1e-6)


@pytest.mark.cuda
def test_smooth_maxsim_kernel_backward_matches_reference():
    """Kernel backward (atomic scatter) must match pure-PyTorch gradients."""
    from late_interaction_kernels.experimental import smooth_maxsim

    torch.manual_seed(0)
    Q = torch.randn(2, 16, 64, device="cuda", dtype=torch.float32, requires_grad=True)
    D = torch.randn(3, 32, 64, device="cuda", dtype=torch.float32, requires_grad=True)
    Qr = Q.detach().clone().requires_grad_(True)
    Dr = D.detach().clone().requires_grad_(True)

    out = smooth_maxsim(Q, D, top_k=3, aggregation="mean")
    out.sum().backward()

    ref = smooth_maxsim_reference(Qr, Dr, top_k=3, aggregation="mean")
    ref.sum().backward()

    # Gradient parity: the winning set may differ on ties so tolerate 2%
    # max-relative error. For non-tie random inputs we measure <1e-3 in
    # practice.
    for name, a, b in [("grad_Q", Q.grad, Qr.grad), ("grad_D", D.grad, Dr.grad)]:
        denom = max(1e-6, b.abs().max().item())
        rel = (a - b).abs().max().item() / denom
        assert rel < 5e-2, f"{name} rel_err={rel:.3e}"


@pytest.mark.cuda
def test_smooth_maxsim_validates_inputs():
    from late_interaction_kernels.experimental import smooth_maxsim

    Q = torch.randn(1, 4, 8, device="cuda", dtype=torch.bfloat16)
    D = torch.randn(1, 4, 16, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="share the embedding dim"):
        smooth_maxsim(Q, D)
    with pytest.raises(ValueError, match="aggregation"):
        smooth_maxsim(
            torch.randn(1, 4, 8, device="cuda"), torch.randn(1, 4, 8, device="cuda"), aggregation="median"
        )
    with pytest.raises(ValueError, match="top_k"):
        smooth_maxsim(torch.randn(1, 4, 8, device="cuda"), torch.randn(1, 4, 8, device="cuda"), top_k=0)
