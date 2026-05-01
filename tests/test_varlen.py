"""Varlen (packed) MaxSim parity tests."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.cuda


def _build_varlen(seqlens, d, dtype=torch.float16, device="cuda"):
    total = sum(seqlens)
    data = torch.randn(total, d, device=device, dtype=dtype)
    cu = torch.zeros(len(seqlens) + 1, device=device, dtype=torch.int32)
    cu[1:] = torch.tensor(seqlens, device=device, dtype=torch.int32).cumsum(0)
    return data, cu


@pytest.mark.parametrize(
    "q_lens,d_lens,d",
    [
        ([32, 32], [64, 128, 32, 200], 128),
        ([5, 10, 1], [50, 60, 1], 128),
        ([32], [300] * 4, 128),
        ([16, 48, 32, 8], [128, 256, 512, 32, 200], 256),
    ],
)
def test_varlen_parity(q_lens, d_lens, d):
    from late_interaction_kernels import maxsim_varlen
    from late_interaction_kernels.reference import maxsim_reference_varlen

    Qp, cu_q = _build_varlen(q_lens, d)
    Dp, cu_d = _build_varlen(d_lens, d)

    fast = maxsim_varlen(Qp, Dp, cu_q, cu_d).float()
    ref = maxsim_reference_varlen(Qp.float(), Dp.float(), cu_q, cu_d)

    err = (fast - ref).abs().max().item()
    denom = max(1.0, ref.abs().max().item())
    assert err / denom < 5e-3, f"err={err} denom={denom}"


def test_varlen_empty_sequence_is_zero():
    from late_interaction_kernels import maxsim_varlen

    Qp, cu_q = _build_varlen([5, 0, 3], 128)  # middle query has 0 tokens
    Dp, cu_d = _build_varlen([8, 16], 128)
    scores = maxsim_varlen(Qp, Dp, cu_q, cu_d)
    assert torch.allclose(scores[1], torch.zeros(2, device="cuda"))


def test_varlen_matches_padded_path():
    """Pack the docs of `maxsim` into cu_seqlens and check scores match."""
    from late_interaction_kernels import maxsim, maxsim_varlen

    Nq, Nd, Lq, Ld, d = 4, 8, 32, 256, 128
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float16)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float16)

    Qp = Q.reshape(-1, d)
    Dp = D.reshape(-1, d)
    cu_q = torch.arange(0, (Nq + 1) * Lq, Lq, device="cuda", dtype=torch.int32)
    cu_d = torch.arange(0, (Nd + 1) * Ld, Ld, device="cuda", dtype=torch.int32)

    a = maxsim(Q, D).float()
    b = maxsim_varlen(Qp, Dp, cu_q, cu_d)
    assert (a - b).abs().max().item() < 1e-3


# -----------------------------------------------------------------------------
# Backward parity: check grad_Q and grad_D match the padded autograd path by
# building two equivalent inputs — one varlen-packed, one padded with masks —
# and comparing the gradients after a scalar loss.
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "q_lens,d_lens,d",
    [
        ([32, 32], [64, 128, 32, 200], 128),
        ([32], [300] * 4, 128),
        ([16, 48, 32, 8], [128, 256, 200], 128),
    ],
)
def test_varlen_backward_matches_padded(q_lens, d_lens, d):
    from late_interaction_kernels import maxsim, maxsim_varlen

    Nq = len(q_lens)
    Nd = len(d_lens)
    max_lq = max(q_lens)
    max_ld = max(d_lens)

    torch.manual_seed(0)
    # Build packed tensors.
    Qp = torch.randn(sum(q_lens), d, device="cuda", dtype=torch.float32)
    Dp = torch.randn(sum(d_lens), d, device="cuda", dtype=torch.float32)
    cu_q = torch.zeros(Nq + 1, device="cuda", dtype=torch.int32)
    cu_q[1:] = torch.tensor(q_lens, device="cuda", dtype=torch.int32).cumsum(0)
    cu_d = torch.zeros(Nd + 1, device="cuda", dtype=torch.int32)
    cu_d[1:] = torch.tensor(d_lens, device="cuda", dtype=torch.int32).cumsum(0)

    # Padded equivalents.
    Q_pad = torch.zeros(Nq, max_lq, d, device="cuda", dtype=torch.float32)
    D_pad = torch.zeros(Nd, max_ld, d, device="cuda", dtype=torch.float32)
    q_mask = torch.zeros(Nq, max_lq, device="cuda", dtype=torch.bool)
    d_mask = torch.zeros(Nd, max_ld, device="cuda", dtype=torch.bool)
    for i, lq in enumerate(q_lens):
        Q_pad[i, :lq] = Qp[cu_q[i] : cu_q[i + 1]]
        q_mask[i, :lq] = True
    for j, ld in enumerate(d_lens):
        D_pad[j, :ld] = Dp[cu_d[j] : cu_d[j + 1]]
        d_mask[j, :ld] = True

    # Varlen autograd path.
    Qp_g = Qp.clone().requires_grad_(True)
    Dp_g = Dp.clone().requires_grad_(True)
    s_var = maxsim_varlen(Qp_g, Dp_g, cu_q, cu_d)

    # Padded autograd path.
    Qp2 = Q_pad.clone().requires_grad_(True)
    Dp2 = D_pad.clone().requires_grad_(True)
    s_pad = maxsim(Qp2, Dp2, q_mask=q_mask, d_mask=d_mask)

    # Same upstream gradient on every score.
    g = torch.randn_like(s_var)
    s_var.backward(g)
    s_pad.backward(g)

    # Unpack padded grads to packed layout for comparison.
    gQ_pad = torch.zeros_like(Qp)
    gD_pad = torch.zeros_like(Dp)
    for i, lq in enumerate(q_lens):
        gQ_pad[cu_q[i] : cu_q[i + 1]] = Qp2.grad[i, :lq]
    for j, ld in enumerate(d_lens):
        gD_pad[cu_d[j] : cu_d[j + 1]] = Dp2.grad[j, :ld]

    err_Q = (Qp_g.grad - gQ_pad).abs().max().item()
    err_D = (Dp_g.grad - gD_pad).abs().max().item()
    denom_Q = max(1.0, gQ_pad.abs().max().item())
    denom_D = max(1.0, gD_pad.abs().max().item())
    assert err_Q / denom_Q < 1e-4, f"grad_Q err={err_Q}"
    assert err_D / denom_D < 1e-4, f"grad_D err={err_D}"


def test_varlen_backward_requires_grad_gate():
    """No argmax is saved when neither input requires grad — sanity check that
    the inference path still returns correct scores."""
    from late_interaction_kernels import maxsim_varlen
    from late_interaction_kernels.reference import maxsim_reference_varlen

    Qp, cu_q = _build_varlen([12, 8, 20], 128)
    Dp, cu_d = _build_varlen([64, 32, 128, 16], 128)
    s_ref = maxsim_reference_varlen(Qp.float(), Dp.float(), cu_q, cu_d)
    s_fast = maxsim_varlen(Qp, Dp, cu_q, cu_d).float()
    assert (s_fast - s_ref).abs().max().item() < 5e-3
