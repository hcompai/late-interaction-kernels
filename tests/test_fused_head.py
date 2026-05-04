"""Parity tests for ``maxsim_from_hidden`` (fused D-side projection + MaxSim).

These are CUDA-only — the kernel is Triton and has no CPU fallback.

We also validate the pure-PyTorch reference on CPU so the parity oracle
itself is exercised in CI smoke runs.
"""

import pytest
import torch

from late_interaction_kernels.reference import (
    maxsim_from_hidden_reference,
    maxsim_reference,
)

# (Nq, Nd, Lq, Ld, d_model, d_out)
SHAPES = [
    (1, 4, 32, 64, 128, 64),
    (1, 8, 32, 128, 384, 96),
    (2, 8, 32, 256, 768, 128),
    (4, 16, 32, 200, 768, 128),  # ModernColBERT-sized
    (1, 32, 32, 180, 768, 128),  # reranking 32 docs
    (1, 4, 32, 2048, 768, 128),  # long-doc
]
SHAPE_IDS = [f"Nq{s[0]}_Nd{s[1]}_Lq{s[2]}_Ld{s[3]}_dm{s[4]}_do{s[5]}" for s in SHAPES]


def _golden(Q_proj, H_d, W, b, d_mask, normalize):
    """Slow but unambiguous: materialize D_proj, then call maxsim_reference."""
    D_proj = torch.nn.functional.linear(H_d.float(), W.float(), b.float() if b is not None else None)
    if normalize:
        D_proj = torch.nn.functional.normalize(D_proj, p=2, dim=-1, eps=1e-12)
    return maxsim_reference(Q_proj.float(), D_proj, d_mask=d_mask)


# --------------------------------------------------------------------------- #
# Reference-on-CPU smoke — runs everywhere, pins the oracle itself.           #
# --------------------------------------------------------------------------- #


def test_reference_matches_manual_on_cpu():
    Nq, Nd, Lq, Ld, d_model, d_out = 2, 3, 16, 32, 64, 32
    Q = torch.randn(Nq, Lq, d_out)
    H_d = torch.randn(Nd, Ld, d_model)
    W = torch.randn(d_out, d_model)
    b = torch.randn(d_out)

    ref = maxsim_from_hidden_reference(Q, H_d, W, b=b, normalize=True)
    manual = _golden(Q, H_d, W, b, d_mask=None, normalize=True)

    torch.testing.assert_close(ref, manual, atol=1e-5, rtol=1e-5)


def test_reference_without_normalize():
    Q = torch.randn(1, 8, 16)
    H_d = torch.randn(4, 12, 32)
    W = torch.randn(16, 32)
    ref = maxsim_from_hidden_reference(Q, H_d, W, normalize=False)
    manual = _golden(Q, H_d, W, b=None, d_mask=None, normalize=False)
    torch.testing.assert_close(ref, manual, atol=1e-5, rtol=1e-5)


def test_reference_with_d_mask_on_cpu():
    Q = torch.randn(2, 8, 16)
    H_d = torch.randn(3, 12, 32)
    W = torch.randn(16, 32)
    d_mask = torch.ones(3, 12, dtype=torch.bool)
    d_mask[0, 5:] = False  # first doc has only 5 real tokens
    d_mask[1, 10:] = False
    ref = maxsim_from_hidden_reference(Q, H_d, W, d_mask=d_mask, normalize=True)
    manual = _golden(Q, H_d, W, b=None, d_mask=d_mask, normalize=True)
    torch.testing.assert_close(ref, manual, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------- #
# CUDA parity tests                                                           #
# --------------------------------------------------------------------------- #

pytestmark_cuda = pytest.mark.cuda


@pytest.mark.cuda
@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_head_parity(shape, dtype, rel):
    from late_interaction_kernels import maxsim_from_hidden

    Nq, Nd, Lq, Ld, d_model, d_out = shape
    torch.manual_seed(0)
    H_d = torch.randn(Nd, Ld, d_model, device="cuda", dtype=dtype)
    W = torch.randn(d_out, d_model, device="cuda", dtype=dtype) * (1.0 / (d_model**0.5))
    b = torch.randn(d_out, device="cuda", dtype=dtype) * 0.01

    # Produce Q via the same projection so parity is against a realistic input.
    H_q = torch.randn(Nq, Lq, d_model, device="cuda", dtype=dtype)
    Q_proj = torch.nn.functional.linear(H_q.float(), W.float(), b.float())
    Q_proj = torch.nn.functional.normalize(Q_proj, dim=-1, eps=1e-12).to(dtype)

    ref = _golden(Q_proj, H_d, W, b, d_mask=None, normalize=True)
    out = maxsim_from_hidden(Q_proj, H_d, W, b=b, normalize=True)

    # bf16 / fp16 tensor-core matmul + two normalizations ≈ 5e-3 relative.
    assert rel(out, ref) < 5e-3, f"rel_err={rel(out, ref):.2e}"


@pytest.mark.cuda
def test_fused_head_no_bias():
    from late_interaction_kernels import maxsim_from_hidden

    Nq, Nd, Lq, Ld, d_model, d_out = 2, 4, 16, 64, 128, 32
    H_d = torch.randn(Nd, Ld, d_model, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(d_out, d_model, device="cuda", dtype=torch.bfloat16) * (1.0 / (d_model**0.5))
    Q = torch.randn(Nq, Lq, d_out, device="cuda", dtype=torch.bfloat16)
    Q = torch.nn.functional.normalize(Q.float(), dim=-1).to(torch.bfloat16)

    out = maxsim_from_hidden(Q, H_d, W, b=None, normalize=True)
    ref = _golden(Q, H_d, W, None, d_mask=None, normalize=True)
    assert (out - ref).abs().max().item() / max(1e-6, ref.abs().max().item()) < 5e-3


@pytest.mark.cuda
def test_fused_head_with_d_mask():
    from late_interaction_kernels import maxsim_from_hidden

    Nq, Nd, Lq, Ld, d_model, d_out = 1, 4, 32, 128, 256, 64
    H_d = torch.randn(Nd, Ld, d_model, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(d_out, d_model, device="cuda", dtype=torch.bfloat16) * (1.0 / (d_model**0.5))
    Q = torch.randn(Nq, Lq, d_out, device="cuda", dtype=torch.bfloat16)
    Q = torch.nn.functional.normalize(Q.float(), dim=-1).to(torch.bfloat16)
    d_mask = torch.ones(Nd, Ld, dtype=torch.bool, device="cuda")
    d_mask[:, Ld // 2 :] = False

    out = maxsim_from_hidden(Q, H_d, W, d_mask=d_mask, normalize=True)
    ref = _golden(Q, H_d, W, None, d_mask=d_mask, normalize=True)
    assert (out - ref).abs().max().item() / max(1e-6, ref.abs().max().item()) < 5e-3


@pytest.mark.cuda
def test_fused_head_matches_unfused_maxsim():
    """The fused kernel must match the canonical `F.linear + normalize + maxsim` path."""
    from late_interaction_kernels import maxsim_from_hidden, maxsim_inference

    Nd, Lq, Ld, d_model, d_out = 8, 32, 200, 768, 128
    H_d = torch.randn(Nd, Ld, d_model, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(d_out, d_model, device="cuda", dtype=torch.bfloat16) * (1.0 / (d_model**0.5))
    b = torch.randn(d_out, device="cuda", dtype=torch.bfloat16) * 0.01
    Q = torch.randn(Lq, d_out, device="cuda", dtype=torch.bfloat16)
    Q = torch.nn.functional.normalize(Q.float(), dim=-1).to(torch.bfloat16)

    D_proj = torch.nn.functional.linear(H_d.float(), W.float(), b.float())
    D_proj = torch.nn.functional.normalize(D_proj, dim=-1).to(torch.bfloat16)

    fused = maxsim_from_hidden(Q, H_d, W, b=b, normalize=True)
    unfused = maxsim_inference(Q, D_proj)

    assert (fused.squeeze(0) - unfused).abs().max().item() / max(1e-6, unfused.abs().max().item()) < 7e-3


@pytest.mark.cuda
def test_fused_head_rejects_bad_shapes():
    from late_interaction_kernels import maxsim_from_hidden

    Q = torch.randn(4, 32, 64, device="cuda", dtype=torch.bfloat16)
    H_d = torch.randn(8, 128, 256, device="cuda", dtype=torch.bfloat16)
    W_bad = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)  # wrong d_out

    with pytest.raises(ValueError, match="W must be"):
        maxsim_from_hidden(Q, H_d, W_bad)
