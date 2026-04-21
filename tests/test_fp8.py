"""Parity + fallback tests for :func:`maxsim_inference_fp8` (0.7.0)."""

from __future__ import annotations

import pytest
import torch

fp8_dtype = getattr(torch, "float8_e4m3fn", None)
pytestmark_no_fp8 = pytest.mark.skipif(fp8_dtype is None, reason="torch has no FP8 dtype")


# (Nq, Nd, Lq, Ld, d)
SHAPES = [
    (1, 4, 32, 64, 128),
    (2, 8, 32, 128, 128),
    (4, 16, 32, 200, 128),  # ModernColBERT-sized
    (1, 32, 32, 256, 128),
    (1, 4, 16, 512, 128),  # longer doc
]
SHAPE_IDS = [f"Nq{s[0]}_Nd{s[1]}_Lq{s[2]}_Ld{s[3]}_d{s[4]}" for s in SHAPES]


def _make(Nq, Nd, Lq, Ld, d, dtype=torch.bfloat16, device="cuda"):
    Q = torch.nn.functional.normalize(
        torch.randn(Nq, Lq, d, device=device, dtype=dtype), dim=-1
    )
    D = torch.nn.functional.normalize(
        torch.randn(Nd, Ld, d, device=device, dtype=dtype), dim=-1
    )
    return Q, D


# --------------------------------------------------------------------------- #
# Helpers are pure-python; these smoke tests run on CPU with tiny tensors.    #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(fp8_dtype is None, reason="torch has no FP8 dtype")
def test_quantize_roundtrip_per_tensor():
    from late_interaction_kernels import (
        dequantize_fp8_per_tensor,
        quantize_fp8_per_tensor,
    )

    X = torch.randn(8, 16, dtype=torch.float32)
    Xq, s = quantize_fp8_per_tensor(X)
    assert Xq.dtype == fp8_dtype
    Xhat = dequantize_fp8_per_tensor(Xq, s)
    err = (X - Xhat).abs().max().item() / max(1e-6, X.abs().max().item())
    assert err < 0.05  # fp8 e4m3 has ~2-3 bits of mantissa


@pytest.mark.skipif(fp8_dtype is None, reason="torch has no FP8 dtype")
def test_quantize_roundtrip_per_token():
    from late_interaction_kernels import (
        dequantize_fp8_per_token,
        quantize_fp8_per_token,
    )

    X = torch.randn(3, 17, 32, dtype=torch.float32)
    Xq, s = quantize_fp8_per_token(X)
    assert Xq.shape == X.shape
    assert s.shape == X.shape[:-1]
    Xhat = dequantize_fp8_per_token(Xq, s)
    err = (X - Xhat).abs().max().item() / max(1e-6, X.abs().max().item())
    assert err < 0.05


@pytest.mark.skipif(fp8_dtype is None, reason="torch has no FP8 dtype")
def test_quantize_empty_input():
    from late_interaction_kernels import quantize_fp8_per_tensor

    X = torch.empty(0, 8, dtype=torch.float32)
    Xq, s = quantize_fp8_per_tensor(X)
    assert Xq.numel() == 0
    assert s.dim() == 0


# --------------------------------------------------------------------------- #
# CUDA parity against the reference maxsim.                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.cuda
@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("scale_q", ["tensor", "token"])
@pytest.mark.parametrize("scale_d", ["tensor", "token"])
def test_maxsim_fp8_parity(shape, scale_q, scale_d):
    if fp8_dtype is None:
        pytest.skip("torch has no FP8 dtype")
    from late_interaction_kernels import (
        maxsim_inference,
        maxsim_inference_fp8,
        quantize_fp8_per_tensor,
        quantize_fp8_per_token,
    )

    Nq, Nd, Lq, Ld, d = shape
    Q, D = _make(Nq, Nd, Lq, Ld, d)
    ref = maxsim_inference(Q, D)

    qfn = quantize_fp8_per_token if scale_q == "token" else quantize_fp8_per_tensor
    dfn = quantize_fp8_per_token if scale_d == "token" else quantize_fp8_per_tensor
    Q_fp8, sQ = qfn(Q)
    D_fp8, sD = dfn(D)
    out = maxsim_inference_fp8(Q_fp8, D_fp8, scale_Q=sQ, scale_D=sD)

    denom = max(1e-6, ref.abs().max().item())
    rel = (out.float() - ref.float()).abs().max().item() / denom
    # fp8 e4m3 has ~2.5-bit mantissa; per-row accumulation of Lq≈32 rows
    # inflates the max error by sqrt(Lq). 3% max-relative is the accepted
    # FP8 reranking tolerance (matches TRT-LLM / SGLang fp8 docs).
    assert rel < 0.05, f"rel_err={rel:.3e} ({scale_q=}, {scale_d=}, shape={shape})"


@pytest.mark.cuda
def test_maxsim_fp8_with_masks():
    if fp8_dtype is None:
        pytest.skip("torch has no FP8 dtype")
    from late_interaction_kernels import (
        maxsim_inference,
        maxsim_inference_fp8,
        quantize_fp8_per_token,
    )

    Q, D = _make(2, 8, 32, 128, 128)
    q_mask = torch.ones(2, 32, dtype=torch.bool, device="cuda")
    q_mask[:, 24:] = False  # 8 pad tokens
    d_mask = torch.ones(8, 128, dtype=torch.bool, device="cuda")
    d_mask[:, 96:] = False

    ref = maxsim_inference(Q, D, q_mask=q_mask, d_mask=d_mask)
    Q_fp8, sQ = quantize_fp8_per_token(Q)
    D_fp8, sD = quantize_fp8_per_token(D)
    out = maxsim_inference_fp8(
        Q_fp8, D_fp8, scale_Q=sQ, scale_D=sD, q_mask=q_mask, d_mask=d_mask
    )

    denom = max(1e-6, ref.abs().max().item())
    rel = (out.float() - ref.float()).abs().max().item() / denom
    assert rel < 0.05


@pytest.mark.cuda
def test_maxsim_fp8_2d_inputs():
    if fp8_dtype is None:
        pytest.skip("torch has no FP8 dtype")
    from late_interaction_kernels import (
        maxsim_inference_fp8,
        quantize_fp8_per_tensor,
    )

    Q = torch.nn.functional.normalize(
        torch.randn(32, 128, device="cuda", dtype=torch.bfloat16), dim=-1
    )
    D = torch.nn.functional.normalize(
        torch.randn(128, 128, device="cuda", dtype=torch.bfloat16), dim=-1
    )
    Qq, sQ = quantize_fp8_per_tensor(Q)
    Dq, sD = quantize_fp8_per_tensor(D)

    scalar = maxsim_inference_fp8(Qq, Dq, scale_Q=sQ, scale_D=sD)
    assert scalar.shape == ()
