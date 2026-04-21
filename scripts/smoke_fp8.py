"""Smoke-test FP8 MaxSim inference (parity vs bf16 kernel, + speed)."""

from __future__ import annotations

import time

import torch

from late_interaction_kernels import (
    maxsim_inference,
    maxsim_inference_fp8,
    quantize_fp8_per_tensor,
    quantize_fp8_per_token,
)


def _make(Nq=4, Nd=64, Lq=32, Ld=256, d=128, dtype=torch.bfloat16, device="cuda"):
    Q = torch.nn.functional.normalize(torch.randn(Nq, Lq, d, device=device, dtype=dtype), dim=-1)
    D = torch.nn.functional.normalize(torch.randn(Nd, Ld, d, device=device, dtype=dtype), dim=-1)
    return Q, D


def parity(scale_q="tensor", scale_d="tensor"):
    torch.manual_seed(0)
    Q, D = _make()
    ref = maxsim_inference(Q, D)

    Q_fp8, sQ = quantize_fp8_per_token(Q) if scale_q == "token" else quantize_fp8_per_tensor(Q)
    D_fp8, sD = quantize_fp8_per_token(D) if scale_d == "token" else quantize_fp8_per_tensor(D)
    out = maxsim_inference_fp8(Q_fp8, D_fp8, scale_Q=sQ, scale_D=sD)

    diff = (out.float() - ref.float()).abs()
    print(
        f"[parity scale_q={scale_q} scale_d={scale_d}] "
        f"mean={diff.mean().item():.4e} max={diff.max().item():.4e} "
        f"rel={(diff / ref.float().abs().clamp_min(1e-6)).mean().item():.3%}"
    )


def bench():
    torch.manual_seed(0)
    Q, D = _make(Nq=8, Nd=4096, Lq=32, Ld=256, d=128)
    # warm
    for _ in range(3):
        _ = maxsim_inference(Q, D)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        _ = maxsim_inference(Q, D)
    torch.cuda.synchronize()
    t_bf16 = (time.perf_counter() - t0) / 20 * 1000

    Q_fp8, sQ = quantize_fp8_per_token(Q)
    D_fp8, sD = quantize_fp8_per_token(D)
    for _ in range(3):
        _ = maxsim_inference_fp8(Q_fp8, D_fp8, scale_Q=sQ, scale_D=sD)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        _ = maxsim_inference_fp8(Q_fp8, D_fp8, scale_Q=sQ, scale_D=sD)
    torch.cuda.synchronize()
    t_fp8 = (time.perf_counter() - t0) / 20 * 1000
    print(
        f"[bench Nq=8 Nd=4096 Lq=32 Ld=256 d=128] bf16={t_bf16:.2f}ms  fp8={t_fp8:.2f}ms  speedup={t_bf16 / t_fp8:.2f}x"
    )


if __name__ == "__main__":
    print("device:", torch.cuda.get_device_name(0))
    parity("tensor", "tensor")
    parity("token", "tensor")
    parity("tensor", "token")
    parity("token", "token")
    bench()
