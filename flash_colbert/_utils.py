"""Small helpers shared across kernels."""

from __future__ import annotations

import functools

import torch


def next_pow2(x: int) -> int:
    """Smallest power of two >= x. `next_pow2(0)` returns 1."""
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


@functools.lru_cache(maxsize=1)
def detect_gpu() -> str:
    """Return a short GPU family string: 'hopper' | 'a100' | 'ada' | 'ampere' | 'generic'."""
    if not torch.cuda.is_available():
        return "generic"
    name = torch.cuda.get_device_name().lower()
    if "h100" in name or "h200" in name:
        return "hopper"
    if "a100" in name:
        return "a100"
    if "l4" in name or "l40" in name or "rtx 40" in name:
        return "ada"
    if "3090" in name or "a10" in name or "a40" in name:
        return "ampere"
    return "generic"


def ensure_contiguous_last(x: torch.Tensor) -> torch.Tensor:
    """Make sure the last dim is contiguous — cheap path for most inputs."""
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


def pick_compute_dtype(Q: torch.Tensor, D: torch.Tensor) -> torch.dtype:
    """Pick the compute dtype for `tl.dot`.

    We honor user intent: if both tensors are fp16/bf16, dot runs in that dtype
    with fp32 accumulator. If either is fp32 we fall back to fp16 on the tile
    (fp32 GEMM doesn't go through tensor cores on H100 anyway).
    """
    if Q.dtype == torch.bfloat16 or D.dtype == torch.bfloat16:
        return torch.bfloat16
    return torch.float16
