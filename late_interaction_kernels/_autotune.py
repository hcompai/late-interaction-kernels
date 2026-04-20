"""Autotune configs per GPU family.

Triton autotune runs each candidate once on the first call for a given key,
caches the winner, and reuses it forever after. We keep lists short — each
extra config costs one real launch.

Family rules of thumb (verified on H100 / A100 and conservative on the rest):
- Small ``d`` (≤ 128): prefer `BLOCK_Q=32-64, BLOCK_D=64-128`.
- Large ``d`` (≥ 512): shrink blocks so the fp16 `Q`/`D` tiles plus the fp32
  `S` tile fit in the SM's shared-memory budget.
- Hopper loves `num_stages ≥ 3` (warp specialization + async copy).
- Ampere / Ada are happiest with `num_stages=2`.

Per-family SRAM budgets (KiB of shared memory the kernel can actually use):
- Hopper (H100 / H200):       228
- Ampere (A100):              164
- Ampere consumer (3090, A10): 100
- Ada (L4, L40, RTX 4090):    100
- Unknown / older:             48 (safe floor)
"""

from __future__ import annotations

import triton

from ._utils import detect_gpu


def _small_d_hopper():
    return [
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_Q": 64, "BLOCK_D": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_Q": 64, "BLOCK_D": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_Q": 128, "BLOCK_D": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_Q": 128, "BLOCK_D": 128}, num_warps=8, num_stages=2),
    ]


def _small_d_ampere():
    """Works on A100, A10, A40, 3090, and is a safe default for Ada (L4, L40, 4090)."""
    return [
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_Q": 64, "BLOCK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 64, "BLOCK_D": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_Q": 128, "BLOCK_D": 64}, num_warps=8, num_stages=1),
    ]


def _large_d_configs():
    """Small-block configs for d ≥ 512 — fit any GPU, any SM."""
    return [
        triton.Config({"BLOCK_Q": 16, "BLOCK_D": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_Q": 16, "BLOCK_D": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 64}, num_warps=4, num_stages=2),
    ]


_SRAM_KIB_BY_FAMILY = {
    "hopper": 228,
    "a100": 164,
    "ampere": 100,
    "ada": 100,
    "generic": 48,
}


def forward_configs():
    gpu = detect_gpu()
    base = _large_d_configs()
    if gpu == "hopper":
        return base + _small_d_hopper()
    if gpu in ("a100", "ampere", "ada"):
        return base + _small_d_ampere()
    return base  # minimal safe shortlist for unknown GPUs


def prune_forward(configs, named_args, **kwargs):
    """Drop configs that overflow shared memory or are oversized for the problem."""
    Lq = named_args.get("Lq", 32)
    d = named_args.get("d", 128)
    gpu = detect_gpu()
    # Reserve 8 KiB for Triton scratch; the rest is ours.
    sram_budget = (_SRAM_KIB_BY_FAMILY.get(gpu, 48) - 8) * 1024

    keep = []
    for cfg in configs:
        bq, bd = cfg.kwargs["BLOCK_Q"], cfg.kwargs["BLOCK_D"]
        # fp16/bf16 Q tile + fp16/bf16 D tile + fp32 S tile.
        need = (bq * d + bd * d) * 2 + bq * bd * 4
        if need > sram_budget:
            continue
        if bq > 2 * Lq:
            continue
        keep.append(cfg)
    # Always return at least two configs so autotune has something to compare.
    return keep or configs[:2]
