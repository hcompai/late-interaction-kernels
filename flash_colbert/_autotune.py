"""Autotune configs per GPU family.

Triton autotune runs each candidate once on the first call for a given key,
caches the winner, and reuses it forever after. Keep the lists small — each
extra config costs one real launch.

Config heuristic (from flash-attention + flash-maxsim lineage, tuned further
with H100 microbenchmarks):

- Small d (<=128): prefer BLOCK_Q=32-64, BLOCK_D=64-128.
- Large d (>=512): shrink blocks to keep shared memory under the roof
  (SRAM_budget ~= 228 KiB on H100, ~= 164 KiB on A100).
- Hopper loves num_stages>=3 (warp specialization + async copy), A100 is
  happiest with num_stages=2.
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


def _small_d_a100():
    return [
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_Q": 64, "BLOCK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 64, "BLOCK_D": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_Q": 128, "BLOCK_D": 64}, num_warps=8, num_stages=1),
    ]


def _large_d_configs():
    return [
        triton.Config({"BLOCK_Q": 16, "BLOCK_D": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_Q": 16, "BLOCK_D": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 64}, num_warps=4, num_stages=2),
    ]


def forward_configs():
    gpu = detect_gpu()
    base = _large_d_configs()
    if gpu == "hopper":
        return base + _small_d_hopper()
    if gpu == "a100":
        return base + _small_d_a100()
    return base + _small_d_a100()  # conservative default


def prune_forward(configs, named_args, **kwargs):
    """Drop configs that exceed shared memory or are larger than the problem."""
    Lq = named_args.get("Lq", 32)
    d = named_args.get("d", 128)
    # Reserve 8 KiB for Triton scratch; the rest is ours.
    sram_budget = 224 * 1024
    keep = []
    for cfg in configs:
        bq, bd = cfg.kwargs["BLOCK_Q"], cfg.kwargs["BLOCK_D"]
        # fp16 Q tile + fp16 D tile + fp32 S tile
        need = (bq * d + bd * d) * 2 + bq * bd * 4
        if need > sram_budget:
            continue
        if bq > 2 * Lq:
            continue
        keep.append(cfg)
    return keep or configs[:2]
