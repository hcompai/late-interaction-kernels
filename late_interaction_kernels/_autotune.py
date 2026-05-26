"""Autotune configs per GPU family.

Triton autotune runs each candidate once on the first call for a given key,
caches the winner, and reuses it forever after. We keep lists short — each
extra config costs one real launch.

Family rules of thumb (verified on H100 / A100 and conservative on the rest):
- Small ``d`` (≤ 128): prefer `BLOCK_Q=32-64, BLOCK_D=64-128`.
- Large ``d`` (≥ 512): shrink blocks so the fp16 `Q`/`D` tiles plus the fp32
  `S` tile fit in the SM's shared-memory budget.
- Hopper loves `num_stages=3` (3-deep async pipeline, one warp group).
- Ampere / Ada are happiest with `num_stages=2`.

Per-family SRAM budgets (KiB of shared memory the kernel can actually use):
- Hopper (H100 / H200):       228
- Ampere (A100):              164
- Ampere consumer (3090, A10): 100
- Ada (L4, L40, RTX 4090):    100
- Unknown / older:             48 (safe floor)
"""

import re

import triton

from late_interaction_kernels._utils import detect_gpu

# Persistent on-disk best-config cache landed in Triton 3.4 (the ``cache_results``
# kwarg on ``triton.autotune``). On older Triton the kwarg doesn't exist and
# passing it would TypeError, so we gate on the version. Effect: first run on
# a fresh machine pays the usual benchmark cost; subsequent processes (CI
# runs, second training epoch, new shell) load the winner from
# ``$TRITON_CACHE_DIR`` and skip the sweep. Triton's cache key already
# includes its version, backend hash, kernel source hash, env-var hash and
# the config list, so it invalidates on its own when any of those change.
_match = re.match(r"(\d+)\.(\d+)", triton.__version__)
_TRITON_VERSION: tuple[int, int] = (int(_match.group(1)), int(_match.group(2)))
_HAS_DISK_CACHE = _TRITON_VERSION >= (3, 4)


def autotune_kwargs() -> dict:
    """Shared ``triton.autotune`` kwargs every LIK kernel injects via ``**``.

    Today only carries ``cache_results=True`` (when supported). If Triton
    grows more first-class autotune knobs in the future (e.g. global cache
    key prefix, remote cache backend) this is where they go so every kernel
    inherits them at once.
    """
    return {"cache_results": True} if _HAS_DISK_CACHE else {}


def _cfg(kwargs, *, num_warps, num_stages):
    """Build a ``triton.Config`` with no warp-spec kwargs (see module docstring)."""
    return triton.Config(kwargs, num_warps=num_warps, num_stages=num_stages)


def _small_d_hopper():
    """H100 winner across our bench shapes is ``BLOCK_Q=64, BLOCK_D=64,
    num_warps=4, num_stages=3``:

    1. WGMMA on bf16 uses ``m64nNk16`` natively, so ``BLOCK_Q=64`` fills
       one warp group's M slot exactly — no masked rows wasted.
    2. ``num_warps=4`` (= one warp group) gets ``tl.dot`` lowered directly
       to ``wgmma.mma_async`` without extra orchestration.
    3. With ``Nq * Nd`` programs on the grid, small per-CTA work means
       more concurrent CTAs, which hides HBM latency better than fewer
       fatter CTAs.

    The other configs cover degenerate shapes (very small ``Lq`` or
    ``Ld``) where the winner above can't fully tile the inner loop.
    """
    return [
        _cfg({"BLOCK_Q": 32, "BLOCK_D": 64}, num_warps=4, num_stages=3),
        _cfg({"BLOCK_Q": 32, "BLOCK_D": 128}, num_warps=4, num_stages=3),
        _cfg({"BLOCK_Q": 64, "BLOCK_D": 64}, num_warps=4, num_stages=3),
        _cfg({"BLOCK_Q": 64, "BLOCK_D": 128}, num_warps=4, num_stages=3),
        _cfg({"BLOCK_Q": 64, "BLOCK_D": 128}, num_warps=8, num_stages=3),
        _cfg({"BLOCK_Q": 128, "BLOCK_D": 64}, num_warps=8, num_stages=2),
        _cfg({"BLOCK_Q": 128, "BLOCK_D": 128}, num_warps=8, num_stages=2),
        _cfg({"BLOCK_Q": 128, "BLOCK_D": 128}, num_warps=8, num_stages=3),
    ]


def _small_d_ampere():
    """Works on A100, A10, A40, 3090, and is a safe default for Ada (L4, L40, 4090)."""
    return [
        _cfg({"BLOCK_Q": 32, "BLOCK_D": 64}, num_warps=4, num_stages=2),
        _cfg({"BLOCK_Q": 32, "BLOCK_D": 128}, num_warps=8, num_stages=2),
        _cfg({"BLOCK_Q": 64, "BLOCK_D": 64}, num_warps=4, num_stages=2),
        _cfg({"BLOCK_Q": 64, "BLOCK_D": 128}, num_warps=8, num_stages=2),
        _cfg({"BLOCK_Q": 128, "BLOCK_D": 64}, num_warps=8, num_stages=1),
    ]


def _large_d_configs():
    """Small-block configs for d ≥ 512 — fit any GPU, any SM."""
    return [
        _cfg({"BLOCK_Q": 16, "BLOCK_D": 16}, num_warps=2, num_stages=2),
        _cfg({"BLOCK_Q": 16, "BLOCK_D": 32}, num_warps=2, num_stages=2),
        _cfg({"BLOCK_Q": 32, "BLOCK_D": 16}, num_warps=2, num_stages=2),
        _cfg({"BLOCK_Q": 32, "BLOCK_D": 32}, num_warps=4, num_stages=2),
        _cfg({"BLOCK_Q": 32, "BLOCK_D": 64}, num_warps=4, num_stages=2),
    ]


# TODO: prune_forward's SRAM model below ignores `num_stages`, so it
# under-estimates the real Triton allocation by ~2x on double-buffered
# configs. The A10G (Ampere consumer, sm_86) hardware limit is ~99 KiB,
# and configs the prune lets through with the full 100 KiB budget
# overshoot at compile time. Band-aid: lower `ampere` to 64 KiB so the
# under-estimate still rejects the problem configs. Proper fix is to
# multiply input-tile bytes by `num_stages` in prune_forward, then
# restore these values. Same likely applies to `ada` (also 100 KiB hw).
_SRAM_KIB_BY_FAMILY = {
    "hopper": 228,
    "a100": 164,
    "ampere": 64,
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
