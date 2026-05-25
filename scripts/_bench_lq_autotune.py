"""Reproduce the variable-Lq autotune stall seen in PyLate end-to-end training.

Tony's PyLate-LIK bench (lightonai/pylate#222) ran 3.1x slower end-to-end than
the einsum reference because the per-batch max query length floats with the
tokenizer output — so each new Lq re-triggered the full Triton autotune sweep
(~21s per new value).

This script reproduces the kernel-level effect on synthetic data, agnostic of
PyLate. It calls ``maxsim()`` in a loop with Lq drawn from a realistic
distribution (8..32, like ColBERT) and reports:

* the per-step wall clock
* the total wall clock for the whole loop
* the autotune cache size at the end

Run on the same H100 with v0.2.0 (no bucketing) and the new branch:

    pip install late-interaction-kernels==0.2.0
    python scripts/_bench_lq_autotune.py
    # then:
    pip install -e .  # the new branch
    python scripts/_bench_lq_autotune.py
"""

from __future__ import annotations

import os
import random
import time

import torch

os.environ.setdefault("LIK_SUPPRESS_NORM_WARN", "1")

import late_interaction_kernels  # noqa: E402
from late_interaction_kernels import maxsim  # noqa: E402
from late_interaction_kernels.forward import _maxsim_fwd_kernel  # noqa: E402


def main() -> None:
    assert torch.cuda.is_available(), "this bench needs a CUDA device"

    device = "cuda"
    dtype = torch.bfloat16

    # Tony's bench used ColQwen2 with B=4, ~5 negatives per query and Lq drawn
    # from a realistic distribution. We use the same broad shape so the kernel
    # sees the same workload — only the autotune-cache pattern is exercised.
    Nq = 4
    Nd = 4 * 6
    d = 128
    Ld = 512

    rng = random.Random(0)
    n_steps = 30
    # Random Lq in 8..32 every step — what Tony actually saw.
    lq_per_step = [rng.randint(8, 32) for _ in range(n_steps)]

    print(f"# version    : late-interaction-kernels {late_interaction_kernels.__version__}")
    print(f"# device     : {torch.cuda.get_device_name(0)}")
    print(f"# shape      : Nq={Nq} Nd={Nd} Ld={Ld} d={d} dtype={dtype}")
    print(f"# n_steps    : {n_steps}")
    print(f"# Lq sample  : {lq_per_step[:10]}...")
    print()

    # Pre-build the D once — the autotune is keyed on Lq + d_pad + masks etc.,
    # not on Nd / Ld, so D can be reused.
    D = torch.randn(Nd, Ld, d, device=device, dtype=dtype)

    timings_ms: list[float] = []
    cache_size_history: list[int] = []
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    for step, lq in enumerate(lq_per_step):
        Q = torch.randn(Nq, lq, d, device=device, dtype=dtype)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = maxsim(Q, D)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e3
        timings_ms.append(dt)
        cache_size_history.append(len(_maxsim_fwd_kernel.cache))
        print(f"  step {step:>2d}  Lq={lq:>2d}  dt={dt:>8.1f} ms  cache={len(_maxsim_fwd_kernel.cache)}")
    torch.cuda.synchronize()
    total_s = time.perf_counter() - wall_start

    print()
    print(f"# total wall-clock : {total_s:>7.2f} s")
    print(f"# autotune entries : {cache_size_history[-1]}")
    print(f"# median step (ms) : {sorted(timings_ms)[len(timings_ms) // 2]:>7.2f}")
    print(f"# max step (ms)    : {max(timings_ms):>7.2f}")
    print(f"# min step (ms)    : {min(timings_ms):>7.2f}")


if __name__ == "__main__":
    main()
