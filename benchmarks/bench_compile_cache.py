"""Benchmark: cold-start cost of variable ``Ld`` on the dense forward kernel.

Reports cold-pass wall-clock + autotune cache size after a sweep of
distinct doc lengths, plus warm/call latency at the middle ``Ld``.
Companion to ``tests/test_compile_cache.py`` (which pins the cache-size
invariant).
"""

import argparse
import json
import os
import time

import torch

from late_interaction_kernels import maxsim
from late_interaction_kernels.forward import _maxsim_fwd_kernel

# 18 distinct Ld values spanning the typical mixed-modality range
# (text Ld≈300, image-augmented Ld≈1024-1280).
LDS = list(range(192, 1281, 64))


def _time(fn, warmup=5, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def _bench_one(Nq: int, Nd: int, Lq: int, d: int, dtype: torch.dtype) -> dict:
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=dtype)
    Ds = [torch.randn(Nd, ld, d, device="cuda", dtype=dtype) for ld in LDS]

    # Amortise CUDA context init + the very first Triton compile so the
    # cold-pass timer measures only the Ld-driven recompile cost.
    _ = maxsim(Q, Ds[0])
    torch.cuda.synchronize()
    _maxsim_fwd_kernel.cache.clear()

    t0 = time.perf_counter()
    for D in Ds:
        _ = maxsim(Q, D)
        torch.cuda.synchronize()
    cold_s = time.perf_counter() - t0

    warm_ms = _time(lambda: maxsim(Q, Ds[len(Ds) // 2]))

    return {
        "Nq": Nq,
        "Nd": Nd,
        "Lq": Lq,
        "d": d,
        "cold_pass_s": cold_s,
        "cache_entries": len(_maxsim_fwd_kernel.cache),
        "warm_per_call_ms": warm_ms,
    }


def main(out_dir: str) -> None:
    gpu = torch.cuda.get_device_name(0).replace(" ", "_")
    shapes = [
        # name, Nq, Nd, Lq, d
        ("text", 2, 4, 32, 128),
        ("colpali", 1, 4, 1024, 128),
    ]

    rows = []
    for name, Nq, Nd, Lq, d in shapes:
        for dtype in (torch.float16, torch.bfloat16):
            r = _bench_one(Nq, Nd, Lq, d, dtype)
            r["name"] = name
            r["dtype"] = "fp16" if dtype is torch.float16 else "bf16"
            rows.append(r)
            print(
                f"{name:10s} {r['dtype']}  "
                f"cold={r['cold_pass_s']:6.2f}s  "
                f"cache_entries={r['cache_entries']}  "
                f"warm/call={r['warm_per_call_ms']:.3f}ms"
            )

    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/compile_cache_{gpu}.json", "w") as f:
        json.dump({"gpu": gpu, "lds": LDS, "rows": rows}, f, indent=2)
    with open(f"{out_dir}/compile_cache_{gpu}.md", "w") as f:
        f.write(f"# Compile-cache bench — {gpu}\n\n")
        f.write(f"Cold pass over {len(LDS)} distinct Ld in [{LDS[0]}, {LDS[-1]}].\n\n")
        f.write("| shape | dtype | Nq | Nd | Lq | d | cold (s) | cache entries | warm/call (ms) |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['name']} | {r['dtype']} | {r['Nq']} | {r['Nd']} | "
                f"{r['Lq']} | {r['d']} | {r['cold_pass_s']:.2f} | "
                f"{r['cache_entries']} | {r['warm_per_call_ms']:.3f} |\n"
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="benchmarks/results")
    args = p.parse_args()
    main(args.outdir)
