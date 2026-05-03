"""MPS forward benchmark: ``torch.compile`` MaxSim vs eager reference.

Measures the high-level :class:`MaxSimScorer` / :func:`retrieve` path on
Apple-Silicon GPUs, which dispatches the dense-MaxSim formula through
``torch.compile`` (``_mps.maxsim_mps``). The eager baseline is the
unfused PyTorch reference — same math, no compile.

We don't bench against ``flash-maxsim`` here (CUDA-only) or a hand-written
Metal kernel: at the time of writing, Apple's MPSGraph already lowers
``einsum`` to ``simdgroup_matrix`` GEMM, and a naive scalar Metal
compute shader can't beat it. See ``docs/design.md`` for the rationale.

Usage::

    python benchmarks/bench_mps.py
    python benchmarks/bench_mps.py --quick
    python benchmarks/bench_mps.py --dtype fp16

Writes ``benchmarks/results/mps_<chip>_<dtype>.{md,json}``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time

import torch

from late_interaction_kernels._mps import maxsim_inference_mps  # noqa: E402
from late_interaction_kernels.reference import maxsim_reference  # noqa: E402

SHAPES = [
    # name, Nq, Nd, Lq, Ld, d
    ("rerank-short", 1, 1000, 32, 300, 128),
    ("rerank-mid", 1, 500, 32, 1024, 128),
    ("rerank-10k", 1, 10000, 32, 300, 128),
    ("colpali", 1, 100, 32, 1024, 128),
    ("train-batch", 32, 32, 32, 200, 128),
    ("edge-d48", 1, 4000, 32, 1024, 48),
    ("edge-d64", 1, 1000, 32, 300, 64),
]


def _sync() -> None:
    torch.mps.synchronize()


def _time_op(fn, warmup: int = 5, iters: int = 30) -> tuple[float, float]:
    """Median + stdev over ``iters`` runs (after ``warmup`` warmup iterations)."""
    for _ in range(warmup):
        fn()
        _sync()
    samples_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        _sync()
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
    med = statistics.median(samples_ms)
    sd = statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0
    return med, sd


def _peak_mb(fn) -> float:
    """Peak driver-side allocator usage (MB) around one call."""
    torch.mps.empty_cache()
    before = torch.mps.driver_allocated_memory()
    out = fn()
    _sync()
    after = torch.mps.driver_allocated_memory()
    del out
    return max(0.0, (after - before) / (1024.0 * 1024.0))


def bench_one(name, Nq, Nd, Lq, Ld, d, dtype):
    Q = torch.randn(Nq, Lq, d, device="mps", dtype=dtype)
    D = torch.randn(Nd, Ld, d, device="mps", dtype=dtype)

    rows = []

    # ``torch.compile``-fused (the default high-level path on MPS).
    t, sd = _time_op(lambda: maxsim_inference_mps(Q, D, normalize=True))
    m = _peak_mb(lambda: maxsim_inference_mps(Q, D, normalize=True))
    rows.append(("compile-fused", t, sd, m))

    # Eager reference for context — same math, no compile.
    try:
        t, sd = _time_op(lambda: maxsim_reference(Q, D, normalize=True))
        m = _peak_mb(lambda: maxsim_reference(Q, D, normalize=True))
        rows.append(("eager reference", t, sd, m))
    except (RuntimeError, torch.cuda.OutOfMemoryError):
        rows.append(("eager reference", float("nan"), float("nan"), float("nan")))

    return rows


def _chip() -> str:
    """Best-effort Apple-Silicon chip label, e.g. ``'M3_Pro'``."""
    if platform.system() != "Darwin":
        return "unknown"
    try:
        import subprocess

        out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        return out.replace("Apple ", "").replace(" ", "_")
    except Exception:
        return "Apple_Silicon"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--quick", action="store_true", help="Skip the biggest shapes.")
    ap.add_argument("--shape", default=None, help="Run only the named shape.")
    ap.add_argument("--outdir", default="benchmarks/results")
    args = ap.parse_args()

    if not torch.backends.mps.is_available():
        sys.exit("MPS not available — these benchmarks need an Apple Silicon machine.")

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    chip = _chip()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Device: mps ({chip})")
    print(f"dtype: {args.dtype}")
    print()

    shapes = SHAPES
    if args.shape:
        shapes = [s for s in shapes if s[0] == args.shape]
        if not shapes:
            sys.exit(f"no shape named {args.shape!r}")
    if args.quick:
        shapes = [s for s in shapes if s[1] * s[3] <= 10_000]

    report = {"chip": chip, "dtype": args.dtype, "shapes": []}

    for name, Nq, Nd, Lq, Ld, d in shapes:
        print(f"== {name:18s}  Nq={Nq} Nd={Nd} Lq={Lq} Ld={Ld} d={d}")
        rows = bench_one(name, Nq, Nd, Lq, Ld, d, dtype)
        for impl, t, sd, m in rows:
            print(f"      {impl:18s}  {t:7.3f} ± {sd:.3f} ms   {m:7.1f} MB")

        compile_t = rows[0][1] if rows[0][1] == rows[0][1] else float("nan")
        eager_t = rows[1][1] if rows[1][1] == rows[1][1] else float("nan")
        speedup = eager_t / compile_t if compile_t > 0 else float("nan")
        print(f"      → compile vs eager: {speedup:.2f}x")
        report["shapes"].append(
            {
                "name": name,
                "shape": [Nq, Nd, Lq, Ld, d],
                "compile_ms": rows[0][1],
                "compile_peak_mb": rows[0][3],
                "eager_ms": rows[1][1],
                "eager_peak_mb": rows[1][3],
                "speedup_compile_vs_eager": speedup,
            }
        )
        print()

    md = [f"# MPS forward benchmark — {chip} ({args.dtype})\n"]
    md.append("30-iter median, ``torch.mps.synchronize`` between calls.\n")
    md.append("| shape | compile ms | eager ms | speedup | compile peak MB | eager peak MB |")
    md.append("| --- | --- | --- | --- | --- | --- |")
    for e in report["shapes"]:
        md.append(
            f"| {e['name']} "
            f"| {e['compile_ms']:.3f} "
            f"| {e['eager_ms']:.3f} "
            f"| {e['speedup_compile_vs_eager']:.2f}x "
            f"| {e['compile_peak_mb']:.1f} "
            f"| {e['eager_peak_mb']:.1f} |"
        )

    out_md = os.path.join(args.outdir, f"mps_{chip}_{args.dtype}.md")
    out_json = os.path.join(args.outdir, f"mps_{chip}_{args.dtype}.json")
    with open(out_md, "w") as f:
        f.write("\n".join(md))
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"→ wrote {out_md}")
    print(f"→ wrote {out_json}")


if __name__ == "__main__":
    main()
