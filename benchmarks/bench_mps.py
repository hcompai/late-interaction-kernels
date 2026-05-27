"""MPS forward benchmark: Metal MMA kernel vs ``torch.compile`` vs eager.

Three implementations land on Apple Silicon:

* ``metal``   — fused ``simdgroup_matrix`` kernel
  (:func:`late_interaction_kernels.metal.maxsim_inference_metal`),
* ``compile`` — ``torch.compile``-fused reference
  (:func:`late_interaction_kernels.mps.compile_dispatch.maxsim_inference_mps`,
  forced via ``LIK_FORCE_MPS_BACKEND=compile``),
* ``eager``   — unfused PyTorch reference (no compile).

The Metal path requires fp16 / bf16 inputs with ``d`` ≤ 128 and
divisible by 8 — the upper bound comes from the register-resident Q
cache (``simdgroup_matrix<T, 8, 8>[M_TILES][K_TILES_MAX]``, sized for
``d / 8 ≤ 16`` and ``Lq / 8 ≤ 4``); going past 128 spills to
threadgroup memory on the M-series register file and the kernel
slows down by 2-4×. Outside the supported envelope, this script runs
only ``compile`` / ``eager``. The default high-level dispatch picks
``metal`` when the shape is large enough for the kernel's launch
overhead to amortise.

Usage::

    python benchmarks/bench_mps.py
    python benchmarks/bench_mps.py --quick
    python benchmarks/bench_mps.py --dtype bf16

Writes ``benchmarks/results/mps_<chip>_<dtype>.{md,json}``.
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time
from typing import Final

import torch

from late_interaction_kernels.mps import metal as _metal
from late_interaction_kernels.reference import maxsim_reference

# (name, Nq, Nd, Lq, Ld, d)
SHAPES: Final[tuple[tuple[str, int, int, int, int, int], ...]] = (
    ("rerank-short", 1, 1000, 32, 300, 128),
    ("rerank-mid", 1, 500, 32, 1024, 128),
    ("rerank-10k", 1, 10000, 32, 300, 128),
    ("colpali", 1, 100, 32, 1024, 128),
    ("colpali-big", 1, 500, 32, 1024, 128),
    ("train-batch", 32, 32, 32, 200, 128),
    ("edge-d48", 1, 4000, 32, 1024, 48),
    ("edge-d64", 1, 1000, 32, 300, 64),
)


def _sync() -> None:
    torch.mps.synchronize()


def _time_op(fn, warmup: int = 10, iters: int = 30) -> tuple[float, float]:
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
    torch.mps.empty_cache()
    before = torch.mps.driver_allocated_memory()
    out = fn()
    _sync()
    after = torch.mps.driver_allocated_memory()
    del out
    return max(0.0, (after - before) / (1024.0 * 1024.0))


def _compile_call(Q, D):
    """Force the compile path even on shapes the heuristic prefers Metal for."""
    from late_interaction_kernels.mps import compile_dispatch as _mps

    return _mps._compile_path(Q, D, q_mask=None, d_mask=None, normalize=True)


def bench_one(name, Nq, Nd, Lq, Ld, d, dtype):
    Q = torch.randn(Nq, Lq, d, device="mps", dtype=dtype)
    D = torch.randn(Nd, Ld, d, device="mps", dtype=dtype)

    rows: list[tuple[str, float, float, float]] = []

    if _metal.is_available() and _metal.supports(Q, D):
        t, sd = _time_op(lambda: _metal.maxsim_inference_metal(Q, D, normalize=True))
        m = _peak_mb(lambda: _metal.maxsim_inference_metal(Q, D, normalize=True))
        rows.append(("metal", t, sd, m))
    else:
        rows.append(("metal", float("nan"), float("nan"), float("nan")))

    t, sd = _time_op(lambda: _compile_call(Q, D))
    m = _peak_mb(lambda: _compile_call(Q, D))
    rows.append(("compile", t, sd, m))

    try:
        t, sd = _time_op(lambda: maxsim_reference(Q, D, normalize=True))
        m = _peak_mb(lambda: maxsim_reference(Q, D, normalize=True))
        rows.append(("eager", t, sd, m))
    except RuntimeError:
        rows.append(("eager", float("nan"), float("nan"), float("nan")))

    return rows


def _chip() -> str:
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
    ap.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="subset of shape names to run; default = all.",
    )
    ap.add_argument("--outdir", default="benchmarks/results")
    args = ap.parse_args()

    if not torch.backends.mps.is_available():
        sys.exit("MPS not available — these benchmarks need an Apple Silicon machine.")

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    chip = _chip()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Device: mps ({chip})")
    print(f"dtype: {args.dtype}")
    metal_ok = _metal.is_available() and dtype in (torch.float16, torch.bfloat16)
    print(f"metal kernel: {'enabled' if metal_ok else 'disabled (dtype/build)'}")
    print()

    if args.only:
        wanted = set(args.only)
        shapes = [s for s in SHAPES if s[0] in wanted]
        if not shapes:
            sys.exit(f"unknown shape(s); pick from: {[s[0] for s in SHAPES]}")
    elif args.quick:
        shapes = [s for s in SHAPES if s[1] * s[3] <= 10_000]
    else:
        shapes = SHAPES

    report = {"chip": chip, "dtype": args.dtype, "shapes": []}

    for name, Nq, Nd, Lq, Ld, d in shapes:
        print(f"== {name:18s}  Nq={Nq} Nd={Nd} Lq={Lq} Ld={Ld} d={d}")
        rows = bench_one(name, Nq, Nd, Lq, Ld, d, dtype)
        timings = {impl: (t, sd, m) for impl, t, sd, m in rows}
        for impl, t, sd, m in rows:
            t_str = f"{t:7.3f} ± {sd:.3f} ms" if t == t else "        n/a       "
            m_str = f"{m:7.1f} MB" if m == m else "    n/a"
            print(f"      {impl:8s}  {t_str}   {m_str}")
        metal_t, _, metal_mb = timings["metal"]
        comp_t, _, comp_mb = timings["compile"]
        eager_t, _, _ = timings["eager"]
        metal_vs_compile = (comp_t / metal_t) if (metal_t == metal_t and comp_t > 0) else None
        compile_vs_eager = (eager_t / comp_t) if (eager_t == eager_t and comp_t > 0) else None
        metal_vs_eager = (eager_t / metal_t) if (metal_t == metal_t and eager_t == eager_t) else None
        if metal_vs_compile is not None:
            print(f"      → metal vs compile: {metal_vs_compile:.2f}x")
        if compile_vs_eager is not None:
            print(f"      → compile vs eager: {compile_vs_eager:.2f}x")
        if metal_vs_eager is not None:
            print(f"      → metal vs eager:   {metal_vs_eager:.2f}x")
        report["shapes"].append(
            {
                "name": name,
                "shape": [Nq, Nd, Lq, Ld, d],
                "metal_ms": metal_t,
                "metal_peak_mb": metal_mb,
                "compile_ms": comp_t,
                "compile_peak_mb": comp_mb,
                "eager_ms": eager_t,
                "speedup_metal_vs_compile": metal_vs_compile,
                "speedup_compile_vs_eager": compile_vs_eager,
                "speedup_metal_vs_eager": metal_vs_eager,
            }
        )
        print()

    def _ratio(v: float | None) -> str:
        return f"{v:.2f}x" if v is not None else "n/a"

    md = [f"# MPS forward benchmark — {chip} ({args.dtype})\n"]
    md.append("30-iter median, `torch.mps.synchronize` between calls.\n")
    md.append(
        "| shape | metal ms | compile ms | eager ms | metal vs eager | metal vs compile | compile vs eager |"
    )
    md.append("| --- | --- | --- | --- | --- | --- | --- |")
    for e in report["shapes"]:
        m_str = f"{e['metal_ms']:.3f}" if e["metal_ms"] == e["metal_ms"] else "n/a"
        md.append(
            f"| {e['name']} | {m_str} | {e['compile_ms']:.3f} | {e['eager_ms']:.3f} "
            f"| {_ratio(e['speedup_metal_vs_eager'])} "
            f"| {_ratio(e['speedup_metal_vs_compile'])} "
            f"| {_ratio(e['speedup_compile_vs_eager'])} |"
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
