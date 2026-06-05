"""MPS forward / training benchmark: Metal MMA kernel vs ``torch.compile`` vs eager.

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

    python benchmarks/kernels/bench_mps.py                     # cross-product forward
    python benchmarks/kernels/bench_mps.py --layout kd         # KD / pairs (4-D D)
    python benchmarks/kernels/bench_mps.py --mode train        # forward + backward
    python benchmarks/kernels/bench_mps.py --quick
    python benchmarks/kernels/bench_mps.py --dtype bf16

Writes ``benchmarks/results/mps_<chip>_<dtype>{_train}.{md,json}``.
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

# KD / pairs layout: ``D.shape == (Nq, K, Ld, d)``, each query owns its
# own K-slab. ``K=1`` is the diagonal "pairs" corner.
# (name, Nq, K, Lq, Ld, d)
KD_SHAPES: Final[tuple[tuple[str, int, int, int, int, int], ...]] = (
    ("kd-rerank-top10", 1, 10, 32, 300, 128),
    ("kd-rerank-top32", 1, 32, 32, 300, 128),
    ("kd-bs4-K10", 4, 10, 32, 300, 128),
    ("kd-bs8-K32", 8, 32, 32, 200, 128),
    ("kd-pairs-bs32", 32, 1, 32, 256, 128),
    ("kd-bs1-K100-long", 1, 100, 32, 1024, 128),
)

# Training shapes: smaller corpora, gradient time matters more than peak
# throughput. (name, Nq, Nd, Lq, Ld, d)
TRAIN_SHAPES: Final[tuple[tuple[str, int, int, int, int, int], ...]] = (
    ("train-bs8", 8, 8, 32, 64, 128),
    ("train-bs16", 16, 16, 32, 128, 128),
    ("train-bs32", 32, 32, 32, 200, 128),
    ("train-bs8-long", 8, 8, 32, 512, 128),
    ("train-d64", 16, 16, 32, 256, 64),
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
    """Force the compile path (3-D and 4-D D both routed through ``_compile_path``)."""
    from late_interaction_kernels.mps import compile_dispatch as _mps

    return _mps._compile_path(Q, D, q_mask=None, d_mask=None, normalize=True)


def _kd_eager_call(Q, D):
    """Eager KD reference (no ``torch.compile``)."""
    from late_interaction_kernels.mps.compile_dispatch import _kd_reference

    return _kd_reference(Q, D, normalize=True)


_kd_compile_call = _compile_call


def _train_step(maxsim_fn, Q, D, gs):
    """Forward + backward; resets ``.grad`` so re-runs don't accumulate."""
    if Q.grad is not None:
        Q.grad = None
    if D.grad is not None:
        D.grad = None
    scores = maxsim_fn(Q, D)
    scores.backward(gs)


def _metal_train_call():
    from late_interaction_kernels.mps.compile_dispatch import maxsim_mps

    def f(Q, D):
        return maxsim_mps(Q, D, normalize=True)

    return f


def _compile_train_call():
    from late_interaction_kernels.mps import compile_dispatch as cd

    def f(Q, D):
        return cd._compile_path(Q, D, q_mask=None, d_mask=None, normalize=True)

    return f


def _eager_train_call():
    def f(Q, D):
        return maxsim_reference(Q, D, normalize=True)

    return f


def bench_one_train(name, Nq, Nd, Lq, Ld, d, dtype):
    """Forward + backward bench. Same shape set / runner as ``bench_one``."""
    Q = torch.randn(Nq, Lq, d, device="mps", dtype=dtype, requires_grad=True)
    D = torch.randn(Nd, Ld, d, device="mps", dtype=dtype, requires_grad=True)
    gs = torch.randn(Nq, Nd, device="mps", dtype=torch.float32)

    rows: list[tuple[str, float, float, float]] = []

    metal_fn = _metal_train_call()
    if _metal.is_available() and _metal.supports(Q, D):
        # Force the Metal autograd path so we measure the kernel and not
        # whichever path the heuristic happens to pick. try/finally so
        # a kernel error doesn't leak the env var into later shapes.
        os.environ["LIK_FORCE_MPS_BACKEND"] = "metal"
        try:
            t, sd = _time_op(lambda: _train_step(metal_fn, Q, D, gs))
            m = _peak_mb(lambda: _train_step(metal_fn, Q, D, gs))
            rows.append(("metal", t, sd, m))
        finally:
            os.environ.pop("LIK_FORCE_MPS_BACKEND", None)
    else:
        rows.append(("metal", float("nan"), float("nan"), float("nan")))

    compile_fn = _compile_train_call()
    try:
        t, sd = _time_op(lambda: _train_step(compile_fn, Q, D, gs))
        m = _peak_mb(lambda: _train_step(compile_fn, Q, D, gs))
        rows.append(("compile", t, sd, m))
    except Exception:
        rows.append(("compile", float("nan"), float("nan"), float("nan")))

    eager_fn = _eager_train_call()
    t, sd = _time_op(lambda: _train_step(eager_fn, Q, D, gs))
    m = _peak_mb(lambda: _train_step(eager_fn, Q, D, gs))
    rows.append(("eager", t, sd, m))

    return rows


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


def bench_one_kd(name, Nq, K, Lq, Ld, d, dtype):
    """KD-layout bench: ``D.shape == (Nq, K, Ld, d)``."""
    Q = torch.randn(Nq, Lq, d, device="mps", dtype=dtype)
    D = torch.randn(Nq, K, Ld, d, device="mps", dtype=dtype)

    rows: list[tuple[str, float, float, float]] = []

    if _metal.is_available() and _metal.supports(Q, D):
        t, sd = _time_op(lambda: _metal.maxsim_inference_metal(Q, D, normalize=True))
        m = _peak_mb(lambda: _metal.maxsim_inference_metal(Q, D, normalize=True))
        rows.append(("metal", t, sd, m))
    else:
        rows.append(("metal", float("nan"), float("nan"), float("nan")))

    try:
        t, sd = _time_op(lambda: _kd_compile_call(Q, D))
        m = _peak_mb(lambda: _kd_compile_call(Q, D))
        rows.append(("compile", t, sd, m))
    except Exception:
        rows.append(("compile", float("nan"), float("nan"), float("nan")))

    t, sd = _time_op(lambda: _kd_eager_call(Q, D))
    m = _peak_mb(lambda: _kd_eager_call(Q, D))
    rows.append(("eager", t, sd, m))

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
    ap.add_argument(
        "--layout",
        choices=["xprod", "kd", "all"],
        default="all",
        help="Which shape set to run: cross-product, KD, or both.",
    )
    ap.add_argument(
        "--mode",
        choices=["fwd", "train"],
        default="fwd",
        help="``fwd`` (default) times the forward only; ``train`` times "
        "forward + backward on the TRAIN_SHAPES set.",
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

    layouts: list[tuple[str, tuple]] = []
    if args.mode == "train":
        # Train mode runs a fixed set of shapes — KD/pairs autograd is
        # routed through the compile path today, so only cross-product
        # shapes belong here.
        layouts.append(("train", TRAIN_SHAPES))
    else:
        if args.layout in ("xprod", "all"):
            layouts.append(("xprod", SHAPES))
        if args.layout in ("kd", "all"):
            layouts.append(("kd", KD_SHAPES))

    if args.only:
        wanted = set(args.only)
        layouts = [(lay, tuple(s for s in shp if s[0] in wanted)) for lay, shp in layouts]
        layouts = [(lay, shp) for lay, shp in layouts if shp]
        if not layouts:
            known = [s[0] for s in SHAPES] + [s[0] for s in KD_SHAPES]
            sys.exit(f"unknown shape(s); pick from: {known}")
    if args.quick:
        layouts = [(lay, tuple(s for s in shp if s[1] * s[3] <= 10_000)) for lay, shp in layouts]

    report = {"chip": chip, "dtype": args.dtype, "shapes": []}

    flat: list[tuple[str, tuple]] = []
    for lay, shp in layouts:
        for s in shp:
            flat.append((lay, s))

    for layout, shape in flat:
        name, Nq, Nd, Lq, Ld, d = shape
        if layout == "kd":
            tag, axis = " [KD]", "K"
        elif layout == "train":
            tag, axis = " [train]", "Nd"
        else:
            tag, axis = "", "Nd"
        print(f"== {name:20s}{tag}  Nq={Nq} {axis}={Nd} Lq={Lq} Ld={Ld} d={d}")
        if layout == "kd":
            rows = bench_one_kd(name, Nq, Nd, Lq, Ld, d, dtype)
        elif layout == "train":
            rows = bench_one_train(name, Nq, Nd, Lq, Ld, d, dtype)
        else:
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
                "layout": layout,
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

    title = "training" if args.mode == "train" else "forward"
    md = [f"# MPS {title} benchmark — {chip} ({args.dtype})\n"]
    md.append("30-iter median, `torch.mps.synchronize` between calls.\n")

    def _write_section(label: str, entries: list[dict]) -> None:
        if not entries:
            return
        md.append(f"\n## {label}\n")
        md.append(
            "| shape | metal ms | compile ms | eager ms "
            "| metal vs eager | metal vs compile | compile vs eager |"
        )
        md.append("| --- | --- | --- | --- | --- | --- | --- |")
        for e in entries:
            m_str = f"{e['metal_ms']:.3f}" if e["metal_ms"] == e["metal_ms"] else "n/a"
            md.append(
                f"| {e['name']} | {m_str} | {e['compile_ms']:.3f} | {e['eager_ms']:.3f} "
                f"| {_ratio(e['speedup_metal_vs_eager'])} "
                f"| {_ratio(e['speedup_metal_vs_compile'])} "
                f"| {_ratio(e['speedup_compile_vs_eager'])} |"
            )

    _write_section(
        "Cross-product (3-D D)",
        [e for e in report["shapes"] if e.get("layout") == "xprod"],
    )
    _write_section(
        "KD / pairs (4-D D)",
        [e for e in report["shapes"] if e.get("layout") == "kd"],
    )
    _write_section(
        "Training (forward + backward)",
        [e for e in report["shapes"] if e.get("layout") == "train"],
    )

    suffix = "_train" if args.mode == "train" else ""
    out_md = os.path.join(args.outdir, f"mps_{chip}_{args.dtype}{suffix}.md")
    out_json = os.path.join(args.outdir, f"mps_{chip}_{args.dtype}{suffix}.json")
    with open(out_md, "w") as f:
        f.write("\n".join(md))
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"→ wrote {out_md}")
    print(f"→ wrote {out_json}")


if __name__ == "__main__":
    main()
