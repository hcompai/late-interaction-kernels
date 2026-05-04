"""Head-to-head benchmark: late-interaction-kernels vs flash-maxsim (roipony/IBM).

Both libraries implement the same core MaxSim math with FlashAttention-style
tiling, so this is the canonical apples-to-apples comparison. We run:

  * forward, bf16 + fp16
  * forward with ``normalize=True`` (we fuse it; flash-maxsim doesn't)
  * training forward + backward (late-interaction-kernels only — flash-maxsim 0.x
    doesn't ship an autograd-aware backward that matches our API)

and report: ms/iter, peak memory, and **speedup ratio**. Negative speedups
(i.e. flash-maxsim is faster) are reported honestly; if that happens on a
given shape we want to know, not hide it.

Usage::

    pip install "flash-maxsim==0.2.0"   # pinned to match the published numbers
    python benchmarks/bench_flash_maxsim.py
    python benchmarks/bench_flash_maxsim.py --quick  # skip biggest shapes
    python benchmarks/bench_flash_maxsim.py --shape train-batch

Writes a Markdown + JSON report to ``benchmarks/results/flash_maxsim_<gpu>_<dtype>.{md,json}``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import torch

from late_interaction_kernels import maxsim, maxsim_inference

try:
    import flash_maxsim  # noqa: F401
    from flash_maxsim import flash_maxsim_batched

    FM_VERSION = getattr(flash_maxsim, "__version__", "unknown")
    HAS_FM = True
except Exception as e:
    FM_VERSION = None
    HAS_FM = False
    _FM_IMPORT_ERR = e


SHAPES = [
    # (name, Nq, Nd, Lq, Ld, d)
    ("rerank-short", 1, 1000, 32, 300, 128),
    ("rerank-long", 1, 1000, 32, 1024, 128),
    ("rerank-very-long", 1, 500, 32, 4096, 128),
    ("rerank-colpali", 1, 500, 1024, 1024, 128),
    ("rerank-10k", 1, 10000, 32, 300, 128),
    ("train-in-batch-32", 32, 32, 32, 200, 128),
    ("train-in-batch-128", 128, 128, 32, 200, 128),
    ("train-long-doc", 16, 16, 32, 2048, 128),
    ("edge-d48", 1, 4000, 32, 2048, 48),
    ("edge-d64", 1, 10000, 32, 300, 64),
]


def cuda_time(fn, warmup: int = 10, iters: int = 50) -> tuple[float, float]:
    """Return (median_ms, stdev_ms) using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    return statistics.median(times_ms), statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0


def peak_mem_mb(fn) -> float:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    del out
    return peak / (1024.0 * 1024.0)


def bench_forward_one(name, Nq, Nd, Lq, Ld, d, dtype, normalize: bool):
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=dtype)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=dtype)
    if normalize:
        Q = torch.nn.functional.normalize(Q, dim=-1)
        D = torch.nn.functional.normalize(D, dim=-1)

    results = {}

    # --- ours -----------------------------------------------------------
    if normalize:
        # fused: undo the outer normalize so we can re-normalize inside.
        Qraw = Q / torch.linalg.vector_norm(Q, dim=-1, keepdim=True).clamp_min(1e-12)
        Draw = D / torch.linalg.vector_norm(D, dim=-1, keepdim=True).clamp_min(1e-12)

        def _ours():
            return maxsim_inference(Qraw, Draw, normalize=True)
    else:

        def _ours():
            return maxsim_inference(Q, D)

    t, sd = cuda_time(_ours)
    m = peak_mem_mb(_ours)
    results["late-interaction-kernels"] = {"ms": t, "ms_stdev": sd, "peak_mb": m}

    # --- flash-maxsim ---------------------------------------------------
    if HAS_FM:
        # flash-maxsim doesn't support fused normalize — emulate it by
        # pre-normalizing when the user asked for normalize=True.
        def _fm():
            return flash_maxsim_batched(Q, D, shared_docs=True)

        try:
            t, sd = cuda_time(_fm)
            m = peak_mem_mb(_fm)
            results["flash-maxsim"] = {"ms": t, "ms_stdev": sd, "peak_mb": m}
        except Exception as e:
            results["flash-maxsim"] = {"error": f"{type(e).__name__}: {e}"}

    # --- naive einsum ---------------------------------------------------
    def _naive():
        S = torch.einsum("ild,jtd->ijlt", Q.float(), D.float())
        return S.max(-1).values.sum(-1)

    try:
        t, sd = cuda_time(_naive)
        m = peak_mem_mb(_naive)
        results["naive-einsum"] = {"ms": t, "ms_stdev": sd, "peak_mb": m}
    except torch.cuda.OutOfMemoryError:
        results["naive-einsum"] = {"error": "OOM"}

    return results


def bench_backward_one(name, Nq, Nd, Lq, Ld, d, dtype):
    """Forward + backward; late-interaction-kernels only (flash-maxsim 0.x lacks a matching bwd API)."""
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=dtype, requires_grad=True)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=dtype, requires_grad=True)
    g = torch.ones(Nq, Nd, device="cuda", dtype=torch.float32)

    def _step():
        s = maxsim(Q, D)
        s.backward(g, retain_graph=False)
        Q.grad = None
        D.grad = None

    t, sd = cuda_time(_step)
    m = peak_mem_mb(lambda: maxsim(Q, D).sum().backward())
    return {"late-interaction-kernels-fwd+bwd": {"ms": t, "ms_stdev": sd, "peak_mb": m}}


def fmt_speedup(baseline, ours):
    if ours is None or baseline is None:
        return "-"
    if "error" in baseline or "error" in ours:
        return "-"
    ratio = baseline["ms"] / ours["ms"]
    sign = "×" if ratio >= 1.0 else "÷"
    return f"{ratio:.2f}{sign}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--outdir", default="benchmarks/results")
    ap.add_argument("--quick", action="store_true", help="Skip the biggest shapes.")
    ap.add_argument("--shape", default=None, help="Run only the shape with this name.")
    ap.add_argument(
        "--no-backward",
        action="store_true",
        help="Skip the training fwd+bwd section (forward-only report).",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA device required.")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    gpu = torch.cuda.get_device_name().replace(" ", "_")
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"dtype: {args.dtype}")
    print(f"flash-maxsim: {'installed ' + str(FM_VERSION) if HAS_FM else 'NOT INSTALLED'}")
    if not HAS_FM:
        print(f"  (import error: {_FM_IMPORT_ERR})")
    print()

    shapes = SHAPES
    if args.shape:
        shapes = [s for s in shapes if s[0] == args.shape]
        if not shapes:
            sys.exit(f"no shape named {args.shape!r}")
    if args.quick:
        shapes = [s for s in shapes if s[1] * s[2] <= 1000]

    report = {"gpu": gpu, "dtype": args.dtype, "fm_version": FM_VERSION, "shapes": []}

    for name, Nq, Nd, Lq, Ld, d in shapes:
        print(f"== {name:25s}  Nq={Nq} Nd={Nd} Lq={Lq} Ld={Ld} d={d}")
        fwd_noN = bench_forward_one(name, Nq, Nd, Lq, Ld, d, dtype, normalize=False)
        fwd_N = bench_forward_one(name, Nq, Nd, Lq, Ld, d, dtype, normalize=True)
        entry = {
            "name": name,
            "shape": [Nq, Nd, Lq, Ld, d],
            "forward_plain": fwd_noN,
            "forward_normalize": fwd_N,
        }
        if not args.no_backward:
            try:
                bwd = bench_backward_one(name, Nq, Nd, Lq, Ld, d, dtype)
                entry["training_fwd_bwd"] = bwd
            except torch.cuda.OutOfMemoryError:
                entry["training_fwd_bwd"] = {"error": "OOM"}

        for section, data in [
            ("forward (no norm)", fwd_noN),
            ("forward (normalize=True)", fwd_N),
        ]:
            print(f"   {section}")
            for impl, d_ in data.items():
                if "error" in d_:
                    print(f"      {impl:30s}  {d_['error']}")
                    continue
                print(
                    f"      {impl:30s}  {d_['ms']:7.3f} ± {d_['ms_stdev']:.3f} ms   {d_['peak_mb']:7.1f} MB"
                )

        if HAS_FM and "flash-maxsim" in fwd_noN and "late-interaction-kernels" in fwd_noN:
            speedup = fmt_speedup(fwd_noN["flash-maxsim"], fwd_noN["late-interaction-kernels"])
            print(f"   → vs flash-maxsim (no norm):  {speedup}")
        report["shapes"].append(entry)
        print()

    # Markdown report
    md = [f"# Head-to-head vs flash-maxsim — {gpu} ({args.dtype})\n"]
    md.append(f"flash-maxsim version: `{FM_VERSION}`.  50-iter median, CUDA events.\n")
    md.append("| shape | impl | fwd (no norm) ms | fwd (norm) ms | peak MB | speedup vs FM |")
    md.append("| --- | --- | --- | --- | --- | --- |")
    for e in report["shapes"]:
        for impl in ["late-interaction-kernels", "flash-maxsim", "naive-einsum"]:
            row_n = e["forward_plain"].get(impl, {})
            row_N = e["forward_normalize"].get(impl, {})

            def cell(r):
                if not r or "error" in r:
                    return r.get("error", "-") if r else "-"
                return f"{r['ms']:.3f}"

            mem = cell(row_n).replace("ms", "").strip()
            if "error" not in row_n and row_n:
                mem = f"{row_n['peak_mb']:.1f}"
            speedup = (
                fmt_speedup(
                    e["forward_plain"].get("flash-maxsim"), e["forward_plain"].get("late-interaction-kernels")
                )
                if impl == "late-interaction-kernels"
                else "-"
            )
            md.append(f"| {e['name']} | {impl} | {cell(row_n)} | {cell(row_N)} | {mem} | {speedup} |")

    out_md = os.path.join(args.outdir, f"flash_maxsim_{gpu}_{args.dtype}.md")
    out_json = os.path.join(args.outdir, f"flash_maxsim_{gpu}_{args.dtype}.json")
    with open(out_md, "w") as f:
        f.write("\n".join(md))
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"→ wrote {out_md}")
    print(f"→ wrote {out_json}")


if __name__ == "__main__":
    main()
