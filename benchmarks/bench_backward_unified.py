"""Backward-pass microbench: unified kernel vs two-pass (atomic, csr).

Reports ms/iter for the backward-only cost at typical training shapes.
The unified kernel should win most training-ish shapes because it
hoists ``Q`` out of the j loop, halving HBM read traffic vs two-pass.

Usage::

    python benchmarks/bench_backward_unified.py --dtype bf16
    python benchmarks/bench_backward_unified.py --quick
"""

import argparse
import json
import os
import statistics
import sys

import torch

from late_interaction_kernels.backward import maxsim_backward, maxsim_backward_unified
from late_interaction_kernels.forward import _run_forward

# (name, Nq, Nd, Lq, Ld, d)
SHAPES = [
    ("train-B16", 16, 16, 32, 200, 128),
    ("train-B32", 32, 32, 32, 200, 128),  # PyLate default
    ("train-B64", 64, 64, 32, 200, 128),
    ("train-B128", 128, 128, 32, 200, 128),
    ("train-B256", 256, 256, 32, 200, 128),
    ("long-doc-B16", 16, 16, 32, 1024, 128),
    ("long-doc-B32", 32, 32, 32, 1024, 128),
    ("colpali-B4", 4, 4, 1024, 1024, 128),
    ("edge-d48-B32", 32, 32, 32, 256, 48),
    ("edge-d64-B32", 32, 32, 32, 256, 64),
    ("large-d-256", 16, 16, 32, 200, 256),
    ("large-d-512", 8, 8, 32, 200, 512),
]


def cuda_time(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times), statistics.stdev(times) if len(times) > 1 else 0.0


def bench(name, Nq, Nd, Lq, Ld, d, dtype):
    torch.manual_seed(0)
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=dtype)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=dtype)
    grad_s = torch.randn(Nq, Nd, device="cuda", dtype=torch.float32)
    _, argmax = _run_forward(Q, D, q_mask=None, d_mask=None, save_argmax=True)

    def run_two_pass_atomic():
        return maxsim_backward(grad_s, Q, D, argmax, None, None, method="atomic")

    def run_two_pass_csr():
        return maxsim_backward(grad_s, Q, D, argmax, None, None, method="csr")

    def run_unified():
        return maxsim_backward_unified(grad_s, Q, D, argmax, q_mask=None)

    t_atom, sd_atom = cuda_time(run_two_pass_atomic)
    t_csr, sd_csr = cuda_time(run_two_pass_csr)
    t_uni, sd_uni = cuda_time(run_unified)

    return {
        "atomic": {"ms": t_atom, "sd": sd_atom},
        "csr": {"ms": t_csr, "sd": sd_csr},
        "unified": {"ms": t_uni, "sd": sd_uni},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--outdir", default="benchmarks/results")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA device required.")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    gpu = torch.cuda.get_device_name().replace(" ", "_")
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Device: {torch.cuda.get_device_name()}   dtype: {args.dtype}")
    print()

    shapes = SHAPES
    if args.quick:
        shapes = [s for s in shapes if s[1] <= 32]

    results = []
    for name, Nq, Nd, Lq, Ld, d in shapes:
        print(f"== {name:20s}  Nq={Nq} Nd={Nd} Lq={Lq} Ld={Ld} d={d}")
        try:
            r = bench(name, Nq, Nd, Lq, Ld, d, dtype)
        except torch.cuda.OutOfMemoryError:
            print("   OOM")
            continue

        base = r["atomic"]["ms"]
        speedup_uni = base / r["unified"]["ms"]
        speedup_csr = base / r["csr"]["ms"]
        print(f"   atomic   {r['atomic']['ms']:6.2f} ± {r['atomic']['sd']:4.2f}  (baseline)")
        print(f"   csr      {r['csr']['ms']:6.2f} ± {r['csr']['sd']:4.2f}  {speedup_csr:5.2f}x vs atomic")
        print(
            f"   unified  {r['unified']['ms']:6.2f} ± {r['unified']['sd']:4.2f}  "
            f"{speedup_uni:5.2f}x vs atomic"
        )
        print()
        results.append(
            {
                "name": name,
                "shape": [Nq, Nd, Lq, Ld, d],
                **r,
                "speedup_unified_vs_atomic": speedup_uni,
                "speedup_csr_vs_atomic": speedup_csr,
            }
        )

    out_md = os.path.join(args.outdir, f"backward_unified_{gpu}_{args.dtype}.md")
    out_json = os.path.join(args.outdir, f"backward_unified_{gpu}_{args.dtype}.json")

    md = [f"# Backward: two-pass vs unified — {gpu} ({args.dtype})\n"]
    md.append("50-iter median, CUDA events. Baseline = two-pass atomic.\n")
    md.append("| shape | atomic (ms) | csr (ms) | unified (ms) | unified speedup |")
    md.append("| --- | --- | --- | --- | --- |")
    for r in results:
        md.append(
            f"| {r['name']} Nq={r['shape'][0]} Lq={r['shape'][2]} Ld={r['shape'][3]} d={r['shape'][4]} "
            f"| {r['atomic']['ms']:.2f} | {r['csr']['ms']:.2f} | {r['unified']['ms']:.2f} "
            f"| **{r['speedup_unified_vs_atomic']:.2f}x** |"
        )
    with open(out_md, "w") as f:
        f.write("\n".join(md))
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"→ wrote {out_md}")
    print(f"→ wrote {out_json}")


if __name__ == "__main__":
    main()
