"""Forward-pass microbenchmark: late-interaction-kernels vs naive einsum vs flash-maxsim.

    python benchmarks/bench_forward.py

Writes a Markdown table + JSON to `results/forward_<gpu>.{md,json}`.
"""

import argparse
import json
import os

import torch

from late_interaction_kernels import maxsim_inference

try:
    from flash_maxsim import flash_maxsim_batched  # optional

    HAS_FM = True
except Exception:
    HAS_FM = False


SHAPES = [
    # name, Nq, Nd, Lq, Ld, d
    ("text-short", 1, 1000, 32, 300, 128),
    ("text-long", 1, 1000, 32, 1024, 128),
    ("text-medium", 1, 1000, 128, 1024, 128),
    ("visual", 1, 1000, 1024, 1024, 128),
    ("corpus-5k", 1, 5000, 32, 300, 128),
    ("corpus-10k", 1, 10000, 32, 300, 128),
    ("train-batch", 32, 32, 32, 300, 128),  # in-batch negatives scenario
    ("train-batch-128", 128, 128, 32, 300, 128),
    ("large-d-512", 1, 1000, 32, 300, 512),
    ("large-d-1024", 1, 500, 32, 300, 1024),
    # Small-d shapes — LateOn-Code-edge (d=48) and mxbai-edge (d=64) style
    # rerankers. Small d makes the GPU HBM-bound on the Q/D loads, so
    # fusion gives a smaller win than at d=128 but memory matters more.
    ("lateon-code-edge-rerank", 1, 1000, 32, 2048, 48),
    ("lateon-code-edge-big", 1, 4000, 32, 2048, 48),
    ("mxbai-edge-rerank", 1, 1000, 32, 300, 64),
    ("mxbai-edge-corpus-10k", 1, 10000, 32, 300, 64),
]


def _time_op(fn, warmup=5, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def _peak_mem(fn):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    del out
    return peak / (1024 * 1024)  # MB


def _naive(Q, D):
    S = torch.einsum("ild,jtd->ijlt", Q.float(), D.float())
    return S.max(-1).values.sum(-1)


def bench_one(name, Nq, Nd, Lq, Ld, d, dtype):
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=dtype)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=dtype)

    rows = []

    # late-interaction-kernels
    t = _time_op(lambda: maxsim_inference(Q, D))
    m = _peak_mem(lambda: maxsim_inference(Q, D))
    rows.append(("late-interaction-kernels", t, m))

    # naive
    try:
        t = _time_op(lambda: _naive(Q, D))
        m = _peak_mem(lambda: _naive(Q, D))
        rows.append(("naive einsum", t, m))
    except torch.cuda.OutOfMemoryError:
        rows.append(("naive einsum", float("nan"), float("nan")))

    # flash-maxsim (if available)
    if HAS_FM:
        try:
            t = _time_op(lambda: flash_maxsim_batched(Q, D, shared_docs=True))
            m = _peak_mem(lambda: flash_maxsim_batched(Q, D, shared_docs=True))
            rows.append(("flash-maxsim", t, m))
        except Exception as e:
            rows.append((f"flash-maxsim (err: {type(e).__name__})", float("nan"), float("nan")))

    return name, dtype, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--outdir", default="benchmarks/results")
    parser.add_argument("--quick", action="store_true", help="Skip the biggest shapes")
    args = parser.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    os.makedirs(args.outdir, exist_ok=True)
    gpu = torch.cuda.get_device_name().replace(" ", "_")

    shapes = [s for s in SHAPES if not args.quick or s[2] <= 1000]
    results = []
    for shape in shapes:
        name, *rest = shape
        print(f"== {name} ==")
        name, dtype_, rows = bench_one(*shape, dtype=dtype)
        for impl, t, m in rows:
            print(f"  {impl:25s}  {t:8.3f} ms   {m:8.2f} MB")
        results.append({"name": name, "shape": shape[1:], "dtype": str(dtype), "rows": rows})

    # Pretty Markdown
    md_lines = [f"# Forward bench — {gpu} ({args.dtype})\n"]
    for r in results:
        md_lines.append(
            f"## {r['name']} — shape Nq={r['shape'][0]} Nd={r['shape'][1]} Lq={r['shape'][2]} Ld={r['shape'][3]} d={r['shape'][4]}\n"
        )
        md_lines.append("| impl | time (ms) | peak mem (MB) |")
        md_lines.append("| --- | --- | --- |")
        for impl, t, m in r["rows"]:
            ts = f"{t:.3f}" if t == t else "OOM"
            ms = f"{m:.2f}" if m == m else "OOM"
            md_lines.append(f"| {impl} | {ts} | {ms} |")
        md_lines.append("")

    out_md = os.path.join(args.outdir, f"forward_{gpu}_{args.dtype}.md")
    out_json = os.path.join(args.outdir, f"forward_{gpu}_{args.dtype}.json")
    with open(out_md, "w") as f:
        f.write("\n".join(md_lines))
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n→ wrote {out_md}")
    print(f"→ wrote {out_json}")


if __name__ == "__main__":
    main()
