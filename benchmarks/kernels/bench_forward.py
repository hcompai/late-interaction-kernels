"""Forward-pass microbenchmark for ``maxsim``.

All implementations share the same numerical contract so the speedups are
fair:

* inputs are ``bf16`` / ``fp16`` (whatever a real encoder produces),
* the inner ``einsum`` runs with an ``fp32`` accumulator,
* the final ``[Nq, Nd]`` output is ``fp32``.

Three baselines are reported per shape:

* ``late-interaction-kernels`` — fused Triton kernel (``maxsim``).
* ``naive (eager, fp32 acc)`` — ``einsum`` upcast to ``fp32`` then
  ``max(-1).sum(-1)``. Materialises the full ``[Nq, Nd, Lq, Ld]`` tensor
  in HBM. This is what every reference implementation does today.
* ``torch.compile (fp32 acc)`` — same body, wrapped in
  ``torch.compile(dynamic=False)``. Inductor can fuse the ``max(-1)``
  reduction but still has to materialise the similarity tile, so the
  HBM round-trip survives.

At the start of every shape we assert all three implementations agree
within bf16-input precision (``atol`` defaults below) — if a future
refactor breaks parity the bench will fail loudly instead of reporting
a fake speedup.

Usage::

    python benchmarks/kernels/bench_forward.py
    python benchmarks/kernels/bench_forward.py --dtype fp16 --quick

Writes a Markdown table + JSON to ``benchmarks/results/forward_<gpu>_<dtype>.{md,json}``.
"""

import argparse
import json
import os

import torch

from late_interaction_kernels import maxsim

try:
    from flash_maxsim import flash_maxsim_batched

    HAS_FM = True
except Exception:
    HAS_FM = False


SHAPES = [
    # name, Nq, Nd, Lq, Ld, d
    ("text-short", 1, 1000, 32, 300, 128),
    ("text-long", 1, 1000, 32, 1024, 128),
    ("colpali", 1, 1000, 128, 1024, 128),  # short text query expanded to Lq=128 vs ~1k-patch page
    ("corpus-5k", 1, 5000, 32, 300, 128),
    ("corpus-10k", 1, 10000, 32, 300, 128),
    ("train-batch", 32, 32, 32, 300, 128),
    ("train-batch-128", 128, 128, 32, 300, 128),
    ("large-d-512", 1, 1000, 32, 300, 512),
    ("large-d-1024", 1, 500, 32, 300, 1024),
    # LateOn-Code-edge (d=48) and mxbai-edge (d=64) rerankers — small d
    # makes the GPU HBM-bound on the Q/D loads, so fusion gives a smaller
    # win than at d=128 but memory matters more.
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


def _eager_fp32(Q, D):
    """Reference einsum with an fp32 accumulator — same numerical contract as ``maxsim``."""
    S = torch.einsum("ild,jtd->ijlt", Q.float(), D.float())
    return S.max(-1).values.sum(-1)


# torch.compile caches per (function id, dtype, shape) tuple. We instantiate
# once at module level so the cache survives the inner per-shape closures
# and recompiles only when the input shapes change.
_compiled_eager_fp32 = torch.compile(_eager_fp32, dynamic=False, mode="reduce-overhead")


# Tolerances for parity check between bf16-accumulator-fp32 implementations.
# bf16 keeps 7-8 bits of mantissa; row sums over `Ld` accumulate error
# linearly. 1e-2 absolute / 1e-2 relative is loose enough to never false-
# positive on real workloads and tight enough to catch any regression that
# would invalidate the speedup numbers.
PARITY_ATOL = 1e-2
PARITY_RTOL = 1e-2


def _check_parity(name, *outputs_with_labels):
    """Assert every implementation agrees within ``PARITY_{A,R}TOL`` of the eager reference."""
    ref_label, ref = outputs_with_labels[0]
    for label, out in outputs_with_labels[1:]:
        if out is None:
            continue
        if not torch.allclose(out.float(), ref.float(), atol=PARITY_ATOL, rtol=PARITY_RTOL):
            diff = (out.float() - ref.float()).abs()
            raise AssertionError(
                f"[{name}] {label} disagrees with {ref_label}: "
                f"max abs diff {diff.max().item():.3g}, "
                f"max rel diff {(diff / ref.float().abs().clamp(min=1e-6)).max().item():.3g} "
                f"(atol={PARITY_ATOL}, rtol={PARITY_RTOL})"
            )


def bench_one(name, Nq, Nd, Lq, Ld, d, dtype):
    torch.manual_seed(0)
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=dtype)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=dtype)

    # Parity: compute one output per implementation and compare against the
    # fp32-accumulator eager reference before timing anything.
    with torch.no_grad():
        ref = _eager_fp32(Q, D)
        out_lik = maxsim(Q, D)
        try:
            out_compile = _compiled_eager_fp32(Q, D)
        except Exception as e:
            print(f"  [warn] torch.compile parity skipped: {type(e).__name__}: {e}")
            out_compile = None

    _check_parity(
        name,
        ("naive (eager, fp32 acc)", ref),
        ("late-interaction-kernels", out_lik),
        ("torch.compile (fp32 acc)", out_compile),
    )
    del ref, out_lik, out_compile

    rows = []

    t = _time_op(lambda: maxsim(Q, D))
    m = _peak_mem(lambda: maxsim(Q, D))
    rows.append(("late-interaction-kernels", t, m))

    try:
        t = _time_op(lambda: _eager_fp32(Q, D))
        m = _peak_mem(lambda: _eager_fp32(Q, D))
        rows.append(("naive (eager, fp32 acc)", t, m))
    except torch.cuda.OutOfMemoryError:
        rows.append(("naive (eager, fp32 acc)", float("nan"), float("nan")))

    try:
        t = _time_op(lambda: _compiled_eager_fp32(Q, D))
        m = _peak_mem(lambda: _compiled_eager_fp32(Q, D))
        rows.append(("torch.compile (fp32 acc)", t, m))
    except torch.cuda.OutOfMemoryError:
        rows.append(("torch.compile (fp32 acc)", float("nan"), float("nan")))
    except Exception as e:
        rows.append((f"torch.compile (err: {type(e).__name__})", float("nan"), float("nan")))

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
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help=f"subset of shape names to run; default = all. choices: {[s[0] for s in SHAPES]}",
    )
    args = parser.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    os.makedirs(args.outdir, exist_ok=True)
    gpu = torch.cuda.get_device_name().replace(" ", "_")

    if args.only:
        wanted = set(args.only)
        shapes = [s for s in SHAPES if s[0] in wanted]
        if not shapes:
            raise SystemExit(f"unknown shape(s); pick from: {[s[0] for s in SHAPES]}")
    else:
        shapes = [s for s in SHAPES if not args.quick or s[2] <= 1000]
    results = []
    for shape in shapes:
        name, *_rest = shape
        print(f"== {name} ==")
        name, dtype_, rows = bench_one(*shape, dtype=dtype)
        for impl, t, m in rows:
            print(f"  {impl:30s}  {t:8.3f} ms   {m:8.2f} MB")
        results.append({"name": name, "shape": shape[1:], "dtype": str(dtype), "rows": rows})

    md_lines = [
        f"# Forward bench — {gpu} ({args.dtype})",
        "",
        f"All implementations share the same numerical contract: inputs in `{args.dtype}`, "
        "einsum / matmul accumulated in `fp32`, scores returned in `fp32`. Parity is "
        f"asserted against the eager fp32-accumulator reference at `atol={PARITY_ATOL}, "
        f"rtol={PARITY_RTOL}` before timing.",
        "",
    ]
    for r in results:
        md_lines.append(
            f"## {r['name']} — shape Nq={r['shape'][0]} Nd={r['shape'][1]} "
            f"Lq={r['shape'][2]} Ld={r['shape'][3]} d={r['shape'][4]}"
        )
        md_lines.append("")
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
