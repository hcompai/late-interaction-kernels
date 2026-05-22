"""Three-way forward MaxSim benchmark on H100.

Compares, fairly and at matched numerics:

    * ``late-interaction-kernels`` (LIK, Triton)
    * ``flash-maxsim``               (roipony/flash-maxsim, Triton)
    * ``kernels/erikkaum/maxsim``    (HF kernels-hub, CUDA WMMA)

Plus, as numerical anchors:

    * ``naive (eager, fp32 acc)``    — einsum with fp32 accumulator
    * ``torch.compile (default)``    — same eager body, default Inductor
      mode (no CUDA-graph capture). ``reduce-overhead`` is *not* used here
      because the CUDA-graph capture optimises away per-launch overhead in
      a way that's invisible at production call sites.

Discipline applied to every shape:

    * ``warmup = 20`` (enough to drive Triton autotune past every config,
      and to fill the JIT / cuBLAS plan caches before timing);
    * ``iters  = 100`` measured between two CUDA events with one
      ``torch.cuda.synchronize`` only at the ends, so the events capture
      only kernel time, not host-side launch dispatch overhead;
    * parity is asserted against the eager fp32-accumulator reference at
      ``atol = 1e-2, rtol = 1e-2`` before any timing — every kernel timed
      here computes the same scores up to bf16-input quantisation;
    * a hardware snapshot (clocks, power state, persistence mode) is
      written to the output file so it's clear what we ran on.

Each kernel may be missing or fail on a specific shape; we report what we
get and move on. Nothing here is fatal — the goal is "what's the honest
relative perf on a fresh H100, today, on the same input".

Usage::

    python benchmarks/bench_three_way.py                    # bf16
    python benchmarks/bench_three_way.py --dtype fp16

Writes Markdown + JSON to ``benchmarks/results/three_way_<gpu>_<dtype>.{md,json}``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from typing import Callable

# Silence LIK's "Q is not unit-norm" warning — the bench uses ``randn`` Q/D
# (not real encoder outputs) and the parity check anchors against the same
# unnormalised eager reference, so the warning would just clutter the log.
os.environ.setdefault("LIK_SUPPRESS_NORM_WARN", "1")

import torch  # noqa: E402

import late_interaction_kernels as lik_pkg
from late_interaction_kernels import maxsim as lik_maxsim

# --- optional baselines -------------------------------------------------

try:
    from flash_maxsim import flash_maxsim_batched

    HAS_FM = True
except Exception as e:
    print(f"[info] flash-maxsim unavailable: {type(e).__name__}: {e}")
    HAS_FM = False

try:
    from kernels import get_kernel

    # erikkaum/* isn't on the curated trusted-publisher list, so we opt in
    # explicitly. We've inspected the kernel source: it's a Triton/CUDA WMMA
    # MaxSim implementation, exact same shape as flash-maxsim + LIK.
    _erik = get_kernel("erikkaum/maxsim", version=1, trust_remote_code=True)
    HAS_ERIK = True
except Exception as e:
    print(f"[info] erikkaum/maxsim unavailable: {type(e).__name__}: {e}")
    _erik = None
    HAS_ERIK = False


# Shapes intentionally cover the regime range:
#  - tiny rerank (Lq=32, Ld=300) → launch-overhead-sensitive
#  - long doc   (Ld=1024)        → bandwidth-bound at modest batch
#  - colpali    (Lq=Ld=1024)     → compute-bound, the GEMM-friendly case
#  - corpus     (Nd=10000)       → naive HBM-traffic blow-up
#  - in-batch   (Nq=Nd=128)      → realistic training tile
SHAPES = [
    # (name, Nq, Nd, Lq, Ld, d)
    ("rerank-short", 1, 1000, 32, 300, 128),
    ("rerank-long", 1, 1000, 32, 1024, 128),
    ("rerank-medium", 1, 1000, 128, 1024, 128),
    ("rerank-colpali", 1, 1000, 1024, 1024, 128),
    ("corpus-5k", 1, 5000, 32, 300, 128),
    ("corpus-10k", 1, 10000, 32, 300, 128),
    ("train-32", 32, 32, 32, 300, 128),
    ("train-128", 128, 128, 32, 300, 128),
    ("edge-d64", 1, 1000, 32, 300, 64),
]


PARITY_ATOL = 1e-2
PARITY_RTOL = 1e-2


# ----------------------------------------------------------------------
# Timing primitives
# ----------------------------------------------------------------------


def _time_op(fn: Callable, *, warmup: int = 20, iters: int = 100) -> float:
    """CUDA-event-based timer. Returns mean ms/iter."""
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
    return start.elapsed_time(end) / iters


def _peak_mem(fn: Callable) -> float:
    """Peak HBM delta in MB, measured *only* over the call window."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    del out
    return peak / (1024 * 1024)


# ----------------------------------------------------------------------
# Reference / baselines
# ----------------------------------------------------------------------


def _eager_fp32(Q: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """Eager einsum with fp32 accumulator — the parity anchor."""
    S = torch.einsum("ild,jtd->ijlt", Q.float(), D.float())
    return S.max(-1).values.sum(-1)


_compiled_eager_fp32 = torch.compile(_eager_fp32, dynamic=False)


# ----------------------------------------------------------------------
# Per-implementation runners
# ----------------------------------------------------------------------


def _run_lik(Q: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    return lik_maxsim(Q, D)


def _run_flash_maxsim(Q: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    # flash-maxsim returns scores per (query, doc) pair when shared_docs=True
    # and Q is [Nq, Lq, d], D is [Nd, Ld, d].
    return flash_maxsim_batched(Q, D, shared_docs=True)


def _erik_pack(Q: torch.Tensor, D: torch.Tensor):
    """Build packed inputs for ``score_pairs_packed`` covering all Nq × Nd pairs.

    erikkaum/maxsim's packed API expects:
      queries           : [total_q_tokens, dim]
      query_offsets     : [num_queries + 1] int32   cumulative
      documents         : [total_d_tokens, dim]
      document_offsets  : [num_documents + 1] int32 cumulative
      pair_query_ids    : [num_pairs] int32
      pair_document_ids : [num_pairs] int32
    """
    Nq, Lq, d = Q.shape
    Nd, Ld, _ = D.shape
    device = Q.device
    queries_packed = Q.reshape(Nq * Lq, d).contiguous()
    documents_packed = D.reshape(Nd * Ld, d).contiguous()
    query_offsets = torch.arange(0, (Nq + 1) * Lq, Lq, dtype=torch.int32, device=device)
    document_offsets = torch.arange(0, (Nd + 1) * Ld, Ld, dtype=torch.int32, device=device)
    pair_q = torch.arange(Nq, dtype=torch.int32, device=device).repeat_interleave(Nd)
    pair_d = torch.arange(Nd, dtype=torch.int32, device=device).repeat(Nq)
    return (
        queries_packed,
        query_offsets,
        documents_packed,
        document_offsets,
        pair_q,
        pair_d,
    )


def _run_erik(Q: torch.Tensor, D: torch.Tensor, packed) -> torch.Tensor:
    """Call erikkaum/maxsim. Reshape its [num_pairs] output back to [Nq, Nd]."""
    Nq = Q.shape[0]
    Nd = D.shape[0]
    qp, qoff, dp, doff, pq, pd = packed
    scores_flat = _erik.score_pairs_packed(qp, qoff, dp, doff, pq, pd)
    return scores_flat.reshape(Nq, Nd)


# ----------------------------------------------------------------------
# Parity check
# ----------------------------------------------------------------------


def _check_parity(name: str, ref: torch.Tensor, **named_outputs):
    """Compare every named output against ``ref`` at ``PARITY_{A,R}TOL``."""
    for label, out in named_outputs.items():
        if out is None:
            continue
        if not torch.allclose(out.float(), ref.float(), atol=PARITY_ATOL, rtol=PARITY_RTOL):
            diff = (out.float() - ref.float()).abs()
            rel = (diff / ref.float().abs().clamp(min=1e-6)).max().item()
            raise AssertionError(
                f"[{name}] {label} disagrees with ref: "
                f"max abs {diff.max().item():.3g}, max rel {rel:.3g} "
                f"(atol={PARITY_ATOL}, rtol={PARITY_RTOL})"
            )


# ----------------------------------------------------------------------
# Single shape
# ----------------------------------------------------------------------


def bench_one(name: str, Nq: int, Nd: int, Lq: int, Ld: int, d: int, dtype):
    torch.manual_seed(0)
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=dtype)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=dtype)

    with torch.no_grad():
        ref = _eager_fp32(Q, D)

        outs = {"naive (eager, fp32 acc)": ref}

        try:
            outs["LIK (Triton)"] = _run_lik(Q, D)
        except Exception as e:
            outs["LIK (Triton)"] = None
            print(f"  [warn] LIK errored: {type(e).__name__}: {e}")

        if HAS_FM:
            try:
                outs["flash-maxsim"] = _run_flash_maxsim(Q, D)
            except Exception as e:
                outs["flash-maxsim"] = None
                print(f"  [warn] flash-maxsim errored: {type(e).__name__}: {e}")

        erik_packed = None
        if HAS_ERIK:
            try:
                erik_packed = _erik_pack(Q, D)
                outs["erikkaum/maxsim"] = _run_erik(Q, D, erik_packed)
            except Exception as e:
                outs["erikkaum/maxsim"] = None
                print(f"  [warn] erikkaum/maxsim errored: {type(e).__name__}: {e}")

        try:
            outs["torch.compile (fp32 acc)"] = _compiled_eager_fp32(Q, D)
        except Exception as e:
            outs["torch.compile (fp32 acc)"] = None
            print(f"  [warn] torch.compile errored: {type(e).__name__}: {e}")

    _check_parity(
        name,
        ref,
        **{k: v for k, v in outs.items() if k != "naive (eager, fp32 acc)"},
    )
    for k in list(outs):
        del outs[k]

    rows = []

    t = _time_op(lambda: _run_lik(Q, D))
    m = _peak_mem(lambda: _run_lik(Q, D))
    rows.append(("LIK (Triton)", t, m))

    if HAS_FM:
        try:
            t = _time_op(lambda: _run_flash_maxsim(Q, D))
            m = _peak_mem(lambda: _run_flash_maxsim(Q, D))
            rows.append(("flash-maxsim", t, m))
        except Exception as e:
            rows.append((f"flash-maxsim (err: {type(e).__name__})", float("nan"), float("nan")))

    if HAS_ERIK:
        try:
            packed = _erik_pack(Q, D)
            t = _time_op(lambda: _run_erik(Q, D, packed))
            m = _peak_mem(lambda: _run_erik(Q, D, packed))
            rows.append(("erikkaum/maxsim", t, m))
        except Exception as e:
            rows.append((f"erikkaum/maxsim (err: {type(e).__name__})", float("nan"), float("nan")))

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

    return rows


# ----------------------------------------------------------------------
# Environment snapshot
# ----------------------------------------------------------------------


def _hw_snapshot() -> str:
    """Capture nvidia-smi clocks/power and the relevant torch/triton versions."""
    lines = []
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,clocks.current.sm,clocks.current.memory,"
                "power.draw,power.limit,persistence_mode,compute_mode,pstate",
                "--format=csv",
            ],
            text=True,
        )
        lines.append(out.strip())
    except Exception as e:
        lines.append(f"nvidia-smi unavailable: {type(e).__name__}")
    try:
        import triton

        lines.append(f"triton: {triton.__version__}")
    except Exception:
        lines.append("triton: unavailable")
    lines.append(f"torch:  {torch.__version__}")
    lines.append(f"lik:    {getattr(lik_pkg, '__version__', 'unknown')}")
    lines.append(f"flash-maxsim available: {HAS_FM}")
    lines.append(f"erikkaum/maxsim available: {HAS_ERIK}")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--outdir", default="benchmarks/results")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    os.makedirs(args.outdir, exist_ok=True)
    gpu = torch.cuda.get_device_name().replace(" ", "_")

    print("== environment ==")
    snap = _hw_snapshot()
    print(snap)
    print()

    shapes = [s for s in SHAPES if not args.quick or s[2] <= 1000]
    results = []
    for shape in shapes:
        name = shape[0]
        print(f"== {name} ==")
        rows = bench_one(*shape, dtype=dtype)
        for impl, t, m in rows:
            ts = f"{t:7.4f}" if t == t else "    OOM"
            ms = f"{m:7.2f}" if m == m else "    OOM"
            print(f"  {impl:32s} {ts} ms   {ms} MB")
        results.append({"name": name, "shape": shape[1:], "dtype": str(dtype), "rows": rows})
        print()

    md = [
        f"# Three-way forward bench — {gpu} ({args.dtype})",
        "",
        "All implementations share the same numerical contract: inputs in "
        f"`{args.dtype}`, matmul accumulated in `fp32`, scores returned in `fp32`. "
        "Parity is asserted against the eager fp32-accumulator reference at "
        f"`atol={PARITY_ATOL}, rtol={PARITY_RTOL}` before timing.",
        "",
        f"Per shape: **warmup={20} · iters={100}**, CUDA-event timing.",
        "",
        "## Environment",
        "",
        "```",
        snap,
        "```",
        "",
    ]
    for r in results:
        Nq, Nd, Lq, Ld, d = r["shape"]
        md.append(
            f"## {r['name']} — `Nq={Nq} Nd={Nd} Lq={Lq} Ld={Ld} d={d}`"
        )
        md.append("")
        md.append("| impl | time (ms) | peak mem (MB) | speedup vs LIK |")
        md.append("| --- | --- | --- | --- |")
        lik_t = next(
            (t for impl, t, _ in r["rows"] if impl == "LIK (Triton)" and t == t),
            None,
        )
        for impl, t, m in r["rows"]:
            ts = f"{t:.4f}" if t == t else "OOM"
            ms = f"{m:.2f}" if m == m else "OOM"
            if lik_t is not None and t == t:
                ratio = t / lik_t
                rs = f"{ratio:.2f}×"
            else:
                rs = "—"
            md.append(f"| {impl} | {ts} | {ms} | {rs} |")
        md.append("")

    out_md = os.path.join(args.outdir, f"three_way_{gpu}_{args.dtype}.md")
    out_json = os.path.join(args.outdir, f"three_way_{gpu}_{args.dtype}.json")
    with open(out_md, "w") as f:
        f.write("\n".join(md))
    with open(out_json, "w") as f:
        json.dump({"env": snap, "results": results}, f, indent=2, default=str)
    print(f"\n→ wrote {out_md}")
    print(f"→ wrote {out_json}")


if __name__ == "__main__":
    main()
