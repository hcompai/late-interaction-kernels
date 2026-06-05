"""Inference-only benchmark for edge / small-d ColBERT models.

Targets the regime where :func:`late_interaction_kernels.maxsim` (under
``torch.inference_mode``) structurally wins against the naive
``einsum → max → sum`` path:

* **small embedding dim** (d ∈ {48, 64}) — LightOn's ``LateOn-Code-edge``
  and Mixedbread's ``mxbai-edge`` family use tiny vectors, trading
  quality for memory + throughput.
* **long context** (Ld ∈ {1024, 4096, 8192}) — the small-d rerankers
  are designed for full-document reranking, where naive einsum has to
  materialize an O(Nq·Nd·Lq·Ld·d) similarity tensor.
* **high batch size** (Nd ∈ {64, 512, 4096, 16384}) — the setting where
  the library replaces a reranker stack. The fused kernel never
  materializes the [Nq, Nd, Lq, Ld] score tensor; naive does.

This complements ``bench_forward.py`` by focusing purely on
``inference_mode`` throughput + peak-memory, with OOM-tolerant naive
fallbacks.

    python benchmarks/kernels/bench_inference_edge.py
    python benchmarks/kernels/bench_inference_edge.py --dtype fp16 --outdir /tmp/lik

Writes a Markdown + JSON report under ``benchmarks/results/``.
"""

import argparse
import json
import os
from collections.abc import Iterable

import torch

from late_interaction_kernels import maxsim

# --------------------------------------------------------------------------- #
# Shapes                                                                       #
# --------------------------------------------------------------------------- #
#
# We sweep a compact grid rather than running every cross-product — each
# entry corresponds to a realistic deployment profile, not just "more
# numbers". The d=128 rows are the reference point: that's the shape
# where the README benchmarks live.


# name, Nq, Nd, Lq, Ld, d
SHAPES: list[tuple[str, int, int, int, int, int]] = [
    # -------- LateOn-Code-edge (d=48, 17M params, code rerank) ----------
    # Rerank a top-1k shortlist at short / long context.
    ("lateon-edge-Nd1k-Ld1024", 1, 1000, 32, 1024, 48),
    ("lateon-edge-Nd1k-Ld4096", 1, 1000, 32, 4096, 48),
    ("lateon-edge-Nd1k-Ld8192", 1, 1000, 32, 8192, 48),
    # High-BS rerank / small-corpus direct scoring.
    ("lateon-edge-Nd4k-Ld1024", 1, 4000, 32, 1024, 48),
    ("lateon-edge-Nd16k-Ld512", 1, 16000, 32, 512, 48),
    # -------- mxbai-edge (d=64, 17M, general rerank) --------------------
    ("mxbai-edge-Nd1k-Ld1024", 1, 1000, 32, 1024, 64),
    ("mxbai-edge-Nd1k-Ld4096", 1, 1000, 32, 4096, 64),
    ("mxbai-edge-Nd4k-Ld1024", 1, 4000, 32, 1024, 64),
    ("mxbai-edge-Nd16k-Ld512", 1, 16000, 32, 512, 64),
    # -------- Reference: ModernColBERT / ColBERTv2 style (d=128) --------
    # Same shapes as above for apples-to-apples delta.
    ("d128-Nd1k-Ld1024", 1, 1000, 32, 1024, 128),
    ("d128-Nd1k-Ld4096", 1, 1000, 32, 4096, 128),
    ("d128-Nd4k-Ld1024", 1, 4000, 32, 1024, 128),
    # -------- Very high BS (serving) ------------------------------------
    ("serving-Nd32k-Ld300", 1, 32000, 32, 300, 128),
]


# --------------------------------------------------------------------------- #
# Timing / memory helpers                                                      #
# --------------------------------------------------------------------------- #


def _time_op(fn, warmup: int = 5, iters: int = 50) -> float:
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
    return start.elapsed_time(end) / iters  # ms / call


def _peak_mem_mb(fn) -> float:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    del out
    torch.cuda.synchronize()
    return peak / (1024 * 1024)


def _naive_maxsim(Q: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """Materializes the [Nq, Nd, Lq, Ld] similarity tensor in fp32."""
    S = torch.einsum("ild,jtd->ijlt", Q.float(), D.float())
    return S.max(-1).values.sum(-1)


# --------------------------------------------------------------------------- #
# Per-shape benchmark                                                          #
# --------------------------------------------------------------------------- #


def bench_one(name: str, Nq: int, Nd: int, Lq: int, Ld: int, d: int, dtype: torch.dtype) -> dict:
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=dtype)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=dtype)

    rows: list[tuple[str, float, float]] = []

    # Bind Q / D as default args so ruff's scope analysis sees them and we
    # still benefit from Python's early-binding semantics inside the closure.

    # --- late-interaction-kernels (inference_mode) --------------------- #
    def _lik(Q=Q, D=D):
        with torch.inference_mode():
            return maxsim(Q, D)

    try:
        t_lik = _time_op(_lik)
        m_lik = _peak_mem_mb(_lik)
    except torch.cuda.OutOfMemoryError:
        t_lik, m_lik = float("nan"), float("nan")
    rows.append(("late-interaction-kernels", t_lik, m_lik))

    # --- naive (fp32 einsum) ------------------------------------------- #
    def _naive(Q=Q, D=D):
        with torch.inference_mode():
            return _naive_maxsim(Q, D)

    try:
        t_naive = _time_op(_naive, iters=10 if Nd * Ld >= 8_000_000 else 50)
        m_naive = _peak_mem_mb(_naive)
    except torch.cuda.OutOfMemoryError:
        t_naive, m_naive = float("nan"), float("nan")
    rows.append(("naive einsum (fp32)", t_naive, m_naive))

    speedup = (t_naive / t_lik) if (t_lik == t_lik and t_naive == t_naive) else float("nan")
    mem_save = (m_naive / m_lik) if (m_lik == m_lik and m_naive == m_naive and m_lik > 0) else float("nan")

    # Free allocations between shapes.
    del Q, D
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    return {
        "name": name,
        "shape": {"Nq": Nq, "Nd": Nd, "Lq": Lq, "Ld": Ld, "d": d},
        "dtype": str(dtype),
        "rows": rows,
        "speedup_x": speedup,
        "memory_ratio_x": mem_save,
    }


# --------------------------------------------------------------------------- #
# Entrypoint                                                                   #
# --------------------------------------------------------------------------- #


def _shapes_arg(iter_: Iterable[str]) -> list[tuple]:
    wanted = set(iter_)
    out = [s for s in SHAPES if s[0] in wanted]
    if not out:
        raise SystemExit(f"unknown shape(s); pick from: {[s[0] for s in SHAPES]}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--outdir", default="benchmarks/results")
    p.add_argument(
        "--only",
        nargs="+",
        default=None,
        help=f"subset of shape names to run; default = all. choices: {[s[0] for s in SHAPES]}",
    )
    p.add_argument("--quick", action="store_true", help="Drop the Ld=8192 / Nd=32k rows.")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable — this benchmark is GPU-only.")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    if args.only:
        shapes = _shapes_arg(args.only)
    elif args.quick:
        shapes = [s for s in SHAPES if s[4] <= 4096 and s[2] <= 4000]
    else:
        shapes = SHAPES

    os.makedirs(args.outdir, exist_ok=True)
    gpu = torch.cuda.get_device_name().replace(" ", "_")

    print(f"GPU: {gpu}   dtype: {args.dtype}   shapes: {len(shapes)}\n")

    results = []
    for shape in shapes:
        name = shape[0]
        print(f"== {name} ==  Nq={shape[1]} Nd={shape[2]} Lq={shape[3]} Ld={shape[4]} d={shape[5]}")
        r = bench_one(*shape, dtype=dtype)
        for impl, t, m in r["rows"]:
            ts = f"{t:8.3f} ms" if t == t else "     OOM"
            ms = f"{m:8.2f} MB" if m == m else "     OOM"
            print(f"  {impl:30s}  {ts}  {ms}")
        sx = r["speedup_x"]
        if sx == sx:
            print(f"  → {sx:.2f}× faster, {r['memory_ratio_x']:.1f}× less peak mem")
        print()
        results.append(r)

    # Markdown summary
    md = [f"# Inference bench (edge / long-context / high-BS) — {gpu} ({args.dtype})\n"]
    md.append(
        "Each row compares `maxsim` (Triton, fused, no score-tensor "
        "materialization) against a naive fp32 einsum → max → sum baseline, "
        "inside `torch.inference_mode()`.\n"
    )
    md.append(
        "| shape | Nq | Nd | Lq | Ld | d | lik (ms) | naive (ms) | speedup | lik mem (MB) | naive mem (MB) | mem save |"
    )
    md.append("| --- | ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:|")
    for r in results:
        lik_t = r["rows"][0][1]
        lik_m = r["rows"][0][2]
        nv_t = r["rows"][1][1]
        nv_m = r["rows"][1][2]
        sx = r["speedup_x"]
        mx = r["memory_ratio_x"]
        md.append(
            f"| {r['name']} | {r['shape']['Nq']} | {r['shape']['Nd']} | {r['shape']['Lq']} | "
            f"{r['shape']['Ld']} | {r['shape']['d']} | "
            f"{(f'{lik_t:.3f}' if lik_t == lik_t else 'OOM')} | "
            f"{(f'{nv_t:.3f}' if nv_t == nv_t else 'OOM')} | "
            f"{(f'{sx:.2f}×' if sx == sx else '—')} | "
            f"{(f'{lik_m:.1f}' if lik_m == lik_m else 'OOM')} | "
            f"{(f'{nv_m:.1f}' if nv_m == nv_m else 'OOM')} | "
            f"{(f'{mx:.1f}×' if mx == mx else '—')} |"
        )

    out_md = os.path.join(args.outdir, f"inference_edge_{gpu}_{args.dtype}.md")
    out_json = os.path.join(args.outdir, f"inference_edge_{gpu}_{args.dtype}.json")
    with open(out_md, "w") as f:
        f.write("\n".join(md) + "\n")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"→ wrote {out_md}")
    print(f"→ wrote {out_json}")


if __name__ == "__main__":
    main()
