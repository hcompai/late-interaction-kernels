"""Compare grad_D paths: ``auto`` (default) vs ``unified`` vs ``lowmem`` vs naive.

Measures end-to-end step time (forward + backward) plus peak memory. The
forward is identical for both flash paths, so the delta isolates the
``grad_D`` path. ``"auto"`` picks ``"lowmem"`` (bf16 grads, ~½ peak memory,
deterministic) for the gradient-heavy shapes and ``"unified"`` (fastest,
fp32 atomics) for the rest.

Shapes match ``bench_backward.py`` plus a stressful retrieval shape
and a "hot bucket" synthetic where every query's argmax collapses to
a single doc-token — the worst case for the atomic scatter.
"""

import argparse
import json
import os
from contextlib import nullcontext

import torch

from late_interaction_kernels import maxsim


def _naive_score(Q, D):
    return torch.einsum("ild,jtd->ijlt", Q, D).max(-1).values.sum(-1)


SHAPES = [
    ("train-32", 32, 32, 32, 128, 128),
    ("train-64", 64, 64, 32, 128, 128),
    ("train-128", 128, 128, 32, 128, 128),
    ("train-256", 256, 256, 32, 128, 128),
    ("train-kd", 32, 8, 32, 300, 128),
    ("retrieval", 16, 512, 32, 300, 128),
    # Long-sequence regimes (ColPali-like).
    ("long-Lq", 4, 16, 1024, 64, 128),
    ("long-both", 4, 8, 512, 512, 128),
    # Tiny Ld — low parallelism, highest atomic contention per output cell.
    ("tiny-Ld", 64, 64, 32, 16, 128),
    ("huge-Nd", 16, 1024, 32, 128, 128),
]


def _bench(fn, warmup=5, iters=30):
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


def _peak_mb(fn) -> float:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2


def _make_step(Q, D, *, backward):
    def _step():
        if Q.grad is not None:
            Q.grad = None
            D.grad = None
        s = maxsim(Q, D, backward=backward)
        s.sum().backward()

    return _step


def _make_naive_step(Q, D):
    def _step():
        if Q.grad is not None:
            Q.grad = None
            D.grad = None
        s = _naive_score(Q.float(), D.float())
        s.sum().backward()

    return _step


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="benchmarks/results")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument(
        "--include-hot",
        action="store_true",
        help="add a synthetic hot-bucket shape (worst case for the atomic scatter)",
    )
    p.add_argument(
        "--only",
        nargs="+",
        default=None,
        help=f"subset of shape names to run; default = all. choices: {[s[0] for s in SHAPES] + ['hot-bucket']}",
    )
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    gpu = torch.cuda.get_device_name().replace(" ", "_")

    rows = []
    shapes = list(SHAPES)
    if args.include_hot:
        shapes.append(("hot-bucket", 32, 32, 32, 128, 128))
    if args.only:
        wanted = set(args.only)
        shapes = [s for s in shapes if s[0] in wanted]
        if not shapes:
            raise SystemExit(f"unknown shape(s); pick from: {[s[0] for s in SHAPES] + ['hot-bucket']}")

    print(
        f"{'shape':<12} {'Nq':>4} {'Nd':>4} {'Lq':>4} {'Ld':>5} {'d':>4}   "
        f"{'auto ms':>8} {'unified':>8} {'lowmem':>8} {'naive ms':>9}  "
        f"{'auto MB':>8}  {'auto×':>5} {'pick':>8}"
    )
    for name, Nq, Nd, Lq, Ld, d in shapes:
        Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float16, requires_grad=True)
        D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float16, requires_grad=True)

        if name == "hot-bucket":
            # Force every query's argmax to t=0 on each doc.
            with torch.no_grad():
                D[:, 0, :] = 100.0

        naive = _make_naive_step(Q, D)

        # The real selector lives in `_MaxSimFn.backward` (autograd.py):
        # `auto` picks `lowmem` for high-contention squares (and all KD
        # layouts) and `unified` otherwise.
        high_contention = Nq >= 256 and Nd >= 256 and Lq <= 64
        pick = "lowmem" if high_contention else "unified"

        t_auto = _bench(_make_step(Q, D, backward="auto"), iters=args.iters)
        t_unified = _bench(_make_step(Q, D, backward="unified"), iters=args.iters)
        t_lowmem = _bench(_make_step(Q, D, backward="lowmem"), iters=args.iters)
        peak_auto_mb = _peak_mb(_make_step(Q, D, backward="auto"))

        try:
            t_naive = _bench(naive, iters=max(5, args.iters // 3))
        except torch.cuda.OutOfMemoryError:
            t_naive = float("nan")

        auto_sp = t_naive / t_auto if t_naive == t_naive else float("nan")
        print(
            f"{name:<12} {Nq:>4} {Nd:>4} {Lq:>4} {Ld:>5} {d:>4}   "
            f"{t_auto:>8.2f} {t_unified:>8.2f} {t_lowmem:>8.2f} {t_naive:>9.2f}  "
            f"{peak_auto_mb:>8.1f}  {auto_sp:>5.2f} {pick:>8}"
        )
        rows.append(
            {
                "name": name,
                "Nq": Nq,
                "Nd": Nd,
                "Lq": Lq,
                "Ld": Ld,
                "d": d,
                "auto_ms": t_auto,
                "unified_ms": t_unified,
                "lowmem_ms": t_lowmem,
                "naive_ms": t_naive,
                "auto_peak_mb": peak_auto_mb,
                "lowmem_vs_unified": t_unified / t_lowmem if t_lowmem else None,
                "auto_pick": pick,
            }
        )

    out = os.path.join(args.outdir, f"backward_method_{gpu}.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"→ wrote {out}")


if __name__ == "__main__":
    # Keep a quieter autotune cache print.
    with nullcontext():
        main()
