"""Backward-pass microbenchmark."""

from __future__ import annotations

import argparse
import json
import os

import torch

from flash_colbert import maxsim


def _naive_score(Q, D):
    return torch.einsum("ild,jtd->ijlt", Q, D).max(-1).values.sum(-1)


SHAPES = [
    ("train-32", 32, 32, 32, 128, 128),
    ("train-64", 64, 64, 32, 128, 128),
    ("train-128", 128, 128, 32, 128, 128),
    ("train-kd", 32, 8, 32, 300, 128),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="benchmarks/results")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    gpu = torch.cuda.get_device_name().replace(" ", "_")

    rows = []
    for name, Nq, Nd, Lq, Ld, d in SHAPES:
        Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float16, requires_grad=True)
        D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float16, requires_grad=True)

        def fast_step():
            if Q.grad is not None:
                Q.grad = None
                D.grad = None
            s = maxsim(Q, D)
            s.sum().backward()

        def naive_step():
            if Q.grad is not None:
                Q.grad = None
                D.grad = None
            s = _naive_score(Q.float(), D.float())
            s.sum().backward()

        t_fast = _bench(fast_step)
        try:
            t_naive = _bench(naive_step)
        except torch.cuda.OutOfMemoryError:
            t_naive = float("nan")
        print(f"{name:12s}  fast={t_fast:7.2f} ms   naive={t_naive:7.2f} ms   speedup={t_naive/t_fast:5.1f}x")
        rows.append({"name": name, "fast_ms": t_fast, "naive_ms": t_naive})

    out = os.path.join(args.outdir, f"backward_{gpu}.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"→ wrote {out}")


if __name__ == "__main__":
    main()
