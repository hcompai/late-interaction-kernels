"""Quick parity bench used by ``scripts/sky_bench_regression_check.yaml``.

Times the LIK forward kernel against the eager fp32-accumulator naive
reference on the same shapes the README headline uses. Designed to run
twice on the same H100 — once with ``late-interaction-kernels==0.1.0``
installed, once with the current source tree — so we can compare the
two on identical hardware / torch / triton.
"""

import argparse
import json
import os

import torch

import late_interaction_kernels as lik

# v0.1.0 ships ``maxsim_inference``; HEAD ships ``maxsim`` with auto-skip
# on no-grad inputs. Both hit the same ``_run_forward(save_argmax=False)``
# code path, so the Triton kernel itself is the unit under test.
try:
    from late_interaction_kernels import maxsim_inference as run_fn

    USED = "maxsim_inference"
except ImportError:
    from late_interaction_kernels import maxsim as run_fn

    USED = "maxsim"


SHAPES = [
    ("text-short", 1, 1000, 32, 300, 128),
    ("text-long", 1, 1000, 32, 1024, 128),
    ("text-medium", 1, 1000, 128, 1024, 128),
    ("visual", 1, 1000, 1024, 1024, 128),
    ("corpus-5k", 1, 5000, 32, 300, 128),
    ("corpus-10k", 1, 10000, 32, 300, 128),
    ("train-batch", 32, 32, 32, 300, 128),
    ("train-batch-128", 128, 128, 32, 300, 128),
    ("large-d-512", 1, 1000, 32, 300, 512),
    ("large-d-1024", 1, 500, 32, 300, 1024),
    ("lateon-edge-rerank", 1, 1000, 32, 2048, 48),
    ("lateon-edge-big", 1, 4000, 32, 2048, 48),
    ("mxbai-edge", 1, 1000, 32, 300, 64),
    ("mxbai-edge-corpus", 1, 10000, 32, 300, 64),
]


def time_op(fn, warmup=5, iters=50):
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


def naive_fp32(Q, D):
    S = torch.einsum("ild,jtd->ijlt", Q.float(), D.float())
    return S.max(-1).values.sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[{args.label}] lik.__version__ = {lik.__version__}  using {USED}")

    results = []
    for name, Nq, Nd, Lq, Ld, d in SHAPES:
        torch.manual_seed(0)
        Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.bfloat16)
        D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.bfloat16)
        t_lik = time_op(lambda: run_fn(Q, D))
        try:
            t_naive = time_op(lambda: naive_fp32(Q, D), iters=25)
        except torch.cuda.OutOfMemoryError:
            t_naive = float("nan")
        ratio = t_naive / t_lik if t_naive == t_naive else float("nan")
        print(
            f"  {name:25s}  LIK {t_lik:8.4f} ms   naive {t_naive:8.4f} ms   speedup {ratio:6.2f}x"
        )
        results.append(
            {
                "name": name,
                "lik_ms": t_lik,
                "naive_ms": t_naive,
                "Nq": Nq,
                "Nd": Nd,
                "Lq": Lq,
                "Ld": Ld,
                "d": d,
            }
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {"label": args.label, "lik_version": lik.__version__, "used": USED, "results": results},
            f,
            indent=2,
        )
    print(f"[{args.label}] wrote {args.out}")


if __name__ == "__main__":
    main()
