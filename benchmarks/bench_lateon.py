"""LateOn-scale benchmark: ModernBERT-base ColBERT across doc lengths.

Covers the LateOn family (all ModernBERT-base, 149 M params, d=128):

  * ``lightonai/LateOn``         — Ld native 300, short-to-medium rerank
  * ``lightonai/LateOn-Code``    — Ld native 8192, long-context code retrieval

Also valid for the predecessor ``lightonai/GTE-ModernColBERT-v1`` (same
architecture and d=128). The kernel is sequence-length-agnostic; this
bench sweeps synthetic Ld ∈ {300, 2k, 4k, 8k, 16k} to characterize the
full envelope.

We measure:

  * Forward throughput vs naive einsum (for inference / reranking).
  * Full forward + backward step (training).
  * Memory footprint — the naive einsum materializes
    Nq·Nd·Lq·Ld = 8192·Nq·Nd·Lq fp32s, which blows up fast at 8k.

Shapes
------

  cb-rerank-300  Nq= 1, Nd=1000, Lq= 32, Ld= 300   : LateOn native reranking
  cb-rerank-8k   Nq= 1, Nd= 256, Lq= 32, Ld=8192   : LateOn-Code reranking
  cb-train-8k    Nq= 8, Nd=  16, Lq= 32, Ld=8192   : LateOn-Code training step
  cb-bigbatch-8k Nq=16, Nd=  32, Lq= 32, Ld=8192   : pushing batch at 8k
  cb-symm-4k     Nq= 4, Nd=   8, Lq= 32, Ld=4096   : halfway (common checkpoint length)
  cb-huge-doc    Nq= 1, Nd=  32, Lq= 32, Ld=16384  : stretch goal, 16k

The naive path will OOM at the largest shapes on an 80 GB H100 — that
is itself the story.
"""

import argparse
import json
import os

import torch

from late_interaction_kernels import maxsim

SHAPES = [
    # name             Nq    Nd    Lq    Ld
    # Baseline short-doc rerank.
    ("cb-rerank-300", 1, 1000, 32, 300),
    # 2k-4k: naive einsum still fits, so we get real speedup + memory ratios.
    ("cb-train-2k", 8, 16, 32, 2048),
    ("cb-train-4k", 8, 16, 32, 4096),
    ("cb-bigbatch-4k", 16, 32, 32, 4096),
    ("cb-rerank-4k", 1, 64, 32, 4096),
    # 8k: ModernColBERT scale. Naive OOMs at training batches.
    ("cb-train-8k", 8, 16, 32, 8192),
    ("cb-bigbatch-8k", 16, 32, 32, 8192),
    ("cb-rerank-8k", 1, 256, 32, 8192),
    # 16k: stretch.
    ("cb-huge-doc", 1, 32, 32, 16384),
]

D_MODEL = 128


def _naive_score(Q, D):
    return torch.einsum("ild,jtd->ijlt", Q, D).max(-1).values.sum(-1)


def _bench(fn, warmup=3, iters=10):
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


def _peak_mem_mb(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="benchmarks/results")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument(
        "--skip-naive-at", type=int, default=4096, help="skip naive for Ld above this threshold (OOM)"
    )
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    gpu = torch.cuda.get_device_name().replace(" ", "_")

    rows = []
    print(
        f"{'shape':<16} {'Nq':>3} {'Nd':>4} {'Lq':>3} {'Ld':>5}   "
        f"{'fwd auto':>8} {'fwd naive':>9}   "
        f"{'bwd atom':>8} {'bwd csr':>8} {'bwd auto':>8} {'bwd naive':>9}   "
        f"{'mem flash':>9} {'mem naive':>9}   {'auto':>5}"
    )

    for name, Nq, Nd, Lq, Ld in SHAPES:
        d = D_MODEL
        # Use fp16 (ColBERT's standard inference dtype).
        Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float16, requires_grad=True)
        D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float16, requires_grad=True)

        run_naive = Ld <= args.skip_naive_at
        big = (Nq * Nd * Lq * d) >= 100_000_000
        long_seq = Lq >= 1024 and Nq * Nd >= 16
        huge_corpus = Nd >= 1024
        pick = "csr" if (big or long_seq or huge_corpus) else "atom"

        # ---- Forward (inference, no autograd) ----
        def fwd_fast():
            maxsim(Q.detach(), D.detach())

        def fwd_naive():
            _naive_score(Q.detach().float(), D.detach().float())

        t_fwd_fast = _bench(fwd_fast, iters=args.iters)
        try:
            t_fwd_naive = _bench(fwd_naive, iters=max(3, args.iters // 2)) if run_naive else float("nan")
        except torch.cuda.OutOfMemoryError:
            t_fwd_naive = float("nan")

        # ---- Backward (training step) ----
        def _make_bwd_step(method):
            def _step():
                if Q.grad is not None:
                    Q.grad = None
                    D.grad = None
                maxsim(Q, D, backward=method).sum().backward()

            return _step

        bwd_step = _make_bwd_step("auto")

        def bwd_naive_step():
            if Q.grad is not None:
                Q.grad = None
                D.grad = None
            _naive_score(Q.float(), D.float()).sum().backward()

        t_bwd_atom = _bench(_make_bwd_step("atomic"), iters=args.iters)
        t_bwd_csr = _bench(_make_bwd_step("csr"), iters=args.iters)
        t_bwd_auto = _bench(bwd_step, iters=args.iters)

        try:
            t_bwd_naive = _bench(bwd_naive_step, iters=max(3, args.iters // 2)) if run_naive else float("nan")
        except torch.cuda.OutOfMemoryError:
            t_bwd_naive = float("nan")

        # ---- Peak memory ----
        try:
            mem_flash = _peak_mem_mb(bwd_step)
        except torch.cuda.OutOfMemoryError:
            mem_flash = float("nan")
        try:
            mem_naive = _peak_mem_mb(bwd_naive_step) if run_naive else float("nan")
        except torch.cuda.OutOfMemoryError:
            mem_naive = float("nan")

        print(
            f"{name:<16} {Nq:>3} {Nd:>4} {Lq:>3} {Ld:>5}   "
            f"{t_fwd_fast:>8.2f} {t_fwd_naive:>9.2f}   "
            f"{t_bwd_atom:>8.2f} {t_bwd_csr:>8.2f} {t_bwd_auto:>8.2f} {t_bwd_naive:>9.2f}   "
            f"{mem_flash:>9.1f} {mem_naive:>9.1f}   {pick:>5}"
        )

        rows.append(
            {
                "name": name,
                "Nq": Nq,
                "Nd": Nd,
                "Lq": Lq,
                "Ld": Ld,
                "d": d,
                "fwd_auto_ms": t_fwd_fast,
                "fwd_naive_ms": t_fwd_naive,
                "bwd_atomic_ms": t_bwd_atom,
                "bwd_csr_ms": t_bwd_csr,
                "bwd_auto_ms": t_bwd_auto,
                "bwd_naive_ms": t_bwd_naive,
                "mem_flash_MB": mem_flash,
                "mem_naive_MB": mem_naive,
                "auto_pick": pick,
            }
        )

    out = os.path.join(args.outdir, f"lateon_{gpu}.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
