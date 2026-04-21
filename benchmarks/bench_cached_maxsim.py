"""Isolated CachedContrastive MaxSim benchmark.

``pylate.losses.CachedContrastive`` chunks the MaxSim into
``(batch_size/mini_batch_size)**2`` Python-level ``colbert_scores`` calls to
avoid OOM (see pylate/losses/cached_contrastive.py: *"We chunk the scores
computation to avoid OOM because MaxSim can get expensive with large batch
sizes/long documents"*).

This bench mirrors that exact chunked pattern on synthesized embeddings (no
encoder), so we can measure the delta that late-interaction-kernels gives on the *MaxSim
step only*. The encoder forward+backward is unchanged regardless of patch,
so stripping it out gives us the honest kernel delta at the shapes that
matter.

Shapes tested:

  * bs=64 / 128 / 256 × Ld=2048 / 4096 / 8192  — the LightOn Reason recipe
  * Always ``Lq=128, d=128``, mini_batch_size=32
  * Embeddings are fp16; scores accumulate in fp32

Metrics reported:

  * ``fwd``                     forward-only wall time (ms), 50-iter median after 5 warmup
  * ``fwd+bwd``                 full forward + backward (``scores.sum().backward()``) (ms)
  * ``peak``                    peak CUDA memory during fwd+bwd (GB)
  * ``tiles``                   number of inner ``colbert_scores`` tiles vanilla makes

Usage
-----
    python benchmarks/bench_cached_maxsim.py
    python benchmarks/bench_cached_maxsim.py --only flash   # skip vanilla when naive OOMs
"""

# ruff: noqa: F821  -- closures capture Q / D_ / d_mask from enclosing run_shape scope

from __future__ import annotations

import argparse
import gc
import json
import os
import time

import torch

SHAPES = [
    # (name,            batch, mini, Lq,  Ld)
    ("reason-bs64-2k", 64, 32, 128, 2048),
    ("reason-bs64-4k", 64, 32, 128, 4096),
    ("reason-bs64-8k", 64, 32, 128, 8192),
    ("reason-bs128-2k", 128, 32, 128, 2048),
    ("reason-bs128-4k", 128, 32, 128, 4096),
    ("reason-bs128-8k", 128, 32, 128, 8192),
    ("reason-bs256-2k", 256, 32, 128, 2048),
    ("reason-bs256-4k", 256, 32, 128, 4096),
    ("reason-bs256-8k", 256, 32, 128, 8192),
]
D_MODEL = 128


def _synth(batch: int, L: int, d: int, device, dtype=torch.float16):
    """Normalized fake embeddings with grad tracking."""
    x = torch.randn(batch, L, d, device=device, dtype=dtype, requires_grad=False)
    x = torch.nn.functional.normalize(x, dim=-1)
    return x.detach().requires_grad_(True)


def _timed(closure, iters=50, warmup=5):
    for _ in range(warmup):
        closure()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        closure()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def _chunked_vanilla(Q, D, d_mask, mini_batch_size: int):
    """Exact shape of pylate.losses.CachedContrastive's double loop over colbert_scores."""
    from pylate.scores import colbert_scores  # vanilla import, not patched

    bs = Q.shape[0]
    rows = []
    for begin in range(0, bs, mini_batch_size):
        end = min(begin + mini_batch_size, bs)
        tiles = []
        for g_start in range(0, bs, mini_batch_size):
            g_end = min(g_start + mini_batch_size, bs)
            tiles.append(
                colbert_scores(
                    Q[begin:end],
                    D[g_start:g_end],
                    d_mask[g_start:g_end],
                )
            )
        rows.append(torch.cat(tiles, dim=1))
    return torch.cat(rows, dim=0)


def _flash_one_call(Q, D, d_mask):
    from late_interaction_kernels import maxsim

    return maxsim(Q, D, d_mask=d_mask.bool() if d_mask is not None else None)


def run_shape(name, batch, mini, Lq, Ld, variants, iters, warmup):
    row = {"shape": name, "batch": batch, "mini": mini, "Lq": Lq, "Ld": Ld}

    for variant in variants:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        Q = _synth(batch, Lq, D_MODEL, "cuda")
        D_ = _synth(batch, Ld, D_MODEL, "cuda")
        d_mask = torch.ones(batch, Ld, device="cuda", dtype=torch.float16)

        def _fwd():
            if variant == "vanilla":
                out = _chunked_vanilla(Q, D_, d_mask, mini)
            else:
                out = _flash_one_call(Q, D_, d_mask)
            out.sum()

        def _fwdbwd():
            Q.grad = None
            D_.grad = None
            if variant == "vanilla":
                out = _chunked_vanilla(Q, D_, d_mask, mini)
            else:
                out = _flash_one_call(Q, D_, d_mask)
            out.sum().backward()

        try:
            fwd_ms = _timed(_fwd, iters=iters, warmup=warmup)
            torch.cuda.reset_peak_memory_stats()
            fwdbwd_ms = _timed(_fwdbwd, iters=iters, warmup=warmup)
            peak_gb = torch.cuda.max_memory_allocated() / 1024**3
            row[f"{variant}_fwd"] = fwd_ms
            row[f"{variant}_fwdbwd"] = fwdbwd_ms
            row[f"{variant}_peak"] = peak_gb
        except torch.cuda.OutOfMemoryError:
            row[f"{variant}_fwd"] = float("nan")
            row[f"{variant}_fwdbwd"] = float("nan")
            row[f"{variant}_peak"] = float("nan")
            row[f"{variant}_err"] = "OOM"

        del Q, D_, d_mask
        gc.collect()
        torch.cuda.empty_cache()

    row["tiles"] = (batch // mini) ** 2
    return row


def fmt(v):
    return f"{v:>7.2f}" if isinstance(v, float) and v == v else "    OOM"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="benchmarks/results")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--only", choices=["both", "vanilla", "flash"], default="both")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    gpu = torch.cuda.get_device_name().replace(" ", "_")

    variants = ["vanilla", "flash"] if args.only == "both" else [args.only]

    print(
        f"{'shape':<20} {'tiles':>5} "
        f"{'v_fwd':>8} {'v_bwd':>8} {'v_peak_GB':>10} "
        f"{'f_fwd':>8} {'f_bwd':>8} {'f_peak_GB':>10} "
        f"{'fwd_x':>6} {'bwd_x':>6} {'mem_x':>6}"
    )
    rows = []
    for name, batch, mini, Lq, Ld in SHAPES:
        row = run_shape(name, batch, mini, Lq, Ld, variants, iters=args.iters, warmup=args.warmup)
        rows.append(row)

        v_fwd = row.get("vanilla_fwd", float("nan"))
        v_bwd = row.get("vanilla_fwdbwd", float("nan"))
        v_peak = row.get("vanilla_peak", float("nan"))
        f_fwd = row.get("flash_fwd", float("nan"))
        f_bwd = row.get("flash_fwdbwd", float("nan"))
        f_peak = row.get("flash_peak", float("nan"))

        def ratio(a, b):
            if a != a or b != b or b == 0:
                return float("nan")
            return a / b

        fwd_x = ratio(v_fwd, f_fwd)
        bwd_x = ratio(v_bwd, f_bwd)
        mem_x = ratio(v_peak, f_peak)
        print(
            f"{name:<20} {row['tiles']:>5} "
            f"{fmt(v_fwd)} {fmt(v_bwd)} {fmt(v_peak)} "
            f"{fmt(f_fwd)} {fmt(f_bwd)} {fmt(f_peak)} "
            f"{fmt(fwd_x)} {fmt(bwd_x)} {fmt(mem_x)}"
        )

    fn = os.path.join(args.outdir, f"cached_maxsim_{gpu}.json")
    with open(fn, "w") as f:
        json.dump({"time_s": time.time(), "rows": rows}, f, indent=2)
    print(f"\nwrote {fn}")


if __name__ == "__main__":
    main()
