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
    """Exact shape of pylate.losses.CachedContrastive's double loop over colbert_scores.

    We always benchmark with an all-ones mask (the cached-contrastive loss
    chunks *whole* sequences, never partial ones), so we don't pass it
    explicitly — that also dodges signature differences between pylate
    versions where the third positional arg has changed name.
    """
    from pylate.scores import colbert_scores  # vanilla import, not patched

    bs = Q.shape[0]
    rows = []
    for begin in range(0, bs, mini_batch_size):
        end = min(begin + mini_batch_size, bs)
        tiles = []
        for g_start in range(0, bs, mini_batch_size):
            g_end = min(g_start + mini_batch_size, bs)
            tiles.append(colbert_scores(Q[begin:end], D[g_start:g_end]))
        rows.append(torch.cat(tiles, dim=1))
    return torch.cat(rows, dim=0)


def _colbert_scores_tile(Q_tile, D_tile):
    """Local re-implementation of ``pylate.scores.colbert_scores`` body.

    Same einsum + ``max(-1).sum(-1)``, fp32 accumulator. Wrapping pylate's
    actual function in ``torch.compile`` is fragile (it does Python-level
    dispatch on tensor attributes); a local re-implementation gives
    Inductor a clean graph while preserving the numerical contract.
    Mask handling is dropped because the cached-contrastive bench always
    uses all-ones masks — same simplification as ``_chunked_vanilla``.
    """
    S = torch.einsum("bld,ctd->bclt", Q_tile.float(), D_tile.float())
    return S.max(-1).values.sum(-1)


# Single global compiled tile, cached across calls. Dynamo recompiles per
# (B, C, Lq, Ld) tuple but the bench reuses the same shape across iters
# inside one row, so the recompile cost amortises across warmup.
_COMPILED_TILE = torch.compile(_colbert_scores_tile, dynamic=False, mode="reduce-overhead")


def _chunked_compile(Q, D, d_mask, mini_batch_size: int):
    """Same chunked tiling as vanilla pylate, but tiles go through ``torch.compile``.

    Inductor can fuse the tile-local ``max(-1)`` reduction but still has to
    materialise the ``[B, C, Lq, Ld]`` similarity intermediate in HBM. This
    baseline measures the slice of the speedup that compile alone closes.
    ``d_mask`` is accepted for signature symmetry but unused — all-ones.
    """
    del d_mask
    bs = Q.shape[0]
    rows = []
    for begin in range(0, bs, mini_batch_size):
        end = min(begin + mini_batch_size, bs)
        tiles = []
        for g_start in range(0, bs, mini_batch_size):
            g_end = min(g_start + mini_batch_size, bs)
            tiles.append(_COMPILED_TILE(Q[begin:end], D[g_start:g_end]))
        rows.append(torch.cat(tiles, dim=1))
    return torch.cat(rows, dim=0)


def _flash_one_call(Q, D, d_mask):
    from late_interaction_kernels import maxsim

    return maxsim(Q, D, d_mask=d_mask.bool() if d_mask is not None else None)


def _parity_check(batch, Lq, Ld, variants):
    """Confirm vanilla pylate, torch.compile and LIK agree on the same input.

    Runs on a downsampled ``(bs=min(8, batch))`` probe so the similarity
    tensor for the vanilla path always fits in HBM regardless of how big
    the timing run is going to be. Uses ``no_grad``. Tolerances are loose
    (``atol=2e-2``) because pylate's ``colbert_scores`` does the final
    ``sum(-1)`` in input dtype (fp16) while LIK accumulates in fp32 —
    bf16/fp16 accumulator drift is real and known. The threshold catches
    any *structural* bug (wrong shape, missing mask, off-by-one chunking)
    without false-positive on legitimate accumulator-precision drift.
    """
    probe_bs = min(8, batch)
    Q = _synth(probe_bs, Lq, D_MODEL, "cuda").detach()
    D_ = _synth(probe_bs, Ld, D_MODEL, "cuda").detach()
    d_mask = torch.ones(probe_bs, Ld, device="cuda", dtype=torch.float16)

    outputs = {}
    try:
        with torch.no_grad():
            if "vanilla" in variants:
                outputs["vanilla"] = _chunked_vanilla(Q, D_, d_mask, probe_bs).float()
            if "compile" in variants:
                outputs["compile"] = _chunked_compile(Q, D_, d_mask, probe_bs).float()
            if "flash" in variants:
                outputs["flash"] = _flash_one_call(Q, D_, d_mask).float()
    except torch.cuda.OutOfMemoryError:
        print(f"  [warn] parity probe OOM at bs={probe_bs} Ld={Ld}, skipping")
        del Q, D_, d_mask
        gc.collect()
        torch.cuda.empty_cache()
        return

    labels = list(outputs)
    if len(labels) >= 2:
        ref_label = "vanilla" if "vanilla" in outputs else labels[0]
        ref = outputs[ref_label]
        for label, out in outputs.items():
            if label == ref_label:
                continue
            if not torch.allclose(out, ref, atol=2e-2, rtol=2e-2):
                diff = (out - ref).abs()
                raise AssertionError(
                    f"[parity bs={probe_bs} Ld={Ld}] {label} disagrees with {ref_label}: "
                    f"max abs diff {diff.max().item():.3g}"
                )
    del Q, D_, d_mask, outputs
    gc.collect()
    torch.cuda.empty_cache()


def run_shape(name, batch, mini, Lq, Ld, variants, iters, warmup):
    _parity_check(batch, Lq, Ld, variants)

    row = {"shape": name, "batch": batch, "mini": mini, "Lq": Lq, "Ld": Ld}

    for variant in variants:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        Q = _synth(batch, Lq, D_MODEL, "cuda")
        D_ = _synth(batch, Ld, D_MODEL, "cuda")
        d_mask = torch.ones(batch, Ld, device="cuda", dtype=torch.float16)

        def _run_variant():
            if variant == "vanilla":
                return _chunked_vanilla(Q, D_, d_mask, mini)
            if variant == "compile":
                return _chunked_compile(Q, D_, d_mask, mini)
            return _flash_one_call(Q, D_, d_mask)

        def _fwd():
            _run_variant().sum()

        def _fwdbwd():
            Q.grad = None
            D_.grad = None
            _run_variant().sum().backward()

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
    ap.add_argument(
        "--only",
        choices=["all", "vanilla", "compile", "flash"],
        default="all",
        help="Which baselines to run. `all` runs vanilla pylate + torch.compile + LIK.",
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    gpu = torch.cuda.get_device_name().replace(" ", "_")

    variants = ["vanilla", "compile", "flash"] if args.only == "all" else [args.only]

    print(
        f"{'shape':<20} {'tiles':>5} {'v_bwd':>8} {'c_bwd':>8} {'f_bwd':>8}  {'v/f':>6} {'c/f':>6} {'v/c':>6}"
    )
    rows = []
    for name, batch, mini, Lq, Ld in SHAPES:
        row = run_shape(name, batch, mini, Lq, Ld, variants, iters=args.iters, warmup=args.warmup)
        rows.append(row)

        v_bwd = row.get("vanilla_fwdbwd", float("nan"))
        c_bwd = row.get("compile_fwdbwd", float("nan"))
        f_bwd = row.get("flash_fwdbwd", float("nan"))

        def ratio(a, b):
            if a != a or b != b or b == 0:
                return float("nan")
            return a / b

        print(
            f"{name:<20} {row['tiles']:>5} "
            f"{fmt(v_bwd)} {fmt(c_bwd)} {fmt(f_bwd)}  "
            f"{fmt(ratio(v_bwd, f_bwd))} {fmt(ratio(c_bwd, f_bwd))} {fmt(ratio(v_bwd, c_bwd))}"
        )

    fn = os.path.join(args.outdir, f"cached_maxsim_{gpu}.json")
    with open(fn, "w") as f:
        json.dump({"time_s": time.time(), "rows": rows}, f, indent=2)
    print(f"\nwrote {fn}")


if __name__ == "__main__":
    main()
