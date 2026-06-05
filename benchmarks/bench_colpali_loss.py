"""Explicit-negative colpali loss benchmark — MaxSim isolation, no encoder.

Times a real ``colpali_engine`` explicit-negative loss head (``ColbertNegativeCELoss``
/ ``ColbertPairwiseNegativeCELoss``) forward+backward on synthetic embeddings,
with and without :func:`patch_colpali_engine`. The encoder is skipped (fake
embeddings), so this isolates exactly the pos (``maxsim_pairs``) + per-query-neg
(4-D ``maxsim``) path the patch fuses.

``--in-batch-weight 0`` (default) isolates the explicit pos/neg fusion this adds.
At the training default 0.5 the in-batch CE term materializes a ``[B, B, Lq, Ld]``
tensor that dominates (and OOMs vanilla on its own at large B) — a different,
already-shipped fusion — so it's off here by default.

    python benchmarks/bench_colpali_loss.py --sweep

Requires colpali_engine (CPU extra ``colpali``; on GPU install out-of-band — see
scripts/sky_colpali_compat_test.yaml for the cu128 dance).
"""

import argparse

import torch


def _fake_embeddings(*shape, device="cuda", dtype=torch.bfloat16):
    """Skip the encoder and synthesize per-token-normalized embeddings."""
    x = torch.randn(*shape, device=device, dtype=dtype)
    return torch.nn.functional.normalize(x, dim=-1).requires_grad_()


def bench(loss_kind, batch_size, n_neg, Lq, Ld, d, in_batch_weight, iters=20, warmup=3, patch=False):
    from colpali_engine.loss.late_interaction_losses import (
        ColbertNegativeCELoss,
        ColbertPairwiseNegativeCELoss,
    )

    from late_interaction_kernels import patch_colpali_engine, unpatch_colpali_engine

    cls = {"colbert_neg": ColbertNegativeCELoss, "pairwise_neg": ColbertPairwiseNegativeCELoss}[loss_kind]
    head = cls(temperature=0.02, normalize_scores=True, in_batch_term_weight=in_batch_weight).to("cuda")

    Q = _fake_embeddings(batch_size, Lq, d)
    pos_D = _fake_embeddings(batch_size, Ld, d)
    neg_D = _fake_embeddings(batch_size, n_neg, Ld, d)

    def step():
        for p in (Q, pos_D, neg_D):
            p.grad = None
        head(Q, pos_D, neg_D).backward()

    if patch:
        patch_colpali_engine()
    try:
        for _ in range(warmup):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            step()
        end.record()
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        return start.elapsed_time(end) / iters, peak_mb
    finally:
        if patch:
            unpatch_colpali_engine()


SWEEP = [
    # (label, loss, batch, n_neg, Lq, Ld, d) — Ld=1030 ~ ColPali visual tokens
    ("colbert-neg-B64-n4", "colbert_neg", 64, 4, 32, 1030, 128),
    ("colbert-neg-B128-n4", "colbert_neg", 128, 4, 32, 1030, 128),
    ("colbert-neg-B128-n8", "colbert_neg", 128, 8, 32, 1030, 128),
    ("colbert-neg-B256-n8", "colbert_neg", 256, 8, 32, 1030, 128),
    ("colbert-neg-B256-n16", "colbert_neg", 256, 16, 32, 1030, 128),
    ("pairwise-neg-B128-n8", "pairwise_neg", 128, 8, 32, 1030, 128),
    ("pairwise-neg-B256-n8", "pairwise_neg", 256, 8, 32, 1030, 128),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="Run the full shape sweep.")
    ap.add_argument(
        "--only",
        nargs="+",
        default=None,
        help=f"subset of sweep labels (implies --sweep); choices: {[s[0] for s in SWEEP]}",
    )
    ap.add_argument("--loss", choices=["colbert_neg", "pairwise_neg"], default="colbert_neg")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-neg", type=int, default=8)
    ap.add_argument("--lq", type=int, default=32)
    ap.add_argument("--ld", type=int, default=1030)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--in-batch-weight", type=float, default=0.0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    if args.only:
        wanted = set(args.only)
        cases = [s for s in SWEEP if s[0] in wanted]
        if not cases:
            raise SystemExit(f"unknown label(s); pick from: {[s[0] for s in SWEEP]}")
    elif args.sweep:
        cases = SWEEP
    else:
        cases = [("single", args.loss, args.batch_size, args.num_neg, args.lq, args.ld, args.d)]

    print(f"GPU: {torch.cuda.get_device_name()}  in_batch_weight={args.in_batch_weight}")
    print(
        f"{'shape':24s}  {'vanilla (ms)':>13s}  {'LIK (ms)':>10s}  "
        f"{'vanilla MB':>12s}  {'LIK MB':>10s}  {'speedup':>8s}"
    )
    print("-" * 92)
    for label, loss_kind, bsz, neg, Lq, Ld, d in cases:
        try:
            torch.cuda.empty_cache()
            t_slow, m_slow = bench(loss_kind, bsz, neg, Lq, Ld, d, args.in_batch_weight, patch=False)
            slow = f"{t_slow:13.2f}  {' ':>10s}  {m_slow:12.1f}"
        except torch.cuda.OutOfMemoryError:
            t_slow = m_slow = None
            slow = f"{'OOM':>13s}  {' ':>10s}  {'OOM':>12s}"
        torch.cuda.empty_cache()
        try:
            t_fast, m_fast = bench(loss_kind, bsz, neg, Lq, Ld, d, args.in_batch_weight, patch=True)
        except torch.cuda.OutOfMemoryError:
            print(f"{label:24s}  {slow}  {'OOM':>10s}")
            continue
        speedup = f"{t_slow / t_fast:7.2f}x" if t_slow else "vanilla-OOM"
        # Reprint with LIK column filled (the vanilla branch left it blank).
        vanilla_ms = f"{t_slow:13.2f}" if t_slow else f"{'OOM':>13s}"
        vanilla_mb = f"{m_slow:12.1f}" if m_slow else f"{'OOM':>12s}"
        print(f"{label:24s}  {vanilla_ms}  {t_fast:10.2f}  {vanilla_mb}  {m_fast:10.1f}  {speedup:>8s}")


if __name__ == "__main__":
    main()
