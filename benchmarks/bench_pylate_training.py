"""End-to-end training step benchmark through PyLate's loss.

Compares an actual `pylate.losses.Contrastive` step with and without our
monkey-patch. Useful to see real wall-clock savings vs a ColBERT training loop.

    python benchmarks/bench_pylate_training.py --batch-size 64 --neg 2

Requires: `pip install pylate` plus a small ColBERT model (we use the default
all-MiniLM-L6-v2 pipeline since it's fast to load).
"""

import argparse

import torch


def _fake_embeddings(bsz, ntokens, d, device, dtype):
    """Skip the actual model forward and synthesize normalized embeddings."""
    x = torch.randn(bsz, ntokens, d, device=device, dtype=dtype)
    return torch.nn.functional.normalize(x, dim=-1)


def bench(batch_size, n_neg, Lq, Ld, d, iters=50, warmup=5, patch=False):
    from late_interaction_kernels.pylate_compat import patch_pylate, unpatch_pylate

    if patch:
        patch_pylate()
    # Import AFTER patching so we get the patched function reference.
    import importlib

    import pylate.scores as pylate_scores

    importlib.reload(pylate_scores)
    colbert_scores = pylate_scores.colbert_scores

    q = _fake_embeddings(batch_size, Lq, d, "cuda", torch.float16).requires_grad_()
    pos = _fake_embeddings(batch_size, Ld, d, "cuda", torch.float16).requires_grad_()
    negs = [_fake_embeddings(batch_size, Ld, d, "cuda", torch.float16).requires_grad_() for _ in range(n_neg)]
    d_mask = torch.ones(batch_size, Ld, device="cuda", dtype=torch.float16)

    def step():
        for p in [q, pos] + negs:
            if p.grad is not None:
                p.grad = None
        scores = torch.cat(
            [colbert_scores(q, p, documents_mask=d_mask) for p in [pos] + negs],
            dim=1,
        )
        labels = torch.arange(batch_size, device="cuda")
        loss = torch.nn.functional.cross_entropy(scores, labels)
        loss.backward()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        step()
    e.record()
    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    if patch:
        unpatch_pylate()
    return s.elapsed_time(e) / iters, peak_mb


SWEEP = [
    # (label, batch, neg, Lq, Ld, d)
    ("pylate-B16-neg1", 16, 1, 32, 200, 128),
    ("pylate-B32-neg1", 32, 1, 32, 200, 128),
    ("pylate-B64-neg1", 64, 1, 32, 200, 128),
    ("pylate-B128-neg1", 128, 1, 32, 200, 128),
    ("long-doc-B32", 32, 1, 32, 1024, 128),
    ("colpali-B4", 4, 1, 1024, 1024, 128),
    ("edge-d48-B64", 64, 1, 32, 256, 48),
    ("edge-d64-B64", 64, 1, 32, 256, 64),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="Run the full training-shape sweep.")
    ap.add_argument(
        "--only",
        nargs="+",
        default=None,
        help=(f"subset of sweep labels to run (implies --sweep); choices: {[s[0] for s in SWEEP]}"),
    )
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--neg", type=int, default=1)
    ap.add_argument("--lq", type=int, default=32)
    ap.add_argument("--ld", type=int, default=200)
    ap.add_argument("--d", type=int, default=128)
    args = ap.parse_args()

    if args.only:
        wanted = set(args.only)
        cases = [s for s in SWEEP if s[0] in wanted]
        if not cases:
            raise SystemExit(f"unknown label(s); pick from: {[s[0] for s in SWEEP]}")
    elif args.sweep:
        cases = SWEEP
    else:
        cases = [("single", args.batch_size, args.neg, args.lq, args.ld, args.d)]

    print(
        f"{'shape':22s}  {'vanilla (ms)':>13s}  {'ours (ms)':>10s}  "
        f"{'vanilla MB':>12s}  {'ours MB':>10s}  {'speedup':>8s}"
    )
    print("-" * 90)
    for label, bsz, neg, Lq, Ld, d in cases:
        try:
            torch.cuda.empty_cache()
            t_slow, m_slow = bench(bsz, neg, Lq, Ld, d, patch=False)
            torch.cuda.empty_cache()
            t_fast, m_fast = bench(bsz, neg, Lq, Ld, d, patch=True)
        except torch.cuda.OutOfMemoryError:
            print(f"{label:22s}  OOM")
            continue
        print(
            f"{label:22s}  {t_slow:13.2f}  {t_fast:10.2f}  "
            f"{m_slow:12.1f}  {m_fast:10.1f}  {t_slow / t_fast:7.2f}x"
        )


if __name__ == "__main__":
    main()
