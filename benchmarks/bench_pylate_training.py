"""End-to-end training step benchmark through PyLate's loss.

Compares an actual `pylate.losses.Contrastive` step with and without our
monkey-patch. Useful to see real wall-clock savings vs a ColBERT training loop.

    python benchmarks/bench_pylate_training.py --batch-size 64 --neg 2

Requires: `pip install pylate` plus a small ColBERT model (we use the default
all-MiniLM-L6-v2 pipeline since it's fast to load).
"""

from __future__ import annotations

import argparse

import torch


def _fake_embeddings(bsz, ntokens, d, device, dtype):
    """Skip the actual model forward and synthesize normalized embeddings."""
    x = torch.randn(bsz, ntokens, d, device=device, dtype=dtype)
    return torch.nn.functional.normalize(x, dim=-1)


def bench(batch_size, n_neg, Lq, Ld, d, iters=50, warmup=5, patch=False):
    from flash_colbert.pylate_compat import patch_pylate, unpatch_pylate

    if patch:
        patch_pylate()
    # Import AFTER patching so we get the patched function reference.
    import importlib

    import pylate.scores as pylate_scores
    importlib.reload(pylate_scores)
    colbert_scores = pylate_scores.colbert_scores

    q = _fake_embeddings(batch_size, Lq, d, "cuda", torch.float16).requires_grad_()
    pos = _fake_embeddings(batch_size, Ld, d, "cuda", torch.float16).requires_grad_()
    negs = [
        _fake_embeddings(batch_size, Ld, d, "cuda", torch.float16).requires_grad_()
        for _ in range(n_neg)
    ]
    d_mask = torch.ones(batch_size, Ld, device="cuda", dtype=torch.float16)

    def step():
        for p in [q, pos] + negs:
            if p.grad is not None:
                p.grad = None
        scores = torch.cat(
            [colbert_scores(q, p, d_mask) for p in [pos] + negs],
            dim=1,
        )
        labels = torch.arange(batch_size, device="cuda")
        loss = torch.nn.functional.cross_entropy(scores, labels)
        loss.backward()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        step()
    e.record()
    torch.cuda.synchronize()
    if patch:
        unpatch_pylate()
    return s.elapsed_time(e) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--neg", type=int, default=1)
    ap.add_argument("--lq", type=int, default=32)
    ap.add_argument("--ld", type=int, default=200)
    ap.add_argument("--d", type=int, default=128)
    args = ap.parse_args()

    torch.cuda.empty_cache()
    t_slow = bench(args.batch_size, args.neg, args.lq, args.ld, args.d, patch=False)
    torch.cuda.empty_cache()
    t_fast = bench(args.batch_size, args.neg, args.lq, args.ld, args.d, patch=True)

    print(f"PyLate Contrastive step:  batch={args.batch_size} neg={args.neg} Lq={args.lq} Ld={args.ld} d={args.d}")
    print(f"  vanilla:       {t_slow:.2f} ms/step")
    print(f"  flash-colbert: {t_fast:.2f} ms/step")
    print(f"  speedup:       {t_slow/t_fast:.2f}x")


if __name__ == "__main__":
    main()
