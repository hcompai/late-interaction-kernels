"""Kernel-level microbenchmark on the `matmul -> mask -> max -> sum` pattern.

Many late-interaction retrieval stacks — plain PyTorch rerankers, PyLate,
and LightOn's FastPlaid (github.com/lightonai/fast-plaid) — all compute the
MaxSim score with the same underlying pattern at their scoring step:

    token_scores = padded_doc_embeddings.matmul(q.transpose(-2, -1))
                   .masked_fill(~mask, -9999.0)
    scores       = token_scores.max_dim(1).sum_dim(-1)

dispatched to `aten::bmm / aten::max / aten::sum`. `late-interaction-kernels`
fuses this pattern into a single Triton kernel. This bench measures the
effect of that fusion at the operation level, in isolation.

It is NOT a comparison of two complete retrieval systems. FastPlaid is a
full search engine (index, IVF probe, orchestration, rerank);
`late-interaction-kernels` is a set of Triton kernels for the scoring
math. They sit in different layers of the stack. The end-to-end numbers
below are provided purely so you can see what fraction of `search()` the
scoring step represents in a production pipeline.

This bench does two things:

1. **Isolated scoring step** at typical reranker shapes
   (`n_full_scores // 4 = 1024` candidate docs per query, which matches
   FastPlaid's defaults), comparing:

     (a) The same ops written in PyTorch: `matmul + masked_fill + max + sum`,
         which dispatch to the identical CUDA kernels libtorch calls.
     (b) `late-interaction-kernels.maxsim` (no-grad).

   This shows the operation-level effect of kernel fusion, independent of
   which library is calling it.

2. **End-to-end `fast_plaid.search()`** on a synthetic index at realistic
   sizes, for context only — so you can see that the exact-rerank MaxSim
   is a small fraction of a complete retrieval call, and understand that
   kernel-level fusion is most valuable in Python-side rerankers and
   long-document regimes where scoring is the bottleneck.

Usage
-----
    # On a CUDA node with fast-plaid + late-interaction-kernels installed:
    python benchmarks/bench_fastplaid.py
    python benchmarks/bench_fastplaid.py --skip-fastplaid   # isolated rerank only
    python benchmarks/bench_fastplaid.py --skip-isolated    # end-to-end only
"""
# ruff: noqa: F821  -- closures capture Q / D_ / d_mask from enclosing scope

import argparse
import gc
import json
import os
import time

import torch

# FastPlaid's real defaults (fast_plaid/search/fast_plaid.py):
#   n_full_scores = 4096
#   After pruning: n_full_scores // 4 = 1024 docs get the exact-rerank MaxSim
# Lq is usually 32 for queries. Ld varies per corpus.
N_RERANK_DOCS = 1024
LQ_DEFAULT = 32
D_MODEL = 128

# Rerank-step shapes to sweep. These are FastPlaid's actual exact-rerank shapes.
RERANK_SHAPES = [
    # (name,            n_docs, Lq,  Ld)
    ("short-200", 1024, 32, 200),  # typical BEIR / MSMARCO
    ("medium-512", 1024, 32, 512),  # typical long-ish passage
    ("long-1024", 1024, 32, 1024),  # long docs
    ("xlong-4096", 1024, 32, 4096),  # ModernColBERT docs
    ("xxlong-8192", 1024, 32, 8192),  # ModernColBERT max
    # Also sweep n_docs for xlong (memory-sensitive):
    ("xlong-4096-256", 256, 32, 4096),
    ("xlong-4096-2048", 2048, 32, 4096),
]


def fastplaid_maxsim_proxy(Q: torch.Tensor, D: torch.Tensor, d_mask: torch.Tensor) -> torch.Tensor:
    """Exact PyTorch transliteration of FastPlaid's Rust `colbert_score_reduce`.

    Mirrors rust/search/search.rs lines 385-402:
        scores = (D @ Q.T)                    # [N, Ld, Lq]
        scores.masked_fill_(~mask, -9999.0)
        max_over_doc_tokens, _ = scores.max(dim=1)   # [N, Lq]
        return max_over_doc_tokens.sum(dim=-1)       # [N]

    Same CUDA kernels as libtorch dispatches from Rust.
    """
    # Q: [Lq, d], D: [N, Ld, d], d_mask: [N, Ld] bool
    token_scores = D.matmul(Q.transpose(-2, -1))  # [N, Ld, Lq]
    padding = ~d_mask.unsqueeze(-1).expand_as(token_scores)
    token_scores = token_scores.masked_fill(padding, -9999.0)
    max_per_doc_token, _ = token_scores.max(dim=1)  # [N, Lq]
    return max_per_doc_token.sum(dim=-1)  # [N]


def lik_maxsim(Q: torch.Tensor, D: torch.Tensor, d_mask: torch.Tensor) -> torch.Tensor:
    """late-interaction-kernels equivalent. No autograd, no saved argmax."""
    from late_interaction_kernels import maxsim

    # late-interaction-kernels wants Q as [Nq, Lq, d]. FastPlaid's rerank is one query at a
    # time, so Nq=1. We squeeze back to [N] at the end to match the proxy.
    scores = maxsim(Q.unsqueeze(0), D, d_mask=d_mask)  # [1, N]
    return scores.squeeze(0)


def _timed(fn, warmup: int = 3, iters: int = 10) -> float:
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


def _peak(fn) -> float:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**3


def run_isolated(args) -> list[dict]:
    print(
        "\n=== (1) Isolated scoring step — PyTorch `bmm + mask + max + sum` vs late-interaction-kernels ==="
    )
    print("Shape: N docs reranked × Ld doc tokens × Lq=32 query tokens × d=128, fp16")
    print(
        f"{'shape':<22} {'fp_ms':>8} {'flash_ms':>10} {'speedup':>8} "
        f"{'fp_peak':>10} {'flash_peak':>12} {'mem_x':>7}"
    )

    rows = []
    for name, n, lq, ld in RERANK_SHAPES:
        Q = torch.nn.functional.normalize(
            torch.randn(lq, D_MODEL, device="cuda", dtype=torch.float16), dim=-1
        )
        D_ = torch.nn.functional.normalize(
            torch.randn(n, ld, D_MODEL, device="cuda", dtype=torch.float16), dim=-1
        )
        # Simulate ~80% valid doc tokens (real corpora are padded).
        lens = torch.randint(int(ld * 0.5), ld + 1, (n,), device="cuda")
        d_mask = torch.arange(ld, device="cuda")[None, :] < lens[:, None]

        row = {"shape": name, "n": n, "lq": lq, "ld": ld}

        try:
            fp_ms = _timed(lambda: fastplaid_maxsim_proxy(Q, D_, d_mask), iters=args.iters)
            fp_peak = _peak(lambda: fastplaid_maxsim_proxy(Q, D_, d_mask))
        except torch.cuda.OutOfMemoryError:
            fp_ms = float("nan")
            fp_peak = float("nan")
            torch.cuda.empty_cache()

        try:
            fl_ms = _timed(lambda: lik_maxsim(Q, D_, d_mask), iters=args.iters)
            fl_peak = _peak(lambda: lik_maxsim(Q, D_, d_mask))
        except torch.cuda.OutOfMemoryError:
            fl_ms = float("nan")
            fl_peak = float("nan")
            torch.cuda.empty_cache()

        # Correctness check (skip if either OOM'd):
        if fp_ms == fp_ms and fl_ms == fl_ms:
            a = fastplaid_maxsim_proxy(Q, D_, d_mask).float()
            b = lik_maxsim(Q, D_, d_mask).float()
            max_abs = (a - b).abs().max().item()
            rel = max_abs / (a.abs().max().item() + 1e-6)
            row["max_abs_err"] = max_abs
            row["rel_err"] = rel
            if rel > 1e-2:
                print(f"  WARNING: {name} rel err = {rel:.3e} (max abs {max_abs:.3e})")

        row.update(dict(fp_ms=fp_ms, flash_ms=fl_ms, fp_peak_gb=fp_peak, flash_peak_gb=fl_peak))
        rows.append(row)

        def r(a, b):
            if a != a or b != b or b == 0:
                return float("nan")
            return a / b

        print(
            f"{name:<22} {fp_ms:>8.3f} {fl_ms:>10.3f} {r(fp_ms, fl_ms):>7.2f}x "
            f"{fp_peak:>9.2f}G {fl_peak:>11.2f}G {r(fp_peak, fl_peak):>6.1f}x"
        )

        del Q, D_, d_mask
        torch.cuda.empty_cache()
    return rows


def build_synthetic_corpus(n_docs: int, ld: int, d: int, device: str):
    """Create a synthetic list of document embeddings. Varying doc length is
    what FastPlaid expects (it computes per-doc lengths for padding)."""
    docs = []
    gen = torch.Generator(device="cpu").manual_seed(0)
    for _ in range(n_docs):
        doc_len = int(torch.randint(max(1, ld // 2), ld + 1, (1,), generator=gen).item())
        v = torch.randn(doc_len, d, generator=gen, dtype=torch.float32)
        v = torch.nn.functional.normalize(v, dim=-1).half()
        docs.append(v)
    return docs


def run_fastplaid(args) -> list[dict]:
    print("\n=== (2) End-to-end fast_plaid.search() (context only, not a head-to-head) ===")
    try:
        from fast_plaid import search as fps
    except ImportError as e:
        print(f"  fast-plaid not installed, skipping. ({e})")
        print("  Install: pip install fast-plaid")
        return []

    rows = []
    # Keep corpus small-ish; we mainly want per-query latency at rerank shapes.
    corpus_sizes = [
        # (name, n_docs, ld)
        ("corpus-10k-200", 10_000, 200),
        ("corpus-10k-512", 10_000, 512),
        ("corpus-10k-1024", 10_000, 1024),
    ]
    if not args.small:
        corpus_sizes.append(("corpus-50k-512", 50_000, 512))

    for name, n_docs, ld in corpus_sizes:
        print(f"\n  building {name} ({n_docs:,} docs × {ld} toks)...")
        index_dir = os.path.join(args.outdir, f"fp_index_{name}")
        if os.path.exists(index_dir):
            import shutil

            shutil.rmtree(index_dir, ignore_errors=True)
        docs = build_synthetic_corpus(n_docs, ld, D_MODEL, "cuda")
        t0 = time.time()
        engine = fps.FastPlaid(index=index_dir, device="cuda")
        engine.create(documents_embeddings=docs, kmeans_niters=2)
        build_s = time.time() - t0
        print(f"  built in {build_s:.1f}s")

        # 100 queries, Lq=32
        n_queries = args.n_queries
        queries = torch.nn.functional.normalize(torch.randn(n_queries, LQ_DEFAULT, D_MODEL), dim=-1)

        # warmup
        engine.search(queries_embeddings=queries[:2], top_k=10, n_full_scores=4096, n_ivf_probe=8)

        for _ in range(args.warmup):
            engine.search(queries_embeddings=queries[:10], top_k=10, n_full_scores=4096, n_ivf_probe=8)

        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.iters):
            _ = engine.search(
                queries_embeddings=queries,
                top_k=10,
                n_full_scores=4096,
                n_ivf_probe=8,
            )
        torch.cuda.synchronize()
        total = (time.time() - t0) / args.iters
        per_q_ms = total * 1000 / n_queries
        print(f"  {name}: {total:.3f}s / {n_queries} queries = {per_q_ms:.2f} ms/query")
        rows.append(
            dict(shape=name, n_docs=n_docs, ld=ld, total_s=total, per_query_ms=per_q_ms, build_s=build_s)
        )

        del engine, docs
        gc.collect()
        torch.cuda.empty_cache()

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="benchmarks/results")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--small", action="store_true", help="skip the 50k corpus")
    ap.add_argument("--skip-isolated", action="store_true")
    ap.add_argument("--skip-fastplaid", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    gpu = torch.cuda.get_device_name().replace(" ", "_") if torch.cuda.is_available() else "cpu"

    out = {"time_s": time.time(), "gpu": gpu}
    if not args.skip_isolated:
        out["isolated"] = run_isolated(args)
    if not args.skip_fastplaid:
        out["fastplaid"] = run_fastplaid(args)

    fn = os.path.join(args.outdir, f"fastplaid_{gpu}.json")
    with open(fn, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {fn}")


if __name__ == "__main__":
    main()
