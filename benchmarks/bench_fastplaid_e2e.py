"""End-to-end fast-plaid rerank comparison.

Fast-plaid's search engine is monolithic Rust, so we can't swap the scoring
loop from Python without forking the crate. Instead we do the next-best
thing: build a real fast-plaid index, then time two pipelines on the same
compressed bytes:

  1. ``engine.search()`` — fast-plaid's full Python + Rust + tch path.
     Includes IVF probing + scoring + top-k. Everything behind the public
     Python API.

  2. ``maxsim_residual_varlen`` on fast-plaid's on-disk compressed tensors.
     This is the scoring slice fast-plaid does internally, rewritten as a
     single fused Triton kernel on ragged (no-pad) inputs.

Fast-plaid serializes the index to standardized ``.npy`` files
(``centroids.npy``, ``bucket_weights.npy``, ``merged_codes.npy``,
``merged_residuals.npy``, ``doclens.*.json``) so we can read them
directly and feed them to our kernel — no Rust changes needed.

What the numbers mean
---------------------

* ``engine.search()`` — user-visible latency (IVF probe + full rerank +
  top-k). This is the number you care about if you call fast-plaid today.
* ``lik_full_rerank`` — our kernel scoring against *all* docs in the
  index. Equivalent to "rerank over the whole corpus" — useful as an
  upper bound on the rerank cost for small indices.
* ``lik_partial_rerank`` — our kernel scoring against a fast-plaid-shaped
  candidate slice (e.g., ``n_full_scores=4096`` random docs), which is
  what fast-plaid actually reranks per query after IVF probing.

If ``lik_partial_rerank`` is comfortably below ``engine.search()``, the
kernel is a viable drop-in for fast-plaid's internal scoring loop.

Usage
-----

    python benchmarks/bench_fastplaid_e2e.py                # default: 10k docs
    python benchmarks/bench_fastplaid_e2e.py --small        # 5k docs, fewer shapes
    python benchmarks/bench_fastplaid_e2e.py --skip-fastplaid  # only our kernel
"""

import argparse
import gc
import glob
import json
import os
import shutil
import time

import numpy as np
import torch

D_MODEL = 128
LQ = 32


def build_corpus(n_docs: int, ld_max: int, seed: int = 0):
    """Synthesize variable-length fp16 doc embeddings the way fast-plaid expects.

    Each document is a ``[Li, D_MODEL]`` fp16 tensor with ``Li`` uniform in
    ``[ld_max/2, ld_max]`` and rows L2-normalized — matching a typical
    ColBERT / ColPali output.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    docs = []
    for _ in range(n_docs):
        li = int(torch.randint(max(1, ld_max // 2), ld_max + 1, (1,), generator=g).item())
        v = torch.randn(li, D_MODEL, generator=g, dtype=torch.float32)
        v = torch.nn.functional.normalize(v, dim=-1).half()
        docs.append(v)
    return docs


def read_fastplaid_index(index_path: str, device: str = "cuda") -> dict:
    """Read the compressed tensors fast-plaid serializes to disk.

    Returns the same quantities the Rust ``construct_index`` receives, so
    our kernel sees *exactly* the bytes fast-plaid scores against.
    """
    with open(os.path.join(index_path, "metadata.json")) as f:
        metadata = json.load(f)
    nbits = int(metadata["nbits"])
    num_chunks = int(metadata["num_chunks"])

    centroids = torch.from_numpy(np.load(os.path.join(index_path, "centroids.npy"))).to(
        device=device, dtype=torch.float16
    )
    bucket_weights = torch.from_numpy(np.load(os.path.join(index_path, "bucket_weights.npy"))).to(
        device=device, dtype=torch.float32
    )

    doc_lens = []
    for i in range(num_chunks):
        dl_path = os.path.join(index_path, f"doclens.{i}.json")
        with open(dl_path) as f:
            doc_lens.extend(json.load(f))
    doc_lengths = torch.tensor(doc_lens, device=device, dtype=torch.int64)

    # merged_codes / merged_residuals are flat [total_tokens, ...] — perfect
    # for our varlen kernel. They may include trailing padding rows; clip to
    # sum(doc_lengths) to be safe.
    merged_codes_path = os.path.join(index_path, "merged_codes.npy")
    merged_residuals_path = os.path.join(index_path, "merged_residuals.npy")
    if not os.path.exists(merged_codes_path):
        # Fall back to per-chunk files if fast-plaid didn't create the merged mmap
        # (it skips the merge for tiny indices).
        chunk_codes = sorted(glob.glob(os.path.join(index_path, "codes.*.npy")))
        chunk_resid = sorted(glob.glob(os.path.join(index_path, "residuals.*.npy")))
        codes_flat = torch.cat([torch.from_numpy(np.load(p)).view(-1).to(torch.int64) for p in chunk_codes])
        residuals_flat = torch.cat(
            [
                torch.from_numpy(np.load(p)).to(torch.uint8).view(-1, int((D_MODEL * nbits + 7) // 8))
                for p in chunk_resid
            ]
        )
    else:
        codes_flat = torch.from_numpy(np.load(merged_codes_path)).view(-1).to(torch.int64)
        residuals_flat = torch.from_numpy(np.load(merged_residuals_path)).to(torch.uint8)
        packed_dim = (D_MODEL * nbits + 7) // 8
        if residuals_flat.dim() == 1:
            residuals_flat = residuals_flat.view(-1, packed_dim)

    total = int(doc_lengths.sum().item())
    codes_flat = codes_flat[:total].to(device)
    residuals_flat = residuals_flat[:total].to(device)

    cu = torch.zeros(doc_lengths.numel() + 1, dtype=torch.int32, device=device)
    cu[1:] = doc_lengths.to(torch.int32).cumsum(0)

    return {
        "nbits": nbits,
        "centroids": centroids,
        "bucket_weights": bucket_weights,
        "codes_flat": codes_flat,
        "residuals_flat": residuals_flat,
        "doc_lengths": doc_lengths,
        "cu_seqlens_d": cu,
        "n_docs": doc_lengths.numel(),
        "total_tokens": total,
        "d": D_MODEL,
    }


def time_cuda(fn, warmup: int = 2, iters: int = 5) -> float:
    """Return median latency in ms, CUDA-synchronized."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def run_corpus(name: str, n_docs: int, ld_max: int, nbits: int, args):
    from late_interaction_kernels.plaid import maxsim_residual_varlen

    print(f"\n{'=' * 70}\n{name}: {n_docs:,} docs × up to {ld_max} toks, nbits={nbits}\n{'=' * 70}")

    idx_dir = os.path.join(args.outdir, f"fp_idx_{name}")
    if os.path.exists(idx_dir):
        shutil.rmtree(idx_dir, ignore_errors=True)

    t0 = time.time()
    docs = build_corpus(n_docs, ld_max)

    engine = None
    build_s = 0.0
    e2e_ms_per_q = None
    if not args.skip_fastplaid:
        try:
            from fast_plaid import search as fps
        except ImportError as e:
            print(f"  fast-plaid not importable: {e}")
        else:
            engine = fps.FastPlaid(index=idx_dir, device="cuda")
            t0 = time.time()
            engine.create(documents_embeddings=docs, kmeans_niters=2, nbits=nbits)
            build_s = time.time() - t0
            print(f"  built fast-plaid index in {build_s:.1f}s")

            queries = torch.nn.functional.normalize(torch.randn(args.n_queries, LQ, D_MODEL), dim=-1)

            def _search():
                _ = engine.search(
                    queries_embeddings=queries,
                    top_k=10,
                    n_full_scores=min(args.n_full_scores, n_docs),
                    n_ivf_probe=8,
                    show_progress=False,
                )

            e2e_ms_total = time_cuda(_search, warmup=args.warmup, iters=args.iters)
            e2e_ms_per_q = e2e_ms_total / args.n_queries
            print(f"  engine.search()        : {e2e_ms_per_q:.2f} ms/query")

    # For the kernel benchmark we need the compressed index. If we skipped
    # fast-plaid we still need *some* index → build a minimal one.
    if engine is None:
        from fast_plaid import search as fps

        engine = fps.FastPlaid(index=idx_dir, device="cuda")
        engine.create(documents_embeddings=docs, kmeans_niters=2, nbits=nbits)
        build_s = time.time() - t0

    del docs
    gc.collect()

    idx = read_fastplaid_index(idx_dir, device="cuda")
    print(
        f"  loaded index: {idx['n_docs']:,} docs, {idx['total_tokens']:,} tokens, "
        f"nbits={idx['nbits']}, centroids={idx['centroids'].shape}, "
        f"bucket_weights={idx['bucket_weights'].shape}"
    )

    Q = torch.nn.functional.normalize(torch.randn(1, LQ, D_MODEL, device="cuda"), dim=-1).to(torch.bfloat16)

    # --- Full corpus rerank (all docs) ---
    def _lik_full():
        _ = maxsim_residual_varlen(
            Q,
            idx["codes_flat"],
            idx["residuals_flat"],
            idx["cu_seqlens_d"],
            idx["centroids"],
            idx["bucket_weights"],
            idx["nbits"],
            normalize=True,
        )

    lik_full_ms = time_cuda(_lik_full, warmup=args.warmup, iters=args.iters)
    print(f"  lik varlen (all docs)  : {lik_full_ms:.2f} ms/query")

    # --- Partial rerank (n_full_scores random docs — fast-plaid shape) ---
    n_cand = min(args.n_full_scores, idx["n_docs"])
    cand_ids = torch.randperm(idx["n_docs"], device="cuda")[:n_cand]
    doc_lengths_cand = idx["doc_lengths"][cand_ids]
    # Gather the per-doc ragged rows. Simple approach: rebuild a flat layout
    # with just the candidate docs. One-time cost per query, realistic to
    # what a Python rerank wrapper around fast-plaid's IVF would do.
    start_per_doc = idx["cu_seqlens_d"][:-1].index_select(0, cand_ids.to(torch.int32))
    # Build a ragged index set via concat in fp. For ~4k docs this is cheap.
    rows_per_doc = [(int(start_per_doc[i].item()), int(doc_lengths_cand[i].item())) for i in range(n_cand)]
    total_cand = int(doc_lengths_cand.sum().item())
    codes_cand = torch.empty(total_cand, dtype=torch.int64, device="cuda")
    resid_cand = torch.empty(total_cand, idx["residuals_flat"].shape[-1], dtype=torch.uint8, device="cuda")
    off = 0
    for s, l in rows_per_doc:
        codes_cand[off : off + l] = idx["codes_flat"][s : s + l]
        resid_cand[off : off + l] = idx["residuals_flat"][s : s + l]
        off += l
    cu_cand = torch.zeros(n_cand + 1, dtype=torch.int32, device="cuda")
    cu_cand[1:] = doc_lengths_cand.to(torch.int32).cumsum(0)

    def _lik_partial():
        _ = maxsim_residual_varlen(
            Q,
            codes_cand,
            resid_cand,
            cu_cand,
            idx["centroids"],
            idx["bucket_weights"],
            idx["nbits"],
            normalize=True,
        )

    lik_partial_ms = time_cuda(_lik_partial, warmup=args.warmup, iters=args.iters)
    print(f"  lik varlen ({n_cand} cands)  : {lik_partial_ms:.2f} ms/query")

    if e2e_ms_per_q is not None:
        speedup_full = e2e_ms_per_q / lik_full_ms
        speedup_partial = e2e_ms_per_q / lik_partial_ms
        print(f"  speedup vs engine.search(): full={speedup_full:.2f}× partial={speedup_partial:.2f}×")

    row = {
        "name": name,
        "n_docs": n_docs,
        "ld_max": ld_max,
        "nbits": nbits,
        "n_full_scores": n_cand,
        "build_s": build_s,
        "engine_search_ms_per_query": e2e_ms_per_q,
        "lik_full_rerank_ms": lik_full_ms,
        "lik_partial_rerank_ms": lik_partial_ms,
        "n_tokens_total": idx["total_tokens"],
    }

    gc.collect()
    torch.cuda.empty_cache()
    if os.path.exists(idx_dir):
        shutil.rmtree(idx_dir, ignore_errors=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="benchmarks/results")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--n-queries", type=int, default=32)
    ap.add_argument("--n-full-scores", type=int, default=4096)
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--skip-fastplaid", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    gpu = torch.cuda.get_device_name().replace(" ", "_") if torch.cuda.is_available() else "cpu"
    print(f"GPU: {gpu}")

    corpora = [
        ("5k-200-nb2", 5_000, 200, 2),
        ("10k-300-nb2", 10_000, 300, 2),
        ("10k-512-nb2", 10_000, 512, 2),
    ]
    if not args.small:
        corpora.append(("10k-512-nb4", 10_000, 512, 4))
        corpora.append(("25k-300-nb2", 25_000, 300, 2))

    rows = []
    for name, n_docs, ld, nbits in corpora:
        try:
            rows.append(run_corpus(name, n_docs, ld, nbits, args))
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED {name}: {e}")
            rows.append({"name": name, "error": str(e)})

    fn = os.path.join(args.outdir, f"fastplaid_e2e_{gpu}.json")
    with open(fn, "w") as f:
        json.dump({"time_s": time.time(), "gpu": gpu, "rows": rows}, f, indent=2)
    print(f"\nwrote {fn}")


if __name__ == "__main__":
    main()
