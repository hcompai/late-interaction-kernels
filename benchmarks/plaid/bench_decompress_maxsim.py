"""Head-to-head bench: fast-plaid's decompress+rerank pipeline vs our fused kernels.

Fast-plaid's per-query exact rerank path (rust/search/search.rs lines 625-656)
executes the following op sequence on already-pruned candidate IDs:

    1. decompress_residuals  (4 × index_select + add + L2-normalize) ->
       [Ntop, d] normalized embeddings.
    2. direct_pad_sequences   (fresh [Ntop, Ld_max, d] scratch + index_put_).
    3. padded @ Q.T            -> [Ntop, Ld_max, Lq].
    4. colbert_score_reduce    (masked_fill(-9999) + max_dim(1) + sum_dim(-1))
       -> [Ntop] final scores.

`maxsim_residual` and `maxsim_residual_varlen` fuse all four steps into one
Triton kernel. This bench measures them against the PyTorch transliteration
of 1-4 (which runs the same CUDA ops libtorch dispatches from the Rust side).

Only an H100/A100-class GPU is meaningful here; the kernel tunes for Hopper.

Usage
-----

    python benchmarks/plaid/bench_decompress_maxsim.py
    python benchmarks/plaid/bench_decompress_maxsim.py --nbits 4
    python benchmarks/plaid/bench_decompress_maxsim.py --short  # quick smoke
"""

import argparse
import gc
import json
import os
import time

import torch

D_MODEL = 128
LQ = 32
N_CENTROIDS = 32_768

# Fast-plaid's real defaults:
#   n_full_scores = 4096 -> pruned to n_full_scores // 4 = 1024 docs for exact rerank.
# Realistic BEIR / MSMARCO doc lengths.
SHAPES = [
    # (name,         n_docs, Ld)
    ("short-200", 1024, 200),
    ("medium-512", 1024, 512),
    ("long-1024", 1024, 1024),
    ("xlong-4096", 1024, 4096),
    ("xxlong-8192", 512, 8192),
    ("tiny-256-docs", 256, 512),
    ("large-2048-docs", 2048, 512),
]


def _build_index(n_docs: int, ld: int, nbits: int, device: str = "cuda", seed: int = 0):
    """Synthesize fast-plaid-shaped quantized inputs: codes, packed residuals,
    codec LUTs, centroids, bucket_weights, doc_lengths.

    Doc lengths are drawn in [ld/2, ld] so every doc is at most `ld` tokens
    (matches fast-plaid's `batch_doc_lengths` distribution).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_buckets = 2**nbits
    codes_per_byte = 8 // nbits
    packed_dim = (D_MODEL * nbits + 7) // 8

    centroids = torch.randn(N_CENTROIDS, D_MODEL, generator=g, dtype=torch.float32) * 0.3
    centroids = torch.nn.functional.normalize(centroids, dim=-1).to(device)
    bucket_weights = torch.linspace(-0.1, 0.1, n_buckets, dtype=torch.float32).to(device)

    doc_lengths = torch.randint(max(1, ld // 2), ld + 1, (n_docs,), generator=g)
    total_tokens = int(doc_lengths.sum().item())

    codes_flat = torch.randint(0, N_CENTROIDS, (total_tokens,), generator=g, dtype=torch.int64)
    # Random per-feature bucket indices, then pack.
    bucket_codes = torch.randint(0, n_buckets, (total_tokens, D_MODEL), generator=g, dtype=torch.int64)
    residuals_flat = torch.zeros(total_tokens, packed_dim, dtype=torch.uint8)
    for f in range(D_MODEL):
        b = f // codes_per_byte
        slot = f % codes_per_byte
        residuals_flat[:, b] |= bucket_codes[:, f].to(torch.uint8) << (slot * nbits)

    # Move to GPU.
    codes_flat = codes_flat.to(device)
    residuals_flat = residuals_flat.to(device)
    doc_lengths = doc_lengths.to(device)

    # Build cumulative offsets for the varlen kernel.
    cu_seqlens = torch.zeros(n_docs + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = doc_lengths.to(torch.int32).cumsum(0)

    # Build *padded* codes + residuals [n_docs, ld_max, *] for the dense kernel
    # and for the fast-plaid-style op sequence.
    ld_max = int(doc_lengths.max().item())
    codes_padded = torch.zeros(n_docs, ld_max, dtype=torch.int64, device=device)
    residuals_padded = torch.zeros(n_docs, ld_max, packed_dim, dtype=torch.uint8, device=device)
    offsets = torch.cat([torch.zeros(1, dtype=torch.int64, device=device), doc_lengths.cumsum(0)[:-1]])
    for i in range(n_docs):
        lo = int(offsets[i].item())
        hi = lo + int(doc_lengths[i].item())
        codes_padded[i, : hi - lo] = codes_flat[lo:hi]
        residuals_padded[i, : hi - lo] = residuals_flat[lo:hi]

    # Pre-compute the per-doc-id and per-token-id scatter indices once — they're
    # fixed for a given corpus and reused every query.
    doc_ids = torch.arange(n_docs, device=device).repeat_interleave(doc_lengths)
    tok_ids = torch.cat([torch.arange(int(l.item()), device=device) for l in doc_lengths.cpu()])

    return {
        "n_docs": n_docs,
        "ld_max": ld_max,
        "total_tokens": total_tokens,
        "nbits": nbits,
        "packed_dim": packed_dim,
        "codes_per_byte": codes_per_byte,
        "centroids": centroids,
        "bucket_weights": bucket_weights,
        "codes_flat": codes_flat,
        "residuals_flat": residuals_flat,
        "cu_seqlens_d": cu_seqlens,
        "codes_padded": codes_padded,
        "residuals_padded": residuals_padded,
        "doc_lengths": doc_lengths,
        "doc_ids_scatter": doc_ids,
        "tok_ids_scatter": tok_ids,
    }


def fastplaid_exact_pipeline(idx: dict, Q: torch.Tensor) -> torch.Tensor:
    """PyTorch transliteration of fast-plaid's per-query scoring slice.

    Mirrors the op sequence from ``rust/search/search.rs``:

      1. ``decompress_residuals`` (lines 53-107): ragged
         ``index_select(centroids, codes)`` + bit-unpack of the packed
         residual bytes + ``index_select(bucket_weights, ...)`` + add +
         L2-normalize -> ``[total_d_tokens, d]`` fp32 unit vectors.
      2. ``direct_pad_sequences`` (rust/search/padding.rs lines 61-108):
         allocate a fresh ``[n_docs, ld_max, d]`` scratch and scatter the
         ragged rows into it.
      3. ``matmul`` (line 654): ``padded @ Q.T`` -> ``[n_docs, ld_max, Lq]``.
      4. ``colbert_score_reduce`` (lines 385-402): ``masked_fill(~mask, -9999)``
         + ``max_dim(1)`` + ``sum_dim(-1)`` -> ``[n_docs]``.

    We use direct bit-shift unpacking instead of fast-plaid's 256-entry
    ``byte_reversed_bits_map`` + ``bucket_weight_indices_lookup`` combo so
    the reference matches our kernel's math bit-for-bit (modulo the bf16
    matmul in the fused kernel). The LUT route and the shift route do the
    same reads and the same adds; total decompression cost is memory-bandwidth
    bound and essentially identical either way — we've confirmed this with a
    side benchmark.
    """
    centroids = idx["centroids"]
    codes = idx["codes_flat"]
    residuals = idx["residuals_flat"]
    bucket_weights = idx["bucket_weights"]
    doc_lengths = idx["doc_lengths"]
    doc_ids = idx["doc_ids_scatter"]
    tok_ids = idx["tok_ids_scatter"]
    n_docs = idx["n_docs"]
    ld_max = idx["ld_max"]
    nbits = idx["nbits"]
    codes_per_byte = idx["codes_per_byte"]
    d = D_MODEL

    num_emb = codes.shape[0]

    # --- decompress_residuals (ragged, direct bit-shift) ---
    cent = centroids.index_select(0, codes)  # [N, d] fp32

    # Unpack `[N, packed_dim]` bytes into `[N, d]` bucket codes via a single
    # broadcasted shift+mask — the same math our kernel does in registers.
    rs = residuals.to(torch.int32)  # [N, packed_dim]
    feat_idx = torch.arange(d, device=codes.device)
    byte_idx = (feat_idx // codes_per_byte).to(torch.long)
    slot_idx = feat_idx % codes_per_byte
    shift = slot_idx * nbits
    mask_val = (1 << nbits) - 1
    # bytes_per_feat: [N, d] where bytes_per_feat[n, f] = rs[n, byte_idx[f]].
    bytes_per_feat = rs.index_select(1, byte_idx)  # [N, d]
    bucket_codes = (bytes_per_feat >> shift) & mask_val  # [N, d]
    bucket_vals = bucket_weights.index_select(0, bucket_codes.reshape(-1)).view(num_emb, d)

    decomp = cent + bucket_vals  # [N, d] fp32
    norms = decomp.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    normalized = (decomp / norms).to(Q.dtype)

    # --- direct_pad_sequences ---
    # Fresh scratch per query, matching fast-plaid's `get_scratch` which also
    # allocates a new tensor instead of caching.
    padded = torch.zeros(n_docs, ld_max, d, dtype=Q.dtype, device=Q.device)
    padded[doc_ids, tok_ids] = normalized

    pos = torch.arange(ld_max, device=Q.device).unsqueeze(0)
    mask = pos < doc_lengths.unsqueeze(-1)

    # --- matmul + colbert_score_reduce ---
    token_scores = padded.matmul(Q.transpose(-2, -1))  # [n_docs, ld_max, Lq]
    padding = ~mask.unsqueeze(-1).expand_as(token_scores)
    token_scores = token_scores.masked_fill(padding, -9999.0)
    max_per_dt = token_scores.max(dim=1).values  # [n_docs, Lq]
    return max_per_dt.sum(dim=-1)  # [n_docs]


def lik_dense(idx: dict, Q: torch.Tensor) -> torch.Tensor:
    """Our existing fused kernel over the padded [Nd, ld_max, packed_dim] format."""
    from late_interaction_kernels.plaid import maxsim_residual

    return maxsim_residual(
        Q.unsqueeze(0),  # [1, Lq, d]
        idx["codes_padded"],
        idx["residuals_padded"],
        idx["doc_lengths"],
        idx["centroids"],
        idx["bucket_weights"],
        nbits=idx["nbits"],
        normalize=True,
    ).squeeze(0)


def lik_varlen(idx: dict, Q: torch.Tensor, max_ld: int) -> torch.Tensor:
    """Our new fused kernel over the ragged (cu_seqlens) format — the native
    storage fast-plaid already uses for `doc_codes_strided`."""
    from late_interaction_kernels.plaid import maxsim_residual_varlen

    return maxsim_residual_varlen(
        Q.unsqueeze(0),
        idx["codes_flat"],
        idx["residuals_flat"],
        idx["cu_seqlens_d"],
        idx["centroids"],
        idx["bucket_weights"],
        nbits=idx["nbits"],
        max_seqlen_d=max_ld,
        normalize=True,
    ).squeeze(0)


def _timed(fn, warmup: int = 5, iters: int = 20) -> float:
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
    return torch.cuda.max_memory_allocated() / 1024**2  # MB


def run(args) -> list[dict]:
    gpu = torch.cuda.get_device_name()
    print(f"GPU: {gpu}")
    print(f"nbits: {args.nbits}, dtype: {args.dtype}, d: {D_MODEL}, Lq: {LQ}\n")

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    header = (
        f"{'shape':<22} {'fp_ms':>8} {'lik_dense_ms':>13} {'lik_varlen_ms':>15} "
        f"{'vs_dense':>9} {'vs_varlen':>10} {'fp_MB':>8} {'varlen_MB':>10}"
    )
    print(header)
    print("-" * len(header))

    if args.only:
        wanted = set(args.only)
        shapes = [s for s in SHAPES if s[0] in wanted]
        if not shapes:
            raise SystemExit(f"unknown shape(s); pick from: {[s[0] for s in SHAPES]}")
    elif args.short:
        shapes = SHAPES[:3]
    else:
        shapes = SHAPES
    rows = []
    for name, n_docs, ld in shapes:
        idx = _build_index(n_docs, ld, args.nbits, seed=args.seed)
        Q = torch.nn.functional.normalize(torch.randn(LQ, D_MODEL, device="cuda", dtype=dtype), dim=-1)
        row = {"shape": name, "n_docs": n_docs, "ld": ld, "nbits": args.nbits}

        # Correctness first.
        try:
            ref = fastplaid_exact_pipeline(idx, Q)
            a = lik_dense(idx, Q).float()
            b = lik_varlen(idx, Q, ld).float()
            ref_f = ref.float()
            denom = ref_f.abs().max().item() + 1e-6
            rel_dense = (a - ref_f).abs().max().item() / denom
            rel_varlen = (b - ref_f).abs().max().item() / denom
            row["rel_err_dense"] = rel_dense
            row["rel_err_varlen"] = rel_varlen
            if rel_dense > 5e-2 or rel_varlen > 5e-2:
                print(f"  WARN {name} rel err dense={rel_dense:.3e} varlen={rel_varlen:.3e}")
        except torch.cuda.OutOfMemoryError:
            print(f"  {name}: OOM on correctness check; skipping")
            torch.cuda.empty_cache()
            continue

        # Time.
        try:
            fp_ms = _timed(lambda: fastplaid_exact_pipeline(idx, Q), iters=args.iters)
            fp_mb = _peak(lambda: fastplaid_exact_pipeline(idx, Q))
        except torch.cuda.OutOfMemoryError:
            fp_ms = float("nan")
            fp_mb = float("nan")
            torch.cuda.empty_cache()
        try:
            dense_ms = _timed(lambda: lik_dense(idx, Q), iters=args.iters)
        except torch.cuda.OutOfMemoryError:
            dense_ms = float("nan")
            torch.cuda.empty_cache()
        try:
            varlen_ms = _timed(lambda: lik_varlen(idx, Q, ld), iters=args.iters)
            varlen_mb = _peak(lambda: lik_varlen(idx, Q, ld))
        except torch.cuda.OutOfMemoryError:
            varlen_ms = float("nan")
            varlen_mb = float("nan")
            torch.cuda.empty_cache()

        def r(a, b):
            if a != a or b != b or b == 0:
                return float("nan")
            return a / b

        row.update(
            dict(
                fp_ms=fp_ms,
                dense_ms=dense_ms,
                varlen_ms=varlen_ms,
                fp_peak_mb=fp_mb,
                varlen_peak_mb=varlen_mb,
            )
        )
        rows.append(row)
        print(
            f"{name:<22} {fp_ms:>8.3f} {dense_ms:>13.3f} {varlen_ms:>15.3f} "
            f"{r(fp_ms, dense_ms):>7.2f}x {r(fp_ms, varlen_ms):>9.2f}x "
            f"{fp_mb:>7.1f}M {varlen_mb:>9.1f}M"
        )
        torch.cuda.empty_cache()
        gc.collect()

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="benchmarks/results")
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--nbits", type=int, default=2, choices=[2, 4, 8])
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--short", action="store_true")
    ap.add_argument(
        "--only",
        nargs="+",
        default=None,
        help=f"subset of shape names to run; default = all. choices: {[s[0] for s in SHAPES]}",
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rows = run(args)
    gpu = torch.cuda.get_device_name().replace(" ", "_")
    out = {"time_s": time.time(), "gpu": gpu, "nbits": args.nbits, "dtype": args.dtype, "rows": rows}
    fn = os.path.join(args.outdir, f"decompress_maxsim_{gpu}_nb{args.nbits}_{args.dtype}.json")
    with open(fn, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {fn}")


if __name__ == "__main__":
    main()
