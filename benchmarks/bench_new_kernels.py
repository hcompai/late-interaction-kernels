"""Benchmark: fused MaxSim kernels vs PyTorch reference for new APIs.

Covers maxsim_topk, maxsim_matryoshka, maxsim_xtr, plaid_approx_score,
and maxsim_residual. Each benchmark reports both the kernel time and the
naive PyTorch reference time on the same shape.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F

from late_interaction_kernels import (
    maxsim,
    maxsim_matryoshka,
    maxsim_residual,
    maxsim_topk,
    maxsim_xtr,
    plaid_approx_score,
)
from late_interaction_kernels.reference import (
    maxsim_residual_reference,
    plaid_approx_score_reference,
    xtr_reference,
)


def _time(fn, warmup=5, iters=30):
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


def bench_topk(rows):
    name = "topk"
    # typical reranker: 1 query, 10k docs, top-10
    Q = torch.randn(1, 32, 128, device="cuda", dtype=torch.bfloat16)
    D = torch.randn(10_000, 300, 128, device="cuda", dtype=torch.bfloat16)

    def _ref():
        s = maxsim(Q, D)
        return torch.topk(s, 10, dim=-1)

    def _fast():
        return maxsim_topk(Q, D, 10)

    t_ref = _time(_ref)
    t_fast = _time(_fast)
    rows.append({"name": name, "ref_ms": t_ref, "fast_ms": t_fast, "speedup": t_ref / t_fast})
    print(f"{name:24s}  ref={t_ref:7.3f} ms  fast={t_fast:7.3f} ms  speedup={t_ref / t_fast:.2f}x")


def bench_matryoshka(rows):
    Q = torch.randn(4, 32, 128, device="cuda", dtype=torch.bfloat16)
    D = torch.randn(1000, 300, 128, device="cuda", dtype=torch.bfloat16)
    dims = [32, 64, 128]

    def _ref():
        Qn = F.normalize(Q.float(), dim=-1).to(torch.bfloat16)
        Dn = F.normalize(D.float(), dim=-1).to(torch.bfloat16)
        return [maxsim(Qn[..., :k].contiguous(), Dn[..., :k].contiguous()) for k in dims]

    def _fast():
        return maxsim_matryoshka(Q, D, dims, normalize=True)

    t_ref = _time(_ref)
    t_fast = _time(_fast)
    rows.append({"name": "matryoshka-3dims", "ref_ms": t_ref, "fast_ms": t_fast, "speedup": t_ref / t_fast})
    print(
        f"{'matryoshka-3dims':24s}  ref={t_ref:7.3f} ms  fast={t_fast:7.3f} ms  speedup={t_ref / t_fast:.2f}x"
    )


def bench_xtr(rows):
    Q = torch.randn(2, 32, 128, device="cuda", dtype=torch.bfloat16)
    D = torch.randn(500, 300, 128, device="cuda", dtype=torch.bfloat16)

    for k in (5, 20):

        def _ref(k=k):
            return xtr_reference(Q, D, top_k=k)

        def _fast(k=k):
            return maxsim_xtr(Q, D, top_k=k)

        t_ref = _time(_ref)
        t_fast = _time(_fast)
        rows.append({"name": f"xtr-k{k}", "ref_ms": t_ref, "fast_ms": t_fast, "speedup": t_ref / t_fast})
        print(
            f"{'xtr-k' + str(k):24s}  ref={t_ref:7.3f} ms  fast={t_fast:7.3f} ms  speedup={t_ref / t_fast:.2f}x"
        )


def bench_plaid_approx(rows):
    n_centroids, Lq, B, max_Ld = 32_000, 32, 8192, 200
    qcs = torch.randn(n_centroids, Lq, device="cuda", dtype=torch.float32)
    codes = torch.randint(0, n_centroids, (B, max_Ld), device="cuda", dtype=torch.int64)
    doc_lens = torch.randint(max_Ld // 2, max_Ld + 1, (B,), device="cuda", dtype=torch.int64)

    def _ref():
        return plaid_approx_score_reference(qcs, codes, doc_lens)

    def _fast():
        return plaid_approx_score(qcs, codes, doc_lens)

    t_ref = _time(_ref)
    t_fast = _time(_fast)
    rows.append(
        {"name": "plaid-approx-8k-docs", "ref_ms": t_ref, "fast_ms": t_fast, "speedup": t_ref / t_fast}
    )
    print(f"{'plaid-approx':24s}  ref={t_ref:7.3f} ms  fast={t_fast:7.3f} ms  speedup={t_ref / t_fast:.2f}x")


def _mk_quant_index(Nd, max_Ld, d, n_cent, nbits):
    torch.manual_seed(0)
    centroids = torch.randn(n_cent, d, device="cuda", dtype=torch.float32) * 0.3
    buckets = torch.linspace(-0.1, 0.1, 2**nbits, device="cuda", dtype=torch.float32)
    codes_per_byte = 8 // nbits
    packed_dim = (d * nbits + 7) // 8
    codes = torch.randint(0, n_cent, (Nd, max_Ld), device="cuda", dtype=torch.int64)
    b_codes = torch.randint(0, 2**nbits, (Nd, max_Ld, d), device="cuda", dtype=torch.int64)
    res = torch.zeros(Nd, max_Ld, packed_dim, device="cuda", dtype=torch.uint8)
    for f in range(d):
        b_idx = f // codes_per_byte
        slot = f % codes_per_byte
        res[..., b_idx] |= b_codes[..., f].to(torch.uint8) << (slot * nbits)
    dlens = torch.randint(max_Ld // 2, max_Ld + 1, (Nd,), device="cuda", dtype=torch.int64)
    return centroids, buckets, codes, res, dlens


def bench_plaid_residual(rows):
    for nbits in (2, 4):
        Nd, max_Ld, d, n_cent = 256, 300, 128, 4096
        cent, buckets, codes, res, dlens = _mk_quant_index(Nd, max_Ld, d, n_cent, nbits)
        Q = torch.randn(1, 32, d, device="cuda", dtype=torch.bfloat16)

        def _ref(nbits=nbits):
            return maxsim_residual_reference(
                Q,
                codes,
                res,
                dlens,
                cent,
                buckets,
                nbits=nbits,
                normalize=True,
            )

        def _fast(nbits=nbits):
            return maxsim_residual(
                Q,
                codes,
                res,
                dlens,
                cent,
                buckets,
                nbits=nbits,
                normalize=True,
            )

        t_ref = _time(_ref)
        t_fast = _time(_fast)
        rows.append(
            {"name": f"residual-{nbits}bit", "ref_ms": t_ref, "fast_ms": t_fast, "speedup": t_ref / t_fast}
        )
        print(
            f"{'residual-' + str(nbits) + 'bit':24s}  ref={t_ref:7.3f} ms  fast={t_fast:7.3f} ms  speedup={t_ref / t_fast:.2f}x"
        )


def main(out_dir: str):
    gpu = torch.cuda.get_device_name(0).replace(" ", "_")
    rows = []
    bench_topk(rows)
    bench_matryoshka(rows)
    bench_xtr(rows)
    bench_plaid_approx(rows)
    bench_plaid_residual(rows)

    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/new_kernels_{gpu}.json", "w") as f:
        json.dump({"gpu": gpu, "rows": rows}, f, indent=2)
    with open(f"{out_dir}/new_kernels_{gpu}.md", "w") as f:
        f.write(f"# New kernels bench — {gpu}\n\n")
        f.write("| kernel | PyTorch ref (ms) | late-interaction-kernels (ms) | speedup |\n")
        f.write("|---|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['name']} | {r['ref_ms']:.3f} | {r['fast_ms']:.3f} | {r['speedup']:.2f}× |\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="benchmarks/results")
    args = p.parse_args()
    main(args.outdir)
