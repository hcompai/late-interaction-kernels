"""Benchmark: fused backward paths shipped in 0.5.0.

Covers:
    * ``maxsim_residual`` forward + backward (grad_Q) vs "dense unpack + maxsim"
      PyTorch autograd.
    * ``maxsim_varlen`` forward + backward vs the padded ``maxsim`` autograd
      path on an equivalent shape.

Both benchmarks are kernel-level microbenchmarks and do *not* reflect any
full-system behavior — they just measure the step this library replaces.
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from late_interaction_kernels import maxsim, maxsim_residual, maxsim_varlen
from late_interaction_kernels.reference import unpack_residuals_reference


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


# -----------------------------------------------------------------------------
# 1. Residual forward+backward (nbits=2/4/8, training on compressed embeddings)
# -----------------------------------------------------------------------------


def bench_residual_backward(rows):
    """Sweep nbits × {PyLate-style small batch, ColBERTv2-rerank batch}.

    Two regimes:
    * ``small`` — ``Nq=8, Nd=64, Ld=300`` — a quick sanity datapoint.
    * ``rerank`` — ``Nq=4, Nd=512, Ld=300`` — a realistic 512-candidate
      rerank batch where the dense ``[Nd, Ld, d]`` fp32 scratch starts to
      matter (≈ 60 MB at d=128 ⇒ HBM-bandwidth-bound for the reference).
    """
    torch.manual_seed(0)
    d = 128
    n_centroids = 1024

    configs = [
        {"tag": "small", "Nq": 8, "Lq": 32, "Nd": 64, "max_Ld": 300},
        {"tag": "rerank", "Nq": 4, "Lq": 32, "Nd": 512, "max_Ld": 300},
    ]

    for cfg in configs:
        Nq, Lq, Nd, max_Ld = cfg["Nq"], cfg["Lq"], cfg["Nd"], cfg["max_Ld"]
        tag = cfg["tag"]
        for nbits in (2, 4, 8):
            n_buckets = 2**nbits
            codes_per_byte = 8 // nbits
            packed_dim = (d * nbits + 7) // 8

            centroids = torch.randn(n_centroids, d, device="cuda", dtype=torch.float32) * 0.3
            bucket_weights = torch.linspace(-0.1, 0.1, n_buckets, device="cuda", dtype=torch.float32)
            codes = torch.randint(0, n_centroids, (Nd, max_Ld), device="cuda", dtype=torch.int64)
            bucket_codes = torch.randint(0, n_buckets, (Nd, max_Ld, d), device="cuda", dtype=torch.int64)
            residuals = torch.zeros(Nd, max_Ld, packed_dim, device="cuda", dtype=torch.uint8)
            for f in range(d):
                byte_idx = f // codes_per_byte
                slot = f % codes_per_byte
                residuals[..., byte_idx] |= bucket_codes[..., f].to(torch.uint8) << (slot * nbits)
            doc_lengths = torch.randint(max_Ld // 2, max_Ld + 1, (Nd,), device="cuda", dtype=torch.int64)

            Q_fast = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)

            def step_fast():
                Q_fast.grad = None
                s = maxsim_residual(
                    Q_fast,
                    codes,
                    residuals,
                    doc_lengths,
                    centroids,
                    bucket_weights,
                    nbits=nbits,
                    normalize=True,
                )
                s.sum().backward()

            # Reference: unpack once (offline, not timed), then run maxsim with autograd.
            bucket_codes_dec = unpack_residuals_reference(residuals, nbits, d).clamp_min(0)
            emb = centroids[codes] + bucket_weights[bucket_codes_dec]
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1, eps=1e-12).to(torch.bfloat16)
            d_mask = torch.arange(max_Ld, device="cuda").unsqueeze(0) < doc_lengths.unsqueeze(-1)
            Q_ref = Q_fast.detach().clone().requires_grad_(True)

            def step_ref():
                Q_ref.grad = None
                Qn = torch.nn.functional.normalize(Q_ref, p=2, dim=-1, eps=1e-12)
                s = maxsim(Qn, emb, d_mask=d_mask)
                s.sum().backward()

            t_fast = _time(step_fast)
            t_ref = _time(step_ref)

            print(
                f"[residual-bwd {tag} nbits={nbits}] Nq={Nq} Nd={Nd} Lq={Lq} Ld={max_Ld} d={d} | "
                f"fused {t_fast:.2f} ms  |  unpack+maxsim {t_ref:.2f} ms  |  "
                f"speedup {t_ref / t_fast:.2f}x  |  dense emb scratch avoided = "
                f"{Nd * max_Ld * d * 4 / 1e9:.2f} GB"
            )
            rows.append(
                {
                    "kernel": "maxsim_residual_bwd",
                    "tag": tag,
                    "nbits": nbits,
                    "Nq": Nq,
                    "Nd": Nd,
                    "Lq": Lq,
                    "Ld": max_Ld,
                    "d": d,
                    "fused_ms": t_fast,
                    "unpack_maxsim_ms": t_ref,
                    "speedup": t_ref / t_fast,
                    "dense_emb_gb": Nd * max_Ld * d * 4 / 1e9,
                }
            )


# -----------------------------------------------------------------------------
# 2. Varlen forward+backward vs padded autograd on equivalent ragged batch
# -----------------------------------------------------------------------------


def _bench_varlen_once(rows, Nq, Nd, d, q_lens, d_lens, tag):
    max_lq = max(q_lens)
    max_ld = max(d_lens)
    Nq = len(q_lens)

    Qp = torch.randn(sum(q_lens), d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    Dp = torch.randn(sum(d_lens), d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    cu_q = torch.zeros(Nq + 1, device="cuda", dtype=torch.int32)
    cu_q[1:] = torch.tensor(q_lens, device="cuda", dtype=torch.int32).cumsum(0)
    cu_d = torch.zeros(Nd + 1, device="cuda", dtype=torch.int32)
    cu_d[1:] = torch.tensor(d_lens, device="cuda", dtype=torch.int32).cumsum(0)

    def step_var():
        Qp.grad = None
        Dp.grad = None
        s = maxsim_varlen(Qp, Dp, cu_q, cu_d, max_lq, max_ld)
        s.sum().backward()

    # Padded reference.
    Q_pad = torch.zeros(Nq, max_lq, d, device="cuda", dtype=torch.bfloat16)
    D_pad = torch.zeros(Nd, max_ld, d, device="cuda", dtype=torch.bfloat16)
    q_mask = torch.zeros(Nq, max_lq, device="cuda", dtype=torch.bool)
    d_mask = torch.zeros(Nd, max_ld, device="cuda", dtype=torch.bool)
    for i, lq in enumerate(q_lens):
        Q_pad[i, :lq] = Qp.detach()[cu_q[i] : cu_q[i + 1]]
        q_mask[i, :lq] = True
    for j, ld in enumerate(d_lens):
        D_pad[j, :ld] = Dp.detach()[cu_d[j] : cu_d[j + 1]]
        d_mask[j, :ld] = True
    Q_pad = Q_pad.requires_grad_(True)
    D_pad = D_pad.requires_grad_(True)

    def step_pad():
        Q_pad.grad = None
        D_pad.grad = None
        s = maxsim(Q_pad, D_pad, q_mask=q_mask, d_mask=d_mask)
        s.sum().backward()

    t_var = _time(step_var)
    t_pad = _time(step_pad)
    waste = 1.0 - sum(d_lens) / (Nd * max_ld)
    print(
        f"[varlen-bwd {tag}] Nq={Nq} Nd={Nd} d={d} max_ld={max_ld} "
        f"padding_waste={waste:.0%} | "
        f"varlen {t_var:.2f} ms  |  padded {t_pad:.2f} ms  |  "
        f"speedup {t_pad / t_var:.2f}x"
    )
    rows.append(
        {
            "kernel": "maxsim_varlen_bwd",
            "tag": tag,
            "Nq": Nq,
            "Nd": Nd,
            "d": d,
            "max_ld": max_ld,
            "padding_waste": waste,
            "varlen_ms": t_var,
            "padded_ms": t_pad,
            "speedup": t_pad / t_var,
        }
    )


def bench_varlen_backward(rows):
    """Two varlen backward regimes.

    * ``code-retrieval`` — short queries, doc lengths ~ U(32, 512).
      Matches LateOn-Code-edge style: many short docs, some long, ~50 %
      padding waste in the dense path.
    * ``long-doc`` — short queries, doc lengths ~ U(256, 4096). Matches
      a Reason-ModernColBERT-ish regime at smaller batch. The varlen
      kernel's `grad_D` path avoids a ``[Nd, max_ld, d]`` padded fp32
      buffer entirely.
    """
    torch.manual_seed(0)
    Nq, Nd, d = 8, 64, 128

    q_lens = [32] * Nq
    d_lens = torch.randint(32, 512, (Nd,), device="cpu").tolist()
    _bench_varlen_once(rows, Nq, Nd, d, q_lens, d_lens, tag="code-retrieval")

    q_lens = [32] * Nq
    d_lens = torch.randint(256, 4096, (Nd,), device="cpu").tolist()
    _bench_varlen_once(rows, Nq, Nd, d, q_lens, d_lens, tag="long-doc")


def main(out_dir: str):
    assert torch.cuda.is_available(), "CUDA required"
    gpu = torch.cuda.get_device_name(0).replace(" ", "_")
    rows: list[dict] = []

    print("=" * 78)
    print("Residual backward (fused grad_Q through 2/4/8-bit PLAID compression)")
    print("=" * 78)
    bench_residual_backward(rows)

    print()
    print("=" * 78)
    print("Varlen backward (packed grad_Q / grad_D — no repad)")
    print("=" * 78)
    bench_varlen_backward(rows)

    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/backward_0_5_{gpu}.json", "w") as f:
        json.dump({"gpu": gpu, "rows": rows}, f, indent=2)
    with open(f"{out_dir}/backward_0_5_{gpu}.md", "w") as f:
        f.write(f"# 0.5.0 backward kernels — {gpu}\n\n")
        f.write("## Residual backward (train on 2/4/8-bit compressed docs)\n\n")
        f.write(
            "| regime | nbits | Nq×Nd×Ld | fused (ms) | unpack+maxsim (ms) | speedup | dense scratch (GB) |\n"
        )
        f.write("|---|---:|---|---:|---:|---:|---:|\n")
        for r in rows:
            if r["kernel"] == "maxsim_residual_bwd":
                f.write(
                    f"| {r.get('tag', '-')} | {r['nbits']} | "
                    f"{r['Nq']}×{r['Nd']}×{r['Ld']} | "
                    f"{r['fused_ms']:.3f} | {r['unpack_maxsim_ms']:.3f} | "
                    f"{r['speedup']:.2f}× | {r.get('dense_emb_gb', 0):.2f} |\n"
                )
        f.write("\n## Varlen backward (packed grad_Q / grad_D, no repad)\n\n")
        f.write("| regime | Nq×Nd | max_ld | padding waste | varlen (ms) | padded (ms) | speedup |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|\n")
        for r in rows:
            if r["kernel"] == "maxsim_varlen_bwd":
                f.write(
                    f"| {r.get('tag', '-')} | {r['Nq']}×{r['Nd']} | "
                    f"{r['max_ld']} | {r['padding_waste']:.0%} | "
                    f"{r['varlen_ms']:.3f} | {r['padded_ms']:.3f} | "
                    f"{r['speedup']:.2f}× |\n"
                )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="benchmarks/results")
    args = ap.parse_args()
    main(args.outdir)
