"""Training-aware fused head benchmark.

Compares ``maxsim_from_hidden`` (autograd path) against the canonical
unfused ``F.linear + F.normalize + maxsim`` on realistic LateOn /
LateOn-Code / LateOn-Code-edge shapes.

Wall-clock + peak HBM (from the fused head forward + loss + backward
on a fresh allocator each iteration) are printed side by side and
written to ``<outdir>/fused_head_train_<gpu>.{json,md}``.

Usage
-----
::

    python benchmarks/bench_fused_head_train.py
    python benchmarks/bench_fused_head_train.py --only lateon-B4-Ld300
"""

import argparse
import json
import os
import statistics
import time

import torch

from late_interaction_kernels import maxsim
from late_interaction_kernels.fused_head import maxsim_from_hidden


def _unfused(Q, H_d, W, b, *, normalize):
    D = torch.nn.functional.linear(H_d, W, b)
    if normalize:
        D = torch.nn.functional.normalize(D, p=2, dim=-1, eps=1e-12)
    return maxsim(Q, D)


def _time_and_mem(fn, warmup=3, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000)
    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    return statistics.median(samples), peak_mb


SHAPES = [
    # (label, Nq, Nd, Lq, Ld, d_model, d_out)
    # LateOn / LateOn-Code: ModernBERT-base, d_model=768, d_out=128.
    ("lateon-B4-Ld300", 4, 16, 32, 300, 768, 128),
    ("lateon-B8-Ld1k", 8, 32, 32, 1024, 768, 128),
    ("lateon-code-Nd128-Ld1k", 4, 128, 32, 1024, 768, 128),
    ("lateon-code-Nd256-Ld1k", 2, 256, 32, 1024, 768, 128),
    # Regimes where the D_proj scratch is the dominant cost: the fused
    # head only ever touches winners, so peak memory stays bounded.
    ("lateon-code-Nd512-Ld2k", 1, 512, 32, 2048, 768, 128),
    ("lateon-code-Nd1024-Ld2k", 1, 1024, 32, 2048, 768, 128),
    # LateOn-Code-edge: Ettin-17M backbone, d_model=384, d_out=96.
    ("lateon-code-edge-Nd256-Ld4k", 2, 256, 32, 4096, 384, 96),
]


def _step_fused(Q, H_d, W, b, *, normalize):
    for p in (Q, H_d, W, b):
        if p is not None and p.grad is not None:
            p.grad = None
    scores = maxsim_from_hidden(Q, H_d, W, b=b, normalize=normalize)
    scores.sum().backward()


def _step_unfused(Q, H_d, W, b, *, normalize):
    for p in (Q, H_d, W, b):
        if p is not None and p.grad is not None:
            p.grad = None
    scores = _unfused(Q, H_d, W, b, normalize=normalize)
    scores.sum().backward()


def _filter_shapes(only: list[str] | None) -> list[tuple]:
    if not only:
        return list(SHAPES)
    wanted = set(only)
    out = [s for s in SHAPES if s[0] in wanted]
    if not out:
        raise SystemExit(f"unknown shape(s); pick from: {[s[0] for s in SHAPES]}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only",
        nargs="+",
        default=None,
        help=f"subset of shape names to run; default = all. choices: {[s[0] for s in SHAPES]}",
    )
    ap.add_argument("--outdir", default="benchmarks/results")
    args = ap.parse_args()

    gpu = torch.cuda.get_device_name(0).replace(" ", "_")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(
        f"{'shape':<34} {'unfused ms':>12} {'fused ms':>12} "
        f"{'speedup':>9}   {'unfused MB':>12} {'fused MB':>12}   label"
    )
    print("-" * 116)

    rows: list[dict] = []
    for label, Nq, Nd, Lq, Ld, d_model, d_out in _filter_shapes(args.only):
        torch.manual_seed(0)
        dtype = torch.bfloat16
        H_d = torch.randn(Nd, Ld, d_model, device="cuda", dtype=dtype, requires_grad=True)
        W = torch.randn(d_out, d_model, device="cuda", dtype=dtype) * (1.0 / (d_model**0.5))
        W.requires_grad_(True)
        b = (torch.randn(d_out, device="cuda", dtype=dtype) * 0.01).requires_grad_(True)
        Q = torch.nn.functional.normalize(
            torch.randn(Nq, Lq, d_out, device="cuda", dtype=dtype), dim=-1
        ).requires_grad_(True)

        t_unf, m_unf = _time_and_mem(lambda: _step_unfused(Q, H_d, W, b, normalize=True))
        t_fus, m_fus = _time_and_mem(lambda: _step_fused(Q, H_d, W, b, normalize=True))

        shape_str = f"Nd={Nd} Lq={Lq} Ld={Ld} dm={d_model}/do={d_out}"
        print(
            f"{shape_str:<34} {t_unf:>12.3f} {t_fus:>12.3f} "
            f"{t_unf / t_fus:>8.2f}x  {m_unf:>12.1f} {m_fus:>12.1f}   {label}"
        )
        rows.append(
            {
                "label": label,
                "Nq": Nq,
                "Nd": Nd,
                "Lq": Lq,
                "Ld": Ld,
                "d_model": d_model,
                "d_out": d_out,
                "unfused_ms": t_unf,
                "fused_ms": t_fus,
                "speedup": t_unf / t_fus,
                "unfused_peak_mb": m_unf,
                "fused_peak_mb": m_fus,
            }
        )

    os.makedirs(args.outdir, exist_ok=True)
    out_json = os.path.join(args.outdir, f"fused_head_train_{gpu}.json")
    out_md = os.path.join(args.outdir, f"fused_head_train_{gpu}.md")
    with open(out_json, "w") as f:
        json.dump({"gpu": gpu, "rows": rows}, f, indent=2)
    with open(out_md, "w") as f:
        f.write(f"# Fused D-side head (training) — {gpu}\n\n")
        f.write(
            "| label | Nq | Nd | Lq | Ld | d_model | d_out | unfused (ms) | "
            "fused (ms) | speedup | unfused peak (MB) | fused peak (MB) |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['label']} | {r['Nq']} | {r['Nd']} | {r['Lq']} | {r['Ld']} | "
                f"{r['d_model']} | {r['d_out']} | "
                f"{r['unfused_ms']:.3f} | {r['fused_ms']:.3f} | {r['speedup']:.2f}× | "
                f"{r['unfused_peak_mb']:.1f} | {r['fused_peak_mb']:.1f} |\n"
            )
    print(f"\n→ wrote {out_json}")
    print(f"→ wrote {out_md}")


if __name__ == "__main__":
    main()
