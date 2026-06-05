"""Benchmark: fused `normalize=True` vs explicit `F.normalize` + MaxSim.

Shows the HBM-bandwidth saving for the fused path. At large (Nd, Ld, d) the
explicit-normalize path writes + re-reads the entire D tensor; the fused path
does it in SRAM so only the raw D is read from HBM once.
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F

from late_interaction_kernels import maxsim

SHAPES = [
    # name, Nq, Nd, Lq, Ld, d
    ("text-short", 1, 1000, 32, 300, 128),
    ("text-long", 1, 1000, 32, 1024, 128),
    ("bigbatch-300", 32, 32, 32, 300, 128),
    ("bigbatch-2048", 8, 16, 32, 2048, 128),
    ("bigbatch-8192", 8, 16, 32, 8192, 128),
    ("corpus-10k", 1, 10000, 32, 300, 128),
]


def _time(fn, warmup=5, iters=50):
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


def _peak_mb(fn) -> float:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2


def _filter_shapes(only: list[str] | None) -> list[tuple]:
    if not only:
        return list(SHAPES)
    wanted = set(only)
    out = [s for s in SHAPES if s[0] in wanted]
    if not out:
        raise SystemExit(f"unknown shape(s); pick from: {[s[0] for s in SHAPES]}")
    return out


def main(out_dir: str, only: list[str] | None):
    gpu = torch.cuda.get_device_name(0).replace(" ", "_")
    rows = []
    for name, Nq, Nd, Lq, Ld, d in _filter_shapes(only):
        Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.bfloat16)
        D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.bfloat16)

        def _explicit():
            Qn = F.normalize(Q.float(), p=2, dim=-1).to(torch.bfloat16)
            Dn = F.normalize(D.float(), p=2, dim=-1).to(torch.bfloat16)
            return maxsim(Qn, Dn)

        def _fused():
            return maxsim(Q, D, normalize=True)

        t_explicit = _time(_explicit)
        t_fused = _time(_fused)
        m_explicit = _peak_mb(_explicit)
        m_fused = _peak_mb(_fused)

        rows.append(
            {
                "name": name,
                "Nq": Nq,
                "Nd": Nd,
                "Lq": Lq,
                "Ld": Ld,
                "d": d,
                "explicit_ms": t_explicit,
                "fused_ms": t_fused,
                "speedup": t_explicit / t_fused,
                "explicit_peak_mb": m_explicit,
                "fused_peak_mb": m_fused,
            }
        )
        print(
            f"{name:20s}  explicit={t_explicit:6.3f} ms / {m_explicit:7.1f} MB  "
            f"fused={t_fused:6.3f} ms / {m_fused:7.1f} MB  "
            f"speedup={t_explicit / t_fused:4.2f}x"
        )

    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/normalize_{gpu}.json", "w") as f:
        json.dump({"gpu": gpu, "rows": rows}, f, indent=2)
    with open(f"{out_dir}/normalize_{gpu}.md", "w") as f:
        f.write(f"# Fused L2-normalize bench — {gpu}\n\n")
        f.write(
            "| shape | Nq | Nd | Lq | Ld | d | F.normalize+maxsim (ms) | "
            "fused (ms) | speedup | explicit peak (MB) | fused peak (MB) |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['name']} | {r['Nq']} | {r['Nd']} | {r['Lq']} | {r['Ld']} | {r['d']} | "
                f"{r['explicit_ms']:.3f} | {r['fused_ms']:.3f} | {r['speedup']:.2f}× | "
                f"{r['explicit_peak_mb']:.1f} | {r['fused_peak_mb']:.1f} |\n"
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="benchmarks/results")
    p.add_argument(
        "--only",
        nargs="+",
        default=None,
        help=f"subset of shape names to run; default = all. choices: {[s[0] for s in SHAPES]}",
    )
    args = p.parse_args()
    main(args.outdir, args.only)
