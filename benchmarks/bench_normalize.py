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


def main(out_dir: str):
    gpu = torch.cuda.get_device_name(0).replace(" ", "_")
    rows = []
    for name, Nq, Nd, Lq, Ld, d in SHAPES:
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
            }
        )
        print(
            f"{name:20s}  explicit={t_explicit:6.3f} ms  fused={t_fused:6.3f} ms  "
            f"speedup={t_explicit / t_fused:4.2f}x"
        )

    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/normalize_{gpu}.json", "w") as f:
        json.dump({"gpu": gpu, "rows": rows}, f, indent=2)
    with open(f"{out_dir}/normalize_{gpu}.md", "w") as f:
        f.write(f"# Fused L2-normalize bench — {gpu}\n\n")
        f.write("| shape | Nq | Nd | Lq | Ld | d | F.normalize+maxsim (ms) | fused (ms) | speedup |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['name']} | {r['Nq']} | {r['Nd']} | {r['Lq']} | {r['Ld']} | {r['d']} | "
                f"{r['explicit_ms']:.3f} | {r['fused_ms']:.3f} | {r['speedup']:.2f}× |\n"
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="benchmarks/results")
    args = p.parse_args()
    main(args.outdir)
