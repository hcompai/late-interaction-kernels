"""FP8 MaxSim vs bf16 MaxSim — reranking throughput on Hopper.

We measure end-to-end ``maxsim`` (no-grad) wall-clock for a realistic
reranking workload:

* Q: ``[Nq, Lq, d]`` (batch of query embeddings already in HBM)
* D: ``[Nd, Ld, d]`` (candidate docs — the common large axis)

FP8 halves the HBM footprint and doubles the theoretical tensor-core
throughput on Hopper (WGMMA). Expect the biggest wins on HBM-bound
shapes (large ``Nd × Ld``, small ``d``).

Usage
-----
::

    python benchmarks/bench_fp8.py
    python benchmarks/bench_fp8.py --only pylate-rerank-1k long-docs-2k
"""

import argparse
import statistics
import time

import torch

from late_interaction_kernels import maxsim
from late_interaction_kernels.fp8 import maxsim_inference_fp8, quantize_fp8_per_token

SHAPES = [
    # (label, Nq, Nd, Lq, Ld, d)
    ("pylate-rerank-1k", 1, 1024, 32, 128, 128),
    ("pylate-rerank-4k", 1, 4096, 32, 128, 128),
    ("colbert-rerank-8k", 1, 8192, 32, 256, 128),
    ("batched-rerank-16k", 4, 4096, 32, 256, 128),
    ("long-docs-2k", 1, 2048, 32, 512, 128),
    ("lateon-edge-1k", 1, 1024, 32, 128, 96),
    ("lateon-edge-4k", 1, 4096, 32, 128, 96),
]


def _time_ms(fn, warmup=5, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000)
    return statistics.median(samples), statistics.stdev(samples) if len(samples) > 1 else 0.0


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only",
        nargs="+",
        default=None,
        help=f"subset of shape names to run; default = all. choices: {[s[0] for s in SHAPES]}",
    )
    args = ap.parse_args()

    print(f"device: {torch.cuda.get_device_name(0)}")
    print(
        f"{'shape':<20} {'bf16 ms':>10} {'fp8 ms':>10} {'speedup':>10} {'bf16 MB':>10} {'fp8 MB':>10}  note"
    )
    print("-" * 100)
    for label, Nq, Nd, Lq, Ld, d in _filter_shapes(args.only):
        torch.manual_seed(0)
        Q = torch.nn.functional.normalize(torch.randn(Nq, Lq, d, device="cuda", dtype=torch.bfloat16), dim=-1)
        D = torch.nn.functional.normalize(torch.randn(Nd, Ld, d, device="cuda", dtype=torch.bfloat16), dim=-1)
        Q_fp8, sQ = quantize_fp8_per_token(Q)
        D_fp8, sD = quantize_fp8_per_token(D)

        t_bf16, _ = _time_ms(lambda: maxsim(Q, D))
        t_fp8, _ = _time_ms(lambda: maxsim_inference_fp8(Q_fp8, D_fp8, scale_Q=sQ, scale_D=sD))
        m_bf16 = _peak_mb(lambda: maxsim(Q, D))
        m_fp8 = _peak_mb(lambda: maxsim_inference_fp8(Q_fp8, D_fp8, scale_Q=sQ, scale_D=sD))
        shape_str = f"Nd={Nd} Lq={Lq} Ld={Ld} d={d}"
        print(
            f"{shape_str:<20} {t_bf16:>10.3f} {t_fp8:>10.3f} {t_bf16 / t_fp8:>9.2f}x "
            f"{m_bf16:>10.1f} {m_fp8:>10.1f}  {label}"
        )


if __name__ == "__main__":
    main()
