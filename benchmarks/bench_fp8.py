"""FP8 MaxSim vs bf16 MaxSim — reranking throughput on Hopper.

We measure end-to-end ``maxsim_inference`` wall-clock for a realistic
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
"""

from __future__ import annotations

import statistics
import time

import torch

from late_interaction_kernels import maxsim_inference, maxsim_inference_fp8
from late_interaction_kernels.fp8 import quantize_fp8_per_token

SHAPES = [
    # (Nq, Nd, Lq, Ld, d, label)
    (1, 1024, 32, 128, 128, "pylate-rerank-1k"),
    (1, 4096, 32, 128, 128, "pylate-rerank-4k"),
    (1, 8192, 32, 256, 128, "colbert-rerank-8k"),
    (4, 4096, 32, 256, 128, "batched-rerank-16k"),
    (1, 2048, 32, 512, 128, "long-docs-2k"),
    (1, 1024, 32, 128, 96, "lateon-edge-1k"),
    (1, 4096, 32, 128, 96, "lateon-edge-4k"),
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


def main():
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"{'shape':<20} {'bf16 ms':>10} {'fp8 ms':>10} {'speedup':>10}  note")
    print("-" * 80)
    for Nq, Nd, Lq, Ld, d, label in SHAPES:
        torch.manual_seed(0)
        Q = torch.nn.functional.normalize(torch.randn(Nq, Lq, d, device="cuda", dtype=torch.bfloat16), dim=-1)
        D = torch.nn.functional.normalize(torch.randn(Nd, Ld, d, device="cuda", dtype=torch.bfloat16), dim=-1)
        Q_fp8, sQ = quantize_fp8_per_token(Q)
        D_fp8, sD = quantize_fp8_per_token(D)

        t_bf16, _ = _time_ms(lambda: maxsim_inference(Q, D))
        t_fp8, _ = _time_ms(lambda: maxsim_inference_fp8(Q_fp8, D_fp8, scale_Q=sQ, scale_D=sD))
        shape_str = f"Nd={Nd} Lq={Lq} Ld={Ld} d={d}"
        print(f"{shape_str:<20} {t_bf16:>10.3f} {t_fp8:>10.3f} {t_bf16 / t_fp8:>9.2f}x  {label}")


if __name__ == "__main__":
    main()
