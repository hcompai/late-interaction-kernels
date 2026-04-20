# Reproducing the benchmarks

All numbers in the README were measured on a **single H100 80 GB SXM** in
bf16 compute with fp32 accumulator, 50 iterations after 5 warmup, torch
2.8.0, Triton 3.6.0, CUDA 12.9.

## Locally (single GPU)

```bash
pip install -e ".[bench,pylate]"
python benchmarks/bench_forward.py      --dtype bf16
python benchmarks/bench_backward.py
python benchmarks/bench_pylate_training.py --batch-size 128 --neg 2
```

Outputs go to `benchmarks/results/{forward,backward}_<gpu>_<dtype>.{md,json}`.

## On a SkyPilot cluster

```bash
# one-shot CI-style run (provisions, tests, benchmarks, exits)
sky jobs launch scripts/sky_test.yaml

# interactive dev box (8×H100)
sky launch -c flash-colbert-dev scripts/sky_dev.yaml
sky ssh flash-colbert-dev

# inside the pod:
cd ~/sky_workdir && CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_forward.py
```

Note: `sky exec` defaults to `CUDA_VISIBLE_DEVICES=""` — always explicitly
set it when running without the YAML (the YAML's `run:` block already
reserves the full GPU allocation).

## Understanding the numbers

The forward speedup depends heavily on the ratio `Lq · Ld / d_tile` — that
is, how much of the work is the tensor-core GEMM vs the reduction around
it. Two regimes:

- **GEMM-bound** (`Lq · Ld ≫ d_pad`): flash-colbert ≈ peak H100 GEMM, naive
  ≈ peak HBM bandwidth. Speedup comes almost entirely from *not moving*
  `[Nq · Nd · Lq · Ld]` fp32 values through HBM and back. Expected speedup:
  **5–15×**.

- **Reduction-bound** (`Lq · Ld ≲ d_pad`): both implementations are
  bandwidth-bound; naive also wastes HBM on the full `S` tensor, but the
  reduction itself isn't much faster on the Triton side. Expected speedup:
  **1.5–3×**.

Training-batch shapes (`Nq = Nd = 32..128`) fall in between and usually see
**2–5× wall-clock** end-to-end, which lines up with the
`bench_pylate_training.py` numbers in the README.

## Memory

Memory savings are more dramatic than time savings. The naive einsum
allocates `Nq · Nd · Lq · Ld · 4 bytes` fp32 scratch; we allocate only the
final `Nq · Nd · 4 bytes` result plus — if autograd is on — a
`Nq · Nd · Lq · 4 bytes` argmax buffer.

| scenario                               | naive scratch | flash-colbert fwd | flash-colbert fwd + argmax |
| -------------------------------------- | ------------- | ----------------- | -------------------------- |
| `Nq=1, Nd=1000, Lq=32, Ld=300`         | 183 MB        | 4 KB              | 128 KB                     |
| `Nq=128, Nd=128, Lq=32, Ld=300`        | 623 MB        | 64 KB             | 2 MB                       |
| `Nq=1, Nd=1000, Lq=1024, Ld=1024`      | 4.5 GB        | 4 KB              | 4 MB                       |
| `Nq=256, Nd=256, Lq=32, Ld=200` (train)| 5.0 GB        | 256 KB            | 8 MB                       |

This is what unlocks the **large-batch regime**: with flash-colbert you can
keep `[Q, D+, D−]` batches up to ~10× larger than PyLate's vanilla path at
the same HBM budget, giving you many more in-batch negatives.
