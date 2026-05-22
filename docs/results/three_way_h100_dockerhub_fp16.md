# Three-way forward bench — NVIDIA H100 80GB HBM3 (fp16)

Stack: `pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime` (torch 2.10.0+cu126,
Triton 3.6.0, CUDA 12.6, driver 580.126.09). Different container from the
NGC 25.06 run but the same physical H100 SXM5 SKU (verified via GPU UUID).
This is the only stack where `erikkaum/maxsim` has a matching pre-built
variant, so all three kernels can be compared on the same node.

All implementations share the same numerical contract: fp16 inputs, fp32
matmul accumulator, fp32 scores. Parity is asserted against the eager
fp32-accumulator reference at `atol=0.01, rtol=0.01` before timing.

Per shape: **warmup = 20 · iters = 100**, CUDA-event timing.

## Results

| shape | LIK | flash-maxsim | erikkaum/maxsim | naive (fp32 acc) | torch.compile | LIK/FM | erik/LIK | naive/LIK |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `rerank-short` (1,1k,32,300,128) | **0.0630** | 0.0691 | 5.2757 | 0.2578 | 0.1993 | 1.10× | 84× | 4.09× |
| `rerank-long` (1,1k,32,1024,128) | **0.0878** | 0.0912 | 17.8366 | 0.7637 | 0.6197 | 1.04× | 203× | 8.70× |
| `rerank-medium` (1,1k,128,1024,128) | **0.0928** | 0.1007 | 70.9220 | 1.5243 | 1.3587 | 1.09× | 765× | 16.43× |
| `rerank-colpali` (1,1k,1024,1024,128) | 0.6957 | **0.5684** | 570.94 | 9.4472 | 8.9720 | **0.82×** | 821× | 13.58× |
| `corpus-5k` (1,5k,32,300,128) | **0.1266** | 0.1378 | 24.7363 | 1.1547 | 1.1554 | 1.09× | 195× | 9.12× |
| `corpus-10k` (1,10k,32,300,128) | **0.2450** | 0.2473 | 49.3878 | 2.2769 | 2.2779 | 1.01× | 202× | 9.30× |
| `train-32` (32,32,32,300,128) | **0.0638** | 0.0725 | 5.2793 | 0.1286 | 0.1689 | 1.14× | 83× | 2.02× |
| `train-128` (128,128,32,300,128) | **0.2126** | 0.2183 | 81.7137 | 1.5494 | 1.5503 | 1.03× | 384× | 7.29× |
| `edge-d64` (1,1k,32,300,64) | **0.0622** | 0.0712 | 2.6847 | 0.1677 | 0.1681 | 1.14× | 43× | 2.70× |

LIK still wins 8/9 shapes vs flash-maxsim (1.01×–1.14×), loses by 18 % on
`rerank-colpali` — the same shape, same direction as the NGC 25.06 stack,
so it's a kernel-tuning gap, not a Triton-version artefact.

`erikkaum/maxsim` runs **43×–821× slower than LIK on H100**. This is
not a bug in his bench — his README is explicit: "Hopper (sm_90) is
supported via PTX forward-compat but doesn't yet use WGMMA —
Ampere/Lovelace gets the best tuning". On H100 his kernel falls back
to a forward-compat scalar path. The numbers above are what users
would see if they pip-install his kernel on an H100 cluster; on
A100/L4/A10G his published numbers (2.0×–6.2× over naive) likely
hold.

## Cross-stack comparison: kernel cost vs naive cost moves independently

Same physical H100 SXM5 (UUID `GPU-46cc9fa1`), only the container changes.

| shape | NGC 25.06 LIK | dockerhub LIK | NGC 25.06 naive | dockerhub naive |
| --- | --- | --- | --- | --- |
| `rerank-short` | 0.0838 | **0.0630** (-25 %) | 0.2661 | 0.2578 (-3 %) |
| `rerank-long` | 0.0918 | **0.0878** (-4 %) | 0.7899 | 0.7637 (-3 %) |
| `rerank-medium` | 0.1016 | **0.0928** (-9 %) | 1.0960 | **1.5243 (+39 %)** |
| `rerank-colpali` | 0.9428 | **0.6957** (-26 %) | 4.0255 | **9.4472 (+135 %)** |
| `corpus-5k` | 0.1307 | **0.1266** (-3 %) | 1.1761 | 1.1547 (-2 %) |
| `train-128` | 0.3155 | **0.2126** (-33 %) | 0.7054 | **1.5494 (+120 %)** |

On the same hardware, switching only the container changes both the
fused kernel time and the naive einsum time, and they do not move
together. Triton 3.6 (newer) makes LIK 4 %–33 % faster on every shape;
torch 2.10's einsum kernel selection (cuBLAS plan / dispatch path)
slows naive down by 120 %–135 % on the larger shapes. The "speedup
ratio" therefore depends on *which baseline* you compare against, and
that baseline is determined by the surrounding torch / CUDA / Triton
stack, not by the MaxSim kernel.

The original v0.1.0 numbers (text-short 22.7×) reported a LIK time
(0.031 ms) we still can't reach on any container we've tried, AND a
naive time (0.705 ms) that's 2.7×–3.6× slower than what current
torch / cuBLAS gives. Whichever NGC version they ran on was clearly a
different software stack from what NGC 25.06 ships today (which
disagrees with NVIDIA's own release notes claiming "Triton 3.5" —
`triton.__version__` returns 3.3 inside the container).
