# Three-way forward bench — NVIDIA H100 80GB HBM3 (bf16)

Stack: `nvcr.io/nvidia/pytorch:25.06-py3` (torch 2.8.0a0+5228986c39.nv25.06,
Triton 3.3.0, CUDA 12.9, driver 580.126.09). Same physical H100 SXM5 SKU
as the dockerhub run below; only the container changes.

All implementations share the same numerical contract: bf16 inputs, fp32
matmul accumulator, fp32 scores. Parity is asserted against the eager
fp32-accumulator reference at `atol=0.01, rtol=0.01` before timing.

Per shape: **warmup = 20 · iters = 100**, CUDA-event timing. `erikkaum/maxsim`
is unavailable on this stack — Erik publishes pre-built wheels for
torch 2.10–2.12 / cu126–cu132 only, and NGC 25.06 ships torch 2.8 /
cu12.9 which isn't on his variant list.

## Results

| shape | LIK | flash-maxsim | naive (fp32 acc) | torch.compile | LIK/FM | naive/LIK |
| --- | --- | --- | --- | --- | --- | --- |
| `rerank-short` (1,1k,32,300,128) | **0.0838** | 0.0891 | 0.2661 | 0.1820 | 1.06× | 3.18× |
| `rerank-long` (1,1k,32,1024,128) | **0.0918** | 0.0967 | 0.7899 | 0.5447 | 1.05× | 8.60× |
| `rerank-medium` (1,1k,128,1024,128) | **0.1016** | 0.1312 | 1.0960 | 0.8327 | 1.29× | 10.79× |
| `rerank-colpali` (1,1k,1024,1024,128) | 0.9428 | **0.8681** | 4.0255 | 3.4984 | **0.92×** | 4.27× |
| `corpus-5k` (1,5k,32,300,128) | **0.1307** | 0.1418 | 1.1761 | 1.1798 | 1.08× | 9.00× |
| `corpus-10k` (1,10k,32,300,128) | **0.2495** | 0.2522 | 2.3253 | 2.3322 | 1.01× | 9.32× |
| `train-32` (32,32,32,300,128) | **0.0852** | 0.1079 | 0.1305 | 0.1857 | 1.27× | 1.53× |
| `train-128` (128,128,32,300,128) | **0.3155** | 0.3446 | 0.7054 | 0.7186 | 1.09× | 2.24× |
| `edge-d64` (1,1k,32,300,64) | **0.0838** | 0.0890 | 0.1693 | 0.1689 | 1.06× | 2.02× |

LIK wins on 8/9 shapes, by 1.01×–1.29×. Loses by 8 % on
`rerank-colpali` (Lq=Ld=1024, d=128) — the compute-bound regime where
the autotune's largest-block configs would help; that's an actionable
LIK tuning bug, not a correctness one. All shapes pass parity.
