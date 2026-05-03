# Benchmarks

Single H100 80 GB SXM, bf16 compute (fp16 for LateOn / ModernColBERT
shapes), fp32 accumulator, 50 iterations after 5 warmup, `torch 2.8`,
`triton 3.6`, `cuda 12.9`.

## Reproducing

```bash
pip install -e ".[dev,pylate]"

# forward (reranking / inference)
python benchmarks/bench_forward.py

# backward — auto / atomic / csr / unified sweep
python benchmarks/bench_backward_method.py
python benchmarks/bench_backward_unified.py

# LateOn / LateOn-Code / ModernColBERT long-document regime
python benchmarks/bench_lateon.py

# fused L2-normalize
python benchmarks/bench_normalize.py

# fused D-side head (training)
python benchmarks/bench_fused_head_train.py

# FP8 inference (Hopper / Blackwell)
python benchmarks/bench_fp8.py

# PyLate end-to-end
python benchmarks/bench_pylate_training.py --batch-size 128 --neg 2
python benchmarks/bench_cached_maxsim.py
python benchmarks/bench_pylate_lateon.py --recipe contrastive --batch-size 4 --Ld 8192

# 8-GPU DDP CachedContrastive
torchrun --standalone --nproc_per_node=8 \
  benchmarks/bench_pylate_lateon.py --recipe reason \
  --batch-size 32 --mini-batch-size 32 --Ld 2048 --grad-checkpoint --ddp

# Head-to-head vs flash-maxsim (same Triton-MaxSim math)
pip install flash-maxsim
python benchmarks/bench_flash_maxsim.py

# PLAID rerank vs fast-plaid
python benchmarks/bench_decompress_maxsim.py     # vs PyTorch transliteration
python benchmarks/bench_fastplaid_e2e.py         # vs `fast_plaid.engine.search()`

# Apple Silicon (MPS): torch.compile dispatch vs eager reference
python benchmarks/bench_mps.py

# everything at once (writes to $OUTDIR)
OUTDIR=benchmarks/results bash scripts/run_all_benchmarks.sh
```

Results land in `benchmarks/results/*.{json,md}`.

## On a SkyPilot cluster

```bash
sky jobs launch scripts/sky_test.yaml             # CI-style: tests + bench_forward + bench_backward
sky jobs launch scripts/sky_lateon_edge.yaml      # LateOn-Code-edge end-to-end
sky jobs launch scripts/sky_decompress_bench.yaml # PLAID decompress + MaxSim
sky jobs launch scripts/sky_fastplaid_e2e.yaml    # vs `fast_plaid.engine.search()`
```

## Forward (reranking / inference)


| shape                                       | fused    | naive einsum | speedup | scratch    |
| ------------------------------------------- | -------- | ------------ | ------- | ---------- |
| `Nq=1, Nd=1000, Lq=32, Ld=300`              | 0.031 ms | 0.705 ms     | 22.7×   | 183 MB → 0 |
| `Nq=1, Nd=10 000, Lq=32, Ld=300`            | 0.557 ms | 7.112 ms     | 12.8×   | 1.8 GB → 0 |
| `Nq=1, Nd=1000, Lq=1024, Ld=1024` (ColPali) | 1.518 ms | 11.967 ms    | 7.9×    | 4.5 GB → 0 |


### vs `flash-maxsim` (same Triton MaxSim math)

`flash-maxsim` (Roi Pony / IBM) was the first public Triton MaxSim
kernel and the direct inspiration for this library. The numbers below
come from `benchmarks/bench_flash_maxsim.py` on H100 80 GB SXM, bf16,
50-iter median over CUDA events. `flash-maxsim` 0.2.0 has no
`normalize=True` knob and no autograd-aware backward, so we report
plain forward only.


| shape                                             | ours  | flash-maxsim | speedup |
| ------------------------------------------------- | ----- | ------------ | ------- |
| `rerank-short` (Nq=1, Nd=1k, Lq=32, Ld=300)       | 0.096 | 0.109        | 1.13×   |
| `rerank-long` (Nq=1, Nd=1k, Lq=32, Ld=1024)       | 0.153 | 0.168        | 1.10×   |
| `rerank-very-long` (Nq=1, Nd=500, Lq=32, Ld=4096) | 0.237 | 0.252        | 1.06×   |
| `rerank-colpali` (Nq=1, Nd=500, Lq=1024, Ld=1024) | 0.409 | 0.495        | 1.21×   |
| `rerank-10k` (Nq=1, Nd=10k, Lq=32, Ld=300)        | 0.310 | 0.324        | 1.05×   |
| `train-in-batch-32` (Nq=Nd=32, Lq=32, Ld=200)     | 0.087 | 0.100        | 1.15×   |
| `train-in-batch-128` (Nq=Nd=128, Lq=32, Ld=200)   | 0.233 | 0.294        | 1.26×   |
| `train-long-doc` (Nq=Nd=16, Lq=32, Ld=2048)       | 0.088 | 0.088        | 1.01×   |
| `edge-d48` (Nq=1, Nd=4k, Lq=32, Ld=2048, d=48)    | 0.320 | 0.341        | 1.07×   |
| `edge-d64` (Nq=1, Nd=10k, Lq=32, Ld=300, d=64)    | 0.190 | 0.218        | 1.14×   |


flash-maxsim trails by 1.05–1.26× depending on shape; on `train-long-doc`
both kernels are saturated by L2-normalized HBM traffic and tie. The
real differentiators are elsewhere: a fused `normalize=True` (no
extra HBM round-trip), a real autograd-aware backward (`unified` /
`csr` / `atomic`), packed/varlen, PLAID residual decompression, and a
fused D-side projection head — none of which `flash-maxsim` ships.

## Fused L2-normalize

`F.normalize(Q) → maxsim` becomes a single kernel (`normalize=True`).
The fused path normalizes in SRAM, eliminating two HBM round-trips, and
the backward correctly applies the L2-norm Jacobian.


| shape                                 | `F.normalize` + maxsim | fused    | speedup   |
| ------------------------------------- | ---------------------- | -------- | --------- |
| text-short (`Nq=1, Nd=1k, Ld=300`)    | 0.491 ms               | 0.064 ms | **7.7×**  |
| text-long (`Nq=1, Nd=1k, Ld=1024`)    | 1.569 ms               | 0.094 ms | **16.7×** |
| bigbatch-300 (`Nq=32, Nd=32, Ld=300`) | 0.225 ms               | 0.064 ms | 3.5×      |
| bigbatch-2k (`Nq=8, Nd=16, Ld=2048`)  | 0.219 ms               | 0.065 ms | 3.4×      |
| corpus-10k (`Nq=1, Nd=10k, Ld=300`)   | 4.442 ms               | 0.303 ms | **14.7×** |


## PLAID / ColBERTv2

End-to-end vs `fast_plaid.engine.search()`: build the index with
fast-plaid, time `engine.search()`, then load the same compressed
tensors and call `maxsim_residual_varlen`. Same inputs, same outputs.
Reproduce with `scripts/sky_fastplaid_e2e.yaml`.


| corpus shape (nbits=2) | `engine.search()` | `maxsim_residual_varlen` (4 k cands) | speedup   |
| ---------------------- | ----------------- | ------------------------------------ | --------- |
| 5 000 docs × 200 tok   | 23.4 ms           | 1.22 ms                              | **19.2×** |
| 10 000 docs × 300 tok  | 48.6 ms           | 1.69 ms                              | **28.7×** |
| 10 000 docs × 512 tok  | 83.1 ms           | 2.71 ms                              | **30.7×** |


Against a PyTorch transliteration of fast-plaid's exact decompress →
pad → matmul → reduce slice, the fused varlen kernel is **3.4–3.7×
faster at 10–34× less GPU memory** — no
`[Ntop, max_Ld, packed_dim]` padded scratch is allocated.

## Fused D-side head (training)

Replaces `F.linear → F.normalize → maxsim` for the
hidden-state → embedding → MaxSim path. Forward saves an argmax;
backward gathers `H_d` only at winning positions and is closed-form.
H100 bf16, forward + backward:


| shape                                           | unfused  | fused   | speedup   |
| ----------------------------------------------- | -------- | ------- | --------- |
| LateOn `Nd=128, Ld=1024, d_model=768`           | 1.45 ms  | 0.96 ms | **1.52×** |
| LateOn-Code `Nd=512, Ld=2048`                   | 7.31 ms  | 1.67 ms | **4.37×** |
| LateOn-Code `Nd=1024, Ld=2048`                  | 14.15 ms | 3.05 ms | **4.64×** |
| LateOn-Code-edge `Nd=256, Ld=4096, d_model=384` | 5.13 ms  | 1.32 ms | **3.88×** |


The win shrinks as `Nq · Lq / Ld → 1` and as `d_model` shrinks below
~128. Memory and derivation in `[design.md](design.md)`.

## End-to-end LateOn-Code-edge training (17 M encoder)

`losses.Contrastive`, in-batch negatives, bf16, grad-checkpointing,
1×H100. Reproduce with `scripts/sky_lateon_edge.yaml`.


| setup                  | vanilla  | fused    | speedup   | peak (v → f)   |
| ---------------------- | -------- | -------- | --------- | -------------- |
| bs=256, Lq=32, Ld=256  | 114.3 ms | 90.3 ms  | **1.27×** | 11.5 → 9.6 GB  |
| bs=192, Lq=32, Ld=512  | 152.5 ms | 126.0 ms | **1.21×** | 16.6 → 14.7 GB |
| bs=128, Lq=32, Ld=1024 | 223.7 ms | 196.4 ms | **1.14×** | 22.6 → 21.5 GB |
| bs=64, Lq=32, Ld=2048  | 293.2 ms | 281.0 ms | 1.04×     | 25.8 GB        |


Smaller encoder + bigger effective batch ⇒ bigger MaxSim slice ⇒ bigger
end-to-end speedup. These numbers use `patch_pylate()`;
`maxsim_from_hidden_train` adds another ~1–3 ms / step but isn't wired
into PyLate's loss path because PyLate's `Dense` projection runs inside
the encoder forward.

## Backward paths

H100 fp32 hardware atomics are very fast, so CSR only wins at very
large shapes. `auto` picks the right path on every shape we measured.


| shape                         | atomic | csr  | auto | auto picks |
| ----------------------------- | ------ | ---- | ---- | ---------- |
| `train-32` (32 × 32, Lq=32)   | 0.48   | 0.66 | 0.49 | atomic     |
| `train-128` (128 × 128)       | 0.50   | 0.65 | 0.51 | atomic     |
| `train-256` (256 × 256)       | 1.85   | 1.25 | 1.25 | **csr**    |
| `retrieval` (16 × 512, L=300) | 0.47   | 0.65 | 0.48 | atomic     |
| `long-Lq` (Lq=1024)           | 0.85   | 0.65 | 0.66 | **csr**    |
| `huge-Nd` (Nd=1024)           | 0.66   | 0.65 | 0.65 | **csr**    |


CSR is bitwise-reproducible across runs (no atomics); `atomic` /
`unified` drift within fp32 ULP (~1e-6 relative) because reduction
order depends on thread scheduling.

```python
from late_interaction_kernels import set_backward_method
set_backward_method("csr")        # | "atomic" | "unified" | "auto"

# or per call
maxsim(Q, D, normalize=True, backward="csr")
```

## End-to-end PyLate `Contrastive` training


| batch × negs | vanilla PyLate | fused    | speedup |
| ------------ | -------------- | -------- | ------- |
| 64 × 1       | 2.70 ms        | 0.92 ms  | 2.93×   |
| 128 × 2      | 14.95 ms       | 4.86 ms  | 3.08×   |
| 256 × 3      | 75.04 ms       | 26.28 ms | 2.86×   |


## LateOn / ModernColBERT (long documents)

At 2k–4k the naive einsum still fits, so you see speedup *and* memory
ratios. At 8k+ naive OOMs on 80 GB at sane training batch sizes.
Numbers apply equally to `lightonai/LateOn`,
`lightonai/GTE-ModernColBERT-v1`, `lightonai/LateOn-Code` (same
backbone, `d=128`).

MaxSim only (one `colbert_scores` call, fp16, `auto` backward):


| shape                                     | fwd fused | fwd naive | bwd fused | bwd naive | peak fused | peak naive |
| ----------------------------------------- | --------- | --------- | --------- | --------- | ---------- | ---------- |
| `Nq=8, Nd=16, Lq=32, Ld=2048` train-2k    | 0.07 ms   | 0.15 ms   | 0.47 ms   | 0.56 ms   | 96 MB      | 152 MB     |
| `Nq=8, Nd=16, Lq=32, Ld=4096` train-4k    | 0.08 ms   | 0.15 ms   | 0.45 ms   | 0.55 ms   | 129 MB     | 241 MB     |
| `Nq=16,Nd=32, Lq=32, Ld=4096` bigbatch-4k | **0.08**  | 0.34      | **0.46**  | 1.06      | **193 MB** | **672 MB** |
| `Nq=1, Nd=64, Lq=32, Ld=4096` rerank-4k   | 0.07 ms   | 0.24 ms   | 0.46 ms   | 0.71 ms   | 320 MB     | 416 MB     |
| `Nq=8, Nd=16, Lq=32, Ld=8192` train-8k    | 0.07 ms   | **OOM**   | 0.39 ms   | **OOM**   | 192 MB     | OOM        |
| `Nq=16,Nd=32, Lq=32, Ld=8192` bigbatch-8k | 0.15 ms   | **OOM**   | 0.41 ms   | **OOM**   | 320 MB     | OOM        |
| `Nq=1, Nd=256,Lq=32, Ld=8192` rerank-8k   | 0.18 ms   | **OOM**   | 1.12 ms   | **OOM**   | 2.1 GB     | OOM        |
| `Nq=1, Nd=32, Lq=32, Ld=16384` huge-doc   | 0.09 ms   | **OOM**   | 0.42 ms   | **OOM**   | 576 MB     | OOM        |


### LightOn cached-contrastive (MaxSim isolation)

`pylate.losses.CachedContrastive` chunks MaxSim into `(bs / mini)**2`
Python-level calls. The fused kernel collapses that double loop into
one call that never materializes `S`. `Lq=128, d=128, mini=32`:


| shape                                       | tiles | vanilla fwd+bwd | fused fwd+bwd | speedup   | vanilla peak | fused peak | mem×     |
| ------------------------------------------- | ----- | --------------- | ------------- | --------- | ------------ | ---------- | -------- |
| `bs=64, Ld=2048`                            | 4     | 13.5 ms         | 1.3 ms        | **10.3×** | 1.1 GB       | 0.2 GB     | 5.7×     |
| `bs=64, Ld=4096`                            | 4     | 26.4 ms         | 2.2 ms        | **11.9×** | 2.2 GB       | 0.3 GB     | 6.8×     |
| `bs=64, Ld=8192`                            | 4     | 55.3 ms         | 4.0 ms        | **13.9×** | 4.3 GB       | 0.6 GB     | 7.5×     |
| `bs=128, Ld=4096`                           | 16    | 107.1 ms        | 8.0 ms        | **13.3×** | 2.3 GB       | 0.6 GB     | 4.0×     |
| `bs=128, Ld=8192`                           | 16    | 224.0 ms        | 15.7 ms       | **14.3×** | 4.6 GB       | 1.1 GB     | 4.2×     |
| `bs=256, Ld=2048`                           | 64    | 224.5 ms        | 17.7 ms       | **12.7×** | 1.4 GB       | 0.6 GB     | 2.2×     |
| `bs=256, Ld=4096`                           | 64    | 439.9 ms        | 33.1 ms       | **13.3×** | 2.6 GB       | 1.1 GB     | 2.3×     |
| **bs=256, Ld=8192** (LightOn's real recipe) | 64    | **915.9 ms**    | **66.3 ms**   | **13.8×** | **5.1 GB**   | **2.1 GB** | **2.4×** |


### End-to-end on `LateOn` (149 M)

Real `pylate.models.ColBERT("lightonai/LateOn")` (22-layer ModernBERT
with FlashAttention-2, 8192-token context), AdamW, bf16 autocast,
per-rank peak memory.

`losses.Contrastive` (no encoder chunking):


| setup                                  | vanilla PyLate | fused    | speedup | peak (v → f)   |
| -------------------------------------- | -------------- | -------- | ------- | -------------- |
| 1 × H100, bs=8, Lq=32, Ld=2048         | 227.2 ms       | 220.6 ms | 1.03×   | 29.2 → 29.2 GB |
| 1 × H100, bs=8, Lq=32, Ld=4096         | 428.3 ms       | 428.7 ms | 1.00×   | 56.3 → 56.3 GB |
| 1 × H100, bs=4, Lq=32, Ld=8192         | 504.2 ms       | 504.1 ms | 1.00×   | 56.2 → 56.2 GB |
| 8 × H100 DDP, bs=4, Ld=8192 (per-rank) | 505.7 ms       | 504.7 ms | 1.00×   | 56.8 → 56.8 GB |


`losses.CachedContrastive` (`gather_across_devices=True`, grad-ckpt):


| setup                                     | vanilla PyLate | fused     | speedup | peak (per rank) |
| ----------------------------------------- | -------------- | --------- | ------- | --------------- |
| 8 × H100 DDP, bs=4/dev, mini=4, Ld=8192   | 664.9 ms       | 662.1 ms  | 1.00×   | 30.2 GB         |
| 8 × H100 DDP, bs=16/dev, mini=8, Ld=4096  | 1164.2 ms      | 1141.0 ms | 1.02×   | 30.2 GB         |
| 8 × H100 DDP, bs=8/dev, mini=8, Ld=8192   | 1250.4 ms      | 1243.0 ms | 1.01×   | 57.5 GB         |
| 8 × H100 DDP, bs=16/dev, mini=16, Ld=4096 | 1078.1 ms      | 1047.4 ms | 1.03×   | 57.5 GB         |
| 8 × H100 DDP, bs=32/dev, mini=32, Ld=2048 | 1020.1 ms      | 960.2 ms  | 1.06×   | 57.4 GB         |


LateOn's 149 M ModernBERT with FA-2 dominates step time by ~10×, so
end-to-end speedup is **1.00–1.06×** even though MaxSim moves up to
13.8× in isolation. On smaller encoders the slice grows: 17 M
LateOn-Code-edge moves **1.04–1.27×** end-to-end (table above).

## Edge models (`d ∈ {48, 64}`)

Edge ColBERT models (`d ∈ {48, 64}`) are more memory-bound, so the
fused kernel widens its lead. `bench_inference_edge.py`, bf16, 50-iter:


| shape                                       | fused    | naive (fp32) | speedup   | fused mem | naive mem |
| ------------------------------------------- | -------- | ------------ | --------- | --------- | --------- |
| LateOn-Code-edge `Nd=1 000, Ld=1 024, d=48` | 0.072 ms | 0.380 ms     | **5.3×**  | 0.0 MB    | 314 MB    |
| LateOn-Code-edge `Nd=1 000, Ld=4 096, d=48` | 0.137 ms | 1.412 ms     | **10.3×** | 0.0 MB    | 1.2 GB    |
| LateOn-Code-edge `Nd=1 000, Ld=8 192, d=48` | 0.266 ms | 2.910 ms     | **10.9×** | 0.0 MB    | 2.5 GB    |
| LateOn-Code-edge `Nd=16 000, Ld=512, d=48`  | 0.252 ms | 2.897 ms     | **11.5×** | 0.1 MB    | 2.5 GB    |
| mxbai-edge `Nd=1 000, Ld=4 096, d=64`       | 0.172 ms | 1.730 ms     | **10.0×** | 0.0 MB    | 1.5 GB    |
| mxbai-edge `Nd=16 000, Ld=512, d=64`        | 0.331 ms | 3.528 ms     | **10.7×** | 0.1 MB    | 3.0 GB    |


## Where this kernel actually moves the e2e needle

1. **Inference / reranking** — no encoder backward → MaxSim *is* the
  step. 7–23×.
2. **Small-encoder training** — encoder small enough that MaxSim is
  material; LateOn-Code-edge moves 1.04–1.27× end-to-end.
3. **Long-context regimes** (`Ld ≥ 8k`) — fused kernels run, naive
  doesn't.
4. **Compressed indices** — PLAID rerank vs `engine.search()` is
  19–30×.
5. **KD / offline scoring** — teacher + student both do MaxSim, no
  per-step encoder cost to hide behind.

For plain `LateOn` fine-tuning the kernel is essentially free: same
step time, same VRAM, same numerics, one `patch_pylate()` call.

## Memory

Naive allocates `Nq · Nd · Lq · Ld · 4` bytes of fp32 scratch. Fused
keeps the `Nq · Nd` result plus — only if autograd is on — a
`Nq · Nd · Lq` int32 argmax buffer.


| scenario                          | naive scratch | fused fwd | fused fwd + argmax |
| --------------------------------- | ------------- | --------- | ------------------ |
| `Nq=1, Nd=1000, Lq=32, Ld=300`    | 183 MB        | 4 KB      | 128 KB             |
| `Nq=128, Nd=128, Lq=32, Ld=300`   | 623 MB        | 64 KB     | 2 MB               |
| `Nq=1, Nd=1000, Lq=1024, Ld=1024` | 4.5 GB        | 4 KB      | 4 MB               |
| `Nq=16, Nd=32, Lq=32, Ld=8192`    | 2.1 GB        | 64 KB     | 64 KB              |


This is what unlocks large in-batch negatives: at the same HBM budget
you fit ~5–10× more of them than vanilla PyLate.

## Apple Silicon (MPS)

`MaxSimScorer` and `retrieve` work on `mps:0` tensors. Two
implementations land on Apple Silicon and the dispatch picks per call:

* **`metal`** — fused `simdgroup_matrix` kernel
  (`late_interaction_kernels.metal.maxsim_inference_metal`, JIT-compiled
  via `torch.mps.compile_shader`). Forward-only; never materialises the
  `[Nq · Nd · Lq · Ld]` similarity tensor.
* **`compile`** — `torch.compile`-fused dense reference. Autograd-aware,
  carries every training-time call.

Apple M4, fp16, 30-iter median (`benchmarks/bench_mps.py`). `metal`
needs `d ≤ 128` and `d % 8 == 0`; outside that the dispatch transparently
falls back to `compile`.


| shape                                          | metal    | compile  | eager    | metal vs eager | metal vs compile | compile vs eager |
| ---------------------------------------------- | -------- | -------- | -------- | -------------- | ---------------- | ---------------- |
| `rerank-short` (Nq=1, Nd=1000, Lq=32, Ld=300)  | 8.42 ms  | 10.87 ms | 18.68 ms | **2.22×**      | 1.29×            | 1.72×            |
| `rerank-mid` (Nq=1, Nd=500, Lq=32, Ld=1024)    | 16.43 ms | 18.28 ms | 31.13 ms | **1.89×**      | 1.11×            | 1.70×            |
| `rerank-10k` (Nq=1, Nd=10k, Lq=32, Ld=300)     | 55.5 ms  | 93.9 ms  | 170.5 ms | **3.07×**      | 1.69×            | 1.82×            |
| `colpali` (Nq=1, Nd=100, Lq=32, Ld=1024)       | 3.11 ms  | 3.37 ms  | 6.51 ms  | **2.10×**      | 1.09×            | 1.93×            |
| `colpali-big` (Nq=1, Nd=500, Lq=32, Ld=1024)   | 9.84 ms  | 17.27 ms | 28.89 ms | **2.94×**      | 1.76×            | 1.67×            |
| `edge-d48` (Nq=1, Nd=4k, Lq=32, Ld=1024, d=48) | 33.3 ms  | 67.2 ms  | 107.2 ms | **3.22×**      | 2.02×            | 1.59×            |
| `edge-d64` (Nq=1, Nd=1k, Lq=32, Ld=300, d=64)  | 4.82 ms  | 5.40 ms  | 13.06 ms | **2.71×**      | 1.12×            | 2.42×            |
| `train-batch` (Nq=Nd=32, Lq=32, Ld=200)        | 5.73 ms  | 1.64 ms  | 2.33 ms  | 0.41× → 1.42×  | 0.29× (compile)  | 1.42×            |


For someone moving from plain PyTorch to this library, the headline is
`metal vs eager`: **1.9–3.2×** on every realistic inference shape. On
`train-batch` the Metal kernel's launch overhead dominates, so the
dispatch heuristic (`Nq * Nd ≥ 64 ∧ Ld ≥ 192`) routes it to `compile`
automatically — the user still gets 1.42× over eager via that path.
Override with `LIK_FORCE_MPS_BACKEND={metal,compile,reference}` or
`LIK_DISABLE_COMPILE=1` if you need explicit control.

Memory is the second story: because `metal` streams `D` in 32-row tiles
through threadgroup memory, peak working-set is one output tensor
(8 MB) regardless of the corpus size. On `rerank-10k` and `edge-d48`
the compile / eager paths materialise 2.5 GB / 750 MB intermediates —
**~300×** memory reduction.

The Metal kernel uses Apple's 8×8 `simdgroup_matrix` MMA (the Metal
analogue of CUDA tensor cores) on top of a *persistent* threadgroup
design: each launch serves 8 consecutive `j` values, loading `Q` once
into a register-resident `simdgroup_matrix` cache that's reused across
every `(j, d-chunk)` pair (collapsing 8·Ld_chunks redundant LDS reads
into a single load pass). The cooperative D load stages each row
through per-thread registers so the L2-normalize fold pays one LDS
write instead of three. Tolerances stay inside 5e-3 relative for fp16
and 3e-2 for bf16.