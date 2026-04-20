# Benchmarks

All numbers are measured on a **single H100 80 GB SXM** in bf16 compute (fp16 for
ModernColBERT), fp32 accumulator, 50 iterations after 5 warmup, torch 2.8, Triton
3.6, CUDA 12.9.

## Reproducing

```bash
pip install -e ".[dev,pylate]"

# forward (reranking / inference)
python benchmarks/bench_forward.py

# backward + auto / atomic / csr sweep
python benchmarks/bench_backward_method.py

# ModernColBERT long-document regime (Ld ∈ {2k, 4k, 8k, 16k})
python benchmarks/bench_moderncolbert.py

# MaxSim-only PyLate training step (synthetic embeddings)
python benchmarks/bench_pylate_training.py --batch-size 128 --neg 2

# The exact CachedContrastive chunked-MaxSim pattern LightOn uses for
# Reason-ModernColBERT (bs=64..256, Ld=2k..8k)
python benchmarks/bench_cached_maxsim.py

# End-to-end PyLate ModernColBERT training — plain Contrastive
python benchmarks/bench_pylate_moderncolbert.py --recipe contrastive \
    --batch-size 4 --Ld 8192

# End-to-end LightOn Reason-ModernColBERT recipe (CachedContrastive, bf16, grad-ckpt)
torchrun --standalone --nproc_per_node=8 \
    benchmarks/bench_pylate_moderncolbert.py --recipe reason \
    --batch-size 32 --mini-batch-size 32 --Ld 2048 --grad-checkpoint --ddp

# FastPlaid rerank-step comparison (requires `pip install fast-plaid`)
python benchmarks/bench_fastplaid.py
```

Results land in `benchmarks/results/*.json`.

## On a SkyPilot cluster

```bash
# one-shot CI run (provisions, tests, benchmarks, exits)
sky jobs launch scripts/sky_test.yaml

# long-lived dev box (8×H100)
sky launch -c late-interaction-kernels-dev scripts/sky_dev.yaml
sky ssh late-interaction-kernels-dev
# (inside the pod, with a GPU mask):
cd ~/sky_workdir && CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_forward.py
```

## Forward (reranking / inference)


| shape                                       | late-interaction-kernels | naive einsum | speedup | scratch    |
| ------------------------------------------- | ------------- | ------------ | ------- | ---------- |
| `Nq=1, Nd=1000, Lq=32, Ld=300`              | 0.031 ms      | 0.705 ms     | 22.7×   | 183 MB → 0 |
| `Nq=1, Nd=10 000, Lq=32, Ld=300`            | 0.557 ms      | 7.112 ms     | 12.8×   | 1.8 GB → 0 |
| `Nq=1, Nd=1000, Lq=1024, Ld=1024` (ColPali) | 1.518 ms      | 11.967 ms    | 7.9×    | 4.5 GB → 0 |


## Backward paths (`atomic` vs `csr` vs `auto`)

On H100, fp32 hardware `atomic_add` is *very* good, so CSR only wins at very large
shapes. `auto` picks the right path on 11/11 shapes we measured.


| shape                         | atomic | csr  | auto | auto picks |
| ----------------------------- | ------ | ---- | ---- | ---------- |
| `train-32` (32 × 32, Lq=32)   | 0.48   | 0.66 | 0.49 | atomic     |
| `train-128` (128 × 128)       | 0.50   | 0.65 | 0.51 | atomic     |
| `train-256` (256 × 256)       | 1.85   | 1.25 | 1.25 | **csr**    |
| `retrieval` (16 × 512, L=300) | 0.47   | 0.65 | 0.48 | atomic     |
| `long-Lq` (Lq=1024)           | 0.85   | 0.65 | 0.66 | **csr**    |
| `huge-Nd` (Nd=1024)           | 0.66   | 0.65 | 0.65 | **csr**    |


You can force a path:

```python
from late_interaction_kernels import set_backward_method
set_backward_method("atomic")   # | "csr" | "auto"
```

CSR is bitwise-reproducible across runs (no atomics); atomic drifts within fp32 ULP (~1e-6
relative) because `atomic_add` reduction order depends on thread scheduling.

## End-to-end PyLate `Contrastive` training


| batch × negs | vanilla PyLate | late-interaction-kernels | speedup |
| ------------ | -------------- | ------------- | ------- |
| 64 × 1       | 2.70 ms        | 0.92 ms       | 2.93×   |
| 128 × 2      | 14.95 ms       | 4.86 ms       | 3.08×   |
| 256 × 3      | 75.04 ms       | 26.28 ms      | 2.86×   |


## ModernColBERT (long documents)

At 2k–4k the naive einsum still fits, so you can see real speedup *and* memory
ratios. At 8k+ the naive path OOMs on 80 GB at any sane training batch.

MaxSim-only (one `colbert_scores` call, fp16 inputs, `auto` backward):


| shape                                     | fwd flash | fwd naive | bwd flash | bwd naive | peak flash | peak naive |
| ----------------------------------------- | --------- | --------- | --------- | --------- | ---------- | ---------- |
| `Nq=8, Nd=16, Lq=32, Ld=2048` train-2k    | 0.07 ms   | 0.15 ms   | 0.47 ms   | 0.56 ms   | 96 MB      | 152 MB     |
| `Nq=8, Nd=16, Lq=32, Ld=4096` train-4k    | 0.08 ms   | 0.15 ms   | 0.45 ms   | 0.55 ms   | 129 MB     | 241 MB     |
| `Nq=16,Nd=32, Lq=32, Ld=4096` bigbatch-4k | **0.08**  | 0.34      | **0.46**  | 1.06      | **193 MB** | **672 MB** |
| `Nq=1, Nd=64, Lq=32, Ld=4096` rerank-4k   | 0.07 ms   | 0.24 ms   | 0.46 ms   | 0.71 ms   | 320 MB     | 416 MB     |
| `Nq=8, Nd=16, Lq=32, Ld=8192` train-8k    | 0.07 ms   | **OOM**   | 0.39 ms   | **OOM**   | 192 MB     | OOM        |
| `Nq=16,Nd=32, Lq=32, Ld=8192` bigbatch-8k | 0.15 ms   | **OOM**   | 0.41 ms   | **OOM**   | 320 MB     | OOM        |
| `Nq=1, Nd=256,Lq=32, Ld=8192` rerank-8k   | 0.18 ms   | **OOM**   | 1.12 ms   | **OOM**   | 2.1 GB     | OOM        |
| `Nq=1, Nd=32, Lq=32, Ld=16384` huge-doc   | 0.09 ms   | **OOM**   | 0.42 ms   | **OOM**   | 576 MB     | OOM        |


At `bigbatch-4k` the MaxSim kernel alone is **4.3× faster on forward and
2.3× faster on backward**, with **3.5× less peak memory** for that op
(480 MB saved). `auto` picks `atomic` for every ModernColBERT shape
(`Lq=32` ⇒ low atomic contention, `Nd < 1024` ⇒ CSR's sort overhead dominates).

### Isolated MaxSim at the LightOn Reason-ModernColBERT recipe shapes

`pylate.losses.CachedContrastive` handles large effective batches by
manually chunking MaxSim into `(bs / mini)**2` Python-level `colbert_scores`
calls — see the comment in `pylate/losses/cached_contrastive.py`:
*"We chunk the scores computation to avoid OOM because MaxSim can get
expensive with large batch sizes/long documents"*. late-interaction-kernels replaces
that whole double loop with **one** fused call that never materializes
`S`.

These numbers strip out the encoder entirely and measure *just* the MaxSim
part (`Lq=128, d=128, mini_batch_size=32`, the exact Reason-ModernColBERT
hyperparams):


| shape                                         | tiles  | vanilla fwd+bwd | flash fwd+bwd | speedup   | vanilla peak | flash peak | mem×     |
| --------------------------------------------- | ------ | --------------- | ------------- | --------- | ------------ | ---------- | -------- |
| `bs=64, Ld=2048`                              | 4      | 13.5 ms         | 1.3 ms        | **10.3×** | 1.1 GB       | 0.2 GB     | 5.7×     |
| `bs=64, Ld=4096`                              | 4      | 26.4 ms         | 2.2 ms        | **11.9×** | 2.2 GB       | 0.3 GB     | 6.8×     |
| `bs=64, Ld=8192`                              | 4      | 55.3 ms         | 4.0 ms        | **13.9×** | 4.3 GB       | 0.6 GB     | 7.5×     |
| `bs=128, Ld=4096`                             | 16     | 107.1 ms        | 8.0 ms        | **13.3×** | 2.3 GB       | 0.6 GB     | 4.0×     |
| `bs=128, Ld=8192`                             | 16     | 224.0 ms        | 15.7 ms       | **14.3×** | 4.6 GB       | 1.1 GB     | 4.2×     |
| `bs=256, Ld=2048`                             | 64     | 224.5 ms        | 17.7 ms       | **12.7×** | 1.4 GB       | 0.6 GB     | 2.2×     |
| `bs=256, Ld=4096`                             | 64     | 439.9 ms        | 33.1 ms       | **13.3×** | 2.6 GB       | 1.1 GB     | 2.3×     |
| `**bs=256, Ld=8192` (LightOn's real recipe)** | **64** | **915.9 ms**    | **66.3 ms**   | **13.8×** | **5.1 GB**   | **2.1 GB** | **2.4×** |


At LightOn's exact Reason-ModernColBERT shape, vanilla PyLate is spending
**~900 ms per step just on MaxSim fwd+bwd** — that's a full second late-interaction-kernels
gives back. The isolation here is deliberate: it's the purest read on what
the kernel swap alone changes.

### End-to-end training step

Measured with the actual `pylate.models.ColBERT("lightonai/GTE-ModernColBERT-v1")`
(150 M params, 22-layer ModernBERT with flash-attention-2, 8192-token context),
AdamW, `bf16` autocast. Per-rank peak memory. `--grad-checkpoint` enables
gradient checkpointing on the ModernBERT encoder.

**Plain `losses.Contrastive*`* (no encoder chunking):


| setup                                  | vanilla PyLate | late-interaction-kernels | speedup | peak (v → f)   |
| -------------------------------------- | -------------- | ------------- | ------- | -------------- |
| 1 × H100, bs=8, Lq=32, Ld=2048         | 227.2 ms       | 220.6 ms      | 1.03×   | 29.2 → 29.2 GB |
| 1 × H100, bs=8, Lq=32, Ld=4096         | 428.3 ms       | 428.7 ms      | 1.00×   | 56.3 → 56.3 GB |
| 1 × H100, bs=4, Lq=32, Ld=8192         | 504.2 ms       | 504.1 ms      | 1.00×   | 56.2 → 56.2 GB |
| 8 × H100 DDP, bs=4, Ld=8192 (per-rank) | 505.7 ms       | 504.7 ms      | 1.00×   | 56.8 → 56.8 GB |


`**losses.CachedContrastive**` (LightOn's Reason recipe: `gather_across_devices=True`, grad-ckpt, bf16):


| setup                                     | vanilla PyLate | late-interaction-kernels | speedup | peak (per rank) |
| ----------------------------------------- | -------------- | ------------- | ------- | --------------- |
| 8 × H100 DDP, bs=4/dev, mini=4, Ld=8192   | 664.9 ms       | 662.1 ms      | 1.00×   | 30.2 GB         |
| 8 × H100 DDP, bs=16/dev, mini=8, Ld=4096  | 1164.2 ms      | 1141.0 ms     | 1.02×   | 30.2 GB         |
| 8 × H100 DDP, bs=8/dev, mini=8, Ld=8192   | 1250.4 ms      | 1243.0 ms     | 1.01×   | 57.5 GB         |
| 8 × H100 DDP, bs=16/dev, mini=16, Ld=4096 | 1078.1 ms      | 1047.4 ms     | 1.03×   | 57.5 GB         |
| 8 × H100 DDP, bs=32/dev, mini=32, Ld=2048 | 1020.1 ms      | 960.2 ms      | 1.06×   | 57.4 GB         |


**What to take away:**

- Per-step e2e speedup is **1.00–1.06×**. Even in the Reason recipe, the
22-layer ModernBERT forward+backward (with FlashAttention-2 internally)
dominates step time by ~10×.
- Peak per-rank VRAM is **activation-bound from the transformer**, not
MaxSim. late-interaction-kernels saves GB on MaxSim scratch, but that's < 10 % of
peak when the transformer is ~50 GB.
- **The full MaxSim win is real and measurable in isolation** (table
above: up to 13.8× / 2.4× memory at the exact LightOn shape), it's just
that the rest of the step is bigger than MaxSim on these ModernBERT
configurations.

### Where late-interaction-kernels actually moves the e2e needle

1. **Inference / reranking** — no encoder backward → MaxSim *is* the
  step. 7–23× (see top table).
2. `**Ld ≥ 8k` MaxSim on smaller encoders** (ColPali, ColBERTv2 at max
  length) — encoder shrinks enough that MaxSim becomes the dominant op.
3. `**ModernColBERT` inference at long docs** — encoding is one-time,
  reranking is per-query; MaxSim is hit every query.
4. **Offline knowledge-distillation scoring** — teacher + student both do
  MaxSim, no per-step encoder cost to hide behind.
5. **When MaxSim OOM is the limit** (`Ld ≥ 8k`, very large `Nd`) — there
  naive doesn't run at all (see the MaxSim-only table above).

For plain ModernColBERT fine-tuning at current shapes, the kernel is
essentially free: same step time, same VRAM, same numerics (within ULP),
one `patch_pylate()` call.

## FastPlaid rerank step (LightOn's Rust/libtorch engine)

[`lightonai/fast-plaid`](https://github.com/lightonai/fast-plaid) is the
Rust multi-vector search engine replacing the original PLAID. Its final
exact-rerank step at
[`rust/search/search.rs:colbert_score_reduce`](https://github.com/lightonai/fast-plaid/blob/main/rust/search/search.rs)
is the exact `matmul → masked_fill(-9999) → max_dim → sum_dim` pattern
late-interaction-kernels fuses:

```rust
let token_scores = padded_doc_embeddings.matmul(&query.transpose(-2, -1));
let masked       = token_scores.masked_fill(&padding_mask, -9999.0);
let (per_doc, _) = masked.max_dim(1, false);
per_doc.sum_dim_intlist(-1, false, Kind::Float)
```

FastPlaid is faster than Python PLAID because it avoids Python overhead
and uses a `PAD_CACHE` scratch buffer — but the *underlying CUDA kernels*
(`aten::bmm`, `aten::max`, `aten::sum`) are the same ones PyLate
dispatches from Python, so the kernel-level speedup carries over 1:1.

At FastPlaid's default `n_full_scores=4096`, only `n_full_scores / 4 =
1024` docs get the exact MaxSim per query. **Isolated rerank-step
MaxSim at those shapes** (1024 docs × `Ld` × `Lq=32` × `d=128`, fp16):

| shape                  | libtorch (proxy) | late-interaction-kernels | speedup   | peak libtorch | peak flash  | mem×     |
| ---------------------- | ---------------- | ------------- | --------- | ------------- | ----------- | -------- |
| `Ld=200`  (MSMARCO)    | 0.121 ms         | 0.137 ms      | 0.89×     | 0.11 GB       | 0.08 GB     | 1.4×     |
| `Ld=512`  (BEIR-ish)   | 0.212 ms         | 0.137 ms      | **1.55×** | 0.23 GB       | 0.16 GB     | 1.5×     |
| `Ld=1024`              | 0.374 ms         | 0.139 ms      | **2.69×** | 0.94 GB       | 0.28 GB     | **3.3×** |
| `Ld=4096` (ModernCol.) | 1.304 ms         | 0.368 ms      | **3.55×** | 1.97 GB       | 1.04 GB     | 1.9×     |
| `Ld=8192` (ModernCol.) | 2.537 ms         | 0.713 ms      | **3.56×** | 3.35 GB       | 2.05 GB     | 1.6×     |
| `Ld=4096, n_docs=256`  | 0.402 ms         | 0.138 ms      | **2.92×** | 0.89 GB       | 0.28 GB     | 3.1×     |
| `Ld=4096, n_docs=2048` | 2.506 ms         | 0.712 ms      | **3.52×** | 3.41 GB       | 2.05 GB     | 1.7×     |

**End-to-end `fast_plaid.search()`** on synthetic 10 k-doc corpora,
100 queries, default `top_k=10, n_full_scores=4096, n_ivf_probe=8`:

| corpus            | total / 100 q | ms / query | isolated rerank MaxSim / step |
| ----------------- | ------------- | ---------- | ----------------------------- |
| 10 k × `Ld=200`   | 5.3 s         | 53.3 ms    | 0.12 ms (**0.2 %**)           |
| 10 k × `Ld=512`   | 10.2 s        | 101.6 ms   | 0.21 ms (**0.2 %**)           |
| 10 k × `Ld=1024`  | 23.2 s        | 232.2 ms   | 0.37 ms (**0.2 %**)           |

**Honest verdict for a FastPlaid integration:**

- At today's defaults (`n_full_scores=4096`, typical `Ld ≤ 1024`) the
  exact-rerank MaxSim is **< 1 % of `search()`** — IVF probing,
  approximate scoring, and residual decompression dominate.
  late-interaction-kernels would be **neutral** end-to-end even though the kernel
  swap itself is 2.7–3.6× faster.
- Where it *would* matter:
  - **ModernColBERT-scale corpora** (`Ld ≥ 4k`): rerank-step MaxSim grows
    linearly in `Ld`; at `Ld=8k` it's already 2.5 ms / query. A 3.6×
    cut saves ~1.8 ms / query — still < 1 % of total, but the **memory
    drop** (3.3 GB → 2.0 GB scratch) lets you push `n_full_scores`
    without OOM.
  - **Large `n_full_scores` for high-recall rerank** — naive scratch
    grows as `n_full_scores · Ld · Lq · 4` bytes. At
    `n_full_scores=16384, Ld=8192` that's **16 GB fp32**, which OOMs on
    an H100; late-interaction-kernels needs < 100 MB.
- **Integration cost:** non-trivial — FastPlaid's hot path is Rust /
  `tch-rs`, so you'd have to either call Python (`py.import` round-trip
  kills the latency win) or ship a CUDA-side rewrite of the Triton
  kernel. Not worth the engineering effort unless ModernColBERT-scale
  search becomes the primary use-case.

**Recommendation:** for FastPlaid's current workload,
late-interaction-kernels is an *optional* optimization that matters for long-doc
rerank; we don't propose upstreaming it. For PyLate / model-training
workloads and pure inference / reranking in Python, late-interaction-kernels's
gains are larger and immediately available via `patch_pylate()`.

## Understanding the numbers

The forward speedup depends on the ratio `Lq · Ld / d_pad` — how much of the work
is the tensor-core GEMM vs the reduction around it:

- **GEMM-bound** (`Lq · Ld ≫ d_pad`, e.g. ColPali): late-interaction-kernels ≈ peak H100 GEMM;
naive ≈ peak HBM bandwidth. Speedup comes almost entirely from *not moving* the
`[Nq · Nd · Lq · Ld]` fp32 values through HBM. Expected: **5–15×**.
- **Reduction-bound** (`Lq · Ld ≲ d_pad`, tiny docs): both implementations are
bandwidth-bound; speedup comes from skipping the `S`-materialization trip. Expected:
**1.5–3×**.

Training shapes (`Nq = Nd = 32..128`) sit in between and end-to-end land at **2–5×**.

## Memory

Naive allocates `Nq · Nd · Lq · Ld · 4 bytes` fp32 scratch. late-interaction-kernels only
keeps the `Nq · Nd` result plus — if autograd is on — a `Nq · Nd · Lq` int32
argmax buffer.


| scenario                          | naive scratch | flash fwd | flash fwd + argmax |
| --------------------------------- | ------------- | --------- | ------------------ |
| `Nq=1, Nd=1000, Lq=32, Ld=300`    | 183 MB        | 4 KB      | 128 KB             |
| `Nq=128, Nd=128, Lq=32, Ld=300`   | 623 MB        | 64 KB     | 2 MB               |
| `Nq=1, Nd=1000, Lq=1024, Ld=1024` | 4.5 GB        | 4 KB      | 4 MB               |
| `Nq=16, Nd=32, Lq=32, Ld=8192`    | 2.1 GB        | 64 KB     | 64 KB              |


This is what unlocks large-batch training: at the same HBM budget you can run
~5–10× more in-batch negatives than PyLate's vanilla path.