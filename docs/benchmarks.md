# Benchmarks

Single H100 80 GB SXM, bf16 inputs (fp16 for LateOn / ModernColBERT
shapes), fp32 accumulator throughout, 50 iterations after 5 warmup,
`torch 2.8` (NGC 25.06), `triton 3.x`, CUDA 12.9.

**Fair-comparison protocol.** Every speedup on this page is measured at
**matched numerics**: each baseline runs the inner einsum / matmul with
an fp32 accumulator just like the fused kernel, and parity vs the eager
reference is asserted at `atol=1e-2, rtol=1e-2` *before* timing. The
forward and cached-contrastive sections also report a `torch.compile`
column on the same body so you can see what Inductor alone closes.

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
pip install "flash-maxsim==0.2.0"   # pinned: matches the published numbers
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
sky launch -c lik-bench-all scripts/sky_run_all_benchmarks.yaml -y   # every table on this page in one shot (1×H100, ~25 min)
sky launch -c lik-bench scripts/sky_bench_verify.yaml -y   # forward + cached + fused head + fp8 (the README numbers)
sky jobs launch scripts/sky_test.yaml             # CI-style: tests + bench_forward + bench_backward
sky jobs launch scripts/sky_lateon_edge.yaml      # LateOn-Code-edge end-to-end
sky jobs launch scripts/sky_decompress_bench.yaml # PLAID decompress + MaxSim
sky jobs launch scripts/sky_fastplaid_e2e.yaml    # vs `fast_plaid.engine.search()`
```

## Forward (reranking / inference)

**Baseline.** Every implementation on this page runs the einsum with an
**fp32 accumulator** (same numerical contract as the fused kernel) and
reads `bf16` / `fp16` inputs. Parity against the eager reference is
asserted at `atol=1e-2, rtol=1e-2` before timing, so the speedup ratios
are apples-to-apples in precision. The reference is one line:

```python
def eager_fp32(Q, D):
    S = torch.einsum("ild,jtd->ijlt", Q.float(), D.float())
    return S.max(-1).values.sum(-1)
```

We also report a `torch.compile(dynamic=False, mode="reduce-overhead")`
column on the *same* body. Inductor fuses the surrounding ops but still
has to materialise the `[Nq · Nd · Lq · Ld]` similarity tile in HBM
before the `max(-1)` reduction — that materialisation is what the fused
Triton kernel exists to skip. Empirically `torch.compile` lands within
±10% of eager on every shape; on the small `train-batch (Nq=Nd=32)` case
it's actually slower than eager because the per-call dispatch overhead
dominates such tiny work.

Full table (H100 80 GB SXM, NGC 25.06, bf16 inputs, 50-iter median over
CUDA events):


| shape                                                   | LIK      | eager (fp32 acc) | `torch.compile` (fp32 acc) | LIK vs eager | LIK vs compile | naive scratch |
| ------------------------------------------------------- | -------- | ---------------- | -------------------------- | ------------ | -------------- | ------------- |
| text-short `Nq=1, Nd=1k, Lq=32, Ld=300`                 | 0.104 ms | 0.261 ms         | 0.272 ms                   | 2.5×         | 2.6×           | 183 MB → 0    |
| text-long `Nq=1, Nd=1k, Lq=32, Ld=1024`                 | 0.121 ms | 0.791 ms         | 0.878 ms                   | **6.5×**     | **7.3×**       | 626 MB → 0    |
| text-medium `Nq=1, Nd=1k, Lq=128, Ld=1024`              | 0.107 ms | 1.538 ms         | 1.602 ms                   | **14.4×**    | **15.0×**      | 1.0 GB → 0    |
| visual `Nq=1, Nd=1k, Lq=1024, Ld=1024` (ColPali)        | 0.731 ms | 9.356 ms         | 9.172 ms                   | **12.8×**    | **12.5×**      | 4.5 GB → 0    |
| corpus-5k `Nq=1, Nd=5k, Lq=32, Ld=300`                  | 0.128 ms | 1.189 ms         | 1.190 ms                   | 9.3×         | 9.3×           | 916 MB → 0    |
| corpus-10k `Nq=1, Nd=10k, Lq=32, Ld=300`                | 0.247 ms | 2.349 ms         | 2.351 ms                   | 9.5×         | 9.5×           | 1.8 GB → 0    |
| train-batch `Nq=Nd=32, Lq=32, Ld=300`                   | 0.103 ms | 0.125 ms         | 0.169 ms                   | 1.2×         | 1.6×           | 43 MB → 0     |
| train-batch-128 `Nq=Nd=128, Lq=32, Ld=300`              | 0.226 ms | 1.549 ms         | 1.539 ms                   | **6.9×**     | **6.8×**       | 621 MB → 0    |
| large-d-512 `Nq=1, Nd=1k, Lq=32, Ld=300, d=512`         | 0.106 ms | 0.829 ms         | 0.830 ms                   | 7.8×         | 7.8×           | 623 MB → 0    |
| large-d-1024 `Nq=1, Nd=500, Lq=32, Ld=300, d=1024`      | 0.112 ms | 0.820 ms         | 0.821 ms                   | 7.3×         | 7.3×           | 604 MB → 0    |
| lateon-code-edge-rerank `Nd=1k, Ld=2048, d=48`          | 0.099 ms | 0.766 ms         | 0.767 ms                   | **7.7×**     | **7.7×**       | 626 MB → 0    |
| lateon-code-edge-big `Nd=4k, Ld=2048, d=48`             | 0.277 ms | 2.957 ms         | 2.959 ms                   | **10.7×**    | **10.7×**      | 2.5 GB → 0    |
| mxbai-edge-rerank `Nd=1k, Ld=300, d=64`                 | 0.102 ms | 0.167 ms         | 0.169 ms                   | 1.6×         | 1.7×           | 110 MB → 0    |
| mxbai-edge-corpus-10k `Nd=10k, Ld=300, d=64`            | 0.130 ms | 1.397 ms         | 1.397 ms                   | **10.7×**    | **10.7×**      | 1.1 GB → 0    |


On wide shapes (`Lq · Ld` large) LIK beats both baselines by 7-15×; the
HBM round-trip on the similarity tile dominates. On tiny shapes the
kernel-launch + autotune overhead caps the win at ~1.5-2.5×.


### vs `flash-maxsim` (same Triton MaxSim math)

`flash-maxsim` (Roi Pony / IBM) was the first public Triton MaxSim
kernel and the direct inspiration for this library. The numbers below
come from `benchmarks/bench_flash_maxsim.py` on H100 80 GB SXM, bf16,
50-iter median over CUDA events. `flash-maxsim` 0.2.0 has no
`normalize=True` knob and no autograd-aware backward, so we report
plain forward only.


| shape                                             | ours  | flash-maxsim | speedup |
| ------------------------------------------------- | ----- | ------------ | ------- |
| `rerank-short` (Nq=1, Nd=1k, Lq=32, Ld=300)       | 0.136 | 0.135        | 0.99×   |
| `rerank-long` (Nq=1, Nd=1k, Lq=32, Ld=1024)       | 0.191 | 0.194        | 1.02×   |
| `rerank-very-long` (Nq=1, Nd=500, Lq=32, Ld=4096) | 0.253 | 0.299        | 1.18×   |
| `rerank-colpali` (Nq=1, Nd=500, Lq=1024, Ld=1024) | 0.473 | 0.571        | 1.21×   |
| `rerank-10k` (Nq=1, Nd=10k, Lq=32, Ld=300)        | 0.349 | 0.354        | 1.01×   |
| `train-in-batch-32` (Nq=Nd=32, Lq=32, Ld=200)     | 0.129 | 0.132        | 1.02×   |
| `train-in-batch-128` (Nq=Nd=128, Lq=32, Ld=200)   | 0.297 | 0.322        | 1.08×   |
| `train-long-doc` (Nq=Nd=16, Lq=32, Ld=2048)       | 0.114 | 0.111        | 0.97×   |
| `edge-d48` (Nq=1, Nd=4k, Lq=32, Ld=2048, d=48)    | 0.359 | 0.368        | 1.02×   |
| `edge-d64` (Nq=1, Nd=10k, Lq=32, Ld=300, d=64)    | 0.227 | 0.230        | 1.01×   |


The two kernels are within ±3% on tight rerank / short-context shapes
and LIK pulls ahead by 1.08–1.21× on the wide ones (`rerank-very-long`,
`rerank-colpali`, `train-in-batch-128`); on `rerank-short` and
`train-long-doc` both are saturated by L2-normalized HBM traffic and
LIK loses by a fraction of a percent. The real differentiators are
elsewhere: a fused `normalize=True` (no extra HBM round-trip), a real
autograd-aware backward (`unified` / `csr` / `atomic`), packed/varlen,
PLAID residual decompression, and a fused D-side projection head —
none of which `flash-maxsim` ships. `bench_flash_maxsim.py` also
covers KD-layout (`Q[B, Lq, d] × D[B, K, Ld, d] → [B, K]`) and pairwise
(`Q[B, Lq, d] × D[B, Ld, d] → [B]`) forwards where LIK is +3 to +14%.

## Fused L2-normalize

`F.normalize(Q) → maxsim` becomes a single kernel (`normalize=True`).
The fused path normalizes in SRAM, eliminating two HBM round-trips, and
the backward correctly applies the L2-norm Jacobian.


| shape                                  | `F.normalize` + maxsim | fused    | speedup   |
| -------------------------------------- | ---------------------- | -------- | --------- |
| text-short (`Nq=1, Nd=1k, Ld=300`)     | 0.464 ms               | 0.109 ms | 4.3×      |
| text-long (`Nq=1, Nd=1k, Ld=1024`)     | 1.467 ms               | 0.122 ms | **12.0×** |
| bigbatch-300 (`Nq=32, Nd=32, Ld=300`)  | 0.288 ms               | 0.106 ms | 2.7×      |
| bigbatch-2k (`Nq=8, Nd=16, Ld=2048`)   | 0.234 ms               | 0.080 ms | 2.9×      |
| bigbatch-8k (`Nq=8, Nd=16, Ld=8192`)   | 0.289 ms               | 0.127 ms | 2.3×      |
| corpus-10k (`Nq=1, Nd=10k, Ld=300`)    | 4.197 ms               | 0.283 ms | **14.8×** |


## PLAID / ColBERTv2

End-to-end vs `fast_plaid.engine.search()`: build the index with
fast-plaid, time `engine.search()`, then load the same compressed
tensors and call `maxsim_residual_varlen` *on the same compressed
bytes*. To keep the comparison apples-to-apples both pipelines end on
a `torch.topk(scores, k=10)` — `engine.search()` returns the top-10,
so the LIK timer now folds the same final argmax in. Reproduce with
`scripts/sky_fastplaid_e2e.yaml`.

We report two LIK variants:

* **`lik_full + top-k`** — score the *whole* corpus, then top-k. Upper
  bound on the rerank cost; corresponds to "no IVF probe".
* **`lik_partial + top-k`** (4 096 cands) — score the same number of
  candidates fast-plaid keeps after its IVF probe (`n_full_scores=4096`),
  then top-k. Closest apples-to-apples vs `engine.search()` because
  it pays the same rerank workload, minus IVF probing — which
  fast-plaid currently does in Rust and we don't.


| corpus shape (nbits)    | `engine.search()` | `lik_full + top-k` | `lik_partial + top-k` | full speedup | partial speedup |
| ----------------------- | ----------------- | ------------------ | --------------------- | ------------ | --------------- |
| 5 000 docs × 200, nb=2  | 22.95 ms          | 1.48 ms            | 1.25 ms               | **15.5×**    | **18.3×**       |
| 10 000 docs × 300, nb=2 | 45.69 ms          | 3.82 ms            | 1.74 ms               | **12.0×**    | **26.3×**       |
| 10 000 docs × 512, nb=2 | 77.15 ms          | 5.59 ms            | 2.50 ms               | **13.8×**    | **30.9×**       |
| 10 000 docs × 512, nb=4 | 129.59 ms         | 6.02 ms            | 2.75 ms               | **21.5×**    | **47.1×**       |
| 25 000 docs × 300, nb=2 | 73.36 ms          | 9.35 ms            | 1.72 ms               | **7.8×**     | **42.6×**       |


Reading: even with the top-k argmax folded into the LIK side, the
fused kernel is **8-22×** faster than the full fast-plaid pipeline on
"full" (scoring every doc) and **18-47×** on "partial" (matching
fast-plaid's post-IVF rerank workload). At 25k docs the `lik_full`
gap closes because we now do every doc's residual decompress while
fast-plaid's IVF probe scales sub-linearly; the partial column tells
the more honest matched-workload story. Against a PyTorch
transliteration of fast-plaid's exact decompress → pad → matmul →
reduce slice, the fused varlen kernel is **3.0-4.0× faster at 10-34×
less GPU memory** — no `[Ntop, max_Ld, packed_dim]` padded scratch is
allocated.

## Fused D-side head (training)

Replaces `F.linear → F.normalize → maxsim` for the
hidden-state → embedding → MaxSim path. Forward saves an argmax;
backward gathers `H_d` only at winning positions and is closed-form.
Both columns use the same LIK MaxSim — the only difference is whether
the projection / normalize step is folded in:


| shape                                                  | unfused (`F.linear+normalize` then LIK MaxSim) | fused    | speedup   |
| ------------------------------------------------------ | ---------------------------------------------- | -------- | --------- |
| LateOn `Nd=16, Lq=32, Ld=300, d_model=768`             |  0.95 ms                                       |  1.02 ms | 0.93×     |
| LateOn `Nd=32, Lq=32, Ld=1024`                         |  1.21 ms                                       |  1.05 ms | 1.15×     |
| LateOn-Code `Nd=128, Ld=1024`                          |  1.50 ms                                       |  1.01 ms | 1.49×     |
| LateOn-Code `Nd=256, Ld=1024`                          |  2.27 ms                                       |  1.21 ms | **1.88×** |
| LateOn-Code `Nd=512, Ld=2048`                          |  7.40 ms                                       |  1.74 ms | **4.27×** |
| LateOn-Code `Nd=1024, Ld=2048`                         | 14.09 ms                                       |  3.14 ms | **4.49×** |
| LateOn-Code-edge `Nd=256, Ld=4096, d_model=384, d=96`  |  5.24 ms                                       |  1.31 ms | **4.01×** |


The win grows with `Nd · Ld` (more positions for the fused linear to
batch over) and shrinks as `Nq · Lq / Ld → 1`. Memory and derivation
in [`design.md`](design.md).

## FP8 inference (Hopper)

This row compares **two LIK kernels against each other**, not LIK vs
naive PyTorch. `maxsim_inference_fp8` runs the inner matmul in `e4m3`
(fp8) with an fp32 accumulator and `bf16` output; `maxsim` (bf16) is
the same Triton kernel one precision step up. No requantization of
`Q` / `D` is needed if they were stored fp8 to begin with (PyLate's
PLAID index can ship fp8 directly), so the speedup below isolates the
\"swap bf16 tensor cores for fp8 tensor cores\" win.

H100 80 GB SXM, NGC 25.06, `bench_fp8.py`:


| shape                                       | bf16     | fp8      | speedup   | label              |
| ------------------------------------------- | -------- | -------- | --------- | ------------------ |
| `Nd=1k, Lq=32, Ld=128, d=128`               | 0.125 ms | 0.130 ms | 0.96×     | pylate-rerank-1k   |
| `Nd=4k, Lq=32, Ld=128, d=128`               | 0.152 ms | 0.158 ms | 0.96×     | pylate-rerank-4k   |
| `Nd=8k, Lq=32, Ld=256, d=128`               | 0.279 ms | 0.237 ms | 1.18×     | colbert-rerank-8k  |
| `Nd=4k, Lq=32, Ld=256, d=128`               | 0.448 ms | 0.348 ms | **1.29×** | batched-rerank-16k |
| `Nd=2k, Lq=32, Ld=512, d=128`               | 0.198 ms | 0.176 ms | 1.12×     | long-docs-2k       |
| `Nd=1k, Lq=32, Ld=128, d=96`                | 0.123 ms | 0.139 ms | 0.89×     | lateon-edge-1k     |
| `Nd=4k, Lq=32, Ld=128, d=96`                | 0.149 ms | 0.152 ms | 0.98×     | lateon-edge-4k     |


On rerank-sized shapes the bf16 kernel got fast enough in 0.3.0 that
short-context fp8 (`Ld=128`) is now within ±5% of bf16 or slightly
behind — kernel-launch and the e4m3 packing pay back only once
`Lq · Ld` is large enough to dominate. Useful regime is now
`Ld ≥ 256` and corpus ≥ 4k, where fp8 lands 1.12-1.29×. Tolerance vs
bf16 stays inside 5e-3 absolute on every shape we tested — fine for
top-k retrieval where ordering, not exact scores, is what matters.

## End-to-end LateOn-Code-edge training (17 M encoder)

`losses.Contrastive`, in-batch negatives, bf16, grad-checkpointing,
1×H100. Reproduce with `scripts/sky_lateon_edge.yaml`.


| setup                  | vanilla  | fused    | speedup   | peak (v → f)   |
| ---------------------- | -------- | -------- | --------- | -------------- |
| bs=256, Lq=32, Ld=256  | 115.1 ms |  90.3 ms | **1.27×** | 11.5 → 9.6 GB  |
| bs=192, Lq=32, Ld=512  | 153.1 ms | 126.2 ms | **1.21×** | 16.6 → 14.7 GB |
| bs=128, Lq=32, Ld=1024 | 229.9 ms | 205.5 ms | **1.12×** | 22.6 → 21.4 GB |
| bs=64, Lq=32, Ld=2048  | 324.3 ms | 313.6 ms | 1.03×     | 25.8 GB        |


Smaller encoder + bigger effective batch ⇒ bigger MaxSim slice ⇒ bigger
end-to-end speedup. These numbers use `patch_pylate()`;
`maxsim_from_hidden` (autograd path) adds another ~1–3 ms / step but
isn't wired into PyLate's loss path because PyLate's `Dense` projection
runs inside the encoder forward.

## Backward paths

0.3.0 ships a new `unified` path that combines CSR's deterministic
reduction with atomic's parallelism; `auto` picks `unified` on almost
every shape and falls back to `csr` for very-high-contention training
batches (e.g. `Nd ≥ 256` with `Ld = 128`).


| shape                            | atomic | csr  | unified | auto | auto picks |
| -------------------------------- | ------ | ---- | ------- | ---- | ---------- |
| `train-32` (32 × 32, Ld=128)     | 0.62   | 0.77 | 0.44    | 0.45 | unified    |
| `train-128` (128 × 128, Ld=128)  | 0.60   | 0.77 | 0.54    | 0.55 | unified    |
| `train-256` (256 × 256, Ld=128)  | 1.80   | 1.18 | 1.63    | 1.18 | **csr**    |
| `retrieval` (16 × 512, Ld=300)   | 0.53   | 0.71 | 0.77    | 0.77 | unified    |
| `long-Lq` (Lq=1024, Ld=64)       | 0.86   | 0.79 | 0.45    | 0.45 | unified    |
| `huge-Nd` (16 × 1024, Ld=128)    | 0.81   | 0.79 | 1.25    | 1.25 | unified    |


CSR is bitwise-reproducible across runs (no atomics); `atomic` /
`unified` drift within fp32 ULP (~1e-6 relative) because reduction
order depends on thread scheduling. Note: on a few shapes the
hand-picked `atomic` path beats `auto`'s `unified` choice
(`retrieval`, `huge-Nd`); the heuristic favours the more general
`unified` path because it generalises across batch sizes and document
lengths without an autotune step.

```python
# Per-call (recommended):
maxsim(Q, D, normalize=True, backward="csr")   # | "atomic" | "unified" | "auto"
```

## End-to-end PyLate `Contrastive` training


| batch × negs | vanilla PyLate | fused   | speedup   |
| ------------ | -------------- | ------- | --------- |
| 64 × 1       |  1.18 ms       | 1.14 ms | 1.04×     |
| 128 × 2      |  4.75 ms       | 1.85 ms | **2.56×** |
| 256 × 3      | 24.31 ms       | 5.88 ms | **4.13×** |


## LateOn / ModernColBERT (long documents)

At 2k-4k the naive einsum still fits, so you see speedup *and* memory
ratios. At 8k+ naive OOMs on 80 GB at sane training batch sizes —
that's the story this section tells. Numbers apply equally to
`lightonai/LateOn`, `lightonai/GTE-ModernColBERT-v1` and
`lightonai/LateOn-Code` (same backbone, `d=128`).

MaxSim only (one `colbert_scores` call, fp16 inputs + fp32
accumulator, `auto` backward). Parity vs the fp32-acc naive reference
is asserted before timing on every shape that fits in HBM:


| shape                                     | fwd LIK | fwd naive | bwd LIK | bwd naive | peak LIK | peak naive |
| ----------------------------------------- | ------- | --------- | ------- | --------- | -------- | ---------- |
| `Nq=8, Nd=16, Lq=32, Ld=2048` train-2k    | 0.09 ms | 0.15 ms   | 0.56 ms | 0.57 ms   |  96 MB   | 152 MB     |
| `Nq=8, Nd=16, Lq=32, Ld=4096` train-4k    | 0.11 ms | 0.21 ms   | 0.45 ms | 0.59 ms   | 128 MB   | 240 MB     |
| `Nq=16,Nd=32, Lq=32, Ld=4096` bigbatch-4k | **0.13**| 0.65      |**0.52** | 1.82      |**193 MB**|**672 MB**  |
| `Nq=1, Nd=64, Lq=32, Ld=4096` rerank-4k   | 0.10 ms | 0.24 ms   | 0.55 ms | 0.77 ms   | 320 MB   | 416 MB     |
| `Nq=8, Nd=16, Lq=32, Ld=8192` train-8k    | 0.12 ms | **OOM**   | 0.51 ms | **OOM**   | 192 MB   | OOM        |
| `Nq=16,Nd=32, Lq=32, Ld=8192` bigbatch-8k | 0.18 ms | **OOM**   | 0.56 ms | **OOM**   | 321 MB   | OOM        |
| `Nq=1, Nd=256,Lq=32, Ld=8192` rerank-8k   | 0.18 ms | **OOM**   | 1.55 ms | **OOM**   |  2.1 GB  | OOM        |
| `Nq=1, Nd=32, Lq=32, Ld=16384` huge-doc   | 0.13 ms | **OOM**   | 1.05 ms | **OOM**   | 576 MB   | OOM        |


At `Ld ≥ 8k` naive OOMs on the `[Nq, Nd, Lq, Ld]` similarity scratch
(`8 · 16 · 32 · 8192 · 4 bytes ≈ 128 MB *per query position*` — and
`Nq` queries multiply that). The fused kernel never writes that tensor
out, so HBM stays flat with `Ld` and the same shapes that OOM in the
naive column run in ~0.2 ms here. The `bigbatch-4k` column shows what
this buys you when both paths fit: ~5× fwd and ~3.5× bwd wall-clock
advantage *and* ~3.5× less HBM, which is what lets `bs ≥ 16` at
`Ld = 4k` survive at all.


### LightOn cached-contrastive (MaxSim isolation)

`pylate.losses.CachedContrastive` chunks MaxSim into `(bs / mini)**2`
Python-level calls. The fused kernel collapses that double loop into
one call that never materializes `S`. We compare three implementations
on the same chunked tiling (`Lq=128, d=128, mini=32`, fwd + bwd):

* **vanilla** — `pylate.scores.colbert_scores` per tile (current PyLate
  default, fp32 accumulator).
* **`torch.compile`** — local re-implementation of the `colbert_scores`
  body wrapped in `torch.compile(dynamic=False, mode="reduce-overhead")`,
  same numerics as vanilla.
* **LIK** — `late_interaction_kernels.maxsim` over the same chunking.

H100 80 GB SXM, NGC 25.06, parity checked on an 8-row probe before
timing. `torch.compile` lands *below* vanilla here because Dynamo
recompiles on every fresh tile shape and the cuda-graph fast path
trips on the pending-backward state pylate's loss leaves behind:


| shape                          | tiles | vanilla fwd+bwd | `torch.compile` fwd+bwd | LIK fwd+bwd | LIK vs vanilla | LIK vs compile |
| ------------------------------ | ----- | --------------- | ----------------------- | ----------- | -------------- | -------------- |
| `bs=64, Ld=2048`               |   4   |   7.64 ms       |  25.84 ms               |   1.52 ms   | **5.04×**      | **17.03×**     |
| `bs=64, Ld=4096`               |   4   |  14.69 ms       |  51.52 ms               |   2.66 ms   | **5.52×**      | **19.36×**     |
| `bs=64, Ld=8192`               |   4   |  31.73 ms       | 101.97 ms               |   4.84 ms   | **6.55×**      | **21.06×**     |
| `bs=128, Ld=2048`              |  16   |  31.33 ms       | 104.14 ms               |   6.02 ms   | **5.20×**      | **17.30×**     |
| `bs=128, Ld=4096`              |  16   |  60.25 ms       | 207.90 ms               |  12.01 ms   | **5.02×**      | **17.31×**     |
| `bs=128, Ld=8192`              |  16   | 130.91 ms       | 412.87 ms               |  22.17 ms   | **5.91×**      | **18.62×**     |
| `bs=256, Ld=2048`              |  64   | 131.00 ms       | 425.52 ms               |  27.98 ms   | **4.68×**      | **15.21×**     |
| `bs=256, Ld=4096`              |  64   | 254.03 ms       | 844.74 ms               |  49.99 ms   | **5.08×**      | **16.90×**     |
| **`bs=256, Ld=8192`** (real recipe) | 64 | **546.51 ms** | **1683.95 ms**       | **93.55 ms**| **5.84×**      | **18.00×**     |


LIK is a steady **5-6.5×** over vanilla and **15-21×** over the
compiled tile across the whole range, with the win growing with `Ld`.
The vs-`torch.compile` gap widened in 0.3.0: compile now recompiles
every fresh tile shape *and* trips the cuda-graph fast path on the
pending-backward state pylate's loss leaves behind, so the compiled
column lands well above vanilla. The vs-vanilla numbers are tighter
than 0.1.0's headline 13.8× because that figure was measured against
an older PyLate that ran `colbert_scores` in plain Python; the loss
path got faster, the kernel got a fair fight, and we still win by ~5×.


### End-to-end on `LateOn-Code-edge` (17 M, real MS MARCO triplets)

Real `pylate.models.ColBERT("lightonai/LateOn-Code-edge")` (`d=48`,
2 047-token context), AdamW + bf16 autocast, MS MARCO `triplet` split
loaded through `datasets`. We swap `patch_pylate()` on/off and time
full optimizer steps (encoder forward + loss + backward + step). The
kernel is a drop-in — no other code changes between the two columns.
Reproduce with `scripts/sky_pylate_realdata.yaml` and
`benchmarks/bench_pylate_realdata.py`.


| recipe              | setup                              | vanilla PyLate | + LIK    | speedup   |
| ------------------- | ---------------------------------- | -------------- | -------- | --------- |
| `Contrastive`       | bs=16, Lq=32, Ld=256               |  64.2 ms       |  61.6 ms | 1.04×     |
| `CachedContrastive` | bs=64, mini=16, Ld=300, grad-ckpt  | 351.1 ms       | 305.5 ms | **1.15×** |
| `CachedContrastive` | bs=128, mini=16, Ld=512, grad-ckpt | 716.4 ms       | 680.0 ms | 1.05×     |


Reading: on a 17 M encoder where the transformer forward isn't yet
swallowing the whole step, LIK moves the wall-clock by **~15 %** at
`bs=64, Ld=300` where the MaxSim chunked-loss slice still matters.
At `bs=16, Ld=256` the encoder forward is the entire step and LIK
buys ~4 %; at `bs=128, Ld=512` the encoder + grad-checkpointing
dominates again. Same bottleneck story as the LateOn 149 M numbers
from v0.1.0 (`bench_pylate_lateon.py`), just shifted up the batch
axis because the encoder is 9× smaller.

## End-to-end ColQwen2 / ColPali training

Real `colpali_engine.models.ColQwen2("vidore/colqwen2-v1.0")` (Qwen2-VL
2 B backbone + LoRA, `d=128`, multi-vector image / query embeddings),
LoRA-only training (37 M of 2 246 M params trainable — the official
ColPali recipe via `peft.get_peft_model`), AdamW + bf16 weights. We
swap `patch_colpali_engine()` on/off and time full optimizer steps
(vision tower + text encoder forward + ColbertLoss + backward + step).
Reproduce with `scripts/sky_colpali_training.yaml` and
`benchmarks/bench_colpali_training.py` (synthetic) /
`bench_colpali_realdata.py` (real `vidore/docvqa_test_subsampled`).


| loss head           | setup                              | vanilla colpali_engine | + LIK     | speedup   | peak     |
| ------------------- | ---------------------------------- | ---------------------- | --------- | --------- | -------- |
| `ColbertLoss`       | synth bs=4, 448px                  |  386.9 ms              |  370.4 ms | **1.04×** |  9.10 GB |
| `ColbertLoss`       | synth bs=8, 448px, grad-ckpt       |  895.4 ms              |  882.4 ms | 1.01×     |  5.74 GB |
| `ColbertPairwiseCE` | synth bs=4, 448px                  |  366.9 ms              |  363.8 ms | 1.01×     |  9.10 GB |
| `ColbertLoss`       | synth bs=16, 1024px                |  831.8 ms              |  831.7 ms | 1.00×     | 49.61 GB |
| `ColbertLoss`       | synth bs=16, 1024px, grad-ckpt     | 2099.5 ms              | 2093.5 ms | 1.00×     | 11.54 GB |
| `ColbertLoss`       | real DocVQA bs=4                   |  770.7 ms              |  723.6 ms | **1.07×** | 16.20 GB |
| `ColbertLoss`       | real DocVQA bs=8, grad-ckpt        | 1941.3 ms              | 1865.7 ms | 1.04×     |  8.05 GB |


Reading: ColPali's Qwen2-VL-2B backbone has a much heavier
forward+backward than a ModernBERT-149 M ColBERT, and the image
modality blows up Ld (≈1 030 visual tokens at the default 448 px
resolution; ≈3 060 at 1 024 px). So even with LoRA-only training
shrinking AdamW state by ~60×, the *encoder activation-grad
backward* is what dominates the step — LIK lands in the 1.00–1.07×
range here. Best win is the 448 px / bs=4 real-data step (**1.07×**)
where the encoder share is smallest; at 1 024 px the encoder fully
swallows the step. The kernel is a drop-in — no other code changes
between the two columns. Same takeaway as the PyLate `Contrastive`
recipe on the 149 M encoder: when the transformer is the bottleneck,
LIK doesn't move the needle, it just doesn't hurt.

## Edge models (`d ∈ {48, 64}`)

Edge ColBERT models (`d ∈ {48, 64}`) are more memory-bound, so the
fused kernel widens its lead. `bench_inference_edge.py`, bf16, 50-iter:


| shape                                       | fused    | naive (fp32) | speedup   | fused mem | naive mem |
| ------------------------------------------- | -------- | ------------ | --------- | --------- | --------- |
| LateOn-Code-edge `Nd=1 000, Ld=1 024, d=48` | 0.114 ms | 0.398 ms     | **3.5×**  | 0.0 MB    | 314 MB    |
| LateOn-Code-edge `Nd=1 000, Ld=4 096, d=48` | 0.135 ms | 1.492 ms     | **11.0×** | 0.0 MB    | 1.2 GB    |
| LateOn-Code-edge `Nd=1 000, Ld=8 192, d=48` | 0.262 ms | 3.053 ms     | **11.6×** | 0.0 MB    | 2.5 GB    |
| LateOn-Code-edge `Nd=16 000, Ld=512, d=48`  | 0.253 ms | 3.037 ms     | **12.0×** | 0.1 MB    | 2.5 GB    |
| mxbai-edge `Nd=1 000, Ld=4 096, d=64`       | 0.266 ms | 1.755 ms     | **6.6×**  | 0.0 MB    | 1.5 GB    |
| mxbai-edge `Nd=16 000, Ld=512, d=64`        | 0.332 ms | 3.565 ms     | **10.7×** | 0.1 MB    | 3.0 GB    |


## Where this kernel actually moves the e2e needle

1. **Inference / reranking** — no encoder backward → MaxSim *is* the
  step. **1.5–15×** at matched numerics, biggest wins on wide
  `Lq · Ld` shapes.
2. **Small-encoder training** — encoder small enough that MaxSim is
  material; LateOn-Code-edge moves **1.04–1.15×** end-to-end on real
  MS MARCO triplets.
3. **Long-context regimes** (`Ld ≥ 8k`) — fused kernels run, naive
  doesn't.
4. **Compressed indices** — PLAID rerank vs `engine.search()` is
  **8–22×** on full corpus scoring and **18–47×** on the partial
  rerank workload that matches fast-plaid's IVF probe.
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
  (`late_interaction_kernels.mps.metal.maxsim_inference_metal`, JIT-compiled
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