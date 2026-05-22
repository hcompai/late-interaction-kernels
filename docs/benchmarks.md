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


| shape                                              | LIK      | eager (fp32 acc) | `torch.compile` (fp32 acc) | LIK vs eager | LIK vs compile | naive scratch |
| -------------------------------------------------- | -------- | ---------------- | -------------------------- | ------------ | -------------- | ------------- |
| text-short `Nq=1, Nd=1k, Lq=32, Ld=300`            | 0.091 ms | 0.266 ms         | 0.246 ms                   | 2.9×         | 2.7×           | 183 MB → 0    |
| text-long `Nq=1, Nd=1k, Lq=32, Ld=1024`            | 0.093 ms | 0.793 ms         | 0.779 ms                   | **8.5×**     | **8.4×**       | 626 MB → 0    |
| text-medium `Nq=1, Nd=1k, Lq=128, Ld=1024`         | 0.103 ms | 1.102 ms         | 1.071 ms                   | **10.7×**    | **10.4×**      | 1.0 GB → 0    |
| visual `Nq=1, Nd=1k, Lq=1024, Ld=1024` (ColPali)   | 0.949 ms | 4.046 ms         | 3.737 ms                   | 4.3×         | 3.9×           | 4.5 GB → 0    |
| corpus-5k `Nq=1, Nd=5k, Lq=32, Ld=300`             | 0.132 ms | 1.182 ms         | 1.183 ms                   | 9.0×         | 9.0×           | 916 MB → 0    |
| corpus-10k `Nq=1, Nd=10k, Lq=32, Ld=300`           | 0.250 ms | 2.331 ms         | 2.332 ms                   | 9.3×         | 9.3×           | 1.8 GB → 0    |
| train-batch `Nq=Nd=32, Lq=32, Ld=300`              | 0.085 ms | 0.137 ms         | 0.193 ms                   | 1.6×         | 2.3×           | 43 MB → 0     |
| train-batch-128 `Nq=Nd=128, Lq=32, Ld=300`         | 0.317 ms | 0.705 ms         | 0.713 ms                   | 2.2×         | 2.2×           | 621 MB → 0    |
| large-d-512 `Nq=1, Nd=1k, Lq=32, Ld=300, d=512`    | 0.152 ms | 0.845 ms         | 0.847 ms                   | 5.6×         | 5.6×           | 623 MB → 0    |
| large-d-1024 `Nq=1, Nd=500, Lq=32, Ld=300, d=1024` | 0.166 ms | 0.835 ms         | 0.837 ms                   | 5.0×         | 5.0×           | 604 MB → 0    |
| lateon-edge-rerank `Nd=1k, Ld=2048, d=48`          | 0.087 ms | 0.716 ms         | 0.716 ms                   | **8.2×**     | **8.2×**       | 625 MB → 0    |
| lateon-edge-big `Nd=4k, Ld=2048, d=48`             | 0.261 ms | 2.804 ms         | 2.798 ms                   | **10.7×**    | **10.7×**      | 2.5 GB → 0    |
| mxbai-edge `Nd=1k, Ld=300, d=64`                   | 0.086 ms | 0.175 ms         | 0.173 ms                   | 2.0×         | 2.0×           | 110 MB → 0    |
| mxbai-edge-corpus-10k `Nd=10k, Ld=300, d=64`       | 0.159 ms | 1.369 ms         | 1.370 ms                   | **8.6×**     | **8.6×**       | 1.1 GB → 0    |


On wide shapes (`Lq · Ld` large) LIK beats both baselines by 8-11×; the
HBM round-trip on the similarity tile dominates. On tiny shapes the
kernel-launch + autotune overhead caps the win at ~2×.


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


| corpus shape (nbits=2) | `engine.search()` | `lik_full + top-k` | `lik_partial + top-k` | full speedup | partial speedup |
| ---------------------- | ----------------- | ------------------ | --------------------- | ------------ | --------------- |
| 5 000 docs × 200 tok   | 23.24 ms          | 1.48 ms            | 1.25 ms               | **15.7×**    | **18.6×**       |
| 10 000 docs × 300 tok  | 46.56 ms          | 3.75 ms            | 1.71 ms               | **12.4×**    | **27.3×**       |
| 10 000 docs × 512 tok  | 79.62 ms          | 5.56 ms            | 2.49 ms               | **14.3×**    | **32.0×**       |


Reading: even with the top-k argmax folded into the LIK side, the
fused kernel is **15-32×** faster than the full fast-plaid pipeline.
"Partial" is the closer head-to-head (matches fast-plaid's rerank
workload after IVF probing); "full" shows what scoring the entire
corpus from scratch costs us — usually still cheaper than the IVF
probe path in fast-plaid below 10k docs. Against a PyTorch
transliteration of fast-plaid's exact decompress → pad → matmul →
reduce slice, the fused varlen kernel is **3.4-3.7× faster at 10-34×
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
| LateOn `Nd=16, Lq=32, Ld=300, d_model=768`             |  1.16 ms                                       |  0.88 ms | 1.32×     |
| LateOn `Nd=32, Lq=32, Ld=1024`                         |  1.09 ms                                       |  0.91 ms | 1.20×     |
| LateOn-Code `Nd=128, Ld=1024`                          |  1.61 ms                                       |  1.00 ms | 1.61×     |
| LateOn-Code `Nd=256, Ld=1024`                          |  2.48 ms                                       |  1.09 ms | **2.29×** |
| LateOn-Code `Nd=512, Ld=2048`                          |  7.37 ms                                       |  1.80 ms | **4.10×** |
| LateOn-Code `Nd=1024, Ld=2048`                         | 13.95 ms                                       |  3.32 ms | **4.21×** |
| LateOn-Code-edge `Nd=256, Ld=4096, d_model=384, d=96`  |  5.25 ms                                       |  1.35 ms | **3.89×** |


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
| `Nd=1k, Lq=32, Ld=128, d=128`               | 0.300 ms | 0.123 ms | **2.45×** | pylate-rerank-1k   |
| `Nd=4k, Lq=32, Ld=128, d=128`               | 0.336 ms | 0.150 ms | **2.24×** | pylate-rerank-4k   |
| `Nd=8k, Lq=32, Ld=256, d=128`               | 0.459 ms | 0.227 ms | 2.03×     | colbert-rerank-8k  |
| `Nd=4k, Lq=32, Ld=256, d=128`               | 0.621 ms | 0.336 ms | 1.85×     | batched-rerank-16k |
| `Nd=2k, Lq=32, Ld=512, d=128`               | 0.367 ms | 0.162 ms | **2.26×** | long-docs-2k       |
| `Nd=1k, Lq=32, Ld=128, d=96`                | 0.294 ms | 0.118 ms | **2.48×** | lateon-edge-1k     |
| `Nd=4k, Lq=32, Ld=128, d=96`                | 0.332 ms | 0.143 ms | **2.33×** | lateon-edge-4k     |


FP8 gives a flat ~2× across rerank-sized shapes. Tolerance vs bf16
stays inside 5e-3 absolute on every shape we tested — fine for top-k
retrieval where ordering, not exact scores, is what matters.

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
`maxsim_from_hidden` (autograd path) adds another ~1–3 ms / step but
isn't wired into PyLate's loss path because PyLate's `Dense` projection
runs inside the encoder forward.

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
# Per-call (recommended):
maxsim(Q, D, normalize=True, backward="csr")   # | "atomic" | "unified" | "auto"
```

## End-to-end PyLate `Contrastive` training


| batch × negs | vanilla PyLate | fused    | speedup |
| ------------ | -------------- | -------- | ------- |
| 64 × 1       | 2.70 ms        | 0.92 ms  | 2.93×   |
| 128 × 2      | 14.95 ms       | 4.86 ms  | 3.08×   |
| 256 × 3      | 75.04 ms       | 26.28 ms | 2.86×   |


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
| `Nq=8, Nd=16, Lq=32, Ld=2048` train-2k    | 0.10 ms | 0.14 ms   | 0.48 ms | 0.53 ms   |  96 MB   | 152 MB     |
| `Nq=8, Nd=16, Lq=32, Ld=4096` train-4k    | 0.10 ms | 0.21 ms   | 0.47 ms | 0.58 ms   | 128 MB   | 240 MB     |
| `Nq=16,Nd=32, Lq=32, Ld=4096` bigbatch-4k | **0.10**| 0.65      |**0.41** | 1.82      |**193 MB**|**672 MB**  |
| `Nq=1, Nd=64, Lq=32, Ld=4096` rerank-4k   | 0.10 ms | 0.24 ms   | 0.51 ms | 0.77 ms   | 320 MB   | 416 MB     |
| `Nq=8, Nd=16, Lq=32, Ld=8192` train-8k    | 0.10 ms | **OOM**   | 0.49 ms | **OOM**   | 192 MB   | OOM        |
| `Nq=16,Nd=32, Lq=32, Ld=8192` bigbatch-8k | 0.18 ms | **OOM**   | 0.56 ms | **OOM**   | 321 MB   | OOM        |
| `Nq=1, Nd=256,Lq=32, Ld=8192` rerank-8k   | 0.18 ms | **OOM**   | 1.55 ms | **OOM**   |  2.1 GB  | OOM        |
| `Nq=1, Nd=32, Lq=32, Ld=16384` huge-doc   | 0.13 ms | **OOM**   | 1.04 ms | **OOM**   | 576 MB   | OOM        |


At `Ld ≥ 8k` naive OOMs on the `[Nq, Nd, Lq, Ld]` similarity scratch
(`8 · 16 · 32 · 8192 · 4 bytes ≈ 128 MB *per query position*` — and
`Nq` queries multiply that). The fused kernel never writes that tensor
out, so HBM stays flat with `Ld` and the same shapes that OOM in the
naive column run in ~0.2 ms here. The `bigbatch-4k` column shows what
this buys you when both paths fit: same wall-clock advantage (~6×
fwd, ~4× bwd) *and* ~3.5× less HBM, which is what lets `bs ≥ 16` at
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
| `bs=64, Ld=2048`               |   4   |   7.64 ms       |  12.58 ms               |   1.89 ms   | **4.05×**      | **6.66×**      |
| `bs=64, Ld=4096`               |   4   |  14.67 ms       |  24.75 ms               |   3.18 ms   | **4.61×**      | **7.77×**      |
| `bs=64, Ld=8192`               |   4   |  31.64 ms       |  48.68 ms               |   5.70 ms   | **5.55×**      | **8.55×**      |
| `bs=128, Ld=2048`              |  16   |  31.31 ms       |  51.07 ms               |   6.51 ms   | **4.81×**      | **7.84×**      |
| `bs=128, Ld=4096`              |  16   |  60.14 ms       | 100.48 ms               |  13.10 ms   | **4.59×**      | **7.67×**      |
| `bs=128, Ld=8192`              |  16   | 129.83 ms       | 197.71 ms               |  24.57 ms   | **5.28×**      | **8.05×**      |
| `bs=256, Ld=2048`              |  64   | 130.86 ms       | 210.02 ms               |  27.70 ms   | **4.72×**      | **7.58×**      |
| `bs=256, Ld=4096`              |  64   | 252.18 ms       | 413.16 ms               |  51.75 ms   | **4.87×**      | **7.98×**      |
| **`bs=256, Ld=8192`** (real recipe) | 64 | **542.15 ms** | **813.40 ms**        | **99.94 ms**| **5.43×**      | **8.14×**      |


LIK is a steady **4-5.5×** over vanilla and **6.7-8.5×** over the
compiled tile across the whole range, with the win growing slightly
with `Ld`. The current numbers are tighter than 0.1.0's headline 13.8×
because that figure was measured against an older PyLate that ran
`colbert_scores` in plain Python; the loss path got faster, the kernel
got a fair fight, and we still win by ~5×.


### End-to-end on `LateOn-Code-edge` (17 M, real MS MARCO triplets)

Real `pylate.models.ColBERT("lightonai/LateOn-Code-edge")` (`d=48`,
2 047-token context), AdamW + bf16 autocast, MS MARCO `triplet` split
loaded through `datasets`. We swap `patch_pylate()` on/off and time
full optimizer steps (encoder forward + loss + backward + step). The
kernel is a drop-in — no other code changes between the two columns.
Reproduce with `scripts/sky_pylate_realdata.yaml` and
`benchmarks/bench_pylate_realdata.py`.


| recipe              | setup                            | vanilla PyLate | + LIK    | speedup   |
| ------------------- | -------------------------------- | -------------- | -------- | --------- |
| `Contrastive`       | bs=16, Lq=32, Ld=256             |  66.3 ms       |  52.3 ms | **1.27×** |
| `CachedContrastive` | bs=64, mini=16, Ld=300, grad-ckpt | 315.2 ms      | 263.7 ms | **1.20×** |
| `CachedContrastive` | bs=128, mini=16, Ld=512, grad-ckpt | 573.0 ms     | 544.9 ms | 1.05×     |


Reading: on a 17 M encoder where the transformer forward isn't yet
swallowing the whole step, LIK moves the wall-clock by **~20–27 %**
when the MaxSim slice is still material (small-to-medium batch). At
`bs=128, Ld=512` the encoder starts dominating and the e2e gain falls
to ~5 % — same bottleneck story as the LateOn 149 M numbers from
v0.1.0 (`bench_pylate_lateon.py`), just shifted up the batch axis
because the encoder is 9× smaller.

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
  step. **2–11×** at matched numerics.
2. **Small-encoder training** — encoder small enough that MaxSim is
  material; LateOn-Code-edge moves **1.05–1.27×** end-to-end on real
  MS MARCO triplets.
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