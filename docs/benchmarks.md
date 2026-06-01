# Benchmarks

## Configuration

Every number on this page comes from this stack unless a table says otherwise:

| | |
| ----------------- | --------------------------------------------------------- |
| GPU               | 1× H100 80 GB SXM                                         |
| Base image        | NGC 25.06                                                |
| CUDA              | 12.8                                                     |
| PyTorch           | `2.8.0+cu128`                                            |
| Triton            | `3.4`                                                    |
| Inputs            | bf16 (fp16 for LateOn / ModernColBERT shapes)            |
| Accumulator       | fp32 throughout                                          |
| Timing            | 50-iter median over CUDA events, after 5 warmup          |

### Baseline package versions

All baselines are pinned PyPI releases (no git checkouts), installed into the
stack above by the SkyPilot jobs in `scripts/` (see
[`../benchmarks/README.md`](../benchmarks/README.md)).

| package          | version       | role on this page                              |
| ---------------- | ------------- | ---------------------------------------------- |
| `flash-maxsim`   | `0.2.1`       | forward head-to-head and the chunking "vs flash" column |
| `pylate`         | `1.3.3`       | vanilla-PyLate end-to-end baselines (1.5+ broke `patch_pylate`'s `(queries_mask, documents_mask)` signature, so the sweep pins 1.3.3) |
| `fast-plaid`     | `1.4.6.280`   | PLAID rerank vs `engine.search()` (the `.280` suffix is the torch-2.8 wheel) |
| `colpali-engine` | `>=0.3.10`    | ColQwen2 / ColPali end-to-end (`ColbertLoss`, `ColbertPairwiseCE`) |

The ColQwen2 / ColPali section is the one exception to the stack above: its
SkyPilot job pins `torch==2.11.0+cu128` / `torchvision==0.26.0+cu128` to satisfy
`colpali-engine`'s dependency floor, so treat its absolute step times as
same-row vanilla-vs-LIK ratios rather than cross-comparable with the other
tables.

## Fair-comparison protocol

Every speedup on this page is measured at **matched numerics**: each baseline
runs the inner einsum / matmul with an fp32 accumulator just like the fused
kernel, and parity vs the eager reference is asserted at `atol=1e-2, rtol=1e-2`
*before* timing. The forward and cached-contrastive sections also report a
`torch.compile` column on the same body so you can see what Inductor alone
closes.

## Reproducing

Running individual scripts, the full local sweep, the SkyPilot cluster
drivers, and the output format all live in one place:
[`../benchmarks/README.md`](../benchmarks/README.md). This page is the
headline numbers and the analysis behind them.

## Forward (reranking / inference)

The eager reference each shape is checked against is one line:

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

Full table:


| shape                                                   | LIK      | eager (fp32 acc) | `torch.compile` (fp32 acc) | LIK vs eager | LIK vs compile | naive scratch |
| ------------------------------------------------------- | -------- | ---------------- | -------------------------- | ------------ | -------------- | ------------- |
| text-short `Nq=1, Nd=1k, Lq=32, Ld=300`                 | 0.094 ms | 0.263 ms         | 0.273 ms                   | 2.8×         | 2.9×           | 183 MB → 0    |
| text-long `Nq=1, Nd=1k, Lq=32, Ld=1024`                 | 0.096 ms | 0.793 ms         | 0.880 ms                   | **8.3×**     | **9.2×**       | 626 MB → 0    |
| text-medium `Nq=1, Nd=1k, Lq=128, Ld=1024`              | 0.097 ms | 1.538 ms         | 1.603 ms                   | **15.9×**    | **16.5×**      | 1.0 GB → 0    |
| visual `Nq=1, Nd=1k, Lq=1024, Ld=1024` (ColPali)        | 0.675 ms | 9.365 ms         | 9.173 ms                   | **13.9×**    | **13.6×**      | 4.5 GB → 0    |
| corpus-5k `Nq=1, Nd=5k, Lq=32, Ld=300`                  | 0.129 ms | 1.195 ms         | 1.196 ms                   | 9.3×         | 9.3×           | 916 MB → 0    |
| corpus-10k `Nq=1, Nd=10k, Lq=32, Ld=300`                | 0.247 ms | 2.354 ms         | 2.352 ms                   | 9.5×         | 9.5×           | 1.8 GB → 0    |
| train-batch `Nq=Nd=32, Lq=32, Ld=300`                   | 0.097 ms | 0.127 ms         | 0.169 ms                   | 1.3×         | 1.7×           | 43 MB → 0     |
| train-batch-128 `Nq=Nd=128, Lq=32, Ld=300`              | 0.226 ms | 1.539 ms         | 1.540 ms                   | **6.8×**     | **6.8×**       | 621 MB → 0    |
| large-d-512 `Nq=1, Nd=1k, Lq=32, Ld=300, d=512`         | 0.108 ms | 0.837 ms         | 0.837 ms                   | 7.8×         | 7.8×           | 623 MB → 0    |
| large-d-1024 `Nq=1, Nd=500, Lq=32, Ld=300, d=1024`      | 0.112 ms | 0.821 ms         | 0.821 ms                   | 7.3×         | 7.3×           | 604 MB → 0    |
| lateon-code-edge-rerank `Nd=1k, Ld=2048, d=48`          | 0.095 ms | 0.767 ms         | 0.768 ms                   | **8.1×**     | **8.1×**       | 626 MB → 0    |
| lateon-code-edge-big `Nd=4k, Ld=2048, d=48`             | 0.260 ms | 2.961 ms         | 2.961 ms                   | **11.4×**    | **11.4×**      | 2.5 GB → 0    |
| mxbai-edge-rerank `Nd=1k, Ld=300, d=64`                 | 0.096 ms | 0.168 ms         | 0.166 ms                   | 1.7×         | 1.7×           | 110 MB → 0    |
| mxbai-edge-corpus-10k `Nd=10k, Ld=300, d=64`            | 0.130 ms | 1.398 ms         | 1.399 ms                   | **10.8×**    | **10.8×**      | 1.1 GB → 0    |


On wide shapes (`Lq · Ld` large) LIK beats both baselines by 7-16×; the
HBM round-trip on the similarity tile dominates. On tiny shapes the
kernel-launch + autotune overhead caps the win at ~1.3-2.9×.

The `naive scratch` column shows naive peak → LIK peak; LIK never
materialises the `[Nq · Nd · Lq · Ld]` similarity tile, so its peak
working set is bounded by the output and ranges from a few KB to
~64 KB across the table — effectively zero next to the naive scratch.


### vs `flash-maxsim` (same Triton MaxSim math)

`flash-maxsim` (Roi Pony / IBM) was the first public Triton MaxSim
kernel and the direct inspiration for this library. The numbers below
come from `bench_flash_maxsim.py`. `flash-maxsim` 0.2.1 has no
`normalize=True` knob and no autograd-aware backward, so we report
plain forward only.


| shape                                             | ours  | flash-maxsim | speedup |
| ------------------------------------------------- | ----- | ------------ | ------- |
| `rerank-short` (Nq=1, Nd=1k, Lq=32, Ld=300)       | 0.106 | 0.111        | 1.05×   |
| `rerank-long` (Nq=1, Nd=1k, Lq=32, Ld=1024)       | 0.167 | 0.173        | 1.04×   |
| `rerank-very-long` (Nq=1, Nd=500, Lq=32, Ld=4096) | 0.227 | 0.255        | 1.12×   |
| `rerank-colpali` (Nq=1, Nd=500, Lq=1024, Ld=1024) | 0.434 | 0.510        | 1.17×   |
| `rerank-10k` (Nq=1, Nd=10k, Lq=32, Ld=300)        | 0.322 | 0.329        | 1.02×   |
| `train-in-batch-32` (Nq=Nd=32, Lq=32, Ld=200)     | 0.098 | 0.102        | 1.05×   |
| `train-in-batch-128` (Nq=Nd=128, Lq=32, Ld=200)   | 0.263 | 0.298        | 1.14×   |
| `train-long-doc` (Nq=Nd=16, Lq=32, Ld=2048)       | 0.089 | 0.092        | 1.04×   |
| `edge-d48` (Nq=1, Nd=4k, Lq=32, Ld=2048, d=48)    | 0.340 | 0.351        | 1.03×   |
| `edge-d64` (Nq=1, Nd=10k, Lq=32, Ld=300, d=64)    | 0.220 | 0.220        | 1.00×   |


LIK meets or beats `flash-maxsim` on every cross-product forward shape:
a dead heat on the tightest ones (`edge-d64`, `rerank-10k`) and 1.12–1.17×
ahead on the wide regimes (`rerank-very-long`, `rerank-colpali`,
`train-in-batch-128`). Both kernels are fused (neither materialises the
score tile), so peak working set is sub-100 KB on every shape and there's
no memory column to compare. The real differentiators are elsewhere: a
fused `normalize=True` (no extra HBM round-trip), a real autograd-aware
backward (`unified` / `csr` / `atomic`), packed/varlen, PLAID residual
decompression, and a fused D-side projection head — none of which
`flash-maxsim` ships. `bench_flash_maxsim.py` also covers KD-layout
(`Q[B, Lq, d] × D[B, K, Ld, d] → [B, K]`, LIK +4 to +17%) and pairwise
(`Q[B, Lq, d] × D[B, Ld, d] → [B]`, LIK within a few percent except the
smallest pair shape, where flash's lighter launch wins).

### Long-query chunking (`Lq > 512`)

For long queries (ColPali-scale `Lq ≈ 1k` visual patches) the un-chunked
kernel serialises a long `static_range` loop inside each program. Since
MaxSim is a sum over query tokens of a per-token max, splitting the query
axis into fixed 128-token chunks and summing the per-chunk scores is
numerically exact, and launches more, shorter programs that keep the H100
busy. It fires only above a measured crossover (`Lq > 512`); shorter
queries fall through to the existing core unchanged. From
`bench_chunking.py`:

| shape                                  | un-chunked | chunked  | speedup  | vs flash |
| -------------------------------------- | ---------- | -------- | -------- | -------- |
| in-batch `Nq=Nd=16, Lq=768, Ld=512`    | 0.260 ms   | 0.147 ms | **1.77×** | 1.38×   |
| in-batch `Nq=Nd=64, Lq=768, Ld=512`    | 1.358 ms   | 0.912 ms | **1.49×** | 1.52×   |
| in-batch `Nq=Nd=32, Lq=1024, Ld=1024`  | 0.731 ms   | 0.588 ms | **1.24×** | 1.53×   |
| tail `Nq=Nd=16, Lq=1030, Ld=1024`      | 0.472 ms   | 0.327 ms | **1.44×** | 1.24×   |
| rerank `Nq=1, Nd=500, Lq=1024, Ld=1024`| 0.436 ms   | 0.440 ms | 0.99×    | 1.18×   |
| boundary `Nq=Nd=128, Lq=512` (≤ cutoff)| 2.278 ms   | 2.288 ms | 1.00×    | 1.49×   |

The split is implemented as a thin dispatch around the shared forward core
(no kernel change) and autograd flows through the reshape + sum, so training
is covered too — neutral-to-better there (large batches `+19–32%`, small
batches sit in the launch-bound noise floor). It also collapses the per-`Lq`
autotune sweep onto a small bounded number of cache entries. Chunking is
cross-product-only: the KD/pairs path (4-D `D`) already saturates its
`Nq·K` grid, so it stays un-chunked.

## Fused L2-normalize

`F.normalize(Q) → maxsim` becomes a single kernel (`normalize=True`).
The fused path normalizes in SRAM, eliminating two HBM round-trips, and
the backward correctly applies the L2-norm Jacobian.


| shape                                  | `F.normalize` + maxsim | fused    | speedup   | explicit peak | fused peak | mem ratio |
| -------------------------------------- | ---------------------- | -------- | --------- | ------------- | ---------- | --------- |
| text-short (`Nq=1, Nd=1k, Ld=300`)     | 0.673 ms               | 0.100 ms | 6.7×      |   368 MB      |   74 MB    | **5.0×**  |
| text-long (`Nq=1, Nd=1k, Ld=1024`)     | 1.623 ms               | 0.099 ms | **16.4×** |  1254 MB      |  250 MB    | **5.0×**  |
| bigbatch-300 (`Nq=32, Nd=32, Ld=300`)  | 0.441 ms               | 0.098 ms | 4.5×      |    12 MB      |    3 MB    | **4.7×**  |
| bigbatch-2k (`Nq=8, Nd=16, Ld=2048`)   | 0.421 ms               | 0.075 ms | 5.6×      |    40 MB      |    8 MB    | **5.0×**  |
| bigbatch-8k (`Nq=8, Nd=16, Ld=8192`)   | 0.590 ms               | 0.131 ms | 4.5×      |   161 MB      |   32 MB    | **5.0×**  |
| corpus-10k (`Nq=1, Nd=10k, Ld=300`)    | 4.357 ms               | 0.284 ms | **15.3×** |  3674 MB      |  733 MB    | **5.0×**  |

The explicit path writes a normalized copy of `D` to HBM and re-reads
it during the matmul; the fused path streams the same data through
threadgroup memory and never writes the intermediate, so peak working
set drops by a steady ~5× across the whole sweep.


## PLAID / ColBERTv2

End-to-end vs `fast_plaid.engine.search()`: build the index with
fast-plaid, time `engine.search()`, then load the same compressed
tensors and call `maxsim_residual_varlen` *on the same compressed
bytes*. To keep the comparison apples-to-apples both pipelines end on
a `torch.topk(scores, k=10)` — `engine.search()` returns the top-10,
so the LIK timer now folds the same final argmax in.

We report two LIK variants:

* **`lik_full + top-k`** — score the *whole* corpus, then top-k. Upper
  bound on the rerank cost; corresponds to "no IVF probe".
* **`lik_partial + top-k`** (4 096 cands) — score the same number of
  candidates fast-plaid keeps after its IVF probe (`n_full_scores=4096`),
  then top-k. Closest apples-to-apples vs `engine.search()` because
  it pays the same rerank workload, minus IVF probing — which
  fast-plaid currently does in Rust and we don't.


| corpus shape (nbits)    | `engine.search()` | `lik_full + top-k` | `lik_partial + top-k` | full speedup | partial speedup | lik_full peak | lik_partial peak |
| ----------------------- | ----------------- | ------------------ | --------------------- | ------------ | --------------- | ------------- | ---------------- |
| 5 000 docs × 200, nb=2  | 23.50 ms          | 1.48 ms            | 1.28 ms               | **15.9×**    | **18.3×**       |   72 MB       |   95 MB          |
| 10 000 docs × 300, nb=2 | 47.27 ms          | 3.78 ms            | 1.73 ms               | **12.5×**    | **27.3×**       |  145 MB       |  180 MB          |
| 10 000 docs × 512, nb=2 | 79.73 ms          | 5.61 ms            | 2.49 ms               | **14.2×**    | **32.0×**       |  217 MB       |  277 MB          |
| 10 000 docs × 512, nb=4 | 138.43 ms         | 6.04 ms            | 2.71 ms               | **22.9×**    | **51.0×**       |  334 MB       |  442 MB          |
| 25 000 docs × 300, nb=2 | 77.24 ms          | 9.25 ms            | 1.71 ms               | **8.3×**     | **45.1×**       |  306 MB       |  341 MB          |


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


| shape                                                  | unfused (`F.linear+normalize` then LIK MaxSim) | fused    | speedup   | unfused peak | fused peak |
| ------------------------------------------------------ | ---------------------------------------------- | -------- | --------- | ------------ | ---------- |
| LateOn `Nd=16, Lq=32, Ld=300, d_model=768`             |  0.93 ms                                       |  0.98 ms | 0.95×     |    84 MB     |   86 MB    |
| LateOn `Nd=32, Lq=32, Ld=1024`                         |  0.93 ms                                       |  0.99 ms | 0.94×     |   201 MB     |  189 MB    |
| LateOn-Code `Nd=128, Ld=1024`                          |  1.47 ms                                       |  1.04 ms | 1.42×     |   608 MB     |  505 MB    |
| LateOn-Code `Nd=256, Ld=1024`                          |  2.34 ms                                       |  1.09 ms | **2.15×** |  1028 MB     |  889 MB    |
| LateOn-Code `Nd=512, Ld=2048`                          |  7.31 ms                                       |  1.71 ms | **4.28×** |  3524 MB     | 3193 MB    |
| LateOn-Code `Nd=1024, Ld=2048`                         | 14.08 ms                                       |  3.15 ms | **4.47×** |  6852 MB     | 6321 MB    |
| LateOn-Code-edge `Nd=256, Ld=4096, d_model=384, d=96`  |  5.23 ms                                       |  1.33 ms | **3.93×** |  1988 MB     | 1631 MB    |


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

From `bench_fp8.py`:


| shape                                       | bf16     | fp8      | speedup   | label              |
| ------------------------------------------- | -------- | -------- | --------- | ------------------ |
| `Nd=1k, Lq=32, Ld=128, d=128`               | 0.124 ms | 0.133 ms | 0.94×     | pylate-rerank-1k   |
| `Nd=4k, Lq=32, Ld=128, d=128`               | 0.154 ms | 0.158 ms | 0.98×     | pylate-rerank-4k   |
| `Nd=8k, Lq=32, Ld=256, d=128`               | 0.278 ms | 0.234 ms | 1.19×     | colbert-rerank-8k  |
| `Nd=4k, Lq=32, Ld=256, d=128`               | 0.442 ms | 0.346 ms | **1.28×** | batched-rerank-16k |
| `Nd=2k, Lq=32, Ld=512, d=128`               | 0.194 ms | 0.173 ms | 1.13×     | long-docs-2k       |
| `Nd=1k, Lq=32, Ld=128, d=96`                | 0.123 ms | 0.137 ms | 0.90×     | lateon-edge-1k     |
| `Nd=4k, Lq=32, Ld=128, d=96`                | 0.148 ms | 0.155 ms | 0.96×     | lateon-edge-4k     |


On rerank-sized shapes the bf16 kernel got fast enough in 0.3.0 that
short-context fp8 (`Ld=128`) is now within ±5% of bf16 or slightly
behind — kernel-launch and the e4m3 packing pay back only once
`Lq · Ld` is large enough to dominate. Useful regime is now
`Ld ≥ 256` and corpus ≥ 4k, where fp8 lands 1.12-1.29×. Tolerance vs
bf16 stays inside 5e-3 absolute on every shape we tested — fine for
top-k retrieval where ordering, not exact scores, is what matters.

Peak working set is identical between the two columns (same Q/D
tensors live in HBM either way; both kernels are fused so neither
allocates score-tile scratch). The memory win of fp8 is upstream of
this bench — if the index is *stored* fp8 (PyLate's PLAID index can
ship that way), `D` itself is 2× smaller in HBM and the corpus
fits in half the RAM.

## End-to-end LateOn-Code-edge training (17 M encoder)

`losses.Contrastive`, in-batch negatives, bf16, grad-checkpointing,
1×H100.


| setup                  | vanilla  | fused    | speedup   | peak (v → f)   |
| ---------------------- | -------- | -------- | --------- | -------------- |
| bs=256, Lq=32, Ld=256  | 116.2 ms |  90.8 ms | **1.28×** | 11.5 → 9.6 GB  |
| bs=192, Lq=32, Ld=512  | 152.2 ms | 127.9 ms | **1.19×** | 16.6 → 14.7 GB |
| bs=128, Lq=32, Ld=1024 | 231.7 ms | 208.3 ms | **1.11×** | 22.6 → 21.4 GB |
| bs=64, Lq=32, Ld=2048  | 322.5 ms | 318.5 ms | 1.01×     | 25.8 GB        |


Smaller encoder + bigger effective batch ⇒ bigger MaxSim slice ⇒ bigger
end-to-end speedup. These numbers use `patch_pylate()`;
`maxsim_from_hidden` (autograd path) adds another ~1–3 ms / step but
isn't wired into PyLate's loss path because PyLate's `Dense` projection
runs inside the encoder forward.

## Backward paths

Two paths matter. `unified` is the fastest single-pass backward: it scatters
`grad_D` with fp32 atomics, accumulating in a full-size fp32 buffer that is
cast to the input dtype at the end. `lowmem` is the memory-optimal one: each
output row is destination-owned, so it accumulates in fp32 *registers* and
writes `grad_Q` / `grad_D` straight in the input dtype — no full-size fp32
buffer, no fp32→bf16 transient, no atomics (hence deterministic). `atomic`
(two-pass) and `csr` (sorted reduction) remain available for back-compat.

`auto` sends the gradient-heavy shapes — knowledge-distillation / hard-negative
layouts (where `grad_D` is `n_neg`-inflated) and large high-contention in-batch
squares — to `lowmem`, and everything else to `unified`. The split is exact:
`lowmem`'s `grad_D` is a one-hot matmul whose wasted FLOPs scale with `Lq`, so
it ties or beats `unified` for short queries but loses on long-query×long-doc
cross-products, which stay on `unified`.

Realistic shapes (H100, bf16, fwd+bwd peak memory / step time):

| shape                          | `unified`        | `lowmem`         |
| ------------------------------ | ---------------- | ---------------- |
| pylate-text B256 Lq32 Ld300    | 96 MB / 2.23 ms  | **52 MB / 2.10 ms** |
| colpali B128 Lq32 Ld1030       | 141 MB / 1.38 ms | **72 MB** / 1.57 ms |
| colpali-neg B128×n8            | 1086 MB / 0.86 ms| **543 MB / 0.71 ms** |
| colpali-neg B256×n16           | 4329 MB / 2.36 ms| **2165 MB / 1.37 ms** |
| longq B64 Lq1030 Ld1030        | **179 MB / 5.6 ms** | 107 MB / 9.0 ms |

So on the hard-negative path `lowmem` roughly halves peak memory *and* runs
~1.7× faster; on long-query cross-products `unified` keeps the speed edge.
`lowmem` and `csr` are deterministic across runs; `atomic` / `unified` drift
within fp32 ULP (reduction order depends on scheduling).

```python
# Per-call override (auto is recommended):
maxsim(Q, D, normalize=True, backward="lowmem")  # | "unified" | "auto" | "atomic" | "csr"
```

## End-to-end PyLate `Contrastive` training


| batch × negs | vanilla PyLate | fused   | speedup   |
| ------------ | -------------- | ------- | --------- |
| 64 × 1       |  1.36 ms       | 1.16 ms | 1.17×     |
| 128 × 2      |  4.75 ms       | 1.85 ms | **2.57×** |
| 256 × 3      | 24.28 ms       | 5.91 ms | **4.11×** |


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
| `Nq=8, Nd=16, Lq=32, Ld=2048` train-2k    | 0.08 ms | 0.15 ms   | 0.65 ms | 0.64 ms   |  96 MB   | 152 MB     |
| `Nq=8, Nd=16, Lq=32, Ld=4096` train-4k    | 0.09 ms | 0.21 ms   | 0.61 ms | 0.64 ms   | 128 MB   | 240 MB     |
| `Nq=16,Nd=32, Lq=32, Ld=4096` bigbatch-4k | **0.12**| 0.65      |**0.54** | 1.82      |**193 MB**|**672 MB**  |
| `Nq=1, Nd=64, Lq=32, Ld=4096` rerank-4k   | 0.09 ms | 0.24 ms   | 0.53 ms | 0.78 ms   | 320 MB   | 416 MB     |
| `Nq=8, Nd=16, Lq=32, Ld=8192` train-8k    | 0.12 ms | **OOM**   | 0.43 ms | **OOM**   | 192 MB   | OOM        |
| `Nq=16,Nd=32, Lq=32, Ld=8192` bigbatch-8k | 0.18 ms | **OOM**   | 0.44 ms | **OOM**   | 321 MB   | OOM        |
| `Nq=1, Nd=256,Lq=32, Ld=8192` rerank-8k   | 0.18 ms | **OOM**   | 1.36 ms | **OOM**   |  2.1 GB  | OOM        |
| `Nq=1, Nd=32, Lq=32, Ld=16384` huge-doc   | 0.13 ms | **OOM**   | 0.55 ms | **OOM**   | 576 MB   | OOM        |


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

Parity is checked on an 8-row probe before
timing. `torch.compile` lands *below* vanilla here because Dynamo
recompiles on every fresh tile shape and the cuda-graph fast path
trips on the pending-backward state pylate's loss leaves behind:


| shape                          | tiles | vanilla fwd+bwd | `torch.compile` fwd+bwd | LIK fwd+bwd | LIK vs vanilla | LIK vs compile |
| ------------------------------ | ----- | --------------- | ----------------------- | ----------- | -------------- | -------------- |
| `bs=64, Ld=2048`               |   4   |   7.64 ms       |  25.99 ms               |   1.27 ms   | **6.02×**      | **20.46×**     |
| `bs=64, Ld=4096`               |   4   |  14.67 ms       |  51.55 ms               |   2.43 ms   | **6.04×**      | **21.21×**     |
| `bs=64, Ld=8192`               |   4   |  31.72 ms       | 101.89 ms               |   4.59 ms   | **6.91×**      | **22.20×**     |
| `bs=128, Ld=2048`              |  16   |  31.32 ms       | 104.63 ms               |   5.17 ms   | **6.06×**      | **20.24×**     |
| `bs=128, Ld=4096`              |  16   |  60.15 ms       | 207.57 ms               |  11.04 ms   | **5.45×**      | **18.80×**     |
| `bs=128, Ld=8192`              |  16   | 129.85 ms       | 410.47 ms               |  21.18 ms   | **6.13×**      | **19.38×**     |
| `bs=256, Ld=2048`              |  64   | 130.86 ms       | 424.27 ms               |  25.87 ms   | **5.06×**      | **16.40×**     |
| `bs=256, Ld=4096`              |  64   | 252.76 ms       | 841.58 ms               |  47.92 ms   | **5.27×**      | **17.56×**     |
| **`bs=256, Ld=8192`** (real recipe) | 64 | **544.52 ms** | **1668.55 ms**       | **91.51 ms**| **5.95×**      | **18.23×**     |


LIK is a steady **5.0-6.9×** over vanilla and **16-22×** over the
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


| recipe              | setup                              | vanilla PyLate | + LIK    | speedup   |
| ------------------- | ---------------------------------- | -------------- | -------- | --------- |
| `Contrastive`       | bs=16, Lq=32, Ld=256               |  64.3 ms       |  60.9 ms | 1.06×     |
| `CachedContrastive` | bs=64, mini=16, Ld=300, grad-ckpt  | 317.2 ms       | 318.5 ms | 1.00×     |
| `CachedContrastive` | bs=128, mini=16, Ld=512, grad-ckpt | 692.9 ms       | 687.5 ms | 1.01×     |


Reading: on a 17 M encoder the transformer forward already swallows
most of the step, so LIK lands in the 1.00–1.06× band across all three
shapes. `bs=16, Ld=256` buys ~6 %; the two larger batches sit flat at
~1×. Same bottleneck story as the LateOn 149 M numbers from v0.1.0
(`bench_pylate_lateon.py`), just shifted up the batch axis because the
encoder is 9× smaller. The `bs=64, Ld=300` shape used to be the
headline win (1.15× in 0.3.0); it regressed to ~1.00× in this sweep
and is queued for investigation in 0.3.1 (see CHANGELOG).

## End-to-end ColQwen2 / ColPali training

Real `colpali_engine.models.ColQwen2("vidore/colqwen2-v1.0")` (Qwen2-VL
2 B backbone + LoRA, `d=128`, multi-vector image / query embeddings),
LoRA-only training (37 M of 2 246 M params trainable — the official
ColPali recipe via `peft.get_peft_model`), AdamW + bf16 weights. We
swap `patch_colpali_engine()` on/off and time full optimizer steps
(vision tower + text encoder forward + ColbertLoss + backward + step).
Synthetic shapes and real `vidore/docvqa_test_subsampled` are both covered.


| loss head           | setup                              | vanilla colpali_engine | + LIK     | speedup   | peak     |
| ------------------- | ---------------------------------- | ---------------------- | --------- | --------- | -------- |
| `ColbertLoss`       | synth bs=4, 448px                  |  379.8 ms              |  375.8 ms | 1.01×     |  9.10 GB |
| `ColbertLoss`       | synth bs=8, 448px, grad-ckpt       |  931.5 ms              |  911.3 ms | 1.02×     |  5.74 GB |
| `ColbertPairwiseCE` | synth bs=4, 448px                  |  373.3 ms              |  386.1 ms | 0.97×     |  9.10 GB |
| `ColbertLoss`       | real DocVQA bs=4                   |  794.3 ms              |  781.4 ms | 1.02×     | 16.20 GB |
| `ColbertLoss`       | real DocVQA bs=8, grad-ckpt        | 2017.2 ms              | 1975.9 ms | 1.02×     |  8.05 GB |


Reading: ColPali's Qwen2-VL-2B backbone has a much heavier
forward+backward than a ModernBERT-149 M ColBERT, and the image
modality blows up Ld (≈1 030 visual tokens at the default 448 px
resolution). So even with LoRA-only training shrinking AdamW state
by ~60×, the *encoder activation-grad backward* is what dominates
the step — LIK lands in the 0.97–1.02× range here. The kernel is a
drop-in — no other code changes between the two columns. Same
takeaway as the PyLate `Contrastive` recipe on the 149 M encoder:
when the transformer is the bottleneck, LIK doesn't move the needle,
it just doesn't hurt. The 0.97× `ColbertPairwiseCE bs=4` cell is at
the edge of run-to-run noise on a step this encoder-dominated.

## Edge models (`d ∈ {48, 64}`)

Edge ColBERT models (`d ∈ {48, 64}`) are more memory-bound, so the
fused kernel widens its lead. From `bench_inference_edge.py`:


| shape                                       | fused    | naive (fp32) | speedup   | fused mem | naive mem |
| ------------------------------------------- | -------- | ------------ | --------- | --------- | --------- |
| LateOn-Code-edge `Nd=1 000, Ld=1 024, d=48` | 0.106 ms | 0.397 ms     | **3.8×**  | 0.0 MB    | 314 MB    |
| LateOn-Code-edge `Nd=1 000, Ld=4 096, d=48` | 0.135 ms | 1.490 ms     | **11.0×** | 0.0 MB    | 1.2 GB    |
| LateOn-Code-edge `Nd=1 000, Ld=8 192, d=48` | 0.262 ms | 3.039 ms     | **11.6×** | 0.0 MB    | 2.5 GB    |
| LateOn-Code-edge `Nd=16 000, Ld=512, d=48`  | 0.254 ms | 3.036 ms     | **12.0×** | 0.1 MB    | 2.5 GB    |
| mxbai-edge `Nd=1 000, Ld=4 096, d=64`       | 0.172 ms | 1.754 ms     | **10.2×** | 0.0 MB    | 1.5 GB    |
| mxbai-edge `Nd=16 000, Ld=512, d=64`        | 0.331 ms | 3.565 ms     | **10.8×** | 0.1 MB    | 3.0 GB    |


## Where this kernel actually moves the e2e needle

1. **Inference / reranking** — no encoder backward → MaxSim *is* the
  step. **1.7–15×** at matched numerics, biggest wins on wide
  `Lq · Ld` shapes.
2. **Small-encoder training** — encoder small enough that MaxSim is
  material; LateOn-Code-edge moves **1.00–1.06×** end-to-end on real
  MS MARCO triplets in this sweep (was 1.04–1.15× pre-0.3.0; the
  `bs=64, Ld=300` headline is queued for 0.3.1 — see CHANGELOG).
3. **Long-context regimes** (`Ld ≥ 8k`) — fused kernels run, naive
  doesn't.
4. **Compressed indices** — PLAID rerank vs `engine.search()` is
  **8–23×** on full corpus scoring and **18–51×** on the partial
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
| `Nq=128, Nd=128, Lq=32, Ld=300`   | 621 MB        | 64 KB     | 2 MB               |
| `Nq=1, Nd=1000, Lq=1024, Ld=1024` | 4.5 GB        | 4 KB      | 4 MB               |
| `Nq=16, Nd=32, Lq=32, Ld=8192`    | 2.1 GB        | 64 KB     | 64 KB              |


This is what unlocks large in-batch negatives: at the same HBM budget
you fit ~5–10× more of them than vanilla PyLate.

## Apple Silicon (MPS)

`MaxSimScorer` and `retrieve` work on `mps:0` tensors. Two
implementations land on Apple Silicon and the dispatch picks per call:

* **`metal`** — fused `simdgroup_matrix` kernels for both forward
  (`maxsim_inference_metal`) and forward + backward (`maxsim_train_metal`
  / `maxsim_backward_metal`), JIT-compiled via `torch.mps.compile_shader`.
  Never materialises the `[Nq · Nd · Lq · Ld]` similarity tensor and now
  also handles autograd-tracking training calls (via `_MaxSimFnMetal`)
  when the dtype / shape suit the kernel.
* **`compile`** — `torch.compile`-fused dense reference. Autograd-aware
  fallback for fp32 inputs, `d > 128`, and tiny shapes the Metal launch
  overhead doesn't amortise on.

MacBook Air Apple M4 (2025, 16 GB unified memory), fp16, 30-iter median
(`benchmarks/bench_mps.py`). `metal` needs `d ≤ 128` and `d % 8 == 0`;
outside that the dispatch transparently falls back to `compile`.


| shape                                          | metal    | compile  | eager    | metal vs eager | metal vs compile | compile vs eager |
| ---------------------------------------------- | -------- | -------- | -------- | -------------- | ---------------- | ---------------- |
| `rerank-short` (Nq=1, Nd=1000, Lq=32, Ld=300)  | 8.45 ms  | 18.33 ms | 18.04 ms | **2.14×**      | 2.17×            | 0.98×            |
| `rerank-mid` (Nq=1, Nd=500, Lq=32, Ld=1024)    | 15.86 ms | 65.40 ms | 31.28 ms | **1.97×**      | 4.12×            | 0.48×            |
| `rerank-10k` (Nq=1, Nd=10k, Lq=32, Ld=300)     | 57.33 ms | 184.88 ms| 181.66 ms| **3.17×**      | 3.22×            | 0.98×            |
| `colpali` (Nq=1, Nd=100, Lq=32, Ld=1024)       | 2.94 ms  | 13.51 ms | 6.45 ms  | **2.19×**      | 4.59×            | 0.48×            |
| `colpali-big` (Nq=1, Nd=500, Lq=32, Ld=1024)   | 16.11 ms | 66.91 ms | 30.15 ms | **1.87×**      | 4.15×            | 0.45×            |
| `edge-d48` (Nq=1, Nd=4k, Lq=32, Ld=1024, d=48) | 31.98 ms | 457.1 ms | 111.7 ms | **3.49×**      | 14.29×           | 0.24×            |
| `edge-d64` (Nq=1, Nd=1k, Lq=32, Ld=300, d=64)  | 4.67 ms  | 14.09 ms | 10.55 ms | **2.26×**      | 3.02×            | 0.75×            |
| `train-batch` (Nq=Nd=32, Lq=32, Ld=200)        | 5.44 ms  |  9.76 ms |  2.36 ms | 0.43×          | 1.80× (compile)  | 0.24×            |


For someone moving from plain PyTorch to this library, the headline is
`metal vs eager`: **1.9–3.5×** on every realistic inference shape. On
`train-batch` the Metal kernel's launch overhead dominates and eager
actually wins (2.36 ms vs metal's 5.44 ms) — the dispatch heuristic
(`Nq * Nd ≥ 64 ∧ Ld ≥ 192`) is tuned to route this regime away from
metal. Override with `LIK_FORCE_MPS_BACKEND={metal,compile,reference}`
or `LIK_DISABLE_COMPILE=1` if you need explicit control.

Memory is the second story: because `metal` streams `D` in 32-row tiles
through threadgroup memory, peak working-set is one output tensor
(8 MB) regardless of the corpus size. On `rerank-10k` the compile /
eager paths materialise 2.5–4.0 GB intermediates and on `edge-d48`
0.75–1.5 GB — **~95–500×** memory reduction.

The Metal kernel uses Apple's 8×8 `simdgroup_matrix` MMA (the Metal
analogue of CUDA tensor cores) on top of a *persistent* threadgroup
design: each launch serves 8 consecutive `j` values, loading `Q` once
into a register-resident `simdgroup_matrix` cache that's reused across
every `(j, d-chunk)` pair (collapsing 8·Ld_chunks redundant LDS reads
into a single load pass). The cooperative D load stages each row
through per-thread registers so the L2-normalize fold pays one LDS
write instead of three. Tolerances stay inside 5e-3 relative for fp16
and 3e-2 for bf16.

The same kernel family also covers training on Apple Silicon: an
argmax-saving forward (`maxsim_train_metal`) plus a single-pass
backward (`maxsim_backward_metal`) that mirrors Triton's
`maxsim_backward_unified` — `grad_Q` is row-owned, `grad_D` scatters
via `atomic_uint` CAS (M1-compatible), and the L2-normalize Jacobian
is folded into the kernel writes. `KD / pairs` layouts (4-D `D` of
shape `[Nq, K, Ld, d]`) route to the same launch via a runtime flag
bit, matching PyLate's `colbert_kd_scores` / `colbert_scores_pairwise`.

Training (forward + backward) on the same MacBook Air M4, fp16, 30-iter
median:


| shape                                       | metal   | compile  | eager   | metal vs eager | metal vs compile |
| ------------------------------------------- | ------- | -------- | ------- | -------------- | ---------------- |
| `train-bs8` (Nq=Nd=8, Lq=32, Ld=64)         | 0.52 ms |  0.52 ms | 1.34 ms | **2.58×**      | 1.00×            |
| `train-bs16` (Nq=Nd=16, Lq=32, Ld=128)      | 2.35 ms |  3.37 ms | 1.93 ms | 0.82×          | 1.44×            |
| `train-bs32` (Nq=Nd=32, Lq=32, Ld=200)      | 6.19 ms | 24.69 ms | 7.43 ms | **1.20×**      | **3.99×**        |
| `train-bs8-long` (Nq=Nd=8, Lq=32, Ld=512)   | 1.44 ms |  4.84 ms | 2.29 ms | **1.60×**      | **3.36×**        |
| `train-d64` (Nq=Nd=16, Lq=32, Ld=256, d=64) | 1.65 ms |  4.96 ms | 2.83 ms | **1.71×**      | **3.00×**        |


Same launch-overhead story as forward `train-batch`: on `train-bs16`
(`Nq · Nd · Ld = 32 768`, matmul work ≈ 17 M FMAs split across the two
backward passes) Metal's per-launch fixed cost doesn't amortise against
eager's already-fast native ops and the kernel runs 0.82× of eager —
reproducibly across runs. From `Ld ≥ 200` or `d = 64` (which halves
matmul cost relative to the scatter) Metal pulls ahead 1.2–1.7×, and
the gap vs `torch.compile` opens to 3–4× because Inductor still
materialises the `[Nq, Nd, Lq, Ld]` score tile for the backward.