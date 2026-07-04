# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Dropped the speculative upper bounds on runtime and optional dependencies
  (`torch`, `triton`, `pylate`, `colpali-engine`): they are floor-only now, so
  a downstream env can resolve them freely. `patch_pylate()` / `patch_colpali_engine()`
  already no-op above the native-LIK cutoff (`pylate>=1.5.1`, `colpali-engine>=0.3.17`),
  so no future major can reach the monkeypatched internals. Dev-tool caps
  (`pytest`, `ruff`, `ty`, `numpy`) stay — they never reach downstream installers.

## [0.4.4] - 2026-06-10

### Fixed

- `maxsim(normalize=False)` ran a `.item()` norm-check device sync on every
  call; it now runs once per process, restoring CUDA-graph capture on the
  PyLate / colpali-engine hot path.

### Changed

- PLAID centroid codes are handled as int32 end to end (any integer dtype is
  still accepted; out-of-range codes in `plaid_approx_score` clamp to
  centroid 0).
- Batch sizes (`Nq`, `Nd`) are runtime kernel arguments, not constexpr — no
  more recompiles per batch shape under dynamic batching.
- GPU family detection is keyed on compute capability, not the device name;
  Blackwell now gets first-class autotune configs.
- The varlen, packed-pairs and residual backward launches are autotuned like
  the dense backwards (up to 1.42× on H100 at training shapes).
- Backward gradient buffers that the kernels overwrite in full use
  `torch.empty` instead of `torch.zeros` (atomic-scatter buffers stay zeroed).
- The fp8 autotune key no longer splits on mask presence.

## [0.4.3] - 2026-06-10

### Fixed

- **Chunked top-k no longer crashes when `top_k` exceeds the merge width.**
  `retrieve(Q, D, top_k=10, chunk=4)` (and the CUDA `maxsim_topk` it wraps)
  raised "selected index k out of range": each chunk contributes at most
  `chunk` columns, but the running merge always asked `torch.topk` for `k`.
  The merge now clamps `k` to the merged width; the final output still
  honors the documented `min(top_k, Nd)` contract on both paths.
- **`MaxSimScorer.forward()` respects a caller's `torch.no_grad()`.** The
  non-inference branch of `_score` forced `torch.enable_grad()`, so
  `MaxSimScorer()(Q, D)` inside `torch.no_grad()` returned a tensor with
  `requires_grad=True`. It now uses a null context and leaves the caller's
  grad mode untouched.
- **`maxsim_residual_varlen` handles an empty corpus (`Nd == 0`).** The
  launcher used grid `Nq * max(Nd, 1)` while the kernel computed
  `pid // Nd` with a constexpr `Nd=0`. The empty `[Nq, 0]` result is now
  returned before any launch.
- **`pack_padded` handles an empty batch (`B == 0`).** It crashed on
  `qlen.max()` of a zero-element tensor; it now returns the trivially-empty
  `PackedBatch` up front, so `maxsim_padded` agrees with the CPU reference
  (`[0, C]`).
- **PyLate legacy `mask=` is forwarded on the fallback path.** The patched
  `colbert_scores` / `colbert_kd_scores` / `colbert_scores_pairwise` mapped
  a legacy `mask=` into the fused path but called the original function
  with a still-`None` `documents_mask` when deferring to PyLate (CPU,
  sub-Ampere, `LIK_DISABLE=1`), silently dropping the doc mask.
- **`maxsim_backward_lowmem` raises a clean `RuntimeError` without
  Triton/CUDA** instead of failing inside the launch — same guard as
  `maxsim_backward_unified`, so `backward/__init__`'s import-anywhere
  promise holds.
- **`prune_forward` falls back to the smallest-footprint configs.** When
  every autotune config overflowed the shared-memory budget, the fallback
  returned the first two list entries — configs just pruned for exceeding
  the budget. It now returns the two with the smallest estimated SMEM
  footprint.
- **The KD path (`maxsim` with 4-D `D`) validates device agreement** for
  `Q` / `D` / masks like the 3-D path, instead of surfacing an internal
  kernel error on mixed devices.
- **FP8 fallback really works without Triton.** `maxsim_inference_fp8`'s
  documented "transparent fallback" imported the Triton-backed `maxsim`
  unconditionally; it now dispatches like `retrieve` (Triton on CUDA,
  compiled reference on MPS, eager reference elsewhere), so the dequantized
  bf16 fallback runs on CPU-only installs.

### Changed

- **`maxsim_varlen` autotune sweeps are amortized across batch shapes.**
  `max_lq` / `max_ld` were constexpr autotune keys with no bucketing, so —
  contrary to the previous docs — every distinct `(max_lq, max_ld)` pair
  re-triggered the full autotune sweep *and* recompiled the kernel. They
  are now exact runtime loop bounds (like the dense kernel's `Ld`), and
  the autotune cache is keyed on power-of-two-bucketed copies (floor 16)
  passed as key-only arguments: one sweep per bucket, zero masked
  iterations added, argmax buffer still sized on the exact `max_lq`.
  Results are unchanged. The PLAID kernels (`maxsim_residual` /
  `maxsim_residual_varlen`) deliberately keep their exact `max_Ld` key: it
  is a property of the compressed index, stable across calls, so the sweep
  amortizes to one — bucketing the loop bound there measured 30-50%
  steady-state throughput loss on `Ld=300` corpora for no benefit, which
  is also why the varlen bounds stay exact.
- **`max_seqlen_*` arguments are documented as hard loop bounds, not
  hints**, in `maxsim_varlen`, `score_pairs_packed`, and
  `maxsim_residual_varlen`: a too-small value silently truncated tokens and
  returned wrong scores. Caller-supplied values are now checked against the
  `cu_seqlens` maxima with an on-device `torch._assert_async` (no D2H sync,
  same trade-off as `pack_padded`'s length checks).
- **`maxsim_residual` squeezes a 2-D `Q` back to `[Nd]`** instead of
  returning `[1, Nd]`, matching `maxsim_residual_varlen` and the `maxsim`
  wrapper's convention. Behavior change for callers that relied on the
  un-squeezed shape.
- Removed the dead `B` constexpr from `_plaid_approx_score_kernel` (it
  forced a recompile per batch size), the unused `Lq` / `Ld` placeholder
  params from the varlen forward kernel, and the dead `Nq` constexpr from
  `_varlen_bwd_dQ_kernel`. The varlen backward kernels also take `max_lq`
  as a runtime arg instead of a constexpr (mirroring the `score_pairs`
  backward kernels), so a new query-length maximum no longer recompiles
  them.
- Docstring corrections: `maxsim_inference_fp8` no longer mentions argmax
  ties (the inference kernel never computes an argmax), and the `lowmem`
  backward docs say grads are written in the *input* dtype (fp16 / bf16 /
  fp32), not "bf16 grads".

### Removed

- The interactive kernel picker page (`docs/choose-a-kernel.html`). It predates
  the API consolidation: with `maxsim` dispatching on layout and the patchers /
  native PyLate & colpali-engine backends covering the framework paths, a
  decision tree over many entry points no longer reflects the library. The
  README "Choose a kernel" section and links are gone with it.

## [0.4.2] - 2026-06-09

### Added

- **Real-recipe e2e training benchmarks for ColQwen2 and PyLate**
  (`bench_colpali_e2e.py`, `bench_pylate_e2e.py`). Both instrument the loss
  head to record per-MaxSim-call VRAM in-train, replay each recorded shape on
  an isolated graph (exact forward/saved/backward brackets), and treat OOM as
  a recorded sweep outcome rather than a crash; `--variant vanilla|lik`
  toggles the patch, with `summarize_*_e2e.py` + `scripts/sky_*_e2e.yaml`
  driving the sweep (fresh process per cell). Measured on 1×H100 80 GB:
  ColQwen2's MaxSim op costs 7.81 GiB vanilla vs 61 MiB with LIK at B=128
  (~130×), step time at parity, and vanilla OOMs at B=128 (a 1.81 GiB request
  with 25 GiB reserved-but-unallocated) where LIK trains it — 2× batch
  headroom; PyLate (grad-ckpt regime) drops step peak 54.1 → 29.7 GiB at
  B=512, runs 1.07–1.12× faster per step, and trains B=1024 where vanilla
  OOMs. The ColQwen2 bench targets released colpali-engine 0.3.16 and shims
  its two `ContrastiveTrainer` bugs under transformers 5.x (fixed upstream in
  [colpali#412](https://github.com/illuin-tech/colpali/pull/412), unreleased).
  Tables in `docs/benchmarks.md`.

### Changed

- **`patch_pylate()` / `patch_colpali_engine()` defer to the native LIK
  backends.** PyLate ≥ 1.5.1 ([pylate#222](https://github.com/lightonai/pylate/pull/222))
  and colpali-engine ≥ 0.3.17 ([colpali#412](https://github.com/illuin-tech/colpali/pull/412))
  now ship their own LIK dispatch (`pip install "pylate[lik]"` /
  `"colpali-engine[lik]"`, via `auto` / `PYLATE_SCORES_BACKEND` /
  `COLPALI_SCORES_BACKEND`). On those versions the patches are deprecated
  no-ops that detect native support by package version and step aside (patching
  PyLate would also break `ColBERTScores`, which forwards `backend=`); older
  versions are unaffected. The native backends call `maxsim` / `maxsim_pairs`
  / `maxsim_mps` by keyword, so those signatures are now pinned by a test.
- **`benchmarks/` is grouped per comparison stack** — `kernels/` (incl. the
  platform-specific `bench_mps.py`), `plaid/`, `colpali/`, and `pylate/`, each
  e2e bench next to its summarizer. Pure moves: `--only` tags and JSON output
  names are unchanged, so existing results stay comparable. `bench_lateon.py`
  → `kernels/bench_longdoc.py` (the value is the long-document regime, Ld up
  to 16 384), and the `sky_run_all_benchmarks.yaml` `RUN_ONLY` tag `lateon` →
  `longdoc`.

### Fixed

- **`patch_pylate()` works on PyLate 1.5 again.** 1.5 renamed the scoring
  module (`pylate.scores.scores` → `pylate.scores.colbert`) and rerouted the
  contrastive losses through `ColBERTScores`; the patch now detects the
  layout, patches the defining module (covering the loss path), and rewrites
  only `Distillation`'s import-time capture on 1.5. The pylate extra's
  `>=1.3.3,<2` range is accurate again — no more 1.3.3 pin.

### Removed

- The previous e2e training benches (`bench_colpali_training.py`,
  `bench_colpali_realdata.py`, `bench_pylate_training.py`,
  `bench_pylate_realdata.py`, `bench_pylate_lateon.py`), their shared
  `_bench_common.py`, and the `sky_colpali_benchmark.yaml` /
  `sky_pylate_benchmark.yaml` jobs — superseded by the e2e harnesses above
  (`bench_colpali_loss.py` is kept; historical numbers stay in
  `docs/benchmarks.md`). Plus four stale one-offs: `bench_backward_0_5.py`,
  `bench_fastplaid.py`, `bench_training.py`, and the autotune-persistence
  reproducer (`scripts/_bench_autotune_persistence.py` +
  `scripts/sky_bench_autotune_persistence.yaml`).

## [0.4.1] - 2026-06-03

### Fixed

- **Variable-length training no longer pays repeated autotune sweeps.**
  On ColQwen2 / ColPali training with variable query lengths, a fresh 5–10 s
  Triton autotune sweep fired every time a query batch first toggled its mask
  presence — as late as step 14, costing up to **1.6× end-to-end** on
  `vidore/docvqa_test_subsampled`. Two causes, both fixed: (1) `has_q_mask` /
  `has_d_mask` were in the forward and backward autotune keys (they are
  `constexpr` toggles that change codegen but not the winning `(BLOCK_Q,
  BLOCK_D, num_warps, num_stages)` tile); and (2) Triton's autotuner also keys
  on the dtype of every tensor argument, and the absent-mask placeholder was
  `Q` (bf16) rather than the real mask dtype (int8) — so present-vs-absent
  re-split the cache regardless of the named key. Absent optional args now use
  a dtype-matched placeholder (`autotune_placeholder`), and the mask flags are
  out of the keys. Autotune reuses the cached config across mask combinations
  (Triton still JIT-compiles a correct, separately specialized kernel per
  `constexpr` value); steady-state numerics and the selected configs are
  unchanged.

## [0.4.0] - 2026-06-03

### Added

- **`lowmem` backward — ~half the training peak memory, deterministic.**
  A new destination-owned backward that accumulates `grad_Q` / `grad_D` in
  fp32 registers and writes them straight in the input dtype (bf16/fp16) — no
  full-size fp32 gradient buffer, no fp32→bf16 transient, and no atomics
  (so it is bitwise-reproducible). `auto` now routes the gradient-heavy
  shapes to it: knowledge-distillation / hard-negative layouts (where
  `grad_D` is `n_neg`-inflated) and large high-contention in-batch squares;
  `unified` still handles the common and long-query cross-products, where it
  is fastest. Measured on H100 (bf16, fwd+bwd): a `B256 × 16-neg` ColPali
  step drops from 4.3 GB / 2.36 ms to **2.2 GB / 1.37 ms** (≈½ memory,
  ~1.7× faster); `pylate-text B256` from 96 MB to 52 MB. Gradients match
  `unified` to bf16 rounding. Select per-call with `backward="lowmem"`.
- **colpali_engine explicit-negative loss heads now fused.**
  `patch_colpali_engine()` previously accelerated only the in-batch CE term
  of `ColbertNegativeCELoss` / `ColbertPairwiseNegativeCELoss`; their
  explicit positive and per-query negative scoring stayed on the unfused
  einsum. The positive term now routes through `maxsim_pairs` (diagonal
  `q[i]·d[i]`) and the `[B, n_neg, Ld, d]` negative slab through 4-D
  `maxsim` (KD layout), so neither materialises the similarity tensor; the
  in-batch term keeps reusing the already-patched inner head (no double
  work). Pos/neg fusion is CUDA-only — MPS / CPU fall back to the original
  einsum for those terms while the in-batch term still accelerates. The 4-D
  negative backward `auto`-routes to `lowmem`, making the fused heads a
  training memory win too (peak ~13–28% below vanilla, widening with
  `B × n_neg`) on top of the speedup (up to **4.31×** at `B256 × 16-neg`
  in the MaxSim-isolation bench).
- **`colpali` install extra** — `pip install
  "late-interaction-kernels[colpali]"` pulls `colpali-engine>=0.3.10,<1`
  for `patch_colpali_engine()`. CPU-only in CI (colpali_engine's
  torchvision tree conflicts with the CUDA torch wheel, so it is never
  co-activated with `torch-cuda`; the GPU parity tests install it
  out-of-band), mirroring the `pylate` extra.

### Changed

- **Long-query forward chunking — broadly faster at ColPali scale.**
  `maxsim()` now splits queries with `Lq > 512` into fixed 128-token
  chunks, scores each chunk as an independent query through the shared
  `_maxsim_cross` core, and sums the per-chunk MaxSim back per original
  query. Summing a per-token max over query tokens is exact, so forward
  and backward are numerically identical to the un-chunked path
  (autograd flows through the reshape + sum). Long queries launch more,
  shorter programs that fill the GPU instead of serialising one long
  `static_range` loop, and the kernel always sees `Lq == 128`, so the
  autotune cache collapses onto a small constant (one entry, plus one
  more for tail-padded `has_q_mask=True`) instead of one per length
  bucket. Measured on H100 (bf16) with `bench_chunking.py`, vs the
  un-chunked path: **+49–77% at `Lq=768`, and at `Lq=1024` from +24%
  in-batch to roughly break-even for rerank**.
  Shorter queries (ColBERT `Lq≤32`, long-doc `Lq≤512`) fall through to
  the existing core unchanged — no regression. Chunking is
  cross-product-only; the KD / pairs path (4-D `D`) is unaffected and
  long-`Lq` KD should use `maxsim_varlen`.
- **Autotuned backward launch params — faster training step.** The
  backward kernels previously launched with Triton's stock
  `num_warps=4`. Each is one program per output row streaming a single
  `d_pad` vector through a doc loop, so 4 warps over-subscribe the
  narrow program — the H100 optimum is 1–2 warps. Every backward kernel
  is now `@triton.autotune`d over a small `(num_warps, num_stages)`
  grid via a shared `backward/_autotune.py` config module. The key
  mirrors the forward autotuner (`Lq`, `d_pad`, layout flags; `Nd` /
  `Ld` stay out), so the cache holds one entry per regime rather than
  one per batch size, and atomic-accumulating kernels use
  `reset_to_zero` so autotune trials don't pile onto each other.
  Measured on H100 (bf16), tuning lifts `auto` by **~1.2–1.45× across
  the training shapes** (see the backward table in `benchmarks.md`),
  the largest gain on the high-contention `train-256` reduction, all at
  lower peak memory.

### Removed

- [breaking] **Backward methods `atomic` and `csr`.** The dense `grad_D`
  strategies collapse to two: `unified` (fastest, fp32 atomics) and
  `lowmem` (memory-optimal, deterministic). The legacy two-pass `atomic`
  path was strictly dominated by `unified`, and `csr`'s determinism niche
  is now covered by `lowmem`, so both were deleted along with the CSR
  build/sort machinery. `backward=` now accepts
  `"auto" | "unified" | "lowmem"`; passing `"atomic"` or `"csr"` now
  raises `ValueError`. The `auto` default is unaffected.

### Fixed

- **`maxsim_from_hidden` backward leaked a spurious gradient for fully
  d-masked documents.** A document with every token masked out scores 0
  in the forward, but the backward gathered a stale index-0 winner and
  added a non-zero contribution to `grad_Q` / `grad_H_d` / `grad_W` /
  `grad_b`. The fused-head kernel now writes a `-1` argmax sentinel for
  query rows with no valid winner and the backward gates on it, matching
  the main `maxsim` path and the unfused reference (zero gradient).
- Forward-kernel autotune config pruning now sizes its shared-memory
  estimate with the padded embedding dim (`next_pow2(d)`) instead of the
  raw `d`. For non-power-of-2 `d` the old estimate undercounted SMEM by
  up to ~2x and could admit configs that overflow at launch.
- `maxsim_residual` now raises on zero-length documents when `Q` requires
  grad. An empty doc has no MaxSim winner, so the backward had no correct
  gradient and would gather a stale index-0 winner; it now fails fast.
  Inference (no grad) is unchanged and still scores an empty doc 0.

## [0.3.0] - 2026-05-28

### Added

- **Training on Apple Silicon.** New `maxsim_train_metal` (forward +
  saved argmax) and `maxsim_backward_metal` Metal kernels mirror the
  Triton `maxsim_backward_unified` API; a `_MaxSimFnMetal(autograd.
  Function)` wires them into `maxsim_mps` so the full training path
  (forward, backward, L2-normalize Jacobian) runs on Metal instead of
  falling back to `torch.compile`. Inference and unsupported-dtype
  paths are unchanged.
- **KD / pairs layout on Metal (4-D `D`).** The MPS Metal kernel now
  accepts `D` as `[Nq, K, Ld, d]` directly — PyLate's
  `colbert_kd_scores` and `colbert_scores_pairwise` shapes hit the
  Metal kernel instead of falling back to `torch.compile`.
- **`maxsim()` 4-D dispatch + `maxsim_pairs()` on CUDA.** `maxsim(Q, D)`
  now dispatches on `D.dim()`: 3-D stays on the in-batch cross-product
  path, 4-D `[Nq, K, Ld, d]` runs as a single fused KD launch. New
  `maxsim_pairs(Q, D)` covers the `[B, Lq, d] × [B, Ld, d] → [B]`
  case. PyLate's `colbert_kd_scores` Python `for`-loop (one
  `maxsim()` call per query) collapses to one kernel launch —
  ~10× faster than the loop at `B=64, K=32` and the pairwise
  `B=5000` shape went from 355 ms (old `maxsim_varlen` packing
  path) → 0.18 ms (beats `flash-maxsim`).
- Every benchmark now reports peak VRAM (`max_memory_allocated()`) per
  variant in stdout and JSON; `bench_fp8.py` and `bench_fused_head_train.py`
  also gained `--outdir` JSON + Markdown sidecars. `benchmarks/README.md`
  documents the unified CLI and the SkyPilot driver.

### Changed

- **CUDA autotune & dispatch overhaul — broadly faster across kernels.**
  Five independent wins on the Triton path:
  - **`Lq` bucketed to next pow-of-2** in the `maxsim()` wrapper
    (#62). Variable-`Lq` training (ColBERT / ColPali, where the
    tokenizer's per-batch `max(Lq)` floats step-to-step) used to
    re-trigger the full autotune sweep on every novel `Lq`. Now
    collapses to ≤ 9 cache entries. Measured **4.7× faster
    end-to-end** on 30 H100 steps with `Lq ∈ [8, 32]` (median
    step 588 ms → 0.19 ms).
  - **KD + pairwise folded into the fast forward + backward
    kernels** (#66). A single `kd_layout: tl.constexpr` switches
    `d_global = pid % Nd` (in-batch) vs `pid` (KD / pairs) so
    PyLate's `colbert_kd_scores` (4-D `D`) and
    `colbert_scores_pairwise` shapes use the same dense fast path
    as in-batch instead of routing through `score_pairs_packed`'s
    packing layer. KD `B=64, K=32` now beats `flash-maxsim`
    (lik/flash 0.94×), down from 1.4–7.7× slower pre-PR.
  - **Persistent on-disk autotune cache** via Triton ≥ 3.4's
    `cache_results=True` on all eight autotuned kernels (#64).
    First run on a machine still pays the ~4 s sweep; every
    subsequent process / CI job / training restart loads the JSON
    winner and skips bench. **10.6× faster** cold start on the
    second process. Feature-detected — older Triton (3.0–3.3)
    silently keeps the in-memory-only behaviour, no dependency
    floor bump.
  - **Small-input forward bypass** for `Nq*Nd ≤ 500 && d ≤ 256`
    with `save_argmax=False` (#64). Fixed-config launch
    (`BLOCK_Q=32, BLOCK_D=64, num_warps=4, num_stages=2`); cold
    call drops from ~0.5–1 ms (autotune sweep) to sub-millisecond.
    Closes the gap to `flash-maxsim`'s `_maxsim_fwd_kernel_small`
    on REPL / unit-test shapes.
  - **New Hopper autotune config** `BLOCK_Q=128, BLOCK_D=128,
    num_warps=8, num_stages=3` (#57). Closes a 0.85× regression
    vs `flash-maxsim` on the compute-bound colpali rerank shape
    (`Nq=1, Nd=500, Lq=Ld=1024, d=128` — now 1.23×). Only picked
    on the shape it was designed for; no regression elsewhere.
  - **`normalize` out of the forward autotune key** (#64). The two
    constexpr branches still produce distinct binaries; they now
    share one autotune entry instead of two. Cache cardinality
    halves.
- **Benchmark CLI unified.** Experiment subsets are now `--only NAME ...`
  on every script (replacing the legacy `--shape` / `--shapes` flags and
  the older `--only` for variant selection, which moved to `--variants`).
  `scripts/sky_run_all_benchmarks.yaml` and the per-domain Sky yamls
  accept a `RUN_ONLY` env to pick a subset of tags.
- **SkyPilot bench yamls consolidated.** Four operator-facing files now
  cover every bench run: `sky_benchmark_smoke_test.yaml` (was
  `sky_run_benchmarks.yaml`), `sky_run_all_benchmarks.yaml`,
  `sky_pylate_benchmark.yaml` (new, folds the three previous
  `sky_lateon_edge.yaml` / `sky_pylate_realdata{,_long}.yaml`), and
  `sky_colpali_benchmark.yaml` (was `sky_colpali_training.yaml`).
- **MPS range refreshed (M4 2025).** Inference `metal vs eager` is now
  **1.9–3.5×** and `metal vs compile` **2.2–14.3×** (vs 1.9–3.2× and
  1.1–2.0× in 0.2.0); the gap vs `torch.compile` widened because MPS
  Inductor regressed sharply on long-`Ld` inputs. New training (fwd +
  bwd) table lands alongside, with the Metal backward 1.2–1.7× over
  eager and 3–4× over `torch.compile` once shapes amortise launch
  overhead. Full tables in `docs/benchmarks.md` Apple Silicon section.
- **JSON peak-VRAM keys standardized** to `<variant>_peak_mb` across
  `bench_pylate_lateon`, `bench_pylate_realdata`, `bench_colpali_training`,
  `bench_colpali_realdata`, `bench_cached_maxsim`, `bench_fastplaid`, and
  `bench_lateon`. Breaking for anyone parsing `benchmarks/results/*.json`
  directly (previously a mix of `peak_gb` / `_peak` / `mem_*_MB`).
- Removed unreferenced exports `pylate_compat._bool_mask` (shadowed by
  `_mask_as_bool`) and `mps.is_mps_tensor`.
- GPU CI moved from the GitHub-hosted runner to AWS CodeBuild (A10G) and
  no longer auto-runs on push to `main`; opt-in via the `run-gpu-tests`
  PR label or `workflow_dispatch`.

### Removed

- [breaking] **Experimental kernels.** `late_interaction_kernels.experimental`
  and its three research variants (`soft_maxsim`, `smooth_maxsim`,
  `maxsim_matryoshka`) are gone, along with `reference.maxsim_reference_soft`,
  `tests/test_{soft,smooth,matryoshka}.py`, and the two soft-maxsim cases
  in `tests/test_robustness.py`. None of them shipped to PyLate,
  colpali_engine, FastPlaid, or NextPlaid; folding research kernels into
  prod was the same mistake as `maxsim_xtr` in 0.2.0. Users on a research
  path can vendor the kernel source from the pre-0.3.0 git history.
- [breaking] **Deprecated `*_inference` shims and `maxsim_from_hidden_train`.**
  The four `DeprecationWarning` shims from 0.2.0 are removed:
  - `late_interaction_kernels.maxsim_inference` → `maxsim(...)`
  - `late_interaction_kernels.fused_head.maxsim_from_hidden_train` →
    `maxsim_from_hidden(...)`
  - `late_interaction_kernels.varlen.maxsim_varlen_inference` →
    `maxsim_varlen(...)`
  - `late_interaction_kernels.plaid.maxsim_residual_inference` →
    `maxsim_residual(...)`

  Each surviving function already auto-skips the saved argmax buffer
  when no input has `requires_grad=True`, so behaviour is unchanged.
- [breaking] **`set_backward_method` / `get_backward_method`** removed
  (deprecated in 0.2.0). Migration: replace `set_backward_method("csr")`
  with `maxsim(..., backward="csr")` (or `MaxSimScorer(backward="csr")`).
  `maxsim()`'s `backward=None` now resolves directly to `"auto"` instead
  of reading a module-level global.
- [breaking] `reference.xtr_reference` and its three CPU-only tests in
  `tests/test_reference_cpu.py` — the XTR Triton kernel was already
  deleted in 0.2.0.

### Fixed

- **Masked-row gradient poisoning (correctness).** When every doc token
  was masked for a `(query, doc)` pair, the Triton forward saved an
  argmax of `0` instead of a sentinel, and the unified / atomic
  backwards atomic-added a spurious `grad_scores[i, j] * Q[i, s, :]`
  into `grad_D[d_global, 0, :]`. Forward now initialises the running
  argmax to `-1`; the unified / atomic backwards skip the scatter on
  `t < 0`. CSR backward is unaffected (sorting naturally drops the
  sentinel). Matches the equivalent fix on the MPS Metal backward.

### Known regressions (queued for 0.3.1)

Four shape-specific perf regressions vs 0.2.0 on the H100 sweep
(NGC 25.06 / torch 2.8 / triton 3.x). All correctness assertions hold;
the likely root cause for each is a winning autotune config rejected
by the tighter `prune_forward` SRAM model (#73).

- `bench_forward` `text-long` (Nq=1, Nd=1k, Lq=32, Ld=1024):
  0.093 → 0.121 ms (+30 %).
- `bench_inference_edge` `LateOn-Code-edge Nd=1k, Ld=1024, d=48`:
  0.072 → 0.114 ms (+58 %).
- `bench_pylate_realdata` `Contrastive bs=16, Lq=32, Ld=256`: e2e step
  52.3 → 61.6 ms (+18 %); vanilla baseline unchanged.
- `bench_pylate_realdata` `CachedContrastive bs=64, mini=16, Lq=32,
  Ld=300, grad-ckpt`: e2e step 305.5 → 318.5 ms — former 1.15× win
  vs vanilla collapsed to 1.00×. Highest-priority of the four.

## [0.2.0] - 2026-05-22

### Added

- Self-hosted GPU CI workflow (`.github/workflows/gpu-ci.yml`) that runs the
  CUDA-marked tests on push to `main`, on PRs touching kernel-related files,
  on `workflow_dispatch`, or on PRs labelled `run-gpu-ci`. CPU-only CI was
  split into `.github/workflows/cpu-ci.yml`; both workflows now trigger only
  when their path filters match.
- Interactive kernel picker (`docs/choose-a-kernel.html`) and HTML playbook
  (`docs/how-it-works.html`) to help pick the right kernel for a workload.
- **`patch_colpali_engine()` / `unpatch_colpali_engine()`** — colpali_engine
  drop-in mirroring `patch_pylate`. Monkey-patches
  `BaseVisualRetrieverProcessor.score_multi_vector` and the three in-batch
  loss heads (`ColbertLoss`, `ColbertPairwiseCELoss`, `ColbertSigmoidLoss`)
  to route their `einsum("bnd,csd->bcns") + amax(-1) + sum(-2)` through the
  fused kernel. Negative-mining siblings (`ColbertNegativeCELoss`,
  `ColbertPairwiseNegativeCELoss`) inherit the in-batch term through their
  `self.inner_loss` reference. Falls back to the original implementation
  for `use_smooth_max=True`, `LIK_DISABLE=1`, sub-Ampere CUDA, CPU
  tensors, and `d < 8`.
- **`maxsim_padded`** — padded-input reranking helper (inspired by
  https://github.com/ErikKaum/maxsim). Takes `[B, Lq, d]` / `[B, C, Ld, d]`
  tensors with per-row lengths, returns `[B, C]` fp32. Autograd-aware on
  every device: CUDA dispatches to the fused pair-list scatter kernel
  (forward + backward), CPU / MPS fall back to the pure-PyTorch reference.
  The underlying `pack_padded(...)` building block (which converts to the
  packed `cu_seqlens` layout with a single combined `max_seqlen_q` /
  `max_seqlen_d` device→host sync) is available from
  `late_interaction_kernels.padded`.
- Fused backward for `score_pairs_packed`. The pair-list scatter kernel
  now saves a `[num_pairs, max_lq]` argmax buffer when either input has
  `requires_grad=True` and produces `grad_Q` / `grad_D` directly on the
  packed layout via two atomic-add scatter kernels. Pair-list training is
  now `O(num_pairs · max_lq · d)` on both passes; no `[Nq, Nd]`
  materialisation, no varlen-style off-diagonal compute. Pure inference
  pays no overhead (`save_argmax=False`).

### Changed

- `maxsim` now auto-skips the saved argmax buffer when neither input
  has `requires_grad=True`, matching the dispatch already shipped by
  `maxsim_varlen` and `maxsim_residual`. `maxsim_inference` is now a
  thin deprecation shim that forwards to `maxsim`.
- `maxsim_from_hidden` is now autograd-aware (gradients flow into
  whichever of `Q` / `H_d` / `W` / `b` carry `requires_grad=True`); the
  forward-only path is auto-dispatched when none of them do.
  `maxsim_from_hidden_train` is now a thin deprecation shim.
- [breaking] Bumped minimum PyTorch from `2.1` to `2.5`. Older releases
  are no longer tested and the `torch._assert_async` bounds check in
  `pack_padded` now assumes the symbol is present unconditionally.
- Replaced the unconditional CPU-only `torch` pin with explicit
  `torch-cpu` / `torch-cuda` optional extras so CUDA installs no longer
  pull a CPU wheel by default.
- Cleaned the H100 autotune pool (`_autotune.py::_small_d_hopper`). Dropped
  the two `warp_spec=True` configs that have been silent no-ops since
  Triton 3.5 removed the `num_consumer_groups` / `num_buffers_warp_spec`
  kwargs (the API moved to compiler-driven warp specialization — without
  the kwargs, those entries duplicated other configs in the pool and
  occasionally won the autotune sample on noise alone). Also resized
  `BLOCK_Q=32, BLOCK_D=128` from `num_warps=8` to `num_warps=4` so it
  matches the WGMMA warp-group size we actually want, and added the
  matching `BLOCK_Q=64, BLOCK_D=128, num_warps=4, num_stages=3` row.

### Deprecated

- `set_backward_method` / `get_backward_method` now emit
  `DeprecationWarning`. The process-wide global has no functional
  advantage over the per-call `backward=` kwarg on `maxsim` /
  `MaxSimScorer` and complicates reasoning in multi-thread / multi-rank
  setups. Migration: replace `set_backward_method("csr")` with
  `maxsim(..., backward="csr")` (or `MaxSimScorer(backward="csr")`).
  The globals will be removed in the next breaking release.
- `late_interaction_kernels.maxsim_inference` — use `maxsim(...)` directly.
- `late_interaction_kernels.fused_head.maxsim_from_hidden_train` — use
  `maxsim_from_hidden(...)` directly.
- [breaking] `maxsim_inference_scatter` → `score_pairs_packed`; module
  `scatter.py` → `score_pairs.py`. Shorter name, matches prior art in
  https://github.com/ErikKaum/maxsim. Kernel, signature, and semantics
  are identical.
- [breaking] Trimmed the top-level public surface to everyday API only:
  `MaxSimScorer`, `retrieve`, `patch_pylate` / `unpatch_pylate`, `maxsim`,
  `maxsim_inference`, `maxsim_varlen`, `maxsim_padded`, and the `reference`
  module. Lower-level / niche kernels must now be imported from their
  submodule:
  - `score_pairs_packed` → `late_interaction_kernels.score_pairs`
  - `pack_padded` / `PackedBatch` → `late_interaction_kernels.padded`
  - `maxsim_from_hidden` / `maxsim_from_hidden_train` → `late_interaction_kernels.fused_head`
  - `plaid_approx_score` / `maxsim_residual` / `maxsim_residual_varlen`
    → `late_interaction_kernels.plaid`
  - `maxsim_inference_fp8` → `late_interaction_kernels.fp8`
  - `set_backward_method` / `get_backward_method` → `late_interaction_kernels.autograd`
- [breaking] Module relocations following the submodule reorganisation.
  Direct imports of the old paths now raise `ImportError`:
  - `late_interaction_kernels._mps` → `late_interaction_kernels.mps.compile_dispatch`
  - `late_interaction_kernels.metal` → `late_interaction_kernels.mps.metal`
  - `late_interaction_kernels.backward_csr` → `late_interaction_kernels.backward.csr`
  - `late_interaction_kernels.backward_unified` → `late_interaction_kernels.backward.unified`
  - `late_interaction_kernels.{soft,smooth,matryoshka,xtr}` →
    `late_interaction_kernels.experimental.{soft,smooth,matryoshka,xtr}`

### Removed

- [breaking] Top-level deprecation shims for `maxsim_forward`, `maxsim_topk`,
  `maxsim_residual_inference`, `maxsim_varlen_inference`,
  `maxsim_matryoshka`, `maxsim_xtr`, `soft_maxsim`, `smooth_maxsim`,
  `quantize_fp8_per_tensor`, `quantize_fp8_per_token`,
  `dequantize_fp8_per_tensor`, `dequantize_fp8_per_token`. Import from
  their submodules directly: `late_interaction_kernels.{forward, topk,
  plaid, varlen, experimental, fp8}`.
- [breaking] `maxsim_xtr` (XTR top-K aggregation, the experimental kernel
  exposed at `late_interaction_kernels.experimental.xtr`). The kernel only
  ever shipped as a research curiosity; nothing in `MaxSimScorer`,
  `retrieve`, `patch_pylate`, or `patch_colpali_engine` used it. Users
  who still need XTR aggregation can take the kernel source from a
  pre-0.2.0 release or compose `maxsim` with a `topk + sum` on the
  output. The companion test (`tests/test_xtr.py`) is gone too.

### Fixed

- Interactive kernel picker (`docs/choose-a-kernel.html`) now surfaces
  `maxsim`, `maxsim_varlen`, `score_pairs_packed`, `maxsim_residual`, and
  `maxsim_residual_varlen` under the "My own training / inference code"
  branch (in addition to "Raw kernel functions"). Previously the combo
  *custom code + training + packed cu_seqlens* returned "No exact match".
- Kernel picker shows a composition recipe when the combo
  *varlen + top-k retrieval* is selected (no single fused kernel covers
  that today — the answer is `maxsim_varlen` followed by `torch.topk`).
  The picker still falls back to the generic "No exact match" message
  for combinations no recipe covers.
- `late_interaction_kernels.backward.atomic` referenced
  `late_interaction_kernels.backward.backward_csr` — a stale path from the
  submodule rename that would have raised `ModuleNotFoundError` the first
  time the `auto` backward path picked CSR on a real GPU. Pointed at the
  correct module (`...backward.csr`).
- `score_pairs_packed` no longer recompiles or re-autotunes per distinct
  `(max_lq, max_ld)`. Both were `tl.constexpr` and part of the autotune
  key, so each distinct max-seqlen bucket triggered a fresh compile +
  autotune sweep — the same trap `Ld` fell into on the dense forward in
  0.1.0. The kernel now keys only on `d_pad`. Pinned by
  `tests/test_compile_cache.py` (single autotune entry across 5 distinct
  `max_ld` / `max_lq` values).

### Documentation

- Spell out what the H100 forward table compares against (eager fp32
  reference) and why a `torch.compile` baseline isn't included on the CUDA
  side: Inductor still has to materialize the `[Nq · Nd · Lq · Ld]`
  similarity tensor before `max(-1)`, which is exactly the HBM round-trip
  the fused kernel exists to avoid.
- README banner, usage-context clarifications, and a restructured
  how-it-works walkthrough.

## [0.1.0] - 2026-05-06

### Fixed

- Triton kernels no longer recompile or re-autotune per distinct `Ld`.
  `Ld` was declared `tl.constexpr` and (for the autotuned forwards) keyed
  the autotuner — but inside the kernels it only drives a runtime
  `range(0, Ld, BLOCK_D)` loop, so variable-length training was paying
  one Triton recompile + one autotune sweep per distinct doc length.
  `Ld` is now a runtime arg and is out of the autotune key across
  `forward`, `soft`, `smooth`, `fp8`, `fused_head`, `matryoshka`, and the
  three backward kernels. Measured 9.3× faster cold start on H100 (4
  distinct `Ld` values, fp16); steady-state per-call performance
  unchanged. Pinned by `tests/test_compile_cache.py`.

### Added

- **Apple Silicon (MPS) — fused `simdgroup_matrix` Metal kernel** for forward
  MaxSim (`late_interaction_kernels.metal.maxsim_inference_metal`),
  JIT-compiled via `torch.mps.compile_shader`. Persistent threadgroups serve
  8 consecutive `j` values per launch, `Q` is register-resident across every
  `(j, d-chunk)` pair, and the cooperative D load stages each row through
  per-thread registers so the optional L2-normalize fold pays one threadgroup
  write per element instead of three. Forward-only; never materialises the
  `[Nq · Nd · Lq · Ld]` similarity tensor.
- **MPS dispatch** (`late_interaction_kernels._mps`) — `torch.compile`-fused
  reference (autograd-aware) for training calls, with the Metal kernel
  selected for inference when its envelope holds (fp16 / bf16, `d ≤ 128`,
  `d % 8 == 0`, `Nq · Nd ≥ 64 ∧ Ld ≥ 192`). Compile-time MSL errors and
  device-side faults fall back transparently to the compile path.
- **`patch_pylate()` MPS routing** — `pylate.scores.colbert_scores` /
  `colbert_kd_scores` now route MPS tensors through `maxsim_mps`. The
  `maxsim` Triton import is now lazy, so `pylate_compat` is importable on
  machines without Triton (e.g. macOS).
- Env overrides: `LIK_FORCE_MPS_BACKEND={metal,compile,reference}`,
  `LIK_DISABLE_COMPILE=1`, `LIK_MPS_METAL_MIN_BATCH`, `LIK_MPS_METAL_MIN_LD`.
- `benchmarks/bench_mps.py` benches Metal / `torch.compile` / eager
  side-by-side and reports `metal vs eager`, `metal vs compile`, and
  `compile vs eager` ratios. Apple M4 fp16: 1.9–3.2× over eager
  (1.1–2.0× over `torch.compile`) on realistic inference shapes.
- `benchmarks/bench_flash_maxsim.py` is back in the runner script and the
  documentation; pinned to `flash-maxsim==0.2.0` so the published numbers
  are reproducible.
- 87 new MPS tests across `tests/test_mps.py`, `tests/test_mps_metal.py`,
  and `tests/test_pylate_compat_mps.py` (parity, masks, autograd, dispatch
  fallbacks, env overrides, KD layout, PyLate routing).

### Changed

- `MaxSimScorer` / `_score()` now raises an explicit `ValueError` when
  `Q.device != D.device` instead of silently dropping through to the eager
  reference and surfacing an opaque `RuntimeError` from `torch.matmul` —
  same contract `retrieve()` already enforced.
- `docs/benchmarks.md` Apple Silicon section rewritten with the Metal
  numbers, the dispatch heuristic, and the headline `metal vs eager` ratio.
- **Minimum Python is now 3.10** (was 3.9). `pyproject.toml` bumps
  `requires-python = ">=3.10"`, the Python classifiers, and
  `tool.ruff.target-version = "py310"`. `uv.lock` regenerated; the CI matrix
  was already 3.10 / 3.11 / 3.12.

### Removed

- All `from __future__ import annotations` lines across the package, tests,
  benchmarks, and examples (67 files). Annotations now use the native PEP 604
  / PEP 585 syntax (`X | Y`, `list[X]`, `dict[K, V]`) that Python 3.10
  supports at runtime — no compatibility shim needed.

### Fixed

- `patch_pylate()` previously gated on `q.is_cuda` and silently fell through
  to PyLate's reference implementation for MPS tensors. Replaced with a
  per-call `_device_path(Q, D) → {"cuda", "mps", None}` switch so Mac users
  get the same one-liner upgrade as CUDA users.
- `benchmarks/results/` is now `.gitignore`d; benchmark outputs are no
  longer tracked in version control.
- **MPS `torch.compile` path no longer trips the inductor symbolic-shape
  bug.** On torch 2.8 / nightly, MPS inductor fails to lower
  `S.max(dim=-1)` when the reduction axis is symbolic
  (`cannot determine truth value of Relational: s12 <= 1024` from
  `codegen_iteration_ranges_entry`). Switched the compile call from
  `dynamic=True` to `dynamic=False` so PyTorch's dynamo cache transparently
  recompiles per `(Nq, Nd, Lq, Ld)` tuple instead — fine for typical
  inference where shapes are stable, and shape-varying workloads can fall
  back to the Metal kernel. Unblocks the 28 MPS tests that were skipped on
  this bug; **167 / 167 pass on macOS** with no skips on the dispatch /
  metal / `pylate_compat_mps` suites.
- **`README.md` header restored.** The Python-3.10 bump (#24) accidentally
  stripped `<div align="center">`, the badge `![]()` image syntax, and
  reformatted a handful of markdown tables — leaving the landing page
  un-centered with plain-text "badges" instead of the shields.io images.
  Restored from the pre-#24 revision and re-applied the
  `python-3.10–3.12` shield change. No other content changes.

## [0.0.1] - 2026-05-02

Fused Triton kernels for late-interaction (MaxSim) scoring, with a high-level
PyTorch API and PyLate drop-in.

### Added

- **Core MaxSim kernels** — `maxsim` (autograd-aware) and `maxsim_inference`
with fused L2-normalize, mask handling, and a `unified` / `csr` / `atomic`
backward selector (`set_backward_method`, default `auto`).
- **Ragged / packed batches** — `maxsim_varlen` over `cu_seqlens`-indexed
flat buffers, autograd-aware on both `Q` and `D`.
- **Pair-list scoring** — `maxsim_inference_scatter` scores arbitrary
`(query_index, doc_index)` pairs from packed batches and returns
`[num_pairs]` directly (vLLM-style reranker scheduling).
- **Fused D-side head** — `maxsim_from_hidden` (inference) and
`maxsim_from_hidden_train` (closed-form backward) apply
projection + L2-normalize + MaxSim in a single pass over raw
`[Nd, Ld, d_model]` hidden states.
- **PLAID / ColBERTv2** — `plaid_approx_score` (approximate scoring) and
`maxsim_residual` / `maxsim_residual_varlen` (exact rerank with on-the-fly
2/4/8-bit residual decompression + L2-normalize + MaxSim, forward-only on
varlen).
- **FP8 inference** — `maxsim_inference_fp8` with per-tensor / per-token
e4m3 inputs, fp32 accumulator, and a score-tie fallback harness.
- **High-level API** — `MaxSimScorer(nn.Module)` and `retrieve(Q, D, top_k)`,
both with transparent pure-PyTorch CPU fallback so training and retrieval
code is unit-testable on macOS / Windows / CPU-only CI.
- **PyLate drop-in** — `patch_pylate` / `unpatch_pylate` patch
`colbert_scores` and `colbert_kd_scores` across `Contrastive`,
`CachedContrastive`, and `Distillation`. `LIK_DISABLE=1` is the
process-wide kill switch.
- **Experimental kernels** — `late_interaction_kernels.experimental` ships
`soft_maxsim`, `smooth_maxsim`, `maxsim_xtr`, and `maxsim_matryoshka`.
- **FP8 helpers** — `late_interaction_kernels.fp8` exposes per-tensor /
per-token quantize / dequantize utilities.
- Per-GPU autotune (Hopper / Ampere / Ada / generic) with shared-memory
pruning; warp specialization on Triton ≥ 3.2 with transparent fallback.
- Pure-PyTorch reference (`late_interaction_kernels.reference`) used as
ground truth in tests and as the CPU fallback path.
- Test suite covering forward / backward parity, varlen, soft/smooth,
edge cases, PyLate compatibility, CPU fallback, and `gradcheck` on the
high-level API.
- Benchmarks for every kernel, plus end-to-end PyLate / LateOn training
and retrieval scripts under `benchmarks/` and `scripts/`.