# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-05-26

### Removed

- [breaking] **Experimental kernels.** The `late_interaction_kernels.experimental`
  package and its three research variants (`soft_maxsim`, `smooth_maxsim`,
  `maxsim_matryoshka`) are gone, along with `reference.maxsim_reference_soft`
  and the matching `tests/test_{soft,smooth,matryoshka}.py` plus the two
  soft-maxsim cases in `tests/test_robustness.py`. None of them shipped to
  PyLate, colpali_engine, FastPlaid, or NextPlaid; folding research kernels
  into prod was the same mistake as `maxsim_xtr` in 0.2.0. Users on a
  research path can vendor the kernel source from the pre-0.3.0 git
  history.
- [breaking] **Deprecated `*_inference` shims and `maxsim_from_hidden_train`.**
  The four shims marked `DeprecationWarning` in 0.2.0 ("will be removed in
  the next breaking release") are removed:
  - `late_interaction_kernels.maxsim_inference` → `maxsim(...)`
  - `late_interaction_kernels.fused_head.maxsim_from_hidden_train` →
    `maxsim_from_hidden(...)`
  - `late_interaction_kernels.varlen.maxsim_varlen_inference` →
    `maxsim_varlen(...)`
  - `late_interaction_kernels.plaid.maxsim_residual_inference` →
    `maxsim_residual(...)`

  Each surviving function already auto-skips the saved argmax buffer
  when no input has `requires_grad=True`, so behaviour is unchanged.
- [breaking] **`set_backward_method` / `get_backward_method`** removed.
  They were deprecated in 0.2.0 and carried no expressivity over the
  per-call `backward=` kwarg on `maxsim(...)` / `MaxSimScorer(...)`.
  Migration: replace `set_backward_method("csr")` with
  `maxsim(..., backward="csr")` (or `MaxSimScorer(backward="csr")`).
  `maxsim()`'s `backward=None` now resolves directly to `"auto"`
  instead of reading a module-level global.
- [breaking] `reference.xtr_reference` and its three CPU-only tests
  in `tests/test_reference_cpu.py`. The XTR Triton kernel was already
  deleted in 0.2.0; the orphaned PyTorch helper is now gone too.

### Changed

- Internal dead code removed: `pylate_compat._bool_mask` (shadowed by
  `_mask_as_bool`) and `mps.is_mps_tensor` (exported but unreferenced).
- `plaid._maxsim_residual_forward` and `maxsim_residual_varlen` now call
  `Q.contiguous()` directly; the previous `ensure_contiguous_last(Q).contiguous()`
  pattern was a no-op pair (the unconditional `.contiguous()` already
  covered every stride layout).

### Fixed

- `docs/benchmarks.md` Apple Silicon section now points at
  `late_interaction_kernels.mps.metal.maxsim_inference_metal` (the
  module was relocated under `mps/` in 0.2.0).

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