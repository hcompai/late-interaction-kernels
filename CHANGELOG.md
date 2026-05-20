# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

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

### Added

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

### Fixed

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