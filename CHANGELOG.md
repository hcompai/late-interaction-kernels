# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
