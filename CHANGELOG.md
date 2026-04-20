# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.3.0 — Rename to `late-interaction-kernels`, pro-grade repo polish

### Changed (breaking)
- **Package renamed** `flash_colbert` → `late_interaction_kernels`. The new
  name is model-agnostic: this library is a fused late-interaction /
  MaxSim scoring kernel, not a ColBERT-specific project.
- Environment kill-switch `FLASH_COLBERT_DISABLE` → `LIK_DISABLE`.
- Distribution name on PyPI: `late-interaction-kernels`.
- All imports change: `from flash_colbert import ...` →
  `from late_interaction_kernels import ...`.

### Added
- `py.typed` marker (PEP 561) for downstream type checking.
- `CONTRIBUTING.md`, `SECURITY.md`, GitHub issue / PR templates.
- CI matrix over Python 3.10, 3.11, 3.12 on CPU; concurrency cancellation.
- Full Apache 2.0 license text + proper authorship notice
  (Aurélien Lac, Tony Wu).
- BibTeX citation block in the README.

### Unchanged
- All kernel behavior, numerics, autotune, API surface. A mechanical
  rename is the only migration needed.

### Migration

```python
# before
from flash_colbert import maxsim, patch_pylate

# after
from late_interaction_kernels import maxsim, patch_pylate
```

```bash
# before
FLASH_COLBERT_DISABLE=1 python train.py

# after
LIK_DISABLE=1 python train.py
```

## 0.2.2 — FastPlaid comparison, PyLate big-training analysis

### Added
- **`benchmarks/bench_fastplaid.py`** — head-to-head against LightOn's
  [FastPlaid](https://github.com/lightonai/fast-plaid) Rust engine at its
  real rerank-step shapes (1024 docs × `Ld` × 32, fp16). Measures both the
  isolated MaxSim (PyTorch proxy for FastPlaid's `tch-rs` `matmul + mask +
  max + sum`) and the end-to-end `fast_plaid.search()` latency on
  synthetic corpora.

### Findings
- FastPlaid's rerank is the same `matmul → mask → max → sum` that PyLate
  dispatches, so the kernel-level win carries 1:1: **2.7–3.6×** faster at
  `Ld ≥ 1024`, roughly **neutral** for MSMARCO-scale `Ld ≈ 200`.
- `fast_plaid.search()` spends **< 1 %** of wall time on the exact MaxSim
  rerank at today's defaults — IVF probe, approximate scoring, and
  residual decompression dominate. late-interaction-kernels is only worth integrating
  into FastPlaid for **ModernColBERT-scale corpora** (`Ld ≥ 4k`) or when
  `n_full_scores` is pushed high enough that naive scratch OOMs.

### Docs
- New "FastPlaid rerank step" section in `docs/benchmarks.md` with the
  full table + decomposition + honest verdict on integration.
- README scenario matrix now includes the two FastPlaid rows
  (`Ld ≤ 1k` neutral, `Ld ≥ 4k` 2.7–3.6× on the rerank step).

## 0.2.1 — Honest ModernColBERT training numbers, CachedContrastive bench

### Added
- **`benchmarks/bench_cached_maxsim.py`** — isolated MaxSim benchmark at the
  exact shapes LightOn uses to train
  [Reason-ModernColBERT](https://huggingface.co/lightonai/Reason-ModernColBERT)
  (bs=64/128/256, Ld=2k/4k/8k, Lq=128, mini=32). Mirrors the chunked
  `(bs/mini)**2` Python loop in `pylate.losses.CachedContrastive` vs one
  fused flash call.
- **`bench_pylate_moderncolbert.py --recipe reason`** drives the real
  `CachedContrastive` pipeline with `gather_across_devices`, bf16 autocast,
  and gradient checkpointing — matches LightOn's training recipe.

### Findings
- Isolated MaxSim at `bs=256, Ld=8192` (the actual Reason recipe):
  **13.8× faster** fwd+bwd (916 → 66 ms), **2.4× less peak memory**
  (5.1 → 2.1 GB).
- End-to-end CachedContrastive training step on 8×H100 DDP: **1.00–1.06×**
  wall-clock (the 22-layer ModernBERT still dominates by ~10×). **Free**
  swap: same numerics, same VRAM, one `patch_pylate()` call.
- Corrected earlier claim: PyLate's `CachedContrastive` *does* chunk MaxSim
  (it's a `(bs/mini)**2` Python loop of plain `colbert_scores` einsums);
  late-interaction-kernels collapses it to one fused call.

### Changed
- README + docs/benchmarks.md rewritten with the honest e2e numbers and a
  "when does late-interaction-kernels move the needle?" matrix.

## 0.2.0 — CSR backward, auto selector, ModernColBERT benchmarks, polish

### Added
- **Scatter-free CSR backward path** (`set_backward_method("csr")`): per-doc cub
  radix sort of the argmax buffer + `torch.searchsorted` for a CSR row_ptr,
  then one Triton program per `(j, t)` output cell does a non-atomic bucket
  reduction. Zero contention, one write per cell.
- **`"auto"` backward selector** (new default): empirical heuristic picks CSR
  when `Nq·Nd·Lq·d ≥ 1e8`, `Lq ≥ 1024`, or `Nd ≥ 1024`; atomic otherwise.
  Matches the best manually-chosen path on 11/11 shapes we measured.
- **ModernColBERT benchmarks** (`benchmarks/bench_moderncolbert.py`): long-document
  `Ld ∈ {4k, 8k, 16k}` sweep. Naive PyLate OOMs on 80 GB H100 at 8k; late-interaction-kernels
  runs these shapes in 0.1–0.2 ms forward / ≤ 0.5 ms backward.
- **Per-family SRAM budgets** in autotune (Hopper/A100/Ampere/Ada/generic) —
  prevents shared-memory overflows on consumer cards.
- `patch_pylate` / `unpatch_pylate` + the method selector exposed on the top
  level (`from late_interaction_kernels import patch_pylate, set_backward_method`).

### Changed
- **Default backward path is now `"auto"`** (was implicit atomic in 0.1).
- Consolidated test suite: `test_forward.py` + `test_backward.py` +
  `test_edge_cases.py` + `test_varlen.py` + `test_soft.py` + `test_pylate_compat.py`.
  Shared `rel_err` helper in `conftest.py`. Every test has a docstring
  explaining *why* it exists.
- Simpler `pyproject.toml` extras: one `dev` with everything needed for tests
  and benchmarks, plus `pylate` for the drop-in.
- CI actually runs the CPU-safe tests (reference implementation + imports)
  in addition to linting.
- Cleaner README with accurate numbers (including the ModernColBERT regime).

### Performance (1×H100 80 GB HBM3, fp16 inputs, end-to-end forward+backward)
- Default `auto` matches the best manually-chosen path on all measured shapes.
- **CSR wins by ~1.3×–1.5×** on `train-256`, `long-Lq`, `huge-Nd`.
- **Atomic wins by ~1.2×–1.4×** on small / medium training shapes.
- **ModernColBERT (`Ld=8192`)**: forward 0.08–0.18 ms, backward 0.45–1.1 ms;
  naive path OOMs.

## 0.1.0 — initial release

### Added
- Fused Triton forward kernel (`maxsim_forward`) with:
  - fp32 accumulator, bf16 / fp16 / fp32 inputs;
  - fused `q_mask` (skiplist) and `d_mask` (attention mask), `-inf` applied
    inside the kernel so masked positions can never win the max;
  - per-GPU autotune (Hopper / Ampere shortlists + shared-mem pruning);
  - optional argmax output for the backward pass.
- Autograd-aware `maxsim` (`torch.autograd.Function`) wiring the forward +
  exact backward.
- Scatter-free `grad_Q` + deterministic fp32-atomic `grad_D`.
- `soft_maxsim` — log-sum-exp streaming kernel (β-relaxation).
- `maxsim_varlen` — packed kernel (cu_seqlens, FlashAttention-style).
- PyLate drop-in (`patch_pylate` / `unpatch_pylate` / `LIK_DISABLE=1`
  kill-switch).
- Pure-PyTorch reference used as ground truth in tests.
- 60 tests covering forward parity, gradient parity, varlen, soft, edge cases,
  PyLate compatibility.
- SkyPilot YAMLs for one-shot CI and a long-lived dev box.

### Performance (H100 80 GB, bf16, SXM)
- Forward: 7.9× – 22.7× vs naive einsum, ~4500× less HBM scratch.
- Backward: 1.3× – 2.3× vs naive.
- End-to-end PyLate `Contrastive` training step: 2.86× – 3.08× vs vanilla.
