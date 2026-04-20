# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.5.0 — Fused backward for `maxsim_residual`, autograd-aware `maxsim_varlen`

Closes the training-side gaps in the 0.4.0 kernel set. You can now train
through every scoring path the library ships — compressed/quantized
documents, ragged / packed batches, and the dense padded flow — without
ever materializing the `[Nq, Nd, Lq, Ld]` similarity tensor. All changes
are additive; the 0.4 API is unchanged.

### Added

- **Fused backward for `maxsim_residual`** — `grad_Q` is computed in a
single Triton kernel that re-decompresses the winning centroid +
residual per `(query, doc, q_tok)` in SRAM, applies the L2-norm
Jacobian, and accumulates straight into the `grad_Q` row it owns. You
can now train directly on 2/4/8-bit PLAID / ColBERTv2 compressed
document embeddings with no dense unpack and no `[Nd, Ld, d]` fp32
scratch. Only `Q` is treated as differentiable — `codes` / `residuals`
are integer tensors, and `centroids` / `bucket_weights` are frozen
k-means artefacts (their gradient is never needed for retriever
fine-tuning). See `tests/test_plaid.py::test_residual_backward_*`.
- **Autograd-aware `maxsim_varlen`** — the packed kernel now saves an
argmax buffer when either input requires grad, and a new fused
backward produces `grad_Q` and `grad_D` directly on the packed
`[sum_L, d]` layout. No repad, no `pad_sequence`, no ragged → dense
conversion round-trip. `grad_Q` is row-owned (scatter-free);
`grad_D` uses the same fp32 `atomic_add` path the padded kernel
already uses. See `tests/test_varlen.py::test_varlen_backward_*`.
- **Inference-only aliases** that skip the argmax save:
`maxsim_residual_inference(...)`, `maxsim_varlen_inference(...)`.
- **Benchmark `benchmarks/bench_backward_0_5.py`** — fused residual
fwd+bwd vs "dense unpack + maxsim" autograd, and fused varlen fwd+bwd
vs the padded autograd path on equivalent ragged batches.
- `**docs/supported_models.md**` — up-to-date list of the PyLate model
families the library speeds up out of the box, with a dedicated entry
for LightOn's LateOn-Code family and an honest note on where ColGrep
fits (and does not fit) in the story.

### Changed

- `maxsim_residual` now returns an autograd-aware result when `Q`
  requires grad — previously it silently dropped gradient. Behavior on
  non-grad inputs is unchanged and byte-for-byte identical to 0.4.
- `maxsim_varlen` likewise is now autograd-aware when either input
  requires grad. Previous 0.4.x callers that pass non-grad tensors see
  no change.

### Fixed

- **`patch_pylate()` signature compatibility with current PyLate.** The
  patched `colbert_scores` / `colbert_kd_scores` now match PyLate's
  current signature `(Q, D, queries_mask=None, documents_mask=None)`
  exactly. Previously we only accepted a single positional `mask=` kwarg,
  which meant `Contrastive`, `CachedContrastive`, and `Distillation`
  (which all call into the score fn with keyword args `queries_mask=` /
  `documents_mask=`) would raise `TypeError` after `patch_pylate()`.
  The legacy `mask=` kwarg is still silently accepted so the patch
  remains a drop-in.
- **Pinned PyLate to ≥ 1.3.3** in `pyproject.toml`'s `[pylate]` extra
  (PyLate 1.3 is the first release with the new mask-kwarg signature).
  Older PyLate (1.2.x) is no longer targeted — use
  `late-interaction-kernels==0.4.x` if you need PyLate 1.2 compatibility.

### Tests

- **`tests/test_pylate_compat.py`** rewritten to target only the current
  PyLate signature. Adds coverage for `colbert_kd_scores` (distillation
  call site), CPU fallback, `LIK_DISABLE=1` kill switch, and verifying
  that `unpatch_pylate()` fully restores PyLate's original references
  in every loss module.
- **`tests/test_robustness.py`** — a new senior-review-grade test file:
  forward determinism (bitwise equality on repeated calls), gradcheck
  on `soft_maxsim` (smooth forward → valid FD gradcheck), `torch.compile`
  smoke, `torch.inference_mode()` / `torch.no_grad()` VRAM accounting
  (argmax buffer is actually skipped), CUDA Graphs capture + replay,
  API contract errors (shape / device mismatch), small-magnitude
  numerical stability, and `soft_maxsim` backward determinism.

### Honest performance notes

- Fused residual backward is a **small wins at small `Nd` (training
  regime)** path: ~1.3–1.5× faster than `unpack → maxsim` at
  `Nq=8, Nd=64, Ld=300`. At rerank scale (`Nd ≥ 512`) the reference
  `unpack`-once + `maxsim` path is faster wall-clock-wise because the
  decompression amortizes across query tokens. The fused path still
  wins on **VRAM** at every shape (no dense `[Nd, Ld, d]` fp32
  scratch). If you need autograd at large `Nd` use the dense unpack
  path; if you only need inference use `maxsim_residual_inference`.
- Fused varlen backward is a **free win (~1.05–1.1×)** at typical code-
  retrieval and long-doc shapes, with zero repadding. The real value
  is correctness: ragged inputs that go in never have to come out as
  padded.

## 0.4.0 — New kernels: fused normalize, top-k, Matryoshka, XTR, PLAID

Broadens the library from "one MaxSim kernel" to a full set of fused
late-interaction building blocks. Every new kernel has parity tests (CPU
reference + GPU kernel) and a benchmark. All changes are additive — the
0.3.x API is unchanged.

### Added

- `**normalize=True**` on `maxsim`, `maxsim_inference`, `maxsim_matryoshka`,
`maxsim_residual`. L2-normalization is fused into the forward kernel (no HBM
round-trip) and the backward correctly applies the L2-norm Jacobian.
Measured **3.4× – 16.7× faster** than explicit `F.normalize + maxsim`
on H100 (see `benchmarks/bench_normalize.py`).
- `**maxsim_topk(Q, D, k, chunk=None)`** — `(values, indices)` top-k in one
call; `chunk` bounds peak memory for very large corpora.
- `**maxsim_matryoshka(Q, D, dims=[...], normalize=False)`** — evaluate
MaxSim at multiple embedding-prefix dims in a single launch (1.56× on 3
dims vs running 3 separate kernels).
- `**maxsim_xtr(Q, D, top_k=k, normalize_by_k=True)**` — XTR-style
top-k-aggregated MaxSim. `top_k=1` path dispatches to the fused MaxSim;
a fully-fused in-kernel heap for `top_k > 1` is on the roadmap.
- `**plaid_approx_score(query_centroid_scores, codes, doc_lengths)**` —
PLAID/ColBERTv2 approximate-scoring step (gather → mask → max → sum)
fused in a single kernel: **~20× faster** than the dense PyTorch gather.
- `**maxsim_residual(Q, codes, residuals, doc_lengths, centroids, bucket_weights, nbits=2|4|8, normalize=True)`** — PLAID exact rerank
with on-the-fly 2/4/8-bit residual decompression + L2-normalize + MaxSim,
all in one kernel. **~20× faster** than the dense unpack+normalize+MaxSim
PyTorch reference.
- **CPU parity tests** (`tests/test_reference_cpu.py`) covering every new
reference so the library ships with green tests even on macOS / CPU CI.
- `**benchmarks/run_all_benchmarks.sh`** + `scripts/sky_full.yaml` —
one-command "run the whole suite on an H100 and dump artifacts".

### Changed

- Consolidated **all PyTorch reference implementations** in
`late_interaction_kernels.reference` (previously scattered across
`plaid.py` / `xtr.py`). References are now importable on CPU-only
platforms without Triton.
- `pylate_compat` now also patches `pylate.losses.distillation.colbert_kd_scores`
and accepts the PyLate signature `(queries_embeddings, documents_embeddings, queries_mask=None, documents_mask=None)`.

### Fixed

- `test_pylate_colbert_scores_patched` now normalizes inputs before
comparison (matches real PyLate encode()) so the mask-semantics gap
(PyLate multiplies by 0, we use `-inf`) doesn't dominate random-data
tolerance.

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
- `CONTRIBUTING.md`, GitHub issue / PR templates.
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

## 0.2.2 — Kernel-level microbenchmark on the `matmul → mask → max → sum` pattern

### Added

- `**benchmarks/bench_fastplaid.py**` — microbenchmark of the shared
`matmul → mask → max → sum` scoring pattern used by many
late-interaction rerankers, at shapes typical of a 1024-candidate
rerank step (`Ld ∈ {200..8192}`, `Lq=32`, `d=128`, fp16). Compares a
PyTorch transliteration of the pattern to `maxsim_inference`. Also
runs `fast_plaid.search()` end-to-end on a synthetic corpus so the
MaxSim step can be seen in context.

### Findings

- The fused kernel vs the PyTorch `matmul + mask + max + sum` pattern:
**2.7–3.6×** at `Ld ≥ 1024`, launch-overhead-bound at MSMARCO-scale
`Ld ≈ 200`.
- A production retrieval engine's `search()` spends **< 1 %** of wall
time on this scoring step — indexing, IVF probe, approximate scoring,
and residual decompression dominate. Kernel-level fusion is most
useful in Python-side rerankers and long-document regimes where
MaxSim is actually the bottleneck.

### Docs

- New kernel-level microbenchmark section in `docs/benchmarks.md`.

## 0.2.1 — Honest ModernColBERT training numbers, CachedContrastive bench

### Added

- `**benchmarks/bench_cached_maxsim.py`** — isolated MaxSim benchmark at the
exact shapes LightOn uses to train
[Reason-ModernColBERT](https://huggingface.co/lightonai/Reason-ModernColBERT)
(bs=64/128/256, Ld=2k/4k/8k, Lq=128, mini=32). Mirrors the chunked
`(bs/mini)**2` Python loop in `pylate.losses.CachedContrastive` vs one
fused flash call.
- `**bench_pylate_moderncolbert.py --recipe reason`** drives the real
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
- `**"auto"` backward selector** (new default): empirical heuristic picks CSR
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

