# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased — final 0.9.0 polish (release candidate)

Adds the last missing kernel for vLLM-style reranker scheduling and
finishes trimming the public surface. Documentation rewritten end-to-end.

### Added

- `maxsim_inference_scatter(Q, D, cu_q, cu_d, pair_q, pair_d)` —
  pair-list MaxSim. Scores arbitrary `(query_index, doc_index)` pairs
  from packed batches and returns `[num_pairs]` directly. Skips the
  `[Nq, Nd]` allocation when the pair list is sparse — typical
  reranker scheduling inside vLLM and similar serving stacks.
- `maxsim_residual_varlen` — fused PLAID decompress + L2-normalize +
  MaxSim over `cu_seqlens`-indexed flat buffers (the on-disk layout
  fast-plaid and ColBERTv2 already use). Forward only. ~30× less GPU
  memory and 3.4–3.7× faster than a PyTorch transliteration; 19–30×
  vs `fast_plaid.engine.search()` end-to-end.
- `triton.autotune` on `maxsim_residual` and `maxsim_residual_varlen`,
  keyed on `(Lq, max_Ld, d_pad, nbits, normalize, SAVE_ARGMAX)`.
- `late_interaction_kernels.experimental` — research kernels
  (`soft_maxsim`, `smooth_maxsim`, `maxsim_xtr`, `maxsim_matryoshka`).
- `late_interaction_kernels.fp8` — FP8 quantize / dequantize helpers.

### Changed

- `maxsim_residual` skips the argmax live-update on the inference path,
  reducing register pressure.
- L2-norm uses `tl.rsqrt` instead of `tl.sqrt + reciprocal`.
- `maxsim_residual*` accept centroids in `fp16` / `bf16` directly.
- README / `docs/*.md` rewritten: shorter, no marketing prose, every
  cross-reference checked.

### Deprecated (back-compat shim, removal scheduled post-0.9)

- `maxsim_topk` → use `retrieve(Q, D, top_k=...)` (same semantics, CPU
  fallback included). The kernel still lives at
  `late_interaction_kernels.topk`.
- `maxsim_residual_inference` → `maxsim_residual` skips the argmax save
  when `Q.requires_grad=False`. Same speed, same numerics.
- `maxsim_varlen_inference` → `maxsim_varlen` skips the argmax save
  when neither input has `requires_grad=True`.
- `maxsim_forward` → `maxsim_inference` (or `maxsim` for autograd).
- Research kernels and FP8 helpers moved to their own submodules
  (`experimental` / `fp8`).

### Removed

- `benchmarks/bench_new_kernels.py`, `scripts/sky_dev.yaml`,
  `scripts/sky_full.yaml`, `scripts/sky_quick_run.yaml`,
  `scripts/sky_pylate_only.yaml`, `scripts/smoke_fp8.py` — superseded
  by per-kernel benchmarks and the `sky_test.yaml` / `sky_lateon_edge.yaml`
  workflow.

## 0.9.0 — User-friendly API layer, cleaner defaults, docs audit

This is an ergonomics-and-honesty release. The kernels are unchanged —
every speedup number in the README / `docs/benchmarks.md` still
reproduces bit-for-bit. What changed is the surface area:

- A high-level `nn.Module` (`MaxSimScorer`) and a top-level `retrieve()`
entry point so you don't have to assemble `maxsim + topk + chunk`
yourself.
- Per-call `backward=` kwarg on `maxsim`, so different experiments can
pick different `grad_D` paths without global state.
- A sweep of documentation fixes where code and prose had drifted out
of sync after 0.6 / 0.7 / 0.8.

### Added

- `**MaxSimScorer(nn.Module)`** — stateless scoring layer with
`normalize=True`, `backward="auto"` defaults and an optional
`mask_pad_token=` shortcut. Composes with any encoder module and
round-trips through `torch.compile`.
- `**retrieve(Q, D, top_k, *, chunk=None, normalize=True)**` — the
one-liner answer to "how do I actually search 100k docs". Wraps
`maxsim_topk` with friendlier defaults and clearer docs.
- **CPU / non-Triton fallback for the high-level API** —
`MaxSimScorer` and `retrieve` are now importable *and runnable* on
macOS / Windows / CPU-only CI. They transparently dispatch to the
pure-PyTorch reference, preserving the full API contract (including
autograd and `torch.autograd.gradcheck`). Lets you unit-test
training / retrieval code locally before renting a GPU.
- **CPU-reachable test suite** — `tests/test_retrieve_cpu.py`
(30 assertions incl. `gradcheck`) plus a `test_public_all_exports_are_resolvable`
guard that flags `__all__` drift without needing a GPU.
- **Per-call `backward=` kwarg on `maxsim**` —
`maxsim(Q, D, ..., backward="csr")` pins a single call's `grad_D`
path without touching global state. `set_backward_method(...)` still
works as the process-wide default.
- **Unnormalized-input warning** — `maxsim(..., normalize=False)`
emits a one-time `UserWarning` when Q's median token L2 norm is
clearly ≠ 1.0 (the top footgun for users coming from PyLate).
Silence with `LIK_SUPPRESS_NORM_WARN=1`.
- **Loss-module patch warning** — `patch_pylate()` now emits a
`RuntimeWarning` if it cannot reach one of the internal PyLate loss
symbols (e.g. PyLate refactors `contrastive.colbert_scores`),
instead of silently leaving that loss unpatched.

### Changed

- **Design doc rewrite** — `docs/design.md` §Backward now describes
the real `unified` + `csr` selector. The 0.5.x atomic-vs-CSR
heuristic had survived the 0.6.0 rewrite in prose only.
- `**docs/rfc/0.6.0.md` and `docs/rfc/0.7.0.md**` — now clearly marked
as historical planning documents, with a banner pointing to the
`CHANGELOG` for what actually shipped. In particular, 0.7.0's
"persistent SMEM-cached fused head" is documented as **not** the
route 0.8.0 took (closed-form backward, same perf outcome).
- `**maxsim` / `maxsim_inference` docstrings** — clarified `normalize`
default behavior and the per-call `backward=` semantics.
- **Varlen input validation** — `_varlen_forward` now raises
`ValueError` with actionable messages instead of bare `assert`s.
- `**bench_backward_method.py**` — added a `unified ms` column and
fixed the `auto_pick` annotation to match the real selector
(`unified` vs `csr`, never `atomic`).
- `**bench_cached_maxsim.py**` — defaults bumped to `50` iters / `5`
warmup to match the rest of the bench suite (docs/benchmarks.md
claim is now consistent with every script).
- **README Quickstart** — restructured around the three canonical
entry points (`patch_pylate`, `MaxSimScorer`, `retrieve`) instead
of only PyLate + raw `maxsim_inference`.
- **Module-preamble docstrings** (`forward.py`, `smooth.py`, `fp8.py`,
`backward_csr.py`, `backward_unified.py`) — trimmed the RFC-style
motivation/derivation sections and cross-link to `docs/design.md`
for the long form.
- `**maxsim_reference` dtype policy** — preserves fp64 inputs instead
of down-casting to fp32. Required for `torch.autograd.gradcheck` on
the high-level API. fp16 / bf16 still promote to fp32 as before; fp32
behavior is unchanged.
- **Test naming** — `test_contrastive_loss_uses_flash` →
`test_contrastive_loss_uses_patched_scores` (and sibling cached /
distillation tests). There is no FlashAttention anywhere in this
project; the historical name was misleading.
- `**CONTRIBUTING.md**` — PR checklist no longer references a
non-existent "Unreleased" CHANGELOG section.

### Deprecated

- `**late_interaction_kernels.maxsim_forward**` — still importable
(emits `DeprecationWarning`), scheduled for removal. Use
`maxsim_inference` for reranking or
`from late_interaction_kernels.forward import maxsim_forward` if
you genuinely need the low-level primitive.
- `**maxsim_varlen_inference**` — redundant alias, emits
`DeprecationWarning`. `maxsim_varlen` auto-skips the argmax save
when neither input has `requires_grad=True`.

### Removed

- Nothing. This release is strictly additive + deprecation.

## 0.8.0 — Closed-form fused-head backward, LateOn integration

Perf + integration release. The 0.7.0 `maxsim_from_hidden_train` shipped
an autograd-aware wrapper that was **numerically correct** (≤ 2 % RMS
vs unfused) but **~2× slower** than plain `F.linear + F.normalize + maxsim`, because the backward rebuilt the winners slice through Python
autograd. 0.8.0 rewrites it to a closed-form backward and drops the
rebuild entirely. It also aligns benchmarks and docs with LightOn's
new `[lightonai/LateOn](https://huggingface.co/lightonai/LateOn)`
model family.

### Added

- **End-to-end `LateOn-Code-edge` training benchmark** (17 M Ettin
encoder) at huge-batch / long-context shapes — the 1 × H100 numbers
measured here are **1.04–1.27× end-to-end** (see README “End-to-end
LateOn-Code-edge training” table). The small encoder makes MaxSim a
material slice of the step, so the kernel swap actually moves the
wall-clock. Reproduce with `scripts/sky_lateon_edge.yaml`.
- `**benchmarks/bench_fused_head_train.py**` — microbench of the
rewritten `maxsim_from_hidden_train` vs the unfused
`F.linear + F.normalize + maxsim` path across LateOn / LateOn-Code /
LateOn-Code-edge shapes. Reports forward, backward, total, and peak
memory.

### Changed

- `**maxsim_from_hidden_train` backward** — closed-form instead of a
Python autograd rebuild. Forward still saves only `argmax`, `H_d`,
`Q`, `W`, `b`; backward:
  - gathers `H_win` at winning positions,
  - recomputes `D_unnorm_win = H_win @ W + b` and `D_hat_win` in
  `compute_dtype` (bf16/fp16 × fp32 accumulator),
  - applies the L2-normalize Jacobian directly (no `autograd.grad`),
  - computes `grad_Q`, `grad_W`, `grad_b` with plain `einsum` /
  `matmul`,
  - scatters `grad_H` into a tensor allocated in `H_d.dtype` via
  `index_add_` (no large fp32 intermediates).
  Measured on 1 × H100 bf16, `LateOn` / `LateOn-Code-edge` shapes:
  **1.01–4.64× faster** than the unfused reference, with peak memory
  on par with or below it. No change to numerics (same atol/rtol
  tests pass as in 0.7.0).
- **Default benchmark model renamed** — `bench_moderncolbert.py` →
`bench_lateon.py`, `bench_pylate_moderncolbert.py` →
`bench_pylate_lateon.py`, default checkpoint flipped to
`lightonai/LateOn`. `lightonai/GTE-ModernColBERT-v1` remains
supported and numbers remain valid — same ModernBERT-base backbone.
- **Docs** — `docs/design.md` gets a new *Fused heads* section
describing the 0.6+ inference head, the 0.8.0 closed-form backward,
the 0.7.0 top-K argmax save for `smooth_maxsim`, the FP8 inference
path, and Triton 3.2+ warp specialization. `docs/supported_models.md`
and `docs/benchmarks.md` are refreshed for LateOn.

## 0.7.0 — FP8 inference, smooth top-K MaxSim, warp-specialized autotune

Feature release. Adds Hopper/Blackwell FP8 inference, a top-K MaxSim
variant with smoother training gradients, warp-specialized autotune
configs (FA-3 style), and an autograd-aware training wrapper for
`maxsim_from_hidden`. See
`[docs/rfc/0.7.0.md](docs/rfc/0.7.0.md)` for design.

### Added

- **FP8 MaxSim inference** (`maxsim_fp8`, `quantize_fp8`) — per-tensor
/ per-token e4m3 inputs with fp32 accumulator and a score-tie
fallback harness that re-runs `(i, j)` cells in bf16 when the FP8
score is within a ULP-equivalent threshold of the runner-up.
Preserves top-K ranking on retrieval benchmarks at ~80 % of the raw
tensor-core speedup.
- `**smooth_maxsim`** — finite-K variant of MaxSim: score is the mean
of the top-K per-query-token inner products. Implemented as a
streaming argmax-union Triton kernel that writes a
`[Nq·Nd·Lq·K]` argmax buffer consumed by an `index_add`-based
backward. `K=1` degenerates to hard MaxSim; `K ≥ 4` gives softer
gradients than the β-tuned `soft_maxsim` without a temperature
knob.
- **Warp specialization** (Triton ≥ 3.2) — producer/consumer schedule
wired into the MaxSim forward autotune, with transparent fallback
on older Triton. Overlaps `D_tile` loads with the previous
`Q·Dᵀ` issue, buying another ~10–20 % on H100.
- `**maxsim_from_hidden_train`** — autograd-aware training wrapper
around `maxsim_from_hidden` (experimental in 0.7.0; backward rebuilt
the winners slice in Python autograd — **correct (≤ 2 % RMS vs
unfused) but ~2× slower**; 0.8.0 rewrites this in closed form).

## 0.6.0 — Unified backward + fused inference head

Performance release. Core training path is now a single fused
backward kernel (FA-2 style); inference gets a fused D-side head that
avoids multi-GB intermediates on large corpora. See
`[docs/rfc/0.6.0.md](docs/rfc/0.6.0.md)` for the full design and
HBM-level motivation.

### Added

- **Unified backward Triton kernel** (`maxsim_backward_unified`,
`set_backward_method("unified")`) — single-pass `grad_Q` + `grad_D`
fused kernel. Per `(i, s)` program hoists `Q[i, s, :]` out of the
`j` loop, so the two-pass `Nd`-fold reload of `Q` for `grad_D`
collapses to a single load. Halves HBM read traffic vs the two-pass
backward on MaxSim-dominant shapes.
  Measured on H100 bf16 (50-iter median, in ms):

  | Shape                             | atomic (two-pass) | unified   | Speedup        |
  | --------------------------------- | ----------------- | --------- | -------------- |
  | train-B32 (PyLate default)        | 0.14              | 0.11      | **1.33×**      |
  | train-B128                        | 0.41              | 0.37      | 1.11×          |
  | **colpali-B4 (Lq=1024)**          | **0.90**          | **0.11**  | **8.35×**      |
  | long-doc-B32 (Ld=1024)            | 0.14              | 0.11      | **1.30×**      |
  | edge-d48 / edge-d64 / large-d-512 | 0.14              | 0.10–0.11 | **1.32–1.34×** |

  End-to-end PyLate training step (full contrastive loss + backward):
  **1.12–2.67× wall-clock** across typical shapes, with the biggest
  win at B=128 (**2.67×**) and ColPali-style B=4, Lq=1024 (**1.03×
  end-to-end, 8.35× on the backward alone**).
- `**set_backward_method("unified")`** exposed alongside the existing
`"atomic"`, `"csr"`, `"auto"` modes. The `"auto"` heuristic now
defaults to `unified` for every training shape except very
high-contention batches (`Nq ≥ 256 ∧ Nd ≥ 256 ∧ Lq ≤ 64`), where
CSR's determinism advantage still wins. No user action is needed
to benefit.
- `**maxsim_from_hidden(Q, H_d, W, b=, d_mask=, normalize=)**` —
inference-only fused kernel. Takes pre-projected queries and **raw
hidden-state** documents (`[Nd, Ld, d_model]`), applies the
projection + L2-normalize + MaxSim in a single pass so the
`[Nd, Ld, d_out]` intermediate never hits HBM. Target use case:
reranking a large corpus stored on disk as `[Nd, Ld, 768]` ModernBERT
hidden states — the intermediate `D_proj` can be multi-GB and OOMs
on edge models with `Nd > 100k`. Not autograd-aware (training-side
fusion requires a persistent kernel, deferred to 0.7.0; the HBM
analysis is in the RFC).
- `**benchmarks/bench_flash_maxsim.py`** — head-to-head microbenchmark
against `flash-maxsim` (roipony/IBM). 50-iter median, reports
ms/iter, stdev, peak memory, and speedup across ten shapes
(rerank, training batch, long-doc, edge models). H100 bf16 results
show **1.01–1.26× forward speedup** vs `flash-maxsim` on every
shape, and this library additionally ships a backward, quantized
(`maxsim_residual`), ragged (`maxsim_varlen`), soft
(`soft_maxsim`), and retrieval-top-K (`maxsim_topk`) path that
`flash-maxsim` does not.
- `**benchmarks/bench_backward_unified.py`** — microbench of the
new unified kernel vs the two-pass atomic and CSR variants.
- `**docs/rfc/0.6.0.md**` — public RFC for the 0.6.0 → 0.8.0
performance track. Includes:
  - HBM accounting showing why naive training-side fused-head is a
  net loss (4× more hidden-state reads) and why 0.7.0 needs a
  persistent kernel to reclaim it.
  - `D_proj` scratch estimates showing the `maxsim_from_hidden` win
  (up to 46 GB saved on `Nd=1M` corpora with `d_out=96`).
  - Expected speedups per milestone and validation plan.

### Changed

- `set_backward_method` now accepts `"unified"` and documents the new
default heuristic in `"auto"` mode. Callers that explicitly pin
`"atomic"` or `"csr"` are unaffected.

### Deferred to 0.7.0

- **Top-K argmax save** for smoother `soft_maxsim` backward gradients.
Needs a streaming Triton top-K kernel (Triton has no usable native
`tl.topk` for this pattern; online heap maintenance across tiles is
a proper kernel project).
- **Persistent kernel + SMEM-cached fused head for training**. The
naive training-side fused head reads `H [d_model]` 4× more than
`D_proj [d_out]`; only a persistent kernel that keeps H or Q
cached across the cross-`(i, j)` loop makes this a win. See RFC.
- **FP8 path** (Hopper WGMMA + Blackwell). Needs a per-token scale
  - score-tie fallback harness before numerical parity claims.
- **Warp specialization** (Flash-Attn-3 style). Requires Triton 3.2+.

## 0.5.1 — Packed-training cookbook, repo trim

Documentation and repo-hygiene release. No kernel changes. All 0.5.0 APIs
are unchanged and numerically identical.

### Added

- `**docs/packed_training.md`** — cookbook for wiring `maxsim_varlen` into a
heterogeneous-length training loop: when packing is worth it (a
padding-waste rule of thumb), the three pieces a packed pipeline needs
(collator, varlen-aware encoder forward, packed loss), correctness
checks against the padded path, and caveats around `torch.compile`,
gradient checkpointing, and distributed training.
- `**examples/packed_training.py**` — runnable padded-vs-packed
comparison on synthetic long-tailed data. Reports per-step loss,
wall-clock, and peak memory so the padding-waste vs packing-win
tradeoff is visible on your hardware. Runs on CPU for smoke-testing.

### Removed

- `CODE_OF_CONDUCT.md`, `SECURITY.md` — redundant boilerplate for a
small kernel library. Apache-2.0 `LICENSE` and `CONTRIBUTING.md` stay
as the canonical contributor-facing docs.
- `docs/liger.md` — speculative upstreaming essay; the Related projects
section in the README keeps the short factual reference to Liger.

### Changed

- `README.md`: dropped the `CODE_OF_CONDUCT.md` / `docs/liger.md` links,
added a pointer from the Varlen section to `docs/packed_training.md`
and `examples/packed_training.py`.
- `.github/ISSUE_TEMPLATE/config.yml`: replaced the SECURITY contact
link with a Packed-training docs entry.

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
- `**soft_maxsim` is now autograd-aware** — wrapped in a
`torch.autograd.Function` with a Triton forward (fp16 / bf16) and a
stable fp32 PyTorch reference backward (softmax-reweighted einsum).
fp32 / fp64 / CPU inputs transparently fall back to a pure-PyTorch
reference forward so `torch.autograd.gradcheck` passes cleanly on
fp64. Previous callers that only used the forward see no change
beyond the output now having a `grad_fn`.
- `**maxsim` and `maxsim_inference` now validate the shape / device
contract up front** (`Q.shape[-1] == D.shape[-1]`, same device, masks
on matching devices). Previously a mismatch could silently produce
garbage scores; now it raises `ValueError` with a clear message.

### Fixed

- `**patch_pylate()` signature compatibility with current PyLate.** The
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

- `**tests/test_pylate_compat.py`** rewritten to target only the current
PyLate signature. Adds coverage for `colbert_kd_scores` (distillation
call site), CPU fallback, `LIK_DISABLE=1` kill switch, and verifying
that `unpatch_pylate()` fully restores PyLate's original references
in every loss module.
- `**tests/test_robustness.py**` — a new senior-review-grade test file:
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

- `**normalize=True`** on `maxsim`, `maxsim_inference`, `maxsim_matryoshka`,
`maxsim_residual`. L2-normalization is fused into the forward kernel (no HBM
round-trip) and the backward correctly applies the L2-norm Jacobian.
Measured **3.4× – 16.7× faster** than explicit `F.normalize + maxsim`
on H100 (see `benchmarks/bench_normalize.py`).
- `**maxsim_topk(Q, D, k, chunk=None)`** — `(values, indices)` top-k in one
call; `chunk` bounds peak memory for very large corpora.
- `**maxsim_matryoshka(Q, D, dims=[...], normalize=False)`** — evaluate
MaxSim at multiple embedding-prefix dims in a single launch (1.56× on 3
dims vs running 3 separate kernels).
- `**maxsim_xtr(Q, D, top_k=k, normalize_by_k=True)`** — XTR-style
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
- `**bench_pylate_lateon.py --recipe reason`** drives the real
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
- **ModernColBERT benchmarks** (`benchmarks/bench_lateon.py`): long-document
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

