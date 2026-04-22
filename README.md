<div align="center">

# ⚡ late-interaction-kernels

**Fused Triton kernels for late-interaction (MaxSim) scoring.**
_ColBERT · ColPali · ModernColBERT · LateOn · LateOn-Code · ColBERTv2 · PyLate-native._

[![CI](https://github.com/hcompai/late-interaction-kernels/actions/workflows/ci.yml/badge.svg)](https://github.com/hcompai/late-interaction-kernels/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A5%202.1-ee4c2c.svg)](https://pytorch.org/)
[![Triton](https://img.shields.io/badge/Triton-%E2%89%A5%203.0-9146ff.svg)](https://github.com/triton-lang/triton)
[![PyLate](https://img.shields.io/badge/PyLate-%E2%89%A5%201.3-00b4d8.svg)](https://github.com/lightonai/pylate)

[**Install**](#install) · [**Quickstart**](#quickstart) · [**Benchmarks**](#benchmarks) · [**API**](#api-reference) · [**Supported models**](docs/supported_models.md) · [**Design**](docs/design.md)

</div>

---

## Highlights

- **One-line PyLate drop-in** — `patch_pylate()`. Numerically identical,
  zero config, `LIK_DISABLE=1` kill-switch.
- **Reranking / inference: 7–23× faster** than PyTorch einsum, ~0 scratch.
  Long documents (`Ld ≥ 8k`) run where naive OOMs.
- **LightOn cached-contrastive recipe: up to 13.8×** on the MaxSim +
  backward slice at `bs=256, Ld=8k`.
- **LateOn-Code-edge (17 M) end-to-end training: 1.04–1.27×** on 1×H100,
  up to **2 GB/rank saved**. Small encoder ⇒ MaxSim is a material slice
  of the step.
- **Fused D-side head for training: 1.5–4.6× faster** than
  `F.linear + F.normalize + maxsim`, same memory.
- **FP8 MaxSim inference** (Hopper WGMMA) up to **1.4×** at large `Nd`.

All numbers on 1×H100 80 GB SXM, bf16 / fp16 compute, 50-iter median.
Every baseline is the same operation written in plain PyTorch — not a
comparison against any other library.

## What this is (and isn't)

A small library of fused Triton kernels for the **scoring math** used by
ColBERT, ColPali, ModernColBERT, LateOn, LateOn-Code and ColBERTv2. Every
kernel is a numerically-matching drop-in for the equivalent PyTorch
expression.

It is **not** a search engine — no index, no IVF, no orchestration. For
end-to-end retrieval use [FastPlaid](https://github.com/lightonai/fast-plaid),
[NextPlaid / ColGrep](https://github.com/lightonai/next-plaid), PLAID, or
plain PyLate. This library is what their MaxSim math *could* compile
down to.

---

## Install

```bash
pip install late-interaction-kernels
```

Needs CUDA + Triton (auto-installed on Linux) for the fused kernels. On
macOS / Windows / CPU-only machines, `MaxSimScorer`, `retrieve`, and
`late_interaction_kernels.reference` still work — they transparently
dispatch to a pure-PyTorch reference, so you can develop and unit-test
your training / retrieval code locally before renting a GPU.

```bash
# development
git clone https://github.com/hcompai/late-interaction-kernels
cd late-interaction-kernels && pip install -e ".[dev,pylate]" && pytest -q
```

PyLate drop-in targets **PyLate ≥ 1.3**.

---

## Quickstart

### 1. PyLate training, one line

```python
from late_interaction_kernels import patch_pylate

patch_pylate()          # swaps colbert_scores / colbert_kd_scores /
                        # Contrastive / CachedContrastive
# ... your PyLate training code is unchanged ...
```

`LIK_DISABLE=1` (checked per call) makes every patched entry point
delegate back to vanilla PyLate without un-installing the patches — a
runtime kill switch you can flip inside a job without restarting.

### 2. Custom training — `MaxSimScorer`

```python
import torch
from late_interaction_kernels import MaxSimScorer

scorer = MaxSimScorer(normalize=True)             # nn.Module, no parameters

scores = scorer(Q, D, q_mask=q_mask, d_mask=d_mask)   # [Nq, Nd] fp32
scores.mean().backward()                          # gradients flow into Q, D
```

`MaxSimScorer` is a stateless `nn.Module` that composes cleanly with any
encoder, trainer or `torch.compile` wrapper. Defaults match
ColBERT / ColPali / LateOn scoring semantics.

### 3. Retrieval — `retrieve(Q, corpus, top_k=...)`

```python
from late_interaction_kernels import retrieve

scores, indices = retrieve(Q, D, top_k=100, chunk=4096)
# scores, indices are both [Nq, 100] — docs ranked per query
```

`chunk=` bounds peak HBM at `Nq · (chunk + top_k)` instead of
`Nq · Nd`, so 100 k-doc corpora fit in a single call.

### 4. Reranking — raw `maxsim_inference`

```python
import torch
from late_interaction_kernels import maxsim_inference

Q = torch.randn(32, 128, device="cuda", dtype=torch.float16)          # [Lq, d]
D = torch.randn(1000, 300, 128, device="cuda", dtype=torch.float16)   # [Nd, Ld, d]

scores = maxsim_inference(Q, D, normalize=True)   # [1000] fp32
top10  = scores.topk(10)
```

`normalize=True` fuses `F.normalize(Q) + F.normalize(D) + maxsim`
into a single kernel (3–17× faster than the explicit version).

More patterns — Matryoshka, XTR, soft / smooth-MaxSim, ColBERTv2
residuals, varlen / packed, FP8 — live in the
[API reference](#api-reference) below with one-line examples and
links to the dedicated docs.

---

## When it helps

| Scenario                                        | End-to-end effect                        |
| ----------------------------------------------- | ---------------------------------------- |
| Reranking / inference with pre-computed embeds  | **7–23× faster**, ~0 scratch             |
| LateOn / ModernColBERT inference at `Ld ≥ 8k`   | **runs; naive einsum OOMs**              |
| Offline KD scoring (teacher + student)          | **5–14×** (no encoder bwd)               |
| ColBERTv2 rerank on compressed (2-bit) docs     | **~20×** vs PyTorch `unpack + einsum`    |
| PyLate MaxSim + loss + backward slice           | **1.12–2.67×** (unified backward)        |
| **LateOn-Code-edge end-to-end training (17 M)** | **1.04–1.27×**, up to 2 GB/rank saved    |
| LateOn / ModernColBERT training (149 M)         | 1.00–1.06× (encoder-bound, free swap)    |

**The honest summary.** On a full 149 M LateOn / ModernColBERT training
step the encoder dominates by ~10×, so the kernel is effectively free.
The moment MaxSim stops being negligible — inference, reranking, long
docs, big effective batch, KD, compressed indices, or a small encoder
like LateOn-Code-edge — the fused path actually moves. `patch_pylate()`
is a one-liner either way.

---

## Benchmarks

1×H100 80 GB SXM, bf16 (fp16 for LateOn / ModernColBERT shapes), fp32
accumulator, 50-iter median. Full tables and reproduction commands:
[`docs/benchmarks.md`](docs/benchmarks.md).

### Reranking / inference (vs naive PyTorch einsum)

| Shape                                              | lik       | naive      | Speedup    |
| -------------------------------------------------- | --------- | ---------- | ---------- |
| `Nq=1, Nd=1 000, Lq=32, Ld=300`                    | 0.031 ms  | 0.705 ms   | **22.7×**  |
| `Nq=1, Nd=10 000, Lq=32, Ld=300`                   | 0.557 ms  | 7.112 ms   | **12.8×**  |
| `Nq=1, Nd=1 000, Lq=1024, Ld=1024` (ColPali)       | 1.518 ms  | 11.967 ms  | **7.9×**   |
| LateOn-Code-edge, `Nd=1 000, Ld=8 192, d=48`       | 0.266 ms  | 2.910 ms   | **10.9×**  |
| Serving, `Nd=32 000, Lq=32, Ld=300, d=128`         | 0.766 ms  | 7.488 ms   | **9.8×**   |

Long-context (`Ld ∈ {8k, 16k}`) rows that OOM on naive still run here:
`Nq=1, Nd=256, Ld=8192` ⇒ 0.18 ms fwd / 1.12 ms bwd at 2.1 GB peak.

### LightOn cached-contrastive recipe (MaxSim + backward slice)

The `(bs / mini)²` Python loop inside `CachedContrastive` collapses to
one fused call. Same hyperparameters LightOn uses for long-doc training
(`Lq=128, mini=32`):

| Shape             | Vanilla fwd+bwd | Fused fwd+bwd | Speedup   | Vanilla peak | Fused peak |
| ----------------- | --------------- | ------------- | --------- | ------------ | ---------- |
| `bs=64, Ld=8192`  | 55.3 ms         | 4.0 ms        | **13.9×** | 4.3 GB       | 0.6 GB     |
| `bs=128, Ld=8192` | 224.0 ms        | 15.7 ms       | **14.3×** | 4.6 GB       | 1.1 GB     |
| `bs=256, Ld=8192` | **915.9 ms**    | **66.3 ms**   | **13.8×** | **5.1 GB**   | **2.1 GB** |

### End-to-end LateOn-Code-edge training (17 M encoder)

`Contrastive` loss, in-batch negatives, bf16, grad-ckpt, 1×H100:

| Setup                  | Vanilla  | late-interaction-kernels | Speedup   | Peak (vanilla → fused) |
| ---------------------- | -------- | ------------------------ | --------- | ---------------------- |
| bs=256, Lq=32, Ld=256  | 114.3 ms | 90.3 ms                  | **1.27×** | 11.5 → 9.6 GB          |
| bs=192, Lq=32, Ld=512  | 152.5 ms | 126.0 ms                 | **1.21×** | 16.6 → 14.7 GB         |
| bs=128, Lq=32, Ld=1024 | 223.7 ms | 196.4 ms                 | **1.14×** | 22.6 → 21.5 GB         |
| bs=64,  Lq=32, Ld=2048 | 293.2 ms | 281.0 ms                 | 1.04×     | 25.8 GB                |

Reproduce with `scripts/sky_lateon_edge.yaml`. Smaller encoder + bigger
effective batch ⇒ bigger slice for MaxSim ⇒ bigger e2e speedup.

These numbers use `patch_pylate()` — the drop-in that swaps PyLate's
`colbert_scores` for `maxsim`. **The fused head
(`maxsim_from_hidden_train`) is not wired in** because PyLate applies
the `Dense` projection inside the encoder forward, so `colbert_scores`
only ever sees `[Nd, Ld, d_out]` tensors. A custom trainer that skips
PyLate's `Dense` and hands raw `[Nd, Ld, d_model]` hidden states to
`maxsim_from_hidden_train` directly would recover the extra
`F.linear + normalize` passes — at these shapes that's another ~1–3 ms
on the step (encoder-bound), but substantially more peak-memory headroom
at `Ld ≥ 4k` since the `[bs, Ld, d_out]` scratch disappears. See the
[Fused D-side head section](#fused-d-side-head-for-custom-trainers-with-raw-hidden-states)
below.

### End-to-end LateOn / ModernColBERT training (149 M encoder)

`CachedContrastive`, AdamW, bf16, grad-ckpt. Numbers apply equally to
`lightonai/LateOn`, `lightonai/LateOn-Code`, and
`lightonai/GTE-ModernColBERT-v1` (same ModernBERT-base backbone):

| 8×H100 DDP setup         | Vanilla | lik     | Speedup | Peak / rank |
| ------------------------ | ------- | ------- | ------- | ----------- |
| bs=4/dev,  mini=4,  Ld=8192 | 665 ms  | 662 ms  | 1.00×   | 30.2 GB     |
| bs=16/dev, mini=16, Ld=4096 | 1078 ms | 1047 ms | 1.03×   | 57.5 GB     |
| bs=32/dev, mini=32, Ld=2048 | 1020 ms | 960 ms  | 1.06×   | 57.4 GB     |

The ModernBERT encoder (forward+backward with FlashAttention-2)
dominates the step — the kernel is a free, numerically-identical swap
here. The win shows up in the MaxSim slice (up to 13.8× above) and in
smaller-encoder settings (1.04–1.27× above).

### Fused D-side head (training, 0.8.0)

Replaces `F.linear → F.normalize → maxsim` for the hidden-state →
embeddings → MaxSim path. Forward runs one kernel with argmax save;
backward is closed-form (no autograd rebuild). Measured on H100 bf16,
forward + backward:

| Shape                                            | Unfused  | Fused   | Speedup   |
| ------------------------------------------------ | -------- | ------- | --------- |
| LateOn `Nd=128, Ld=1024, d_model=768`            | 1.45 ms  | 0.96 ms | **1.52×** |
| LateOn-Code `Nd=512, Ld=2048`                    | 7.31 ms  | 1.67 ms | **4.37×** |
| LateOn-Code `Nd=1024, Ld=2048`                   | 14.15 ms | 3.05 ms | **4.64×** |
| LateOn-Code-edge `Nd=256, Ld=4096, d_model=384`  | 5.13 ms  | 1.32 ms | **3.88×** |

Design details and the closed-form gradient derivation:
[`docs/design.md` → Fused heads](docs/design.md#fused-heads-v06).

---

## API reference

Most users only need `patch_pylate()`, `MaxSimScorer` or `retrieve`.
Everything else is for when you're writing a custom trainer,
reranker, or research pipeline.

### High-level (recommended)

| Symbol                                       | What it does                                                                 |
| -------------------------------------------- | ---------------------------------------------------------------------------- |
| `patch_pylate()` / `unpatch_pylate()`        | One-line PyLate drop-in. Numerically identical, `LIK_DISABLE=1` kill-switch. |
| `MaxSimScorer(normalize=True, backward=...)` | `nn.Module` scoring layer. Autograd-aware, composes with any encoder.        |
| `retrieve(Q, D, top_k, chunk=...)`           | Top-k retrieval in one call with optional chunking for huge corpora.         |

### Core scoring

| Function                                       | What it does                                                                 |
| ---------------------------------------------- | ---------------------------------------------------------------------------- |
| `maxsim(Q, D, q_mask, d_mask, *, backward=)`   | Autograd-aware MaxSim. Used by `patch_pylate()`; per-call `backward=` kwarg. |
| `maxsim_inference(Q, D, ...)`                  | Forward-only MaxSim (skips argmax save). Use at inference.                   |
| `maxsim_varlen(Qp, Dp, cu_q, cu_d)`            | Packed-layout MaxSim, autograd-aware. Auto-skips argmax save when no grad. See [packed training cookbook](docs/packed_training.md). |
| `maxsim_topk(Q, D, k=10, chunk=...)`           | MaxSim + fused `top_k`, with optional chunking for huge `Nd`.                |
| `maxsim_matryoshka(Q, D, dims=[...])`          | Score at K truncated dimensions in one kernel call.                          |
| `maxsim_xtr(Q, D, top_k=5)`                    | XTR-style: sum of top-K doc tokens per query token.                          |
| `soft_maxsim(Q, D, beta=10)`                   | Log-sum-exp relaxation. `β → ∞` recovers hard MaxSim.                        |
| `smooth_maxsim(Q, D, top_k=4, ...)`            | Top-K aggregate (denser grads, streaming Triton top-K).                      |

### Fused D-side head (for custom trainers with raw hidden states)

`patch_pylate()` does **not** install these — PyLate already stores
*projected, normalized* token embeddings. Use these when your training
or inference loop keeps raw `[Nd, Ld, d_model]` ModernBERT / encoder
hidden states and applies the projection + normalize yourself.

**Which API do I pick?**

- `[Nd, Ld, d_out]` already projected + normalized (PyLate, sentence-transformers,
  most reranking code) → use `MaxSimScorer` / `maxsim` / `maxsim_inference`.
- `[Nd, Ld, d_model]` raw encoder hidden states + a separate `Dense` projection
  → use `maxsim_from_hidden(_train)`; it fuses `F.linear + normalize + maxsim`
  into one kernel and never materializes the `[Nd, Ld, d_out]` scratch.

Only the **D side** is fused. `Q_proj` is expected already projected + normalized
(query-side scratch is `[Nq, Lq, d_out]` — small, so not worth fusing). The
asymmetry is deliberate.

| Function                                  | When to use                                           |
| ----------------------------------------- | ----------------------------------------------------- |
| `maxsim_from_hidden(Q_proj, H_d, W, b)`   | **Inference.** Fuses `H_d @ W.T + b → normalize → maxsim` in one kernel; `[Nd, Ld, d_out]` scratch is never materialized. |
| `maxsim_from_hidden_train(Q_proj, H_d, W, b)` | **Training (autograd-aware).** Same forward + closed-form backward. Gradients flow to `Q_proj`, `H_d`, `W`, `b`. |

The two functions mirror `maxsim_inference` / `maxsim` — inference skips
the argmax save, training keeps it. Example:

```python
from late_interaction_kernels import maxsim_from_hidden_train

H_d.requires_grad_(True)
W.requires_grad_(True)
scores = maxsim_from_hidden_train(Q_proj, H_d, W, b=b, normalize=True)
scores.sum().backward()
```

**Why the backward is fast.** Forward saves which doc token won MaxSim per
query token (the argmax), so backward only has to project `H_d` at the
`Nq · Nd · Lq` winning positions instead of all `Nd · Ld`. For LateOn-Code
training shapes (`Lq=32, Ld=2048`), that's ≈1.5 % of the matmul work —
where the 4.6× fwd+bwd speedup comes from. The win shrinks as `Nq · Lq /
Ld` grows toward 1: roughly square, pure-reranking shapes (`Nq · Lq ≈ Ld`)
see little benefit, and small-`d_model` shapes (`<128`) see little benefit
because the `F.linear` pass is already cheap.

Full derivation + memory accounting:
[`docs/design.md` → Fused heads](docs/design.md#fused-heads-v06).

### ColBERTv2 (compressed index)

| Function                                                      | What it does                                                    |
| ------------------------------------------------------------- | --------------------------------------------------------------- |
| `plaid_approx_score(q_cent, codes, doc_lens)`                 | IVF prune step: score queries vs centroid codes.                |
| `maxsim_residual(Q, codes, residuals, ..., nbits=2\|4\|8)`    | Fused decompress + normalize + MaxSim. Autograd on `Q`.         |
| `maxsim_residual_inference(...)`                              | Same, skips argmax save.                                        |

Useful for building a pure-Python ColBERTv2 reranker or fine-tuning
the query encoder directly against a compressed PLAID index.

### FP8 inference (Hopper / Blackwell)

```python
from late_interaction_kernels import maxsim_inference_fp8, quantize_fp8_per_token

Q_fp8, sQ = quantize_fp8_per_token(Q)
D_fp8, sD = quantize_fp8_per_token(D)
scores = maxsim_inference_fp8(Q_fp8, D_fp8, scale_Q=sQ, scale_D=sD)
```

Auto-falls back to dequantized bf16 on pre-Hopper GPUs. Relative error
≤ 0.2 % on normalized embeddings.

### Knobs

| Call                                                               | What it controls                                                                                        |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| `maxsim(..., backward="auto" \| "unified" \| "atomic" \| "csr")`   | **Per-call** `grad_D` strategy — safe to mix across experiments. `"auto"` is the default.               |
| `set_backward_method(method)` / `get_backward_method()`            | Process-wide default (kept for back-compat; prefer the per-call kwarg).                                 |
| `LIK_DISABLE=1` (env, checked per call)                            | Patched entry points delegate to vanilla PyLate. Runtime kill switch; no restart needed.                |
| `LIK_SUPPRESS_NORM_WARN=1` (env)                                   | Silence the one-shot warning when `maxsim(..., normalize=False)` gets a clearly-unnormalized Q.         |

Details on backward-method selection:
[`docs/design.md` → Backward](docs/design.md#backward).

---

## Hardware support

| GPU family                  | Status         | Notes                                            |
| --------------------------- | -------------- | ------------------------------------------------ |
| H100 / H200 (Hopper)        | primary target | autotuned, FP8 WGMMA, warp-spec (Triton ≥ 3.2)   |
| A100 (Ampere 80 GB)         | supported      | separate autotune shortlist                      |
| L4, L40, RTX 4090 (Ada)     | supported      | generic shortlist                                |
| A10, A40, RTX 3090 (Ampere) | supported      | generic shortlist                                |
| Older / unknown CUDA        | works          | conservative default configs                     |
| CPU / macOS / Windows       | reference only | pure-PyTorch `late_interaction_kernels.reference` |

Autotune runs once per unique `(Lq, Ld, d, masks)` signature and caches
the winner — zero overhead after warmup.

---

## Design

The forward tiles `Q` and `D` into SRAM and streams
`matmul → mask → max → running-sum`, never writing the
`[Nq · Nd · Lq · Ld]` similarity tensor to HBM. The backward either
recomputes the argmax and uses `atomic_add` (fast for moderate shapes)
or reuses a saved argmax via a CSR-style bucket reduction
(deterministic, wins at long sequences). On top of that, v0.6+ adds a
unified single-pass backward, fused D-side heads (inference + training,
closed-form backward in 0.8.0), FP8 inference, smooth top-K, and
Triton-3.2 warp specialization.

Full walk-through — tiling, mask semantics, argmax save, backward
variants, fused heads, numerics: [`docs/design.md`](docs/design.md).

---

## Development

```bash
pip install -e ".[dev,pylate]"
pytest tests/ -q                   # auto-skips CUDA tests without a GPU
ruff check . && ruff format --check .

# benchmarks (single H100)
python benchmarks/bench_forward.py
python benchmarks/bench_pylate_lateon.py   # real PyLate training step
```

Contributions welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md). Past
releases and full kernel-by-kernel history: [`CHANGELOG.md`](CHANGELOG.md).

---

## Citation

```bibtex
@software{late_interaction_kernels_2026,
  author  = {Lac, Aurélien and Wu, Tony},
  title   = {{late-interaction-kernels}: Fused Triton kernels for late-interaction scoring},
  year    = {2026},
  url     = {https://github.com/hcompai/late-interaction-kernels},
}
```

## Authors

**Aurélien Lac** · **Tony Wu** — H Company · 2026

## Related projects

- [**PyLate**](https://github.com/lightonai/pylate) — training framework
  for late-interaction models. `patch_pylate()` hooks directly into it.
- [**FastPlaid**](https://github.com/lightonai/fast-plaid) — Rust /
  `tch-rs` multi-vector search engine. Different category: end-to-end
  retrieval engine vs kernel library. On the shared `bmm → mask → max
  → sum` scoring step the fused kernel is **1.55–3.56×** faster at
  `Ld ∈ {512, 1k, 4k, 8k}` ([numbers](docs/benchmarks.md#kernel-level-microbenchmark-on-the-matmul--mask--max--sum-pattern)).
- [**NextPlaid / ColGrep**](https://github.com/lightonai/next-plaid) —
  LightOn's Rust CLI built on FastPlaid, optimized for on-disk
  code-search indexes.
- [**flash-maxsim**](https://github.com/roipony/flash-maxsim) (IBM) —
  the first public Triton MaxSim kernel. Direct inspiration;
  `late-interaction-kernels` extends it with fused masking, varlen,
  backward, FP8, smooth / soft variants, ColBERTv2 kernels, and the
  PyLate drop-in.
- [**FlashAttention**](https://github.com/Dao-AILab/flash-attention) —
  the IO-aware tiling pattern this kernel is a strict subset of.

## License

Apache 2.0 — see [`LICENSE`](LICENSE). Copyright 2026 Aurélien Lac
and Tony Wu.
