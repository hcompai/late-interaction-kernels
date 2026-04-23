<div align="center">

# ⚡ late-interaction-kernels

**Fused Triton kernels for MaxSim scoring.**
ColBERT · ColPali · ModernColBERT · LateOn · LateOn-Code · ColBERTv2 · PyLate-native.

[![CI](https://github.com/hcompai/late-interaction-kernels/actions/workflows/ci.yml/badge.svg)](https://github.com/hcompai/late-interaction-kernels/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9–3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A5%202.1-ee4c2c.svg)](https://pytorch.org/)
[![Triton](https://img.shields.io/badge/Triton-%E2%89%A5%203.0-9146ff.svg)](https://github.com/triton-lang/triton)
[![PyLate](https://img.shields.io/badge/PyLate-%E2%89%A5%201.3-00b4d8.svg)](https://github.com/lightonai/pylate)

[Install](#install) · [Quickstart](#quickstart) · [Speedups](#speedups-on-h100) · [API](#api) · [Benchmarks](docs/benchmarks.md) · [Design](docs/design.md) · [Models](docs/supported_models.md)

</div>

---

Drop-in, numerically-identical replacements for the MaxSim math that
late-interaction models compute during training, reranking and retrieval.
One line patches PyLate; the rest is `nn.Module` and function-level APIs
for custom pipelines.

> Not a search engine. For end-to-end retrieval use
> [FastPlaid](https://github.com/lightonai/fast-plaid),
> [NextPlaid / ColGrep](https://github.com/lightonai/next-plaid)
> or [PyLate](https://github.com/lightonai/pylate).
> This library is the MaxSim math their reranking / training steps compile down to.

---

## Install

```bash
pip install late-interaction-kernels
```

Linux + CUDA for the fused kernels. CPU / macOS / Windows still works —
`MaxSimScorer`, `retrieve` and `late_interaction_kernels.reference` fall
back to a pure-PyTorch implementation, so you can develop and unit-test
training / retrieval code locally before renting a GPU.

PyLate drop-in targets PyLate ≥ 1.3.

---

## Quickstart

**Speed up PyLate — one line:**

```python
from late_interaction_kernels import patch_pylate

patch_pylate()
# ... your PyLate training / rerank code is unchanged ...
```

`LIK_DISABLE=1` in the environment makes patched entry points fall back to
vanilla PyLate at runtime — no restart needed.

**Score MaxSim in any training loop:**

```python
from late_interaction_kernels import MaxSimScorer

scorer = MaxSimScorer(normalize=True)            # nn.Module, no parameters
scores = scorer(Q, D, q_mask=q_mask, d_mask=d_mask)  # [Nq, Nd] fp32
scores.mean().backward()
```

**Top-k retrieval:**

```python
from late_interaction_kernels import retrieve

scores, indices = retrieve(Q, D, top_k=100, chunk=4096)
# both [Nq, 100] — chunk= bounds peak HBM at Nq · (chunk + top_k)
```

**PLAID / ColBERTv2 rerank on compressed, ragged docs:**

```python
from late_interaction_kernels import maxsim_residual_varlen

# codes_flat:    [total_tokens] uint8
# residuals_flat:[total_tokens, packed_dim] uint8
# cu_seqlens_d:  [Nd+1] int32 — fast-plaid / ColBERTv2 on-disk layout
scores = maxsim_residual_varlen(
    Q, codes_flat, residuals_flat, cu_seqlens_d,
    centroids=centroids, bucket_weights=bucket_weights,
    nbits=2, normalize=True,
)  # [Nd] fp32
```

One kernel does decompress + L2-normalize + MaxSim. No padded scratch.

---

## Speedups on H100

1×H100 80 GB SXM, bf16 / fp16 compute, fp32 accumulator, 50-iter median.
Every baseline is the same operation in plain PyTorch.

| Workload                                                | Speedup            |
| ------------------------------------------------------- | ------------------ |
| Reranking / inference vs naive einsum                   | **7–23×**          |
| Long-context (`Ld ≥ 8k`) reranking                      | runs; naive OOMs   |
| PyLate cached-contrastive MaxSim + backward             | up to **13.8×**    |
| PLAID rerank vs `fast_plaid.engine.search()`            | **19–30×**         |
| Fused D-side head (training)                            | **1.5–4.6×**       |
| FP8 MaxSim inference (Hopper)                           | up to **1.4×**     |
| LateOn-Code-edge end-to-end training (17 M)             | **1.04–1.27×**     |
| LateOn / ModernColBERT end-to-end training (149 M)      | 1.00–1.06× (free)  |

ModernBERT-class encoders dominate step time, so on a full 149 M training
run the kernel is essentially a free swap. Anywhere MaxSim stops being
negligible — inference, reranking, long docs, big effective batch, KD,
compressed indices, small encoders — the fused path moves.

Full tables, shapes and reproduction commands:
[**docs/benchmarks.md**](docs/benchmarks.md).

---

## API

Most users only need `patch_pylate()`, `MaxSimScorer` or `retrieve`.

| Symbol                                  | What it does                                                              |
| --------------------------------------- | ------------------------------------------------------------------------- |
| `patch_pylate()` / `unpatch_pylate()`   | One-line PyLate drop-in. `LIK_DISABLE=1` kill switch.                     |
| `MaxSimScorer(normalize=, backward=)`   | Stateless `nn.Module`, autograd-aware.                                    |
| `retrieve(Q, D, top_k, chunk=)`         | Top-k retrieval, chunked for huge corpora.                                |
| `maxsim` / `maxsim_inference`           | Core MaxSim, dense layout (autograd / forward-only).                      |
| `maxsim_varlen`                         | Packed (`cu_seqlens`) layout. Autograd-aware.                             |
| `maxsim_topk(Q, D, k=, chunk=)`         | MaxSim + top-k in one call.                                               |
| `maxsim_from_hidden(_train)`            | Fused D-side `Linear → Normalize → MaxSim`, no `[Nd, Ld, d_out]` scratch. |
| `maxsim_residual(_inference / _varlen)` | Fused PLAID / ColBERTv2 decompress + normalize + MaxSim.                  |
| `plaid_approx_score`                    | IVF prune step for ColBERTv2.                                             |
| `maxsim_inference_fp8`                  | FP8 tensor-core MaxSim (Hopper / Blackwell). Auto-fallback to bf16.       |

Submodules:

- `late_interaction_kernels.fp8` — FP8 quantize / dequantize helpers
  (per-tensor, per-token).
- `late_interaction_kernels.experimental` — research kernels
  (`soft_maxsim`, `smooth_maxsim`, `maxsim_matryoshka`, `maxsim_xtr`).
- `late_interaction_kernels.reference` — pure-PyTorch reference
  implementations, importable on any platform.

Config:

| Knob                                                   | Effect                                                              |
| ------------------------------------------------------ | ------------------------------------------------------------------- |
| `maxsim(..., backward="auto" \| "unified" \| "atomic" \| "csr")` | Per-call `grad_D` strategy. `"auto"` picks per shape.     |
| `set_backward_method(...)` / `get_backward_method()`   | Process-wide default (back-compat; prefer per-call kwarg).          |
| `LIK_DISABLE=1`                                        | Patched entry points delegate to vanilla PyLate.                    |
| `LIK_SUPPRESS_NORM_WARN=1`                             | Silence the "looks unnormalized" one-shot warning.                  |

Detailed walk-through of each kernel, the autograd graph, the backward
variants and the numerics: [**docs/design.md**](docs/design.md).

---

## Hardware

Primary target: **H100 / H200** (autotuned, FP8 WGMMA, warp-spec on
Triton ≥ 3.2).
Also tuned for **A100**, **Ada** (L4 / L40 / 4090) and **Ampere**
(A10 / A40 / 3090). Older / unknown CUDA falls back to conservative
default configs. CPU / macOS / Windows get the pure-PyTorch reference.

Autotune runs once per unique `(Lq, Ld, d, masks)` signature and caches
the winner — zero overhead after warmup.

---

## Development

```bash
git clone https://github.com/hcompai/late-interaction-kernels
cd late-interaction-kernels
pip install -e ".[dev,pylate]"

pytest -q                               # auto-skips CUDA tests without a GPU
ruff check . && ruff format --check .

python benchmarks/bench_forward.py      # see benchmarks/ for the full set
```

[`CONTRIBUTING.md`](CONTRIBUTING.md) for the contribution workflow,
[`CHANGELOG.md`](CHANGELOG.md) for the kernel-by-kernel release history.

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

**Aurélien Lac · Tony Wu** — H Company · 2026

## License

Apache 2.0 — see [`LICENSE`](LICENSE). Copyright 2026 Aurélien Lac and Tony Wu.

## Related

[PyLate](https://github.com/lightonai/pylate) ·
[FastPlaid](https://github.com/lightonai/fast-plaid) ·
[NextPlaid / ColGrep](https://github.com/lightonai/next-plaid) ·
[flash-maxsim](https://github.com/roipony/flash-maxsim) (IBM, the first
public Triton MaxSim — direct inspiration) ·
[FlashAttention](https://github.com/Dao-AILab/flash-attention).
