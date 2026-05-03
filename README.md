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

Drop-in, numerically-identical replacements for the MaxSim math behind
late-interaction training, reranking and retrieval. One line patches
PyLate; the rest is `nn.Module` and function-level APIs for custom
pipelines.

> Not a search engine. For end-to-end retrieval use
> [FastPlaid](https://github.com/lightonai/fast-plaid),
> [NextPlaid / ColGrep](https://github.com/lightonai/next-plaid),
> or [PyLate](https://github.com/lightonai/pylate).
> This library is the MaxSim math their reranking / training compiles down to.

---

## Install

```bash
pip install late-interaction-kernels
```

| Platform                          | Path                                                       |
| --------------------------------- | ---------------------------------------------------------- |
| Linux + CUDA (sm_75+)             | Fused Triton kernels — full speedups in [Speedups](#speedups-on-h100). |
| macOS (Apple Silicon, MPS)        | Fused Metal `simdgroup_matrix` kernel for inference, `torch.compile` for training. |
| CPU / Windows / anything else     | Eager pure-PyTorch reference, autograd-aware.              |

`MaxSimScorer`, `retrieve` and `late_interaction_kernels.reference`
import and run on every platform, so training and retrieval code is
unit-testable on a laptop before renting a GPU. The PyLate drop-in
targets PyLate ≥ 1.3.

---

## Quickstart

**Speed up PyLate, one line:**

```python
from late_interaction_kernels import patch_pylate

patch_pylate()
# PyLate training / rerank code is unchanged
```

`LIK_DISABLE=1` in the environment makes patched entry points fall back
to vanilla PyLate at runtime.

**Score MaxSim in any training loop:**

```python
from late_interaction_kernels import MaxSimScorer

scorer = MaxSimScorer(normalize=True)                # nn.Module, no parameters
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

# fast-plaid / ColBERTv2 on-disk layout
scores = maxsim_residual_varlen(
    Q, codes_flat, residuals_flat, cu_seqlens_d,
    centroids=centroids, bucket_weights=bucket_weights,
    nbits=2, normalize=True,
)  # [Nd] fp32 — one kernel does decompress + L2-normalize + MaxSim
```

---

## Speedups on H100

1×H100 80 GB SXM, bf16 / fp16 compute, fp32 accumulator, 50-iter median.
Every baseline is the same operation in plain PyTorch.

| Workload                                            | Speedup           |
| --------------------------------------------------- | ----------------- |
| Reranking / inference vs naive einsum               | **7–23×**         |
| Long-context (`Ld ≥ 8k`) reranking                  | runs; naive OOMs  |
| PyLate cached-contrastive MaxSim + backward         | up to **13.8×**   |
| PLAID rerank vs `fast_plaid.engine.search()`        | **19–30×**        |
| Fused D-side head (training)                        | **1.5–4.6×**      |
| FP8 MaxSim inference (Hopper)                       | up to **1.4×**    |
| LateOn-Code-edge end-to-end training (17 M)         | **1.04–1.27×**    |
| LateOn / ModernColBERT end-to-end training (149 M)  | 1.00–1.06× (free) |

ModernBERT-class encoders dominate step time, so on a full 149 M
training run the kernel is essentially a free swap. Wherever MaxSim
stops being negligible — inference, reranking, long docs, big effective
batch, KD, compressed indices, small encoders — the fused path moves.

Full tables, shapes and reproduction commands:
[`docs/benchmarks.md`](docs/benchmarks.md).

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
| `maxsim_inference_scatter`              | Pair-list reranking on packed batches (vLLM-style scheduling).            |
| `maxsim_from_hidden(_train)`            | Fused D-side `Linear → Normalize → MaxSim`, no `[Nd, Ld, d_out]` scratch. |
| `maxsim_residual(_varlen)`              | Fused PLAID / ColBERTv2 decompress + normalize + MaxSim.                  |
| `plaid_approx_score`                    | IVF prune step for ColBERTv2.                                             |
| `maxsim_inference_fp8`                  | FP8 tensor-core MaxSim (Hopper / Blackwell). Auto-fallback to bf16.       |

Submodules:

- `late_interaction_kernels.fp8` — FP8 quantize / dequantize helpers
  (per-tensor and per-token).
- `late_interaction_kernels.experimental` — research kernels
  (`soft_maxsim`, `smooth_maxsim`, `maxsim_matryoshka`, `maxsim_xtr`).
- `late_interaction_kernels.reference` — pure-PyTorch reference
  implementations, importable on any platform.

Config:

| Knob                                                              | Effect                                                              |
| ----------------------------------------------------------------- | ------------------------------------------------------------------- |
| `maxsim(..., backward="auto" \| "unified" \| "atomic" \| "csr")`  | Per-call `grad_D` strategy. `"auto"` picks per shape.               |
| `set_backward_method(...)` / `get_backward_method()`              | Process-wide default (back-compat; prefer per-call kwarg).          |
| `LIK_DISABLE=1`                                                   | Patched entry points delegate to vanilla PyLate.                    |
| `LIK_SUPPRESS_NORM_WARN=1`                                        | Silence the "looks unnormalized" one-shot warning.                  |
| `LIK_DISABLE_COMPILE=1`                                           | Skip `torch.compile` on the MPS path (eager fallback).              |
| `LIK_FORCE_MPS_BACKEND={metal,compile,reference}`                 | Pin the MPS dispatch (default: heuristic on shape).                 |

Walk-through of every kernel, the autograd graph, the backward variants
and the numerics: [`docs/design.md`](docs/design.md).

---

## Hardware

Primary target: **H100 / H200** (autotuned, FP8 WGMMA, warp-specialized
on Triton ≥ 3.2). Also tuned for **A100**, **Ada** (L4 / L40 / 4090) and
**Ampere** (A10 / A40 / 3090). Older / unknown CUDA falls back to a
conservative shortlist.

**Apple Silicon (MPS)** ships two paths and picks per call:

* a fused **Metal `simdgroup_matrix`** kernel (forward-only) — beats
  the compile path 1.1–2.0× on realistic inference shapes and uses
  ~300× less peak memory on big corpora because it never materialises
  `[Nq · Nd · Lq · Ld]`. Persistent threadgroups serve 8 consecutive
  `j`s per launch and keep `Q` register-resident across every
  `(j, d-chunk)`;
* a **`torch.compile`-fused** reference (autograd-aware) — carries
  every training-time call and small-batch inference where the Metal
  kernel's launch overhead doesn't amortise.

See [`docs/benchmarks.md`](docs/benchmarks.md#apple-silicon-mps) for
shapes and numbers.

Autotune runs once per unique `(Lq, Ld, d, masks)` signature on CUDA
and caches the winner; the MPS compile cache keys on `(dtype,
normalize, has_q_mask, has_d_mask)` and amortises after the first
call.

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

[`CONTRIBUTING.md`](CONTRIBUTING.md) for the contribution workflow.
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
