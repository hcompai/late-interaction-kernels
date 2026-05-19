# late-interaction-kernels

[![CI](https://github.com/hcompai/late-interaction-kernels/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/hcompai/late-interaction-kernels/actions/workflows/ci.yml)
[![Version](https://img.shields.io/pypi/v/late-interaction-kernels?color=%2334D058&label=pypi%20package)](https://pypi.org/project/late-interaction-kernels/)
[![Downloads](https://static.pepy.tech/badge/late-interaction-kernels)](https://pepy.tech/project/late-interaction-kernels)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

---

[[Benchmarks]](docs/benchmarks.md)
[[Design]](docs/design.md)
[[Models]](docs/supported_models.md)
[[Changelog]](CHANGELOG.md)

Fused Triton kernels for MaxSim — the late-interaction scoring at the heart of ColBERT, ColPali, ModernColBERT, LateOn and ColBERTv2. Numerically identical to plain PyTorch, with an `nn.Module`, a function-level API, and a one-line PyLate drop-in. This is **not** a search engine; for end-to-end retrieval use [PyLate](https://github.com/lightonai/pylate), [FastPlaid](https://github.com/lightonai/fast-plaid) or [NextPlaid](https://github.com/lightonai/next-plaid) — this library is the MaxSim math they compile down to.

## Install

```bash
uv add late-interaction-kernels       # or: pip install late-interaction-kernels
```

Linux + CUDA (sm_75+) hits the fused Triton path. macOS (Apple Silicon, MPS) uses a fused Metal `simdgroup_matrix` kernel for inference and `torch.compile` for training. CPU / Windows fall back to an autograd-aware pure-PyTorch reference, so training and retrieval code is unit-testable on a laptop. The PyLate drop-in targets PyLate ≥ 1.3.

## Quickstart

The one-liner — speed up PyLate without touching your code:

```python
from late_interaction_kernels import patch_pylate

patch_pylate()
```

Set `LIK_DISABLE=1` in the environment to fall back to vanilla PyLate at runtime.

Custom training loop:

```python
from late_interaction_kernels import MaxSimScorer

scorer = MaxSimScorer(normalize=True)                # nn.Module, no parameters
scores = scorer(Q, D, q_mask=q_mask, d_mask=d_mask)  # [Nq, Nd] fp32
scores.mean().backward()
```

Top-k retrieval over a corpus:

```python
from late_interaction_kernels import retrieve

scores, indices = retrieve(Q, D, top_k=100, chunk=4096)
# both [Nq, 100] — chunk= bounds peak HBM at Nq · (chunk + top_k)
```

PLAID / ColBERTv2 rerank on compressed, ragged docs lives in `late_interaction_kernels.plaid`.

## Benchmarks

1×H100, bf16/fp16, 50-iter median, vs the same op in plain PyTorch:

| Workload                                           | Speedup           |
| -------------------------------------------------- | ----------------- |
| Reranking / inference vs naive einsum              | 7–23×             |
| Long-context (`Ld ≥ 8k`) reranking                 | runs; naive OOMs  |
| PyLate cached-contrastive MaxSim + backward        | up to 13.8×       |
| PLAID rerank vs `fast_plaid.engine.search()`       | 19–30×            |
| Fused D-side head (training)                       | 1.5–4.6×          |
| FP8 MaxSim inference (Hopper)                      | up to 1.4×        |
| End-to-end training of a 149 M encoder             | 1.00–1.06× (free) |

Wherever MaxSim stops being negligible — inference, reranking, long docs, big effective batch, KD, compressed indices, small encoders — the fused path moves. Full tables and reproduction commands: [`docs/benchmarks.md`](docs/benchmarks.md).

## API

Three symbols cover most use cases:

| Symbol                                | What it does                                                          |
| ------------------------------------- | --------------------------------------------------------------------- |
| `patch_pylate()` / `unpatch_pylate()` | One-line PyLate drop-in. `LIK_DISABLE=1` kill switch.                 |
| `MaxSimScorer(normalize=, backward=)` | Stateless `nn.Module`, autograd-aware.                                |
| `retrieve(Q, D, top_k, chunk=)`       | Top-k retrieval, chunked for huge corpora.                            |

Function-level entry points (`maxsim`, `maxsim_inference`, `maxsim_varlen`, `maxsim_padded`) sit alongside these on the top-level package. Niche kernels live in submodules: `padded`, `score_pairs`, `fused_head`, `plaid`, `fp8`, `experimental`, `reference`. Walk-through of every kernel, the autograd graph, the backward variants and the numerics: [`docs/design.md`](docs/design.md).

<details>
<summary>Configuration knobs (env vars + kwargs)</summary>

| Knob                                                              | Effect                                                            |
| ----------------------------------------------------------------- | ----------------------------------------------------------------- |
| `maxsim(..., backward="auto" \| "unified" \| "atomic" \| "csr")`  | Per-call `grad_D` strategy. `"auto"` picks per shape.             |
| `set_backward_method(...)` / `get_backward_method()`              | Process-wide default — `late_interaction_kernels.autograd`.       |
| `LIK_DISABLE=1`                                                   | Patched entry points delegate to vanilla PyLate.                  |
| `LIK_SUPPRESS_NORM_WARN=1`                                        | Silence the "looks unnormalized" one-shot warning.                |
| `LIK_DISABLE_COMPILE=1`                                           | Skip `torch.compile` on the MPS path (eager fallback).            |
| `LIK_FORCE_MPS_BACKEND={metal,compile,reference}`                 | Pin the MPS dispatch (default: heuristic on shape).               |

</details>

<details>
<summary>Hardware support &amp; autotune details</summary>

Primary target: **H100 / H200** (autotuned, FP8 WGMMA, warp-specialized on Triton ≥ 3.2). Also tuned for **A100**, **Ada** (L4 / L40 / 4090) and **Ampere** (A10 / A40 / 3090). Older / unknown CUDA falls back to a conservative shortlist.

**Apple Silicon (MPS)** ships two paths and picks per call: a fused Metal `simdgroup_matrix` kernel (forward-only, 1.9–3.2× over plain PyTorch on realistic inference shapes, ~300× less peak memory because it never materialises `[Nq · Nd · Lq · Ld]`), and a `torch.compile`-fused reference for training-time calls and small-batch inference where the Metal launch overhead doesn't amortise.

Autotune runs once per unique `(Lq, Ld, d, masks)` signature on CUDA and caches the winner; the MPS compile cache keys on `(dtype, normalize, has_q_mask, has_d_mask)` and amortises after the first call.

</details>

## Development

```bash
git clone https://github.com/hcompai/late-interaction-kernels
cd late-interaction-kernels
uv sync --extra dev --extra pylate           # CPU torch by default; see note
uv run pytest -q                             # CUDA tests auto-skip without a GPU
uv run ruff check . && uv run ruff format --check .
```

`uv sync` installs CPU-only torch (per `[tool.uv.sources]` in `pyproject.toml`) so CI doesn't pull the multi-GB CUDA wheel. For local GPU dev:

```bash
UV_INDEX=https://download.pytorch.org/whl/cu124 uv sync --extra dev --extra pylate
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the contribution workflow.

## Citation

```bibtex
@software{late_interaction_kernels_2026,
  author  = {Lac, Aurélien and Wu, Tony},
  title   = {{late-interaction-kernels}: Fused Triton kernels for late-interaction scoring},
  year    = {2026},
  url     = {https://github.com/hcompai/late-interaction-kernels},
}
```

Aurélien Lac · Tony Wu — H Company · Apache 2.0 (see [`LICENSE`](LICENSE)). Direct inspiration: [flash-maxsim](https://github.com/roipony/flash-maxsim) (IBM, the first public Triton MaxSim) and [FlashAttention](https://github.com/Dao-AILab/flash-attention).
