<div align="center">

# late-interaction-kernels

<img src="assets/banner.webp" alt="late-interaction-kernels banner" />

[![ColBERT](https://img.shields.io/badge/ColBERT-2004.12832-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2004.12832)
[![PyLate](https://img.shields.io/badge/PyLate-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/lightonai/pylate)
[![colpali-engine](https://img.shields.io/badge/colpali--engine-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/illuin-tech/colpali)
[![Hugging Face](https://img.shields.io/badge/Hcompany-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/Hcompany)

[![CI](https://github.com/hcompai/late-interaction-kernels/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/hcompai/late-interaction-kernels/actions/workflows/ci.yml)
[![Version](https://img.shields.io/pypi/v/late-interaction-kernels?color=%2334D058&label=pypi%20package)](https://pypi.org/project/late-interaction-kernels/)
[![Downloads](https://static.pepy.tech/badge/late-interaction-kernels)](https://pepy.tech/project/late-interaction-kernels)

---

[[How it works]](https://hcompai.github.io/late-interaction-kernels/how-it-works.html)
[[Kernel picker]](https://hcompai.github.io/late-interaction-kernels/choose-a-kernel.html)
[[Benchmarks]](docs/benchmarks.md)
[[Design]](docs/design.md)
[[Supported models]](docs/supported_models.md)
[[Changelog]](CHANGELOG.md)

</div>

> [!NOTE]
> Full algorithmic walkthrough, animations and benchmark plots live on the docs site: **[hcompai.github.io/late-interaction-kernels](https://hcompai.github.io/late-interaction-kernels/how-it-works.html)**.

## Introduction

`late-interaction-kernels` provides fused Triton kernels for **MaxSim**, the late-interaction scoring used by ColBERT, ColPali, ModernColBERT, LateOn and ColBERTv2. The kernels are numerically identical to plain PyTorch and come with three APIs:

- a one-line PyLate drop-in (`patch_pylate()`),
- a stateless `nn.Module` (`MaxSimScorer`) for custom training loops,
- function-level entry points (`maxsim`, `maxsim_varlen`, `maxsim_padded`, ...) for everything else.

This is **not** a search engine. For end-to-end training or retrieval use [PyLate](https://github.com/lightonai/pylate), [FastPlaid](https://github.com/lightonai/fast-plaid) or [NextPlaid](https://github.com/lightonai/next-plaid). This library is the MaxSim math they compile down to.

## Install

```bash
pip install late-interaction-kernels
```

| Platform                       | Backend                                                                       |
| ------------------------------ | ----------------------------------------------------------------------------- |
| Linux + CUDA (sm_75+)          | Fused Triton kernels (autotuned, FP8 on Hopper).                              |
| macOS (Apple Silicon, MPS)     | Fused Metal `simdgroup_matrix` for inference, `torch.compile` for training.   |
| CPU / Windows                  | Autograd-aware pure-PyTorch reference.                                        |

## Quickstart

### Patch PyLate (one line)

```python
from late_interaction_kernels import patch_pylate

patch_pylate()
# PyLate training / rerank code is unchanged
```

Set `LIK_DISABLE=1` in the environment to fall back to vanilla PyLate at runtime.

### Custom training loop

```python
from late_interaction_kernels import MaxSimScorer

scorer = MaxSimScorer(normalize=True)                # nn.Module, no parameters
scores = scorer(Q, D, q_mask=q_mask, d_mask=d_mask)  # [Nq, Nd] fp32
scores.mean().backward()
```

### Top-k retrieval

```python
from late_interaction_kernels import retrieve

scores, indices = retrieve(Q, D, top_k=100, chunk=4096)
# both [Nq, 100]; chunk= bounds peak HBM at Nq * (chunk + top_k)
```

### PLAID / ColBERTv2 on compressed, ragged docs

```python
from late_interaction_kernels.plaid import maxsim_residual_varlen

scores = maxsim_residual_varlen(
    Q, codes_flat, residuals_flat, cu_seqlens_d,
    centroids=centroids, bucket_weights=bucket_weights,
    nbits=2, normalize=True,
)  # [Nd] fp32; one kernel does decompress + L2-normalize + MaxSim
```

## Benchmarks

1×H100 80GB SXM, bf16 inputs / fp32 accumulator, 50-iter median. All
speedups are measured at **matched numerics** — every baseline runs the
einsum with an fp32 accumulator (same as the fused kernel), and parity
is asserted at `atol=1e-2` before timing.

| Workload                                                    | Speedup            |
| ----------------------------------------------------------- | ------------------ |
| Reranking / inference (vs eager fp32-acc *and* `torch.compile`) | 2-11×          |
| Long-context (`Ld ≥ 8k`) MaxSim fwd+bwd                     | runs; naive OOMs   |
| PyLate cached-contrastive MaxSim + backward (vs vanilla)    | 4.0-5.5×           |
| PLAID rerank vs `fast_plaid.engine.search()` (incl. top-k)  | 19-32×             |
| Fused D-side head (training)                                | 1.2-4.2×           |
| FP8 MaxSim inference vs same kernel in bf16 (Hopper)        | 1.9-2.5×           |
| LateOn-Code-edge training (real MS MARCO triplets)          | 1.05-1.27× e2e     |

`torch.compile` is within ±5% of eager on every forward shape because
Inductor still has to materialise the `[Nq · Nd · Lq · Ld]` similarity
tensor before the `max(-1)` reduction — that materialisation *is* what
the fused kernel exists to skip. Full tables and reproduction commands:
[`docs/benchmarks.md`](docs/benchmarks.md).

## Choose a kernel

<table>
<tr>
<td width="55%" valign="middle">

Not sure which entry point fits your stack? The docs site ships an interactive decision tree that narrows the public API down to the right function in four questions (stack · phase · layout · workload):

**👉 [hcompai.github.io/late-interaction-kernels/choose-a-kernel.html](https://hcompai.github.io/late-interaction-kernels/choose-a-kernel.html#choose-a-kernel)**

</td>
<td width="45%" valign="middle" align="center">

<a href="https://hcompai.github.io/late-interaction-kernels/choose-a-kernel.html#choose-a-kernel">
  <img src="assets/kernel_picker_widget_preview.webp" alt="Pick a kernel · interactive decision tree" width="420">
</a>

</td>
</tr>
</table>

## API

| Symbol                                | What it does                                                          |
| ------------------------------------- | --------------------------------------------------------------------- |
| `patch_pylate()` / `unpatch_pylate()` | One-line PyLate drop-in. `LIK_DISABLE=1` kill switch.                 |
| `MaxSimScorer(normalize=, backward=)` | Stateless `nn.Module`, autograd-aware.                                |
| `retrieve(Q, D, top_k, chunk=)`       | Top-k retrieval, chunked for huge corpora.                            |
| `maxsim`                              | Core MaxSim. Dispatches on `D.dim()`: 3D → in-batch `[Nq, Nd]`, 4D → per-query KD candidates `[Nq, K]` (one fused launch, no Python loop). Autograd-aware. |
| `maxsim_varlen`                       | Packed (`cu_seqlens`) layout. Autograd-aware.                         |
| `maxsim_padded`                       | Padded reranking wrapper: packs internally, returns `[B, C]` fp32.    |

Other kernels are in submodules: `padded`, `score_pairs`, `fused_head`, `plaid`, `fp8`, `reference`. See [`docs/design.md`](docs/design.md) for details on every kernel, the autograd graph and the backward variants.

<details>
<summary><strong>🔽 Configuration knobs (env vars + kwargs)</strong></summary>

| Knob                                                              | Effect                                                            |
| ----------------------------------------------------------------- | ----------------------------------------------------------------- |
| `maxsim(..., backward="auto" \| "unified" \| "atomic" \| "csr")`  | Per-call `grad_D` strategy. `"auto"` picks per shape.             |
| `patch_colpali_engine()` / `unpatch_colpali_engine()`             | colpali_engine drop-in: loss + scoring routes through the kernel. |
| `LIK_DISABLE=1`                                                   | Patched entry points delegate to vanilla PyLate.                  |
| `LIK_SUPPRESS_NORM_WARN=1`                                        | Silence the "looks unnormalized" one-shot warning.                |
| `LIK_DISABLE_COMPILE=1`                                           | Skip `torch.compile` on the MPS path (eager fallback).            |
| `LIK_FORCE_MPS_BACKEND={metal,compile,reference}`                 | Pin the MPS dispatch.                                             |

</details>

## Development

```bash
git clone https://github.com/hcompai/late-interaction-kernels
cd late-interaction-kernels
uv sync --extra dev --extra pylate --extra torch-cuda   # GPU dev; use --extra torch-cpu on CPU-only boxes
uv run pytest -q                                        # CUDA tests auto-skip without a GPU
uv run ruff check . && uv run ruff format --check .
```

> [!NOTE]
> Pick exactly one of `--extra torch-cuda` (pulls torch from the CUDA index — `cu124`) or `--extra torch-cpu` (CPU-only wheel, what CI uses). The two are declared as conflicting in `pyproject.toml` so the lockfile resolves cleanly for both. On macOS, `--extra torch-cpu` falls back to PyPI's default (MPS-capable) wheel automatically.

GPU tests run automatically on every push to `main`. To run them on a PR, apply the `run-gpu-tests` label.

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
