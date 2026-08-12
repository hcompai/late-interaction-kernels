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
[[Benchmarks]](docs/benchmarks.md)
[[Design]](docs/design.md)
[[Changelog]](CHANGELOG.md)

</div>

<table>
<tr>
<td width="55%" valign="middle">

The docs explain tiling, online max, and the backward pass. They also include animations and benchmark plots:

**👉 [hcompai.github.io/late-interaction-kernels](https://hcompai.github.io/late-interaction-kernels/how-it-works.html)**

</td>
<td width="45%" valign="middle" align="center">

<a href="https://hcompai.github.io/late-interaction-kernels/how-it-works.html">
  <img src="assets/how_it_works_preview.webp" alt="How it works · design walkthrough preview" width="420">
</a>

</td>
</tr>
</table>

## Introduction

**MaxSim** is the scoring method used by ColBERT, ColPali, ModernColBERT, LateOn, and ColBERTv2. It compares every query token with every document token. A direct implementation creates a full `[Nq, Nd, Lq, Ld]` similarity tensor in GPU memory before reducing it. That tensor can exceed available memory at large batch sizes, and reading and writing it slows the GPU.

`late-interaction-kernels` provides fused Triton and Metal kernels that produce the same scores without writing that tensor. Each kernel calculates similarities, takes the maximum, and optionally L2-normalizes the inputs in one launch. The results match plain PyTorch, use less GPU memory, and run faster.

[PyLate](https://github.com/lightonai/pylate) and [colpali-engine](https://github.com/illuin-tech/colpali) support the kernels natively. Install their optional extra and `auto` dispatch uses them without code changes. You can also call `MaxSimScorer` in a custom training loop, or call functions such as `maxsim`, `maxsim_varlen`, and `maxsim_padded` directly.

## Install

```bash
pip install late-interaction-kernels
```

| Platform                       | Backend                                                                       |
| ------------------------------ | ----------------------------------------------------------------------------- |
| Linux + CUDA (sm_75+)          | Fused Triton kernels (autotuned, FP8 on Hopper/Blackwell).                    |
| macOS (Apple Silicon, MPS)     | Fused Metal `simdgroup_matrix` kernels for inference and training (fp16 / bf16, `d ≤ 128`); `torch.compile` fallback otherwise. |
| CPU / Windows                  | Autograd-aware pure-PyTorch reference.                                        |

## Quickstart

### Score directly (`maxsim` / `maxsim_pairs`)

`maxsim` is the lowest-level public function. It supports autograd and masks, and selects the layout from `D.dim()`. The same call handles in-batch scoring and knowledge distillation in one fused launch. When neither input needs gradients, it skips the argmax buffer used by the backward pass.

```python
from late_interaction_kernels import maxsim, maxsim_pairs

# in-batch:  Q[Nq, Lq, d] × D[Nd, Ld, d]    → [Nq, Nd]
scores = maxsim(Q, D, q_mask=q_mask, d_mask=d_mask, normalize=True)

# KD / hard-negative:  D is 4D [Nq, K, Ld, d]  → [Nq, K]   (one launch, no Python loop)
scores = maxsim(Q, D_kd, q_mask=q_mask, d_mask=d_mask_kd)

# pairwise (diagonal):  Q[B, Lq, d] × D[B, Ld, d]  → [B]
scores = maxsim_pairs(Q, D, q_mask=q_mask, d_mask=d_mask)
```

### PyLate & colpali-engine

Both libraries include a native LIK backend. Install the optional extra and their `auto` dispatch selects it without code changes. Set `PYLATE_SCORES_BACKEND=lik` or `COLPALI_SCORES_BACKEND=lik` to select it explicitly. For older versions, the `patch_*` functions replace scoring and loss at import time. Set `LIK_DISABLE=1` to use the original implementation.

<table>
<tr>
<td width="50%" valign="top">

**PyLate ≥ 1.5.1**

```bash
pip install "pylate[lik]"
```

PyLate < 1.5.1:

```python
import late_interaction_kernels as lik
lik.patch_pylate()
```

</td>
<td width="50%" valign="top">

**colpali-engine ≥ 0.3.17**

```bash
pip install "colpali-engine[lik]"
```

colpali-engine < 0.3.17:

```python
import late_interaction_kernels as lik
lik.patch_colpali_engine()
```

</td>
</tr>
</table>

### Top-k retrieval

Score `Q` against a large corpus and return the top `k` documents for each query. `retrieve` does not create the full `[Nq, Nd]` score matrix. Set `chunk=` to process documents in tiles and limit peak GPU memory.

```python
from late_interaction_kernels import retrieve

scores, indices = retrieve(Q, D, top_k=100, chunk=4096)
# both [Nq, 100]; chunk= bounds peak HBM at Nq * (chunk + top_k)
```

<details>
<summary><strong>PLAID</strong>: compressed, ragged ColBERTv2 indexes</summary>

<br>

Use this for PLAID indexes, where documents use centroid codes and residuals with variable lengths. One kernel decompresses the data, L2-normalizes it, and calculates MaxSim. It never writes a decoded tensor to GPU memory.

```python
from late_interaction_kernels.plaid import maxsim_residual_varlen

scores = maxsim_residual_varlen(
    Q, codes_flat, residuals_flat, cu_seqlens_d,
    centroids=centroids, bucket_weights=bucket_weights,
    nbits=2, normalize=True,
)  # [Nd] fp32; one kernel does decompress + L2-normalize + MaxSim
```

</details>

<details>
<summary><strong>Custom training loop</strong>: stateless <code>MaxSimScorer</code> module</summary>

<br>

`MaxSimScorer` is a stateless `nn.Module` wrapper around `maxsim`. Use it in a training loop when you need autograd-aware late-interaction scoring without PyLate.

```python
from late_interaction_kernels import MaxSimScorer

scorer = MaxSimScorer(normalize=True)                # nn.Module, no parameters
scores = scorer(Q, D, q_mask=q_mask, d_mask=d_mask)  # [Nq, Nd] fp32
scores.mean().backward()
```

</details>

## Benchmarks

Benchmarks use one 80 GB H100 SXM, bf16 inputs, an fp32 accumulator, and the median of 50 runs. Each baseline also uses an fp32 accumulator, so it has the same numeric precision as the fused kernel. The benchmark checks parity at `atol=1e-2` before timing.

### Speed

|             | Rerank /<br>inference | PyLate<br>cached-contrastive | PLAID rerank<br>vs `fast_plaid` | Fused D-head<br>(training) | FP8 vs bf16<br>(Hopper) | LateOn-Code-edge<br>e2e |
| ----------- | --------------------- | ---------------------------- | ------------------------------- | -------------------------- | ----------------------- | ----------------------- |
| **Speedup** | 1.7-16×               | 5.0-6.9×                     | 8-23× full<br>18-51× partial    | 0.94-4.5×                  | 1.1-1.3×                | 1.00-1.06×              |

Rerank compares against the eager fp32-accumulator path and `torch.compile`. PLAID rerank includes top k. The fused D-head is faster as `Nd * Ld` grows. It is slightly slower for the two smallest LateOn shapes at 0.94 to 0.95x, and at least 1.4x faster from `Nd=128, Ld=1024`, which includes every ColBERT and ColPali-scale shape. FP8 results use `Ld >= 256`.

[`docs/benchmarks.md`](docs/benchmarks.md) contains full tables and commands to reproduce them. [`benchmarks/README.md`](benchmarks/README.md) explains the benchmark scripts, including `--only`, `--variants`, and running a filtered set on a SkyPilot cluster.

### Memory

The naive einsum creates the full fp32 `[Nq * Nd * Lq * Ld]` similarity tensor before `max(-1)`. Its peak memory is larger than the tensor because fp32 copies of the inputs are also in memory. The fused kernel streams document tiles through SRAM and returns only `[Nq, Nd]` scores. Training also stores a `[Nq * Nd, Lq]` int32 argmax buffer.

| shape                                     | naive scratch | fused fwd | fused fwd + bwd |
| ----------------------------------------- | ------------- | --------- | --------------- |
| `Nq=1, Nd=1k, Lq=32, Ld=300`              | 183 MB        | 4 KB      | 128 KB          |
| `Nq=1, Nd=1k, Lq=128, Ld=1024` (ColPali)  | 1.0 GB        | 4 KB      | 512 KB          |
| `Nq=16, Nd=32, Lq=32, Ld=8192`            | 2.1 GB        | 2 KB      | 64 KB           |

The ColPali row assumes a short text query expanded to `Lq = 128`
(ColBERT-style query augmentation) against a `Ld ≈ 1024`-patch page.

The fused kernel supports long contexts with `Ld >= 8k` that exceed the memory available to the naive path. It also fits about 5 to 10 times more in-batch negatives within the same GPU memory budget.

In a ColQwen2 training run on an 80 GB H100, using LoRA, gradient checkpointing, and `vidore/colpali_train_set`, colpali-engine runs out of memory at `batch=128`. MaxSim uses 7.8 GiB, while the fused kernel uses 61 MiB. The fused kernel doubles the maximum batch size without increasing step time.

For shapes with large gradients, `auto` selects `lowmem`. That backward path writes `grad_Q` and `grad_D` in the input dtype. It avoids a full fp32 buffer and atomics, and it is deterministic. For example, it reduces the backward peak for a ColPali step with `B=256` and 16 negatives from 4.3 GB to 2.2 GB. See [`docs/benchmarks.md`](docs/benchmarks.md#memory) for the full tables.

## API

| Symbol                                                | What it does                                                          |
| ----------------------------------------------------- | --------------------------------------------------------------------- |
| `patch_pylate()` / `unpatch_pylate()`                 | One-line PyLate drop-in. `LIK_DISABLE=1` kill switch.                 |
| `patch_colpali_engine()` / `unpatch_colpali_engine()` | One-line colpali_engine drop-in (loss + scoring route through the kernel). |
| `MaxSimScorer(normalize=, backward=)`                 | Stateless `nn.Module`, autograd-aware.                                |
| `retrieve(Q, D, top_k, chunk=)`                       | Top-k retrieval, chunked for huge corpora.                            |
| `maxsim`                                              | Core MaxSim. Dispatches on `D.dim()`: 3D → in-batch `[Nq, Nd]`, 4D → per-query KD candidates `[Nq, K]` (one fused launch, no Python loop). Autograd-aware. |
| `maxsim_pairs`                                        | Diagonal pairs `Q[B, Lq, d] × D[B, Ld, d] → [B]`. K=1 case of the KD path; never builds the `[B, B]` cross product. Autograd-aware. |
| `maxsim_varlen`                                       | Packed (`cu_seqlens`) layout. Autograd-aware.                         |
| `maxsim_padded`                                       | Padded reranking wrapper: packs internally, returns `[B, C]` fp32.    |

Other kernels are in the `padded`, `score_pairs`, `fused_head`, `plaid`, `fp8`, and `reference` submodules. [`docs/design.md`](docs/design.md) explains each kernel, the autograd graph, and the backward options.

<details>
<summary><strong>🔽 Configuration knobs (env vars + kwargs)</strong></summary>

| Knob                                                              | Effect                                                            |
| ----------------------------------------------------------------- | ----------------------------------------------------------------- |
| `maxsim(..., backward="auto" \| "unified" \| "lowmem")`    | Per-call backward strategy. `"auto"` picks per shape: `"lowmem"` (bf16 grads, ~½ peak memory, deterministic) where gradient buffers dominate, `"unified"` (fastest) elsewhere. |
| `LIK_DISABLE=1`                                                   | Patched entry points delegate to vanilla PyLate / colpali_engine. |
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

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the contribution workflow, including how GPU tests run.

## Related projects

<details>
<summary><strong>⚡ MaxSim implementations</strong></summary>

<br>

- [roipony/flash-maxsim](https://github.com/roipony/flash-maxsim) — fused Triton kernel that tiles the similarity matrix in SRAM instead of materialising it in HBM.
- [erikkaum/maxsim](https://github.com/erikkaum/maxsim) — exact MaxSim with hand-written CUDA (NVIDIA) and Metal (Apple Silicon) kernels; avoids materialising the similarity matrix on either backend.
- [mixedbread-ai/maxsim-cpu](https://github.com/mixedbread-ai/maxsim-cpu) — Rust + SIMD CPU implementation (libxsmm on x86, Accelerate on ARM) for environments without a GPU.

</details>

<details>
<summary><strong>🏋️ Late interaction training libraries</strong></summary>

<br>

- [lightonai/pylate](https://github.com/lightonai/pylate) — ColBERT-style training and retrieval on top of Sentence Transformers; native LIK backend since [pylate#222](https://github.com/lightonai/pylate/pull/222).
- [illuin-tech/colpali](https://github.com/illuin-tech/colpali) — training and inference for ColPali / ColQwen2 visual late-interaction retrievers; native LIK backend since [colpali#412](https://github.com/illuin-tech/colpali/pull/412).
- [stanford-futuredata/ColBERT](https://github.com/stanford-futuredata/ColBERT) — the original late-interaction retriever, with ColBERTv2 training and PLAID indexing.

</details>

<details>
<summary><strong>🔍 Late interaction retrieval engines</strong></summary>

<br>

- [lightonai/fast-plaid](https://github.com/lightonai/fast-plaid) — fast PLAID index + search engine for ColBERT-style multi-vector retrieval.
- [lightonai/next-plaid](https://github.com/lightonai/next-plaid) — LightOn's next-generation PLAID engine (home of the Rust ColGrep runtime).

</details>

## Citation

```bibtex
@software{late_interaction_kernels_2026,
  author  = {Lac, Aurélien and Wu, Tony},
  title   = {{late-interaction-kernels}: Fused Triton kernels for late-interaction scoring},
  year    = {2026},
  url     = {https://github.com/hcompai/late-interaction-kernels},
}
```
