<div align="center">

# late-interaction-kernels

**Fused Triton kernels for late-interaction (MaxSim) scoring.**
_ColBERT · ColPali · ModernColBERT · PyLate · FastPlaid._

[![CI](https://github.com/hcompai/late-interaction-kernels/actions/workflows/ci.yml/badge.svg)](https://github.com/hcompai/late-interaction-kernels/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A5%202.1-ee4c2c.svg)](https://pytorch.org/)
[![Triton](https://img.shields.io/badge/Triton-%E2%89%A5%203.0-9146ff.svg)](https://github.com/triton-lang/triton)

</div>

---

A single IO-aware kernel that replaces PyLate's
`torch.einsum("ild,jtd->ijlt") → max → sum` (and FastPlaid's
`matmul → mask → max_dim → sum_dim`) with a FlashAttention-style streaming
reduction. The `[Nq · Nd · Lq · Ld]` similarity tensor is never materialized
in HBM — just the `[Nq · Nd]` scores come out.

| Where it wins                                          | Win                          |
| :----------------------------------------------------- | :--------------------------- |
| Reranking / inference with pre-computed embeddings     | **7–23× faster, ~0 scratch** |
| Fused `normalize=True` vs `F.normalize + maxsim`       | **3–17× faster**             |
| PLAID approximate scoring (`plaid_approx_score`)       | **~20× faster** than gather+mask+max in PyTorch |
| PLAID residual rerank (`maxsim_residual`, 2/4/8-bit)   | **~20× faster** than dense unpack + normalize + MaxSim |
| Matryoshka multi-dim scoring (`maxsim_matryoshka`)     | **1.6× faster** than K separate MaxSim calls |
| CachedContrastive "chunked MaxSim" step                | **up to 13.8×** faster       |
| ModernColBERT `Ld ≥ 8k` — MaxSim step                  | **runs; naive PyTorch OOMs** |
| FastPlaid exact rerank (`Ld ≥ 4k`)                     | **2.7–3.6×** faster          |
| Plain ModernColBERT training step (encoder-dominated)  | 1.00–1.06× (a free upgrade)  |

Every result is on a single H100 80 GB, bf16/fp16 compute, fp32 accumulator,
50-iter averages. Full tables, setup, and reproduction: [`docs/benchmarks.md`](docs/benchmarks.md).

---

## Contents

1. [Install](#install)
2. [Quickstart](#quickstart)
3. [When to reach for it](#when-to-reach-for-it)
4. [Benchmarks](#benchmarks)
5. [Advanced usage](#advanced-usage)
6. [Hardware support](#hardware-support)
7. [Design](#design)
8. [Development](#development)
9. [Citation](#citation)
10. [Authors](#authors)
11. [Acknowledgements](#acknowledgements)
12. [License](#license)

---

## Install

```bash
pip install late-interaction-kernels
```

Requires CUDA + Triton (auto-installed on Linux). On macOS / Windows the
reference PyTorch implementation is importable as
`late_interaction_kernels.reference.maxsim_reference` — useful for
correctness checks and CI.

Development install:

```bash
git clone https://github.com/hcompai/late-interaction-kernels
cd late-interaction-kernels
pip install -e ".[dev,pylate]"
pytest -q
```

---

## Quickstart

### Reranking / inference

```python
import torch
from late_interaction_kernels import maxsim_inference

Q = torch.randn(32, 128, device="cuda", dtype=torch.float16)         # [Lq, d]
D = torch.randn(1000, 300, 128, device="cuda", dtype=torch.float16)  # [Nd, Ld, d]

scores = maxsim_inference(Q, D)        # [1000] fp32
topk   = scores.topk(10)
```

### Training (autograd-aware)

```python
from late_interaction_kernels import maxsim

Q = torch.nn.Parameter(torch.randn(32,  32, 128, device="cuda"))   # [B, Lq, d]
D = torch.nn.Parameter(torch.randn(32, 128, 128, device="cuda"))   # [B, Ld, d]

scores = maxsim(Q, D)                       # [B, B] fp32
scores.sum().backward()                     # gradients flow to Q and D
```

### Mask-aware

```python
scores = maxsim(Q, D, q_mask=q_mask, d_mask=d_mask)    # masked tokens can't win the max
```

### PyLate drop-in — one line

```python
from late_interaction_kernels import patch_pylate
patch_pylate()   # swaps colbert_scores / colbert_kd_scores / Contrastive / CachedContrastive

# ...rest of your PyLate code is unchanged...
```

Kill-switch: `LIK_DISABLE=1` reverts to vanilla PyTorch at import time.

### Fused L2-normalize (skip the HBM round-trip)

```python
# Instead of F.normalize(Q) + F.normalize(D) + maxsim(Qn, Dn):
scores = maxsim_inference(Q, D, normalize=True)   # fused — 3–17× faster
```

Works for `maxsim`, `maxsim_inference`, `maxsim_matryoshka`, and
`maxsim_residual`. Backward is correct (L2-norm Jacobian is applied).

### Top-K retrieval

```python
from late_interaction_kernels import maxsim_topk

values, indices = maxsim_topk(Q, D, k=10)          # [1, 10], [1, 10]
# For very large corpora, chunk D to bound peak memory:
values, indices = maxsim_topk(Q, D, k=10, chunk=2048)
```

### Matryoshka multi-dim scoring

```python
from late_interaction_kernels import maxsim_matryoshka

# Scores at 32, 64, and 128 dims in a single kernel launch.
scores = maxsim_matryoshka(Q, D, dims=[32, 64, 128], normalize=True)
# scores.shape == [len(dims), Nq, Nd]
```

### XTR-style top-k aggregation

```python
from late_interaction_kernels import maxsim_xtr

scores = maxsim_xtr(Q, D, top_k=5)    # sum of top-5 doc-token scores per query token
```

### PLAID (ColBERTv2) kernels

```python
from late_interaction_kernels import plaid_approx_score, maxsim_residual

# 1) Approximate scoring via pre-computed query↔centroid scores:
scores = plaid_approx_score(query_centroid_scores, codes, doc_lengths)  # [n_docs]

# 2) Exact rerank with fused 2/4/8-bit residual decompression + normalize + MaxSim:
scores = maxsim_residual(
    Q, codes, residuals, doc_lengths,
    centroids, bucket_weights,
    nbits=2, normalize=True,
)
```

Both are drop-in replacements for FastPlaid's
`index_select → pad → colbert_score_reduce` and
`decompress → F.normalize → einsum → max → sum` pipelines — fused in a
single Triton kernel.

---

## When to reach for it

| scenario                                        | end-to-end win                |
| :---------------------------------------------- | :---------------------------- |
| Reranking / inference with pre-computed embeds  | **7–23× faster, ~0 scratch**  |
| ModernColBERT inference at `Ld ≥ 8k`            | **runs; naive OOMs**          |
| Offline KD scoring (teacher + student)          | **5–14×** (no encoder bwd)    |
| CachedContrastive training (Reason recipe)      | 1.00–1.06× (free swap)        |
| Plain Contrastive ModernColBERT training        | 1.00× (free swap)             |
| FastPlaid rerank — typical `Ld ≤ 1k` corpora    | neutral (< 1 % of search)     |
| FastPlaid rerank — `Ld ≥ 4k` ModernColBERT      | 2.7–3.6× on the rerank step   |
| Any workload where `Nq · Nd · Lq · Ld` is tight | **lets you raise batch size** |

The honest story: on a full ModernColBERT *training* step the 22-layer
transformer dominates by ~10×, so the kernel is roughly free (same wall-clock,
same VRAM, deterministic numerics). The full value surfaces on MaxSim-heavy
workloads — inference, reranking, long documents, large negative sets, KD.

---

## Benchmarks

All numbers: single **H100 80 GB SXM**, bf16 compute (fp16 for ModernColBERT),
fp32 accumulator, 50-iter average, PyTorch 2.8, Triton 3.6, CUDA 12.9. See
[`docs/benchmarks.md`](docs/benchmarks.md) for the full tables + reproduction.

### Reranking / inference

| shape                                              | late-interaction-kernels | naive einsum | speedup   |
| :------------------------------------------------- | -----------------------: | -----------: | :-------: |
| text — `Nq=1, Nd=1 000, Lq=32, Ld=300`             |                 0.031 ms |     0.705 ms | **22.7×** |
| corpus-10k — `Nq=1, Nd=10 000, Ld=300`             |                 0.557 ms |     7.112 ms | **12.8×** |
| ColPali-scale — `Nq=1, Nd=1 000, Lq=1024, Ld=1024` |                 1.518 ms |    11.967 ms |  **7.9×** |

### CachedContrastive chunked MaxSim — LightOn's Reason-ModernColBERT shapes

The `(bs / mini)²` Python loop inside `CachedContrastive` collapses to one
fused call. At the exact recipe used to train
[Reason-ModernColBERT](https://huggingface.co/lightonai/Reason-ModernColBERT):

| shape (Lq=128, mini=32)        | vanilla fwd+bwd | flash fwd+bwd | speedup   | vanilla peak | flash peak |  mem×    |
| :----------------------------- | --------------: | ------------: | :-------: | -----------: | ---------: | :------: |
| `bs=64, Ld=8192`               |         55.3 ms |        4.0 ms | **13.9×** |       4.3 GB |     0.6 GB | **7.5×** |
| `bs=128, Ld=8192`              |        224.0 ms |       15.7 ms | **14.3×** |       4.6 GB |     1.1 GB | **4.2×** |
| **`bs=256, Ld=8192` (Reason)** |    **915.9 ms** |   **66.3 ms** | **13.8×** |   **5.1 GB** | **2.1 GB** | **2.4×** |

### End-to-end ModernColBERT training (full encoder + loss)

Real model: `pylate.models.ColBERT("lightonai/GTE-ModernColBERT-v1")` →
`CachedContrastive` → AdamW, bf16 autocast, gradient checkpointing.

| setup                                   | vanilla | late-interaction-kernels | speedup | peak / rank |
| :-------------------------------------- | ------: | -----------------------: | :-----: | ----------: |
| 8×H100 DDP, bs=4/dev,  mini=4,  Ld=8192 |  665 ms |                   662 ms |  1.00×  |     30.2 GB |
| 8×H100 DDP, bs=16/dev, mini=16, Ld=4096 | 1078 ms |                  1047 ms |  1.03×  |     57.5 GB |
| 8×H100 DDP, bs=32/dev, mini=32, Ld=2048 | 1020 ms |                   960 ms |  1.06×  |     57.4 GB |

### New kernels (v0.4) vs PyTorch reference

| kernel                       |  PyTorch ref | late-interaction-kernels |  speedup   |
| :--------------------------- | -----------: | -----------------------: | :--------: |
| `maxsim_matryoshka` (3 dims) |      0.74 ms |                  0.47 ms |   1.56×    |
| `maxsim_topk` (10k docs)     |      0.30 ms |                  0.30 ms |   1.03×    |
| `plaid_approx_score` (8k×200)|      0.69 ms |                 0.033 ms | **20.58×** |
| `maxsim_residual` 2-bit      |      3.51 ms |                  0.16 ms | **21.70×** |
| `maxsim_residual` 4-bit      |      3.52 ms |                  0.18 ms | **19.95×** |

### Fused L2-normalize vs `F.normalize` + MaxSim

| shape                                    | explicit | fused | speedup |
| :--------------------------------------- | -------: | ----: | :-----: |
| `Nq=1, Nd=1 000, Lq=32, Ld=300`          | 0.49 ms  | 0.064 ms | **7.7×** |
| `Nq=1, Nd=1 000, Lq=32, Ld=1024`         | 1.57 ms  | 0.094 ms | **16.7×** |
| `Nq=1, Nd=10 000, Lq=32, Ld=300`         | 4.44 ms  | 0.30 ms  | **14.7×** |

### Long-document MaxSim (`Ld ∈ {2k, 4k, 8k, 16k}`)

At 2–4k naive still fits (so you see direct speedup). At 8k+ naive OOMs.

| shape                                  |   flash fwd |   naive fwd |   flash bwd |   naive bwd |    flash peak | naive peak |
| :------------------------------------- | ----------: | ----------: | ----------: | ----------: | ------------: | ---------: |
| `Nq=8, Nd=16, Ld=2048`                 |     0.07 ms |     0.15 ms |     0.47 ms |     0.56 ms |         96 MB |     152 MB |
| `Nq=16, Nd=32, Ld=4096` (bigbatch)     | **0.08 ms** |     0.34 ms | **0.46 ms** |     1.06 ms |    **193 MB** |     672 MB |
| `Nq=8, Nd=16, Ld=8192`                 |     0.07 ms |     **OOM** |     0.39 ms |     **OOM** |        192 MB |        OOM |
| `Nq=1, Nd=256, Ld=8192` (rerank-8k)    |     0.18 ms |     **OOM** |     1.12 ms |     **OOM** |        2.1 GB |        OOM |
| `Nq=1, Nd=32, Ld=16384`                |     0.09 ms |     **OOM** |     0.42 ms |     **OOM** |        576 MB |        OOM |

---

## Advanced usage

### Soft (log-sum-exp) MaxSim for training stability

```python
from late_interaction_kernels import soft_maxsim
scores = soft_maxsim(Q, D, beta=10.0)    # β → ∞ recovers hard max
```

Denser gradient — every doc token contributes a softmax-weighted share. Use
`β = 5..10` from scratch, `β = 20..50` for fine-tuning, fall back to `maxsim`
for eval.

### Varlen / packed inputs (no padding)

```python
from late_interaction_kernels import maxsim_varlen

# Q_packed: [sum(Lq_i), d]     D_packed: [sum(Ld_j), d]
# cu_seqlens_{q,d}: [N+1] int32, FlashAttention convention
scores = maxsim_varlen(Q_packed, D_packed, cu_seqlens_q, cu_seqlens_d)
```

### Backward-path selection

Two `grad_D` paths — `"auto"` picks per shape (default):

```python
from late_interaction_kernels import set_backward_method
set_backward_method("auto")     # heuristic (default)
set_backward_method("atomic")   # fp32 atomic_add — fast for small / medium
set_backward_method("csr")      # sort + bucket reduce — wins at long seqs / huge Nd
```

CSR is bitwise-deterministic across runs; atomic drifts by ~1e-6 relative
(`atomic_add` reduction order depends on the scheduler).

---

## Hardware support

| GPU family                  | Status         | Notes                                                  |
| :-------------------------- | :------------- | :----------------------------------------------------- |
| H100 / H200 (Hopper)        | primary target | autotuned shortlist, all benchmarks from here          |
| A100 (Ampere 80 GB)         | supported      | separate autotune shortlist                            |
| L4, L40, RTX 4090 (Ada)     | supported      | generic shortlist                                      |
| A10, A40, RTX 3090 (Ampere) | supported      | generic shortlist                                      |
| Older / unknown CUDA        | works          | conservative default configs                           |
| CPU / macOS / Windows       | reference only | `late_interaction_kernels.reference` (pure PyTorch)    |

The kernel autotunes once per unique `(Lq, Ld, d, masks)` signature and caches
the winner — zero overhead after warmup.

---

## Design

Short read: [`docs/design.md`](docs/design.md) walks through the
FlashAttention-style tiling, the fused masking, the argmax save for backward,
and the two `grad_D` paths.

One-paragraph version: forward tiles `Q` and `D` into SRAM, streams the inner
`matmul → max → running-sum` without ever writing the `[Nq · Nd · Lq · Ld]`
similarity tensor back to HBM. The backward either recomputes the argmax on
the fly and uses `atomic_add` (fast for moderate shapes) or reuses a saved
argmax via a CSR-style bucket reduction (deterministic, wins at long
sequences).

---

## Development

```bash
pip install -e ".[dev,pylate]"

# fast: CPU reference + non-GPU tests
pytest tests/ -q

# full: GPU kernels, CSR / atomic sweep, PyLate end-to-end
pytest tests/ -q --runxfail       # on a CUDA box

# benchmarks (single H100)
python benchmarks/bench_forward.py
python benchmarks/bench_backward_method.py
python benchmarks/bench_moderncolbert.py
python benchmarks/bench_cached_maxsim.py
python benchmarks/bench_fastplaid.py         # pip install fast-plaid first
```

Contributions welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md).

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

---

## Authors

**Aurélien Lac** · **Tony Wu**
H Company · 2026

---

## Acknowledgements

- [FlashAttention](https://github.com/Dao-AILab/flash-attention) —
  the IO-aware tiling pattern this kernel is a strict subset of.
- [flash-maxsim](https://github.com/roipony/flash-maxsim) by IBM Research —
  the first public Triton MaxSim kernel for ColBERT / ColPali and the direct
  inspiration for this project. late-interaction-kernels extends it with
  fused masking, varlen / packed inputs, a training-grade deterministic
  backward (CSR), and the PyLate drop-in. See their README for the
  single-query inference numbers (1920×–9438× VRAM reduction); our tables
  quote *training-context* peak memory (argmax + grads + normalization
  buffers) which is a smaller ratio.
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) — the autotune
  and `torch.autograd.Function` idioms. See
  [`docs/liger.md`](docs/liger.md) for upstream-merge thoughts.
- [PyLate](https://github.com/lightonai/pylate) — the late-interaction
  training framework we optimize for (and test against).
- [FastPlaid](https://github.com/lightonai/fast-plaid) — LightOn's
  Rust/libtorch multi-vector search engine; benchmarked head-to-head in
  [`docs/benchmarks.md`](docs/benchmarks.md).

---

## License

Apache 2.0 — see [`LICENSE`](LICENSE). Copyright 2026 Aurélien Lac and Tony Wu.
