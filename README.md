# late-interaction-kernels

**Fused Triton kernels for late-interaction (MaxSim) scoring.**
*ColBERT · ColPali · ModernColBERT · LateOn · LateOn-Code · ColBERTv2 · PyLate-native.*

[Install](#install) · [Quickstart](#quickstart) · [Benchmarks](#benchmarks) · [Supported models](docs/supported_models.md) · [Design](docs/design.md)

---

## Highlights

- **One-line PyLate drop-in** — `patch_pylate()`. Numerically identical,
  zero config, `LIK_DISABLE=1` kill-switch.
- **Reranking / inference**: **7–23× faster** than PyTorch einsum, ~0
  scratch. Long documents (`Ld ≥ 8k`) run where naive OOMs.
- **LightOn cached-contrastive recipe**: up to **13.8×** on the
  MaxSim + backward slice at `bs=256, Ld=8k`.
- **LateOn-Code-edge (17 M) end-to-end PyLate training**: **1.04–1.27×**
  on 1×H100 with peak-memory savings up to **2 GB/rank**. Small
  encoder ⇒ MaxSim is a material slice of the step.
  → [End-to-end LateOn-Code-edge training](#end-to-end-lateon-code-edge-training-17-m-encoder--loss).
- **0.8.0 — closed-form fused D-side head for training**
  (`maxsim_from_hidden_train`): **1.5–4.6× faster** than the unfused
  `F.linear + F.normalize + maxsim` triple, peak memory ≤ unfused.
- **FP8 MaxSim inference** (Hopper WGMMA): up to **1.4×** at large `Nd`,
  ≤ 0.2 % relative error vs bf16.
- **Smooth top-K MaxSim** — denser training gradients at essentially
  hard-MaxSim cost.
- **ColBERTv2**: `plaid_approx_score` + 2/4/8-bit `maxsim_residual`,
  autograd-aware on `Q` since 0.5.0.
- **Varlen / packed**: `maxsim_varlen` autograd-aware, no
  `pad_sequence` on either pass.

All numbers on 1×H100 80 GB SXM, bf16 / fp16 compute, fp32 accumulator,
50-iter median. Every baseline is the same operation written in plain
PyTorch — not a comparison against any other library or engine.

## What this is

A small library of fused Triton kernels for the **scoring math** used
by late-interaction retrievers — ColBERT, ColPali, ModernColBERT,
ColBERTv2, LateOn, LateOn-Code. Every kernel is a numerically-matching
drop-in for the corresponding PyTorch expression:

- `maxsim` — the core `einsum → max → sum` pattern, fused
  FlashAttention-style so the `[Nq · Nd · Lq · Ld]` similarity tensor
  is never written to HBM.
- Companion kernels: fused L2-normalize, top-k retrieval, Matryoshka
  multi-dim scoring, XTR-style top-k aggregation, log-sum-exp
  relaxation (`soft_maxsim`), smooth top-K MaxSim (`smooth_maxsim`).
- Fused D-side heads: `maxsim_from_hidden` (inference) and
  `maxsim_from_hidden_train` (training, autograd-aware, closed-form
  backward as of 0.8.0).
- ColBERTv2: `plaid_approx_score` (IVF prune over centroid codes) and
  `maxsim_residual` (2/4/8-bit decompression + normalize + MaxSim,
  autograd-aware on `Q`).
- Varlen / packed: `maxsim_varlen`, autograd-aware end-to-end.
- `patch_pylate()` — one-line PyLate drop-in.

## What this is not

- **Not a search engine.** No index, no IVF disk format, no
  orchestration. For end-to-end retrieval use
  [FastPlaid](https://github.com/lightonai/fast-plaid),
  [NextPlaid / ColGrep](https://github.com/lightonai/next-plaid),
  PLAID, or plain PyLate — this library is what their MaxSim math
  *could* compile down to.
- **Not a new model or loss.** Numerically-matching drop-ins only.

## At a glance


| Kernel / path                                           | Baseline (same shape)                 | Speedup                      |
| ------------------------------------------------------- | ------------------------------------- | ---------------------------- |
| `maxsim_inference` — rerank / inference                 | `einsum + max + sum` (PyTorch)        | **7–23×**, ~0 scratch        |
| `maxsim_inference_fp8` (Hopper WGMMA, 0.7.0)            | `maxsim_inference` (bf16)             | **up to 1.4×** at `Nd ≥ 4k`  |
| `maxsim(..., normalize=True)`                           | `F.normalize + maxsim`                | **3–17×**                    |
| `maxsim_matryoshka` (K dims at once)                    | K separate MaxSim calls               | **1.6×**                     |
| `maxsim_from_hidden` (inference, 0.6.0)                 | `F.linear + F.normalize + maxsim`     | avoids `D_proj` scratch      |
| `maxsim_from_hidden_train` (training, **0.8.0**)        | `F.linear + F.normalize + maxsim`     | **1.5–4.6×** (closed-form bwd) |
| `smooth_maxsim` (top-K aggregate, 0.7.0)                | `torch.topk + mean + sum`             | smoother grads, O(K) bwd     |
| `plaid_approx_score` (ColBERTv2 IVF step)               | gather + mask + max + sum (PyTorch)   | **~20×**                     |
| `maxsim_residual` (2/4/8-bit) fwd+bwd                   | unpack + normalize + MaxSim (PyTorch) | **~20×**                     |
| `maxsim_varlen` fwd+bwd (ragged, no repad)              | pad + mask + maxsim autograd          | see `bench_backward_0_5`     |
| CachedContrastive chunked MaxSim (PyLate training)      | vanilla PyLate                        | **up to 13.8×** (MaxSim slice) |
| PyLate MaxSim + loss + backward (0.6.0 unified bwd)     | vanilla PyLate (same slice)           | **1.12–2.67×**               |
| **LateOn-Code-edge** end-to-end training (17 M encoder) | vanilla PyLate                        | **1.04–1.27×**, up to 2 GB saved |
| LateOn / ModernColBERT training (149 M encoder)         | vanilla PyLate                        | 1.00–1.06× (encoder-bound)   |
| Long-doc MaxSim at `Ld ≥ 8k`                            | naive einsum                          | **runs; naive OOMs**         |


Full tables, setup, and reproduction:
[`docs/benchmarks.md`](docs/benchmarks.md). Per-model guidance
(ColBERT v2, LateOn, LateOn-Code, LateOn-Code-edge, GTE-ModernColBERT,
mxbai-edge, ColPali): [`docs/supported_models.md`](docs/supported_models.md).

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

The PyLate drop-in targets PyLate **≥ 1.3** (the first release with the
`(queries_mask, documents_mask)` scoring signature).

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

`maxsim_inference` skips the argmax buffer save that the training path
needs — use it whenever you don't need gradients.

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
`maxsim_residual`. Backward is correct (the L2-norm Jacobian is applied).

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

### Fused D-side head (inference, 0.6.0)

When the on-disk format is raw hidden states `[Nd, Ld, d_model]` (the
last layer of a ModernBERT encoder), the standard rerank pipeline
materializes a big `D_proj = F.normalize(H_d @ W.T)` intermediate
before calling MaxSim. For large corpora this scratch is multi-GB
(e.g. 4.4 GB at `Nd=100k, Ld=180, d_out=128`) and OOMs on edge models
at `Nd > 1M`.

`maxsim_from_hidden` fuses projection + L2-normalize + MaxSim into
one kernel, so that scratch is never allocated:

```python
from late_interaction_kernels import maxsim_from_hidden

# Q_proj: already projected + normalized queries, [Lq, d_out]
# H_d:    raw hidden states, [Nd, Ld, d_model]
# W, b:   projection head (same convention as nn.Linear)
scores = maxsim_from_hidden(Q_proj, H_d, W, b=b, normalize=True)
```

Inference-only. The autograd-aware training variant ships in 0.7.0 —
see below.

### Fused D-side head (training, **new in 0.8.0**)

`maxsim_from_hidden_train` is the autograd-aware sibling of
`maxsim_from_hidden`. Forward runs the fused kernel with an argmax
save; backward gathers `H_d` at winners only, does the projection +
normalize + MaxSim gradient in closed form with bf16/fp16 matmuls
(fp32 accumulator), and scatters into `grad_H_d` with a single
`index_add_`. No autograd rebuild, no fp32 `F.linear` recompute.

```python
from late_interaction_kernels import maxsim_from_hidden_train

H_d.requires_grad_(True)
W.requires_grad_(True)
scores = maxsim_from_hidden_train(Q_proj, H_d, W, b=b, normalize=True)
loss = scores.sum()
loss.backward()     # grads flow to Q_proj, H_d, W, b as appropriate
```

Drop-in replacement for the unfused `F.linear → F.normalize → maxsim`
composition when `H_d`, `W` (or `Q_proj`) need gradients and you want
to skip the `[Nd, Ld, d_out]` `D_proj` scratch entirely. Numerically
matches the unfused path to bf16 tolerance (< 2 % RMS on gradients,
gradcheck clean on fp32). On long-doc training shapes where
`Nq · Lq ≪ Ld`, the backward does ≈ `Nq · Lq / Ld` the matmul work
of the unfused triple. Measured on H100 80 GB SXM, bf16 autocast,
forward + backward (reproduce with `benchmarks/bench_fused_head_train.py`):

| Shape                                         | Unfused  | Fused    | Speedup   |
| --------------------------------------------- | -------- | -------- | --------- |
| LateOn `Nd=128, Ld=1024, d_model=768`         | 1.45 ms  | 0.96 ms  | **1.52×** |
| LateOn-Code `Nd=256, Ld=1024`                 | 2.25 ms  | 1.02 ms  | **2.21×** |
| LateOn-Code `Nd=512, Ld=2048`                 | 7.31 ms  | 1.67 ms  | **4.37×** |
| LateOn-Code `Nd=1024, Ld=2048`                | 14.15 ms | 3.05 ms  | **4.64×** |
| LateOn-Code-edge `Nd=256, Ld=4096, d_model=384` | 5.13 ms  | 1.32 ms  | **3.88×** |

Peak memory is on par with or below the unfused path across all
shapes — no more `[Nd, Ld, d_out]` `D_proj` scratch.

### FP8 MaxSim inference (Hopper / Blackwell, **new in 0.7.0**)

Halve the HBM footprint and double the tensor-core throughput on Hopper
(`WGMMA`) — ideal for large-corpus reranking where the kernel is
HBM-bound:

```python
from late_interaction_kernels import (
    maxsim_inference_fp8,
    quantize_fp8_per_token,
)

Q_fp8, sQ = quantize_fp8_per_token(Q)   # or quantize_fp8_per_tensor(Q)
D_fp8, sD = quantize_fp8_per_token(D)
scores = maxsim_inference_fp8(Q_fp8, D_fp8, scale_Q=sQ, scale_D=sD)
```

Auto-falls back to dequantized bf16 with a one-time warning on
pre-Hopper GPUs or when Triton doesn't support fp8 `tl.dot` — the API
contract is preserved so calling code never has to branch.

### Smooth top-K MaxSim (training, **new in 0.7.0**)

Hard MaxSim gives 100 % of the gradient to a single doc token per query
token. `smooth_maxsim` aggregates the **top-K** doc tokens per query
token so the training signal is denser — same backward cost as hard
MaxSim times a tiny constant (no `[Nq, Nd, Lq, Ld]` scratch like
`soft_maxsim`):

```python
from late_interaction_kernels import smooth_maxsim

scores = smooth_maxsim(Q, D, top_k=4, aggregation="mean")
scores.sum().backward()
```

`top_k=1, aggregation="sum"` is bit-identical to hard `maxsim`. Use
`top_k ∈ [2, 8]` for smoother gradients during pre-training; fall back
to `maxsim` for eval.

### ColBERTv2 kernels (centroid codes + packed residuals)

```python
from late_interaction_kernels import plaid_approx_score, maxsim_residual

# 1) Approximate scoring via pre-computed query↔centroid scores
#    (IVF-style prune step over centroid codes):
scores = plaid_approx_score(query_centroid_scores, codes, doc_lengths)  # [n_docs]

# 2) Exact rerank with fused 2/4/8-bit residual decompression + normalize + MaxSim.
#    Autograd-aware on Q since 0.5.0 — train the query encoder directly on
#    the quantized document index, no dense unpack, no [Nd, Ld, d] fp32 scratch.
Q = Q.requires_grad_(True)
scores = maxsim_residual(
    Q, codes, residuals, doc_lengths,
    centroids, bucket_weights,
    nbits=2, normalize=True,
)
scores.sum().backward()        # grad_Q is fused; codes / centroids get none
```

These replace the `index_select → pad → mask → max → sum` and
`decompress → F.normalize → einsum → max → sum` patterns with a single
Triton kernel each. They let you build a ColBERTv2-style reranker — or
fine-tune against an already-compressed PLAID index — in pure Python
without hand-writing the fused ops. Use `maxsim_residual_inference` if
you don't need gradients and want to skip the argmax save.

### Varlen / packed inputs (autograd-aware since 0.5.0)

Short queries + widely-varying document lengths (typical in code search
or crawl corpora) no longer need `pad_sequence`:

```python
from late_interaction_kernels import maxsim_varlen

# Q_packed: [sum(Lq_i), d]     D_packed: [sum(Ld_j), d]
# cu_seqlens_{q,d}: [N+1] int32, FlashAttention convention
scores = maxsim_varlen(Q_packed, D_packed, cu_seqlens_q, cu_seqlens_d)
scores.sum().backward()        # grad_Q and grad_D produced on the packed layout
```

`grad_Q` is row-owned (scatter-free); `grad_D` uses the same fp32
`atomic_add` path the padded kernel uses. Use
`maxsim_varlen_inference` to skip the argmax save on the reranker path.

Wiring this into a heterogeneous-length training loop needs a packed
collator and a varlen-aware encoder forward. The padded kernel (what
`patch_pylate()` installs) stays the zero-config path; packing is opt-in
and pays off mainly on long-tailed corpora like code or crawl data. See
[`docs/packed_training.md`](docs/packed_training.md) for a cookbook and
[`examples/packed_training.py`](examples/packed_training.py) for a
runnable padded-vs-packed comparison.

---

## When it helps

| Scenario                                        | End-to-end effect                        |
| ----------------------------------------------- | ---------------------------------------- |
| Reranking / inference with pre-computed embeds  | **7–23× faster, ~0 scratch**             |
| LateOn / ModernColBERT inference at `Ld ≥ 8k`   | **runs; naive einsum OOMs**              |
| Offline KD scoring (teacher + student)          | **5–14×** (no encoder bwd)               |
| ColBERTv2 rerank on compressed (2-bit) docs     | **~20×** vs PyTorch unpack + einsum      |
| PyLate MaxSim + loss + backward slice (0.6.0)   | **1.12–2.67×** (unified backward)        |
| **LateOn-Code-edge** end-to-end training (17 M) | **1.04–1.27×**, up to 2 GB/rank saved    |
| LateOn / ModernColBERT training (149 M)         | 1.00–1.06× (free swap, encoder-bound)    |
| Any workload where `Nq · Nd · Lq · Ld` is tight | lets you raise batch / context size      |

**The honest summary.** On a full 149 M LateOn / ModernColBERT
training step the encoder dominates by ~10×, so the kernel is
effectively free — same wall-clock, same VRAM, deterministic
numerics. The moment MaxSim stops being negligible — inference,
reranking, long docs, big effective batch, KD scoring, compressed
indices, *or a small encoder like LateOn-Code-edge* — the fused path
actually moves. `patch_pylate()` is a one-liner either way;
`LIK_DISABLE=1` reverts to vanilla.

---

## Benchmarks

All numbers: 1×H100 80 GB SXM, bf16 compute (fp16 for LateOn / ModernColBERT
shapes), fp32 accumulator, 50-iter median, PyTorch 2.8, Triton 3.6, CUDA
12.9. Full tables + reproduction:
[`docs/benchmarks.md`](docs/benchmarks.md).

### Head-to-head vs `flash-maxsim` (IBM/roipony) — forward


| Shape                                          | late-interaction-kernels | flash-maxsim | Speedup   |
| ---------------------------------------------- | ------------------------ | ------------ | --------- |
| rerank-short (`Nq=1, Nd=1 000, Lq=32, Ld=300`) | 0.096 ms                 | 0.109 ms     | **1.13×** |
| rerank-long (`Nd=1 000, Ld=2 048`)             | 0.153 ms                 | 0.168 ms     | **1.10×** |
| rerank-colpali (`Lq/Ld=1024`)                  | 0.409 ms                 | 0.495 ms     | **1.21×** |
| rerank-10k (`Nq=1, Nd=10 000`)                 | 0.310 ms                 | 0.324 ms     | **1.05×** |
| train-in-batch-32                              | 0.087 ms                 | 0.100 ms     | **1.15×** |
| train-in-batch-128                             | 0.233 ms                 | 0.294 ms     | **1.26×** |
| edge-d48 (`Nq=1, Nd=4 000, d=48`)              | 0.320 ms                 | 0.341 ms     | **1.07×** |
| edge-d64 (`Nq=1, Nd=10 000, d=64`)             | 0.190 ms                 | 0.218 ms     | **1.14×** |


`late-interaction-kernels` additionally ships fused **backward**,
**quantized** (`maxsim_residual`), **ragged** (`maxsim_varlen`),
**soft** (`soft_maxsim`), and **top-K retrieval** (`maxsim_topk`)
paths that `flash-maxsim` does not. Reproduce with
[`benchmarks/bench_flash_maxsim.py`](benchmarks/bench_flash_maxsim.py).

### Unified backward (0.6.0) — PyLate MaxSim + loss + backward slice

The `"unified"` backward kernel is now the default for every shape
below `B=256`. This table isolates the **scoring slice** of a PyLate
training step — `colbert_scores(...) → cross_entropy → backward`
with **synthetic (pre-computed) embeddings, no encoder**. That is the
subset of work this library owns end-to-end:


| Shape                      | vanilla PyLate | 0.6.0 (unified) | Slice speedup |
| -------------------------- | -------------- | --------------- | ------------- |
| `B=16, Ld=200, d=128`      | 1.08 ms        | 0.94 ms         | **1.15×**     |
| `B=32, Ld=200, d=128`      | 0.95 ms        | 0.84 ms         | **1.12×**     |
| `B=64, Ld=200, d=128`      | 0.97 ms        | 0.84 ms         | **1.15×**     |
| `**B=128, Ld=200, d=128`** | **3.16 ms**    | **1.19 ms**     | **2.67×**     |
| `B=32, Ld=1 024, d=128`    | 1.02 ms        | 0.85 ms         | **1.21×**     |
| `B=64, Ld=256, d=48`       | 1.07 ms        | 0.94 ms         | **1.14×**     |


Reproduce with
[`benchmarks/bench_pylate_training.py --sweep`](benchmarks/bench_pylate_training.py).

On a **full end-to-end training step** with the 149 M LateOn /
LateOn-Code encoder, the ModernBERT forward+backward dominates by
~10×, so overall wall-clock stays at 1.00–1.06×. The 17 M
**LateOn-Code-edge** is a different story — MaxSim is a material
slice of the step there and the kernel moves end-to-end by
**1.04–1.27×**: see
[End-to-end LateOn-Code-edge training](#end-to-end-lateon-code-edge-training-17-m-encoder--loss).

### Reranking / inference


| Shape                                              | late-interaction-kernels | Naive einsum | Speedup   |
| -------------------------------------------------- | ------------------------ | ------------ | --------- |
| text — `Nq=1, Nd=1 000, Lq=32, Ld=300`             | 0.031 ms                 | 0.705 ms     | **22.7×** |
| corpus-10k — `Nq=1, Nd=10 000, Ld=300`             | 0.557 ms                 | 7.112 ms     | **12.8×** |
| ColPali-scale — `Nq=1, Nd=1 000, Lq=1024, Ld=1024` | 1.518 ms                 | 11.967 ms    | **7.9×**  |


### FP8 inference (Hopper WGMMA, 0.7.0)

`maxsim_inference_fp8` uses Hopper's fp8 tensor cores (`WGMMA`) with an
fp32 accumulator. Per-tensor and per-token scales are both supported;
on-the-fly dequant fallback kicks in on pre-Hopper GPUs. The real wins
show up on HBM-bound shapes (large `Nd × Ld`, small `d`) where the 2×
footprint reduction actually matters:

| Shape                                      | bf16 ms  | fp8 ms   | Speedup   |
| ------------------------------------------ | -------- | -------- | --------- |
| pylate-rerank-8k (`Nd=8 192, Ld=256, d=128`) | 0.237    | 0.189    | **1.26×** |
| batched-rerank-16k (`Nq=4, Nd=4 096, Ld=256`) | 0.404   | 0.286    | **1.42×** |
| long-docs-2k (`Nd=2 048, Ld=512, d=128`)   | 0.155    | 0.137    | **1.13×** |
| colbert-short-rerank (`Nd=4 096, Ld=128`)  | 0.116    | 0.116    | 1.00×     |

Relative numerical error vs the bf16 reference: ≤ 0.2 % on normalized
embeddings (within published fp8 reranking tolerances).
Reproduce with [`benchmarks/bench_fp8.py`](benchmarks/bench_fp8.py).

### Edge rerankers at long context + high BS (0.5.0)

`maxsim_inference` never materializes the `[Nq, Nd, Lq, Ld]` similarity
tensor — naive einsum does. That turns into a structural win for
small-`d` edge rerankers (LateOn-Code-edge `d=48`, mxbai-edge `d=64`)
at the shapes they're actually deployed on. Run with
`python benchmarks/bench_inference_edge.py`.


| Shape                                                | late-interaction-kernels | Naive einsum (fp32) | Speedup   | Peak mem (lik → naive) |
| ---------------------------------------------------- | ------------------------ | ------------------- | --------- | ---------------------- |
| LateOn-Code-edge — `Nd=1 000, Lq=32, Ld=1 024, d=48` | 0.072 ms                 | 0.380 ms            | **5.3×**  | 0.0 MB → 314 MB        |
| LateOn-Code-edge — `Nd=1 000, Lq=32, Ld=4 096, d=48` | 0.137 ms                 | 1.412 ms            | **10.3×** | 0.0 MB → 1.2 GB        |
| LateOn-Code-edge — `Nd=1 000, Lq=32, Ld=8 192, d=48` | 0.266 ms                 | 2.910 ms            | **10.9×** | 0.0 MB → 2.5 GB        |
| LateOn-Code-edge — `Nd=16 000, Lq=32, Ld=512, d=48`  | 0.252 ms                 | 2.897 ms            | **11.5×** | 0.1 MB → 2.5 GB        |
| mxbai-edge — `Nd=1 000, Lq=32, Ld=4 096, d=64`       | 0.172 ms                 | 1.730 ms            | **10.0×** | 0.0 MB → 1.5 GB        |
| mxbai-edge — `Nd=16 000, Lq=32, Ld=512, d=64`        | 0.331 ms                 | 3.528 ms            | **10.7×** | 0.1 MB → 3.0 GB        |
| d=128 reference — `Nd=1 000, Lq=32, Ld=4 096`        | 0.333 ms                 | 3.055 ms            | **9.2×**  | 0.0 MB → 2.5 GB        |
| Serving — `Nd=32 000, Lq=32, Ld=300, d=128`          | 0.766 ms                 | 7.488 ms            | **9.8×**  | 0.1 MB → 5.9 GB        |


All rows fit end-to-end inside `torch.inference_mode()` on a single
H100; the naive path would bottleneck any realistic rerank pipeline on
HBM alone. See [`benchmarks/bench_inference_edge.py`](benchmarks/bench_inference_edge.py)
for the full sweep (13 shapes).

### CachedContrastive chunked MaxSim — LightOn's long-doc recipe shapes

The `(bs / mini)²` Python loop inside `CachedContrastive` collapses to
one fused call. Same hyperparameters LightOn uses to train long-doc
ColBERT checkpoints on the ModernBERT-base family (LateOn-Code, the
former Reason-ModernColBERT recipe, …):


| Shape (Lq=128, mini=32)   | Vanilla fwd+bwd | Flash fwd+bwd | Speedup   | Vanilla peak | Flash peak | Mem×     |
| ------------------------- | --------------- | ------------- | --------- | ------------ | ---------- | -------- |
| `bs=64, Ld=8192`          | 55.3 ms         | 4.0 ms        | **13.9×** | 4.3 GB       | 0.6 GB     | **7.5×** |
| `bs=128, Ld=8192`         | 224.0 ms        | 15.7 ms       | **14.3×** | 4.6 GB       | 1.1 GB     | **4.2×** |
| `**bs=256, Ld=8192`**     | **915.9 ms**    | **66.3 ms**   | **13.8×** | **5.1 GB**   | **2.1 GB** | **2.4×** |


### End-to-end LateOn training (full encoder + loss)

Real model: `pylate.models.ColBERT("lightonai/LateOn")` →
`CachedContrastive` → AdamW, bf16 autocast, gradient checkpointing.
Numbers apply equally to `lightonai/LateOn-Code` and the predecessor
`lightonai/GTE-ModernColBERT-v1` (same ModernBERT-base backbone,
149 M params, `d=128`).


| Setup                                   | Vanilla | late-interaction-kernels | Speedup | Peak / rank |
| --------------------------------------- | ------- | ------------------------ | ------- | ----------- |
| 8×H100 DDP, bs=4/dev, mini=4, Ld=8192   | 665 ms  | 662 ms                   | 1.00×   | 30.2 GB     |
| 8×H100 DDP, bs=16/dev, mini=16, Ld=4096 | 1078 ms | 1047 ms                  | 1.03×   | 57.5 GB     |
| 8×H100 DDP, bs=32/dev, mini=32, Ld=2048 | 1020 ms | 960 ms                   | 1.06×   | 57.4 GB     |


### End-to-end LateOn-Code-edge training (17 M encoder + loss)

`LateOn-Code-edge` is the 17 M-parameter Ettin-backbone sibling of
`LateOn`. The encoder is so small that MaxSim becomes a meaningful
slice of the step, and the fused kernels move end-to-end PyLate
training — measured on a single H100, `Contrastive` loss with
in-batch negatives, bf16 autocast, gradient checkpointing:


| Setup (`Contrastive`, grad-ckpt, bf16, 1×H100) | Vanilla  | late-interaction-kernels | Speedup   | Peak (vanilla → flash) |
| ---------------------------------------------- | -------- | ------------------------ | --------- | ---------------------- |
| bs=256, Lq=32, Ld=256                          | 114.3 ms | 90.3 ms                  | **1.27×** | 11.5 GB → 9.6 GB       |
| bs=192, Lq=32, Ld=512                          | 152.5 ms | 126.0 ms                 | **1.21×** | 16.6 GB → 14.7 GB      |
| bs=128, Lq=32, Ld=1024                         | 223.7 ms | 196.4 ms                 | **1.14×** | 22.6 GB → 21.5 GB      |
| bs=64,  Lq=32, Ld=2048                         | 293.2 ms | 281.0 ms                 | 1.04×     | 25.8 GB                |

Reproduce with `scripts/sky_lateon_edge.yaml`. The pattern is
consistent: the smaller the encoder and the larger the effective
batch, the bigger the slice MaxSim owns — and the closer the
end-to-end speedup tracks the isolated-kernel speedup in the tables
above. For pure-inference reranking on the same encoder the fused
MaxSim path is **5–11× faster** — see
[Edge rerankers at long context + high BS](#edge-rerankers-at-long-context--high-bs-050).


### New kernels (v0.4) vs pure-PyTorch reference

Each reference is a plain-PyTorch transliteration of the same operation
(dense einsum + masks + reduce + optional normalize). These numbers are
*not* a comparison against any other library.


| Kernel                        | PyTorch ref | late-interaction-kernels | Speedup    |
| ----------------------------- | ----------- | ------------------------ | ---------- |
| `maxsim_matryoshka` (3 dims)  | 0.74 ms     | 0.47 ms                  | 1.56×      |
| `maxsim_topk` (10k docs)      | 0.30 ms     | 0.30 ms                  | 1.03×      |
| `plaid_approx_score` (8k×200) | 0.69 ms     | 0.033 ms                 | **20.58×** |
| `maxsim_residual` 2-bit       | 3.51 ms     | 0.16 ms                  | **21.70×** |
| `maxsim_residual` 4-bit       | 3.52 ms     | 0.18 ms                  | **19.95×** |


### Fused L2-normalize vs `F.normalize` + MaxSim


| Shape                            | Explicit | Fused    | Speedup   |
| -------------------------------- | -------- | -------- | --------- |
| `Nq=1, Nd=1 000, Lq=32, Ld=300`  | 0.49 ms  | 0.064 ms | **7.7×**  |
| `Nq=1, Nd=1 000, Lq=32, Ld=1024` | 1.57 ms  | 0.094 ms | **16.7×** |
| `Nq=1, Nd=10 000, Lq=32, Ld=300` | 4.44 ms  | 0.30 ms  | **14.7×** |


### Long-document MaxSim (`Ld ∈ {2k, 4k, 8k, 16k}`)

At 2–4k naive still fits (so you see direct speedup). At 8k+ naive OOMs.


| Shape                               | Flash fwd   | Naive fwd | Flash bwd   | Naive bwd | Flash peak | Naive peak |
| ----------------------------------- | ----------- | --------- | ----------- | --------- | ---------- | ---------- |
| `Nq=8, Nd=16, Ld=2048`              | 0.07 ms     | 0.15 ms   | 0.47 ms     | 0.56 ms   | 96 MB      | 152 MB     |
| `Nq=16, Nd=32, Ld=4096` (bigbatch)  | **0.08 ms** | 0.34 ms   | **0.46 ms** | 1.06 ms   | **193 MB** | 672 MB     |
| `Nq=8, Nd=16, Ld=8192`              | 0.07 ms     | **OOM**   | 0.39 ms     | **OOM**   | 192 MB     | OOM        |
| `Nq=1, Nd=256, Ld=8192` (rerank-8k) | 0.18 ms     | **OOM**   | 1.12 ms     | **OOM**   | 2.1 GB     | OOM        |
| `Nq=1, Nd=32, Ld=16384`             | 0.09 ms     | **OOM**   | 0.42 ms     | **OOM**   | 576 MB     | OOM        |


---

## Advanced usage

### Soft (log-sum-exp) MaxSim for training stability

```python
from late_interaction_kernels import soft_maxsim
scores = soft_maxsim(Q, D, beta=10.0)    # β → ∞ recovers hard max
```

Denser gradient — every doc token contributes a softmax-weighted share.
Use `β = 5..10` from scratch, `β = 20..50` for fine-tuning, fall back to
`maxsim` for eval.

### Backward-path selection

Two `grad_D` paths — `"auto"` picks per shape (default):

```python
from late_interaction_kernels import set_backward_method
set_backward_method("auto")     # heuristic (default — uses "unified" for almost everything)
set_backward_method("unified")  # 0.6.0: single-pass fused grad_Q + grad_D (FA-2 style)
set_backward_method("atomic")   # two-pass, fp32 atomic_add grad_D — fallback
set_backward_method("csr")      # two-pass, sort + bucket reduce — wins at very high contention (B≥256)
```

The unified backward was added in 0.6.0 and hoists `Q[i, s, :]` out of
the inner doc-batch loop, roughly halving HBM read traffic vs the
two-pass variant. It is now the default for every training shape
except very high-contention batches (`Nq ≥ 256 ∧ Nd ≥ 256 ∧ Lq ≤ 64`),
where CSR's determinism still wins. End-to-end PyLate training step
speedups range from **1.12× (small batch) to 2.67× (B=128)** over
unpatched PyLate on H100.

CSR is bitwise-deterministic across runs; atomic drifts by ~1e-6
relative (the `atomic_add` reduction order depends on the scheduler).

---

## Hardware support


| GPU family                  | Status         | Notes                                               |
| --------------------------- | -------------- | --------------------------------------------------- |
| H100 / H200 (Hopper)        | primary target | autotuned shortlist (incl. warp-specialized configs on Triton 3.2+), FP8 WGMMA path |
| A100 (Ampere 80 GB)         | supported      | separate autotune shortlist                         |
| L4, L40, RTX 4090 (Ada)     | supported      | generic shortlist                                   |
| A10, A40, RTX 3090 (Ampere) | supported      | generic shortlist                                   |
| Older / unknown CUDA        | works          | conservative default configs                        |
| CPU / macOS / Windows       | reference only | `late_interaction_kernels.reference` (pure PyTorch) |


The kernel autotunes once per unique `(Lq, Ld, d, masks)` signature and
caches the winner — zero overhead after warmup.

---

## Design

Short read: [`docs/design.md`](docs/design.md) walks through the
FlashAttention-style tiling, the fused masking, the argmax save for the
backward, and the two `grad_D` paths.

One-paragraph version: the forward tiles `Q` and `D` into SRAM, streams
the inner `matmul → max → running-sum` without ever writing the
`[Nq · Nd · Lq · Ld]` similarity tensor back to HBM. The backward either
recomputes the argmax on the fly and uses `atomic_add` (fast for
moderate shapes) or reuses a saved argmax via a CSR-style bucket
reduction (deterministic, wins at long sequences).

---

## Development

```bash
pip install -e ".[dev,pylate]"

# Unit tests (auto-skips CUDA tests if no GPU)
pytest tests/ -q

# Lint / format
ruff check .
ruff format --check .

# Benchmarks (single H100)
python benchmarks/bench_forward.py
python benchmarks/bench_backward_method.py
python benchmarks/bench_backward_0_5.py          # 0.5.0 — fused residual + varlen bwd
python benchmarks/bench_pylate_lateon.py  # real PyLate training step
python benchmarks/bench_fastplaid.py             # pip install fast-plaid first
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

**Aurélien Lac** · **Tony Wu** — H Company · 2026

---

## Related projects

The late-interaction ecosystem this library slots into:

- **[PyLate](https://github.com/lightonai/pylate)** — training framework
for late-interaction models. We provide a `patch_pylate()` drop-in and
test against it in CI.
- **[FastPlaid](https://github.com/lightonai/fast-plaid)** — a Rust /
`tch-rs` multi-vector search engine with its own index, IVF probe, and
rerank pipeline. A different category of software: FastPlaid is an
end-to-end search engine; this library is a set of fused kernels you
can call from Python. [`docs/benchmarks.md`](docs/benchmarks.md)
includes a small kernel-level comparison on the shared
`matmul → mask → max → sum` pattern.
- **[NextPlaid / ColGrep](https://github.com/lightonai/next-plaid)** —
LightOn's Rust CLI / serving stack built on top of FastPlaid,
optimised for on-disk code-search indexes. Complementary to this
library, which targets the Python training + in-memory rerank path.
- **[flash-maxsim](https://github.com/roipony/flash-maxsim)** by IBM
Research — the first public Triton MaxSim kernel for ColBERT /
ColPali, and the direct inspiration for this project.
`late-interaction-kernels` extends it with fused masking, varlen /
packed inputs, a fused `normalize=True` path, a training-grade
deterministic backward (CSR), Matryoshka / top-k / XTR / ColBERTv2
companion kernels, and the PyLate drop-in. See their README for
single-query inference memory numbers (a different regime than the
training-context peaks we report).
- **[FlashAttention](https://github.com/Dao-AILab/flash-attention)** —
the IO-aware tiling pattern this kernel is a strict subset of.
- **[Liger-Kernel](https://github.com/linkedin/Liger-Kernel)** — source
of the autotune / `torch.autograd.Function` idioms used here.

---

## License

Apache 2.0 — see [`LICENSE`](LICENSE). Copyright 2026 Aurélien Lac and Tony Wu.