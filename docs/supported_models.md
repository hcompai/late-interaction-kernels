# Supported models

`late-interaction-kernels` is a **kernel** library — not a model and not a
retrieval engine. It accelerates any late-interaction scoring path that
boils down to

```
scores[i, j] = sum_s max_t  Q[i, s, :]  ·  D[j, t, :]
```

so in practice it supports **any PyLate ColBERT checkpoint** on GPU. This
page lists model families we have explicitly tested against and how they
slot into the API.

> The Python / PyTorch path is what this library speeds up. Runtimes that
> don't go through PyTorch — ONNX Runtime, TensorRT, llama.cpp, Rust
> inference engines — are out of scope; see the [ColGrep note](#colgrep-and-nextplaid)
> at the end of this page.

## PyLate-native ColBERT checkpoints (drop-in)

For any checkpoint loadable via `pylate.models.ColBERT`, the one-liner is:

```python
from late_interaction_kernels import patch_pylate

patch_pylate()  # replaces pylate.scores.colbert_scores and .colbert_kd_scores
```

From there, every PyLate component that scores via MaxSim — `Contrastive`,
`CachedContrastive`, `Distillation`, `rerank()` — transparently goes
through the fused kernel. Set `LIK_DISABLE=1` in the env to fall back to
vanilla PyLate for a one-off comparison.

### Tested checkpoints

| Family | Checkpoint | `d` | `max_len` | Where it helps |
|---|---|---:|---:|---|
| ColBERT v2 | [`colbert-ir/colbertv2.0`](https://huggingface.co/colbert-ir/colbertv2.0) | 128 | 512 | Training + Python-side reranking |
| PyLate GTE | [`lightonai/GTE-ModernColBERT-v1`](https://huggingface.co/lightonai/GTE-ModernColBERT-v1) | 128 | 8192 | Long-context training (`Ld ∈ {4k, 8k}`) |
| Reason | [`lightonai/Reason-ModernColBERT`](https://huggingface.co/lightonai/Reason-ModernColBERT) | 128 | 8192 | `CachedContrastive` at `bs=256, Ld=8k` |
| mxbai edge | [`mixedbread-ai/mxbai-edge-colbert-v0-17m`](https://huggingface.co/mixedbread-ai/mxbai-edge-colbert-v0-17m) | 48 | 2048 | Edge / small-d (see below) |
| mxbai edge 32m | [`mixedbread-ai/mxbai-edge-colbert-v0-32m`](https://huggingface.co/mixedbread-ai/mxbai-edge-colbert-v0-32m) | 64 | 2048 | Edge / small-d |
| **LateOn-Code** | [`lightonai/LateOn-Code-edge`](https://huggingface.co/lightonai/LateOn-Code-edge) | 48 | 2047 | Code-retrieval fine-tuning |
| **LateOn-Code** | [`lightonai/LateOn-Code`](https://huggingface.co/lightonai/LateOn-Code) | 128 | 8192 | Code-retrieval fine-tuning |
| ColPali (via PyLate) | any ColPali checkpoint | 128 | — | VLM training through CachedContrastive |

### Short-`d` regimes (LateOn-Code-edge, mxbai-edge, ColPali-small)

With `d ∈ {48, 64}` the kernel is **more memory-bound** than at `d = 128`,
because the tensor-core math per HBM byte drops. That's good news for this
library: fusing `matmul → mask → max → sum` removes exactly the HBM
round-trip the reference path wastes. Measured speedups at `d = 48` are in
line with the `d = 128` numbers; we ship a benchmark config for each in
`benchmarks/bench_forward.py`.

If you want to verify on your hardware before depending on this, the
two-command test is:

```bash
python benchmarks/bench_moderncolbert.py --d 48  # or 64
python benchmarks/bench_pylate_moderncolbert.py --model lightonai/LateOn-Code-edge
```

## Compressed / quantized indices (PLAID / ColBERTv2)

`maxsim_residual` takes the compressed format PLAID uses — `codes` +
`residuals` (2/4/8-bit packed) + a centroid / bucket table — and scores
queries against it with on-the-fly decompression. From 0.5.0 onward the
backward is also fused, so you can **train the query encoder directly on
the quantized document index** without ever materializing dense fp16
embeddings:

```python
from late_interaction_kernels import maxsim_residual

scores = maxsim_residual(
    Q,                              # requires_grad fine
    codes, residuals, doc_lengths,  # int, non-differentiable
    centroids, bucket_weights,      # frozen k-means artefacts
    nbits=2,
    normalize=True,
)
scores.sum().backward()             # grad_Q is fused; codes / centroids get none
```

Use `maxsim_residual_inference` if you only need reranking — it skips the
argmax save and saves `Nq * Nd * Lq * 4` bytes of VRAM.

**When does fused-backward residual actually help vs `unpack + maxsim`?**
The fused path avoids ever materializing the dense `[Nd, Ld, d]` fp32
embedding, so the VRAM story is unambiguous. For wall-clock time, the
crossover we measured on H100 is roughly:

- small `Nd` (≤ 128) — typical training / distillation batches —
  fused is **~1.3–1.5×** faster at 2/4/8-bit.
- large `Nd` (≥ 512) — reranker-scale — fused is **~2–3× slower**
  because the decompression is re-run per query-token during backward
  while the reference amortizes it across a single `unpack` call.

If you're training, stay with the fused path (small `Nd`, VRAM wins).
If you're scoring thousands of candidates with autograd enabled (rare —
you probably want `maxsim_residual_inference` here), fall back to the
dense unpack + `maxsim` autograd path. See
`benchmarks/bench_backward_0_5.py` for the exact numbers.

## Ragged / packed batches (code search, heterogeneous doc lengths)

If your documents have widely different lengths — typical in code search,
commit-message retrieval, or crawl corpora — `maxsim_varlen` avoids the
50 %-waste `pad_sequence` that PyLate's `rerank()` does by default. From
0.5.0 the packed path is autograd-aware end-to-end (`grad_Q` and
`grad_D` are produced directly on the `[sum_L, d]` layout):

```python
from late_interaction_kernels import maxsim_varlen

scores = maxsim_varlen(Qp, Dp, cu_seqlens_q, cu_seqlens_d)
loss = -scores.diag().mean()
loss.backward()
```

## ColGrep and NextPlaid

LightOn's [ColGrep](https://github.com/lightonai/next-plaid/tree/main/colgrep)
is a **Rust** command-line tool built on **NextPlaid** (a Rust multi-vector
database) and inference is handled by **ONNX Runtime**. It is not a
PyTorch / Python code path, so `late-interaction-kernels` cannot (and
does not try to) replace any of its kernels.

Where this library *does* apply in the ColGrep / LateOn-Code story:

- **Training the LateOn-Code checkpoints themselves** — these are PyLate
  models, trained with standard PyLate loops. `patch_pylate()` drops in
  unchanged and accelerates the MaxSim scoring step in `Contrastive` /
  `CachedContrastive` / `Distillation`.
- **Custom Python-side rerankers** over a LateOn-Code corpus (e.g. a
  bespoke research pipeline that runs PyTorch inference directly instead
  of via ColGrep) get the fused forward and — if you're distilling or
  fine-tuning — the fused backward.

ColGrep's own `colgrep search` path is completely independent of this
library and is the right tool for the CLI / agent use case.
