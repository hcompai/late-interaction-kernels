# Supported models

`late-interaction-kernels` accelerates any scoring path that reduces to

```
score[i, j] = sum_s  max_t  ⟨Q[i, s], D[j, t]⟩
```

so in practice it works with **every PyLate ColBERT checkpoint** on GPU.
The list below is what we test against explicitly.

| Family               | Checkpoint                                                                                                  | `d` | `max_len` | Notable use case                          |
| -------------------- | ----------------------------------------------------------------------------------------------------------- | --- | --------- | ----------------------------------------- |
| ColBERT v2           | [`colbert-ir/colbertv2.0`](https://huggingface.co/colbert-ir/colbertv2.0)                                   | 128 | 512       | Training + Python-side reranking          |
| LateOn               | [`lightonai/LateOn`](https://huggingface.co/lightonai/LateOn)                                               | 128 | 8192      | Long-context training (`Ld ∈ {4k, 8k}`)   |
| LateOn-Code          | [`lightonai/LateOn-Code`](https://huggingface.co/lightonai/LateOn-Code)                                     | 128 | 8192      | Code-retrieval fine-tuning                |
| LateOn-Code-edge     | [`lightonai/LateOn-Code-edge`](https://huggingface.co/lightonai/LateOn-Code-edge)                           | 48  | 2048      | Edge rerank, small `d`                    |
| GTE-ModernColBERT    | [`lightonai/GTE-ModernColBERT-v1`](https://huggingface.co/lightonai/GTE-ModernColBERT-v1)                   | 128 | 8192      | Same backbone as LateOn                   |
| Reason               | [`lightonai/Reason-ModernColBERT`](https://huggingface.co/lightonai/Reason-ModernColBERT)                   | 128 | 8192      | `CachedContrastive` at `bs=256, Ld=8k`    |
| mxbai edge           | [`mixedbread-ai/mxbai-edge-colbert-v0-17m`](https://huggingface.co/mixedbread-ai/mxbai-edge-colbert-v0-17m) | 48  | 2048      | Edge / small-d                            |
| mxbai edge 32m       | [`mixedbread-ai/mxbai-edge-colbert-v0-32m`](https://huggingface.co/mixedbread-ai/mxbai-edge-colbert-v0-32m) | 64  | 2048      | Edge / small-d                            |
| ColPali (via PyLate) | any ColPali checkpoint                                                                                      | 128 | —         | VLM training through `CachedContrastive`  |

Drop-in for any of these via:

```python
from late_interaction_kernels import patch_pylate
patch_pylate()
```

For the relevant speedup ranges and the exact shapes that move the needle
on each family, see [`docs/benchmarks.md`](benchmarks.md). Edge models
(`d ∈ {48, 64}`) hit the largest reranking-side wins because the kernel
is more memory-bound at those sizes — see the dedicated section in
`docs/benchmarks.md`.

## Compressed indices (PLAID / ColBERTv2)

`maxsim_residual` (dense) and `maxsim_residual_varlen` (ragged) take the
PLAID compressed format — `(codes, residuals, centroids, bucket_weights)`
with 2/4/8-bit packed residuals — and decompress on-the-fly in SRAM.
Reranking on top of a FastPlaid IVF probe goes through
`maxsim_residual_varlen`; `maxsim_residual` is the dense version, with a
fused backward when `Q.requires_grad=True`.

## Out of scope

Non-PyTorch inference engines — ONNX Runtime, TensorRT, llama.cpp, the
Rust runtime behind LightOn's [ColGrep](https://github.com/lightonai/next-plaid/tree/main/colgrep)
— don't go through the Python kernel path and aren't accelerated here.
Training the underlying ColBERT checkpoints (which is a PyTorch / PyLate
job) does benefit.
