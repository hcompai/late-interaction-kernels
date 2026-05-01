# Packed (varlen) training

`maxsim_varlen` is a fused, autograd-aware MaxSim that takes packed
inputs (`[sum(L_i), d]` + `cu_seqlens`). This page is a cookbook for
wiring it into a training loop when document lengths are heterogeneous.

`patch_pylate()` only hooks the padded kernel — the packed path changes
the data pipeline and only pays off in a specific regime, so it's
opt-in.

## When packing is worth it

Save proportional to padding waste. Rule of thumb:

```
waste = 1 − (Σ real_tokens) / (batch_size · max_len_in_batch)
```

| Workload                                             | Typical waste | Worth packing? |
| ---------------------------------------------------- | ------------- | -------------- |
| ColBERT / MS MARCO (short queries, ~180-tok docs)    | ~15 %         | No             |
| ModernColBERT, homogeneous long docs                 | ~5–10 %       | No             |
| ColPali (fixed long query, short docs)               | huge (query)  | Only if you pack the encoder too |
| Code search (64–2048 tokens, heavy tail)             | 40–60 %       | **Yes**        |
| Crawl / web snippets mixed with long docs            | 40–70 %       | **Yes**        |

Packing only the MaxSim while keeping a padded encoder saves the MaxSim
cost but **not** the encoder cost — which is usually where padding waste
actually burns. End-to-end wins need a packed encoder (FlashAttention
varlen, ModernBERT native varlen, or an `unpad → encode → repack`
adapter). Measure `waste` first; below 20 % the added complexity isn't
worth it.

## Pieces

A packed pipeline needs three components. None ship in this library —
`maxsim_varlen` is the kernel they plug into.

1. **Packed collator** — emits `input_ids`, `cu_seqlens`, `max_seqlen`
   instead of `[B, L_max] + attention_mask`.
2. **Varlen encoder forward** — native varlen, or
   `unpad → encode → repack`.
3. **Packed loss** — calls `maxsim_varlen` on the packed embeddings,
   reshapes the `[Nq, Nd]` scores the way the loss expects.

## End-to-end snippet

```python
import torch
import torch.nn.functional as F

from late_interaction_kernels import maxsim_varlen


def pack(seqs):
    """seqs: list of [L_i, d] tensors. Returns (packed, cu_seqlens, max_L)."""
    packed = torch.cat(seqs, dim=0)
    lens = torch.tensor([s.shape[0] for s in seqs], dtype=torch.int32)
    cu = torch.zeros(len(seqs) + 1, dtype=torch.int32, device=packed.device)
    cu[1:] = lens.cumsum(0)
    return packed, cu.to(packed.device), int(lens.max())


def encode_padded_then_repack(encoder, input_ids_list):
    """Padded encoder → packed embeddings, costs one gather per call."""
    lens = [x.numel() for x in input_ids_list]
    L_max = max(lens)
    B = len(input_ids_list)
    padded = torch.zeros(B, L_max, dtype=torch.long, device=input_ids_list[0].device)
    for i, x in enumerate(input_ids_list):
        padded[i, : lens[i]] = x
    emb = F.normalize(encoder(padded), dim=-1)
    return pack([emb[i, : lens[i]] for i in range(B)])


def step(encoder, queries, positives, negatives):
    Qp, cu_q, _ = encode_padded_then_repack(encoder, queries)
    Pp, cu_p, _ = encode_padded_then_repack(encoder, positives)
    Np, cu_n, _ = encode_padded_then_repack(encoder, negatives)
    Dp = torch.cat([Pp, Np], dim=0)
    cu_d = torch.cat([cu_p, cu_p[-1] + cu_n[1:]], dim=0)

    scores = maxsim_varlen(Qp, Dp, cu_q, cu_d)          # [B, 2B]
    B = cu_q.numel() - 1
    labels = torch.arange(B, device=scores.device)
    return F.cross_entropy(scores, labels)
```

A runnable version with a synthetic dataset and a padded baseline is in
[`examples/packed_training.py`](../examples/packed_training.py).

## Sanity check

`tests/test_varlen.py` exercises parity against the padded path. When
debugging your own pipeline, the first thing to check is:

```python
scores_packed = maxsim_varlen(Qp, Dp, cu_q, cu_d)
scores_padded = maxsim(Q_padded, D_padded, q_mask=q_mask, d_mask=d_mask)
torch.testing.assert_close(scores_packed, scores_padded, atol=1e-3, rtol=1e-3)
```

If those disagree, the bug is in the collator, not in the kernel.

## Caveats

* **`torch.compile(fullgraph=True)`** around the full step is not
  validated; the kernel itself is compiled in Triton.
* **Distributed training**: varlen `grad_D` is `atomic_add` and is not
  bitwise-deterministic across ranks. There is no CSR backward for
  varlen today.
* **Empty sequences** (`L_i = 0`) are handled — score contribution is
  zero — but usually indicate an upstream bug; guard in the collator.
