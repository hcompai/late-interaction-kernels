# Packed / varlen training

`late-interaction-kernels` ships `maxsim_varlen` — a fused, autograd-aware
MaxSim that consumes **packed** inputs (`[sum(L_i), d]` tensors + FlashAttention
`cu_seqlens`). This doc is a cookbook for wiring it into a training loop when
your documents have widely varying lengths.

It is **opt-in**. `patch_pylate()` only hooks up the padded kernel — the packed
path is a separate pipeline and does not ship as a one-liner. That is
intentional: the padded kernel is numerically identical and risk-free; the
packed path changes the data pipeline and only pays off in a specific regime.

---

## When packing is worth it

Packing saves wall-clock and VRAM in proportion to the **padding waste** in
your batches. A rule of thumb:

```
waste = 1 - (sum_of_real_tokens / (batch_size * max_len_in_batch))
```

| Workload                                                   | Typical waste | Worth packing? |
| ---------------------------------------------------------- | ------------- | -------------- |
| ColBERT / MS MARCO (short queries, ~180-token docs)        | ~15 %         | No             |
| ModernColBERT / long docs, homogeneous length              | ~5-10 %       | No             |
| ColPali (fixed 1030-token query, short docs)               | huge (query)  | Only if you pack the encoder too |
| Code search (64–2048 tokens, heavy tail)                   | 40-60 %       | **Yes**        |
| Crawl corpora, web snippets mixed with full documents      | 40-70 %       | **Yes**        |

Packing only the MaxSim step while keeping a padded encoder saves the MaxSim
cost but **not** the encoder cost — which is usually where the padding waste
actually burns. End-to-end wins require a packed encoder forward too
(FlashAttention varlen, `ModernBERT` native varlen path, or an
`unpad → encode → repack` adapter around a standard encoder).

Before writing any code, measure your `waste` on a real batch. If it's below
20 % you will not recoup the added complexity.

---

## The three pieces

A packed training pipeline needs three components. None of them ship in this
library today; `maxsim_varlen` is the last-mile kernel they plug into.

1. **A packed collator** — replaces `pylate.utils.ColBERTCollator`. Emits
   `input_ids`, `cu_seqlens`, and `max_seqlen` instead of `[B, L_max]` +
   `attention_mask`.
2. **A varlen-aware encoder forward** — either a native varlen encoder, or a
   thin `unpad → encoder(padded) → repack` adapter around a standard encoder.
3. **A packed loss** — calls `maxsim_varlen` on the packed embeddings, then
   reshapes the `[Nq, Nd]` scores the way your loss expects (e.g.
   `cross_entropy` over `[B, 2B]` for `Contrastive` with one positive + one
   negative).

The kernel does not care which encoder you use. It only needs the two packed
embedding tensors and their `cu_seqlens`.

---

## End-to-end snippet

The following snippet is deliberately self-contained. It uses a random encoder
(so it runs on CPU for smoke-testing) and a synthetic batch. Replace the
encoder with your real model and the collator with a dataloader wrapper.

```python
import torch
import torch.nn.functional as F

from late_interaction_kernels import maxsim_varlen


# ---------- 1. Packed collator (toy version) ----------------------------

def pack(seqs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, int]:
    """seqs: list of [L_i, d] token-embedding tensors (variable L_i)."""
    packed = torch.cat(seqs, dim=0)                        # [sum(L_i), d]
    lens = torch.tensor([s.shape[0] for s in seqs], dtype=torch.int32)
    cu = torch.zeros(len(seqs) + 1, dtype=torch.int32, device=packed.device)
    cu[1:] = lens.cumsum(0)
    return packed, cu.to(packed.device), int(lens.max())


# ---------- 2. Encoder adapter (padded encoder → packed embeddings) ------

def encode_padded_then_repack(encoder, input_ids_list):
    """Thin adapter: pad → encode → unpad → pack.

    Works with any [B, L_max, d]-shaped encoder. The repack costs one
    gather, which is cheap compared to the encoder forward itself.
    Replace with a native varlen forward if your encoder has one.
    """
    lens = [x.numel() for x in input_ids_list]
    L_max = max(lens)
    B = len(input_ids_list)

    padded = torch.zeros(B, L_max, dtype=torch.long, device=input_ids_list[0].device)
    mask = torch.zeros(B, L_max, dtype=torch.bool, device=padded.device)
    for i, x in enumerate(input_ids_list):
        padded[i, : lens[i]] = x
        mask[i, : lens[i]] = True

    emb = encoder(padded)                                  # [B, L_max, d]
    emb = F.normalize(emb, dim=-1)
    return pack([emb[i, : lens[i]] for i in range(B)])     # packed + cu_seqlens


# ---------- 3. Packed contrastive loss -----------------------------------

def packed_contrastive_step(encoder, queries, positives, negatives):
    """One training step with in-batch negatives on packed tensors.

    queries / positives / negatives: list of [L_i] int64 input_id tensors.
    """
    Qp, cu_q, _ = encode_padded_then_repack(encoder, queries)
    Pp, cu_p, _ = encode_padded_then_repack(encoder, positives)
    Np, cu_n, _ = encode_padded_then_repack(encoder, negatives)

    # Concatenate positives and negatives along the "doc" axis.
    Dp = torch.cat([Pp, Np], dim=0)
    cu_d = torch.cat([cu_p, cu_p[-1] + cu_n[1:]], dim=0)

    scores = maxsim_varlen(Qp, Dp, cu_q, cu_d)             # [B, 2B]
    B = cu_q.numel() - 1
    labels = torch.arange(B, device=scores.device)         # positive at i == i
    return F.cross_entropy(scores, labels)
```

A runnable version with synthetic data and a side-by-side padded baseline
lives at [`examples/packed_training.py`](../examples/packed_training.py).

---

## Correctness checks

`maxsim_varlen` is tested against the padded path in
[`tests/test_varlen.py`](../tests/test_varlen.py):

- `test_varlen_parity` — packed forward matches padded forward to
  `atol=1e-3`.
- `test_varlen_matches_padded_path` — same but with mixed `fp16` / `bf16`
  dtypes.
- `test_varlen_backward_matches_padded` — gradients on packed inputs match
  the padded autograd reference.
- `test_varlen_backward_requires_grad_gate` — packed backward is only wired
  when inputs require grad (inference stays argmax-save-free).

When debugging a packed pipeline, the first sanity check is:

```python
scores_packed = maxsim_varlen(Qp, Dp, cu_q, cu_d)
scores_padded = maxsim(Q_padded, D_padded, q_mask=q_mask, d_mask=d_mask)
torch.testing.assert_close(scores_packed, scores_padded, atol=1e-3, rtol=1e-3)
```

If these disagree, the bug is in the collator, not in the kernel.

---

## Caveats

- **The kernel is tested with eager autograd.** `torch.compile(fullgraph=True)`
  around the full training step has not been validated; the kernel itself is
  compiled in Triton.
- **Gradient checkpointing across `maxsim_varlen`** should work (the custom
  `autograd.Function` reruns the forward on recompute) but is not exercised
  in CI. If you rely on it in prod, add a parity test against the padded
  path.
- **Distributed training**: `grad_D` uses `atomic_add`. The numerics are
  stable but not bitwise-deterministic across ranks, same as the padded
  kernel. Use the `csr` backward path (padded kernel) if you need bitwise
  determinism — there is no CSR backward for varlen today.
- **Empty sequences** (zero-length) are handled by the kernel (scores
  contribution is zero, see `test_varlen_empty_sequence_is_zero`) but
  usually indicate a bug upstream — guard in your collator.

---

## Why this isn't a shipped `PackedCollator`

We considered shipping `PackedCollator` and `VarlenContrastive` classes as a
drop-in for PyLate's padded collator and `Contrastive` loss. We didn't,
because:

1. The encoder side is where padding waste usually burns; a collator that only
   feeds `maxsim_varlen` (and keeps the encoder padded) delivers a fraction of
   the possible win while adding a parallel API surface to maintain.
2. A faithful replacement would need to track PyLate's trainer API (features
   dict, skiplist, prompts, distillation labels). That's a moving target and
   belongs upstream in PyLate, not here.
3. The three components above are short enough to be a cookbook rather than a
   new module. A real-world integration always has project-specific wiring
   (skiplist tokens, prompt prefixes, KD teacher scores) that a generic class
   would force users to override anyway.

If you build a packed pipeline on top of this kernel and want to upstream the
glue, open an issue — we'll happily link to it from here.
