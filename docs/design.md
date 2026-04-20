# Design notes

## Problem statement

For two batches of token embeddings

    Q ∈ ℝ^{Nq × Lq × d}        D ∈ ℝ^{Nd × Ld × d}

the ColBERT MaxSim score is

    score[i, j] = Σ_s  max_t  ⟨ Q[i, s], D[j, t] ⟩

with optional per-batch boolean masks on both sides (`q_mask` drops rows from
the outer sum, `d_mask` hides doc tokens from the inner max). The output is
`[Nq, Nd]`.

The obvious implementation — PyLate's — materializes the full similarity
tensor `S ∈ ℝ^{Nq × Nd × Lq × Ld}`, applies the mask by multiplication, then
reduces. Two problems:

1. **Memory**: `S` is the dominant term and scales as `Nq · Nd · Lq · Ld`.
   For a mid-size training step (`Nq=Nd=128, Lq=32, Ld=300, d=128`) that's
   ~600 MB fp32 per score tensor; at ColPali scale (`Lq=Ld=1024`) it's
   4.5 GB per tensor. Training loops allocate 3–4 such tensors per step.
2. **Time**: materializing then reducing requires a full HBM round-trip. The
   einsum itself already hits the 3 TB/s memory bandwidth ceiling on H100.

Both issues go away if we never put `S` in HBM.

## Core idea

Borrow the outer-product tiling of FlashAttention. For each `(q_batch, d_batch)`
pair we assign one program; inside it we stream `D` in tiles while holding a
`[BLOCK_Q]` running maximum in registers:

```
for q_start in 0, BLOCK_Q, 2·BLOCK_Q, ..., Lq:      (outer — static)
    load Q_block    [BLOCK_Q × d]           → SRAM
    m      ← −inf   [BLOCK_Q]
    argmax ← 0      [BLOCK_Q]

    for d_start in 0, BLOCK_D, 2·BLOCK_D, ..., Ld:  (inner — dynamic)
        load D_block [BLOCK_D × d]          → SRAM
        S = Q_block @ D_blockᵀ              (tensor cores, fp32 acc)
        S ← where(d_active, S, −inf)        (mask fused in)
        tile_max    = max(S, axis=1)
        tile_argmax = argmax(S, axis=1) + d_start
        update      = tile_max > m
        argmax ← where(update, tile_argmax, argmax)
        m       ← max(m, tile_max)

    m ← where(finite(m) & q_active, m, 0)           (masked rows → 0)
    score_acc += sum(m)

store scores[q_batch, d_batch] ← score_acc
store argmax[q_batch, d_batch] ← argmax             (if needed for backward)
```

This is exactly the FlashAttention kernel with the softmax replaced by `max`
and the `P·V` projection dropped. The online-max reduction has no
normalization term and is trivially numerically exact — unlike online
softmax, there is nothing to rescale.

Memory: each program uses `BLOCK_Q·d + BLOCK_D·d + BLOCK_Q·BLOCK_D` fp16
values in SRAM, plus one `[BLOCK_Q]` fp32 accumulator. For
`BLOCK_Q=BLOCK_D=64, d=128` that's ~40 KiB — fits 4× in an H100 SM.

## Mask handling

PyLate and `flash-maxsim` handle masks differently:

- PyLate multiplies similarity by the mask *after* the einsum (`S * mask`).
  A masked position becomes `0`, which can still win the `max` if the true
  scores were all negative.
- `flash-maxsim` applies the mask *after* the forward kernel (post-scale).
- `flash-colbert` applies the mask *inside* the kernel, before every
  `tl.max` reduction, by overwriting masked positions with `−inf`. This has
  two benefits:
  1. Masked positions never influence the argmax (semantic correctness).
  2. One fewer pass over the output tensor (we never need a post-kernel
     elementwise op).

## Backward

Two observations:

- `max` is subdifferentiable: the subgradient picks one winner per (i, j, s)
  triple and zero everywhere else. For `d(max(·))/dS[·]` we need just the
  argmax index.
- The sum-over-s and `q_mask` are linear in `S`, so

      grad_Q[i, s, :] = q_mask[i, s] · Σ_j grad_scores[i, j] · D[j, argmax[i,j,s], :]
      grad_D[j, t, :] = Σ_{(i,s) : argmax[i,j,s] == t} q_mask[i, s] · grad_scores[i, j] · Q[i, s, :]

The forward kernel optionally writes `[Nq · Nd, Lq]` int32 argmax indices —
that's 4 MB for `Nq=Nd=64, Lq=256`, acceptable at every training scale we
care about.

`grad_Q` is embarrassingly parallel: one program per `(i, s)` loops over
`j ∈ Nd`, gathers `D[j, argmax[i,j,s], :]`, weighted sum, write.

`grad_D` has output contention: the same `(j, t)` slot receives contributions
from many `(i, s)` pairs. We use `tl.atomic_add` in **fp32** (not fp16 like
`flash-maxsim`'s backward). H100 fp32 atomics run at ~1 cycle / element and
are fully deterministic under tie-breaking rules that don't depend on
thread-execution order, so the result is bitwise reproducible between runs
as long as argmax ties break the same way (and they do — `tl.argmax` is
deterministic).

Alternative backward path (future work — toggle `save_argmax=False`): redo
the forward max in the backward, picking the winner on the fly. Saves the
argmax buffer at the cost of one extra GEMM. Analogous to FlashAttention's
recompute backward.

## Varlen / packed path

For reranking and ColPali-style workloads, per-doc length variance is large;
padding everything to `Ld_max` wastes ~50 % of FLOPs and memory. We support a
varlen path where both `Q` and `D` arrive as stacked `[total_tokens, d]`
tensors with cumulative-length offsets (FlashAttention's `cu_seqlens`
convention).

The kernel is the padded one with two changes:

1. `q_hi, q_lo = cu_q[i], cu_q[i+1]`; loads use the offset base.
2. The outer bound is `max_lq` (the static upper bound used for
   constant-folding) but the `q_valid = q_off < (q_hi - q_lo)` check
   short-circuits past-end tiles.

Empty sequences (`q_hi == q_lo`) produce a score of 0 without entering the
loop — same semantics as the padded path.

## Log-sum-exp (soft) variant

    score_soft[i, j] = (1/β) · Σ_s logsumexp_t(β · ⟨Q[i,s], D[j,t]⟩)

With `β → ∞` this recovers hard MaxSim. For training, a finite `β` produces
softer gradients: all doc tokens contribute a softmax-weighted share, not
just the argmax. This is structurally a FlashAttention kernel (`Q·K → softmax
→ output`) with the output being a scalar instead of a value projection —
we reuse the online-softmax streaming update `(m, ℓ)`.

Recommended `β`:
- Training from scratch: `β = 5..10` (prevents winner-takes-all collapse).
- Fine-tuning: `β = 20..50`.
- Eval: fall back to `maxsim` (`β → ∞`).

## Autotune

We autotune over `BLOCK_Q × BLOCK_D × num_warps × num_stages` on the product
key `(Lq, Ld, d_pad, has_q_mask, has_d_mask)`. Configs are pruned by:

- Shared-memory budget (≤ 224 KiB on H100, ≤ 164 KiB on A100).
- Block larger than the problem (`BLOCK_Q > 2·Lq`).

The shortlist is tuned for H100 vs A100/A10 (Hopper loves
`num_stages=3` plus warp-specialized `tl.dot`; A100 prefers `num_stages=2`).
Autotune runs once per key per process and caches the winner.

## Numerical accuracy

We accumulate in fp32 everywhere except the `Q·Dᵀ` GEMM, where we use
bf16/fp16 inputs with fp32 accumulator (the tensor-core native path). After
the GEMM the `max` and `sum` reductions are all in fp32 — no more precision
loss after the GEMM.

Parity tests enforce `max |out − ref| / max |ref| < 5 × 10⁻³` for fp16 and
`2 × 10⁻²` for bf16 inputs, matching the expected ULP drift of a
tensor-core GEMM of the given shape.
