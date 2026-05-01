# Design notes

## Problem

For two batches of token embeddings

    Q ∈ ℝ^{Nq × Lq × d}        D ∈ ℝ^{Nd × Ld × d}

the ColBERT MaxSim score is

    score[i, j] = Σ_s  max_t  ⟨Q[i, s], D[j, t]⟩

with optional per-batch boolean masks (`q_mask` drops rows from the sum,
`d_mask` hides doc tokens from the inner max). Output: `[Nq, Nd]`.

PyLate's reference implementation materializes the full similarity tensor
`S ∈ ℝ^{Nq · Nd · Lq · Ld}` before reducing. At ColPali scale
(`Lq = Ld = 1024`) that's 4.5 GB per tensor, and training loops allocate
3-4 such tensors per step. The fused kernel never puts `S` in HBM.

## Forward

One Triton program per `(q_batch, d_batch)`. Inside, stream `D` in tiles
while holding a `[BLOCK_Q]` running max in registers:

```
for q_start in 0, BLOCK_Q, ..., Lq:
    load Q_block [BLOCK_Q × d]                 → SRAM
    m      ← −inf [BLOCK_Q]
    argmax ← 0    [BLOCK_Q]

    for d_start in 0, BLOCK_D, ..., Ld:
        load D_block [BLOCK_D × d]             → SRAM
        S = Q_block @ D_blockᵀ                 (tensor cores, fp32 acc)
        S ← where(d_active, S, −inf)           (mask fused in)
        tile_max    = max(S, axis=1)
        tile_argmax = argmax(S, axis=1) + d_start
        argmax ← where(tile_max > m, tile_argmax, argmax)
        m       ← max(m, tile_max)

    m ← where(finite(m) & q_active, m, 0)
    score_acc += sum(m)

store scores[q_batch, d_batch] ← score_acc
store argmax[q_batch, d_batch] ← argmax        (if needed)
```

Same outer-product tiling as FlashAttention with the softmax replaced by
`max` and the `P · V` projection dropped. Online max is numerically exact
(nothing to rescale).

SRAM use per program: `BLOCK_Q · d + BLOCK_D · d + BLOCK_Q · BLOCK_D` fp16
values + a `[BLOCK_Q]` fp32 accumulator. ~40 KiB at typical block sizes,
fits 4× per H100 SM.

### Mask handling

Masked positions are written as `−inf` *before* every `tl.max` reduction, so
they can't influence the argmax even when scores would otherwise be
negative. PyLate (post-multiplied 0/1 mask) and `flash-maxsim` (post-kernel)
both apply the mask later — neither reads back from HBM.

## Backward

The forward is `score = Σ_s max_t S` with `S` linear in Q and D. `max` is
sub-differentiable: only the argmax position contributes a gradient.

    grad_Q[i, s] = q_mask[i, s] · Σ_j grad_scores[i, j] · D[j, argmax[i,j,s]]
    grad_D[j, t] = Σ_{(i,s): argmax[i,j,s] == t} q_mask[i, s] · grad_scores[i, j] · Q[i, s]

The forward optionally writes `[Nq · Nd, Lq]` int32 argmax indices (4 MB
for `Nq = Nd = 64, Lq = 256`).

`grad_Q` is embarrassingly parallel: one program per `(i, s)` loops over
`j ∈ Nd`, gathers `D[j, argmax[i,j,s]]`, weighted-sum, write.

`grad_D` has output contention: many `(i, s)` pairs map to the same
`(j, t)`. Three kernels, selected by `set_backward_method(...)` or per-call
via `maxsim(..., backward=...)`:

1. **`unified`** (default for typical training shapes) — single-pass fused
   `grad_Q + grad_D` in one kernel. Hoists `Q[i, s]` out of the doc-batch
   loop, halving HBM read traffic versus the two-pass variants. `grad_D`
   uses fp32 `tl.atomic_add`.

2. **`csr`** (auto-picked at very high contention:
   `Nq ≥ 256 ∧ Nd ≥ 256 ∧ Lq ≤ 64`) — sort `(i, s)` by argmax into per-`j`
   CSR buckets; each `(j, t)` program reduces its bucket in registers and
   writes once. **Bitwise-deterministic across runs.** Build cost: a few
   hundred µs of cub radix-sort.

3. **`atomic`** (legacy two-pass, never picked by `auto`) — kept for
   benchmarking and as a fallback on GPUs with degraded fp32 atomics.

| method     | bitwise-reproducible | when to pick                          |
| ---------- | :------------------: | ------------------------------------- |
| `unified`  | no (atomic, ≤1e-6 r) | default for training                  |
| `csr`      | yes                  | regression suites, very high contention |
| `atomic`   | no (atomic, ≤1e-6 r) | fallback only                         |

`tl.argmax` is stable (lowest-index tie-break) on all three paths — only
the `grad_D` reduction order differs.

## Varlen / packed path

For reranking and ColPali-style heterogeneous lengths, padding to `Ld_max`
wastes ~50 % of FLOPs and memory. `maxsim_varlen` consumes packed
`[total_tokens, d]` tensors with `cu_seqlens` offsets (FlashAttention
convention). The kernel is the padded one with two changes:

1. `q_lo, q_hi = cu_q[i], cu_q[i+1]`; loads use the offset base.
2. The outer bound is `max_lq` (compile-time constant); the runtime
   `q_valid` mask drops past-end tiles.

Empty sequences contribute zero, same as the padded path. The fused
backward writes per-pair argmax into the packed gradient layout via fp32
`atomic_add`.

`maxsim_inference_scatter` is the same forward but for arbitrary
`(query_index, doc_index)` pairs from the packed batches — output is
`[num_pairs]` rather than `[Nq, Nd]`. Use it when the pair list is
sparse relative to `Nq · Nd` (typical reranker scheduling, e.g. inside
vLLM).

## PLAID / ColBERTv2

`maxsim_residual` and `maxsim_residual_varlen` decompress
`(codes, residuals)` to dense embeddings on the fly in SRAM, optionally
L2-normalize, and run MaxSim — all in a single pass.

* `codes[..., t]` indexes into a centroid table.
* `residuals[..., t, :]` packs `8 / nbits` bucket codes per byte
  (little-endian within the byte). `nbits ∈ {2, 4, 8}`.
* `bucket_weights[b]` is the scalar offset added on top of the centroid
  feature.

Output never materializes a dense `[Nd, Ld, d]` embedding tensor.
`maxsim_residual_varlen` also skips the `[Ntop, max_Ld, packed_dim]`
attention-mask scratch. `plaid_approx_score` is the IVF-prune step
(gather centroid scores, mask, max-then-sum) fused into one kernel.

## Fused D-side head

`maxsim_from_hidden` and `maxsim_from_hidden_train` fold
`F.linear(H_d, W, b) → F.normalize → MaxSim` into a single kernel. The
`[Nd, Ld, d_out]` `D_proj` scratch is never written to HBM; for corpora
stored as `[Nd, Ld, d_model]` ModernBERT hidden states this saves
multi-GB of read traffic.

The training backward is closed-form. The forward saves an `[Nq · Nd, Lq]`
argmax. Backward gathers `H_d` only at winning positions
(`Nq · Nd · Lq` rows, typically <10 % of `Nd · Ld`), recomputes
`D_unnorm_win = H_win @ Wᵀ + b` in `compute_dtype` with fp32 accumulator,
applies the L2-normalize Jacobian directly, and computes
`grad_Q / grad_W / grad_b` with three small matmuls. `grad_H_d` is
scattered with `index_add_` in the input dtype — no fp32
`[Nd · Ld, d_model]` buffer.

The win shrinks as `Nq · Lq / Ld → 1` (square reranking-shaped batches)
and as `d_model` shrinks below ~128.

## Autotune

Tuned over `BLOCK_Q × BLOCK_D × num_warps × num_stages` keyed on
`(Lq, Ld, d_pad, mask flags, normalize)`. Configs are pruned by:

* shared-memory budget (≤ 224 KiB on H100, ≤ 164 KiB on A100),
* `BLOCK_Q > 2 · Lq` (block bigger than the problem).

Hopper shortlist enables `num_consumer_groups` warp specialization
(Triton ≥ 3.2, FA-3 style); fallback is transparent on older Triton.
Autotune runs once per key per process and caches the winner.

## Numerical accuracy

Inputs can be fp16 / bf16 / fp32; the GEMM uses bf16/fp16 inputs with
fp32 accumulator (the tensor-core native path), and every reduction
afterwards is in fp32.

Parity tests enforce `max |out − ref| / max |ref| < 5 × 10⁻³` for fp16
and `2 × 10⁻²` for bf16, matching the expected ULP drift of a
tensor-core GEMM.
