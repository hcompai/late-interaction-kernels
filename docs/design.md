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

### Long-query chunking

For long queries (`Lq > 512`, ColPali-scale visual patches), `maxsim()` splits
`Q` into fixed 128-token chunks and scores each as an independent batch through
the same kernel, then sums the per-chunk scores back per original query:

    nc = ceil(Lq / 128)
    Qc = Q.reshape(Nq * nc, 128, d)          # [Nq*nc, 128, d]
    scores_c = kernel(Qc, D, ...)             # [Nq*nc, Nd]
    scores = scores_c.view(Nq, nc, Nd).sum(1) # [Nq, Nd]

Because MaxSim is a sum over query tokens of a per-token max, this is
numerically exact. Autograd flows through the reshape and sum unchanged.

Two benefits:
* **GPU utilisation.** Without chunking, a long query serialises a long
  `static_range(Lq / BLOCK_Q)` loop inside each program. Chunking launches
  more, shorter programs that fill the SM array concurrently.
* **Autotune cache collapse.** The kernel always sees `Lq == 128`, so
  all ColPali-scale queries share one autotune entry (two with tail-padding)
  instead of one per `Lq` bucket.

The split fires only when `Lq > 512`; shorter queries fall through to the
direct kernel call unchanged. Chunking is cross-product-only — the KD /
pairs path (`maxsim` with 4-D `D`) is unaffected.

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
`(j, t)`. Kernels selected per-call via `maxsim(..., backward=...)`:

1. **`unified`** (default for cross-products) — single-pass fused
   `grad_Q + grad_D` in one kernel. Hoists `Q[i, s]` out of the doc-batch
   loop, halving HBM read traffic versus the two-pass variants. `grad_D`
   uses fp32 `tl.atomic_add`, accumulated in a full-size fp32 buffer that is
   cast to the input dtype at the end.

2. **`lowmem`** (auto-picked where gradient buffers dominate: KD /
   hard-negative layouts, and high contention `Nq ≥ 256 ∧ Nd ≥ 256 ∧
   Lq ≤ 64`) — each output row is destination-owned. A one-hot matmul over
   the saved argmax reduces `grad_D` in fp32 registers and stores it directly
   in the input dtype; `grad_Q` reuses the row-owned kernel into a bf16
   buffer. No full-size fp32 buffer, no fp32→bf16 transient, no atomics —
   so it roughly halves backward peak memory and is **deterministic**.

3. **`csr`** / **`atomic`** — sorted-argmax reduction and legacy two-pass;
   kept for back-compat and as fallbacks.

| method     | bitwise-reproducible | when to pick                          |
| ---------- | :------------------: | ------------------------------------- |
| `unified`  | no (atomic, ≤1e-6 r) | default cross-product (fastest)       |
| `lowmem`   | yes                  | KD / hard-negatives, high contention (½ memory) |
| `csr`      | yes                  | back-compat / determinism             |
| `atomic`   | no (atomic, ≤1e-6 r) | fallback only                         |

`tl.argmax` is stable (lowest-index tie-break) on all paths — only the
`grad_D` reduction order differs.

### Backward launch-param autotuning

Backward kernels have no block tiling to sweep, so they are autotuned over
`num_warps × num_stages` only (see `## Autotune → Backward`).

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

`score_pairs_packed` is the same forward but for arbitrary
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

`maxsim_from_hidden` folds `F.linear(H_d, W, b) → F.normalize → MaxSim`
into a single kernel; the backward (auto-dispatched on `requires_grad`)
gathers `H_d` only at winning positions. The `[Nd, Ld, d_out]`
`D_proj` scratch is never written to HBM; for corpora stored as
`[Nd, Ld, d_model]` ModernBERT hidden states this saves multi-GB of
read traffic.

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

### Forward

Tuned over `BLOCK_Q × BLOCK_D × num_warps × num_stages` keyed on
`(Lq, Ld, d_pad, mask flags, normalize)`. Configs are pruned by:

* shared-memory budget (≤ 224 KiB on H100, ≤ 164 KiB on A100),
* `BLOCK_Q > 2 · Lq` (block bigger than the problem).

Hopper shortlist enables `num_consumer_groups` warp specialization
(Triton ≥ 3.2, FA-3 style); fallback is transparent on older Triton.

With long-query chunking (see `## Forward → Long-query chunking`) ColPali-scale
queries always present `Lq = 128` to the kernel, collapsing the forward
autotune cache onto a small constant regardless of the original query length.

### Backward

Tuned over `num_warps × num_stages` (no block-tiling dimension) keyed on
`(Lq, d_pad, mask flags, kd_layout)`, so one entry covers all batch sizes in a
training regime.

Autotune runs once per key per process and caches the winner.

## Numerical accuracy

Inputs can be fp16 / bf16 / fp32; the GEMM uses bf16/fp16 inputs with
fp32 accumulator (the tensor-core native path), and every reduction
afterwards is in fp32.

Parity tests enforce `max |out − ref| / max |ref| < 5 × 10⁻³` for fp16
and `2 × 10⁻²` for bf16, matching the expected ULP drift of a
tensor-core GEMM.
