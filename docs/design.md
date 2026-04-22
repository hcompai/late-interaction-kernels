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
- `late-interaction-kernels` applies the mask *inside* the kernel, before every
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

`grad_D` has output contention: the same `(j, t)` slot receives
contributions from many `(i, s)` pairs. Three implementations live in
the codebase, selected by `set_backward_method(...)` (process-wide) or
per-call via `maxsim(..., backward="...")`:

1. **Unified** (`"unified"`, default for almost every training shape
   since 0.6.0) — single-pass fused `grad_Q` + `grad_D` in one Triton
   kernel, FlashAttention-2 style. Hoists `Q[i, s, :]` out of the
   inner doc-batch loop, roughly halving HBM read traffic versus the
   two-pass variants. `grad_D` accumulates with `tl.atomic_add` in
   fp32 (same numerical characteristics as `"atomic"` below). This is
   what you want for virtually all in-batch contrastive training.

2. **CSR** (`"csr"`, wins at very high `grad_D` contention) —
   scatter-free bucketed reduction. We sort the argmax per doc-batch
   into a CSR structure `(row_ptr, perm)` where
   `perm[j, row_ptr[j, t] : row_ptr[j, t+1]]` lists every `(i, s)`
   whose argmax is `t`. Each `(j, t)` program reduces its bucket in
   registers and writes once — no atomics at all, **bitwise-reproducible
   across runs**. Build cost is `Nd` independent cub radix-sorts of
   `Nq · Lq` int32 keys plus a batched `searchsorted`; typically a few
   hundred µs. Wins when `Nq ≥ 256 ∧ Nd ≥ 256 ∧ Lq ≤ 64`.

3. **Atomic** (`"atomic"`, legacy two-pass) — the pre-0.6.0 path, kept
   for comparison and as a fallback on GPUs where `tl.atomic_add(fp32)`
   is degraded. Reduction order depends on thread scheduling
   (≤ 1e-6 relative ULP drift, far below fp16 / bf16 input noise).

### The `"auto"` selector

`maxsim(..., backward="auto")` (the default) picks between `"unified"`
and `"csr"` based on problem shape:

```python
# late_interaction_kernels/autograd.py, _MaxSimFn.backward
high_contention = Nq >= 256 and Nd >= 256 and Lq <= 64
method = "csr" if high_contention else "unified"
```

The heuristic is empirical from a sweep on 1×H100; in every other case,
`"unified"` dominates because a single fused pass over `Q/D/argmax`
beats CSR's sort + empty-bucket overhead even at modest contention.
See `benchmarks/bench_backward_unified.py` for the sweep.

`"atomic"` is **never picked by `"auto"`** since 0.6.0 — it exists for
pinned-method benchmarking and as a manual fallback.

### Numerical determinism

| method     | bitwise-reproducible across runs | notes                            |
| ---------- | :------------------------------: | -------------------------------- |
| `unified`  | ✗ (`atomic_add`, ≤ 1e-6 rel.)   | fastest training default          |
| `csr`      | ✓                                | choose for regression reproducibility |
| `atomic`   | ✗ (`atomic_add`, ≤ 1e-6 rel.)   | legacy two-pass                   |

Argmax selection itself is fully deterministic (`tl.argmax` is stable)
in all three paths — only the `grad_D` reduction order differs.

### Alternative backward path (roadmap)

Recompute-style backward — toggle `save_argmax=False`, redo the forward
`max` in the backward and pick the winner on the fly. Saves the argmax
buffer at the cost of one extra GEMM. Analogous to FlashAttention's
recompute backward. Not yet shipped.

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

## Fused heads (v0.6+)

Two fused heads sit on top of the core MaxSim kernel and exist because the
naive `F.linear → F.normalize → maxsim` pipeline does three extra HBM
round-trips:

- **`maxsim_from_hidden`** (inference). Streams the hidden state
  `H ∈ ℝ^{Nd·Ld·d_model}` through a single kernel that fuses
  `D = H @ W + b`, L2-normalize, and MaxSim in one pass. `D_proj` is
  never written to HBM; for `d_model=768, d_out=128, Ld=8192` that
  saves 4 × `Nd·Ld·d_out` bytes of read traffic. Two-stage pipeline
  (load `H_tile` → project → normalize → score) keeps SMEM pressure
  under the H100 budget.
- **`maxsim_from_hidden_train`** (training, autograd-aware). Same
  forward as inference but saves the argmax-gather `H_win` instead of
  the whole `D_proj`, and in 0.8.0 computes the backward in **closed
  form** — no autograd rebuild.

  *Why the backward is much cheaper than the unfused path.* MaxSim's
  forward already commits to one winning doc token per query token:
  the argmax. Only those positions receive gradient signal. The unfused
  path doesn't know this and reprojects + normalizes all `Nd · Ld`
  tokens a second time on the way back. The fused backward gathers
  `H_d` at the `Nq · Nd · Lq` winning positions only — typically
  `Nq · Lq / Ld ≈ 1–10 %` of the work for ColBERT-style training
  shapes — then does the projection, normalize-VJP and MaxSim-VJP in
  closed form:

  ```
  grad_D_hat_win  = einsum('ijs,jsv->ijsv', grad_scores, Q)         # bf16 GEMM
  grad_D_unnorm   = normalize_vjp(grad_D_hat_win, D_hat_win, ‖D_unnorm‖)  # elementwise
  grad_Q          = einsum('ijsv,ijsv->isv', grad_D_hat_win, D_hat_win) ← trivial
  grad_H_win      = grad_D_unnorm @ Wᵀ                              # bf16 GEMM, fp32 acc
  grad_W, grad_b  = H_winᵀ @ grad_D_unnorm ,  grad_D_unnorm.sum(0)
  grad_H          = zeros[H_d.dtype].index_add_(argmax_flat, grad_H_win)
  ```

  The only fp32 allocations are small reductions (`grad_W`, `grad_Q`,
  per-token norms). `grad_H` lives in the input dtype and is scattered
  with `index_add_`, so peak memory is on par with or below the
  unfused path at long `Ld`. The speedup shrinks as `Nq · Lq / Ld → 1`
  (reranking-shaped square batches) and as `d_model` shrinks below
  ~128 (the `F.linear` pass stops being the bottleneck).

## Smooth MaxSim & top-K save (v0.7)

`smooth_maxsim` is a finite-K variant of `maxsim`:

    score_smooth[i, j] = Σ_s  mean( topK_t ⟨Q[i,s], D[j,t]⟩ )

For `K=1` it degenerates to hard MaxSim; for larger `K` it behaves like
a truncated log-sum-exp without the `β` tuning knob, and the backward
shares the gradient evenly across the `K` winners. The kernel is a
streaming argmax-union loop: at each `BLOCK_D` tile we run `K` rounds
of `tl.argmax`, masking out previously picked indices. The `[Nq·Nd·Lq·K]`
argmax buffer is written once in the forward and consumed by an
`index_add`-based backward.

## FP8 inference path (v0.7)

On Hopper/Blackwell the `Q·Dᵀ` GEMM can be issued in FP8 (e4m3) with
per-tensor or per-token scales, with the fp32 accumulator re-used
unchanged. The streaming max is unaffected. We provide `quantize_fp8`
helpers that pick a conservative power-of-two scale and a score-tie
fallback harness that re-runs a given `(i, j)` in bf16 when the FP8
score is within a configurable ULP-equivalent threshold of the runner
up — this preserves top-K ranking on retrieval benchmarks while
keeping ~80 % of the raw tensor-core speedup.

## Warp specialization (v0.7, Triton ≥ 3.2)

On Triton 3.2+ we compile the MaxSim forward with a warp-specialized
producer/consumer schedule (load-half / compute-half) via the
`tl.async_copy` and explicit `num_consumer_groups` hints, mirroring the
FA-3 pattern. On H100 this overlaps the `D_tile` load with the
previous `Q·Dᵀ` issue, buying another 10–20 % over the FA-2-style
pipeline. We fall back to the default schedule on older Triton
transparently.
