# Changelog

## 0.2.0 — CSR backward + auto-selector

### Added
- Scatter-free CSR (`"csr"`) backward path for `grad_D`. Per doc-batch
  cub radix sort of the argmax buffer, `torch.searchsorted` to build
  CSR row_ptr, then one Triton program per `(j, t)` output cell does
  a non-atomic bucket reduction. Zero contention, one write per cell.
- `flash_colbert.set_backward_method("auto"|"csr"|"atomic")` +
  `get_backward_method()` for process-wide selection.
- `"auto"` (default): empirical three-term heuristic picks CSR on
  (`Nq·Nd·Lq·d ≥ 1e8` | `Lq ≥ 1024` | `Nd ≥ 1024`), atomic otherwise.
- `benchmarks/bench_backward_method.py` — CSR vs atomic vs naive sweep
  across 10 training / retrieval / long-sequence shapes.
- 20 new tests (`tests/test_backward_csr.py`) — parity, equivalence,
  hot-bucket stress, empty-bucket handling, determinism, bf16, mask
  interactions, non-power-of-two `d`, `auto`-picks-correct-path.

### Changed
- Default grad_D path is now `"auto"` (was implicit `"atomic"` in 0.1).
- `docs/design.md` expanded with CSR design + measured numbers + the
  honest finding that CSR **does not always win** on H100 (its win
  regime is large-batch or long-sequence training).

### Performance (1×H100 80 GB HBM3, fp16 inputs, end-to-end forward+backward)
- Default `auto` matches the best manually-chosen path on 11/11 shapes.
- **CSR wins by ~1.3×–1.5×** on `train-256`, `long-Lq`, `huge-Nd`.
- **Atomic wins by ~1.2×–1.4×** on small/medium training shapes — H100
  hardware atomics coalesce at L2 faster than the CSR `sort` amortizes
  for those workloads.

## 0.1.0 — initial release

### Added
- Fused Triton forward kernel (`maxsim_forward`) with:
  - FP32 accumulator, bf16 / fp16 / fp32 inputs.
  - Fused `q_mask` (skiplist) and `d_mask` (attention mask), `-inf` applied
    inside the kernel so masked positions can never win the max.
  - Per-GPU autotune (Hopper / Ampere shortlists + shared-mem pruning).
  - Optional argmax output for the backward pass.
- Autograd-aware `maxsim` (`torch.autograd.Function`) wiring the forward +
  exact backward.
- Scatter-free `grad_Q` + deterministic fp32-atomic `grad_D`.
- `soft_maxsim` log-sum-exp streaming kernel (β-relaxation).
- `maxsim_varlen` packed kernel (cu_seqlens, FlashAttention-style).
- `flash_colbert.pylate_compat` drop-in monkey-patch:
  `patch_pylate()` / `unpatch_pylate()` / `FLASH_COLBERT_DISABLE=1`
  environment kill-switch.
- Pure-PyTorch reference (`maxsim_reference`, `maxsim_reference_soft`,
  `maxsim_reference_varlen`) used as ground truth in tests.
- 60 tests: forward parity (9 shapes × 2 dtypes), gradient parity with/without
  masks, varlen parity, soft-max approximation, edge cases (single-token,
  non-power-of-two `d`, non-contiguous inputs, fully-masked rows), PyLate
  compatibility.
- Benchmarks: forward (vs naive einsum, optional vs flash-maxsim),
  backward, end-to-end PyLate `Contrastive` step.
- SkyPilot YAMLs for one-shot CI (`sky jobs launch scripts/sky_test.yaml`)
  and a long-lived dev box (`sky launch -c ... scripts/sky_dev.yaml`).

### Performance (H100 80 GB, bf16, SXM)
- Forward: 7.9× – 22.7× vs naive einsum, ~4500× less HBM scratch.
- Backward: 1.3× – 2.3× vs naive.
- End-to-end PyLate `Contrastive` training step: 2.86× – 3.08× vs vanilla.
