# Changelog

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
