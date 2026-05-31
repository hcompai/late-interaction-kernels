# benchmarks/

Scripts that measure `late-interaction-kernels` against the reference
implementations it replaces. The headline numbers + analysis live in
[`../docs/benchmarks.md`](../docs/benchmarks.md); this file is the
operator's guide to *what's here and how to drive it*.

## Common CLI conventions

All bench scripts follow the same flag shape:

| flag | purpose |
| --- | --- |
| `--only NAME [NAME ...]` | Run a subset of the script's hard-coded experiment / shape list (default: run all). |
| `--variants {...}` | Run a subset of impl variants (e.g. `vanilla` / `lik` / `flash` / `compile` / `both` / `all`). Only present in benches that compare variants in lockstep. |
| `--outdir DIR` | Where to write `*.json` and `*.md` artifacts (default `benchmarks/results`). |
| `--dtype {bf16,fp16}` | Input dtype where applicable. Accumulator is always fp32. |
| `--quick` | Pre-baked "small subset" filter on a few scripts; prefer `--only` for explicit control. |

Every script also records peak GPU memory per variant — `peak_mb` in
JSON output (uniformly MB across the directory), and GB-formatted in
stdout where the values warrant it (e.g. e2e training shapes at 25+
GB). Handy for comparing memory footprints alongside wall-clock.

Pass `--help` on any script for the full list of options and the
choices accepted by `--only`.

## What each script measures

### Microbenchmarks (kernel-level)

| script | what it isolates |
| --- | --- |
| `bench_forward.py` | Fused forward `maxsim` vs eager fp32 einsum vs `torch.compile` across 14 shapes. |
| `bench_inference_edge.py` | Small-d (`d ∈ {48, 64}`) edge ColBERT regimes, `inference_mode`. |
| `bench_normalize.py` | Fused `normalize=True` vs explicit `F.normalize` + `maxsim`. |
| `bench_backward_method.py` | grad_D paths: `auto` vs `unified` vs `csr` vs `atomic` vs naive. |
| `bench_backward_unified.py` | Backward-only timing of the unified kernel vs two-pass paths. |
| `bench_training.py` | Full training step (forward + backward) speed and peak memory, with flash as an external reference. |
| `bench_backward_0_5.py` | Fused `maxsim_residual` / `maxsim_varlen` backward vs "unpack + autograd". |
| `bench_lateon.py` | LateOn / LateOn-Code shapes (Ld up to 16 384, d=128). |
| `bench_compile_cache.py` | Cold-pass autotune cost across 18 distinct `Ld` values. |
| `bench_flash_maxsim.py` | Head-to-head vs `flash-maxsim` (same Triton-MaxSim math). |
| `bench_fp8.py` | FP8 inference on Hopper vs bf16 `maxsim`. |
| `bench_fused_head_train.py` | Fused `maxsim_from_hidden` (head + L2-normalize + maxsim) vs unfused. |

### PLAID / FastPlaid

| script | what it isolates |
| --- | --- |
| `bench_fastplaid.py` | Isolated rerank step (`bmm + mask + max + sum`) vs `maxsim` on the same shapes. |
| `bench_fastplaid_e2e.py` | `fast_plaid.engine.search()` vs our scoring kernel on the same on-disk compressed index. |
| `bench_decompress_maxsim.py` | Fast-plaid's decompress + rerank pipeline vs `maxsim_residual` / `maxsim_residual_varlen`. |
| `bench_cached_maxsim.py` | PyLate `CachedContrastive`'s chunked MaxSim vs vanilla vs `torch.compile` vs LIK. |

### End-to-end (real model + loss)

| script | what it drives |
| --- | --- |
| `bench_pylate_training.py` | PyLate `Contrastive` step on synthesized embeddings (no encoder). |
| `bench_pylate_lateon.py` | Real `pylate.models.ColBERT` (LateOn / LateOn-Code) — Contrastive or CachedContrastive. DDP-aware. |
| `bench_pylate_realdata.py` | Same as above but on real MS MARCO triplets. |
| `bench_colpali_training.py` | `colpali_engine.ColQwen2` step on synthetic images + queries. |
| `bench_colpali_realdata.py` | Same as above on real `vidore/docvqa_test_subsampled`. |

### Platform-specific

| script | platform |
| --- | --- |
| `bench_mps.py` | Apple Silicon (MPS). Metal kernel vs `torch.compile` vs eager. |

## Running

### One script

```bash
python benchmarks/bench_forward.py
python benchmarks/bench_forward.py --only text-short text-long      # subset of shapes
python benchmarks/bench_cached_maxsim.py --variants flash           # skip vanilla pylate
```

### All scripts locally

```bash
OUTDIR=benchmarks/results bash scripts/run_all_benchmarks.sh
```

Failures in one script do not stop the rest — each is wrapped in a
non-fatal `run()` helper.

### On a SkyPilot cluster

```bash
# every table in docs/benchmarks.md (1×H100, ~25 min)
sky launch -c lik-bench-all scripts/sky_run_all_benchmarks.yaml -y

# subset by bench tag (README headline numbers only, ~5 min)
sky launch -c lik-bench-smoke scripts/sky_run_all_benchmarks.yaml -y \
    --env RUN_ONLY="forward cached_maxsim fused_head_train fp8"
```

`RUN_ONLY` accepts a space-separated list of bench tags; see the
`envs:` block in `scripts/sky_run_all_benchmarks.yaml` for the full
tag list.

The other `scripts/sky_*.yaml` files target specific tables (PLAID,
ColPali training, LateOn-Code-edge, etc.) — see each file's header for
the exact scope.

## Output

Each script writes to `benchmarks/results/` (overridable via
`--outdir`). Per-script artifacts:

* `*.json` — full result rows including per-variant timing, peak VRAM,
  and shape metadata. The format that the headline tables in
  `docs/benchmarks.md` are generated from.
* `*.md` — pre-rendered Markdown table for the same data, ready to
  paste into a report.

The directory is `.gitignore`d. **Do not commit anything under
`benchmarks/results/`** (see `../AGENTS.md`).

## Conventions

* **Numerics.** All baselines run their inner einsum / matmul with an
  fp32 accumulator (matching the fused kernel) and read bf16 / fp16
  inputs. Parity is asserted at `atol=1e-2, rtol=1e-2` *before*
  timing so the speedup ratios are apples-to-apples.
* **Warmup.** Every timed loop runs ≥ 5 untimed warmup iterations
  before the measurement window.
* **Timing.** CUDA events for kernel-level benches; `time.perf_counter`
  with explicit `torch.cuda.synchronize()` for end-to-end benches.
* **Memory.** Peak VRAM via `torch.cuda.reset_peak_memory_stats()` +
  `max_memory_allocated()` around each variant.
