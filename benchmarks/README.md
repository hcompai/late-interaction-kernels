# benchmarks/

Scripts that measure `late-interaction-kernels` against the implementations it
replaces. The headline numbers and the analysis behind them live in
[`../docs/benchmarks.md`](../docs/benchmarks.md), along with the hardware, the
software stack, and the pinned baseline versions every table was produced on.
This file is the operator's guide: what each script measures and how to run it.

## Common CLI conventions

All bench scripts share the same flag shape:

| flag | purpose |
| --- | --- |
| `--only NAME [NAME ...]` | Run a subset of the script's experiment / shape list (default: all). |
| `--variants {...}` | Run a subset of impl variants (e.g. `vanilla` / `lik` / `flash` / `compile` / `both` / `all`). Only in benches that compare variants in lockstep. |
| `--outdir DIR` | Where to write `*.json` and `*.md` artifacts (default `benchmarks/results`). |
| `--dtype {bf16,fp16}` | Input dtype where applicable. The accumulator is always fp32. |
| `--quick` | Pre-baked small-subset filter on a few scripts; prefer `--only` for explicit control. |

Every script records peak GPU memory per variant: `peak_mb` in the JSON output
(uniformly MB across the directory), and GB-formatted in stdout where the
values warrant it (e2e training shapes reach 25+ GB). Pass `--help` on any
script for its full option list and the values `--only` accepts.

`bench_colpali_e2e.py` and `bench_pylate_e2e.py` are the exception to the flag
shape: they run a single (batch size, variant) cell per process — so an OOM is
an isolated, recorded outcome — and take
`--variant vanilla|lik --batch-size B --output FILE.json`. The sweep loops live
in `scripts/sky_colpali_e2e.yaml` / `scripts/sky_pylate_e2e.yaml`.

## What each script measures

### Microbenchmarks (kernel-level)

| script | what it isolates |
| --- | --- |
| `bench_forward.py` | Fused forward `maxsim` vs eager fp32 einsum vs `torch.compile` across 14 shapes. |
| `bench_chunking.py` | Long-query (`Lq > 512`) chunking: forward + training vs the un-chunked core, with flash as an external reference. |
| `bench_inference_edge.py` | Small-d (`d ∈ {48, 64}`) edge ColBERT regimes, `inference_mode`. |
| `bench_normalize.py` | Fused `normalize=True` vs explicit `F.normalize` + `maxsim`. |
| `bench_backward_method.py` | grad_D paths: `auto` vs `unified` vs `lowmem` vs naive. |
| `bench_backward_lowmem.py` | Backward time + peak memory: `lowmem` vs `unified` on PyLate/ColPali shapes, with flash-maxsim and a PyLate-naive einsum baseline. |
| `bench_lateon.py` | LateOn / LateOn-Code shapes (Ld up to 16 384, d=128). |
| `bench_compile_cache.py` | Cold-pass autotune cost across 18 distinct `Ld` values. |
| `bench_flash_maxsim.py` | Head-to-head vs `flash-maxsim` (same Triton-MaxSim math). |
| `bench_fp8.py` | FP8 inference on Hopper vs bf16 `maxsim`. |
| `bench_fused_head_train.py` | Fused `maxsim_from_hidden` (head + L2-normalize + maxsim) vs unfused. |

### PLAID / FastPlaid

| script | what it isolates |
| --- | --- |
| `bench_fastplaid_e2e.py` | `fast_plaid.engine.search()` vs our scoring kernel on the same on-disk compressed index. |
| `bench_decompress_maxsim.py` | Fast-plaid's decompress + rerank pipeline (PyTorch transliteration) vs `maxsim_residual` / `maxsim_residual_varlen`. |
| `bench_cached_maxsim.py` | PyLate `CachedContrastive`'s chunked MaxSim vs vanilla vs `torch.compile` vs LIK. |

### End-to-end (real model + loss)

| script | what it drives |
| --- | --- |
| `bench_colpali_e2e.py` | Real ColQwen2 + LoRA training steps (`ColModelTraining` recipe) on a subset of `vidore/colpali_train_set`, with the loss head instrumented: per-MaxSim-call VRAM (in-train + exact isolated replay), step times, whole-run peak, OOM-as-outcome. `--variant vanilla\|lik` toggles `patch_colpali_engine()`. Adapted from the harness in [colpali PR #412](https://github.com/illuin-tech/colpali/pull/412). |
| `bench_pylate_e2e.py` | The PyLate sibling: real `SentenceTransformerTrainer` steps (GTE-ModernColBERT-v1 + `Contrastive`) on MS MARCO triplets, same instrumentation/replay/OOM-as-outcome design. `--variant vanilla\|lik` toggles `patch_pylate()`; `--score-mini-batch-size` exercises PyLate's own chunking mitigation. |
| `bench_colpali_loss.py` | MaxSim loss-head isolation on synthetic embeddings (no encoder) — the explicit-negative heads. |
| `summarize_colpali_e2e.py` / `summarize_pylate_e2e.py` | Render the e2e sweep results: op-VRAM table, batch-ceiling table, log-log plot. |

### Platform-specific

| script | platform |
| --- | --- |
| `bench_mps.py` | Apple Silicon (MPS). Metal kernel vs `torch.compile` vs eager. |

## Running

Install the package from the repo root first (`pip install -e ".[dev,pylate]"`).
The comparison benches each need their baseline installed too; the exact pinned
versions are in [`../docs/benchmarks.md`](../docs/benchmarks.md#baseline-package-versions):

| baseline | needed by |
| --- | --- |
| `flash-maxsim` | `bench_flash_maxsim.py`, `bench_forward.py`, `bench_chunking.py` |
| `fast-plaid` | `bench_fastplaid_e2e.py` |
| `colpali-engine` (`==0.3.16`) | `bench_colpali_e2e.py`, `bench_colpali_loss.py` |
| `pylate` (`==1.5.0`) | `bench_pylate_e2e.py`, `bench_cached_maxsim.py` (installed by the `pylate` extra) |

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

A failure in one script does not stop the rest; each runs inside a non-fatal
`run()` helper.

### On a SkyPilot cluster

The SkyPilot jobs in `scripts/` provision a 1×H100 box, install the pinned
baselines, and run the benches.

```bash
# every table in docs/benchmarks.md (~25 min)
sky launch -c lik-bench-all scripts/sky_run_all_benchmarks.yaml -y

# a subset, by bench tag (~5 min)
sky launch -c lik-bench-smoke scripts/sky_run_all_benchmarks.yaml -y \
    --env RUN_ONLY="forward cached_maxsim fused_head_train fp8"
```

`RUN_ONLY` takes a space-separated list of bench tags; the `envs:` block in
`scripts/sky_run_all_benchmarks.yaml` lists them all. The per-stack jobs target
specific tables and pin their own baselines:

| job | covers |
| --- | --- |
| `scripts/sky_colpali_e2e.yaml` | ColQwen2 e2e MaxSim-VRAM batch sweep (vanilla vs LIK) + optional loss isolation |
| `scripts/sky_pylate_e2e.yaml` | PyLate e2e MaxSim-VRAM batch sweep (vanilla vs LIK + chunked-vanilla cells) |
| `scripts/sky_benchmark_smoke_test.yaml` | two-bench smoke test (`bench_forward` + `bench_backward_method`) |
| `scripts/sky_decompress_bench.yaml` | PLAID decompress + MaxSim |
| `scripts/sky_fastplaid_e2e.yaml` | rerank vs `fast_plaid.engine.search()` |

See each file's header for its exact scope.

## Output

Each script writes to `benchmarks/results/` (overridable via `--outdir`):

* `*.json` — full result rows including per-variant timing, peak VRAM, and
  shape metadata. This is the format the headline tables in `docs/benchmarks.md`
  are generated from.
* `*.md` — pre-rendered Markdown table for the same data, ready to paste into a
  report.

The directory is `.gitignore`d. **Do not commit anything under
`benchmarks/results/`** (see `../AGENTS.md`).

## Conventions

* **Numerics.** Baselines run their inner einsum / matmul with an fp32
  accumulator (matching the fused kernel) and read bf16 / fp16 inputs; parity is
  asserted before timing. See the fair-comparison protocol in
  [`../docs/benchmarks.md`](../docs/benchmarks.md) for the exact tolerance.
* **Warmup.** Every timed loop runs ≥ 5 untimed warmup iterations before the
  measurement window.
* **Timing.** CUDA events for kernel-level benches; `time.perf_counter` with an
  explicit `torch.cuda.synchronize()` for end-to-end benches.
* **Memory.** Peak VRAM via `torch.cuda.reset_peak_memory_stats()` +
  `max_memory_allocated()` around each variant.
