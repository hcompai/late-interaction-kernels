# `late-interaction-kernels` vs `personal/maxsim`

Side-by-side comparison of the two MaxSim repositories on this machine.

| Path | Role |
| --- | --- |
| `/Users/tony.wu/code/late-interaction-kernels` | H Company's production library — fused **Triton/Metal** MaxSim kernels with full training/retrieval surface. |
| `/Users/tony.wu/code/personal/maxsim` | Erik Kaum's experimental **HF `kernels`-hub** package — single forward-only MaxSim kernel, hand-written **CUDA + Metal**. |

Both projects compute the same primitive

```
score(q, d) = Σ_i max_j ⟨q_i, d_j⟩
```

without materialising the full `[Lq, Ld]` similarity matrix, but they sit at very different points on the maturity / scope curve.

---

## 1. Scope & surface

| Aspect | `late-interaction-kernels` | `personal/maxsim` |
| --- | --- | --- |
| Public entry points | `MaxSimScorer`, `retrieve`, `patch_pylate`, `maxsim`, `maxsim_inference`, `maxsim_varlen`, `maxsim_inference_scatter`, `maxsim_from_hidden(_train)`, `maxsim_residual(_varlen)`, `plaid_approx_score`, `maxsim_inference_fp8` + `experimental.{soft_maxsim, smooth_maxsim, maxsim_matryoshka, maxsim_xtr}` | `score_pairs_packed`, `score_candidates_padded` (+ `*_reference`) |
| Forward | ✅ (dense, varlen, scatter, residual, FP8) | ✅ (packed ragged + padded wrapper) |
| Backward / autograd | ✅ — `maxsim`, `maxsim_varlen`, `maxsim_from_hidden_train`, three backward strategies (`unified`, `atomic`, `csr`, `auto`) | ❌ Forward-only (README: "No backward pass") |
| Top-k retrieval | ✅ `retrieve(Q, D, top_k, chunk)` with chunked HBM bound | ❌ |
| PLAID / ColBERTv2 compressed indices | ✅ `maxsim_residual{,_varlen}`, `plaid_approx_score` | ❌ |
| FP8 (Hopper) | ✅ `maxsim_inference_fp8` + `fp8` submodule | ❌ |
| PyLate drop-in | ✅ `patch_pylate()` / `unpatch_pylate()`, `LIK_DISABLE` kill switch | ❌ |
| Experimental kernels | ✅ `soft`, `smooth`, `matryoshka`, `xtr` | ❌ |
| Reference (pure PyTorch) | ✅ `late_interaction_kernels.reference` | ✅ `maxsim.reference` |

`late-interaction-kernels` is a multi-kernel **library**; `maxsim` is essentially **one well-engineered kernel** plus its packed/padded Python API.

---

## 2. Implementation strategy

| | `late-interaction-kernels` | `personal/maxsim` |
| --- | --- | --- |
| GPU language | **Triton** (Python-DSL JIT) | **Hand-written CUDA** (`.cu`, ~27 KB), bf16 WMMA / `mma.sync.16x16x16` |
| Apple Silicon | Metal `simdgroup_matrix` kernel + `torch.compile` fallback (per-call heuristic) | Metal `simdgroup_matrix` kernel (`maxsim.metal` 58 KB + `maxsim.mm` 31 KB ObjC++) |
| CPU / other | Pure-PyTorch `reference` (autograd-aware) | Pure-PyTorch `reference` |
| Packaging | Standard PyPI wheel via `hatchling` + `hatch-vcs`; `pip install late-interaction-kernels` | **Hugging Face `kernels` hub** package (`build.toml`, `kernel-builder`, Nix `flake.nix`); `kernels.get_kernel("erikkaum/maxsim", version=1)` |
| Build system | `pyproject.toml`, `uv` | `build.toml` + `flake.nix` + `justfile` (HF Jobs for CUDA builds, Docker linux/amd64 fallback) |
| Autotune / caching | Per-`(Lq, Ld, d, masks)` autotune cache on CUDA; MPS compile cache keyed on `(dtype, normalize, has_q_mask, has_d_mask)` | Static kernel — fast path requires `dim % 16 == 0` (CUDA) / `dim % 8 == 0` (Metal); scalar fallback otherwise |

---

## 3. Hardware support

| Target | LIK | maxsim |
| --- | --- | --- |
| NVIDIA Ampere (sm_80, A100/A10) | ✅ tuned | ✅ tuned |
| Ada / Lovelace (sm_86 / 89, L4/L40/4090) | ✅ tuned | ✅ tuned |
| **Hopper (sm_90, H100/H200)** | ✅ primary target — FP8 WGMMA, warp-specialized (Triton ≥ 3.2) | ⚠️ PTX forward-compat only; **no WGMMA** |
| Blackwell | ✅ (FP8 path auto-fallback) | — |
| Apple Silicon (MPS) | ✅ Metal kernel **+** `torch.compile` (auto-picked per call) | ✅ Metal kernel |
| CPU / Windows | ✅ eager reference | reference only |

---

## 4. Repo layout & weight

`late-interaction-kernels` (≈21 k LOC across `late_interaction_kernels/`, tests, benchmarks, docs):

```
late_interaction_kernels/  forward.py varlen.py backward{,_csr,_unified}.py
                           autograd.py retrieve.py topk.py scatter.py
                           plaid.py fused_head.py fp8.py matryoshka.py
                           metal.py _mps.py pylate_compat.py reference.py
                           experimental/{soft,smooth,xtr,matryoshka}.py ...
benchmarks/  18 bench scripts (forward, backward, fp8, mps, fastplaid, lateon...)
tests/       25 test modules
docs/        benchmarks.md, design.md, packed_training.md, supported_models.md
scripts/     SkyPilot YAMLs for cloud benchmarking
```

`personal/maxsim` (≈4 k LOC, mostly the two GPU kernels):

```
maxsim_cuda/    maxsim.cu (27 KB) + dev_binding.cpp
maxsim_metal/   maxsim.metal (58 KB) + maxsim.mm (31 KB)
torch-ext/      torch_binding.{cpp,h} + maxsim/{__init__.py, reference.py}
tests/          6 test modules (correctness, dtypes, padded, ragged)
benchmarks/     benchmark.py + run_local.py
scripts/        cuda_dev.py, cuda_bench.py, cuda_release.sh  (HF Jobs dev loop)
example.py, justfile, build.toml, flake.nix
```

---

## 5. API ergonomics

**Calling MaxSim on a packed ragged batch**

```python
# late-interaction-kernels
from late_interaction_kernels import maxsim_varlen
scores = maxsim_varlen(Q_packed, cu_q, D_packed, cu_d, ...)

# personal/maxsim
from kernels import get_kernel
maxsim = get_kernel("erikkaum/maxsim", version=1)
scores = maxsim.score_pairs_packed(
    queries, query_offsets, documents, document_offsets,
    pair_query_ids, pair_document_ids,
)
```

Key API difference: `maxsim.score_pairs_packed` takes **explicit `(query_id, doc_id)` pair lists**, much like the LIK `maxsim_inference_scatter` reranking entry point — it's pair-list scoring, not a `[Nq, Nd]` matrix. LIK's `maxsim` / `maxsim_inference` return the full cross-score matrix; LIK's `maxsim_inference_scatter` is the closer analogue.

`score_candidates_padded` mirrors a `[B, C, Ld, D]` reranking layout; LIK covers the same shape implicitly via `maxsim_inference` + masks.

---

## 6. Benchmarks (as reported in each README)

| Workload | `late-interaction-kernels` (H100) | `personal/maxsim` (A10G / L4 / A100, M3 Pro) |
| --- | --- | --- |
| Reranking vs naive einsum | **7–23×** | **2.0–5.3×** (per-GPU, see table) |
| Long-context (Ld ≥ 8k) | runs; naive OOMs | **1.9–6.2×** (LongDocStress) |
| PyLate cached-contrastive + bwd | up to **13.8×** | n/a (no backward) |
| PLAID rerank vs FastPLAID | **19–30×** | n/a |
| FP8 MaxSim (Hopper) | up to **1.4×** | n/a |
| LateOn-Code-edge e2e training | **1.04–1.27×** | n/a |
| MPS (Apple M3 Pro, fp16, dim=128) | 1.9–3.2× over PyTorch, 1.1–2.0× over `torch.compile` | **2.2–3.8×** vs naive PyTorch |

LIK's numbers are end-to-end on H100 and cover training; `maxsim` benches forward-only on Ampere/Lovelace + Apple Silicon, where it's competitive on inference shapes but doesn't claim Hopper or training workloads.

---

## 7. Licensing, authorship, intent

| | LIK | maxsim |
| --- | --- | --- |
| License | Apache-2.0 | Apache-2.0 |
| Authors | Aurélien Lac, Tony Wu (H Company) | Erik Kaum (`erikkaum`) — referenced as `personal/maxsim` |
| Distribution | PyPI (`pip install late-interaction-kernels`) | HF kernels hub (`erikkaum/maxsim`) |
| Stated goal | Drop-in replacement for the MaxSim math in PyLate / FastPLAID training + retrieval pipelines | Demonstrate a clean exact MaxSim kernel packaged as a `kernels` hub artifact |

---

## 8. TL;DR

- **`late-interaction-kernels`** — production library. Triton-first, full training+retrieval surface, FP8/Hopper-tuned, PyLate drop-in, autograd backwards, PLAID/ColBERTv2 compressed support, MPS path, 25 test modules. Owns the long-context + training story.
- **`personal/maxsim`** — focused, hand-written CUDA + Metal MaxSim forward, packaged for the HF `kernels` hub. Smaller surface (two entry points), forward-only, no backward / no FP8 / no PLAID, but interesting as a single-purpose kernel with its own padded-API design and HF-Jobs-based CUDA dev loop.

If you want **one thing** to take from `maxsim` into LIK: its `score_pairs_packed` explicit pair-list API (avoid forming a `[Nq, Nd]` score matrix when the caller already has a reranking pair list) and the `_pack_padded` zero-sync conversion to packed layout are clean ideas. The reverse direction — what `maxsim` is missing — is essentially everything LIK does on top of the forward kernel: backward, retrieval, PLAID, FP8, PyLate integration, autotune.
