# HuggingFace Kernels build

`late-interaction-kernels` is published to the
[HuggingFace Kernel Hub](https://github.com/huggingface/kernels) as
`Hcompany/late-interaction-kernels`. The Hub build exposes the **forward /
inference surface** of this repo: MaxSim, varlen MaxSim, FP8 MaxSim,
fused-head MaxSim, PLAID approx / residual. No `pip install`, no local
compile — the kernel is fetched and cached per-revision by `kernels`.

For training, autograd-aware scoring with backward variants, retrieval
helpers, PyLate patching, padded reranking, and research kernels, install
the PyPI package directly (`pip install late-interaction-kernels`).

## Consumer quickstart

```python
import torch
from kernels import get_kernel

lik = get_kernel("Hcompany/late-interaction-kernels")

# Dense MaxSim
scores = lik.maxsim_inference(Q, D, normalize=True)              # [Nq, Nd]

# Packed / variable-length docs
scores = lik.maxsim_varlen(
    Q_packed, D_packed, cu_seqlens_q, cu_seqlens_d, normalize=True,
)

# FP8 tensor-core MaxSim (Hopper / Blackwell; auto-fallback to bf16 elsewhere)
scores = lik.maxsim_inference_fp8(Q_fp8, D_fp8, scale_Q, scale_D)

# PLAID / ColBERTv2 rerank on compressed residuals
scores = lik.maxsim_residual_inference(
    Q, codes, residuals, centroids=centroids, bucket_weights=bw, nbits=2,
)

# ... and via the kernelize() layer path
from kernels import LayerRepository, use_kernel_mapping, kernelize, Mode

mapping = {"MaxSim": {
    "cuda": LayerRepository(
        repo_id="Hcompany/late-interaction-kernels", layer_name="MaxSim",
    ),
}}
with use_kernel_mapping(mapping):
    model = kernelize(model, mode=Mode.INFERENCE | Mode.TORCH_COMPILE)
```

## Public surface (v1)

| Symbol                       | Purpose                                                |
| ---------------------------- | ------------------------------------------------------ |
| `maxsim`                     | Core MaxSim (autograd-aware). `has_backward` on the layer is False, but the functional API still supports `.backward()` for consumers who need training gradients. |
| `maxsim_inference`           | Forward-only MaxSim. Smaller peak memory than `maxsim`. |
| `maxsim_varlen`              | Packed-tensor layout with `cu_seqlens` (autograd-aware; auto-skips argmax when no grad needed). |
| `maxsim_varlen_inference`    | Explicit forward-only variant of `maxsim_varlen`.      |
| `maxsim_inference_fp8`       | FP8 tensor-core path (Hopper+). Auto-fallback to bf16. |
| `maxsim_from_hidden`         | Fused D-side `Linear → Normalize → MaxSim` (forward).  |
| `plaid_approx_score`         | IVF prune step for ColBERTv2.                           |
| `maxsim_residual`            | Fused PLAID / ColBERTv2 decompress + MaxSim (autograd-aware). |
| `maxsim_residual_inference`  | Explicit forward-only variant of `maxsim_residual`.    |
| `maxsim_residual_varlen`     | Varlen PLAID residual rerank.                          |
| `layers.MaxSim`              | Stateless `nn.Module` for `LayerRepository` + `kernelize()`. |

Backward / CSR / unified training kernels, retrieval helpers (`retrieve`,
`MaxSimScorer`), padded reranking (`pack_padded`, `maxsim_padded`),
pair-list scoring (`score_pairs_packed`), experimental variants (`soft`,
`smooth`, `matryoshka`, `xtr`) and PyLate compat stay PyPI-only.

## Repository layout on the source side

```text
late-interaction-kernels/
├── build.toml                               # HF Kernels manifest
├── flake.nix                                # kernel-builder flake input
├── torch-ext/
│   └── late_interaction_kernels/            # HF-facing package (symlinks + shims)
│       ├── __init__.py                      # curated inference surface
│       ├── layers.py                        # stateless MaxSim nn.Module
│       ├── backward/                        # directory symlink → ../../late_interaction_kernels/backward
│       └── *.py                             # file symlinks → ../../late_interaction_kernels/*.py
└── late_interaction_kernels/                # canonical PyPI source (unchanged)
```

Files under `torch-ext/late_interaction_kernels/` that aren't hand-written
(`__init__.py`, `layers.py`) are **symlinks** into the main package — one
source of truth. `kernel-builder` dereferences symlinks when it copies the
tree into `build/torch-universal/late_interaction_kernels/`, so the Hub
repo ends up with real files.

The `backward/` subpackage is a directory symlink — `kernel-builder` copies
all four files (`__init__.py`, `atomic.py`, `csr.py`, `unified.py`) in one
shot.

## Maintainer: build + publish

You need [Nix](https://nixos.org) and the HuggingFace kernel-builder. Once:

```bash
curl -fsSL https://raw.githubusercontent.com/huggingface/kernels/main/install.sh | bash
```

Then from the repo root:

```bash
kernel-builder build-and-copy -L   # → build/torch-universal/late_interaction_kernels/
kernel-builder testshell           # nix shell with torch + triton + test deps
pytest tests/test_hf_kernels_layer.py -v
```

To publish to `Hcompany/late-interaction-kernels` (version bumps the
branch to `v<version>` per `build.toml`):

```bash
export HF_TOKEN=hf_...              # write access to the Hcompany org
kernel-builder build-and-upload
```

Or tag `kernels-vN` to let
`.github/workflows/kernels-publish.yml` do the upload.

## Staging first

Before bumping the real `v1` branch, publish to a personal staging repo:

```bash
# temporarily flip build.toml:
#   [general.hub]
#   repo-id = "<your-user>/late-interaction-kernels-staging"

kernel-builder build-and-upload

# in a clean venv with only torch + triton + kernels:
python -c "from kernels import get_kernel; \
  lik = get_kernel('<your-user>/late-interaction-kernels-staging'); \
  print(dir(lik))"
```

Only flip the `repo-id` back to `Hcompany/late-interaction-kernels` once
round-trip + parity tests pass.

## Versioning

- `build.toml:[general].version = N` ⇒ published to branch `vN` on the
  Hub. Keep `v1` stable; bump to `v2` for any incompatible change.
- Adding a new kernel / argument on an existing op is *not* a version
  bump — consumers pinning `version=1` still work.
- Removing or renaming an export, or changing tensor layout / dtype
  expectations, **is** a version bump.

## Known limitations (v1)

- **FP8 MaxSim** requires Hopper+ (`sm_90`). On older GPUs,
  `maxsim_inference_fp8` dequantizes to bf16 and calls the regular
  MaxSim forward — correct, but no FP8 speedup. (This lazy fallback
  imports `from .autograd import maxsim_inference` internally, which is
  why `autograd.py` is bundled in the Hub build even though backward
  isn't advertised.)
- **macOS / Windows** — Triton isn't available, so `get_kernel()` will
  fail at import time on non-Linux. Matches the PyPI package.
- **No backward API** is surfaced through the `MaxSim` layer
  (`has_backward = False`). Users who need training gradients should
  either call `lik.maxsim(...).backward()` on the functional API
  directly, or install the PyPI package for the full training surface.
