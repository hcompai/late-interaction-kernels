# Should late-interaction-kernels be a Liger-Kernel addition?

Short answer: **it could live in Liger, and a PR would likely be accepted, but the standalone library is a better home today.** Here is the reasoning.

## What Liger optimizes for

Liger-Kernel is focused on [kernels useful inside transformer model
training loops](https://github.com/linkedin/Liger-Kernel):

- Layer-level fused kernels (RMSNorm, SwiGLU, GeGLU, GroupNorm).
- Loss-level fused kernels (fused cross-entropy, JSD, fused linear
  cross-entropy).
- Positional encoding kernels (RoPE).

All of them plug into an LLM `forward()` — typically inside Hugging Face
`transformers` modules — with a `liger_kernel.transformers` patch API
(`apply_liger_kernel_to_llama`, `apply_liger_kernel_to_qwen2`, etc.). They
are **layer replacements**, not algorithm replacements.

MaxSim is structurally different:

- It's not a layer in a transformer forward. It lives *after* the token
  embeddings leave the model, inside the **loss function** of a retrieval
  system.
- It's specific to the **ColBERT / ColPali / late-interaction** family —
  not part of the GPT / Llama / Qwen / Gemma call graph that Liger
  currently patches.
- The shape invariants are different (`[Nq × Lq × d]` vs the familiar
  `[B × T × H]`).

## Where it could slot in

There is precedent for loss-fusion kernels in Liger:

- `liger_kernel/ops/fused_linear_cross_entropy.py`
- `liger_kernel/ops/jsd.py` / `js_div.py`
- `liger_kernel/ops/kl_div.py`

A `liger_kernel/ops/maxsim.py` kernel and a `MaxSimLoss` wrapper in
`liger_kernel/transformers/` would fit that folder convention. The Triton
style (`@triton.autotune` + `torch.autograd.Function`) matches Liger
perfectly — our `autograd.py` is deliberately written in that exact idiom
(compare `late_interaction_kernels/autograd.py` to `liger_kernel/ops/softmax.py`).

## Why we're not doing that today

1. **Scope mismatch**: Liger's patch API targets HuggingFace
   `transformers` model classes. PyLate is a `sentence-transformers`
   derivative — its `ColBERT` class is not a HF model class. Plumbing a
   `apply_liger_kernel_to_pylate` is doable but crosses a project
   boundary.

2. **Varlen + masks are first-class here, not in Liger**: late-interaction-kernels
   carries a packed/varlen kernel and the mask-fused kernels that retrieval
   workloads actually need. Liger's kernels mostly operate on packed
   `[B·T]` tensors without the retrieval-specific mask semantics.

3. **The PyLate drop-in belongs near PyLate, not near Liger**: the
   `patch_pylate()` logic imports `pylate.scores` and `pylate.losses`;
   pinning that inside Liger adds a non-transformers dependency to an
   LLM-training library.

4. **Audience**: Liger users are LLM trainers. late-interaction-kernels users are
   retrieval people. Keeping them in separate installs lets each track its
   own release cadence (e.g., we can ship a fix for a PyLate version bump
   without waiting on a Liger release).

## What I would do

- **Today**: ship `late-interaction-kernels` as its own `pip install late-interaction-kernels`
  package — which is what this repo is.
- **If LinkedIn accepts it**: open a Liger PR that vendors the
  `late_interaction_kernels.forward`, `backward`, `autograd` modules under
  `liger_kernel/ops/maxsim.py` + `liger_kernel/chunked_loss/maxsim_loss.py`
  (Liger already has a `chunked_loss` submodule for memory-efficient
  losses). That delivery form reuses Liger's autotune caches and test
  harness, and extends Liger's reach into the retrieval space. The
  standalone PyLate drop-in would stay here as a thin wrapper around the
  Liger version.
- **Long term**: if Hugging Face `transformers` or
  `sentence-transformers` adopts late-interaction as a first-class model
  class (the current `ColBERT` architecture is very close to one), the
  `MaxSimLoss` would fit naturally into
  `liger_kernel.transformers.maxsim_loss`.

## Summary

|                              | late-interaction-kernels (standalone) | as Liger-Kernel op |
| ---------------------------- | -------------------------- | ------------------ |
| Code style fits              | ✓ (modeled after it)       | ✓                 |
| Maintenance overhead         | 1 repo                     | cross-project     |
| PyLate integration           | first-class                | awkward           |
| Varlen / mask fusion         | first-class                | new territory     |
| Release independence         | ✓                          | blocked on Liger  |
| Visibility to LLM engineers  | −                          | ✓                 |
| Visibility to IR engineers   | ✓                          | −                 |

Best of both worlds is probably to keep late-interaction-kernels as the canonical
place for the kernel and publish a thin Liger adapter once the Liger team
is interested — which is what `docs/liger.md` will become: a "how to wire
this into Liger" guide.
