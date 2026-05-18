# Implementation notes — port from `personal/maxsim`

Running log of decisions, tradeoffs, and changes while porting the two
maxsim ideas worth taking into LIK:

1. an explicit pair-list MaxSim API (already present in LIK as
   `maxsim_inference_scatter` — no porting needed, see below);
2. a zero-sync padded → packed conversion helper (new — `pack_padded`);
3. an ergonomic padded reranking wrapper that uses (1) + (2) (new —
   `maxsim_padded`).

Each section below was written at the time the corresponding change
landed and is not edited retroactively.

---

## Scope

### Already covered in LIK — not re-implemented

* `score_pairs_packed` is **functionally equivalent to LIK's
  `maxsim_inference_scatter`** (`late_interaction_kernels/scatter.py`).
  Same signature shape: packed `Q`, packed `D`, two `cu_seqlens`, two
  pair-index tensors, returns `[num_pairs]` fp32. Same forward-only
  contract. The Triton kernel even uses the same per-pair program-id
  scheduling.

  → Decision: do **not** rename `maxsim_inference_scatter`. It's part of
  the 0.1.0 public API and is documented in the README ("Pair-list
  reranking on packed batches"). A rename would be a breaking change
  for any caller. The maxsim name `score_pairs_packed` is arguably
  clearer, but renaming-with-deprecation is more churn than the win
  justifies.

### Genuinely new

* `pack_padded` — `[B, Lq, D]` + `[B, C, Ld, D]` → packed layout
  (`Q_packed`, `D_packed`, `cu_seqlens_q`, `cu_seqlens_d`,
  `pair_q_idx`, `pair_d_idx`, `max_seqlen_q`) with **no `.item()`
  syncs in the hot path**. The current LIK doesn't have this; users
  who only have padded reranking tensors have to write their own pack
  loop.
* `maxsim_padded` — high-level reranking entry point that takes the
  padded `[B, Lq, D]` / `[B, C, Ld, D]` layout, packs internally, and
  returns `[B, C]` fp32 scores. Dispatches CUDA → Triton kernel,
  everything else → reference.

---

## Decisions

### D1. Naming: `cu_seqlens_*` not `*_offsets`

maxsim uses `query_offsets` / `document_offsets`. LIK consistently uses
`cu_seqlens_q` / `cu_seqlens_d` (FlashAttention convention) across
`varlen.py`, `scatter.py`, `reference.py`, the docs and the
benchmarks. I kept LIK's naming so the new helpers compose
unsurprisingly with the rest of the library.

### D2. Structured return type for `pack_padded`

maxsim returns a 7-tuple. That's hard to read at the call site (which
slot was `pair_q_idx` again?). I used a `@dataclass(frozen=True)`
called `PackedBatch` — fields are positional-unpack-safe via
`__iter__` so existing tuple-style call patterns still work, but
keyword access (`batch.cu_seqlens_q`) is preferred.

This is the only place in `late_interaction_kernels/` that uses a
dataclass; no pydantic anywhere in the library. Going lighter (plain
dataclass) felt consistent.

### D3. Default to **no** device-side length validation

maxsim's `_pack_padded` defaults to `validate=False` because the
checks (`(qlen <= 0).any().item()`) each force a D2H sync — which
defeats the whole point of having a zero-sync pack helper. I match
that default and gate the validating path behind `validate=True`. The
shape checks (which don't touch device data) are unconditional.

The single unavoidable sync is `int(qlen.max().item())` so we can
return `max_seqlen_q` as a Python int — `maxsim_inference_scatter`
needs it to size the kernel. Worth eating *one* sync to skip the four
the kernel would otherwise do.

### D4. Dispatch in `maxsim_padded`

CUDA + Triton: pack + `maxsim_inference_scatter`.
Anywhere else (CPU, MPS, Windows): pack + `maxsim_reference_scatter`.

No fused-Metal path for the scatter kernel exists yet in LIK
(`metal.py` only covers the dense `Q × D` reranker), so the MPS user
falls back to the pure-PyTorch reference. Noted as a follow-up.

### D5. Pair ordering: row-major `(b, c)`

Pair index `b * C + c` ⇒ `pair_q_idx[k] = b`, `pair_d_idx[k] = b * C + c`.
This is the same ordering maxsim uses; it means the `[num_pairs]`
kernel output can be reshaped to `[B, C]` with a plain `.view(B, C)`.

### D6. Pack uses `boolean-mask + gather`, not `index_select`

A boolean mask `[B, Lq_max] < qlen[:, None]` followed by
`flat[mask.reshape(-1)]` is a single fused gather kernel under
`torch.compile`/eager and matches maxsim's approach. Tried writing it
with `pad_sequence`-style scatter loops — slower and not
sync-free. Boolean-mask gather it is.

### D7. References live in `reference.py`, not a new submodule

LIK keeps every reference function in `late_interaction_kernels/reference.py`
(it's already 372 lines, but everything is there). I added
`pack_padded_reference` and `maxsim_padded_reference` to that same
file rather than starting a new module. Easier to grep.

---

## Tradeoffs not pursued

* **Pad-position invariance on the kernel side**: maxsim's `padded`
  Metal kernel reads padded strides directly without packing, which
  means it never even sees padded tokens. LIK's `maxsim_padded`
  packs-then-calls-`scatter`, so the pack step does the masking. Same
  numerics, but on hot reranking workloads the pack is one extra
  kernel launch + one allocation. A future native padded kernel would
  shave that — explicitly out of scope here.
* **`score_pairs_packed` alias**: tempted to add it as a friendly
  alias in `__init__.py` since the maxsim name is clearer. Decided
  against it for now — adds API surface without behaviour. Revisit if
  users actually trip on the `maxsim_inference_scatter` name.
* **Renaming/restructuring tests**: maxsim's `tests/` is flatter and
  uses a `_helpers.py` module for shared fixtures. LIK's is bigger
  (25 modules) and per-kernel. A wholesale reorg would be churn; I
  only added `tests/test_padded.py` for the new code and left the
  rest alone.

---

## Files touched

(filled in as commits land)
