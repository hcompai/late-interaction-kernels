# flash-colbert

**Fused Triton kernels for ColBERT / ColPali late-interaction MaxSim — with
masks, packed varlen input, and a training-grade backward.**

Drop it into [PyLate](https://github.com/lightonai/pylate) with one line, get
**~3× faster training** and **~12× faster reranking** on H100.

<p align="center">
  <a href="#benchmarks">Benchmarks</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#pylate-drop-in">PyLate drop-in</a> ·
  <a href="docs/design.md">Design notes</a> ·
  <a href="docs/liger.md">Should this be a Liger kernel?</a>
</p>

---

## Why

ColBERT-style **late interaction** scoring is

$$ \operatorname{MaxSim}(Q, D) = \sum_{s} \max_{t} Q_s \cdot D_t $$

PyLate computes this in one line of `torch.einsum`:

```python
S = torch.einsum("ild,jtd->ijlt", Q, D)
scores = S.max(-1).values.sum(-1)
```

That einsum materializes a `[Nq, Nd, Lq, Ld]` tensor in HBM. For a modest
`Nq=Nd=128, Lq=32, Ld=300, d=128` training batch that's already
**600 MB** of throw-away memory per score tensor; ColPali (`Lq=Ld=1024`)
blows up to **4.5 GB** per batch. The einsum itself is memory-bound: on H100
it ranges from 0.7 ms (text reranking) to **12 ms (visual)** per call — and
it's typically called three times per training step (anchor, positive, N
negatives).

`flash-colbert` fuses the whole operation into a single IO-aware Triton kernel
that **never materializes the similarity matrix in HBM**. The max reduction
happens in registers while each `[BLOCK_Q, BLOCK_D]` tile is still in SRAM,
exactly like the inner loop of FlashAttention — just without the softmax and
without a V projection.

## Features

| | |
| :-- | :-- |
| Fused forward kernel (Triton) | ✓ |
| Mask-fused forward (no `-inf` post-processing) | ✓ |
| Query-side skiplist / pad-mask | ✓ |
| Document-side attention mask | ✓ |
| Packed / varlen inputs (no padding) | ✓ |
| Differentiable (`torch.autograd.Function`) | ✓ |
| `grad_Q` scatter-free | ✓ |
| `grad_D` via fp32 atomics (deterministic) | ✓ |
| Log-sum-exp (soft) relaxation | ✓ |
| fp16 / bf16 / fp32 inputs with fp32 accumulator | ✓ |
| Autotuned per GPU family (Hopper / Ampere) | ✓ |
| PyLate `colbert_scores` drop-in replacement | ✓ |
| Multi-GPU DDP compatible | ✓ (it's just a kernel) |

## Benchmarks

Measured on **H100 80 GB SXM** inside a SkyPilot-launched Kubernetes pod,
bf16 compute, fp32 accumulator, 50 iterations after 5 warmup.

### Reranking / inference

```
text-short    Nq=1  Nd=1000  Lq=32  Ld=300  d=128
  flash-colbert   0.031 ms     0.00 MB
  naive einsum    0.705 ms   183.12 MB   (22.7× slower, 5800× more memory)

visual        Nq=1  Nd=1000  Lq=1024 Ld=1024 d=128     (ColPali scale)
  flash-colbert   1.518 ms     0.00 MB
  naive einsum   11.967 ms  4500.50 MB   (7.9× slower, 45 000× more memory)

corpus-10k    Nq=1  Nd=10000 Lq=32  Ld=300  d=128
  flash-colbert   0.557 ms
  naive einsum    7.112 ms                (12.8× slower)
```

### End-to-end PyLate `Contrastive` training step

```
batch=64  neg=1  Lq=32 Ld=200 d=128
  vanilla PyLate:   2.70 ms / step
  flash-colbert:    0.92 ms / step       (2.93× faster)

batch=128 neg=2
  vanilla PyLate:  14.95 ms / step
  flash-colbert:    4.86 ms / step       (3.08× faster)

batch=256 neg=3
  vanilla PyLate:  75.04 ms / step
  flash-colbert:   26.28 ms / step       (2.86× faster)
```

Full tables: [`benchmarks/results/forward_H100_bf16.md`](benchmarks/results/forward_H100_bf16.md).

## Quickstart

```bash
pip install flash-colbert
```

```python
import torch
from flash_colbert import maxsim_inference

Q = torch.randn(32, 128, device="cuda", dtype=torch.float16)           # [Lq, d]
D = torch.randn(1000, 300, 128, device="cuda", dtype=torch.float16)    # [Nd, Ld, d]

scores = maxsim_inference(Q, D)           # shape [1000] — fp32
top = scores.topk(10)
```

For training, use the autograd-aware `maxsim`:

```python
from flash_colbert import maxsim

Q = torch.nn.Parameter(torch.randn(32, 32, 128, device="cuda"))        # [B, Lq, d]
D = torch.nn.Parameter(torch.randn(32, 128, 128, device="cuda"))
scores = maxsim(Q, D)                     # [B, B]
scores.sum().backward()                   # gradients flow into Q and D
```

## PyLate drop-in

One line gives every PyLate call site (`colbert_scores`,
`colbert_kd_scores`, `Contrastive`, `CachedContrastive`, `rerank`) the fast
path with no code changes:

```python
from flash_colbert.pylate_compat import patch_pylate
patch_pylate()

# ...the rest of your PyLate pipeline, unchanged...
from pylate import models, losses
model = models.ColBERT(model_name_or_path="lightonai/GTE-ModernColBERT-v1")
loss  = losses.Contrastive(model=model)
```

Disable at runtime: `FLASH_COLBERT_DISABLE=1` in the environment falls back
to PyTorch. `unpatch_pylate()` restores the original functions.

## Variants

### Log-sum-exp soft MaxSim

For training stability (denser gradient: all doc tokens contribute, not just
the argmax):

```python
from flash_colbert import soft_maxsim
scores = soft_maxsim(Q, D, beta=10.0)     # beta → ∞ recovers hard max
```

### Varlen / packed inputs (no padding)

Avoid `pad_sequence` and its memory waste:

```python
from flash_colbert import maxsim_varlen

# Q_packed: [sum(Lq_i), d]   D_packed: [sum(Ld_j), d]
# cu_seqlens_{q,d}: [N+1] int32 cumulative offsets (FlashAttention convention)
scores = maxsim_varlen(Q_packed, D_packed, cu_seqlens_q, cu_seqlens_d)
```

## Design in 60 seconds

```
   Q_block             Q_block              Q_block
   ┌────┐              ┌────┐               ┌────┐
   │ Lq │              │ Lq │               │ Lq │
   │ ·  │              │ ·  │               │ ·  │
   │ d  │              │ d  │               │ d  │
   └─┬──┘              └─┬──┘               └─┬──┘
     │ loaded to SRAM    │                    │
     ▼                   ▼                    ▼
   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
   │  S = Q·Dᵀ    │     │  S = Q·Dᵀ    │     │  S = Q·Dᵀ    │
   │   tile 1     │     │   tile 2     │     │   tile 3     │
   │  mask→−inf   │  →  │  mask→−inf   │  →  │  mask→−inf   │  → sum m
   │  max(m, S)   │     │  max(m, S)   │     │  max(m, S)   │
   └──────────────┘     └──────────────┘     └──────────────┘
                        registers only — no HBM roundtrip
```

One program per (q_batch, d_batch). FP32 accumulator, bf16/fp16 tensor-core
GEMM inside `tl.dot`. Mask-fused: `-inf` applied _inside_ the `tl.where` so
masked tokens can never win the max even if they happened to have a large
positive dot product on a non-masked run.

See [`docs/design.md`](docs/design.md) for the full write-up.

## What about Liger?

Short answer: **yes it could live there, but it stands better on its own.**
See [`docs/liger.md`](docs/liger.md) for the detailed argument.

## Testing

```bash
pytest tests/ -v
```

60 tests, including numerical parity across 9 shapes × 2 dtypes, gradient
parity vs PyTorch autograd, varlen parity, non-power-of-two embedding dims,
non-contiguous inputs, empty-mask rows, and PyLate compatibility.

CI launches on an H100 via `sky jobs launch scripts/sky_test.yaml`.

## Roadmap

- [ ] Recompute-style backward (trade the argmax buffer for one extra GEMM;
      useful at very large `Lq`).
- [ ] `maxsim_multihead` for the `colbert_kd_scores` case (per-query doc sets)
      as a single batched kernel, removing the Python for-loop.
- [ ] `maxsim_topk` returning the top-k doc tokens per query token, for
      interpretability / late-interaction routing.
- [ ] INT8 quantized forward (follow-on to `flash-maxsim_int8`).
- [ ] FP8 (Hopper E4M3) with per-tile rescaling.

## Acknowledgements

- [**FlashAttention**](https://github.com/Dao-AILab/flash-attention) —
  the IO-aware tiled attention pattern this kernel is a strict subset of.
- [**flash-maxsim**](https://github.com/xhluca/flash-maxsim) — first
  public Triton kernel for ColBERT MaxSim. `flash-colbert` extends it with
  fused masking, varlen inputs, a training-grade backward, and the PyLate
  drop-in.
- [**Liger-Kernel**](https://github.com/linkedin/Liger-Kernel) — the
  autotune + `torch.autograd.Function` idioms are lifted straight from the
  Liger softmax kernel.
- [**PyLate**](https://github.com/lightonai/pylate) — late-interaction
  training framework we optimize for.

## License

Apache 2.0. See [`LICENSE`](LICENSE).
