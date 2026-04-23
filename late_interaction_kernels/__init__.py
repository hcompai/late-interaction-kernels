"""late-interaction-kernels: fused Triton kernels for late-interaction scoring.

Most users only need two things::

    from late_interaction_kernels import patch_pylate, MaxSimScorer

    patch_pylate()                       # one-line PyLate speedup
    scorer = MaxSimScorer(normalize=True) # nn.Module for custom training

Public surface
--------------

High-level (recommended)
    * ``MaxSimScorer`` / ``retrieve`` — nn.Module and top-level retrieval
      helper. Batteries included: mask handling, chunking, normalize, mixed
      precision.
    * ``patch_pylate()`` / ``unpatch_pylate()`` — one-line PyLate drop-in.

Core MaxSim
    * ``maxsim(Q, D, q_mask=, d_mask=, normalize=, backward="auto")`` — autograd-aware.
    * ``maxsim_inference(...)`` — no saved argmax, inference-only.
    * ``soft_maxsim(...)`` — log-sum-exp relaxation (dense gradient).
    * ``smooth_maxsim(..., top_k=)`` — top-K argmax save with sparse
      smoother gradient (middle ground between hard max and LSE).
    * ``maxsim_varlen(...)`` — packed / ragged inputs, autograd-aware.
      Forward-only when neither input has ``requires_grad=True``; no
      separate ``_inference`` alias is needed anymore.

FP8 (Hopper / Blackwell)
    * ``maxsim_inference_fp8(Q_fp8, D_fp8, scale_Q=, scale_D=)`` — fp8
      tensor-core MaxSim with per-tensor or per-token scales. Auto-falls
      back to bf16 on non-Hopper GPUs.
    * ``quantize_fp8_per_tensor / _per_token(X)`` — helpers.

Retrieval / fused heads
    * ``maxsim_from_hidden(Q, H_d, W, b=, normalize=)`` — fused D-side
      projection + L2-normalize + MaxSim, inference-only. Saves the
      ``[Nd, Ld, d_out]`` scratch from materializing ``D_proj``.
    * ``maxsim_from_hidden_train(Q, H_d, W, b=, normalize=)`` — same
      forward, fully autograd-aware; backs-props into ``H_d``, ``W``,
      ``b``, ``Q`` without re-materializing ``D_proj``.
    * ``maxsim_topk(Q, D, k, ...)`` — top-k docs + indices in one call.
    * ``plaid_approx_score(qcs, codes, doc_lengths)`` — ColBERTv2 IVF
      approximate scoring step, fused.
    * ``maxsim_residual(Q, codes, residuals, ...)`` — fused PLAID /
      ColBERTv2 2/4/8-bit decompression + L2-normalize + MaxSim (exact
      rerank step). Autograd-aware on Q; lets you train directly on
      compressed doc embeddings.
    * ``maxsim_residual_inference(...)`` — no saved argmax, inference-only.
    * ``maxsim_residual_varlen(Q, codes_flat, residuals_flat, cu_seqlens_d, ...)``
      — same fused decompress + MaxSim but over ragged (``cu_seqlens``-indexed)
      codes / residuals, matching the storage format fast-plaid / ColBERTv2
      use internally. Skips the ``[Ntop, max_Ld, packed_dim]`` padded scratch
      and attention mask. Inference-only.

Variants
    * ``maxsim_matryoshka(Q, D, dims=[...])`` — multi-dim scoring in one pass.
    * ``maxsim_xtr(Q, D, top_k=5)`` — XTR top-k aggregated MaxSim.

Advanced
    * ``set_backward_method(method)`` / ``get_backward_method()`` select
      the process-wide default ``grad_D`` path. Valid values:
      ``"auto" | "unified" | "csr" | "atomic"``. Since 0.6.0 ``"auto"``
      picks between ``"unified"`` and ``"csr"`` — ``"atomic"`` is a
      legacy two-pass fallback. Prefer the per-call ``backward=`` kwarg
      on ``maxsim`` / ``MaxSimScorer`` over the global.
"""

__version__ = "0.9.0.dev0"

# The Triton kernels are not importable on platforms without Triton (macOS,
# Windows without a CUDA build). We still want ``import late_interaction_kernels`` and
# the reference implementation to work there — the library doubles as a
# correctness reference in those environments. Only ``patch_pylate`` and the
# kernel-backed APIs are gated on Triton being available.
try:
    import triton  # noqa: F401

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

if _HAS_TRITON:
    from .autograd import (
        get_backward_method,
        maxsim,
        maxsim_inference,
        set_backward_method,
    )
    from .fp8 import (
        dequantize_fp8_per_tensor,
        dequantize_fp8_per_token,
        maxsim_inference_fp8,
        quantize_fp8_per_tensor,
        quantize_fp8_per_token,
    )
    from .fused_head import maxsim_from_hidden, maxsim_from_hidden_train
    from .matryoshka import maxsim_matryoshka
    from .plaid import (
        maxsim_residual,
        maxsim_residual_inference,
        maxsim_residual_varlen,
        plaid_approx_score,
    )
    from .pylate_compat import patch_pylate, unpatch_pylate
    from .smooth import smooth_maxsim
    from .soft import soft_maxsim
    from .topk import maxsim_topk
    from .varlen import maxsim_varlen, maxsim_varlen_inference
    from .xtr import maxsim_xtr
else:  # pragma: no cover

    def _needs_triton(*_args, **_kwargs):  # type: ignore[no-redef]
        raise RuntimeError(
            "late-interaction-kernels's GPU kernels require Triton, which isn't installed on "
            "this platform. Install a CUDA-enabled Triton (Linux only) or use the "
            "reference implementations in `late_interaction_kernels.reference`."
        )

    maxsim = maxsim_inference = _needs_triton
    maxsim_from_hidden = maxsim_from_hidden_train = _needs_triton
    maxsim_inference_fp8 = _needs_triton
    quantize_fp8_per_tensor = quantize_fp8_per_token = _needs_triton
    dequantize_fp8_per_tensor = dequantize_fp8_per_token = _needs_triton
    soft_maxsim = smooth_maxsim = maxsim_varlen = maxsim_varlen_inference = _needs_triton
    maxsim_topk = maxsim_matryoshka = maxsim_xtr = _needs_triton
    plaid_approx_score = maxsim_residual = maxsim_residual_inference = maxsim_residual_varlen = _needs_triton
    set_backward_method = get_backward_method = _needs_triton
    patch_pylate = unpatch_pylate = _needs_triton

# `MaxSimScorer` and `retrieve` transparently fall back to the pure-PyTorch
# reference on non-Triton platforms, so they're always importable — a big
# UX win for macOS / CI users developing training code locally.
from . import reference  # noqa: E402,F401  — always importable (pure PyTorch)
from .retrieve import MaxSimScorer, retrieve  # noqa: E402


def __getattr__(name: str):
    """Module-level ``__getattr__`` for deprecated / moved symbols (PEP 562)."""
    if name == "maxsim_forward":
        # Demoted to private in 0.9.0. The forward primitive is available as
        # `from late_interaction_kernels.forward import maxsim_forward` for
        # advanced users; the module-level re-export is scheduled for removal.
        import warnings

        warnings.warn(
            "`late_interaction_kernels.maxsim_forward` is deprecated since 0.9.0 and "
            "will be removed in a future release. It is a forward-only primitive "
            "with no autograd — use `maxsim_inference(Q, D, ...)` for reranking "
            "(skips the argmax save) or `maxsim(Q, D, ...)` if you need gradients. "
            "If you really want the low-level primitive, import it from the "
            "private module: `from late_interaction_kernels.forward import maxsim_forward`.",
            DeprecationWarning,
            stacklevel=2,
        )
        if _HAS_TRITON:
            from .forward import maxsim_forward as _mf

            return _mf
        return _needs_triton
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    # high-level
    "MaxSimScorer",
    "retrieve",
    "patch_pylate",
    "unpatch_pylate",
    # core MaxSim
    "maxsim",
    "maxsim_inference",
    "maxsim_from_hidden",
    "maxsim_from_hidden_train",
    "maxsim_inference_fp8",
    "quantize_fp8_per_tensor",
    "quantize_fp8_per_token",
    "dequantize_fp8_per_tensor",
    "dequantize_fp8_per_token",
    "smooth_maxsim",
    "soft_maxsim",
    "maxsim_varlen",
    "maxsim_varlen_inference",
    # retrieval
    "maxsim_topk",
    "plaid_approx_score",
    "maxsim_residual",
    "maxsim_residual_inference",
    "maxsim_residual_varlen",
    # variants
    "maxsim_matryoshka",
    "maxsim_xtr",
    # configuration
    "set_backward_method",
    "get_backward_method",
    "reference",
]
