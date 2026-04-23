"""late-interaction-kernels: fused Triton kernels for late-interaction scoring.

Most users only need two things::

    from late_interaction_kernels import patch_pylate, MaxSimScorer

    patch_pylate()                         # one-line PyLate speedup
    scorer = MaxSimScorer(normalize=True)  # nn.Module for custom training

Public surface
--------------

High-level
    * ``MaxSimScorer`` / ``retrieve`` — ``nn.Module`` and top-level retrieval
      helper. Batteries included: masks, chunking, normalize, mixed precision.
    * ``patch_pylate()`` / ``unpatch_pylate()`` — one-line PyLate drop-in.

Core MaxSim
    * ``maxsim(Q, D, q_mask=, d_mask=, normalize=, backward="auto")`` —
      autograd-aware.
    * ``maxsim_inference(...)`` — no saved argmax, inference-only.
    * ``maxsim_varlen(Qp, Dp, cu_q, cu_d)`` — packed / ragged inputs,
      autograd-aware. Auto-skips the argmax save when neither input needs
      gradients (``maxsim_varlen_inference`` is a thin alias kept for 0.9.x
      back-compat).
    * ``maxsim_topk(Q, D, k=, chunk=)`` — MaxSim + top-k in one call.

Fused heads (raw-hidden-state training / inference)
    * ``maxsim_from_hidden(Q, H_d, W, b=, normalize=)`` — fused D-side
      projection + L2-normalize + MaxSim, inference. Skips the
      ``[Nd, Ld, d_out]`` scratch.
    * ``maxsim_from_hidden_train(...)`` — same forward, autograd-aware;
      back-props into ``H_d``, ``W``, ``b``, ``Q``.

PLAID / ColBERTv2 rerank
    * ``plaid_approx_score(q_cent, codes, doc_lengths)`` — IVF approximate
      scoring step.
    * ``maxsim_residual(Q, codes, residuals, ...)`` — fused decompress +
      L2-normalize + MaxSim, autograd on ``Q``.
    * ``maxsim_residual_inference(...)`` — forward-only.
    * ``maxsim_residual_varlen(Q, codes_flat, residuals_flat, cu_seqlens_d, ...)``
      — same fused decompress + MaxSim over ragged (``cu_seqlens``-indexed)
      inputs, matching the layout fast-plaid / ColBERTv2 use on disk.
      Forward-only; no ``[Ntop, max_Ld, packed_dim]`` scratch, no attention
      mask.

FP8 (Hopper / Blackwell)
    * ``maxsim_inference_fp8(Q_fp8, D_fp8, scale_Q=, scale_D=)`` — fp8
      tensor-core MaxSim. Auto-falls back to bf16 off Hopper.

Experimental (research-grade, kept out of the top-level surface)
    * ``from late_interaction_kernels.experimental import`` — ``soft_maxsim``,
      ``smooth_maxsim``, ``maxsim_xtr``, ``maxsim_matryoshka``.

Low-level FP8 helpers live at ``late_interaction_kernels.fp8``
(``quantize_fp8_per_tensor / _per_token`` and their dequant counterparts).

Tuning knobs
    * ``set_backward_method(method)`` / ``get_backward_method()`` — global
      default for ``grad_D``. Valid values: ``"auto" | "unified" | "csr" |
      "atomic"``. Prefer the per-call ``backward=`` kwarg on ``maxsim`` /
      ``MaxSimScorer``.
"""

__version__ = "0.9.0.dev0"

# The Triton kernels aren't importable on platforms without Triton (macOS,
# Windows without a CUDA build). We still want ``import late_interaction_kernels``
# and the reference implementation to work there — the library doubles as a
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
    from .fp8 import maxsim_inference_fp8
    from .fused_head import maxsim_from_hidden, maxsim_from_hidden_train
    from .plaid import (
        maxsim_residual,
        maxsim_residual_inference,
        maxsim_residual_varlen,
        plaid_approx_score,
    )
    from .pylate_compat import patch_pylate, unpatch_pylate
    from .topk import maxsim_topk
    from .varlen import maxsim_varlen, maxsim_varlen_inference
else:  # pragma: no cover

    def _needs_triton(*_args, **_kwargs):  # type: ignore[no-redef]
        raise RuntimeError(
            "late-interaction-kernels's GPU kernels require Triton, which isn't "
            "installed on this platform. Install a CUDA-enabled Triton (Linux only) "
            "or use the reference implementations in `late_interaction_kernels.reference`."
        )

    maxsim = maxsim_inference = _needs_triton
    maxsim_from_hidden = maxsim_from_hidden_train = _needs_triton
    maxsim_inference_fp8 = _needs_triton
    maxsim_varlen = maxsim_varlen_inference = _needs_triton
    maxsim_topk = _needs_triton
    plaid_approx_score = _needs_triton
    maxsim_residual = maxsim_residual_inference = maxsim_residual_varlen = _needs_triton
    set_backward_method = get_backward_method = _needs_triton
    patch_pylate = unpatch_pylate = _needs_triton

# `MaxSimScorer` and `retrieve` transparently fall back to the pure-PyTorch
# reference on non-Triton platforms, so they're always importable — a UX win
# for macOS / CI users developing training code locally.
from . import reference  # noqa: E402,F401  — always importable (pure PyTorch)
from .retrieve import MaxSimScorer, retrieve  # noqa: E402

# --- Deprecated re-exports ----------------------------------------------------
#
# Four research kernels moved to `late_interaction_kernels.experimental` and
# four FP8 quantization helpers moved to `late_interaction_kernels.fp8`. We
# keep them importable from the top level with a DeprecationWarning so 0.9.x
# code keeps working. Removal is scheduled for the first post-0.9 release.

_DEPRECATED_EXPERIMENTAL = {
    "maxsim_matryoshka": "late_interaction_kernels.experimental",
    "maxsim_xtr": "late_interaction_kernels.experimental",
    "soft_maxsim": "late_interaction_kernels.experimental",
    "smooth_maxsim": "late_interaction_kernels.experimental",
}

_DEPRECATED_FP8_HELPERS = {
    "quantize_fp8_per_tensor": "late_interaction_kernels.fp8",
    "quantize_fp8_per_token": "late_interaction_kernels.fp8",
    "dequantize_fp8_per_tensor": "late_interaction_kernels.fp8",
    "dequantize_fp8_per_token": "late_interaction_kernels.fp8",
}


def __getattr__(name: str):
    """Module-level ``__getattr__`` for deprecated / moved symbols (PEP 562)."""
    if name == "maxsim_forward":
        # Demoted to private in 0.9.0. The forward primitive is available as
        # `from late_interaction_kernels.forward import maxsim_forward`.
        import warnings

        warnings.warn(
            "`late_interaction_kernels.maxsim_forward` is deprecated since 0.9.0 and "
            "will be removed in a future release. It's a forward-only primitive with "
            "no autograd — use `maxsim_inference(Q, D, ...)` for reranking or "
            "`maxsim(Q, D, ...)` for gradients. If you really want the low-level "
            "primitive, import it from the private module: "
            "`from late_interaction_kernels.forward import maxsim_forward`.",
            DeprecationWarning,
            stacklevel=2,
        )
        if _HAS_TRITON:
            from .forward import maxsim_forward as _mf

            return _mf
        return _needs_triton

    if name in _DEPRECATED_EXPERIMENTAL:
        import warnings

        new_home = _DEPRECATED_EXPERIMENTAL[name]
        warnings.warn(
            f"`late_interaction_kernels.{name}` moved to `{new_home}` and will be "
            f"removed from the top-level import in a future release. "
            f"Use `from {new_home} import {name}`.",
            DeprecationWarning,
            stacklevel=2,
        )
        if _HAS_TRITON:
            from .experimental import (
                maxsim_matryoshka as _mm,
            )
            from .experimental import (
                maxsim_xtr as _mx,
            )
            from .experimental import (
                smooth_maxsim as _sm,
            )
            from .experimental import (
                soft_maxsim as _so,
            )

            return {
                "maxsim_matryoshka": _mm,
                "maxsim_xtr": _mx,
                "smooth_maxsim": _sm,
                "soft_maxsim": _so,
            }[name]
        return _needs_triton

    if name in _DEPRECATED_FP8_HELPERS:
        import warnings

        new_home = _DEPRECATED_FP8_HELPERS[name]
        warnings.warn(
            f"`late_interaction_kernels.{name}` moved to `{new_home}` and will be "
            f"removed from the top-level import in a future release. "
            f"Use `from {new_home} import {name}`.",
            DeprecationWarning,
            stacklevel=2,
        )
        if _HAS_TRITON:
            from .fp8 import (
                dequantize_fp8_per_tensor as _dqt,
            )
            from .fp8 import (
                dequantize_fp8_per_token as _dqk,
            )
            from .fp8 import (
                quantize_fp8_per_tensor as _qt,
            )
            from .fp8 import (
                quantize_fp8_per_token as _qk,
            )

            return {
                "quantize_fp8_per_tensor": _qt,
                "quantize_fp8_per_token": _qk,
                "dequantize_fp8_per_tensor": _dqt,
                "dequantize_fp8_per_token": _dqk,
            }[name]
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
    "maxsim_varlen",
    "maxsim_varlen_inference",
    "maxsim_topk",
    # fused heads
    "maxsim_from_hidden",
    "maxsim_from_hidden_train",
    # PLAID / ColBERTv2
    "plaid_approx_score",
    "maxsim_residual",
    "maxsim_residual_inference",
    "maxsim_residual_varlen",
    # FP8 inference
    "maxsim_inference_fp8",
    # configuration
    "set_backward_method",
    "get_backward_method",
    "reference",
]
