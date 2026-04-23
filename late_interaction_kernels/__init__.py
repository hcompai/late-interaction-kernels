"""Fused Triton kernels for late-interaction (MaxSim) scoring.

    from late_interaction_kernels import patch_pylate, MaxSimScorer, retrieve

    patch_pylate()                         # speed up PyLate in one line
    scorer = MaxSimScorer(normalize=True)  # nn.Module for custom training
    scores, idx = retrieve(Q, D, top_k=100)

See README.md for the full API, benchmarks and supported models.
FP8 helpers live in ``late_interaction_kernels.fp8``.
Research kernels (soft / smooth / Matryoshka / XTR MaxSim) live in
``late_interaction_kernels.experimental``.
"""

__version__ = "0.9.0.dev0"

# Triton isn't available everywhere (macOS, Windows without CUDA). On those
# platforms we still import the package and expose the pure-PyTorch reference;
# only the fused kernel entry points are gated on Triton being present.
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

# `MaxSimScorer` and `retrieve` are always importable: they fall back to the
# pure-PyTorch reference when Triton isn't available, so training / retrieval
# code can still be developed and unit-tested on macOS or CI.
from . import reference  # noqa: E402,F401  — always importable (pure PyTorch)
from .retrieve import MaxSimScorer, retrieve  # noqa: E402

# Symbols that moved out of the top level in 0.9.0. Re-exported here with a
# DeprecationWarning so 0.9.x code keeps working; removed in a future release.

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
            "`late_interaction_kernels.maxsim_forward` is deprecated since 0.9.0 "
            "and will be removed in a future release. Use `maxsim_inference(Q, D, ...)` "
            "for reranking or `maxsim(Q, D, ...)` for gradients. The low-level "
            "primitive is still available at "
            "`late_interaction_kernels.forward.maxsim_forward`.",
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
