"""Fused Triton kernels for late-interaction (MaxSim) scoring.

Common entry points::

    from late_interaction_kernels import patch_pylate, MaxSimScorer, retrieve

    patch_pylate()                         # PyLate drop-in
    scorer = MaxSimScorer(normalize=True)  # nn.Module, autograd-aware
    scores, idx = retrieve(Q, D, top_k=100)

See the README for the full API and benchmarks.
FP8 helpers live in ``late_interaction_kernels.fp8``.
Research kernels live in ``late_interaction_kernels.experimental``.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("late-interaction-kernels")
except PackageNotFoundError:  # pragma: no cover — running from a source tree without install
    __version__ = "0.0.0+unknown"

# The kernels need Triton (Linux + CUDA). On macOS / Windows we still want
# `import late_interaction_kernels` to succeed so users can develop against
# the pure-PyTorch reference and `MaxSimScorer` / `retrieve` fallbacks.
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
        maxsim_residual_varlen,
        plaid_approx_score,
    )
    from .pylate_compat import patch_pylate, unpatch_pylate
    from .scatter import maxsim_inference_scatter
    from .varlen import maxsim_varlen
else:  # pragma: no cover

    def _needs_triton(*_args, **_kwargs):  # type: ignore[no-redef]
        raise RuntimeError(
            "late-interaction-kernels GPU kernels require Triton, which isn't "
            "installed on this platform. Install a CUDA-enabled Triton (Linux only) "
            "or use `late_interaction_kernels.reference` for the pure-PyTorch path."
        )

    maxsim = maxsim_inference = _needs_triton
    maxsim_from_hidden = maxsim_from_hidden_train = _needs_triton
    maxsim_inference_fp8 = _needs_triton
    maxsim_varlen = _needs_triton
    plaid_approx_score = _needs_triton
    maxsim_residual = maxsim_residual_varlen = _needs_triton
    maxsim_inference_scatter = _needs_triton
    set_backward_method = get_backward_method = _needs_triton
    patch_pylate = unpatch_pylate = _needs_triton

# `MaxSimScorer` and `retrieve` are always importable: they fall back to the
# pure-PyTorch reference on platforms without Triton, so training and
# retrieval code can be unit-tested locally.
from . import reference  # noqa: E402,F401
from .retrieve import MaxSimScorer, retrieve  # noqa: E402

# Symbols moved out of the top level. Still importable, with a
# `DeprecationWarning`. Scheduled for removal one release after 0.9.x.
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
    """PEP 562 — re-export deprecated / moved symbols with a warning."""
    import warnings

    if name == "maxsim_forward":
        warnings.warn(
            "`late_interaction_kernels.maxsim_forward` is deprecated. Use "
            "`maxsim_inference` for reranking, `maxsim` for gradients, or "
            "import the primitive from `late_interaction_kernels.forward`.",
            DeprecationWarning,
            stacklevel=2,
        )
        if _HAS_TRITON:
            from .forward import maxsim_forward as _mf

            return _mf
        return _needs_triton

    if name == "maxsim_topk":
        warnings.warn(
            "`maxsim_topk` is deprecated; use `retrieve(Q, D, top_k=...)` "
            "(same semantics, transparent CPU fallback). Still importable from "
            "`late_interaction_kernels.topk`.",
            DeprecationWarning,
            stacklevel=2,
        )
        if _HAS_TRITON:
            from .topk import maxsim_topk as _mt

            return _mt
        return _needs_triton

    if name == "maxsim_residual_inference":
        warnings.warn(
            "`maxsim_residual_inference` is deprecated; `maxsim_residual` "
            "auto-skips the argmax save when `Q.requires_grad=False`.",
            DeprecationWarning,
            stacklevel=2,
        )
        if _HAS_TRITON:
            from .plaid import maxsim_residual_inference as _mri

            return _mri
        return _needs_triton

    if name == "maxsim_varlen_inference":
        warnings.warn(
            "`maxsim_varlen_inference` is deprecated; `maxsim_varlen` "
            "auto-skips the argmax save when neither input requires grad.",
            DeprecationWarning,
            stacklevel=2,
        )
        if _HAS_TRITON:
            from .varlen import maxsim_varlen_inference as _mvi

            return _mvi
        return _needs_triton

    if name in _DEPRECATED_EXPERIMENTAL:
        new_home = _DEPRECATED_EXPERIMENTAL[name]
        warnings.warn(
            f"`late_interaction_kernels.{name}` moved to `{new_home}`. Use `from {new_home} import {name}`.",
            DeprecationWarning,
            stacklevel=2,
        )
        if _HAS_TRITON:
            from . import experimental

            return getattr(experimental, name)
        return _needs_triton

    if name in _DEPRECATED_FP8_HELPERS:
        new_home = _DEPRECATED_FP8_HELPERS[name]
        warnings.warn(
            f"`late_interaction_kernels.{name}` moved to `{new_home}`. Use `from {new_home} import {name}`.",
            DeprecationWarning,
            stacklevel=2,
        )
        if _HAS_TRITON:
            from . import fp8

            return getattr(fp8, name)
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
    # reranking on packed batches
    "maxsim_inference_scatter",
    # fused D-side head
    "maxsim_from_hidden",
    "maxsim_from_hidden_train",
    # PLAID / ColBERTv2
    "plaid_approx_score",
    "maxsim_residual",
    "maxsim_residual_varlen",
    # FP8 inference
    "maxsim_inference_fp8",
    # configuration
    "set_backward_method",
    "get_backward_method",
    "reference",
]
