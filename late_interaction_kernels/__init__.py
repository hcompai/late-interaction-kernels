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

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

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

# Cross-platform high-level entry points:
# * `MaxSimScorer` / `retrieve` fall back to the pure-PyTorch reference on
#   machines without Triton, so training and retrieval code is unit-testable
#   on a laptop;
# * `patch_pylate` dispatches per-call: CUDA → Triton kernel, MPS →
#   `torch.compile`-fused path, anything else → PyLate's own implementation.
from . import reference  # noqa: E402,F401
from .padded import PackedBatch, maxsim_padded, pack_padded  # noqa: E402
from .pylate_compat import patch_pylate, unpatch_pylate  # noqa: E402
from .retrieve import MaxSimScorer, retrieve  # noqa: E402

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
    # padded-input reranking
    "pack_padded",
    "maxsim_padded",
    "PackedBatch",
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
