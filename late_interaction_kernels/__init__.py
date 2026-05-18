"""Fused Triton kernels for late-interaction (MaxSim) scoring.

Common entry points::

    from late_interaction_kernels import patch_pylate, MaxSimScorer, retrieve

    patch_pylate()                         # PyLate drop-in
    scorer = MaxSimScorer(normalize=True)  # nn.Module, autograd-aware
    scores, idx = retrieve(Q, D, top_k=100)

The top-level surface is intentionally small. Niche / lower-level kernels live
in submodules and must be imported explicitly:

- pair-list scoring → ``late_interaction_kernels.score_pairs``
- padded → packed building blocks → ``late_interaction_kernels.padded``
- fused D-side head → ``late_interaction_kernels.fused_head``
- PLAID / ColBERTv2 → ``late_interaction_kernels.plaid``
- FP8 inference → ``late_interaction_kernels.fp8``
- backward-method config → ``late_interaction_kernels.autograd``
- research variants → ``late_interaction_kernels.experimental``

See the README for the full API and benchmarks.
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
    from late_interaction_kernels.autograd import maxsim, maxsim_inference
    from late_interaction_kernels.varlen import maxsim_varlen
else:  # pragma: no cover

    def _needs_triton(*_args, **_kwargs):  # type: ignore[no-redef]
        raise RuntimeError(
            "late-interaction-kernels GPU kernels require Triton, which isn't "
            "installed on this platform. Install a CUDA-enabled Triton (Linux only) "
            "or use `late_interaction_kernels.reference` for the pure-PyTorch path."
        )

    maxsim = maxsim_inference = _needs_triton
    maxsim_varlen = _needs_triton

# Cross-platform high-level entry points:
# * `MaxSimScorer` / `retrieve` / `maxsim_padded` fall back to the pure-PyTorch
#   reference on machines without Triton, so training and retrieval code is
#   unit-testable on a laptop;
# * `patch_pylate` dispatches per-call: CUDA → Triton kernel, MPS →
#   `torch.compile`-fused path, anything else → PyLate's own implementation.
from late_interaction_kernels import reference  # noqa: E402,F401
from late_interaction_kernels.padded import maxsim_padded  # noqa: E402
from late_interaction_kernels.pylate_compat import patch_pylate, unpatch_pylate  # noqa: E402
from late_interaction_kernels.retrieve import MaxSimScorer, retrieve  # noqa: E402

__all__ = [
    "__version__",
    # high-level
    "MaxSimScorer",
    "retrieve",
    "maxsim_padded",
    "patch_pylate",
    "unpatch_pylate",
    # core MaxSim
    "maxsim",
    "maxsim_inference",
    "maxsim_varlen",
    # ground-truth reference module
    "reference",
]
