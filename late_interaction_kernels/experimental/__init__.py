"""Research MaxSim variants. Import explicitly, not at the top level.

    from late_interaction_kernels.experimental import (
        maxsim_matryoshka,   # Matryoshka: score K truncated dims at once
        maxsim_xtr,          # XTR top-K aggregation
        soft_maxsim,         # log-sum-exp relaxation
        smooth_maxsim,       # top-K smoother
    )

Requires CUDA + Triton.
"""

try:
    import triton  # noqa: F401

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False


if _HAS_TRITON:
    from ..matryoshka import maxsim_matryoshka
    from ..smooth import smooth_maxsim
    from ..soft import soft_maxsim
    from ..xtr import maxsim_xtr
else:  # pragma: no cover

    def _needs_triton(*_args, **_kwargs):
        raise RuntimeError(
            "late-interaction-kernels experimental kernels require Triton, which "
            "isn't installed on this platform. Install a CUDA-enabled Triton "
            "(Linux only) to use them."
        )

    maxsim_matryoshka = maxsim_xtr = soft_maxsim = smooth_maxsim = _needs_triton

__all__ = [
    "maxsim_matryoshka",
    "maxsim_xtr",
    "soft_maxsim",
    "smooth_maxsim",
]
