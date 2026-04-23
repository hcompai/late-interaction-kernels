"""Research-grade kernels that haven't earned a spot in the top-level API.

They work and have tests, but nobody is shipping production code on top of
them today. Kept in the package so paper reproductions and ablations stay
possible; kept out of the top-level import so the main surface is the four
things almost everyone actually uses (``maxsim``, ``maxsim_varlen``, the
``maxsim_residual*`` family, ``maxsim_from_hidden*``).

Promotion criteria: show up here with a real user story and numbers, and
we'll move it up. Removal criteria: a release cycle with no open issues
referencing it.

    from late_interaction_kernels.experimental import (
        maxsim_matryoshka,   # multi-dim scoring in one pass
        maxsim_xtr,          # XTR top-k aggregation
        soft_maxsim,         # log-sum-exp relaxation
        smooth_maxsim,       # top-K smoother
    )

These require CUDA + Triton, same as the rest of the kernel-backed APIs.
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
