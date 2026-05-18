"""Backward (grad_D) strategies for MaxSim.

Three implementations; :mod:`late_interaction_kernels.autograd` picks
among them via the ``backward=`` kwarg.

- :mod:`.atomic`  — atomic-add scatter; fast for large ``Nd``.
- :mod:`.unified` — warp-shuffle reduce; fast for small/mid ``Nd``.
- :mod:`.csr`     — CSR-sorted argmax; fast for sparse argmax patterns.
"""

try:
    import triton  # noqa: F401

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

# unified.py guards Triton internally; both functions are always importable.
from late_interaction_kernels.backward.unified import (
    maxsim_backward_unified,
    maxsim_backward_unified_reference,
)

if _HAS_TRITON:
    # atomic.py and csr.py use @triton.jit at module level — only import on CUDA.
    from late_interaction_kernels.backward.atomic import maxsim_backward
    from late_interaction_kernels.backward.csr import maxsim_backward_csr_dD
else:  # pragma: no cover — Triton missing; bind stubs so `from .backward import X` works.

    def _needs_triton(*_args, **_kwargs):
        raise RuntimeError(
            "late-interaction-kernels backward kernels require Triton, which isn't "
            "installed on this platform (Linux + CUDA only)."
        )

    maxsim_backward = _needs_triton
    maxsim_backward_csr_dD = _needs_triton

__all__ = [
    "maxsim_backward",
    "maxsim_backward_unified",
    "maxsim_backward_unified_reference",
    "maxsim_backward_csr_dD",
]
