"""Backward (grad_D) strategies for MaxSim.

:mod:`late_interaction_kernels.autograd` picks among them via ``backward=``.

- :mod:`.unified` — fp32 atomic scatter; fastest single-pass path.
- :mod:`.lowmem`  — destination-owned, bf16 grads (no fp32 buffer, no atomics);
  lower peak memory and deterministic. Default where grad buffers dominate.
- :mod:`.atomic`  — two-pass atomic scatter.
- :mod:`.csr`     — CSR-sorted argmax reduction.
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
    # These use @triton.jit at module level — only import on CUDA.
    from late_interaction_kernels.backward.atomic import maxsim_backward
    from late_interaction_kernels.backward.csr import maxsim_backward_csr_dD
    from late_interaction_kernels.backward.lowmem import maxsim_backward_lowmem

__all__ = [
    "maxsim_backward",
    "maxsim_backward_unified",
    "maxsim_backward_unified_reference",
    "maxsim_backward_csr_dD",
    "maxsim_backward_lowmem",
]
