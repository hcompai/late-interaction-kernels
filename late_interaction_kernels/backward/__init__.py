"""Backward (grad_D) strategies for MaxSim.

:mod:`late_interaction_kernels.autograd` picks among them via ``backward=``.

- :mod:`.unified` — fp32 atomic scatter; fastest single-pass path.
- :mod:`.lowmem`  — destination-owned, grads written in the input dtype (no
  fp32 buffer, no atomics); lower peak memory and deterministic. Default
  where grad buffers dominate.
"""

# Both modules guard Triton internally (@triton.jit kernels live behind a
# _HAS_TRITON check); their launcher functions import fine on CPU and raise
# only if actually called without a CUDA Triton.
from late_interaction_kernels.backward.lowmem import maxsim_backward_lowmem
from late_interaction_kernels.backward.unified import (
    maxsim_backward_unified,
    maxsim_backward_unified_reference,
)

__all__ = [
    "maxsim_backward_unified",
    "maxsim_backward_unified_reference",
    "maxsim_backward_lowmem",
]
