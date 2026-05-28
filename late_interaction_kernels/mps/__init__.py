"""Apple Silicon (MPS) MaxSim backends.

- :mod:`.compile_dispatch` — ``torch.compile``-fused path (autograd-aware,
  any dtype / shape) plus the dispatcher that routes between this and
  the Metal kernel.
- :mod:`.metal`            — fused ``simdgroup_matrix`` Metal kernels
  for both forward (inference) and forward + backward (training) on
  fp16 / bf16 inputs with ``d ≤ 128``.

:func:`maxsim_mps` and :func:`maxsim_inference_mps` dispatch between the two
based on shape, dtype, and the ``LIK_FORCE_MPS_BACKEND`` env variable.
"""

from late_interaction_kernels.mps.compile_dispatch import (
    maxsim_inference_mps,
    maxsim_mps,
)

__all__ = ["maxsim_mps", "maxsim_inference_mps"]
