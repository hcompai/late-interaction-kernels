"""Apple Silicon (MPS) MaxSim backends.

- :mod:`.compile_dispatch` — ``torch.compile``-fused path (autograd-aware,
  all shapes), and the dispatcher that routes between this and the Metal
  kernel.
- :mod:`.metal`            — fused ``simdgroup_matrix`` Metal kernel
  (forward-only, big-batch inference).

:func:`maxsim_mps` and :func:`maxsim_inference_mps` dispatch between the two
based on shape and the ``LIK_FORCE_MPS_BACKEND`` env variable.
"""

from late_interaction_kernels.mps.compile_dispatch import (
    maxsim_inference_mps,
    maxsim_mps,
)

__all__ = ["maxsim_mps", "maxsim_inference_mps"]
