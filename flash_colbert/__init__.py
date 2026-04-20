"""flash-colbert: fused Triton kernels for ColBERT / ColPali late-interaction.

High-level API
--------------
    from flash_colbert import maxsim, maxsim_inference, soft_maxsim, maxsim_varlen

    scores = maxsim(Q, D, q_mask, d_mask)   # differentiable
    scores = maxsim_inference(Q, D, ...)    # no autograd, no saved argmax
    scores = soft_maxsim(Q, D, ..., beta=10.0)
    scores = maxsim_varlen(Q_packed, D_packed, cu_seqlens_q, cu_seqlens_d)

PyLate drop-in
--------------
    from flash_colbert.pylate_compat import patch_pylate, unpatch_pylate
    patch_pylate()

Reference
---------
    from flash_colbert.reference import maxsim_reference, maxsim_reference_soft
"""

__version__ = "0.2.0"

from .autograd import (
    get_backward_method,
    maxsim,
    maxsim_inference,
    set_backward_method,
)
from .forward import maxsim_forward
from .soft import soft_maxsim
from .varlen import maxsim_varlen

__all__ = [
    "maxsim",
    "maxsim_inference",
    "maxsim_forward",
    "soft_maxsim",
    "maxsim_varlen",
    "set_backward_method",
    "get_backward_method",
    "__version__",
]
