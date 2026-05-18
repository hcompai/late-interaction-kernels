"""Fused Triton kernels for late-interaction (MaxSim) scoring — HF Kernels build.

Loaded via::

    from kernels import get_kernel
    lik = get_kernel("Hcompany/late-interaction-kernels")

The v1 surface is the forward / inference path of ColBERT-family scoring:
MaxSim, varlen MaxSim, FP8 MaxSim, fused-head MaxSim, PLAID approx / residual.
The ``maxsim`` autograd entry point is also exported and supports backward for
users who want training gradients, but the ``MaxSim`` nn.Module exposed in
:mod:`late_interaction_kernels.layers` is forward-only (``has_backward=False``).

See the PyPI package ``late-interaction-kernels`` for the full API surface
(retrieval helpers, PyLate compatibility, padded reranking, experimental
variants).
"""

from late_interaction_kernels import layers
from late_interaction_kernels.autograd import maxsim, maxsim_inference
from late_interaction_kernels.fp8 import maxsim_inference_fp8
from late_interaction_kernels.fused_head import maxsim_from_hidden
from late_interaction_kernels.plaid import (
    maxsim_residual,
    maxsim_residual_inference,
    maxsim_residual_varlen,
    plaid_approx_score,
)
from late_interaction_kernels.varlen import maxsim_varlen, maxsim_varlen_inference

__kernel_metadata__ = {
    "license": "apache-2.0",
}

__all__ = [
    "__kernel_metadata__",
    "layers",
    # core MaxSim
    "maxsim",
    "maxsim_inference",
    "maxsim_varlen",
    "maxsim_varlen_inference",
    # fused head (forward)
    "maxsim_from_hidden",
    # FP8 inference
    "maxsim_inference_fp8",
    # PLAID / ColBERTv2 (forward)
    "plaid_approx_score",
    "maxsim_residual",
    "maxsim_residual_inference",
    "maxsim_residual_varlen",
]
