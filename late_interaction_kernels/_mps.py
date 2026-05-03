"""MPS dispatch for the high-level :func:`retrieve` and :class:`MaxSimScorer`.

Apple Silicon GPUs run PyTorch through the MPS backend, which has no
Triton support. The kernels in :mod:`late_interaction_kernels.forward`
and friends therefore can't run on ``mps:0`` tensors.

Instead, we hand the dense MaxSim formula in
:func:`~late_interaction_kernels.reference.maxsim_reference` to
``torch.compile``. Inductor lowers the einsum + max + sum chain to a
single Metal compute graph that's typically ≈2× faster than eager on
M-series GPUs. The compile cost is amortised by caching the compiled
callable per ``(dtype, normalize, has_q_mask, has_d_mask)`` signature.

The compile path is autograd-aware: gradients flow through ``Q`` and
``D`` exactly as they do in the eager reference. We therefore reuse it
for both inference and training-time scoring on MPS.

Why not a hand-written Metal kernel? Apple's MPSGraph already lowers
matmul to ``simdgroup_matrix`` (the Metal MMA primitive), which a naive
scalar Metal kernel can't beat. Outperforming MPSGraph requires a
proper ``simdgroup_matrix`` GEMM with MaxSim fused on top — a separate
piece of work tracked in the design notes.
"""

from __future__ import annotations

import os
import threading
from typing import Callable

import torch

from .reference import maxsim_reference

# Module-level compile cache. Keys: ``(dtype, normalize, has_q_mask, has_d_mask)``.
# Values: the compiled callable. Compile is single-threaded — guard with a lock
# so the first call from multi-threaded code doesn't double-compile.
_compile_lock = threading.Lock()
_compiled_cache: dict[tuple, Callable] = {}


def _disable_compile() -> bool:
    """Honour ``LIK_DISABLE_COMPILE=1`` so users can opt out."""
    return os.environ.get("LIK_DISABLE_COMPILE", "0") == "1"


def _compile_key(
    Q: torch.Tensor,
    normalize: bool,
    q_mask: torch.Tensor | None,
    d_mask: torch.Tensor | None,
) -> tuple:
    return (Q.dtype, bool(normalize), q_mask is not None, d_mask is not None)


def _get_compiled(key: tuple) -> Callable:
    fn = _compiled_cache.get(key)
    if fn is not None:
        return fn
    with _compile_lock:
        fn = _compiled_cache.get(key)
        if fn is None:
            # ``dynamic=True`` lets a single compile handle every (Nq, Nd, Lq, Ld)
            # shape — important on Mac where shape variance per call is normal.
            fn = torch.compile(
                maxsim_reference,
                mode="reduce-overhead",
                dynamic=True,
                fullgraph=False,
            )
            _compiled_cache[key] = fn
    return fn


def is_mps_tensor(x: torch.Tensor) -> bool:
    return x.device.type == "mps"


def maxsim_mps(
    Q: torch.Tensor,
    D: torch.Tensor,
    q_mask: torch.Tensor | None = None,
    d_mask: torch.Tensor | None = None,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """``torch.compile``-fused MaxSim on MPS. Autograd-aware."""
    if _disable_compile():
        return maxsim_reference(Q, D, q_mask=q_mask, d_mask=d_mask, normalize=normalize)
    fn = _get_compiled(_compile_key(Q, normalize, q_mask, d_mask))
    return fn(Q, D, q_mask=q_mask, d_mask=d_mask, normalize=normalize)


def maxsim_inference_mps(
    Q: torch.Tensor,
    D: torch.Tensor,
    q_mask: torch.Tensor | None = None,
    d_mask: torch.Tensor | None = None,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Inference-only MPS path. Same compile cache as :func:`maxsim_mps`."""
    with torch.no_grad():
        return maxsim_mps(Q, D, q_mask=q_mask, d_mask=d_mask, normalize=normalize)


__all__ = ["is_mps_tensor", "maxsim_mps", "maxsim_inference_mps"]
