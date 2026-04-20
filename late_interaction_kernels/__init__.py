"""late-interaction-kernels: fused Triton kernels for ColBERT / ColPali / ModernColBERT MaxSim.

Top-level API
-------------

    from late_interaction_kernels import maxsim, maxsim_inference, soft_maxsim, maxsim_varlen

    # autograd-aware (training)
    scores = maxsim(Q, D, q_mask=..., d_mask=...)              # [Nq, Nd]

    # inference-only (no saved argmax, lighter)
    scores = maxsim_inference(Q, D, q_mask=..., d_mask=...)

    # log-sum-exp relaxation (denser gradient)
    scores = soft_maxsim(Q, D, beta=10.0)

    # packed / varlen inputs (no padding waste)
    scores = maxsim_varlen(Q_packed, D_packed, cu_seqlens_q, cu_seqlens_d)

PyLate drop-in
--------------

    from late_interaction_kernels import patch_pylate, unpatch_pylate
    patch_pylate()

Backward-path selector (advanced)
---------------------------------

    from late_interaction_kernels import set_backward_method, get_backward_method
    set_backward_method("auto")     # default: heuristic picker
    set_backward_method("atomic")   # fp32 atomic_add
    set_backward_method("csr")      # sort + bucket reduce

Reference implementation (CPU / debugging)
------------------------------------------

    from late_interaction_kernels.reference import maxsim_reference
"""

__version__ = "0.3.0"

# The Triton kernels are not importable on platforms without Triton (macOS,
# Windows without a CUDA build). We still want ``import late_interaction_kernels`` and
# the reference implementation to work there — the library doubles as a
# correctness reference in those environments. Only ``patch_pylate`` and the
# kernel-backed APIs are gated on Triton being available.
try:
    import triton  # noqa: F401

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

if _HAS_TRITON:
    from .autograd import (
        get_backward_method,
        maxsim,
        maxsim_inference,
        set_backward_method,
    )
    from .forward import maxsim_forward
    from .pylate_compat import patch_pylate, unpatch_pylate
    from .soft import soft_maxsim
    from .varlen import maxsim_varlen
else:  # pragma: no cover

    def _needs_triton(*_args, **_kwargs):  # type: ignore[no-redef]
        raise RuntimeError(
            "late-interaction-kernels's GPU kernels require Triton, which isn't installed on "
            "this platform. Install a CUDA-enabled Triton (Linux only) or use "
            "`late_interaction_kernels.reference.maxsim_reference` for a pure-PyTorch fallback."
        )

    maxsim = maxsim_inference = maxsim_forward = _needs_triton
    soft_maxsim = maxsim_varlen = _needs_triton
    set_backward_method = get_backward_method = _needs_triton
    patch_pylate = unpatch_pylate = _needs_triton

from . import reference  # noqa: E402,F401  — always importable (pure PyTorch)

__all__ = [
    "__version__",
    "maxsim",
    "maxsim_inference",
    "maxsim_forward",
    "soft_maxsim",
    "maxsim_varlen",
    "set_backward_method",
    "get_backward_method",
    "patch_pylate",
    "unpatch_pylate",
    "reference",
]
