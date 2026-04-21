"""late-interaction-kernels: fused Triton kernels for late-interaction scoring.

Public surface
--------------

Training / general MaxSim
    * ``maxsim(Q, D, q_mask=, d_mask=, normalize=)`` — autograd-aware.
    * ``maxsim_inference(...)`` — no saved argmax, inference-only.
    * ``soft_maxsim(...)`` — log-sum-exp relaxation (dense gradient).
    * ``maxsim_varlen(...)`` — packed / ragged inputs, autograd-aware.
    * ``maxsim_varlen_inference(...)`` — no saved argmax, inference-only.

Retrieval
    * ``maxsim_topk(Q, D, k, ...)`` — top-k docs + indices in one call.
    * ``plaid_approx_score(qcs, codes, doc_lengths)`` — ColBERTv2 IVF
      approximate scoring step, fused.
    * ``maxsim_residual(Q, codes, residuals, ...)`` — fused PLAID /
      ColBERTv2 2/4/8-bit decompression + L2-normalize + MaxSim (exact
      rerank step). Autograd-aware on Q; lets you train directly on
      compressed doc embeddings.
    * ``maxsim_residual_inference(...)`` — no saved argmax, inference-only.

Late-interaction variants
    * ``maxsim_matryoshka(Q, D, dims=[...])`` — multi-dim scoring in one pass.
    * ``maxsim_xtr(Q, D, top_k=5)`` — XTR top-k aggregated MaxSim.

PyLate drop-in
    * ``patch_pylate()`` / ``unpatch_pylate()``.

Advanced
    * ``set_backward_method("auto" | "csr" | "atomic")`` selects the grad_D path.
"""

__version__ = "0.5.1"

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
    from .matryoshka import maxsim_matryoshka
    from .plaid import maxsim_residual, maxsim_residual_inference, plaid_approx_score
    from .pylate_compat import patch_pylate, unpatch_pylate
    from .soft import soft_maxsim
    from .topk import maxsim_topk
    from .varlen import maxsim_varlen, maxsim_varlen_inference
    from .xtr import maxsim_xtr
else:  # pragma: no cover

    def _needs_triton(*_args, **_kwargs):  # type: ignore[no-redef]
        raise RuntimeError(
            "late-interaction-kernels's GPU kernels require Triton, which isn't installed on "
            "this platform. Install a CUDA-enabled Triton (Linux only) or use the "
            "reference implementations in `late_interaction_kernels.reference`."
        )

    maxsim = maxsim_inference = maxsim_forward = _needs_triton
    soft_maxsim = maxsim_varlen = maxsim_varlen_inference = _needs_triton
    maxsim_topk = maxsim_matryoshka = maxsim_xtr = _needs_triton
    plaid_approx_score = maxsim_residual = maxsim_residual_inference = _needs_triton
    set_backward_method = get_backward_method = _needs_triton
    patch_pylate = unpatch_pylate = _needs_triton

from . import reference  # noqa: E402,F401  — always importable (pure PyTorch)

__all__ = [
    "__version__",
    # core MaxSim
    "maxsim",
    "maxsim_inference",
    "maxsim_forward",
    "soft_maxsim",
    "maxsim_varlen",
    "maxsim_varlen_inference",
    # retrieval
    "maxsim_topk",
    "plaid_approx_score",
    "maxsim_residual",
    "maxsim_residual_inference",
    # variants
    "maxsim_matryoshka",
    "maxsim_xtr",
    # configuration
    "set_backward_method",
    "get_backward_method",
    # pylate
    "patch_pylate",
    "unpatch_pylate",
    "reference",
]
