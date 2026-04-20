"""`torch.autograd.Function` wrapper — the user-facing API for training."""

from __future__ import annotations

import torch

from .backward import maxsim_backward
from .forward import _run_forward, maxsim_forward

_BACKWARD_METHOD = "auto"  # module-level toggle, flipped by `set_backward_method`


def set_backward_method(method: str) -> None:
    """Select the grad_D path used by ``maxsim.backward``.

    Valid values:

    * ``"auto"`` (default) — pick ``csr`` vs ``atomic`` based on workload
      size (``Nq * Nd * Lq * d`` vs 1e8 on H100). Best default for both
      small-batch and large-batch training.
    * ``"csr"`` — scatter-free bucketed reduction. Faster on large/long
      shapes (train-256+, ColPali-style, large corpora).
    * ``"atomic"`` — fp32 ``tl.atomic_add``. Faster on common PyLate shapes
      (train-32..128) on H100 thanks to its hardware-accelerated atomics.

    Process-wide — intended for benchmarking, not per-call configuration.
    """
    global _BACKWARD_METHOD
    if method not in ("csr", "atomic", "auto"):
        raise ValueError(f"method must be 'csr', 'atomic', or 'auto', got {method!r}")
    _BACKWARD_METHOD = method


def get_backward_method() -> str:
    return _BACKWARD_METHOD


class _MaxSimFn(torch.autograd.Function):
    """Autograd function for fused MaxSim, 3-D shapes, with saved argmax."""

    @staticmethod
    def forward(ctx, Q, D, q_mask, d_mask):
        # All shape-normalization happens here; we expect Q=[Nq,Lq,d], D=[Nd,Ld,d]
        scores, argmax = _run_forward(Q, D, q_mask, d_mask, save_argmax=True)
        ctx.save_for_backward(Q, D, argmax, q_mask, d_mask)
        ctx.backward_method = _BACKWARD_METHOD
        return scores

    @staticmethod
    def backward(ctx, grad_scores):
        Q, D, argmax, q_mask, d_mask = ctx.saved_tensors
        grad_scores = grad_scores.contiguous().to(torch.float32)
        grad_Q, grad_D = maxsim_backward(
            grad_scores,
            Q,
            D,
            argmax,
            q_mask,
            d_mask,
            method=ctx.backward_method,
        )
        # masks receive no gradient
        return grad_Q, grad_D, None, None


def maxsim(
    Q: torch.Tensor,
    D: torch.Tensor,
    q_mask: torch.Tensor | None = None,
    d_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable fused MaxSim. Drop-in for PyLate's `colbert_scores`.

    Inputs:
        Q: [Nq, Lq, d] or [Lq, d]
        D: [Nd, Ld, d] or [Ld, d]
        q_mask, d_mask: bool tensors matching the first two dims of Q / D.

    Output:
        scores: [Nq, Nd], or squeezed accordingly. Always fp32.

    Notes:
        * Gradients flow into Q and D. Masks are non-differentiable.
        * Uses FP32 accumulation inside the kernel; input can be fp16 / bf16 / fp32.
    """
    q_was_2d = Q.dim() == 2
    d_was_2d = D.dim() == 2
    if q_was_2d:
        Q = Q.unsqueeze(0)
    if d_was_2d:
        D = D.unsqueeze(0)
    if q_mask is not None and q_mask.dim() == 1:
        q_mask = q_mask.unsqueeze(0)
    if d_mask is not None and d_mask.dim() == 1:
        d_mask = d_mask.unsqueeze(0)

    Q = Q.contiguous()
    D = D.contiguous()
    q_mask_i8 = q_mask.contiguous().to(torch.int8) if q_mask is not None else None
    d_mask_i8 = d_mask.contiguous().to(torch.int8) if d_mask is not None else None

    scores = _MaxSimFn.apply(Q, D, q_mask_i8, d_mask_i8)

    if q_was_2d and d_was_2d:
        return scores.reshape(())
    if q_was_2d:
        return scores.squeeze(0)
    if d_was_2d:
        return scores.squeeze(-1)
    return scores


# Inference-only alias that does not save argmax (smaller memory)
def maxsim_inference(
    Q: torch.Tensor,
    D: torch.Tensor,
    q_mask: torch.Tensor | None = None,
    d_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Same as `maxsim` but no autograd / no saved argmax. Use for reranking."""
    scores, _ = maxsim_forward(Q, D, q_mask=q_mask, d_mask=d_mask, save_argmax=False)
    return scores
