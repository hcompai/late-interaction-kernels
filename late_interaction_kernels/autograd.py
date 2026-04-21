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
    def forward(ctx, Q, D, q_mask, d_mask, normalize):
        # All shape-normalization happens here; we expect Q=[Nq,Lq,d], D=[Nd,Ld,d]
        scores, argmax = _run_forward(Q, D, q_mask, d_mask, save_argmax=True, normalize=normalize)
        ctx.save_for_backward(Q, D, argmax, q_mask, d_mask)
        ctx.backward_method = _BACKWARD_METHOD
        ctx.normalize = normalize
        return scores

    @staticmethod
    def backward(ctx, grad_scores):
        Q, D, argmax, q_mask, d_mask = ctx.saved_tensors
        grad_scores = grad_scores.contiguous().to(torch.float32)

        if ctx.normalize:
            # The forward computed scores against Q_hat = Q / ||Q|| and D_hat = D / ||D||.
            # We need grad w.r.t. the *unnormalized* Q and D. We get that by
            # (a) running the existing backward against the normalized tensors to get
            # grad_Q_hat, grad_D_hat, then (b) applying the L2-normalize Jacobian.
            # This is a small number of extra ops per token, still fully on-GPU.
            q_norm = torch.linalg.vector_norm(Q, dim=-1, keepdim=True).clamp_min(1e-6)
            d_norm = torch.linalg.vector_norm(D, dim=-1, keepdim=True).clamp_min(1e-6)
            Q_hat = Q / q_norm
            D_hat = D / d_norm
            grad_Qh, grad_Dh = maxsim_backward(
                grad_scores,
                Q_hat,
                D_hat,
                argmax,
                q_mask,
                d_mask,
                method=ctx.backward_method,
            )
            # d Qhat / d Q = (I - Qhat Qhat^T) / ||Q||
            grad_Q = (grad_Qh - (grad_Qh * Q_hat).sum(-1, keepdim=True) * Q_hat) / q_norm
            grad_D = (grad_Dh - (grad_Dh * D_hat).sum(-1, keepdim=True) * D_hat) / d_norm
        else:
            grad_Q, grad_D = maxsim_backward(
                grad_scores,
                Q,
                D,
                argmax,
                q_mask,
                d_mask,
                method=ctx.backward_method,
            )
        # masks and normalize receive no gradient
        return grad_Q, grad_D, None, None, None


def maxsim(
    Q: torch.Tensor,
    D: torch.Tensor,
    q_mask: torch.Tensor | None = None,
    d_mask: torch.Tensor | None = None,
    *,
    normalize: bool = False,
) -> torch.Tensor:
    """Differentiable fused MaxSim. Drop-in for PyLate's `colbert_scores`.

    Args:
        Q: [Nq, Lq, d] or [Lq, d]
        D: [Nd, Ld, d] or [Ld, d]
        q_mask, d_mask: bool tensors matching the first two dims of Q / D.
        normalize: if True, L2-normalize Q and D per-token inside the kernel
            (saves one HBM round-trip vs ``F.normalize(Q) → maxsim``). The
            gradient correctly accounts for the normalization so the op is a
            true drop-in for ``maxsim(F.normalize(Q), F.normalize(D))``.

    Returns:
        scores: [Nq, Nd], fp32. Squeezed to match 2-D inputs.

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

    # Shape / device contract — fail fast with a clear message so user code
    # doesn't silently corrupt memory or produce garbage scores.
    if Q.shape[-1] != D.shape[-1]:
        raise ValueError(
            f"Q and D must share the embedding dim; got Q.shape[-1]={Q.shape[-1]} "
            f"vs D.shape[-1]={D.shape[-1]}."
        )
    if Q.device != D.device:
        raise ValueError(
            f"Q and D must be on the same device; got Q.device={Q.device} vs "
            f"D.device={D.device}."
        )
    if q_mask is not None and q_mask.device != Q.device:
        raise ValueError(
            f"q_mask must be on the same device as Q; got {q_mask.device} vs {Q.device}."
        )
    if d_mask is not None and d_mask.device != D.device:
        raise ValueError(
            f"d_mask must be on the same device as D; got {d_mask.device} vs {D.device}."
        )

    Q = Q.contiguous()
    D = D.contiguous()
    q_mask_i8 = q_mask.contiguous().to(torch.int8) if q_mask is not None else None
    d_mask_i8 = d_mask.contiguous().to(torch.int8) if d_mask is not None else None

    scores = _MaxSimFn.apply(Q, D, q_mask_i8, d_mask_i8, normalize)

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
    *,
    normalize: bool = False,
) -> torch.Tensor:
    """Same as ``maxsim`` but no autograd / no saved argmax. Use for reranking.

    Supports ``normalize=True`` to fuse L2-norm with the MaxSim reduction —
    this is the fast path for typical ColBERT reranking (the unnormalized
    embeddings are dequantized or encoder-output fp16).
    """
    if Q.shape[-1] != D.shape[-1]:
        raise ValueError(
            f"Q and D must share the embedding dim; got Q.shape[-1]={Q.shape[-1]} "
            f"vs D.shape[-1]={D.shape[-1]}."
        )
    if Q.device != D.device:
        raise ValueError(
            f"Q and D must be on the same device; got Q.device={Q.device} vs "
            f"D.device={D.device}."
        )
    scores, _ = maxsim_forward(
        Q,
        D,
        q_mask=q_mask,
        d_mask=d_mask,
        save_argmax=False,
        normalize=normalize,
    )
    return scores
