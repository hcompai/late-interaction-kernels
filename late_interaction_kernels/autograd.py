"""`torch.autograd.Function` wrapper — the user-facing API for training."""

from __future__ import annotations

import os
import warnings

import torch

from .backward import maxsim_backward
from .backward_unified import maxsim_backward_unified
from .forward import _run_forward, maxsim_forward

_BACKWARD_METHOD = "auto"  # module-level toggle, flipped by `set_backward_method`

_VALID_METHODS = ("auto", "atomic", "csr", "unified")

# One-shot flag so we don't spam the user's logs if they happen to pass
# unnormalized inputs inside a tight training loop.
_WARNED_UNNORMALIZED = False


def set_backward_method(method: str) -> None:
    """Set the **process-wide default** ``grad_D`` path.

    Prefer the per-call ``backward=`` kwarg on :func:`maxsim` (or on
    :class:`~late_interaction_kernels.MaxSimScorer`). This global is kept
    for backward compatibility and for pinning a single method across an
    entire benchmark run.

    Valid values:

    * ``"auto"`` (default) — pick ``"unified"`` for almost every shape,
      ``"csr"`` for very high ``grad_D`` contention
      (``Nq ≥ 256 ∧ Nd ≥ 256 ∧ Lq ≤ 64``). ``"atomic"`` is never picked
      by auto since 0.6.0.
    * ``"unified"`` — single-pass fused ``grad_Q + grad_D`` kernel (0.6.0).
      What ``auto`` picks for all typical training shapes.
    * ``"csr"`` — scatter-free bucketed reduction. Deterministic across
      runs, wins at very high contention.
    * ``"atomic"`` — legacy two-pass, fp32 ``tl.atomic_add``. Kept for
      benchmarking / fallback on GPUs with degraded atomic_add.
    """
    global _BACKWARD_METHOD
    if method not in _VALID_METHODS:
        raise ValueError(f"method must be one of {_VALID_METHODS}, got {method!r}")
    _BACKWARD_METHOD = method


def get_backward_method() -> str:
    return _BACKWARD_METHOD


def _maybe_warn_unnormalized(Q: torch.Tensor) -> None:
    """Warn once if Q is clearly not L2-normalized and the user didn't ask for `normalize=True`.

    ColBERT / ColPali / LateOn scores are always computed on L2-normalized
    token embeddings. Calling ``maxsim`` on raw encoder outputs silently
    produces different score scales than PyLate — a common footgun that
    eats hours of debugging. We emit a one-time warning if the token
    norms look far from 1. Disable with ``LIK_SUPPRESS_NORM_WARN=1``.
    """
    global _WARNED_UNNORMALIZED
    if _WARNED_UNNORMALIZED or os.environ.get("LIK_SUPPRESS_NORM_WARN", "0") == "1":
        return
    # Cheap sanity check: a handful of token norms.
    with torch.no_grad():
        sample = Q.detach()
        # Flatten leading dims, inspect up to the first 64 tokens.
        sample = sample.reshape(-1, sample.shape[-1])[:64]
        if sample.numel() == 0:
            return
        norms = sample.float().norm(dim=-1)
        med = norms.median().item()
    if not (0.9 <= med <= 1.1):
        _WARNED_UNNORMALIZED = True
        warnings.warn(
            "late-interaction-kernels: `maxsim(...)` was called with `normalize=False` but the "
            f"median Q token L2 norm is {med:.3f} (expected ≈1.0 for late-interaction scoring). "
            "ColBERT / ColPali / LateOn-style models score on L2-normalized token embeddings — "
            "pass `normalize=True` to fuse L2-norm into the kernel, or pre-normalize with "
            "`F.normalize(Q, dim=-1)`. Silence this warning with `LIK_SUPPRESS_NORM_WARN=1`.",
            UserWarning,
            stacklevel=3,
        )


class _MaxSimFn(torch.autograd.Function):
    """Autograd function for fused MaxSim, 3-D shapes, with saved argmax."""

    @staticmethod
    def forward(ctx, Q, D, q_mask, d_mask, normalize, backward_method):
        # All shape-normalization happens here; we expect Q=[Nq,Lq,d], D=[Nd,Ld,d]
        scores, argmax = _run_forward(Q, D, q_mask, d_mask, save_argmax=True, normalize=normalize)
        ctx.save_for_backward(Q, D, argmax, q_mask, d_mask)
        ctx.backward_method = backward_method
        ctx.normalize = normalize
        return scores

    @staticmethod
    def backward(ctx, grad_scores):
        Q, D, argmax, q_mask, d_mask = ctx.saved_tensors
        grad_scores = grad_scores.contiguous().to(torch.float32)

        # Resolve "auto" once, using problem shape. Unified is the default
        # for almost every training shape (see
        # benchmarks/bench_backward_unified.py); CSR only wins at very
        # high grad_D contention (B >= 256 with typical Lq / Ld).
        method = ctx.backward_method
        if method == "auto":
            Nq, Lq, _ = Q.shape
            Nd = D.shape[0]
            high_contention = Nq >= 256 and Nd >= 256 and Lq <= 64
            method = "csr" if high_contention else "unified"

        def _bwd(Qt, Dt):
            if method == "unified":
                return maxsim_backward_unified(grad_scores, Qt, Dt, argmax, q_mask=q_mask, method="atomic")
            return maxsim_backward(
                grad_scores,
                Qt,
                Dt,
                argmax,
                q_mask,
                d_mask,
                method=method,
            )

        if ctx.normalize:
            # The forward computed scores against Q_hat = Q / ||Q|| and D_hat = D / ||D||.
            # We need grad w.r.t. the *unnormalized* Q and D. We get that by
            # (a) running the existing backward against the normalized tensors to get
            # grad_Q_hat, grad_D_hat, then (b) applying the L2-normalize Jacobian.
            q_norm = torch.linalg.vector_norm(Q, dim=-1, keepdim=True).clamp_min(1e-6)
            d_norm = torch.linalg.vector_norm(D, dim=-1, keepdim=True).clamp_min(1e-6)
            Q_hat = Q / q_norm
            D_hat = D / d_norm
            grad_Qh, grad_Dh = _bwd(Q_hat, D_hat)
            # d Qhat / d Q = (I - Qhat Qhat^T) / ||Q||
            grad_Q = (grad_Qh - (grad_Qh * Q_hat).sum(-1, keepdim=True) * Q_hat) / q_norm
            grad_D = (grad_Dh - (grad_Dh * D_hat).sum(-1, keepdim=True) * D_hat) / d_norm
        else:
            grad_Q, grad_D = _bwd(Q, D)
        # masks, normalize, backward_method receive no gradient
        return grad_Q, grad_D, None, None, None, None


def maxsim(
    Q: torch.Tensor,
    D: torch.Tensor,
    q_mask: torch.Tensor | None = None,
    d_mask: torch.Tensor | None = None,
    *,
    normalize: bool = False,
    backward: str | None = None,
) -> torch.Tensor:
    """Differentiable fused MaxSim. Drop-in for PyLate's ``colbert_scores``.

    Args:
        Q: ``[Nq, Lq, d]`` or ``[Lq, d]``.
        D: ``[Nd, Ld, d]`` or ``[Ld, d]``.
        q_mask, d_mask: bool tensors matching the first two dims of Q / D.
            ``True`` = valid token, ``False`` = padding / masked.
        normalize: if ``True``, L2-normalize Q and D per-token inside the
            kernel (saves one HBM round-trip vs ``F.normalize(Q) → maxsim``).
            The gradient correctly accounts for the normalization so the op
            is a true drop-in for ``maxsim(F.normalize(Q), F.normalize(D))``.
            **Recommended for ColBERT / ColPali / LateOn**, which always
            score on L2-normalized embeddings.
        backward: per-call ``grad_D`` path override —
            ``"auto" | "unified" | "csr" | "atomic" | None``.
            ``None`` (default) defers to the process-wide
            :func:`set_backward_method` setting.

    Returns:
        scores: ``[Nq, Nd]`` fp32. Squeezed to match 2-D inputs.

    Notes:
        * Gradients flow into Q and D. Masks are non-differentiable.
        * FP32 accumulation inside the kernel; inputs can be fp16 / bf16 / fp32.
        * If Q doesn't look L2-normalized and ``normalize=False``, a one-time
          warning is emitted (silence with ``LIK_SUPPRESS_NORM_WARN=1``).
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
            f"Q and D must be on the same device; got Q.device={Q.device} vs D.device={D.device}."
        )
    if q_mask is not None and q_mask.device != Q.device:
        raise ValueError(f"q_mask must be on the same device as Q; got {q_mask.device} vs {Q.device}.")
    if d_mask is not None and d_mask.device != D.device:
        raise ValueError(f"d_mask must be on the same device as D; got {d_mask.device} vs {D.device}.")

    if backward is None:
        method = _BACKWARD_METHOD
    elif backward not in _VALID_METHODS:
        raise ValueError(f"backward= must be one of {_VALID_METHODS} or None, got {backward!r}")
    else:
        method = backward

    if not normalize:
        _maybe_warn_unnormalized(Q)

    Q = Q.contiguous()
    D = D.contiguous()
    q_mask_i8 = q_mask.contiguous().to(torch.int8) if q_mask is not None else None
    d_mask_i8 = d_mask.contiguous().to(torch.int8) if d_mask is not None else None

    scores = _MaxSimFn.apply(Q, D, q_mask_i8, d_mask_i8, normalize, method)

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
            f"Q and D must be on the same device; got Q.device={Q.device} vs D.device={D.device}."
        )
    if not normalize:
        _maybe_warn_unnormalized(Q)
    scores, _ = maxsim_forward(
        Q,
        D,
        q_mask=q_mask,
        d_mask=d_mask,
        save_argmax=False,
        normalize=normalize,
    )
    return scores
