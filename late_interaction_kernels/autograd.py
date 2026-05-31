"""User-facing autograd wrapper for the fused MaxSim kernel."""

import os
import warnings

import torch
import torch.nn.functional as F

from late_interaction_kernels._utils import next_pow2
from late_interaction_kernels.backward import maxsim_backward, maxsim_backward_unified
from late_interaction_kernels.forward import _run_forward

# Smallest BLOCK_Q across our autotune pools is 16, so any Lq below that
# would be pruned to a fallback config anyway. Use 16 as the bucket floor.
_LQ_BUCKET_FLOOR = 16

# Above this we stop bucketing and pass Lq through with a one-shot warning.
# The cap is set to cover the realistic dense-MaxSim workload range:
# ColBERT (≤ 32), ColPali (~1030 visual patches → bucket 2048), long-doc
# rerank up to ~4 k. Past 4096 the static_range unroll over Lq/BLOCK_Q
# starts to dominate compile time and the caller is in genuine long-context
# territory where :func:`maxsim_varlen` is the right tool (it buckets on
# ``max_lq`` over a ``range`` loop, no static unroll).
_LQ_BUCKET_CEIL = 4096

_WARNED_LQ_OVER_CEIL = False


def _bucket_lq(Q: torch.Tensor, q_mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Round Lq up to the next power of two so Triton's autotune cache reuses
    a config across batches with slightly different query lengths.

    Without this, every distinct ``Lq`` (e.g. 7, 9, 12, 17, ...) re-triggers
    the full autotune sweep — variable-length training paid up to ~21 s of
    pure overhead per new value. Bucketing to {16, 32, ..., 2048, 4096}
    caps the cache at 9 entries while keeping ``Lq`` constexpr inside the
    kernel (preserving the ``tl.static_range`` unroll).

    Pads ``Q`` with zeros along the ``Lq`` axis and extends (or creates)
    ``q_mask`` so the kernel ignores the padded rows in the max reduction
    and the backward zero-grads them.

    Past ``_LQ_BUCKET_CEIL`` we emit a one-shot warning and pass Lq through
    (each value gets its own autotune entry — the v0.2.0 behaviour for any
    Lq). Use :func:`maxsim_varlen` for genuine long-context workloads.
    """
    Lq = Q.shape[-2]
    if Lq > _LQ_BUCKET_CEIL:
        global _WARNED_LQ_OVER_CEIL
        if not _WARNED_LQ_OVER_CEIL:
            warnings.warn(
                f"maxsim: Lq={Lq} > {_LQ_BUCKET_CEIL}; falling back to per-Lq autotune "
                "(each distinct value re-triggers the Triton sweep). For genuine "
                "long-context use `maxsim_varlen`, which buckets on `max_lq` via a "
                "non-unrolled loop and avoids this entirely.",
                RuntimeWarning,
                stacklevel=3,
            )
            _WARNED_LQ_OVER_CEIL = True
        return Q, q_mask
    bucket = max(_LQ_BUCKET_FLOOR, next_pow2(Lq))
    if bucket == Lq:
        return Q, q_mask

    pad = bucket - Lq
    Q = F.pad(Q, (0, 0, 0, pad))
    if q_mask is None:
        q_mask = torch.ones(Q.shape[:-1], dtype=torch.bool, device=Q.device)
        q_mask[..., Lq:] = False
    else:
        q_mask = F.pad(q_mask, (0, pad), value=False)
    return Q, q_mask


# Query-token chunk size for the large-Lq forward. Splitting Lq into fixed
# blocks of this many tokens and summing the per-block MaxSim is exact
# (MaxSim is a sum over query tokens of a per-token max), but launches
# ``ceil(Lq / chunk)`` more programs — which keeps the H100 busy when a long
# query (ColPali ~1k visual patches) would otherwise serialise a long
# ``static_range`` loop inside one program. Picking a power-of-two chunk also
# pins the kernel's ``Lq`` constexpr to a single value, so the autotune cache
# collapses every large-Lq workload onto one entry instead of one per bucket.
_LQ_CHUNK = 128

# Only chunk once the query is long enough that the per-program serial loop
# dominates the extra-program launch/scheduling overhead. Below the crossover
# the un-chunked launch keeps the grid lean (and is already the faster path);
# above it, chunking both fills the GPU and shortens the inner loop. The
# crossover measured on H100 sits between Lq=512 (neutral) and Lq=1024 (up to
# +40%), so we trigger strictly above 512 — short/medium queries (ColBERT
# Lq≤32, Lq=256/512 long-form) are left untouched.
_LQ_CHUNK_MIN = 512


def _should_chunk_lq(Lq: int) -> bool:
    return Lq > _LQ_CHUNK_MIN


_VALID_METHODS = ("auto", "atomic", "csr", "unified")

# One-shot flag so we don't spam the user's logs if they happen to pass
# unnormalized inputs inside a tight training loop.
_WARNED_UNNORMALIZED = False


def _maybe_warn_unnormalized(Q: torch.Tensor) -> None:
    """Warn once when ``normalize=False`` is paired with non-normalized Q.

    ColBERT / ColPali / LateOn always score L2-normalized tokens. Calling
    ``maxsim`` on raw encoder outputs silently produces different score
    scales than PyLate. Silence with ``LIK_SUPPRESS_NORM_WARN=1``.
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
            f"late-interaction-kernels: `maxsim(..., normalize=False)` but Q's median L2 norm "
            f"is {med:.3f} (ColBERT-style models expect ≈1.0). Pass `normalize=True` to fuse "
            "the L2-norm into the kernel, or pre-normalize with `F.normalize(Q, dim=-1)`. "
            "Silence with `LIK_SUPPRESS_NORM_WARN=1`.",
            UserWarning,
            stacklevel=3,
        )


class _MaxSimFn(torch.autograd.Function):
    """Fused MaxSim with saved argmax, 3-D inputs.

    When ``kd_layout=True``, ``D`` is the flat ``[Nq * K, Ld, d]`` view of a
    KD/pairs batch (see :func:`_maxsim_kd_fast`); the forward kernel uses
    ``d_global = pid`` so each query reads its own slab.
    """

    @staticmethod
    def forward(ctx, Q, D, q_mask, d_mask, normalize, backward_method, kd_layout):
        scores, argmax = _run_forward(
            Q, D, q_mask, d_mask, save_argmax=True, normalize=normalize, kd_layout=kd_layout
        )
        ctx.save_for_backward(Q, D, argmax, q_mask, d_mask)
        ctx.backward_method = backward_method
        ctx.normalize = normalize
        ctx.kd_layout = kd_layout
        return scores

    @staticmethod
    def backward(ctx, grad_scores):
        Q, D, argmax, q_mask, d_mask = ctx.saved_tensors
        grad_scores = grad_scores.contiguous().to(torch.float32)
        kd_layout = ctx.kd_layout

        # `auto` -> `unified` for typical training shapes; `csr` only when
        # `grad_D` contention is very high (large square batches, short Lq).
        # KD/pairs has no cross-query contention on grad_D (each pair owns
        # its own slab), so we always pick the cheaper unified path there.
        method = ctx.backward_method
        if method == "auto":
            if kd_layout:
                method = "unified"
            else:
                Nq, Lq, _ = Q.shape
                Nd = D.shape[0]
                high_contention = Nq >= 256 and Nd >= 256 and Lq <= 64
                method = "csr" if high_contention else "unified"

        def _bwd(Qt, Dt):
            if method == "unified":
                return maxsim_backward_unified(
                    grad_scores, Qt, Dt, argmax, q_mask=q_mask, method="atomic", kd_layout=kd_layout
                )
            return maxsim_backward(
                grad_scores,
                Qt,
                Dt,
                argmax,
                q_mask,
                d_mask,
                method=method,
                kd_layout=kd_layout,
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
        # masks, normalize, backward_method, kd_layout receive no gradient.
        return grad_Q, grad_D, None, None, None, None, None


def _maxsim_kd(
    Q: torch.Tensor,
    D: torch.Tensor,
    q_mask: torch.Tensor | None,
    d_mask: torch.Tensor | None,
    *,
    normalize: bool,
    backward_method: str,
) -> torch.Tensor:
    """KD layout ``[Nq, Lq, d] x [Nq, K, Ld, d] -> [Nq, K]``.

    Reshapes ``D`` to the flat ``[Nq * K, Ld, d]`` view and dispatches to the
    same fast forward kernel as in-batch, with ``kd_layout=True`` so each
    program reads its own ``K``-slab (no cross-product, no packing). One
    fused launch, full ``tl.static_range`` unroll over ``Lq``, identical
    backward kernels as the cross-product path (KD just skips the CSR
    branch because there's no cross-query contention on ``grad_D``).
    """
    if Q.dim() != 3:
        raise ValueError(f"KD layout (D.dim()==4) needs Q to be [Nq, Lq, d]; got Q.shape={tuple(Q.shape)}")
    Nq, Lq, d_q = Q.shape
    Nq_d, K, Ld, d_d = D.shape
    if Nq != Nq_d:
        raise ValueError(
            f"KD layout needs D.shape[0] == Q.shape[0]; got Q.shape[0]={Nq} vs D.shape[0]={Nq_d}."
        )
    if d_q != d_d:
        raise ValueError(f"Q and D must share the embedding dim; got {d_q} vs {d_d}.")
    if q_mask is not None and q_mask.shape != (Nq, Lq):
        raise ValueError(f"q_mask must be [Nq={Nq}, Lq={Lq}] for KD layout; got {tuple(q_mask.shape)}.")
    if d_mask is not None and d_mask.shape != (Nq, K, Ld):
        raise ValueError(
            f"d_mask must be [Nq={Nq}, K={K}, Ld={Ld}] for KD layout; got {tuple(d_mask.shape)}."
        )

    # ``F.pad`` + ``reshape(Nq * K, ...)`` keeps strides contiguous so the view
    # below is a real zero-copy reinterpretation, not a hidden materialise.
    Q = Q.contiguous()
    D = D.contiguous()
    D_flat = D.view(Nq * K, Ld, d_d)
    d_mask_flat = d_mask.contiguous().view(Nq * K, Ld) if d_mask is not None else None

    Q, q_mask = _bucket_lq(Q, q_mask)
    q_mask_i8 = q_mask.contiguous().to(torch.int8) if q_mask is not None else None
    d_mask_i8 = d_mask_flat.to(torch.int8) if d_mask_flat is not None else None

    if Q.requires_grad or D.requires_grad:
        scores = _MaxSimFn.apply(Q, D_flat, q_mask_i8, d_mask_i8, normalize, backward_method, True)
    else:
        scores, _ = _run_forward(
            Q, D_flat, q_mask_i8, d_mask_i8, save_argmax=False, normalize=normalize, kd_layout=True
        )
    # scores comes out as [Nq, K] — exactly the requested KD shape.
    return scores


def _maxsim_cross(
    Q: torch.Tensor,
    D: torch.Tensor,
    q_mask: torch.Tensor | None,
    d_mask: torch.Tensor | None,
    normalize: bool,
    method: str,
) -> torch.Tensor:
    """Cross-product core: ``Q[Nq, Lq, d] x D[Nd, Ld, d] -> [Nq, Nd]``.

    Buckets ``Lq`` for autotune-cache reuse, then runs the fused kernel
    (autograd-aware when either input needs a gradient, plain forward
    otherwise). Inputs are assumed already 3-D and validated by the caller.
    """
    Q, q_mask = _bucket_lq(Q, q_mask)

    Q = Q.contiguous()
    D = D.contiguous()
    q_mask_i8 = q_mask.contiguous().to(torch.int8) if q_mask is not None else None
    d_mask_i8 = d_mask.contiguous().to(torch.int8) if d_mask is not None else None

    # Skip the argmax save when neither input needs a backward — the fused
    # kernel is otherwise identical. Same pattern as `maxsim_varlen` and
    # `maxsim_residual`.
    if Q.requires_grad or D.requires_grad:
        return _MaxSimFn.apply(Q, D, q_mask_i8, d_mask_i8, normalize, method, False)
    scores, _ = _run_forward(
        Q, D, q_mask_i8, d_mask_i8, save_argmax=False, normalize=normalize, kd_layout=False
    )
    return scores


def _maxsim_cross_chunked(
    Q: torch.Tensor,
    D: torch.Tensor,
    q_mask: torch.Tensor | None,
    d_mask: torch.Tensor | None,
    normalize: bool,
    method: str,
) -> torch.Tensor:
    """Large-Lq cross product via query-token chunking.

    Reshapes the ``Lq`` axis into ``nc = ceil(Lq / _LQ_CHUNK)`` blocks, scores
    each ``(query, chunk)`` as an independent query through :func:`_maxsim_cross`
    (so the kernel always sees ``Lq == _LQ_CHUNK``), then sums the chunk scores
    back per original query. Numerically identical to the un-chunked path.
    """
    Nq, Lq, d = Q.shape
    nc = (Lq + _LQ_CHUNK - 1) // _LQ_CHUNK
    pad = nc * _LQ_CHUNK - Lq
    if pad:
        # Mask the padding tokens out: a zero row would otherwise contribute a
        # spurious max. Fold into the caller's q_mask when present.
        Q = F.pad(Q, (0, 0, 0, pad))
        if q_mask is None:
            q_mask = torch.ones(Nq, Lq + pad, dtype=torch.bool, device=Q.device)
            q_mask[:, Lq:] = False
        else:
            q_mask = F.pad(q_mask, (0, pad), value=False)

    Qc = Q.reshape(Nq * nc, _LQ_CHUNK, d)
    qmc = q_mask.reshape(Nq * nc, _LQ_CHUNK) if q_mask is not None else None
    scores = _maxsim_cross(Qc, D, qmc, d_mask, normalize, method)  # [Nq*nc, Nd]
    return scores.view(Nq, nc, scores.shape[-1]).sum(dim=1)


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

    Dispatches on ``D``'s rank so the same call covers in-batch scoring and
    per-query candidate scoring (PyLate's ``colbert_kd_scores`` shape):

    Args:
        Q: ``[Nq, Lq, d]`` or ``[Lq, d]``.
        D: one of

            * ``[Ld, d]`` — single doc → returns a scalar (with 2-D ``Q``)
              or ``[Nq]`` (with 3-D ``Q``).
            * ``[Nd, Ld, d]`` — in-batch cross product → returns ``[Nq, Nd]``.
            * ``[Nq, K, Ld, d]`` — per-query candidate lists (KD layout) →
              returns ``[Nq, K]``. Internally dispatches the same fast
              forward kernel as in-batch, with a constexpr flag that switches
              the per-program doc index from ``pid % Nd`` (cross-product) to
              ``pid`` (each query owns its K-slab). One launch, no Python
              loop, no packing.
        q_mask, d_mask: bool tensors (``True`` = valid token). Shapes mirror
            the matching ``Q``/``D`` axes. Any boolean pattern is supported
            (masked positions get ``-inf`` scores before the row max).
        normalize: L2-normalize Q and D per-token inside the kernel. Set to
            ``True`` for ColBERT / ColPali / LateOn-style scoring.
        backward: ``grad_D`` strategy
            (``"auto" | "unified" | "csr" | "atomic"``). ``None`` (default)
            is treated as ``"auto"``. KD/pairs always use ``unified`` (no
            cross-query contention on ``grad_D``).

    Returns:
        scores: fp32, shape as above.

    Inputs can be fp16 / bf16 / fp32 (fp32 accumulator). Gradients flow
    into Q and D; masks are non-differentiable.
    """
    if backward is None:
        method_for_kd = "auto"
    elif backward not in _VALID_METHODS:
        raise ValueError(f"backward= must be one of {_VALID_METHODS} or None, got {backward!r}")
    else:
        method_for_kd = backward

    # KD layout: D is [Nq, K, Ld, d]. Delegate to the unified fast path,
    # which runs the standard forward kernel with ``kd_layout=True`` so each
    # program reads its own K-slab — same kernel, same backward family, no
    # Python loop, no packing overhead.
    if D.dim() == 4:
        return _maxsim_kd(Q, D, q_mask, d_mask, normalize=normalize, backward_method=method_for_kd)

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

    method = method_for_kd  # already resolved + validated above

    if not normalize:
        _maybe_warn_unnormalized(Q)

    # Long queries (ColPali-scale Lq) split into fixed 128-token chunks: more
    # programs on the grid, shorter per-program loops, and a single autotune
    # entry. The per-chunk MaxSim sums back to the exact full-Lq score, and
    # the sum is a plain tensor op so autograd flows through unchanged.
    if _should_chunk_lq(Q.shape[-2]):
        scores = _maxsim_cross_chunked(Q, D, q_mask, d_mask, normalize, method)
    else:
        scores = _maxsim_cross(Q, D, q_mask, d_mask, normalize, method)

    if q_was_2d and d_was_2d:
        return scores.reshape(())
    if q_was_2d:
        return scores.squeeze(0)
    if d_was_2d:
        return scores.squeeze(-1)
    return scores


def maxsim_pairs(
    Q: torch.Tensor,
    D: torch.Tensor,
    q_mask: torch.Tensor | None = None,
    d_mask: torch.Tensor | None = None,
    *,
    normalize: bool = False,
    backward: str | None = None,
) -> torch.Tensor:
    """Diagonal (paired) MaxSim — for PyLate's ``colbert_scores_pairwise``.

    Computes one score per same-index pair ``(Q[i], D[i])`` and returns
    ``[B]``. Internally this is the ``K = 1`` case of the KD layout, so it
    routes through the same fast forward kernel as
    :func:`maxsim` — no full ``[B, B]`` cross product, no Python loop, no
    packing.

    Args:
        Q: ``[B, Lq, d]``.
        D: ``[B, Ld, d]``. ``B`` must match ``Q``'s.
        q_mask: optional ``[B, Lq]`` bool mask.
        d_mask: optional ``[B, Ld]`` bool mask.
        normalize: per-token L2-normalize inside the kernel.
        backward: same semantics as :func:`maxsim` (auto / unified / atomic).

    Returns:
        scores: ``[B]`` fp32.
    """
    if Q.dim() != 3 or D.dim() != 3:
        raise ValueError(
            f"maxsim_pairs needs both Q and D to be [B, L, d]; got Q.shape={tuple(Q.shape)}, "
            f"D.shape={tuple(D.shape)}."
        )
    if Q.shape[0] != D.shape[0]:
        raise ValueError(f"maxsim_pairs needs Q.shape[0] == D.shape[0]; got {Q.shape[0]} vs {D.shape[0]}.")
    # Same code path as KD with K=1; the ``unsqueeze`` is a view (free).
    D_kd = D.unsqueeze(1)
    d_mask_kd = d_mask.unsqueeze(1) if d_mask is not None else None
    out = maxsim(Q, D_kd, q_mask=q_mask, d_mask=d_mask_kd, normalize=normalize, backward=backward)
    return out.squeeze(-1)
