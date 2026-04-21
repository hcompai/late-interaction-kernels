"""Variable-length (packed) MaxSim kernel — forward + fused backward.

This is the **zero-padding** path: queries and docs arrive as `[total_tokens, d]`
tensors, and per-sequence lengths come via `cu_seqlens` (CSR-style offsets),
exactly like FlashAttention's varlen mode.

Why it matters for ColBERT / ColPali
------------------------------------
PyLate's ``rerank()`` does ``torch.nn.utils.rnn.pad_sequence`` on documents
before scoring. That's a deep-copy of every doc up to ``Ld_max``, with ~50 %
waste on realistic distributions (some docs 32 tokens, some 512). Varlen
skips padding entirely in both the forward and the backward.

Forward API
-----------
    scores = maxsim_varlen(
        Q_packed,            # [sum(Lq_i), d]
        D_packed,            # [sum(Ld_j), d]
        cu_seqlens_q,        # [Nq + 1] int32, cumulative sum
        cu_seqlens_d,        # [Nd + 1] int32, cumulative sum
        max_seqlen_q,        # int (optional, inferred if omitted)
        max_seqlen_d,        # int
    )                        # -> [Nq, Nd] fp32

Backward
--------
When ``Q_packed`` or ``D_packed`` requires grad, the forward saves an argmax
buffer of shape ``[Nq, Nd, max_seqlen_q]`` (int32, ``-1`` for invalid query
positions) and a fused backward kernel recovers ``grad_Q`` (row-owned, no
atomics) and ``grad_D`` (fp32 ``atomic_add`` into the packed grad tensor).

Internally we dispatch one Triton program per ``(q_batch, d_batch)`` pair.
Each program reads its own ``cu_seqlens`` entries, bounds-checks loads, and
runs the same inner loop as the padded kernel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._autotune import forward_configs, prune_forward
from ._utils import next_pow2, pick_compute_dtype


@triton.autotune(
    configs=forward_configs(),
    key=["max_lq", "max_ld", "d_pad"],
    prune_configs_by={"early_config_prune": prune_forward},
)
@triton.jit
def _varlen_fwd_kernel(
    Q_ptr,
    D_ptr,
    cu_q_ptr,
    cu_d_ptr,
    scores_ptr,
    argmax_ptr,
    Nq: tl.constexpr,
    Nd: tl.constexpr,
    max_lq: tl.constexpr,
    max_ld: tl.constexpr,
    d: tl.constexpr,
    d_pad: tl.constexpr,
    stride_q_t,
    stride_q_k,
    stride_d_t,
    stride_d_k,
    stride_s_n,
    stride_s_d,
    stride_am_n,
    stride_am_d,
    stride_am_l,
    Lq: tl.constexpr,
    Ld: tl.constexpr,  # unused, kept for autotune key compat
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    SAVE_ARGMAX: tl.constexpr,
):
    pid = tl.program_id(0)
    q_idx = pid // Nd
    d_idx = pid % Nd

    q_lo = tl.load(cu_q_ptr + q_idx).to(tl.int32)
    q_hi = tl.load(cu_q_ptr + q_idx + 1).to(tl.int32)
    d_lo = tl.load(cu_d_ptr + d_idx).to(tl.int32)
    d_hi = tl.load(cu_d_ptr + d_idx + 1).to(tl.int32)

    lq = q_hi - q_lo
    ld = d_hi - d_lo

    k_off = tl.arange(0, d_pad)
    k_mask = k_off < d
    score_acc = tl.zeros([], dtype=tl.float32)

    # Empty sequences contribute zero.
    if lq == 0 or ld == 0:
        tl.store(scores_ptr + q_idx * stride_s_n + d_idx * stride_s_d, score_acc)
        if SAVE_ARGMAX:
            # Fill the argmax row with -1 so the backward skips it cleanly.
            for q_start in range(0, max_lq, BLOCK_Q):
                q_off = q_start + tl.arange(0, BLOCK_Q)
                tl.store(
                    argmax_ptr + q_idx * stride_am_n + d_idx * stride_am_d + q_off * stride_am_l,
                    tl.full([BLOCK_Q], -1, dtype=tl.int32),
                    mask=q_off < max_lq,
                )
        return

    for q_start in range(0, max_lq, BLOCK_Q):
        q_off = q_start + tl.arange(0, BLOCK_Q)
        q_valid = q_off < lq

        Q_block = tl.load(
            Q_ptr + (q_lo + q_off)[:, None] * stride_q_t + k_off[None, :] * stride_q_k,
            mask=q_valid[:, None] & k_mask[None, :],
            other=0.0,
        ).to(COMPUTE_DTYPE)

        m = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
        am = tl.zeros([BLOCK_Q], dtype=tl.int32)

        for d_start in range(0, max_ld, BLOCK_D):
            d_off = d_start + tl.arange(0, BLOCK_D)
            d_valid = d_off < ld

            D_block = tl.load(
                D_ptr + (d_lo + d_off)[:, None] * stride_d_t + k_off[None, :] * stride_d_k,
                mask=d_valid[:, None] & k_mask[None, :],
                other=0.0,
            ).to(COMPUTE_DTYPE)

            S = tl.dot(Q_block, tl.trans(D_block), out_dtype=tl.float32)
            S = tl.where(d_valid[None, :], S, float("-inf"))
            tile_max = tl.max(S, axis=1)
            tile_arg = tl.argmax(S, axis=1).to(tl.int32) + d_start
            update = tile_max > m
            m = tl.where(update, tile_max, m)
            am = tl.where(update, tile_arg, am)

        m = tl.where(q_valid & (m != float("-inf")), m, 0.0)
        score_acc += tl.sum(m)

        if SAVE_ARGMAX:
            # Store -1 for padding query positions; the bwd uses it as "skip".
            am_out = tl.where(q_valid, am, -1)
            tl.store(
                argmax_ptr + q_idx * stride_am_n + d_idx * stride_am_d + q_off * stride_am_l,
                am_out,
                mask=q_off < max_lq,
            )

    tl.store(scores_ptr + q_idx * stride_s_n + d_idx * stride_s_d, score_acc)


# -----------------------------------------------------------------------------
# Backward: grad_Q — one program per (q_batch, q_token_abs)
# -----------------------------------------------------------------------------


@triton.jit
def _varlen_bwd_dQ_kernel(
    D_ptr,  # [sum_Ld, d]
    cu_q_ptr,  # [Nq+1]
    cu_d_ptr,  # [Nd+1]
    argmax_ptr,  # [Nq, Nd, max_lq]
    grad_s_ptr,  # [Nq, Nd] fp32
    grad_Q_ptr,  # [sum_Lq, d] fp32
    Nq: tl.constexpr,
    Nd: tl.constexpr,
    max_lq: tl.constexpr,
    d: tl.constexpr,
    d_pad: tl.constexpr,
    stride_d_t,
    stride_d_k,
    stride_am_n,
    stride_am_d,
    stride_am_l,
    stride_gs_n,
    stride_gs_d,
    stride_gq_t,
    stride_gq_k,
):
    pid = tl.program_id(0)
    q_idx = pid // max_lq
    s = pid % max_lq

    q_lo = tl.load(cu_q_ptr + q_idx).to(tl.int32)
    q_hi = tl.load(cu_q_ptr + q_idx + 1).to(tl.int32)
    lq = q_hi - q_lo

    if s >= lq:
        return

    k = tl.arange(0, d_pad)
    km = k < d
    acc = tl.zeros([d_pad], dtype=tl.float32)

    for j in range(0, Nd):
        d_lo = tl.load(cu_d_ptr + j).to(tl.int32)
        d_hi = tl.load(cu_d_ptr + j + 1).to(tl.int32)
        ld = d_hi - d_lo

        t = tl.load(argmax_ptr + q_idx * stride_am_n + j * stride_am_d + s * stride_am_l).to(tl.int32)
        # t == -1 on empty docs or invalid positions; skip those safely.
        valid = (t >= 0) & (ld > 0)
        if valid:
            gs = tl.load(grad_s_ptr + q_idx * stride_gs_n + j * stride_gs_d).to(tl.float32)
            v = tl.load(
                D_ptr + (d_lo + t) * stride_d_t + k * stride_d_k,
                mask=km,
                other=0.0,
            ).to(tl.float32)
            acc += gs * v

    tl.store(
        grad_Q_ptr + (q_lo + s) * stride_gq_t + k * stride_gq_k,
        acc,
        mask=km,
    )


# -----------------------------------------------------------------------------
# Backward: grad_D — atomic scatter. One program per (q_batch, d_batch).
# -----------------------------------------------------------------------------


@triton.jit
def _varlen_bwd_dD_kernel(
    Q_ptr,  # [sum_Lq, d]
    cu_q_ptr,
    cu_d_ptr,
    argmax_ptr,
    grad_s_ptr,
    grad_D_ptr,  # [sum_Ld, d] fp32
    Nd: tl.constexpr,
    max_lq: tl.constexpr,
    d: tl.constexpr,
    d_pad: tl.constexpr,
    stride_q_t,
    stride_q_k,
    stride_am_n,
    stride_am_d,
    stride_am_l,
    stride_gs_n,
    stride_gs_d,
    stride_gd_t,
    stride_gd_k,
):
    pid = tl.program_id(0)
    i = pid // Nd
    j = pid % Nd

    q_lo = tl.load(cu_q_ptr + i).to(tl.int32)
    q_hi = tl.load(cu_q_ptr + i + 1).to(tl.int32)
    lq = q_hi - q_lo
    d_lo = tl.load(cu_d_ptr + j).to(tl.int32)

    if lq == 0:
        return

    gs = tl.load(grad_s_ptr + i * stride_gs_n + j * stride_gs_d).to(tl.float32)

    k = tl.arange(0, d_pad)
    km = k < d

    for s in range(0, max_lq):
        if s < lq:
            t = tl.load(argmax_ptr + i * stride_am_n + j * stride_am_d + s * stride_am_l).to(tl.int32)
            if t >= 0:
                qv = tl.load(
                    Q_ptr + (q_lo + s) * stride_q_t + k * stride_q_k,
                    mask=km,
                    other=0.0,
                ).to(tl.float32)
                tl.atomic_add(
                    grad_D_ptr + (d_lo + t) * stride_gd_t + k * stride_gd_k,
                    gs * qv,
                    mask=km,
                )


# -----------------------------------------------------------------------------
# Python launcher + autograd wrapper
# -----------------------------------------------------------------------------


def _varlen_forward(
    Q_packed: torch.Tensor,
    D_packed: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_d: torch.Tensor,
    max_seqlen_q: int | None,
    max_seqlen_d: int | None,
    save_argmax: bool,
):
    if Q_packed.dim() != 2:
        raise ValueError(
            f"Q_packed must be 2-D [sum(Lq_i), d]; got shape {tuple(Q_packed.shape)} "
            f"(ndim={Q_packed.dim()}). For padded inputs use `maxsim(Q, D, ...)` instead."
        )
    if D_packed.dim() != 2:
        raise ValueError(
            f"D_packed must be 2-D [sum(Ld_j), d]; got shape {tuple(D_packed.shape)} "
            f"(ndim={D_packed.dim()}). For padded inputs use `maxsim(Q, D, ...)` instead."
        )
    d = Q_packed.shape[1]
    if D_packed.shape[1] != d:
        raise ValueError(
            f"Q_packed and D_packed must share the embedding dim; got "
            f"Q_packed.shape[1]={Q_packed.shape[1]} vs D_packed.shape[1]={D_packed.shape[1]}."
        )

    cu_seqlens_q = cu_seqlens_q.to(torch.int32).contiguous()
    cu_seqlens_d = cu_seqlens_d.to(torch.int32).contiguous()
    Nq = cu_seqlens_q.numel() - 1
    Nd = cu_seqlens_d.numel() - 1

    if max_seqlen_q is None:
        max_seqlen_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()) if Nq else 0
    if max_seqlen_d is None:
        max_seqlen_d = int((cu_seqlens_d[1:] - cu_seqlens_d[:-1]).max().item()) if Nd else 0

    d_pad = next_pow2(d)
    compute_dtype = pick_compute_dtype(Q_packed, D_packed)
    tl_dtype = tl.float16 if compute_dtype == torch.float16 else tl.bfloat16

    scores = torch.zeros(Nq, Nd, device=Q_packed.device, dtype=torch.float32)

    Q_packed = Q_packed.contiguous()
    D_packed = D_packed.contiguous()

    if save_argmax and max_seqlen_q > 0:
        argmax = torch.full((Nq, Nd, max_seqlen_q), -1, device=Q_packed.device, dtype=torch.int32)
        am_strides = (argmax.stride(0), argmax.stride(1), argmax.stride(2))
    else:
        argmax = torch.empty(1, device=Q_packed.device, dtype=torch.int32)
        am_strides = (0, 0, 0)

    _varlen_fwd_kernel[(Nq * Nd,)](
        Q_packed,
        D_packed,
        cu_seqlens_q,
        cu_seqlens_d,
        scores,
        argmax,
        Nq,
        Nd,
        max_seqlen_q,
        max_seqlen_d,
        d,
        d_pad,
        Q_packed.stride(0),
        Q_packed.stride(1),
        D_packed.stride(0),
        D_packed.stride(1),
        scores.stride(0),
        scores.stride(1),
        am_strides[0],
        am_strides[1],
        am_strides[2],
        max_seqlen_q,
        max_seqlen_d,  # Lq, Ld placeholders
        COMPUTE_DTYPE=tl_dtype,
        SAVE_ARGMAX=save_argmax,
    )
    return scores, (argmax if save_argmax else None), max_seqlen_q, max_seqlen_d


class _MaxSimVarlenFn(torch.autograd.Function):
    """Autograd-aware varlen MaxSim."""

    @staticmethod
    def forward(ctx, Q, D, cu_q, cu_d, max_q, max_d):
        scores, argmax, max_q, max_d = _varlen_forward(Q, D, cu_q, cu_d, max_q, max_d, save_argmax=True)
        ctx.save_for_backward(Q, D, cu_q.to(torch.int32), cu_d.to(torch.int32), argmax)
        ctx.max_q = max_q
        ctx.max_d = max_d
        return scores

    @staticmethod
    def backward(ctx, grad_scores):
        Q, D, cu_q, cu_d, argmax = ctx.saved_tensors
        max_q = ctx.max_q
        grad_scores = grad_scores.contiguous().to(torch.float32)

        _, d = Q.shape
        Nq = cu_q.numel() - 1
        Nd = cu_d.numel() - 1
        d_pad = next_pow2(d)

        grad_Q = torch.zeros_like(Q, dtype=torch.float32)
        grad_D = torch.zeros_like(D, dtype=torch.float32)

        if max_q > 0:
            _varlen_bwd_dQ_kernel[(Nq * max_q,)](
                D,
                cu_q,
                cu_d,
                argmax,
                grad_scores,
                grad_Q,
                Nq,
                Nd,
                max_q,
                d,
                d_pad,
                D.stride(0),
                D.stride(1),
                argmax.stride(0),
                argmax.stride(1),
                argmax.stride(2),
                grad_scores.stride(0),
                grad_scores.stride(1),
                grad_Q.stride(0),
                grad_Q.stride(1),
                num_warps=4,
                num_stages=2,
            )

            _varlen_bwd_dD_kernel[(Nq * Nd,)](
                Q,
                cu_q,
                cu_d,
                argmax,
                grad_scores,
                grad_D,
                Nd,
                max_q,
                d,
                d_pad,
                Q.stride(0),
                Q.stride(1),
                argmax.stride(0),
                argmax.stride(1),
                argmax.stride(2),
                grad_scores.stride(0),
                grad_scores.stride(1),
                grad_D.stride(0),
                grad_D.stride(1),
                num_warps=4,
                num_stages=2,
            )

        return (
            grad_Q.to(Q.dtype),
            grad_D.to(D.dtype),
            None,  # cu_q
            None,  # cu_d
            None,  # max_q
            None,  # max_d
        )


def maxsim_varlen(
    Q_packed: torch.Tensor,
    D_packed: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_d: torch.Tensor,
    max_seqlen_q: int | None = None,
    max_seqlen_d: int | None = None,
) -> torch.Tensor:
    """Run MaxSim on packed (no-padding) inputs. Autograd-aware.

    Args:
        Q_packed: ``[sum(Lq_i), d]`` fp16/bf16/fp32. Requires-grad honored.
        D_packed: ``[sum(Ld_j), d]``. Requires-grad honored.
        cu_seqlens_q: ``[Nq+1]`` int32, cumulative sums so seq ``i`` is
            ``Q_packed[cu[i] : cu[i+1]]``.
        cu_seqlens_d: ``[Nd+1]`` int32, same for docs.
        max_seqlen_q / max_seqlen_d: hints for tile counts. Computed from
            ``cu_seqlens`` if not passed.

    Returns:
        scores: ``[Nq, Nd]`` fp32.

    Notes:
        * When either input requires grad, the forward saves an argmax
          buffer of shape ``[Nq, Nd, max_seqlen_q]`` (int32). The fused
          backward then produces ``grad_Q`` and ``grad_D`` directly on the
          packed layout — no repad, no materialized ``[Nq, Nd, Lq, Ld]``.
        * For pure reranking, just pass tensors with ``requires_grad=False``
          — the argmax save is skipped automatically. The separate
          ``maxsim_varlen_inference`` alias is deprecated since 0.9.0.
    """
    if Q_packed.requires_grad or D_packed.requires_grad:
        return _MaxSimVarlenFn.apply(
            Q_packed,
            D_packed,
            cu_seqlens_q.to(torch.int32),
            cu_seqlens_d.to(torch.int32),
            max_seqlen_q,
            max_seqlen_d,
        )
    scores, _, _, _ = _varlen_forward(
        Q_packed, D_packed, cu_seqlens_q, cu_seqlens_d, max_seqlen_q, max_seqlen_d, False
    )
    return scores


def maxsim_varlen_inference(
    Q_packed: torch.Tensor,
    D_packed: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_d: torch.Tensor,
    max_seqlen_q: int | None = None,
    max_seqlen_d: int | None = None,
) -> torch.Tensor:
    """Deprecated alias for :func:`maxsim_varlen`.

    :func:`maxsim_varlen` already skips the argmax save when neither input
    requires gradients; this alias exists for backward compatibility only
    and emits a :class:`DeprecationWarning`. Scheduled for removal in a
    future release.
    """
    import warnings

    warnings.warn(
        "`maxsim_varlen_inference` is deprecated since 0.9.0 — use "
        "`maxsim_varlen(...)` directly. The argmax save is skipped "
        "automatically when neither input has `requires_grad=True`.",
        DeprecationWarning,
        stacklevel=2,
    )
    scores, _, _, _ = _varlen_forward(
        Q_packed, D_packed, cu_seqlens_q, cu_seqlens_d, max_seqlen_q, max_seqlen_d, False
    )
    return scores
