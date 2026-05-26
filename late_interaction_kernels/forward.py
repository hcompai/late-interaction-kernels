"""Fused MaxSim forward kernel (FlashAttention-style tiling).

One program per ``(q_batch, d_batch)``. ``[Nq · Nd · Lq · Ld]`` similarities
never hit HBM. Optional ``save_argmax`` writes a ``[Nq · Nd, Lq]`` int32
buffer used by the training backward paths. See ``docs/design.md`` for the
algorithm.
"""

import torch
import triton
import triton.language as tl

from late_interaction_kernels._autotune import autotune_kwargs, forward_configs, prune_forward
from late_interaction_kernels._utils import ensure_contiguous_last, next_pow2, pick_compute_dtype


@triton.autotune(
    configs=forward_configs(),
    # ``Ld`` stays out of the key: it only drives a runtime ``range(0, Ld,
    # BLOCK_D)`` loop, so keying on it would force one recompile + one
    # autotune sweep per distinct doc length. ``normalize`` is also
    # intentionally absent — it's a constexpr toggle for ~3 extra ops in the
    # inner Ld loop, doesn't shift register pressure enough to change the
    # winning ``(BLOCK_Q, BLOCK_D, num_warps, num_stages)`` config. Keeping
    # it would double the cache cardinality for zero perf win (verified on
    # H100, A100 sweeps).
    key=["Lq", "d_pad", "has_q_mask", "has_d_mask"],
    prune_configs_by={"early_config_prune": prune_forward},
    **autotune_kwargs(),
)
@triton.jit
def _maxsim_fwd_kernel(
    Q_ptr,
    D_ptr,
    q_mask_ptr,
    d_mask_ptr,
    scores_ptr,
    argmax_ptr,
    Nq: tl.constexpr,
    Nd: tl.constexpr,
    Lq: tl.constexpr,
    Ld,
    d: tl.constexpr,
    d_pad: tl.constexpr,
    stride_q_n,
    stride_q_l,
    stride_q_d,
    stride_d_n,
    stride_d_l,
    stride_d_d,
    stride_s_n,
    stride_s_d,
    stride_qm_n,
    stride_qm_l,
    stride_dm_n,
    stride_dm_l,
    stride_a_pair,
    stride_a_lq,
    has_q_mask: tl.constexpr,
    has_d_mask: tl.constexpr,
    save_argmax: tl.constexpr,
    normalize: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    q_idx = pid // Nd
    d_idx = pid % Nd

    k_off = tl.arange(0, d_pad)
    k_mask = k_off < d

    score_acc = tl.zeros([], dtype=tl.float32)

    # Outer loop over query-token tiles. `static_range` unrolls when Lq/BLOCK_Q
    # is small and known at compile time, which is usually the case here (Lq is
    # often 32 or 128).
    for q_start in tl.static_range(0, Lq, BLOCK_Q):
        q_off = q_start + tl.arange(0, BLOCK_Q)
        q_valid = q_off < Lq

        if has_q_mask:
            qm = tl.load(
                q_mask_ptr + q_idx * stride_qm_n + q_off * stride_qm_l,
                mask=q_valid,
                other=0,
            ).to(tl.int1)
            q_active = q_valid & qm
        else:
            q_active = q_valid

        Q_block_f32 = tl.load(
            Q_ptr + q_idx * stride_q_n + q_off[:, None] * stride_q_l + k_off[None, :] * stride_q_d,
            mask=q_valid[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        if normalize:
            # Per-row L2 normalize in registers. Matches F.normalize(..., p=2, dim=-1)
            # with the same eps=1e-12 clamp as PyLate's upstream calls.
            q_norm_sq = tl.sum(Q_block_f32 * Q_block_f32, axis=1)
            q_inv = 1.0 / tl.sqrt(tl.maximum(q_norm_sq, 1e-12))
            Q_block_f32 = Q_block_f32 * q_inv[:, None]
        Q_block = Q_block_f32.to(COMPUTE_DTYPE)

        m = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
        m_idx = tl.full([BLOCK_Q], 0, dtype=tl.int32)

        for d_start in range(0, Ld, BLOCK_D):
            d_off = d_start + tl.arange(0, BLOCK_D)
            d_valid = d_off < Ld

            if has_d_mask:
                dm = tl.load(
                    d_mask_ptr + d_idx * stride_dm_n + d_off * stride_dm_l,
                    mask=d_valid,
                    other=0,
                ).to(tl.int1)
                d_active = d_valid & dm
            else:
                d_active = d_valid

            D_block_f32 = tl.load(
                D_ptr + d_idx * stride_d_n + d_off[:, None] * stride_d_l + k_off[None, :] * stride_d_d,
                mask=d_valid[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            if normalize:
                d_norm_sq = tl.sum(D_block_f32 * D_block_f32, axis=1)
                d_inv = 1.0 / tl.sqrt(tl.maximum(d_norm_sq, 1e-12))
                D_block_f32 = D_block_f32 * d_inv[:, None]
            D_block = D_block_f32.to(COMPUTE_DTYPE)

            # tl.dot needs inputs with the same dtype; Q_block and D_block are
            # both COMPUTE_DTYPE. Accumulator is fp32.
            S = tl.dot(Q_block, tl.trans(D_block), out_dtype=tl.float32)
            S = tl.where(d_active[None, :], S, float("-inf"))

            tile_max = tl.max(S, axis=1)
            if save_argmax:
                tile_argmax = tl.argmax(S, axis=1).to(tl.int32) + d_start
                update = tile_max > m
                m_idx = tl.where(update, tile_argmax, m_idx)
            m = tl.maximum(m, tile_max)

        # Rows where the max stayed at -inf (e.g. no active doc tokens) must
        # contribute 0, not -inf.
        m_finite = m != float("-inf")
        m = tl.where(m_finite & q_active, m, 0.0)
        score_acc += tl.sum(m)

        if save_argmax:
            tl.store(
                argmax_ptr + pid * stride_a_pair + q_off * stride_a_lq,
                m_idx,
                mask=q_valid,
            )

    tl.store(scores_ptr + q_idx * stride_s_n + d_idx * stride_s_d, score_acc)


# Small-input bypass thresholds. Below these we sidestep autotune entirely
# and launch the kernel with a conservative fixed config that fits the SMEM
# budget of every GPU family we support (≤ 96 KiB for d ≤ 256, well under the
# 100 KiB Ampere-consumer floor). The motivation is the same as
# flash-maxsim's ``_maxsim_fwd_kernel_small`` (line 337-348 of their
# ``flash_maxsim.py``): inference / REPL / test calls hit autotune over and
# over for shapes where the autotune *cost* (5+ s) dominates the workload
# (sub-millisecond). For these the optimum-vs-fixed-config gap is single-digit
# percent — well worth trading away to skip the sweep.
_SMALL_BYPASS_NQND = 500  # max Nq * Nd
_SMALL_BYPASS_D = 256  # max embedding dim (SMEM ceiling)
_SMALL_BLOCK_Q = 32
_SMALL_BLOCK_D = 64
_SMALL_NUM_WARPS = 4
_SMALL_NUM_STAGES = 2


def _should_bypass_autotune(Nq: int, Nd: int, d: int, save_argmax: bool) -> bool:
    """Inference-side small-shape bypass. Only safe when we don't need the
    argmax tile (i.e. no backward) — that keeps the bypass kernel identical
    to the autotuned one except for the fixed block sizes.
    """
    return (not save_argmax) and (Nq * Nd <= _SMALL_BYPASS_NQND) and (d <= _SMALL_BYPASS_D)


def _run_forward(
    Q: torch.Tensor,
    D: torch.Tensor,
    q_mask: torch.Tensor | None,
    d_mask: torch.Tensor | None,
    save_argmax: bool,
    normalize: bool = False,
):
    """Launch the forward kernel. Q/D assumed 3-D, contiguous on last dim.

    Returns (scores [Nq, Nd], argmax [Nq*Nd, Lq] or None).
    """
    Nq, Lq, d = Q.shape
    Nd, Ld, _ = D.shape
    d_pad = next_pow2(d)
    compute_dtype = pick_compute_dtype(Q, D)
    tl_dtype = tl.float16 if compute_dtype == torch.float16 else tl.bfloat16

    scores = torch.empty(Nq, Nd, device=Q.device, dtype=torch.float32)
    argmax = torch.empty(Nq * Nd, Lq, device=Q.device, dtype=torch.int32) if save_argmax else None

    has_q_mask = q_mask is not None
    has_d_mask = d_mask is not None
    # Triton doesn't like None pointers; substitute something.
    q_mask_ptr = q_mask if has_q_mask else Q
    d_mask_ptr = d_mask if has_d_mask else D
    argmax_ptr = argmax if save_argmax else scores

    qm_strides = (q_mask.stride(0), q_mask.stride(1)) if has_q_mask else (0, 0)
    dm_strides = (d_mask.stride(0), d_mask.stride(1)) if has_d_mask else (0, 0)
    a_strides = (argmax.stride(0), argmax.stride(1)) if save_argmax else (0, 0)

    grid = (Nq * Nd,)
    args = (
        Q,
        D,
        q_mask_ptr,
        d_mask_ptr,
        scores,
        argmax_ptr,
        Nq,
        Nd,
        Lq,
        Ld,
        d,
        d_pad,
        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        D.stride(0),
        D.stride(1),
        D.stride(2),
        scores.stride(0),
        scores.stride(1),
        qm_strides[0],
        qm_strides[1],
        dm_strides[0],
        dm_strides[1],
        a_strides[0],
        a_strides[1],
        has_q_mask,
        has_d_mask,
        save_argmax,
        normalize,
    )
    if _should_bypass_autotune(Nq, Nd, d, save_argmax):
        # Bypass: launch the underlying JIT directly with fixed constexpr
        # block sizes and fixed launch attrs. ``_maxsim_fwd_kernel.fn`` is the
        # ``JITFunction`` wrapped by ``@triton.autotune`` — calling it via
        # ``[grid](...)`` skips the autotuner entirely, so no sweep, no
        # benchmark, no disk cache touch. Same kernel binary as the autotuned
        # path would compile for this (BLOCK_Q, BLOCK_D) tuple, so the
        # Triton compile cache is shared.
        _maxsim_fwd_kernel.fn[grid](
            *args,
            BLOCK_Q=_SMALL_BLOCK_Q,
            BLOCK_D=_SMALL_BLOCK_D,
            COMPUTE_DTYPE=tl_dtype,
            num_warps=_SMALL_NUM_WARPS,
            num_stages=_SMALL_NUM_STAGES,
        )
    else:
        _maxsim_fwd_kernel[grid](*args, COMPUTE_DTYPE=tl_dtype)
    return scores, argmax


def maxsim_forward(
    Q: torch.Tensor,
    D: torch.Tensor,
    q_mask: torch.Tensor | None = None,
    d_mask: torch.Tensor | None = None,
    *,
    save_argmax: bool = False,
    normalize: bool = False,
):
    """Fused MaxSim forward.

    Args:
        Q: ``[Nq, Lq, d]`` or ``[Lq, d]``.
        D: ``[Nd, Ld, d]`` or ``[Ld, d]``.
        q_mask, d_mask: optional bool masks (``True`` = keep).
        save_argmax: if ``True``, also return the winning doc-token index
            per ``(q_batch, d_batch, q_token)``.
        normalize: L2-normalize Q and D per-token inside the kernel.

    Returns:
        scores: ``[Nq, Nd]`` fp32 (scalar if both inputs were 2-D).
        argmax: ``[Nq*Nd, Lq]`` int32 or ``None``.
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

    Q = ensure_contiguous_last(Q)
    D = ensure_contiguous_last(D)
    if q_mask is not None:
        q_mask = q_mask.contiguous().to(torch.int8)
    if d_mask is not None:
        d_mask = d_mask.contiguous().to(torch.int8)

    scores, argmax = _run_forward(Q, D, q_mask, d_mask, save_argmax, normalize)

    if q_was_2d and d_was_2d:
        return scores.reshape(()), argmax
    if q_was_2d:
        return scores.squeeze(0), argmax
    if d_was_2d:
        return scores.squeeze(-1), argmax
    return scores, argmax
