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
    # ``kd_layout`` *is* in the key because the two modes hit very different
    # D-side cache patterns (in-batch reuses D across queries; KD/pairs
    # don't), and the best (BLOCK_Q, BLOCK_D, num_warps) trade-off shifts.
    key=["Lq", "d_pad", "has_q_mask", "has_d_mask", "kd_layout"],
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
    Nd: tl.constexpr,  # cross-product: doc count; KD/pairs: K per query
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
    kd_layout: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    q_idx = pid // Nd
    d_idx = pid - q_idx * Nd
    # Cross-product: every query scores the same Nd docs → d_global = d_idx
    # KD / pairs:    each query owns its slab of Nd docs in a flattened
    #                D[Nq * Nd, Ld, d] view → d_global = pid (= q_idx*Nd + d_idx).
    # ``d_mask`` follows the same indexing as ``D``.
    if kd_layout:
        d_global = pid
    else:
        d_global = d_idx

    emb_off = tl.arange(0, d_pad)
    emb_mask = emb_off < d

    score_acc = tl.zeros([], dtype=tl.float32)

    # Outer loop over query-token tiles. `static_range` unrolls when Lq/BLOCK_Q
    # is small and known at compile time, which is usually the case here (Lq is
    # often 32 or 128).
    for q_start in tl.static_range(0, Lq, BLOCK_Q):
        q_off = q_start + tl.arange(0, BLOCK_Q)
        q_valid = q_off < Lq

        if has_q_mask:
            q_mask_val = tl.load(
                q_mask_ptr + q_idx * stride_qm_n + q_off * stride_qm_l,
                mask=q_valid,
                other=0,
            ).to(tl.int1)
            q_active = q_valid & q_mask_val
        else:
            q_active = q_valid

        Q_block_f32 = tl.load(
            Q_ptr + q_idx * stride_q_n + q_off[:, None] * stride_q_l + emb_off[None, :] * stride_q_d,
            mask=q_valid[:, None] & emb_mask[None, :],
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
        # -1 sentinel so a fully-d-masked (i, j, s) row keeps a unique value
        # in the saved argmax buffer; the backward gates loads + atomic-adds
        # on `t >= 0` to skip those rows instead of writing a spurious
        # contribution into `grad_D[d_global, 0, :]`.
        m_idx = tl.full([BLOCK_Q], -1, dtype=tl.int32)

        for d_start in range(0, Ld, BLOCK_D):
            d_off = d_start + tl.arange(0, BLOCK_D)
            d_valid = d_off < Ld

            if has_d_mask:
                d_mask_val = tl.load(
                    d_mask_ptr + d_global * stride_dm_n + d_off * stride_dm_l,
                    mask=d_valid,
                    other=0,
                ).to(tl.int1)
                d_active = d_valid & d_mask_val
            else:
                d_active = d_valid

            D_block_f32 = tl.load(
                D_ptr + d_global * stride_d_n + d_off[:, None] * stride_d_l + emb_off[None, :] * stride_d_d,
                mask=d_valid[:, None] & emb_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            if normalize:
                d_norm_sq = tl.sum(D_block_f32 * D_block_f32, axis=1)
                d_inv = 1.0 / tl.sqrt(tl.maximum(d_norm_sq, 1e-12))
                D_block_f32 = D_block_f32 * d_inv[:, None]
            D_block = D_block_f32.to(COMPUTE_DTYPE)

            # Q_block and D_block share COMPUTE_DTYPE (tl.dot requires matching
            # operand dtypes); accumulate in fp32.
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

    # Output is always [Nq, Nd] (Nd = K_per_query in KD/pairs mode).
    tl.store(scores_ptr + q_idx * stride_s_n + d_idx * stride_s_d, score_acc)


# Small-input bypass: for tiny shapes (inference / REPL / tests) the autotune
# sweep (~5 s) dominates the actual work (sub-ms), and the optimum-vs-fixed
# gap is single-digit percent. The fixed config below fits ≤ 96 KiB SMEM so
# it lands on every supported GPU family.
_SMALL_BYPASS_NQND = 500  # max Nq * Nd  (grid size)
_SMALL_BYPASS_LQLD = 200_000  # max Lq * Ld (per-program work)
_SMALL_BYPASS_D = 256  # max embedding dim (SMEM ceiling)
_SMALL_BLOCK_Q = 32
_SMALL_BLOCK_D = 64
_SMALL_NUM_WARPS = 4
_SMALL_NUM_STAGES = 2


def _should_bypass_autotune(Nq: int, Nd: int, Lq: int, Ld: int, d: int, save_argmax: bool) -> bool:
    """Inference-side small-shape bypass. Only safe when we don't need the
    argmax tile (i.e. no backward) — that keeps the bypass kernel identical
    to the autotuned one except for the fixed block sizes.

    Both gates have to hold:

    * ``Nq * Nd <= 500`` — small grid, otherwise the autotuner amortizes its
      sweep across enough programs that the fixed config can't catch up.
    * ``Lq * Ld <= 200_000`` — small per-program work, otherwise the shape
      is compute-bound and the fixed ``(BLOCK_Q=32, BLOCK_D=64, warps=4)``
      tile shape loses meaningfully to the Hopper compute-bound winner
      ``(128, 128, warps=8)``. ColPali Lq=Ld=1024 (1M work) is the
      canonical example: bypass cost ~2.4× over autotuned on that shape.

    ``d <= 256`` is the SMEM ceiling for the fixed config — past that the
    operand buffers don't fit.
    """
    return (
        (not save_argmax)
        and (Nq * Nd <= _SMALL_BYPASS_NQND)
        and (Lq * Ld <= _SMALL_BYPASS_LQLD)
        and (d <= _SMALL_BYPASS_D)
    )


def _run_forward(
    Q: torch.Tensor,
    D: torch.Tensor,
    q_mask: torch.Tensor | None,
    d_mask: torch.Tensor | None,
    save_argmax: bool,
    normalize: bool = False,
    *,
    kd_layout: bool = False,
):
    """Launch the forward kernel. Q/D assumed 3-D, contiguous on last dim.

    Args:
        Q: ``[Nq, Lq, d]``.
        D: ``[Nd, Ld, d]`` for the cross-product layout, or
            ``[Nq * K, Ld, d]`` for the KD/pairs layout (in which case the
            caller passes ``kd_layout=True`` and the kernel uses
            ``d_global = pid`` so each query reads its own ``K``-slab).
        q_mask: ``[Nq, Lq]`` int8 or ``None``.
        d_mask: ``[Nd, Ld]`` or ``[Nq*K, Ld]`` int8 (matches ``D``'s first dim).
        save_argmax: write the ``[Nq*Nd_eff, Lq]`` winner buffer.
        normalize: per-token L2 normalize inside the kernel.
        kd_layout: switches from cross-product to per-query slab indexing.

    Returns ``(scores [Nq, Nd_eff], argmax [Nq*Nd_eff, Lq] or None)`` where
    ``Nd_eff`` is ``Nd`` for cross-product or ``K`` for KD/pairs.
    """
    Nq, Lq, d = Q.shape
    if kd_layout:
        # D is the flat [Nq*K, Ld, d] view; recover K from D.shape[0] // Nq.
        Nd_total = D.shape[0]
        if Nd_total % Nq != 0:
            raise ValueError(
                f"kd_layout=True requires D.shape[0] (={Nd_total}) to be a multiple of Nq (={Nq})."
            )
        K = Nd_total // Nq
        Ld = D.shape[1]
        Nd_eff = K  # candidates per query
    else:
        Nd_eff = D.shape[0]  # total doc count in the cross-product batch
        Ld = D.shape[1]
    d_pad = next_pow2(d)
    compute_dtype = pick_compute_dtype(Q, D)
    tl_dtype = tl.float16 if compute_dtype == torch.float16 else tl.bfloat16

    scores = torch.empty(Nq, Nd_eff, device=Q.device, dtype=torch.float32)
    argmax = torch.empty(Nq * Nd_eff, Lq, device=Q.device, dtype=torch.int32) if save_argmax else None

    has_q_mask = q_mask is not None
    has_d_mask = d_mask is not None
    # Triton rejects None pointers; pass a live tensor the kernel won't read.
    q_mask_ptr = q_mask if has_q_mask else Q
    d_mask_ptr = d_mask if has_d_mask else D
    argmax_ptr = argmax if save_argmax else scores

    qm_strides = (q_mask.stride(0), q_mask.stride(1)) if has_q_mask else (0, 0)
    dm_strides = (d_mask.stride(0), d_mask.stride(1)) if has_d_mask else (0, 0)
    a_strides = (argmax.stride(0), argmax.stride(1)) if save_argmax else (0, 0)

    grid = (Nq * Nd_eff,)
    args = (
        Q,
        D,
        q_mask_ptr,
        d_mask_ptr,
        scores,
        argmax_ptr,
        Nq,
        Nd_eff,
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
        kd_layout,
    )
    # Bypass eligibility uses ``Nd_eff`` (= Nd for in-batch, K for KD/pairs)
    # so the per-program work estimate matches what the kernel will actually
    # do. KD/pairs of size (Nq=4, K=16) lands in the bypass band just like
    # in-batch (Nq=4, Nd=16) — the launch budget is the same.
    if _should_bypass_autotune(Nq, Nd_eff, Lq, Ld, d, save_argmax):
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
