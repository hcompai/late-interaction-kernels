"""Low-memory backward: bf16 grads, no full-size fp32 buffers, no atomics.

The other backends accumulate grads into full-size fp32 buffers and cast to
the input dtype at the end, so the cast briefly holds both copies. For the 4-D
hard-negative layout grad_D is n_neg-inflated, making that fp32 buffer + its
transient the largest training allocation.

This path accumulates in an fp32 register inside destination-owned kernels and
stores the input dtype on the single write — same result as ``fp32 →
.to(dtype)`` but with no fp32 buffer, no transient, and deterministic.

* grad_Q: the row-owned ``_bwd_dQ_kernel`` pointed at a bf16 buffer.
* grad_D: one destination-owned kernel for both layouts — each
  ``(slab, doc-tile)`` reduces ``onehot^T @ (g·Q)`` from the saved argmax.
"""

import torch

try:
    import triton
    import triton.language as tl

    from late_interaction_kernels._autotune import autotune_kwargs, forward_configs, prune_forward
    from late_interaction_kernels.backward._autotune import BWD_CONFIGS, BWD_KEY

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

from late_interaction_kernels._utils import autotune_placeholder, next_pow2, pick_compute_dtype

if _HAS_TRITON:
    # grad_Q: one program per (q_batch, q_token), gathers the winning D row per
    # contributing doc. Row-owned full-coverage store — no atomics.
    @triton.autotune(configs=BWD_CONFIGS, key=BWD_KEY)
    @triton.jit
    def _bwd_dQ_kernel(
        D_ptr,
        argmax_ptr,
        grad_s_ptr,
        q_mask_ptr,
        grad_Q_ptr,
        Nq: tl.constexpr,
        Nd: tl.constexpr,  # K_per_query in KD/pairs mode
        Lq: tl.constexpr,
        Ld,
        d: tl.constexpr,
        d_pad: tl.constexpr,
        stride_d_n,
        stride_d_l,
        stride_d_k,
        stride_gs_n,
        stride_gs_d,
        stride_qm_n,
        stride_qm_l,
        stride_gq_n,
        stride_gq_l,
        stride_gq_k,
        stride_a_pair,
        stride_a_lq,
        has_q_mask: tl.constexpr,
        kd_layout: tl.constexpr,
    ):
        pid = tl.program_id(0)
        q_idx = pid // Lq
        s = pid % Lq

        emb_off = tl.arange(0, d_pad)
        emb_mask = emb_off < d
        acc = tl.zeros([d_pad], dtype=tl.float32)

        q_active = True
        if has_q_mask:
            q_mask_val = tl.load(q_mask_ptr + q_idx * stride_qm_n + s * stride_qm_l).to(tl.int1)
            q_active = q_mask_val != 0

        if q_active:
            for d_idx in range(0, Nd):
                gs = tl.load(grad_s_ptr + q_idx * stride_gs_n + d_idx * stride_gs_d).to(tl.float32)
                t = tl.load(argmax_ptr + (q_idx * Nd + d_idx) * stride_a_pair + s * stride_a_lq)
                t = t.to(tl.int32)
                d_global = q_idx * Nd + d_idx if kd_layout else d_idx
                # `t == -1` sentinel: forward had no active doc for this (q, j, s).
                if t >= 0:
                    dv = tl.load(
                        D_ptr + d_global * stride_d_n + t * stride_d_l + emb_off * stride_d_k,
                        mask=emb_mask,
                        other=0.0,
                    ).to(tl.float32)
                    acc += gs * dv

        tl.store(
            grad_Q_ptr + q_idx * stride_gq_n + s * stride_gq_l + emb_off * stride_gq_k,
            acc,
            mask=emb_mask,
        )

    @triton.autotune(
        configs=forward_configs(),
        key=["Lq", "d_pad", "cross"],
        prune_configs_by={"early_config_prune": prune_forward},
        **autotune_kwargs(),
    )
    @triton.jit
    def _bwd_dD_owned_kernel(
        Q_ptr,
        argmax_ptr,
        grad_s_ptr,
        q_mask_ptr,
        grad_D_ptr,
        Nq: tl.constexpr,
        Nd: tl.constexpr,  # cross: doc count; KD: K per query
        Lq: tl.constexpr,
        Ld,
        d: tl.constexpr,
        d_pad: tl.constexpr,
        stride_q_n,
        stride_q_l,
        stride_q_k,
        stride_gs_n,
        stride_gs_d,
        stride_qm_n,
        stride_qm_l,
        stride_gd_n,
        stride_gd_l,
        stride_gd_k,
        stride_a_pair,
        stride_a_lq,
        has_q_mask: tl.constexpr,
        cross: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_D: tl.constexpr,
        COMPUTE_DTYPE: tl.constexpr,
    ):
        n_dtiles = tl.cdiv(Ld, BLOCK_D)
        pid = tl.program_id(0)
        slab = pid // n_dtiles  # = d_global (the grad_D row block owner)
        dt = pid % n_dtiles
        d_start = dt * BLOCK_D
        d_off = d_start + tl.arange(0, BLOCK_D)
        d_valid = d_off < Ld

        emb_off = tl.arange(0, d_pad)
        emb_mask = emb_off < d

        # Which queries contribute to this grad_D slab, and the grad_s column.
        #   cross: slab == j; every query i scatters into D[j]  → loop i.
        #   KD:    slab == i*Nd + j; only query i = slab // Nd  → single i.
        if cross:
            j = slab
            i_lo = 0
            i_hi = Nq
        else:
            j = slab % Nd
            i_lo = slab // Nd
            i_hi = i_lo + 1

        acc = tl.zeros([BLOCK_D, d_pad], dtype=tl.float32)

        for i in range(i_lo, i_hi):
            gs = tl.load(grad_s_ptr + i * stride_gs_n + j * stride_gs_d).to(tl.float32)
            a_row = i * Nd + j  # argmax row index (matches forward layout)

            for q_start in tl.static_range(0, Lq, BLOCK_Q):
                q_off = q_start + tl.arange(0, BLOCK_Q)
                q_valid = q_off < Lq

                a = tl.load(
                    argmax_ptr + a_row * stride_a_pair + q_off * stride_a_lq,
                    mask=q_valid,
                    other=-1,
                ).to(tl.int32)

                active = q_valid & (a >= 0)
                if has_q_mask:
                    qm = tl.load(
                        q_mask_ptr + i * stride_qm_n + q_off * stride_qm_l,
                        mask=q_valid,
                        other=0,
                    ).to(tl.int1)
                    active = active & qm

                # one-hot[s, t] = (winner(s) == d_off[t]) & active(s)
                onehot = (a[:, None] == d_off[None, :]) & active[:, None]
                W = tl.where(onehot, gs, 0.0).to(COMPUTE_DTYPE)  # [BLOCK_Q, BLOCK_D]

                Qb = tl.load(
                    Q_ptr + i * stride_q_n + q_off[:, None] * stride_q_l + emb_off[None, :] * stride_q_k,
                    mask=q_valid[:, None] & emb_mask[None, :],
                    other=0.0,
                ).to(COMPUTE_DTYPE)

                # grad_D_tile[t, :] += Σ_s onehot[s,t]·gs·Q[i,s,:]
                acc += tl.dot(tl.trans(W), Qb, out_dtype=tl.float32)

        tl.store(
            grad_D_ptr + slab * stride_gd_n + d_off[:, None] * stride_gd_l + emb_off[None, :] * stride_gd_k,
            acc,  # fp32 acc cast to grad_D's (bf16/fp16) dtype on store
            mask=d_valid[:, None] & emb_mask[None, :],
        )


def maxsim_backward_lowmem(
    grad_scores: torch.Tensor,
    Q: torch.Tensor,
    D: torch.Tensor,
    argmax: torch.Tensor,
    q_mask: torch.Tensor | None = None,
    *,
    kd_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward producing bf16/fp16 grads directly (no fp32 buffers, no atomics).

    Same gradients as the unified backend to bf16 rounding; deterministic.
    The ``grad_D`` one-hot reduction runs at ``pick_compute_dtype`` matmul
    precision, so fp32 inputs get fp16-precision ``grad_D`` (``grad_Q`` is a
    gather and stays fp32-accurate); use ``unified`` for fp32 accumulation.

    Args:
        grad_scores: ``[Nq, Nd_eff]`` upstream gradient.
        Q: ``[Nq, Lq, d]``.
        D: ``[Nd, Ld, d]`` cross-product, or ``[Nq*K, Ld, d]`` KD/pairs.
        argmax: ``[Nq*Nd_eff, Lq]`` int32 saved by the forward.
        q_mask: optional ``[Nq, Lq]`` int8/bool.
        kd_layout: per-query slab indexing.

    Returns ``(grad_Q, grad_D)`` already in ``Q`` / ``D`` dtypes.
    """
    Nq, Lq, d = Q.shape
    Nd_total, Ld, _ = D.shape
    Nd = Nd_total // Nq if kd_layout else Nd_total
    d_pad = next_pow2(d)

    has_q_mask = q_mask is not None
    # int8 placeholder (not Q) so present-vs-absent q_mask doesn't split the
    # autotune cache via Triton's dtype-keying. See _utils.autotune_placeholder.
    qm_ptr = q_mask if has_q_mask else autotune_placeholder(Q, torch.int8)
    qm_strides = (q_mask.stride(0), q_mask.stride(1)) if has_q_mask else (0, 0)

    # --- grad_Q: row-owned kernel, bf16 output (fp32 accumulate in-kernel) ---
    grad_Q = torch.zeros(Nq, Lq, d, device=Q.device, dtype=Q.dtype)
    _bwd_dQ_kernel[(Nq * Lq,)](
        D,
        argmax,
        grad_scores,
        qm_ptr,
        grad_Q,
        Nq,
        Nd,
        Lq,
        Ld,
        d,
        d_pad,
        D.stride(0),
        D.stride(1),
        D.stride(2),
        grad_scores.stride(0),
        grad_scores.stride(1),
        qm_strides[0],
        qm_strides[1],
        grad_Q.stride(0),
        grad_Q.stride(1),
        grad_Q.stride(2),
        argmax.stride(0),
        argmax.stride(1),
        has_q_mask,
        kd_layout,
    )

    # --- grad_D: one destination-owned matmul kernel, bf16 output ---
    # Cross and KD share it; cross loops all queries per slab, KD owns one.
    grad_D = torch.zeros(Nd_total, Ld, d, device=D.device, dtype=D.dtype)
    compute_dtype = pick_compute_dtype(Q, D)
    tl_dtype = tl.float16 if compute_dtype == torch.float16 else tl.bfloat16

    def grid(meta):
        return (Nd_total * triton.cdiv(Ld, meta["BLOCK_D"]),)

    _bwd_dD_owned_kernel[grid](
        Q,
        argmax,
        grad_scores,
        qm_ptr,
        grad_D,
        Nq,
        Nd,
        Lq,
        Ld,
        d,
        d_pad,
        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        grad_scores.stride(0),
        grad_scores.stride(1),
        qm_strides[0],
        qm_strides[1],
        grad_D.stride(0),
        grad_D.stride(1),
        grad_D.stride(2),
        argmax.stride(0),
        argmax.stride(1),
        has_q_mask,
        not kd_layout,  # cross
        COMPUTE_DTYPE=tl_dtype,
    )

    return grad_Q, grad_D
