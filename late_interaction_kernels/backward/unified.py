"""Single-pass fused ``grad_Q`` + ``grad_D`` kernel (FA-2 style).

Hoists ``Q[i, s, :]`` out of the doc-batch loop, roughly halving HBM
read traffic versus a two-pass backward (separate ``grad_Q`` /
``grad_D`` kernels). Row-owned ``grad_Q`` accumulation (no atomic) +
``tl.atomic_add`` for ``grad_D``. Default ``"auto"`` / ``"unified"``
backward.

See :doc:`../docs/design.md` for the full HBM-traffic derivation and
numerical contract.
"""

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

from late_interaction_kernels._utils import autotune_placeholder, next_pow2
from late_interaction_kernels.backward._autotune import BWD_CONFIGS, BWD_KEY


def maxsim_backward_unified_reference(
    grad_scores: torch.Tensor,
    Q: torch.Tensor,
    D: torch.Tensor,
    argmax: torch.Tensor,
    q_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for the unified backward.

    Shapes match the existing forward:
        grad_scores: [Nq, Nd] fp32
        Q:           [Nq, Lq, d]
        D:           [Nd, Ld, d]
        argmax:      [Nq*Nd, Lq] int32 — winning doc-token index per (i, j, q)
        q_mask:      [Nq, Lq] bool or None

    Returns (grad_Q, grad_D) in the dtypes of Q and D.

    This is the same math as the unified Triton kernel, expressed
    without any kernel fusion. It exists to validate the Triton kernel
    to fp32 tolerance.
    """
    Nq, Lq, d = Q.shape
    Nd, Ld, _ = D.shape
    g = grad_scores.to(torch.float32)  # [Nq, Nd]
    Qf = Q.to(torch.float32)
    Df = D.to(torch.float32)

    m_idx = argmax.view(Nq, Nd, Lq).long()  # [Nq, Nd, Lq]
    # D_win[i, j, q] = D[j, argmax[i, j, q]]
    j_idx = torch.arange(Nd, device=D.device).view(1, Nd, 1).expand(Nq, Nd, Lq)
    D_win = Df[j_idx, m_idx]  # [Nq, Nd, Lq, d]

    if q_mask is not None:
        m = q_mask.to(torch.bool).view(Nq, 1, Lq, 1)  # broadcast mask
    else:
        m = None

    # grad_Q[i, q, :] = sum_j grad_scores[i, j] * D_win[i, j, q, :]
    contrib_q = g.view(Nq, Nd, 1, 1) * D_win  # [Nq, Nd, Lq, d]
    if m is not None:
        contrib_q = contrib_q * m.to(contrib_q.dtype)
    grad_Q = contrib_q.sum(dim=1)  # [Nq, Lq, d]

    # grad_D: scatter contributions into D_win slots.
    #   contrib_d[i, j, q, :] = grad_scores[i, j] * Q[i, q, :]
    contrib_d = g.view(Nq, Nd, 1, 1) * Qf.view(Nq, 1, Lq, d)
    if m is not None:
        contrib_d = contrib_d * m.to(contrib_d.dtype)

    grad_D = torch.zeros(Nd, Ld, d, device=D.device, dtype=torch.float32)
    # Flatten (i, j, q) → assemble indices and index_add_ per-j to avoid
    # a giant scatter. On GPU this reference is slow-but-correct.
    for d_idx in range(Nd):
        m_idx_j = m_idx[:, d_idx, :]  # [Nq, Lq]
        contrib_slice = contrib_d[:, d_idx, :, :]  # [Nq, Lq, d]
        grad_D[d_idx].index_add_(0, m_idx_j.reshape(-1), contrib_slice.reshape(-1, d))

    return grad_Q.to(Q.dtype), grad_D.to(D.dtype)


if _HAS_TRITON:
    # The unified kernel owns a full grad_Q row per program (one (i, s)
    # pair), so no atomics on grad_Q. Inside, it loops over j and:
    #   1) accumulates grad_Q  += grad_scores[i, j] * D[j, argmax[i, j, s]]
    #   2) atomic_add  grad_D[j, argmax[i, j, s]] += grad_scores[i, j] * Q[i, s]
    #
    # The key optimisation vs a two-pass backward (separate grad_Q / grad_D
    # kernels): Q[i, s, :] is hoisted out of the j loop; a standalone dD
    # kernel would reload it Nd times.

    # grad_D accumulates with atomic_add, so it must be re-zeroed before each
    # autotune trial — otherwise trials pile onto each other's results.
    @triton.autotune(configs=BWD_CONFIGS, key=BWD_KEY, reset_to_zero=["grad_D_ptr"])
    @triton.jit
    def _bwd_unified_kernel(
        Q_ptr,
        D_ptr,
        argmax_ptr,
        grad_s_ptr,
        q_mask_ptr,
        grad_Q_ptr,
        grad_D_ptr,
        Nq: tl.constexpr,
        Nd: tl.constexpr,  # K_per_query in KD/pairs mode
        Lq: tl.constexpr,
        Ld,
        d: tl.constexpr,
        d_pad: tl.constexpr,
        stride_q_n,
        stride_q_l,
        stride_q_k,
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
        stride_gd_n,
        stride_gd_l,
        stride_gd_k,
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

        q_active = True
        if has_q_mask:
            q_mask_val = tl.load(q_mask_ptr + q_idx * stride_qm_n + s * stride_qm_l).to(tl.int1)
            q_active = q_mask_val != 0

        # Always zero the grad_Q row (masked rows must produce zeros, not
        # leftover garbage from a previous launch).
        if not q_active:
            tl.store(
                grad_Q_ptr + q_idx * stride_gq_n + s * stride_gq_l + emb_off * stride_gq_k,
                tl.zeros([d_pad], dtype=tl.float32),
                mask=emb_mask,
            )
            return

        # Hoist Q[i, s, :] out of the j loop — amortizes across all j's
        # atomic_add into grad_D. This is the single biggest HBM win
        # vs a two-pass backward.
        qv = tl.load(
            Q_ptr + q_idx * stride_q_n + s * stride_q_l + emb_off * stride_q_k,
            mask=emb_mask,
            other=0.0,
        ).to(tl.float32)

        acc_Q = tl.zeros([d_pad], dtype=tl.float32)

        for d_idx in range(0, Nd):
            gs = tl.load(grad_s_ptr + q_idx * stride_gs_n + d_idx * stride_gs_d).to(tl.float32)
            t = tl.load(argmax_ptr + (q_idx * Nd + d_idx) * stride_a_pair + s * stride_a_lq).to(tl.int32)
            # Cross-product: all queries share D[j, :]. KD/pairs: query i
            # owns its slab so the global doc index is i*Nd + j (= the
            # same value that produced this argmax row in the forward).
            if kd_layout:
                d_global = q_idx * Nd + d_idx
            else:
                d_global = d_idx
            # `t == -1` sentinel: forward saw no active doc for this (i, j, s)
            # row — skip the load + atomic-add so we don't poison
            # `grad_D[d_global, 0, :]`.
            if t >= 0:
                dv = tl.load(
                    D_ptr + d_global * stride_d_n + t * stride_d_l + emb_off * stride_d_k,
                    mask=emb_mask,
                    other=0.0,
                ).to(tl.float32)

                acc_Q += gs * dv

                tl.atomic_add(
                    grad_D_ptr + d_global * stride_gd_n + t * stride_gd_l + emb_off * stride_gd_k,
                    gs * qv,
                    mask=emb_mask,
                )

        tl.store(
            grad_Q_ptr + q_idx * stride_gq_n + s * stride_gq_l + emb_off * stride_gq_k,
            acc_Q,
            mask=emb_mask,
        )


def maxsim_backward_unified(
    grad_scores: torch.Tensor,
    Q: torch.Tensor,
    D: torch.Tensor,
    argmax: torch.Tensor,
    q_mask: torch.Tensor | None = None,
    *,
    kd_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-pass ``grad_Q`` + ``grad_D`` backward (fp32 atomic scatter).

    Hoists ``Q[i, s, :]`` out of the doc-batch loop, roughly halving HBM read
    traffic. ``grad_D`` accumulates in a full-size fp32 buffer via
    ``tl.atomic_add`` and is cast to the input dtype at the end. For the
    memory-optimal, deterministic variant see :func:`maxsim_backward_lowmem`.

    Args:
        grad_scores: ``[Nq, Nd]`` fp32 upstream gradient.
        Q: ``[Nq, Lq, d]``.
        D: ``[Nd, Ld, d]`` cross-product, or ``[Nq*K, Ld, d]`` KD/pairs.
        argmax: ``[Nq*Nd, Lq]`` int32 — the winner buffer from the forward.
        q_mask: optional ``[Nq, Lq]`` mask. Pass it as ``int8`` to share the
            autotune entry with the mask-absent path (the autotuner keys on the
            arg dtype, so a ``bool`` mask gets its own entry/sweep). Internal
            callers already convert to ``int8`` before ``save_for_backward``.

    Returns:
        ``(grad_Q, grad_D)`` cast back to the dtypes of ``Q`` and ``D``.
    """
    if not _HAS_TRITON:  # pragma: no cover
        raise RuntimeError("maxsim_backward_unified requires Triton; install a CUDA-enabled Triton build.")
    if not (Q.is_cuda and D.is_cuda):
        raise RuntimeError("maxsim_backward_unified requires CUDA tensors.")

    Nq, Lq, d = Q.shape
    Nd_total, Ld, _ = D.shape  # in KD/pairs mode this is Nq*K (flat view)
    K = Nd_total // Nq if kd_layout else Nd_total
    d_pad = next_pow2(d)

    grad_Q = torch.empty(Nq, Lq, d, device=Q.device, dtype=torch.float32)
    grad_D = torch.zeros(Nd_total, Ld, d, device=D.device, dtype=torch.float32)

    has_q_mask = q_mask is not None
    # int8 placeholder (not Q) so present-vs-absent q_mask doesn't split the
    # autotune cache via Triton's dtype-keying. See _utils.autotune_placeholder.
    qm_ptr = q_mask if has_q_mask else autotune_placeholder(Q, torch.int8)
    qm_strides = (q_mask.stride(0), q_mask.stride(1)) if has_q_mask else (0, 0)

    _bwd_unified_kernel[(Nq * Lq,)](
        Q,
        D,
        argmax,
        grad_scores,
        qm_ptr,
        grad_Q,
        grad_D,
        Nq,
        K,
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
        grad_scores.stride(0),
        grad_scores.stride(1),
        qm_strides[0],
        qm_strides[1],
        grad_Q.stride(0),
        grad_Q.stride(1),
        grad_Q.stride(2),
        grad_D.stride(0),
        grad_D.stride(1),
        grad_D.stride(2),
        argmax.stride(0),
        argmax.stride(1),
        has_q_mask,
        kd_layout,
    )

    return grad_Q.to(Q.dtype), grad_D.to(D.dtype)
