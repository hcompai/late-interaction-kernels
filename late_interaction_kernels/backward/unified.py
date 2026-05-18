"""Single-pass fused ``grad_Q`` + ``grad_D`` kernel (FA-2 style).

Hoists ``Q[i, s, :]`` out of the doc-batch loop, roughly halving HBM
read traffic versus the two-pass backward. Row-owned ``grad_Q``
accumulation (no atomic) + ``tl.atomic_add`` for ``grad_D``. Default
``"auto"`` / ``"unified"`` backward.

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

from late_interaction_kernels._utils import next_pow2


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

    This is the same math as the two-pass Triton backward, expressed
    without any kernel fusion. It exists to let us validate the
    upcoming unified Triton kernel to fp32 tolerance before shipping it.
    """
    Nq, Lq, d = Q.shape
    Nd, Ld, _ = D.shape
    g = grad_scores.to(torch.float32)  # [Nq, Nd]
    Qf = Q.to(torch.float32)
    Df = D.to(torch.float32)

    am = argmax.view(Nq, Nd, Lq).long()  # [Nq, Nd, Lq]
    # Gather winning D rows:  D_win[i, j, q] = D[j, argmax[i, j, q]]
    j_idx = torch.arange(Nd, device=D.device).view(1, Nd, 1).expand(Nq, Nd, Lq)
    D_win = Df[j_idx, am]  # [Nq, Nd, Lq, d]

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
    for j in range(Nd):
        am_j = am[:, j, :]  # [Nq, Lq]
        cont = contrib_d[:, j, :, :]  # [Nq, Lq, d]
        # grad_D[j].index_add_(0, am_j.flatten(), cont.reshape(-1, d))
        grad_D[j].index_add_(0, am_j.reshape(-1), cont.reshape(-1, d))

    return grad_Q.to(Q.dtype), grad_D.to(D.dtype)


if _HAS_TRITON:
    # The unified kernel owns a full grad_Q row per program (one (i, s)
    # pair), so no atomics on grad_Q. Inside, it loops over j and:
    #   1) accumulates grad_Q  += grad_scores[i, j] * D[j, argmax[i, j, s]]
    #   2) atomic_add  grad_D[j, argmax[i, j, s]] += grad_scores[i, j] * Q[i, s]
    #
    # The key optimisation vs the two-pass backward: Q[i, s, :] is hoisted
    # out of the j loop. The two-pass dD reloads it Nd times.

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
        Nd: tl.constexpr,
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
    ):
        pid = tl.program_id(0)
        i = pid // Lq
        s = pid % Lq

        k = tl.arange(0, d_pad)
        km = k < d

        q_active = True
        if has_q_mask:
            qm = tl.load(q_mask_ptr + i * stride_qm_n + s * stride_qm_l).to(tl.int1)
            q_active = qm != 0

        # Always zero the grad_Q row (masked rows must produce zeros, not
        # leftover garbage from a previous launch).
        if not q_active:
            tl.store(
                grad_Q_ptr + i * stride_gq_n + s * stride_gq_l + k * stride_gq_k,
                tl.zeros([d_pad], dtype=tl.float32),
                mask=km,
            )
            return

        # Hoist Q[i, s, :] out of the j loop — amortizes across all j's
        # atomic_add into grad_D. This is the single biggest HBM win
        # vs the two-pass backward.
        qv = tl.load(
            Q_ptr + i * stride_q_n + s * stride_q_l + k * stride_q_k,
            mask=km,
            other=0.0,
        ).to(tl.float32)

        acc_Q = tl.zeros([d_pad], dtype=tl.float32)

        for j in range(0, Nd):
            gs = tl.load(grad_s_ptr + i * stride_gs_n + j * stride_gs_d).to(tl.float32)
            t = tl.load(argmax_ptr + (i * Nd + j) * stride_a_pair + s * stride_a_lq).to(tl.int32)
            dv = tl.load(
                D_ptr + j * stride_d_n + t * stride_d_l + k * stride_d_k,
                mask=km,
                other=0.0,
            ).to(tl.float32)

            acc_Q += gs * dv

            tl.atomic_add(
                grad_D_ptr + j * stride_gd_n + t * stride_gd_l + k * stride_gd_k,
                gs * qv,
                mask=km,
            )

        tl.store(
            grad_Q_ptr + i * stride_gq_n + s * stride_gq_l + k * stride_gq_k,
            acc_Q,
            mask=km,
        )


def maxsim_backward_unified(
    grad_scores: torch.Tensor,
    Q: torch.Tensor,
    D: torch.Tensor,
    argmax: torch.Tensor,
    q_mask: torch.Tensor | None = None,
    *,
    method: str = "atomic",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-pass ``grad_Q`` + ``grad_D`` backward.

    Produces the same numerical result as the two-pass
    :func:`late_interaction_kernels.backward.maxsim_backward`, to fp32
    tolerance, in roughly half the HBM traffic.

    Args:
        grad_scores: ``[Nq, Nd]`` fp32 upstream gradient.
        Q: ``[Nq, Lq, d]``.
        D: ``[Nd, Ld, d]``.
        argmax: ``[Nq*Nd, Lq]`` int32 — the winner buffer written by the
            forward kernel.
        q_mask: optional ``[Nq, Lq]`` bool mask.
        method: for now, only ``"atomic"`` is exposed. A CSR-deterministic
            path is on the roadmap once bench numbers settle.

    Returns:
        ``(grad_Q, grad_D)`` cast back to the dtypes of ``Q`` and ``D``.
    """
    if method != "atomic":
        raise ValueError(
            f"Only method='atomic' is supported right now; got {method!r}. "
            "A deterministic CSR variant is planned once benchmarks justify it."
        )
    if not _HAS_TRITON:  # pragma: no cover
        raise RuntimeError(
            "maxsim_backward_unified requires Triton; install a CUDA-enabled "
            "Triton or use the two-pass backward."
        )
    if not (Q.is_cuda and D.is_cuda):
        raise RuntimeError("maxsim_backward_unified requires CUDA tensors.")

    Nq, Lq, d = Q.shape
    Nd, Ld, _ = D.shape
    d_pad = next_pow2(d)

    grad_Q = torch.empty(Nq, Lq, d, device=Q.device, dtype=torch.float32)
    grad_D = torch.zeros(Nd, Ld, d, device=D.device, dtype=torch.float32)

    has_q_mask = q_mask is not None
    qm_ptr = q_mask if has_q_mask else Q
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
    )

    return grad_Q.to(Q.dtype), grad_D.to(D.dtype)
