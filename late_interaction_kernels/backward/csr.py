"""Scatter-free CSR backward for ``grad_D`` (deterministic, zero atomics).

Inverts the atomic scatter: sorts ``(i, s)`` pairs by ``argmax`` to
build per-``j`` CSR buckets, then each ``(j, t)`` program reduces its
own bucket into one register and writes once. Wins at very high
``grad_D`` contention (``Nq ≥ 256 ∧ Nd ≥ 256 ∧ Lq ≤ 64``) and when
bitwise reproducibility across runs matters.

See :doc:`../docs/design.md` for the full derivation and the
heuristic the ``"auto"`` selector uses to pick between ``unified``,
``csr``, and ``atomic``.
"""

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

from late_interaction_kernels._utils import next_pow2

# ---------------------------------------------------------------------------
# PyTorch-side CSR construction
# ---------------------------------------------------------------------------


def _build_csr(
    argmax: torch.Tensor,
    Nq: int,
    Nd: int,
    Lq: int,
    Ld: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-doc-batch CSR from argmax.

    Args:
        argmax: [Nq*Nd, Lq] int32 — layout is ``(i * Nd + j, s)``.

    Returns:
        row_ptr: [Nd, Ld+1] int32 — ``row_ptr[j, t]`` is the first index in
            ``perm[j]`` whose key is ``>= t``. ``row_ptr[j, t+1] - row_ptr[j, t]``
            is the size of bucket ``(j, t)``.
        perm:    [Nd, Nq*Lq] int32 — flat ``(i * Lq + s)`` indices sorted by
            the argmax value within each ``j``-row.
    """
    # Group rows by doc-batch j. .contiguous() is required because the
    # Triton kernel reads flat strides off the permuted layout.
    argmax_by_doc = argmax.view(Nq, Nd, Lq).permute(1, 0, 2).contiguous().view(Nd, Nq * Lq)

    # perm[j, :] holds the original flat index k = q_idx*Lq + s, sorted by
    # argmax value within each j-row.
    sorted_argmax, perm = argmax_by_doc.sort(dim=1)

    # row_ptr via batched searchsorted; boundaries[j, :] = [0, 1, ..., Ld].
    boundaries = torch.arange(Ld + 1, device=argmax.device, dtype=sorted_argmax.dtype)
    boundaries = boundaries.unsqueeze(0).expand(Nd, -1).contiguous()
    row_ptr = torch.searchsorted(sorted_argmax, boundaries).to(torch.int32)

    return row_ptr, perm.to(torch.int32)


# ---------------------------------------------------------------------------
# Triton CSR kernel for grad_D
# ---------------------------------------------------------------------------


@triton.jit
def _bwd_dD_csr_kernel(
    Q_ptr,
    perm_ptr,
    row_ptr_ptr,
    grad_s_ptr,
    q_mask_ptr,
    grad_D_ptr,
    Nd: tl.constexpr,
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
    stride_rp_n,
    stride_rp_l,
    stride_perm_n,
    stride_perm_f,
    has_q_mask: tl.constexpr,
):
    """One program per ``(j, t)`` output row.

    Reduces ``grad_D[j, t, :] = Σ gs[i, j] * Q[i, s, :]`` over the bucket
    ``perm[j, row_ptr[j, t] : row_ptr[j, t+1]]``. No atomics, one store.
    """
    pid = tl.program_id(0)
    d_idx = pid // Ld
    t = pid % Ld

    emb_off = tl.arange(0, d_pad)
    emb_mask = emb_off < d

    start = tl.load(row_ptr_ptr + d_idx * stride_rp_n + t * stride_rp_l).to(tl.int32)
    end = tl.load(row_ptr_ptr + d_idx * stride_rp_n + (t + 1) * stride_rp_l).to(tl.int32)

    acc = tl.zeros([d_pad], dtype=tl.float32)

    # Dynamic-bound loop — Triton lowers this to a while loop on the GPU.
    # Empty bucket (start == end) skips the body and we store zeros below.
    for off in range(start, end):
        flat = tl.load(perm_ptr + d_idx * stride_perm_n + off * stride_perm_f).to(tl.int32)
        q_idx = flat // Lq
        s = flat % Lq

        q_active = True
        if has_q_mask:
            q_mask_val = tl.load(q_mask_ptr + q_idx * stride_qm_n + s * stride_qm_l).to(tl.int1)
            q_active = q_mask_val != 0

        if q_active:
            gs = tl.load(grad_s_ptr + q_idx * stride_gs_n + d_idx * stride_gs_d).to(tl.float32)
            qv = tl.load(
                Q_ptr + q_idx * stride_q_n + s * stride_q_l + emb_off * stride_q_k,
                mask=emb_mask,
                other=0.0,
            ).to(tl.float32)
            acc += gs * qv

    tl.store(
        grad_D_ptr + d_idx * stride_gd_n + t * stride_gd_l + emb_off * stride_gd_k,
        acc,
        mask=emb_mask,
    )


# ---------------------------------------------------------------------------
# Public launcher
# ---------------------------------------------------------------------------


def maxsim_backward_csr_dD(
    grad_scores: torch.Tensor,
    Q: torch.Tensor,
    D: torch.Tensor,
    argmax: torch.Tensor,
    q_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Compute ``grad_D`` only, via the CSR path.

    Args match ``maxsim_backward`` in ``backward.py``. ``grad_Q`` is computed
    by the existing non-atomic kernel in ``backward.py``.

    Returns:
        grad_D: [Nd, Ld, d] fp32 (caller casts to D.dtype).
    """
    Nq, Lq, d = Q.shape
    Nd, Ld, _ = D.shape
    d_pad = next_pow2(d)

    row_ptr, perm = _build_csr(argmax, Nq, Nd, Lq, Ld)

    grad_D = torch.zeros(Nd, Ld, d, device=D.device, dtype=torch.float32)

    has_q_mask = q_mask is not None
    qm_ptr = q_mask if has_q_mask else Q
    qm_strides = (q_mask.stride(0), q_mask.stride(1)) if has_q_mask else (0, 0)

    _bwd_dD_csr_kernel[(Nd * Ld,)](
        Q,
        perm,
        row_ptr,
        grad_scores,
        qm_ptr,
        grad_D,
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
        row_ptr.stride(0),
        row_ptr.stride(1),
        perm.stride(0),
        perm.stride(1),
        has_q_mask,
    )
    return grad_D
