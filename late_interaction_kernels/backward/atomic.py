"""Two-pass backward (``grad_Q`` then ``grad_D``).

Forward saves a ``[Nq * Nd, Lq]`` int32 argmax buffer. Backward scatters:

    grad_Q[i, s] = q_active[i, s] · Σ_j grad_scores[i, j] · D[j, argmax[i, j, s]]
    grad_D[j, t] = Σ_{(i,s) : argmax[i, j, s] == t} q_active[i, s] · grad_scores[i, j] · Q[i, s]

``grad_D`` accumulates with fp32 ``atomic_add``. Argmax ties are broken
by ``tl.argmax`` (lowest index), bitwise-reproducible.
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
# grad_Q kernel — one program per (q_batch, q_token)
# ---------------------------------------------------------------------------


@triton.jit
def _bwd_dQ_kernel(
    D_ptr,
    argmax_ptr,
    grad_s_ptr,
    q_mask_ptr,
    grad_Q_ptr,
    Nq: tl.constexpr,
    Nd: tl.constexpr,
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
):
    pid = tl.program_id(0)
    q_idx = pid // Lq
    s = pid % Lq

    k = tl.arange(0, d_pad)
    km = k < d
    acc = tl.zeros([d_pad], dtype=tl.float32)

    q_active = True
    if has_q_mask:
        qm = tl.load(q_mask_ptr + q_idx * stride_qm_n + s * stride_qm_l).to(tl.int1)
        q_active = qm != 0

    if q_active:
        for j in range(0, Nd):
            gs = tl.load(grad_s_ptr + q_idx * stride_gs_n + j * stride_gs_d).to(tl.float32)
            t = tl.load(argmax_ptr + (q_idx * Nd + j) * stride_a_pair + s * stride_a_lq)
            t = t.to(tl.int32)
            v = tl.load(
                D_ptr + j * stride_d_n + t * stride_d_l + k * stride_d_k,
                mask=km,
                other=0.0,
            ).to(tl.float32)
            acc += gs * v

    tl.store(
        grad_Q_ptr + q_idx * stride_gq_n + s * stride_gq_l + k * stride_gq_k,
        acc,
        mask=km,
    )


# ---------------------------------------------------------------------------
# grad_D kernel — atomic-add scatter. One program per (q_batch, d_batch).
# ---------------------------------------------------------------------------


@triton.jit
def _bwd_dD_kernel(
    Q_ptr,
    argmax_ptr,
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
    stride_a_pair,
    stride_a_lq,
    has_q_mask: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid // Nd
    j = pid % Nd

    gs = tl.load(grad_s_ptr + i * stride_gs_n + j * stride_gs_d).to(tl.float32)

    k = tl.arange(0, d_pad)
    km = k < d

    # One query-token at a time (Lq is tiny). For each s, read Q[i, s, :],
    # the winner index t = argmax[i, j, s], and atomic-add gs * Q[i, s, :]
    # into grad_D[j, t, :].
    for s in range(0, Lq):
        q_active = True
        if has_q_mask:
            qm = tl.load(q_mask_ptr + i * stride_qm_n + s * stride_qm_l).to(tl.int1)
            q_active = qm != 0
        if q_active:
            t = tl.load(argmax_ptr + (i * Nd + j) * stride_a_pair + s * stride_a_lq).to(tl.int32)
            qv = tl.load(
                Q_ptr + i * stride_q_n + s * stride_q_l + k * stride_q_k,
                mask=km,
                other=0.0,
            ).to(tl.float32)
            tl.atomic_add(
                grad_D_ptr + j * stride_gd_n + t * stride_gd_l + k * stride_gd_k,
                gs * qv,
                mask=km,
            )


# ---------------------------------------------------------------------------
# Python-side launcher
# ---------------------------------------------------------------------------


def maxsim_backward(
    grad_scores: torch.Tensor,
    Q: torch.Tensor,
    D: torch.Tensor,
    argmax: torch.Tensor,
    q_mask: torch.Tensor | None,
    d_mask: torch.Tensor | None,  # noqa: ARG001 - masked tokens already -inf in fwd
    *,
    method: str = "auto",
):
    """Compute ``grad_Q`` and ``grad_D`` from the saved argmax buffer.

    Args:
        grad_scores: [Nq, Nd] fp32 upstream gradient.
        Q, D, argmax, q_mask, d_mask: as produced by the forward.
        method: one of

            * ``"auto"`` (default) — pick ``csr`` or ``atomic`` based on total
              contention. Empirical crossover on H100 is ~100M atomic ops
              (= ``Nq * Nd * Lq * d``); above that the scatter-free CSR path
              wins, below it the fp32 atomic path is faster because H100's
              atomic_add hardware coalesces at L2 and the extra ``sort`` cost
              of CSR dominates.
            * ``"csr"`` — force the scatter-free path (CSR-bucketed reduction).
              Fastest on large-batch training, long-sequence ColPali-style
              shapes, or GPUs with slower atomics (A100, consumer cards).
            * ``"atomic"`` — force fp32 ``tl.atomic_add``. Fastest on common
              PyLate training shapes (``Nq = Nd ≲ 128``, ``Lq = 32``,
              ``Ld ≲ 300``) on H100.

    Returns:
        (grad_Q, grad_D) with the original dtypes of Q and D.
    """
    if method not in ("csr", "atomic", "auto"):
        raise ValueError(f"method must be 'csr', 'atomic', or 'auto', got {method!r}")

    Nq, Lq, d = Q.shape
    Nd, Ld, _ = D.shape

    if method == "auto":
        # Heuristic derived from bench_backward_method.py on 1×H100. CSR wins
        # when any of three distinct atomic-serialization regimes trigger:
        #   1. Sheer volume:  Nq * Nd * Lq * d ≥ 1e8 total atomic ops.
        #   2. Long-sequence: Lq ≥ 1024 (each atomic-kernel program emits
        #      many writes per doc-batch → per-program serialization).
        #   3. Huge corpus:   Nd ≥ 1024 (massive atomic-kernel grid with
        #      high fan-in contention on each grad_D row).
        # In all other cases, H100's hardware-coalesced fp32 atomics beat
        # CSR's sort + searchsorted + (mostly empty) bucketed reduction.
        big_workload = (Nq * Nd * Lq * d) >= 100_000_000
        long_seq = Lq >= 1024 and (Nq * Nd) >= 16
        huge_corpus = Nd >= 1024
        method = "csr" if (big_workload or long_seq or huge_corpus) else "atomic"
    d_pad = next_pow2(d)

    grad_Q = torch.zeros(Nq, Lq, d, device=Q.device, dtype=torch.float32)

    has_q_mask = q_mask is not None
    qm_ptr = q_mask if has_q_mask else Q
    qm_strides = (q_mask.stride(0), q_mask.stride(1)) if has_q_mask else (0, 0)

    # grad_Q kernel is shared — it already has no atomics (each query-token
    # program owns its output row).
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
    )

    if method == "csr":
        # Imported here to avoid a circular import (csr uses _utils).
        from late_interaction_kernels.backward.csr import maxsim_backward_csr_dD

        grad_D = maxsim_backward_csr_dD(grad_scores, Q, D, argmax, q_mask)
    else:
        grad_D = torch.zeros(Nd, Ld, d, device=D.device, dtype=torch.float32)
        _bwd_dD_kernel[(Nq * Nd,)](
            Q,
            argmax,
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
            argmax.stride(0),
            argmax.stride(1),
            has_q_mask,
        )

    return grad_Q.to(Q.dtype), grad_D.to(D.dtype)
