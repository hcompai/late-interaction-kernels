"""Unified backward kernel — scaffold and reference (0.6.0-dev).

**Status: reference implementation + API scaffold only. The Triton kernel
lands in 0.6.1 after cluster autotune.**

Motivation
----------
The current backward (:mod:`late_interaction_kernels.backward`) runs two
Triton kernels. Each one reloads ``Q``, ``D``, and ``grad_scores`` from
HBM:

    pass 1 (``_bwd_dQ_kernel``):  reads D, argmax, grad_scores -> writes grad_Q
    pass 2 (``_bwd_dD_kernel``):  reads Q, argmax, grad_scores -> writes grad_D

For MaxSim-dominant shapes (``Nq·Nd·Lq·Ld`` large) the backward is
bandwidth-bound. A single fused pass that keeps ``grad_scores`` and the
argmax buffer in SMEM while accumulating both ``grad_Q`` (row-owned) and
``grad_D`` (``atomic_add``) halves HBM traffic. Flash-Attention-2 made
this switch and got their biggest single speedup from it.

This module ships:

  * :func:`maxsim_backward_unified_reference` — pure-PyTorch reference
    that computes the same numbers as the current two-pass backward.
    Used to pin the upcoming Triton kernel's numerics before landing.
  * :func:`maxsim_backward_unified` — placeholder that raises
    ``NotImplementedError``. Kept here so the public module surface is
    stable for 0.6.x import-compat tests.

Correctness contract (for the Triton kernel, when written):
  * For every ``(i, j, q)`` the argmax ``k = argmax[i*Nd+j, q]`` is
    already known. ``grad_Q[i, q]`` is incremented by
    ``grad_scores[i, j] * D[j, k]`` (summed across ``j``). ``grad_D[j, k]``
    is incremented by ``grad_scores[i, j] * Q[i, q]`` (summed across
    ``(i, q)`` — needs an atomic_add or a CSR bucketed reduction).
  * With ``q_mask``, positions where ``q_mask[i, q] == 0`` are skipped
    entirely (contribute zero to both grads).
  * ``d_mask`` is already folded into the forward (masked tokens get
    ``-inf`` and never win the argmax), so the backward does not need
    to re-check it.
"""

from __future__ import annotations

import torch


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


def maxsim_backward_unified(*args, **kwargs):  # noqa: ARG001
    """Unified-pass Triton backward — **not yet implemented**.

    Lands in 0.6.1 once the autotune shortlist is settled on the cluster.
    See :func:`maxsim_backward_unified_reference` for the numerical
    contract and ``docs/rfc/0.6.0.md`` for the design notes.

    Use :func:`late_interaction_kernels.autograd.maxsim` for training — it
    calls the stable 0.5.x two-pass backward and will switch to this
    unified implementation transparently once it lands.
    """
    raise NotImplementedError(
        "maxsim_backward_unified is a 0.6.1 deliverable; use the two-pass "
        "`maxsim_backward` for now. See late_interaction_kernels.backward_unified "
        "for the reference impl and docs/rfc/0.6.0.md for the plan."
    )
