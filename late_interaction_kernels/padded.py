"""Padded-input helpers for batched reranking.

Entry points
------------
pack_padded
    Convert ``[B, Lq, d]`` / ``[B, C, Ld, d]`` padded tensors to the packed
    (``cu_seqlens``) layout used by :func:`score_pairs_packed` and
    :func:`maxsim_varlen`, **without any device→host syncs** in the hot path.

maxsim_padded
    High-level reranking wrapper: takes padded inputs, packs them, scores via
    the Triton scatter kernel on CUDA (reference on everything else), and
    returns ``[B, C]`` fp32.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PackedBatch:
    """Result of :func:`pack_padded`.

    All tensors live on the same device as the inputs.
    """

    Q_packed: torch.Tensor
    """``[sum(Lq_b), d]`` packed query tokens."""

    cu_seqlens_q: torch.Tensor
    """``[B + 1]`` int32 cumulative query lengths."""

    D_packed: torch.Tensor
    """``[sum(Ld_{b,c}), d]`` packed document tokens."""

    cu_seqlens_d: torch.Tensor
    """``[B * C + 1]`` int32 cumulative doc lengths (row-major over ``(b, c)``)."""

    pair_q_idx: torch.Tensor
    """``[B * C]`` int32 — ``pair_q_idx[b * C + c] = b``."""

    pair_d_idx: torch.Tensor
    """``[B * C]`` int32 — ``pair_d_idx[b * C + c] = b * C + c``."""

    max_seqlen_q: int
    """Maximum query length (Python int) — pass to :func:`score_pairs_packed`
    as ``max_seqlen_q`` to skip the kernel's own D2H validation."""

    def __iter__(self):
        """Allow tuple-style unpacking: ``Q, cu_q, D, cu_d, pq, pd, max_lq = pack_padded(...)``."""
        yield self.Q_packed
        yield self.cu_seqlens_q
        yield self.D_packed
        yield self.cu_seqlens_d
        yield self.pair_q_idx
        yield self.pair_d_idx
        yield self.max_seqlen_q


# ---------------------------------------------------------------------------
# Core helper
# ---------------------------------------------------------------------------


def pack_padded(
    queries: torch.Tensor,
    documents: torch.Tensor,
    query_lengths: torch.Tensor,
    doc_lengths: torch.Tensor,
    *,
    validate: bool = False,
) -> PackedBatch:
    """Convert padded reranking tensors to the packed layout.

    Builds a :class:`PackedBatch` from padded ``[B, Lq, d]`` /
    ``[B, C, Ld, d]`` inputs using boolean-mask gather + ``cumsum`` — no
    ``.item()`` syncs except for the single unavoidable
    ``max_seqlen_q = int(qlen.max().item())`` that :func:`score_pairs_packed`
    needs to size shared memory.

    Args:
        queries: ``[B, Lq, d]`` fp16 / bf16 / fp32.
        documents: ``[B, C, Ld, d]`` same dtype as ``queries``.
        query_lengths: ``[B]`` int32/int64. Valid lengths in ``(0, Lq]``.
        doc_lengths: ``[B, C]`` int32/int64. Valid lengths in ``(0, Ld]``.
        validate: when ``True`` performs three extra D2H syncs to catch
            zero-length or out-of-range lengths. Leave ``False`` (default)
            in the hot path; use ``True`` for first-time / debug runs.

    Returns:
        :class:`PackedBatch` with packed tensors, ``cu_seqlens`` offsets,
        pair indices, and ``max_seqlen_q`` as a Python int.

    Shape summary::

        Q_packed:      [sum(query_lengths), d]
        cu_seqlens_q:  [B + 1]          int32
        D_packed:      [sum(doc_lengths.reshape(-1)), d]
        cu_seqlens_d:  [B * C + 1]      int32
        pair_q_idx:    [B * C]          int32  — pair_q_idx[b*C+c] = b
        pair_d_idx:    [B * C]          int32  — pair_d_idx[b*C+c] = b*C+c
        max_seqlen_q:  Python int
    """
    if queries.dim() != 3:
        raise ValueError(f"queries must be [B, Lq, d]; got shape {tuple(queries.shape)}")
    if documents.dim() != 4:
        raise ValueError(f"documents must be [B, C, Ld, d]; got shape {tuple(documents.shape)}")

    B, Lq_max, d = queries.shape
    Bd, C, Ld_max, Dd = documents.shape

    if B != Bd:
        raise ValueError(f"batch size mismatch: queries B={B}, documents B={Bd}")
    if d != Dd:
        raise ValueError(f"embedding dim mismatch: queries d={d}, documents d={Dd}")
    if query_lengths.shape != (B,):
        raise ValueError(f"query_lengths must be [B={B}]; got {tuple(query_lengths.shape)}")
    if doc_lengths.shape != (B, C):
        raise ValueError(f"doc_lengths must be [B={B}, C={C}]; got {tuple(doc_lengths.shape)}")

    device = queries.device
    qlen = query_lengths.to(device=device, dtype=torch.int32)
    dlen = doc_lengths.to(device=device, dtype=torch.int32)

    if validate:
        if (qlen <= 0).any().item():
            raise ValueError("query_lengths must all be > 0")
        if (dlen <= 0).any().item():
            raise ValueError("doc_lengths must all be > 0")
        if (qlen > Lq_max).any().item() or (dlen > Ld_max).any().item():
            raise ValueError("a length exceeds the padded extent")

    # --- gather valid query tokens ---
    q_pos = torch.arange(Lq_max, device=device, dtype=torch.int32)
    q_mask = q_pos[None, :] < qlen[:, None]                     # [B, Lq_max]
    Q_packed = queries.reshape(B * Lq_max, d)[q_mask.reshape(-1)]

    cu_seqlens_q = torch.zeros(B + 1, dtype=torch.int32, device=device)
    cu_seqlens_q[1:] = qlen.cumsum(0)

    # --- gather valid document tokens ---
    d_pos = torch.arange(Ld_max, device=device, dtype=torch.int32)
    d_mask = d_pos[None, None, :] < dlen[:, :, None]            # [B, C, Ld_max]
    D_packed = documents.reshape(B * C * Ld_max, d)[d_mask.reshape(-1)]

    cu_seqlens_d = torch.zeros(B * C + 1, dtype=torch.int32, device=device)
    cu_seqlens_d[1:] = dlen.reshape(-1).cumsum(0)

    # --- pair indices (row-major) ---
    flat_idx = torch.arange(B * C, dtype=torch.int32, device=device)
    pair_q_idx = (flat_idx // C).contiguous()
    pair_d_idx = flat_idx.contiguous()

    # One D2H sync: pull max_seqlen_q so the kernel can skip its own.
    max_seqlen_q = int(qlen.max().item())

    return PackedBatch(
        Q_packed=Q_packed,
        cu_seqlens_q=cu_seqlens_q,
        D_packed=D_packed,
        cu_seqlens_d=cu_seqlens_d,
        pair_q_idx=pair_q_idx,
        pair_d_idx=pair_d_idx,
        max_seqlen_q=max_seqlen_q,
    )


# ---------------------------------------------------------------------------
# High-level padded reranking entry point
# ---------------------------------------------------------------------------


def maxsim_padded(
    queries: torch.Tensor,
    documents: torch.Tensor,
    query_lengths: torch.Tensor,
    doc_lengths: torch.Tensor,
    *,
    validate: bool = False,
) -> torch.Tensor:
    """Score reranking candidates from padded inputs, returning ``[B, C]`` fp32.

    Packs the inputs via :func:`pack_padded` and delegates to
    :func:`score_pairs_packed` on CUDA (Triton) or the pure-PyTorch
    :func:`~late_interaction_kernels.reference.maxsim_reference_scatter`
    on CPU / MPS / any non-CUDA device.

    Args:
        queries: ``[B, Lq, d]``.
        documents: ``[B, C, Ld, d]`` — ``C`` candidates per query.
        query_lengths: ``[B]`` valid query lengths.
        doc_lengths: ``[B, C]`` valid doc lengths.
        validate: forward to :func:`pack_padded`; enables extra D2H checks.

    Returns:
        scores: ``[B, C]`` fp32 on the same device as ``queries``.
    """
    B = queries.shape[0]
    C = documents.shape[1]

    batch = pack_padded(queries, documents, query_lengths, doc_lengths, validate=validate)

    if queries.is_cuda:
        try:
            from late_interaction_kernels.score_pairs import score_pairs_packed

            flat = score_pairs_packed(
                batch.Q_packed,
                batch.D_packed,
                batch.cu_seqlens_q,
                batch.cu_seqlens_d,
                batch.pair_q_idx,
                batch.pair_d_idx,
                max_seqlen_q=batch.max_seqlen_q,
            )
            return flat.view(B, C)
        except ImportError:
            pass  # Triton not available — fall through to reference

    from late_interaction_kernels.reference import maxsim_reference_scatter

    flat = maxsim_reference_scatter(
        batch.Q_packed.float(),
        batch.D_packed.float(),
        batch.cu_seqlens_q,
        batch.cu_seqlens_d,
        batch.pair_q_idx,
        batch.pair_d_idx,
    )
    return flat.view(B, C)


__all__ = ["PackedBatch", "pack_padded", "maxsim_padded"]
