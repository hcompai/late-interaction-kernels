"""PLAID / ColBERTv2 kernels: approximate scoring + fused residual-decompress + MaxSim.

Two Triton kernels for the core operations of a ColBERTv2-style reranker
(approximate scoring over centroid codes, exact rerank over packed
residuals), so you can build one end-to-end in Python without hand-writing
the fused ops.

* :func:`plaid_approx_score` — the approximate scoring step. Gathers
  per-token query↔centroid scores, masks padded positions, takes
  max-over-doc-tokens and sum-over-query-tokens, all in one kernel.

* :func:`maxsim_residual` — the exact rerank step. Takes per-token
  centroid codes + packed residuals (2/4/8-bit) + centroid table + bucket
  weights, decompresses on-the-fly in SRAM, L2-normalizes, and computes
  MaxSim against the query embedding — all in a single kernel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._utils import ensure_contiguous_last, next_pow2

# -----------------------------------------------------------------------------
# C1. plaid_approx_score
# -----------------------------------------------------------------------------


@triton.jit
def _plaid_approx_score_kernel(
    qcs_ptr,  # [n_centroids, Lq] query-centroid scores, fp32
    codes_ptr,  # [B, max_Ld] int64 centroid codes (padded)
    doc_len_ptr,  # [B] int64 real lengths
    out_ptr,  # [B] fp32 scores
    B: tl.constexpr,
    n_centroids: tl.constexpr,
    Lq: tl.constexpr,
    max_Ld: tl.constexpr,
    Lq_pad: tl.constexpr,
    stride_qcs_c,
    stride_qcs_l,
    stride_codes_b,
    stride_codes_d,
    BLOCK_D: tl.constexpr,
):
    """One program per document. Accumulates per-query-token running max
    over doc-token indices, then sums over Lq.
    """
    b = tl.program_id(0)
    doc_len = tl.load(doc_len_ptr + b)

    q_off = tl.arange(0, Lq_pad)
    q_valid = q_off < Lq
    # Running max per query token. -inf for full-length rows; we set empty docs to 0 below.
    m = tl.full([Lq_pad], float("-inf"), dtype=tl.float32)

    for d_start in range(0, max_Ld, BLOCK_D):
        d_off = d_start + tl.arange(0, BLOCK_D)
        d_valid = d_off < doc_len

        codes = tl.load(
            codes_ptr + b * stride_codes_b + d_off * stride_codes_d,
            mask=d_valid,
            other=0,
        ).to(tl.int32)
        # Clamp to valid centroid range so the gather never reads out of bounds.
        codes = tl.where(codes < n_centroids, codes, 0)

        # Gather [BLOCK_D, Lq] scores from query_centroid_scores.
        tile = tl.load(
            qcs_ptr + codes[:, None] * stride_qcs_c + q_off[None, :] * stride_qcs_l,
            mask=d_valid[:, None] & q_valid[None, :],
            other=float("-inf"),
        )
        # Mask out invalid doc positions.
        tile = tl.where(d_valid[:, None], tile, float("-inf"))
        m = tl.maximum(m, tl.max(tile, axis=0))

    # Empty doc -> -inf still; clamp to 0.
    m = tl.where((m != float("-inf")) & q_valid, m, 0.0)
    score = tl.sum(m)
    tl.store(out_ptr + b, score)


def plaid_approx_score(
    query_centroid_scores: torch.Tensor,
    codes: torch.Tensor,
    doc_lengths: torch.Tensor,
) -> torch.Tensor:
    """Fused PLAID-style approximate scoring.

    Performs the ColBERTv2 IVF-prune step — gather per-token centroid
    scores, mask padded positions, max-over-doc, sum-over-query — in a
    single kernel launch. Input codes are padded (not bit-packed); the
    API matches what ColBERTv2 / PLAID-style retrievers pass into their
    ``colbert_score_reduce`` routine.

    Args:
        query_centroid_scores: ``[n_centroids, Lq]`` fp32.
            ``= centroids @ Q.T``, computed once per query.
        codes: ``[B, max_Ld]`` int64 centroid codes. Positions beyond the real
            doc length are masked out via ``doc_lengths``; their values are
            not read.
        doc_lengths: ``[B]`` int64 actual doc lengths (``<= max_Ld``).

    Returns:
        scores: ``[B]`` fp32 approximate MaxSim scores.
    """
    if query_centroid_scores.dim() != 2:
        raise ValueError("query_centroid_scores must be [n_centroids, Lq]")
    if codes.dim() != 2:
        raise ValueError("codes must be [B, max_Ld]")
    if doc_lengths.dim() != 1 or doc_lengths.shape[0] != codes.shape[0]:
        raise ValueError("doc_lengths must be [B] with B matching codes")

    query_centroid_scores = query_centroid_scores.contiguous().to(torch.float32)
    codes = codes.contiguous().to(torch.int64)
    doc_lengths = doc_lengths.contiguous().to(torch.int64)

    n_centroids, Lq = query_centroid_scores.shape
    B, max_Ld = codes.shape
    Lq_pad = next_pow2(Lq)
    BLOCK_D = 128 if max_Ld >= 128 else max(16, next_pow2(max_Ld))

    out = torch.empty(B, device=codes.device, dtype=torch.float32)

    grid = (B,)
    _plaid_approx_score_kernel[grid](
        query_centroid_scores,
        codes,
        doc_lengths,
        out,
        B,
        n_centroids,
        Lq,
        max_Ld,
        Lq_pad,
        query_centroid_scores.stride(0),
        query_centroid_scores.stride(1),
        codes.stride(0),
        codes.stride(1),
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )
    return out


# -----------------------------------------------------------------------------
# C2. maxsim_residual
# -----------------------------------------------------------------------------


@triton.jit
def _maxsim_residual_kernel(
    Q_ptr,  # [Nq, Lq, d] query embeddings (assumed normalized)
    codes_ptr,  # [Nd, Ld]  int64 centroid codes (padded)
    residuals_ptr,  # [Nd, Ld, packed_dim] uint8 packed residuals
    doc_len_ptr,  # [Nd] int64 real doc lengths
    centroids_ptr,  # [n_centroids, d] fp32 centroid table
    bucket_weights_ptr,  # [n_buckets] fp32 residual bucket values
    out_ptr,  # [Nq, Nd] fp32
    Nq: tl.constexpr,
    Nd: tl.constexpr,
    Lq: tl.constexpr,
    max_Ld: tl.constexpr,
    d: tl.constexpr,
    d_pad: tl.constexpr,
    packed_dim: tl.constexpr,
    nbits: tl.constexpr,  # 2, 4, or 8
    codes_per_byte: tl.constexpr,  # 8 / nbits
    stride_q_n,
    stride_q_l,
    stride_q_d,
    stride_codes_n,
    stride_codes_l,
    stride_res_n,
    stride_res_l,
    stride_res_p,
    stride_cent_c,
    stride_cent_d,
    stride_out_n,
    stride_out_d,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    normalize: tl.constexpr,
):
    """One program per (query, doc). For each doc token we:
    1. Read centroid code, gather the centroid[code, :].
    2. Read packed residual bytes, unpack to `d` fp32 bucket-weight values.
    3. emb = centroid + residual (clamped to unit-norm via rsqrt).
    4. Tile-accumulate S = Q @ emb.T, run the MaxSim online-max like the
       main forward kernel.
    """
    pid = tl.program_id(0)
    q_idx = pid // Nd
    d_idx = pid % Nd
    doc_len = tl.load(doc_len_ptr + d_idx)

    k_off = tl.arange(0, d_pad)
    k_mask = k_off < d

    # --- Load one query row tile at a time, but reuse doc emb per d-tile.
    score_acc = tl.zeros([], dtype=tl.float32)

    # Pre-compute which bytes each feature dim comes from. nbits is a constexpr.
    # feature f -> byte index = f // codes_per_byte, slot = f % codes_per_byte.
    byte_idx = (k_off // codes_per_byte).to(tl.int32)
    slot_idx = (k_off % codes_per_byte).to(tl.int32)
    # Mask to extract one code per feature from the byte.
    # shift = slot_idx * nbits ; code = (byte >> shift) & ((1 << nbits) - 1)
    shift = (slot_idx * nbits).to(tl.int32)
    code_mask = tl.full([d_pad], (1 << nbits) - 1, dtype=tl.int32)

    for q_start in tl.static_range(0, Lq, BLOCK_Q):
        q_off = q_start + tl.arange(0, BLOCK_Q)
        q_valid = q_off < Lq
        Qf = tl.load(
            Q_ptr + q_idx * stride_q_n + q_off[:, None] * stride_q_l + k_off[None, :] * stride_q_d,
            mask=q_valid[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        if normalize:
            qn = tl.sum(Qf * Qf, axis=1)
            qinv = 1.0 / tl.sqrt(tl.maximum(qn, 1e-12))
            Qf = Qf * qinv[:, None]
        Q_block = Qf.to(COMPUTE_DTYPE)

        m = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)

        for d_start in range(0, max_Ld, BLOCK_D):
            d_off = d_start + tl.arange(0, BLOCK_D)
            d_valid = d_off < doc_len

            # Load and gather centroid rows.
            cent_codes = tl.load(
                codes_ptr + d_idx * stride_codes_n + d_off * stride_codes_l,
                mask=d_valid,
                other=0,
            ).to(tl.int32)
            cent = tl.load(
                centroids_ptr + cent_codes[:, None] * stride_cent_c + k_off[None, :] * stride_cent_d,
                mask=d_valid[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            # Load packed residual bytes for this tile, expand to per-feature codes.
            # For each (d-token, feature), we index into bytes at position byte_idx[f].
            # BLOCK_D × d_pad uint8 load.
            byte_vals = tl.load(
                residuals_ptr
                + d_idx * stride_res_n
                + d_off[:, None] * stride_res_l
                + byte_idx[None, :] * stride_res_p,
                mask=d_valid[:, None] & k_mask[None, :],
                other=0,
            ).to(tl.int32)
            bucket_codes = (byte_vals >> shift[None, :]) & code_mask[None, :]

            # Gather bucket weights.
            bucket_vals = tl.load(
                bucket_weights_ptr + bucket_codes,
                mask=d_valid[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            emb = cent + bucket_vals

            if normalize:
                en = tl.sum(emb * emb, axis=1)
                einv = 1.0 / tl.sqrt(tl.maximum(en, 1e-12))
                emb = emb * einv[:, None]

            D_block = emb.to(COMPUTE_DTYPE)
            S = tl.dot(Q_block, tl.trans(D_block), out_dtype=tl.float32)
            S = tl.where(d_valid[None, :], S, float("-inf"))
            m = tl.maximum(m, tl.max(S, axis=1))

        m_finite = m != float("-inf")
        m = tl.where(m_finite & q_valid, m, 0.0)
        score_acc += tl.sum(m)

    tl.store(out_ptr + q_idx * stride_out_n + d_idx * stride_out_d, score_acc)


def maxsim_residual(
    Q: torch.Tensor,
    codes: torch.Tensor,
    residuals: torch.Tensor,
    doc_lengths: torch.Tensor,
    centroids: torch.Tensor,
    bucket_weights: torch.Tensor,
    nbits: int,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Fused PLAID residual-decompression + MaxSim.

    Decompresses PLAID / ColBERTv2 compressed embeddings on-the-fly in
    SRAM and scores them against the query — the exact-rerank step, fused
    into a single Triton kernel callable from Python.

    Compressed format (following the PLAID / ColBERTv2 convention):

    * ``codes[n, t]`` is an integer index into ``centroids`` for token ``t``
      of doc ``n``.
    * ``residuals[n, t, :]`` is ``ceil(d * nbits / 8)`` bytes; each byte packs
      ``8 / nbits`` bucket indices (little-endian within the byte).
    * ``bucket_weights[b]`` gives the scalar quantization offset added onto
      the centroid feature.

    Args:
        Q: ``[Nq, Lq, d]`` query embeddings (fp16/bf16/fp32).
        codes: ``[Nd, max_Ld]`` int64 centroid codes.
        residuals: ``[Nd, max_Ld, packed_dim]`` uint8. ``packed_dim = d * nbits / 8``.
        doc_lengths: ``[Nd]`` int64 real lengths.
        centroids: ``[n_centroids, d]`` fp32 centroid table.
        bucket_weights: ``[n_buckets]`` fp32. ``n_buckets = 2 ** nbits``.
        nbits: one of ``{2, 4, 8}``.
        normalize: if True, L2-normalize Q and the reconstructed embedding
            (standard ColBERTv2 / PLAID convention).

    Returns:
        scores: ``[Nq, Nd]`` fp32.
    """
    if nbits not in (2, 4, 8):
        raise ValueError(f"nbits must be 2, 4, or 8; got {nbits}")
    if Q.dim() == 2:
        Q = Q.unsqueeze(0)
    if codes.dim() != 2:
        raise ValueError("codes must be [Nd, max_Ld]")
    if residuals.dim() != 3:
        raise ValueError("residuals must be [Nd, max_Ld, packed_dim]")

    Q = ensure_contiguous_last(Q).contiguous()
    codes = codes.contiguous().to(torch.int64)
    residuals = residuals.contiguous().to(torch.uint8)
    doc_lengths = doc_lengths.contiguous().to(torch.int64)
    centroids = centroids.contiguous().to(torch.float32)
    bucket_weights = bucket_weights.contiguous().to(torch.float32)

    Nq, Lq, d = Q.shape
    Nd, max_Ld = codes.shape
    packed_dim = residuals.shape[-1]
    expected_pd = (d * nbits + 7) // 8
    if packed_dim != expected_pd:
        raise ValueError(
            f"residuals last dim {packed_dim} != expected {expected_pd} for d={d}, nbits={nbits}"
        )

    d_pad = next_pow2(d)
    codes_per_byte = 8 // nbits
    compute_dtype = torch.float16 if Q.dtype == torch.float16 else torch.bfloat16
    tl_dtype = tl.float16 if compute_dtype == torch.float16 else tl.bfloat16

    out = torch.empty(Nq, Nd, device=Q.device, dtype=torch.float32)

    BLOCK_Q = 32 if Lq >= 32 else max(16, next_pow2(Lq))
    BLOCK_D = 64 if max_Ld >= 64 else max(16, next_pow2(max_Ld))

    grid = (Nq * Nd,)
    _maxsim_residual_kernel[grid](
        Q,
        codes,
        residuals,
        doc_lengths,
        centroids,
        bucket_weights,
        out,
        Nq,
        Nd,
        Lq,
        max_Ld,
        d,
        d_pad,
        packed_dim,
        nbits,
        codes_per_byte,
        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        codes.stride(0),
        codes.stride(1),
        residuals.stride(0),
        residuals.stride(1),
        residuals.stride(2),
        centroids.stride(0),
        centroids.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_Q=BLOCK_Q,
        BLOCK_D=BLOCK_D,
        COMPUTE_DTYPE=tl_dtype,
        normalize=normalize,
        num_warps=4,
        num_stages=2,
    )
    return out


# Reference implementations live in ``late_interaction_kernels.reference``
# so they're importable on CPU-only platforms without Triton. Re-export for
# convenience.
from .reference import (  # noqa: E402
    maxsim_residual_reference,
    plaid_approx_score_reference,
)

__all__ = [
    "plaid_approx_score",
    "plaid_approx_score_reference",
    "maxsim_residual",
    "maxsim_residual_reference",
]
