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

from ._autotune import forward_configs, prune_forward
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


@triton.autotune(
    configs=forward_configs(),
    key=["Lq", "max_Ld", "d_pad", "nbits", "normalize", "SAVE_ARGMAX"],
    prune_configs_by={"early_config_prune": prune_forward},
)
@triton.jit
def _maxsim_residual_kernel(
    Q_ptr,  # [Nq, Lq, d] query embeddings (assumed normalized)
    codes_ptr,  # [Nd, Ld]  int64 centroid codes (padded)
    residuals_ptr,  # [Nd, Ld, packed_dim] uint8 packed residuals
    doc_len_ptr,  # [Nd] int64 real doc lengths
    centroids_ptr,  # [n_centroids, d] fp32 centroid table
    bucket_weights_ptr,  # [n_buckets] fp32 residual bucket values
    out_ptr,  # [Nq, Nd] fp32
    argmax_ptr,  # [Nq, Nd, Lq] int32 (only written if SAVE_ARGMAX)
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
    stride_am_n,
    stride_am_d,
    stride_am_l,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    normalize: tl.constexpr,
    SAVE_ARGMAX: tl.constexpr,
):
    """One program per (query, doc). For each doc token we:
    1. Read centroid code, gather the centroid[code, :].
    2. Read packed residual bytes, unpack to `d` fp32 bucket-weight values.
    3. emb = centroid + residual (optionally L2-normalized in SRAM).
    4. Tile-accumulate S = Q @ emb.T, run the MaxSim online-max like the
       main forward kernel.

    If SAVE_ARGMAX is set, we also write the per-query-token winning doc-token
    index to ``argmax[q_idx, d_idx, s]`` — the backward kernel uses this to
    recompute the winner's ``emb`` without materializing the score tensor.
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
        if SAVE_ARGMAX:
            am = tl.zeros([BLOCK_Q], dtype=tl.int32)

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
                einv = tl.rsqrt(tl.maximum(en, 1e-12))
                emb = emb * einv[:, None]

            D_block = emb.to(COMPUTE_DTYPE)
            S = tl.dot(Q_block, tl.trans(D_block), out_dtype=tl.float32)
            S = tl.where(d_valid[None, :], S, float("-inf"))
            tile_max = tl.max(S, axis=1)
            if SAVE_ARGMAX:
                tile_arg = tl.argmax(S, axis=1).to(tl.int32) + d_start
                update = tile_max > m
                am = tl.where(update, tile_arg, am)
            m = tl.maximum(m, tile_max)

        m_finite = m != float("-inf")
        m = tl.where(m_finite & q_valid, m, 0.0)
        score_acc += tl.sum(m)

        if SAVE_ARGMAX:
            # Clamp argmax for invalid query tokens to 0 so the backward's load
            # is always in-bounds (the q_mask/length guard will zero the grad anyway).
            am_safe = tl.where(q_valid, am, 0)
            tl.store(
                argmax_ptr + q_idx * stride_am_n + d_idx * stride_am_d + q_off * stride_am_l,
                am_safe,
                mask=q_valid,
            )

    tl.store(out_ptr + q_idx * stride_out_n + d_idx * stride_out_d, score_acc)


# -----------------------------------------------------------------------------
# Backward: grad_Q for maxsim_residual
# -----------------------------------------------------------------------------


@triton.jit
def _maxsim_residual_bwd_dQ_kernel(
    argmax_ptr,  # [Nq, Nd, Lq] int32
    grad_s_ptr,  # [Nq, Nd] fp32
    codes_ptr,  # [Nd, Ld]  int64 centroid codes
    residuals_ptr,  # [Nd, Ld, packed_dim] uint8
    centroids_ptr,  # [n_centroids, d] fp32
    bucket_weights_ptr,  # [n_buckets] fp32
    grad_Qhat_ptr,  # [Nq, Lq, d] fp32 output
    Nq: tl.constexpr,
    Nd: tl.constexpr,
    Lq: tl.constexpr,
    d: tl.constexpr,
    d_pad: tl.constexpr,
    nbits: tl.constexpr,
    codes_per_byte: tl.constexpr,
    stride_am_n,
    stride_am_d,
    stride_am_l,
    stride_gs_n,
    stride_gs_d,
    stride_codes_n,
    stride_codes_l,
    stride_res_n,
    stride_res_l,
    stride_res_p,
    stride_cent_c,
    stride_cent_d,
    stride_gq_n,
    stride_gq_l,
    stride_gq_k,
    normalize: tl.constexpr,
):
    """One program per (q_idx, s). For each doc j:
      1. Load argmax[q,j,s] = t (the winning doc token).
      2. Gather centroid[codes[j,t]] and unpack residual bucket values.
      3. emb = centroid + residual (optionally L2-normalized).
      4. acc += grad_scores[q, j] * emb.

    This gives grad w.r.t. the *normalized* Q (if normalize=True). The
    Python wrapper applies the Q-side L2-norm Jacobian.
    """
    pid = tl.program_id(0)
    q_idx = pid // Lq
    s = pid % Lq

    k_off = tl.arange(0, d_pad)
    k_mask = k_off < d

    byte_idx = (k_off // codes_per_byte).to(tl.int32)
    slot_idx = (k_off % codes_per_byte).to(tl.int32)
    shift = (slot_idx * nbits).to(tl.int32)
    code_mask = tl.full([d_pad], (1 << nbits) - 1, dtype=tl.int32)

    acc = tl.zeros([d_pad], dtype=tl.float32)

    for j in range(0, Nd):
        gs = tl.load(grad_s_ptr + q_idx * stride_gs_n + j * stride_gs_d).to(tl.float32)
        t = tl.load(argmax_ptr + q_idx * stride_am_n + j * stride_am_d + s * stride_am_l).to(tl.int32)

        cent_code = tl.load(codes_ptr + j * stride_codes_n + t * stride_codes_l).to(tl.int32)
        cent = tl.load(
            centroids_ptr + cent_code * stride_cent_c + k_off * stride_cent_d,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        byte_vals = tl.load(
            residuals_ptr + j * stride_res_n + t * stride_res_l + byte_idx * stride_res_p,
            mask=k_mask,
            other=0,
        ).to(tl.int32)
        bucket_codes = (byte_vals >> shift) & code_mask
        bucket_vals = tl.load(
            bucket_weights_ptr + bucket_codes,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        emb = cent + bucket_vals

        if normalize:
            en = tl.sum(emb * emb)
            einv = 1.0 / tl.sqrt(tl.maximum(en, 1e-12))
            emb = emb * einv

        acc += gs * emb

    tl.store(
        grad_Qhat_ptr + q_idx * stride_gq_n + s * stride_gq_l + k_off * stride_gq_k,
        acc,
        mask=k_mask,
    )


def _maxsim_residual_forward(
    Q: torch.Tensor,
    codes: torch.Tensor,
    residuals: torch.Tensor,
    doc_lengths: torch.Tensor,
    centroids: torch.Tensor,
    bucket_weights: torch.Tensor,
    nbits: int,
    normalize: bool,
    save_argmax: bool,
):
    """Launcher used by both the public inference API and the autograd wrapper."""
    if nbits not in (2, 4, 8):
        raise ValueError(f"nbits must be 2, 4, or 8; got {nbits}")
    if codes.dim() != 2:
        raise ValueError("codes must be [Nd, max_Ld]")
    if residuals.dim() != 3:
        raise ValueError("residuals must be [Nd, max_Ld, packed_dim]")

    Q = ensure_contiguous_last(Q).contiguous()
    codes = codes.contiguous().to(torch.int64)
    residuals = residuals.contiguous().to(torch.uint8)
    doc_lengths = doc_lengths.contiguous().to(torch.int64)
    # Centroids can stay in fp16 / bf16: the kernel casts to fp32 in registers
    # after the load. Forcing fp32 would double the centroid bandwidth (and
    # fast-plaid stores centroids as fp16 on disk anyway).
    if centroids.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        centroids = centroids.to(torch.float32)
    centroids = centroids.contiguous()
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
    if save_argmax:
        argmax = torch.empty(Nq, Nd, Lq, device=Q.device, dtype=torch.int32)
    else:
        argmax = torch.empty(1, device=Q.device, dtype=torch.int32)  # dummy

    am_strides = (argmax.stride(0), argmax.stride(1), argmax.stride(2)) if save_argmax else (0, 0, 0)

    grid = (Nq * Nd,)
    _maxsim_residual_kernel[grid](
        Q,
        codes,
        residuals,
        doc_lengths,
        centroids,
        bucket_weights,
        out,
        argmax,
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
        am_strides[0],
        am_strides[1],
        am_strides[2],
        COMPUTE_DTYPE=tl_dtype,
        normalize=normalize,
        SAVE_ARGMAX=save_argmax,
    )
    return out, (argmax if save_argmax else None), (Q, codes, residuals, centroids, bucket_weights)


class _MaxSimResidualFn(torch.autograd.Function):
    """Autograd wrapper for ``maxsim_residual`` — only ``Q`` is differentiable.

    ``codes`` / ``residuals`` are integer tensors (not differentiable).
    ``centroids`` and ``bucket_weights`` are typically frozen k-means
    quantization artefacts; we do not propagate gradient into them.
    """

    @staticmethod
    def forward(
        ctx,
        Q,
        codes,
        residuals,
        doc_lengths,
        centroids,
        bucket_weights,
        nbits,
        normalize,
    ):
        scores, argmax, packed = _maxsim_residual_forward(
            Q, codes, residuals, doc_lengths, centroids, bucket_weights, nbits, normalize, True
        )
        Qc, codes_c, residuals_c, centroids_c, bucket_weights_c = packed
        ctx.save_for_backward(Qc, codes_c, residuals_c, centroids_c, bucket_weights_c, argmax)
        ctx.nbits = nbits
        ctx.normalize = normalize
        return scores

    @staticmethod
    def backward(ctx, grad_scores):
        Q, codes, residuals, centroids, bucket_weights, argmax = ctx.saved_tensors
        nbits = ctx.nbits
        normalize = ctx.normalize

        grad_scores = grad_scores.contiguous().to(torch.float32)
        Nq, Lq, d = Q.shape
        Nd = codes.shape[0]
        d_pad = next_pow2(d)
        codes_per_byte = 8 // nbits

        grad_Qhat = torch.zeros(Nq, Lq, d, device=Q.device, dtype=torch.float32)

        grid = (Nq * Lq,)
        _maxsim_residual_bwd_dQ_kernel[grid](
            argmax,
            grad_scores,
            codes,
            residuals,
            centroids,
            bucket_weights,
            grad_Qhat,
            Nq,
            Nd,
            Lq,
            d,
            d_pad,
            nbits,
            codes_per_byte,
            argmax.stride(0),
            argmax.stride(1),
            argmax.stride(2),
            grad_scores.stride(0),
            grad_scores.stride(1),
            codes.stride(0),
            codes.stride(1),
            residuals.stride(0),
            residuals.stride(1),
            residuals.stride(2),
            centroids.stride(0),
            centroids.stride(1),
            grad_Qhat.stride(0),
            grad_Qhat.stride(1),
            grad_Qhat.stride(2),
            normalize=normalize,
            num_warps=4,
            num_stages=2,
        )

        if normalize:
            # grad_Qhat is gradient w.r.t. Q_hat = Q / ||Q||. Apply the
            # Q-side L2-normalize Jacobian:
            #   d Qhat / d Q = (I - Qhat Qhat^T) / ||Q||
            q_norm = torch.linalg.vector_norm(Q.float(), dim=-1, keepdim=True).clamp_min(1e-6)
            Q_hat = Q.float() / q_norm
            proj = (grad_Qhat * Q_hat).sum(-1, keepdim=True) * Q_hat
            grad_Q = (grad_Qhat - proj) / q_norm
        else:
            grad_Q = grad_Qhat

        return (
            grad_Q.to(Q.dtype),  # Q
            None,  # codes
            None,  # residuals
            None,  # doc_lengths
            None,  # centroids
            None,  # bucket_weights
            None,  # nbits
            None,  # normalize
        )


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
    """Fused PLAID residual-decompression + MaxSim (autograd-aware on Q).

    Decompresses PLAID / ColBERTv2 compressed embeddings on-the-fly in
    SRAM and scores them against the query — the exact-rerank step, fused
    into a single Triton kernel.

    Compressed format (following the PLAID / ColBERTv2 convention):

    * ``codes[n, t]`` is an integer index into ``centroids`` for token ``t``
      of doc ``n``.
    * ``residuals[n, t, :]`` is ``ceil(d * nbits / 8)`` bytes; each byte packs
      ``8 / nbits`` bucket indices (little-endian within the byte).
    * ``bucket_weights[b]`` gives the scalar quantization offset added onto
      the centroid feature.

    Args:
        Q: ``[Nq, Lq, d]`` query embeddings (fp16/bf16/fp32). Requires-grad
            is honored via a fused backward (see Autograd below).
        codes: ``[Nd, max_Ld]`` int64 centroid codes.
        residuals: ``[Nd, max_Ld, packed_dim]`` uint8. ``packed_dim = d * nbits / 8``.
        doc_lengths: ``[Nd]`` int64 real lengths.
        centroids: ``[n_centroids, d]`` fp32 centroid table.
        bucket_weights: ``[n_buckets]`` fp32. ``n_buckets = 2 ** nbits``.
        nbits: one of ``{2, 4, 8}``.
        normalize: if True, L2-normalize Q and the reconstructed embedding
            (standard ColBERTv2 / PLAID convention). Backward applies the
            Q-side L2-norm Jacobian.

    Returns:
        scores: ``[Nq, Nd]`` fp32.

    Autograd
    --------
    Only ``Q`` is treated as differentiable. ``codes`` / ``residuals`` are
    integer tensors; ``centroids`` and ``bucket_weights`` are typically
    frozen post-k-means and no gradient is propagated into them. The fused
    backward kernel recovers the winning doc token per ``(query, doc, q_tok)``
    from a saved argmax buffer, re-decompresses its embedding in SRAM, and
    accumulates ``grad_Q += grad_scores * emb`` — no dense unpack, no
    ``[Nq, Nd, Lq, Ld]`` scratch.

    Use ``maxsim_residual_inference`` if you don't need gradients and want
    to skip saving the argmax buffer.
    """
    if Q.dim() == 2:
        Q = Q.unsqueeze(0)
    if Q.requires_grad:
        return _MaxSimResidualFn.apply(
            Q, codes, residuals, doc_lengths, centroids, bucket_weights, nbits, normalize
        )
    scores, _, _ = _maxsim_residual_forward(
        Q, codes, residuals, doc_lengths, centroids, bucket_weights, nbits, normalize, False
    )
    return scores


def maxsim_residual_inference(
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
    """Same as ``maxsim_residual`` but skips the argmax save (no autograd).

    Use this for pure reranking / inference — it avoids a ``[Nq, Nd, Lq]``
    int32 buffer that would otherwise be reserved for the backward pass.
    """
    if Q.dim() == 2:
        Q = Q.unsqueeze(0)
    scores, _, _ = _maxsim_residual_forward(
        Q, codes, residuals, doc_lengths, centroids, bucket_weights, nbits, normalize, False
    )
    return scores


# -----------------------------------------------------------------------------
# C3. maxsim_residual_varlen — packed / ragged decompress + MaxSim
# -----------------------------------------------------------------------------
#
# Fast-plaid stores its per-passage codes and residuals in "strided" tensors
# (a concatenated flat buffer + cumulative offsets), which is exactly the
# varlen format. The dense kernel above forces callers to pad every doc up to
# ``max_Ld`` and to materialize a ``[Ntop, max_Ld, packed_dim]`` residual
# tensor — a ``Ntop * max_Ld * packed_dim`` byte-scatter on every query.
# This kernel reads the ragged buffers directly and cuts both the scatter and
# the attention mask.


@triton.autotune(
    configs=forward_configs(),
    key=["Lq", "max_Ld", "d_pad", "nbits", "normalize", "SAVE_ARGMAX"],
    prune_configs_by={"early_config_prune": prune_forward},
)
@triton.jit
def _maxsim_residual_varlen_kernel(
    Q_ptr,  # [Nq, Lq, d] query embeddings (assumed normalized or we normalize)
    codes_flat_ptr,  # [sum_Ld] int64 centroid codes, concatenated over docs
    residuals_flat_ptr,  # [sum_Ld, packed_dim] uint8 packed residuals
    cu_seqlens_d_ptr,  # [Nd + 1] int32 cumulative doc-token offsets
    centroids_ptr,  # [n_centroids, d] fp32
    bucket_weights_ptr,  # [n_buckets] fp32
    out_ptr,  # [Nq, Nd] fp32
    argmax_ptr,  # [Nq, Nd, Lq] int32 (only if SAVE_ARGMAX)
    Nq: tl.constexpr,
    Nd: tl.constexpr,
    Lq: tl.constexpr,
    max_Ld: tl.constexpr,  # worst-case doc length across this batch
    d: tl.constexpr,
    d_pad: tl.constexpr,
    packed_dim: tl.constexpr,
    nbits: tl.constexpr,
    codes_per_byte: tl.constexpr,
    stride_q_n,
    stride_q_l,
    stride_q_d,
    stride_codes_t,  # codes_flat is 1-D so stride is 1, kept for clarity
    stride_res_t,
    stride_res_p,
    stride_cent_c,
    stride_cent_d,
    stride_out_n,
    stride_out_d,
    stride_am_n,
    stride_am_d,
    stride_am_l,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    normalize: tl.constexpr,
    SAVE_ARGMAX: tl.constexpr,
):
    """One program per (query, doc). Same inner math as the padded kernel,
    but the per-doc-token reads index into a ragged `[sum_Ld, *]` flat buffer
    via ``cu_seqlens_d``. No padded scratch, no attention mask.
    """
    pid = tl.program_id(0)
    q_idx = pid // Nd
    d_idx = pid % Nd

    d_lo = tl.load(cu_seqlens_d_ptr + d_idx).to(tl.int32)
    d_hi = tl.load(cu_seqlens_d_ptr + d_idx + 1).to(tl.int32)
    doc_len = d_hi - d_lo

    k_off = tl.arange(0, d_pad)
    k_mask = k_off < d

    score_acc = tl.zeros([], dtype=tl.float32)

    byte_idx = (k_off // codes_per_byte).to(tl.int32)
    slot_idx = (k_off % codes_per_byte).to(tl.int32)
    shift = (slot_idx * nbits).to(tl.int32)
    code_mask = tl.full([d_pad], (1 << nbits) - 1, dtype=tl.int32)

    if doc_len == 0:
        tl.store(out_ptr + q_idx * stride_out_n + d_idx * stride_out_d, score_acc)
        return

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
            qinv = tl.rsqrt(tl.maximum(qn, 1e-12))
            Qf = Qf * qinv[:, None]
        Q_block = Qf.to(COMPUTE_DTYPE)

        m = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
        if SAVE_ARGMAX:
            am = tl.zeros([BLOCK_Q], dtype=tl.int32)

        for d_start in range(0, max_Ld, BLOCK_D):
            d_off = d_start + tl.arange(0, BLOCK_D)
            d_valid = d_off < doc_len
            t_idx = d_lo + d_off  # ragged row indices into flat buffers

            cent_codes = tl.load(
                codes_flat_ptr + t_idx * stride_codes_t,
                mask=d_valid,
                other=0,
            ).to(tl.int32)
            cent = tl.load(
                centroids_ptr + cent_codes[:, None] * stride_cent_c + k_off[None, :] * stride_cent_d,
                mask=d_valid[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            byte_vals = tl.load(
                residuals_flat_ptr
                + t_idx[:, None] * stride_res_t
                + byte_idx[None, :] * stride_res_p,
                mask=d_valid[:, None] & k_mask[None, :],
                other=0,
            ).to(tl.int32)
            bucket_codes = (byte_vals >> shift[None, :]) & code_mask[None, :]

            bucket_vals = tl.load(
                bucket_weights_ptr + bucket_codes,
                mask=d_valid[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            emb = cent + bucket_vals
            if normalize:
                en = tl.sum(emb * emb, axis=1)
                einv = tl.rsqrt(tl.maximum(en, 1e-12))
                emb = emb * einv[:, None]

            D_block = emb.to(COMPUTE_DTYPE)
            S = tl.dot(Q_block, tl.trans(D_block), out_dtype=tl.float32)
            S = tl.where(d_valid[None, :], S, float("-inf"))
            tile_max = tl.max(S, axis=1)
            if SAVE_ARGMAX:
                tile_arg = tl.argmax(S, axis=1).to(tl.int32) + d_start
                update = tile_max > m
                am = tl.where(update, tile_arg, am)
            m = tl.maximum(m, tile_max)

        m_finite = m != float("-inf")
        m = tl.where(m_finite & q_valid, m, 0.0)
        score_acc += tl.sum(m)

        if SAVE_ARGMAX:
            am_safe = tl.where(q_valid, am, 0)
            tl.store(
                argmax_ptr + q_idx * stride_am_n + d_idx * stride_am_d + q_off * stride_am_l,
                am_safe,
                mask=q_valid,
            )

    tl.store(out_ptr + q_idx * stride_out_n + d_idx * stride_out_d, score_acc)


def maxsim_residual_varlen(
    Q: torch.Tensor,
    codes_flat: torch.Tensor,
    residuals_flat: torch.Tensor,
    cu_seqlens_d: torch.Tensor,
    centroids: torch.Tensor,
    bucket_weights: torch.Tensor,
    nbits: int,
    *,
    max_seqlen_d: int | None = None,
    normalize: bool = True,
) -> torch.Tensor:
    """Packed / ragged PLAID residual-decompression + MaxSim (inference only).

    Matches the storage fast-plaid uses internally for ``doc_codes_strided``
    and ``doc_residuals_strided``: a single flat tensor per field plus
    cumulative offsets. This skips the padded ``[Ntop, max_Ld, packed_dim]``
    residual scratch and the attention mask entirely — the two big
    allocations per-query in the standard PLAID pipeline.

    Args:
        Q: ``[Nq, Lq, d]`` (or ``[Lq, d]``) query embeddings.
        codes_flat: ``[total_d_tokens]`` int64 concatenated centroid codes.
        residuals_flat: ``[total_d_tokens, packed_dim]`` uint8 concatenated
            packed residuals, where ``packed_dim = d * nbits / 8``.
        cu_seqlens_d: ``[Nd + 1]`` int32 cumulative offsets into the flat
            buffers (``cu_seqlens_d[0] = 0``, ``cu_seqlens_d[-1] = total_d_tokens``).
        centroids: ``[n_centroids, d]`` fp32.
        bucket_weights: ``[2**nbits]`` fp32.
        nbits: one of ``{2, 4, 8}``.
        max_seqlen_d: worst-case doc length. Inferred from ``cu_seqlens_d``
            if omitted.
        normalize: L2-normalize Q and the reconstructed emb inside the kernel
            (standard PLAID convention).

    Returns:
        scores: ``[Nq, Nd]`` fp32. If ``Q`` was 2-D, returns ``[Nd]``.

    This path is inference-only (no autograd). Use ``maxsim_residual`` if you
    need gradients on ``Q`` — quantization codes are integers and
    non-differentiable by construction either way, but the dense kernel has
    a fused backward.
    """
    if nbits not in (2, 4, 8):
        raise ValueError(f"nbits must be 2, 4, or 8; got {nbits}")
    q_squeeze = Q.dim() == 2
    if q_squeeze:
        Q = Q.unsqueeze(0)
    if codes_flat.dim() != 1:
        raise ValueError("codes_flat must be 1-D [total_d_tokens]")
    if residuals_flat.dim() != 2:
        raise ValueError("residuals_flat must be 2-D [total_d_tokens, packed_dim]")
    if cu_seqlens_d.dim() != 1:
        raise ValueError("cu_seqlens_d must be 1-D [Nd + 1]")

    Q = ensure_contiguous_last(Q).contiguous()
    codes_flat = codes_flat.contiguous().to(torch.int64)
    residuals_flat = residuals_flat.contiguous().to(torch.uint8)
    cu_seqlens_d = cu_seqlens_d.contiguous().to(torch.int32)
    if centroids.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        centroids = centroids.to(torch.float32)
    centroids = centroids.contiguous()
    bucket_weights = bucket_weights.contiguous().to(torch.float32)

    Nq, Lq, d = Q.shape
    Nd = cu_seqlens_d.numel() - 1
    packed_dim = residuals_flat.shape[-1]
    expected_pd = (d * nbits + 7) // 8
    if packed_dim != expected_pd:
        raise ValueError(
            f"residuals_flat last dim {packed_dim} != expected {expected_pd} for d={d}, nbits={nbits}"
        )

    if max_seqlen_d is None:
        # One D2H sync per call; batch your queries if this matters.
        starts = cu_seqlens_d[:-1]
        ends = cu_seqlens_d[1:]
        max_seqlen_d = int((ends - starts).max().item()) if Nd > 0 else 0
    max_seqlen_d = max(int(max_seqlen_d), 1)

    d_pad = next_pow2(d)
    codes_per_byte = 8 // nbits
    compute_dtype = torch.float16 if Q.dtype == torch.float16 else torch.bfloat16
    tl_dtype = tl.float16 if compute_dtype == torch.float16 else tl.bfloat16

    out = torch.empty(Nq, Nd, device=Q.device, dtype=torch.float32)
    argmax = torch.empty(1, device=Q.device, dtype=torch.int32)  # unused placeholder

    grid = (Nq * max(Nd, 1),)
    _maxsim_residual_varlen_kernel[grid](
        Q,
        codes_flat,
        residuals_flat,
        cu_seqlens_d,
        centroids,
        bucket_weights,
        out,
        argmax,
        Nq,
        Nd,
        Lq,
        max_seqlen_d,
        d,
        d_pad,
        packed_dim,
        nbits,
        codes_per_byte,
        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        codes_flat.stride(0),
        residuals_flat.stride(0),
        residuals_flat.stride(1),
        centroids.stride(0),
        centroids.stride(1),
        out.stride(0),
        out.stride(1),
        0,
        0,
        0,
        COMPUTE_DTYPE=tl_dtype,
        normalize=normalize,
        SAVE_ARGMAX=False,
    )
    if q_squeeze:
        return out.squeeze(0)
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
    "maxsim_residual_inference",
    "maxsim_residual_reference",
    "maxsim_residual_varlen",
]
