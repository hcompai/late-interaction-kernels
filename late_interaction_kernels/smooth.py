"""Top-K MaxSim with O(K) backward.

Aggregates the top-K doc tokens per query token; the backward
distributes gradient across those K winners only (O(K) extra work on
top of the hard backward, no ``[Nq, Nd, Lq, Ld]`` scratch).
``top_k=1, aggregation="sum"`` is bit-identical to hard MaxSim.
"""

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

from ._autotune import forward_configs, prune_forward
from ._utils import next_pow2, pick_compute_dtype


def smooth_maxsim_reference(
    Q: torch.Tensor,
    D: torch.Tensor,
    q_mask: torch.Tensor | None = None,
    d_mask: torch.Tensor | None = None,
    *,
    top_k: int = 4,
    aggregation: str = "mean",
    normalize: bool = False,
) -> torch.Tensor:
    """Pure-PyTorch reference: sum over s of aggregate(top-K per (i,j,s)).

    Materializes the full [Nq, Nd, Lq, Ld] similarity — slow, correct.
    """
    if aggregation not in ("mean", "sum"):
        raise ValueError(f"aggregation must be 'mean' or 'sum'; got {aggregation!r}")
    q_squeeze = Q.dim() == 2
    d_squeeze = D.dim() == 2
    if q_squeeze:
        Q = Q.unsqueeze(0)
    if d_squeeze:
        D = D.unsqueeze(0)

    acc_dtype = Q.dtype if Q.dtype == torch.float64 else torch.float32
    Qf = Q.to(acc_dtype)
    Df = D.to(acc_dtype)
    if normalize:
        Qf = torch.nn.functional.normalize(Qf, p=2, dim=-1, eps=1e-12)
        Df = torch.nn.functional.normalize(Df, p=2, dim=-1, eps=1e-12)

    S = torch.einsum("ild,jtd->ijlt", Qf, Df)  # [Nq, Nd, Lq, Ld]
    if d_mask is not None:
        if d_mask.dim() == 1:
            d_mask = d_mask.unsqueeze(0)
        S = S.masked_fill(~d_mask.bool()[None, :, None, :], float("-inf"))

    k = min(top_k, S.shape[-1])
    top = S.topk(k, dim=-1).values  # [Nq, Nd, Lq, k]
    top = torch.where(torch.isfinite(top), top, torch.zeros_like(top))

    if aggregation == "mean":
        row = top.sum(dim=-1) / k
    else:
        row = top.sum(dim=-1)

    if q_mask is not None:
        if q_mask.dim() == 1:
            q_mask = q_mask.unsqueeze(0)
        row = row * q_mask.to(row.dtype)[:, None, :]

    scores = row.sum(dim=-1)  # [Nq, Nd]
    if q_squeeze:
        scores = scores.squeeze(0)
    if d_squeeze:
        scores = scores.squeeze(-1)
    return scores


if _HAS_TRITON:

    @triton.autotune(
        configs=forward_configs(),
        key=["Lq", "d_pad", "K", "has_q_mask", "has_d_mask", "normalize"],
        prune_configs_by={"early_config_prune": prune_forward},
    )
    @triton.jit
    def _smooth_maxsim_fwd_kernel(
        Q_ptr,
        D_ptr,
        q_mask_ptr,
        d_mask_ptr,
        scores_ptr,
        topk_idx_ptr,
        Nq: tl.constexpr,
        Nd: tl.constexpr,
        Lq: tl.constexpr,
        Ld,
        d: tl.constexpr,
        d_pad: tl.constexpr,
        K: tl.constexpr,
        agg_scale,  # fp32 — 1.0 for sum, 1/K for mean
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
        stride_tk_pair,
        stride_tk_lq,
        stride_tk_k,
        has_q_mask: tl.constexpr,
        has_d_mask: tl.constexpr,
        normalize: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_D: tl.constexpr,
        COMPUTE_DTYPE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        q_idx = pid // Nd
        d_idx = pid % Nd

        k_off = tl.arange(0, d_pad)
        k_mask = k_off < d

        score_acc = tl.zeros([], dtype=tl.float32)

        for q_start in tl.static_range(0, Lq, BLOCK_Q):
            q_off = q_start + tl.arange(0, BLOCK_Q)
            q_valid = q_off < Lq

            if has_q_mask:
                qm = tl.load(
                    q_mask_ptr + q_idx * stride_qm_n + q_off * stride_qm_l,
                    mask=q_valid,
                    other=0,
                ).to(tl.int1)
                q_active = q_valid & qm
            else:
                q_active = q_valid

            Q_block_f32 = tl.load(
                Q_ptr + q_idx * stride_q_n + q_off[:, None] * stride_q_l + k_off[None, :] * stride_q_d,
                mask=q_valid[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            if normalize:
                q_norm_sq = tl.sum(Q_block_f32 * Q_block_f32, axis=1)
                q_inv = 1.0 / tl.sqrt(tl.maximum(q_norm_sq, 1e-12))
                Q_block_f32 = Q_block_f32 * q_inv[:, None]
            Q_block = Q_block_f32.to(COMPUTE_DTYPE)

            # Running top-K, sorted loosely by value. We don't enforce a
            # strict sort — we just guarantee the set is the K largest.
            # best_v[s, k] in fp32, best_i[s, k] in int32.
            best_v = tl.full([BLOCK_Q, K], float("-inf"), dtype=tl.float32)
            best_i = tl.zeros([BLOCK_Q, K], dtype=tl.int32)

            for d_start in range(0, Ld, BLOCK_D):
                d_off = d_start + tl.arange(0, BLOCK_D)
                d_valid = d_off < Ld

                if has_d_mask:
                    dm = tl.load(
                        d_mask_ptr + d_idx * stride_dm_n + d_off * stride_dm_l,
                        mask=d_valid,
                        other=0,
                    ).to(tl.int1)
                    d_active = d_valid & dm
                else:
                    d_active = d_valid

                D_block_f32 = tl.load(
                    D_ptr + d_idx * stride_d_n + d_off[:, None] * stride_d_l + k_off[None, :] * stride_d_d,
                    mask=d_valid[:, None] & k_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                if normalize:
                    d_norm_sq = tl.sum(D_block_f32 * D_block_f32, axis=1)
                    d_inv = 1.0 / tl.sqrt(tl.maximum(d_norm_sq, 1e-12))
                    D_block_f32 = D_block_f32 * d_inv[:, None]
                D_block = D_block_f32.to(COMPUTE_DTYPE)

                S = tl.dot(Q_block, tl.trans(D_block), out_dtype=tl.float32)
                S = tl.where(d_active[None, :], S, float("-inf"))

                # Extract top-K via K rounds of (argmax-on-union, mask-out).
                # At each round we pick the single winner from the union of
                # S_tile and the running best_v, write it into slot `ki` of
                # new_{v,i}, and mask out the source position so the next
                # round can't pick it again.
                k_range = tl.arange(0, K)
                bd_range = tl.arange(0, BLOCK_D)

                new_v = tl.full([BLOCK_Q, K], float("-inf"), dtype=tl.float32)
                new_i = tl.zeros([BLOCK_Q, K], dtype=tl.int32)

                S_work = S
                best_v_work = best_v
                best_i_work = best_i
                for ki in tl.static_range(0, K):
                    tile_max_v = tl.max(S_work, axis=1)  # [BLOCK_Q]
                    tile_max_p = tl.argmax(S_work, axis=1).to(tl.int32)
                    best_max_v = tl.max(best_v_work, axis=1)
                    best_max_p = tl.argmax(best_v_work, axis=1).to(tl.int32)

                    use_best = best_max_v >= tile_max_v
                    winner_v = tl.where(use_best, best_max_v, tile_max_v)

                    # Gather best_i at best_max_p.
                    at_best = k_range[None, :] == best_max_p[:, None]  # [BLOCK_Q, K]
                    from_best = tl.sum(tl.where(at_best, best_i_work, tl.zeros_like(best_i_work)), axis=1)
                    from_tile = tile_max_p + d_start
                    winner_i = tl.where(use_best, from_best, from_tile)

                    is_slot = k_range[None, :] == ki  # [1, K]
                    new_v = tl.where(is_slot, winner_v[:, None], new_v)
                    new_i = tl.where(is_slot, winner_i[:, None], new_i)

                    # Mask out winner in its source tensor.
                    at_tile_pos = bd_range[None, :] == tile_max_p[:, None]
                    S_work = tl.where(
                        (~use_best)[:, None] & at_tile_pos,
                        tl.full(S_work.shape, float("-inf"), tl.float32),
                        S_work,
                    )
                    best_v_work = tl.where(
                        use_best[:, None] & at_best,
                        tl.full(best_v_work.shape, float("-inf"), tl.float32),
                        best_v_work,
                    )

                best_v = new_v
                best_i = new_i

            # Store topk idx for this q_start block — flatten K over lq tokens.
            # topk_idx layout: [Nq*Nd, Lq, K], pid-major.
            k_col = tl.arange(0, K)
            tl.store(
                topk_idx_ptr
                + pid * stride_tk_pair
                + q_off[:, None] * stride_tk_lq
                + k_col[None, :] * stride_tk_k,
                best_i,
                mask=q_valid[:, None],
            )

            # Aggregate: per-row sum * agg_scale. Finite-safe: if all K slots
            # are -inf (whole doc masked), treat as 0.
            best_v = tl.where(best_v == float("-inf"), tl.zeros_like(best_v), best_v)
            row_sum = tl.sum(best_v, axis=1) * agg_scale  # [BLOCK_Q]
            row_sum = tl.where(q_active, row_sum, 0.0)
            score_acc += tl.sum(row_sum)

        tl.store(scores_ptr + q_idx * stride_s_n + d_idx * stride_s_d, score_acc)

    @triton.jit
    def _smooth_maxsim_bwd_kernel(
        Q_ptr,
        D_ptr,
        topk_idx_ptr,
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
        K: tl.constexpr,
        agg_scale,
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
        stride_tk_pair,
        stride_tk_lq,
        stride_tk_k,
        has_q_mask: tl.constexpr,
    ):
        pid = tl.program_id(0)
        i = pid // Lq
        s = pid % Lq

        kd = tl.arange(0, d_pad)
        kdm = kd < d

        q_active = True
        if has_q_mask:
            qm = tl.load(q_mask_ptr + i * stride_qm_n + s * stride_qm_l).to(tl.int1)
            q_active = qm != 0

        if not q_active:
            tl.store(
                grad_Q_ptr + i * stride_gq_n + s * stride_gq_l + kd * stride_gq_k,
                tl.zeros([d_pad], dtype=tl.float32),
                mask=kdm,
            )
            return

        qv = tl.load(
            Q_ptr + i * stride_q_n + s * stride_q_l + kd * stride_q_k,
            mask=kdm,
            other=0.0,
        ).to(tl.float32)

        acc_Q = tl.zeros([d_pad], dtype=tl.float32)

        for j in range(0, Nd):
            gs = tl.load(grad_s_ptr + i * stride_gs_n + j * stride_gs_d).to(tl.float32) * agg_scale
            for kk in tl.static_range(0, K):
                t = tl.load(
                    topk_idx_ptr + (i * Nd + j) * stride_tk_pair + s * stride_tk_lq + kk * stride_tk_k
                ).to(tl.int32)
                dv = tl.load(
                    D_ptr + j * stride_d_n + t * stride_d_l + kd * stride_d_k,
                    mask=kdm,
                    other=0.0,
                ).to(tl.float32)
                acc_Q += gs * dv
                tl.atomic_add(
                    grad_D_ptr + j * stride_gd_n + t * stride_gd_l + kd * stride_gd_k,
                    gs * qv,
                    mask=kdm,
                )

        tl.store(
            grad_Q_ptr + i * stride_gq_n + s * stride_gq_l + kd * stride_gq_k,
            acc_Q,
            mask=kdm,
        )


def _smooth_maxsim_triton_forward(
    Q: torch.Tensor,
    D: torch.Tensor,
    q_mask: torch.Tensor | None,
    d_mask: torch.Tensor | None,
    top_k: int,
    agg_scale: float,
    normalize: bool,
):
    Nq, Lq, d = Q.shape
    Nd, Ld, _ = D.shape
    d_pad = next_pow2(d)
    compute_dtype = pick_compute_dtype(Q, D)
    tl_dtype = tl.float16 if compute_dtype == torch.float16 else tl.bfloat16

    scores = torch.empty(Nq, Nd, device=Q.device, dtype=torch.float32)
    topk_idx = torch.empty(Nq * Nd, Lq, top_k, device=Q.device, dtype=torch.int32)

    has_q_mask = q_mask is not None
    has_d_mask = d_mask is not None
    q_mask_ptr = q_mask if has_q_mask else Q
    d_mask_ptr = d_mask if has_d_mask else D
    qm_strides = (q_mask.stride(0), q_mask.stride(1)) if has_q_mask else (0, 0)
    dm_strides = (d_mask.stride(0), d_mask.stride(1)) if has_d_mask else (0, 0)

    _smooth_maxsim_fwd_kernel[(Nq * Nd,)](
        Q,
        D,
        q_mask_ptr,
        d_mask_ptr,
        scores,
        topk_idx,
        Nq,
        Nd,
        Lq,
        Ld,
        d,
        d_pad,
        top_k,
        float(agg_scale),
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
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        has_q_mask,
        has_d_mask,
        normalize,
        COMPUTE_DTYPE=tl_dtype,
    )
    return scores, topk_idx


def _smooth_maxsim_triton_backward(
    grad_scores: torch.Tensor,
    Q: torch.Tensor,
    D: torch.Tensor,
    topk_idx: torch.Tensor,
    q_mask: torch.Tensor | None,
    top_k: int,
    agg_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    Nq, Lq, d = Q.shape
    Nd, Ld, _ = D.shape
    d_pad = next_pow2(d)

    grad_Q = torch.empty(Nq, Lq, d, device=Q.device, dtype=torch.float32)
    grad_D = torch.zeros(Nd, Ld, d, device=D.device, dtype=torch.float32)

    has_q_mask = q_mask is not None
    qm_ptr = q_mask if has_q_mask else Q
    qm_strides = (q_mask.stride(0), q_mask.stride(1)) if has_q_mask else (0, 0)

    _smooth_maxsim_bwd_kernel[(Nq * Lq,)](
        Q,
        D,
        topk_idx,
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
        top_k,
        float(agg_scale),
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
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        has_q_mask,
    )
    return grad_Q.to(Q.dtype), grad_D.to(D.dtype)


class _SmoothMaxSimFn(torch.autograd.Function):
    """Autograd wrapper. Uses the Triton fwd+bwd on CUDA with
    bf16/fp16 inputs; falls back to the pure-PyTorch reference
    otherwise (CPU, fp32, fp64 gradcheck, etc.)."""

    @staticmethod
    def forward(ctx, Q, D, q_mask, d_mask, top_k, agg_scale, normalize):
        use_triton = (
            _HAS_TRITON and Q.is_cuda and Q.dtype in (torch.float16, torch.bfloat16) and D.dtype == Q.dtype
        )
        if use_triton:
            scores, topk_idx = _smooth_maxsim_triton_forward(
                Q, D, q_mask, d_mask, top_k, agg_scale, normalize
            )
            ctx.save_for_backward(Q, D, topk_idx, q_mask, d_mask)
            ctx.top_k = int(top_k)
            ctx.agg_scale = float(agg_scale)
            ctx.normalize = bool(normalize)
            ctx.use_triton = True
            return scores
        # Reference path (also handles fp32/fp64 for gradcheck).
        scores = smooth_maxsim_reference(
            Q,
            D,
            q_mask=q_mask if q_mask is not None else None,
            d_mask=d_mask if d_mask is not None else None,
            top_k=int(top_k),
            aggregation="mean" if abs(agg_scale - 1.0 / max(top_k, 1)) < 1e-9 else "sum",
            normalize=bool(normalize),
        )
        ctx.save_for_backward(Q, D, q_mask, d_mask)
        ctx.top_k = int(top_k)
        ctx.agg_scale = float(agg_scale)
        ctx.normalize = bool(normalize)
        ctx.use_triton = False
        return scores

    @staticmethod
    def backward(ctx, grad_scores):
        grad_scores = grad_scores.contiguous().to(torch.float32)
        if ctx.use_triton:
            Q, D, topk_idx, q_mask, _d_mask = ctx.saved_tensors

            if ctx.normalize:
                # Backprop through L2-normalize Jacobian.
                q_norm = torch.linalg.vector_norm(Q, dim=-1, keepdim=True).clamp_min(1e-6)
                d_norm = torch.linalg.vector_norm(D, dim=-1, keepdim=True).clamp_min(1e-6)
                Q_hat = Q / q_norm
                D_hat = D / d_norm
                gQh, gDh = _smooth_maxsim_triton_backward(
                    grad_scores, Q_hat, D_hat, topk_idx, q_mask, ctx.top_k, ctx.agg_scale
                )
                gQ = (gQh - (gQh * Q_hat).sum(-1, keepdim=True) * Q_hat) / q_norm
                gD = (gDh - (gDh * D_hat).sum(-1, keepdim=True) * D_hat) / d_norm
            else:
                gQ, gD = _smooth_maxsim_triton_backward(
                    grad_scores, Q, D, topk_idx, q_mask, ctx.top_k, ctx.agg_scale
                )
            return gQ, gD, None, None, None, None, None

        # Reference autograd (single path: rebuild the graph in fp32+).
        Q, D, q_mask, d_mask = ctx.saved_tensors
        with torch.enable_grad():
            Qn = Q.detach().requires_grad_(True)
            Dn = D.detach().requires_grad_(True)
            aggregation = "mean" if abs(ctx.agg_scale - 1.0 / max(ctx.top_k, 1)) < 1e-9 else "sum"
            scores = smooth_maxsim_reference(
                Qn,
                Dn,
                q_mask=q_mask,
                d_mask=d_mask,
                top_k=ctx.top_k,
                aggregation=aggregation,
                normalize=ctx.normalize,
            )
            grads = torch.autograd.grad(scores, [Qn, Dn], grad_outputs=grad_scores)
        return grads[0].to(Q.dtype), grads[1].to(D.dtype), None, None, None, None, None


def smooth_maxsim(
    Q: torch.Tensor,
    D: torch.Tensor,
    q_mask: torch.Tensor | None = None,
    d_mask: torch.Tensor | None = None,
    *,
    top_k: int = 4,
    aggregation: str = "mean",
    normalize: bool = False,
) -> torch.Tensor:
    """Top-K-aggregated MaxSim with autograd-aware smoother backward.

    ``score[i, j] = sum_{s ∈ q_mask} aggr_k( top_k_t( Q[i,s] · D[j,t] ) )``

    with ``aggr = mean`` (default) or ``sum``. ``top_k=1, aggregation='sum'``
    is numerically identical to hard :func:`maxsim` (and its gradient).
    For ``top_k > 1`` the gradient is distributed across the ``K`` winning
    doc tokens per query token — denser signal than hard MaxSim without
    the O(Ld) cost of the LSE :func:`soft_maxsim` backward.

    Forward runs the Triton streaming top-K kernel on CUDA fp16/bf16
    inputs; otherwise falls back to the PyTorch reference (including
    gradcheck paths).

    Args:
        Q, D: 2-D or 3-D token embeddings.
        q_mask, d_mask: boolean masks (same convention as :func:`maxsim`).
        top_k: number of winners per query token. Clamped to ``Ld``.
        aggregation: ``'mean'`` (default) or ``'sum'``.
        normalize: L2-normalize Q / D per-token inside the kernel.

    Returns:
        ``scores [Nq, Nd]`` fp32, squeezed to match 2-D inputs.
    """
    if aggregation not in ("mean", "sum"):
        raise ValueError(f"aggregation must be 'mean' or 'sum'; got {aggregation!r}")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1; got {top_k}")

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

    if Q.shape[-1] != D.shape[-1]:
        raise ValueError(
            f"Q and D must share the embedding dim; got Q.shape[-1]={Q.shape[-1]} "
            f"vs D.shape[-1]={D.shape[-1]}."
        )
    if Q.device != D.device:
        raise ValueError(
            f"Q and D must be on the same device; got Q.device={Q.device} vs D.device={D.device}."
        )

    Q = Q.contiguous()
    D = D.contiguous()
    q_mask_c = q_mask.contiguous().to(torch.int8) if q_mask is not None else None
    d_mask_c = d_mask.contiguous().to(torch.int8) if d_mask is not None else None

    Ld = D.shape[1]
    effective_k = min(top_k, Ld)
    agg_scale = (1.0 / effective_k) if aggregation == "mean" else 1.0

    scores = _SmoothMaxSimFn.apply(Q, D, q_mask_c, d_mask_c, effective_k, agg_scale, normalize)

    if q_was_2d and d_was_2d:
        return scores.reshape(())
    if q_was_2d:
        return scores.squeeze(0)
    if d_was_2d:
        return scores.squeeze(-1)
    return scores


__all__ = ["smooth_maxsim", "smooth_maxsim_reference"]
