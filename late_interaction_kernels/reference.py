"""Pure-PyTorch reference implementations of MaxSim.

These are the ground truth against which all Triton kernels are validated.
They are slow and memory-hungry by design — they are not meant to be fast.

Shapes (following PyLate conventions):
    Q: [Nq, Lq, d]   query-side token embeddings
    D: [Nd, Ld, d]   document-side token embeddings
    q_mask: [Nq, Lq]  bool — False positions are ignored on the SUM side
    d_mask: [Nd, Ld]  bool — False positions are ignored on the MAX side
                       (they get -inf before the max reduction)

The scalar score formula is:

    score[i, j] = sum_{s : q_mask[i, s]} max_{t : d_mask[j, t]} Q[i, s] @ D[j, t]

For symmetry with PyLate's `colbert_scores` we return a [Nq, Nd] matrix.
If either tensor is 2-D we reshape to 3-D with a leading 1 and squeeze back.
"""

import torch

NEG_INF = float("-inf")


def maxsim_reference(
    Q: torch.Tensor,
    D: torch.Tensor,
    q_mask: torch.Tensor | None = None,
    d_mask: torch.Tensor | None = None,
    *,
    normalize: bool = False,
) -> torch.Tensor:
    """Dense reference MaxSim with full mask support.

    Fully materializes the [Nq, Nd, Lq, Ld] similarity tensor. Correct, but slow.

    Args:
        Q: [Nq, Lq, d] or [Lq, d] query embeddings.
        D: [Nd, Ld, d] or [Ld, d] document embeddings.
        q_mask: [Nq, Lq] or [Lq] boolean mask. True = keep, False = drop from sum.
        d_mask: [Nd, Ld] or [Ld] boolean mask. True = keep, False = mask out of max.
        normalize: if True, L2-normalize Q and D per-token before the einsum
            (equivalent to ``F.normalize(.., p=2, dim=-1)``).

    Returns:
        scores: [Nq, Nd] float tensor. If inputs were 2-D the matching dim is
                squeezed out.
    """
    q_squeeze = Q.dim() == 2
    d_squeeze = D.dim() == 2
    if q_squeeze:
        Q = Q.unsqueeze(0)
    if d_squeeze:
        D = D.unsqueeze(0)

    Nq, Lq, d = Q.shape
    Nd, Ld, d2 = D.shape
    if d != d2:
        raise ValueError(f"embedding dims don't match: Q has d={d}, D has d={d2}")

    # Preserve fp64 for `torch.autograd.gradcheck`; only promote narrow
    # floats (fp16 / bf16) up to fp32 for a numerically safe reference.
    if Q.dtype in (torch.float16, torch.bfloat16):
        Qf = Q.float()
    else:
        Qf = Q
    if D.dtype in (torch.float16, torch.bfloat16):
        Df = D.float()
    else:
        Df = D
    if normalize:
        Qf = torch.nn.functional.normalize(Qf, p=2, dim=-1, eps=1e-12)
        Df = torch.nn.functional.normalize(Df, p=2, dim=-1, eps=1e-12)
    S = torch.einsum("ild,jtd->ijlt", Qf, Df)  # [Nq, Nd, Lq, Ld]

    if d_mask is not None:
        if d_mask.dim() == 1:
            d_mask = d_mask.unsqueeze(0)
        # -inf on masked doc tokens so they lose the max
        S = S.masked_fill(~d_mask.bool()[None, :, None, :], NEG_INF)

    # If an entire row is -inf (whole doc masked) max() would give -inf; clamp to 0.
    row_max = S.max(dim=-1).values  # [Nq, Nd, Lq]
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))

    if q_mask is not None:
        if q_mask.dim() == 1:
            q_mask = q_mask.unsqueeze(0)
        row_max = row_max * q_mask.to(row_max.dtype)[:, None, :]

    scores = row_max.sum(dim=-1)  # [Nq, Nd]

    if q_squeeze:
        scores = scores.squeeze(0)
    if d_squeeze:
        scores = scores.squeeze(-1)
    return scores


def plaid_approx_score_reference(
    query_centroid_scores: torch.Tensor,
    codes: torch.Tensor,
    doc_lengths: torch.Tensor,
) -> torch.Tensor:
    """Dense-gather PyTorch reference for PLAID-style approximate scoring.

    Mirrors the ColBERTv2 ``index_select(query_centroid_scores) -> pad ->
    colbert_score_reduce`` pipeline but uses a dense masked reduction so it
    runs on any device, including CPU.
    """
    qcs = query_centroid_scores.float()
    codes = codes.long()
    B, max_Ld = codes.shape
    flat = qcs[codes.reshape(-1)].reshape(B, max_Ld, -1)  # [B, max_Ld, Lq]
    pos = torch.arange(max_Ld, device=codes.device).unsqueeze(0)
    mask = pos < doc_lengths.to(codes.device).unsqueeze(-1)
    flat = flat.masked_fill(~mask.unsqueeze(-1), NEG_INF)
    m = flat.max(dim=1).values
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    return m.sum(dim=-1)


def unpack_residuals_reference(residuals: torch.Tensor, nbits: int, d: int) -> torch.Tensor:
    """Dense CPU/GPU reference for PLAID residual bit-unpacking.

    Each byte of ``residuals`` holds ``8 / nbits`` bucket codes, little-endian
    within the byte. The output is ``[..., d]`` int32 bucket indices.
    """
    codes_per_byte = 8 // nbits
    mask = (1 << nbits) - 1
    rs = residuals.to(torch.int32)
    feats = []
    for f in range(d):
        byte_idx = f // codes_per_byte
        slot = f % codes_per_byte
        val = (rs[..., byte_idx] >> (slot * nbits)) & mask
        feats.append(val)
    return torch.stack(feats, dim=-1)


def maxsim_residual_reference(
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
    """PyTorch reference for the PLAID residual decompress + MaxSim kernel."""
    if Q.dim() == 2:
        Q = Q.unsqueeze(0)
    _, _, d = Q.shape
    _, max_Ld = codes.shape
    centroids_f = centroids.float()
    buckets = bucket_weights.float()

    bucket_codes = unpack_residuals_reference(residuals, nbits, d)
    bc = bucket_codes.clamp(min=0, max=buckets.numel() - 1).long()
    bucket_vals = buckets[bc]
    emb = centroids_f[codes.long()] + bucket_vals

    if normalize:
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1, eps=1e-12)
        Qn = torch.nn.functional.normalize(Q.float(), p=2, dim=-1, eps=1e-12)
    else:
        Qn = Q.float()

    pos = torch.arange(max_Ld, device=codes.device).unsqueeze(0)
    d_mask = pos < doc_lengths.to(codes.device).unsqueeze(-1)

    S = torch.einsum("ild,jtd->ijlt", Qn, emb)
    S = S.masked_fill(~d_mask.bool()[None, :, None, :], NEG_INF)
    m = S.max(dim=-1).values
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    return m.sum(dim=-1)


def maxsim_from_hidden_reference(
    Q: torch.Tensor,
    H_d: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor | None = None,
    d_mask: torch.Tensor | None = None,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Reference for ``maxsim_from_hidden``: project D on-the-fly, MaxSim against Q.

    Materializes ``D_proj`` in fp32 — that is the whole point of the fused
    kernel *not* existing here — but this gives us the ground truth for
    parity tests.
    """
    q_squeeze = Q.dim() == 2
    if q_squeeze:
        Q = Q.unsqueeze(0)
    if d_mask is not None and d_mask.dim() == 1:
        d_mask = d_mask.unsqueeze(0)

    D_proj = torch.nn.functional.linear(H_d.float(), W.float(), b.float() if b is not None else None)
    if normalize:
        D_proj = torch.nn.functional.normalize(D_proj, p=2, dim=-1, eps=1e-12)

    scores = maxsim_reference(Q, D_proj, d_mask=d_mask)
    if q_squeeze:
        scores = scores.squeeze(0)
    return scores


def maxsim_reference_varlen(
    Q: torch.Tensor,
    D: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_d: torch.Tensor,
) -> torch.Tensor:
    """Packed/varlen reference: Q/D are 2-D tensors of stacked token embeddings.

    Args:
        Q: [total_q_tokens, d]
        D: [total_d_tokens, d]
        cu_seqlens_q: [Nq + 1] cumulative query token counts (int32).
        cu_seqlens_d: [Nd + 1] cumulative doc   token counts (int32).

    Returns:
        scores: [Nq, Nd]
    """
    Nq = cu_seqlens_q.numel() - 1
    Nd = cu_seqlens_d.numel() - 1
    out = torch.empty(Nq, Nd, device=Q.device, dtype=torch.float32)
    for i in range(Nq):
        qi = Q[cu_seqlens_q[i].item() : cu_seqlens_q[i + 1].item()].float()
        if qi.shape[0] == 0:
            out[i].zero_()
            continue
        for j in range(Nd):
            dj = D[cu_seqlens_d[j].item() : cu_seqlens_d[j + 1].item()].float()
            if dj.shape[0] == 0:
                out[i, j] = 0.0
                continue
            S = qi @ dj.T  # [lq, ld]
            out[i, j] = S.max(dim=-1).values.sum()
    return out


def maxsim_reference_scatter(
    Q_packed: torch.Tensor,
    D_packed: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_d: torch.Tensor,
    pair_q_idx: torch.Tensor,
    pair_d_idx: torch.Tensor,
) -> torch.Tensor:
    """Pair-list reference for ``score_pairs_packed``: scores ``[num_pairs]``."""
    num_pairs = pair_q_idx.numel()
    out = torch.empty(num_pairs, device=Q_packed.device, dtype=torch.float32)
    for k in range(num_pairs):
        i = int(pair_q_idx[k].item())
        j = int(pair_d_idx[k].item())
        qi = Q_packed[cu_seqlens_q[i].item() : cu_seqlens_q[i + 1].item()].float()
        dj = D_packed[cu_seqlens_d[j].item() : cu_seqlens_d[j + 1].item()].float()
        if qi.shape[0] == 0 or dj.shape[0] == 0:
            out[k] = 0.0
            continue
        S = qi @ dj.T
        out[k] = S.max(dim=-1).values.sum()
    return out


def maxsim_padded_reference(
    queries: torch.Tensor,
    documents: torch.Tensor,
    query_lengths: torch.Tensor,
    doc_lengths: torch.Tensor,
) -> torch.Tensor:
    """Reference for :func:`maxsim_padded`.

    Loops over ``(b, c)`` pairs and calls :func:`maxsim_reference` for each.
    Always returns fp32.

    Args:
        queries: ``[B, Lq, d]``
        documents: ``[B, C, Ld, d]``
        query_lengths: ``[B]``
        doc_lengths: ``[B, C]``

    Returns:
        ``[B, C]`` fp32 tensor on the same device as ``queries``.
    """
    B, _, _ = queries.shape
    _, C, _, _ = documents.shape
    qlen = query_lengths.to(torch.int64).cpu().tolist()
    dlen = doc_lengths.to(torch.int64).cpu().tolist()
    out = torch.empty((B, C), dtype=torch.float32, device=queries.device)
    for b in range(B):
        q = queries[b, : qlen[b]].float()
        for c in range(C):
            d = documents[b, c, : dlen[b][c]].float()
            if q.shape[0] == 0 or d.shape[0] == 0:
                out[b, c] = 0.0
                continue
            S = q @ d.T  # [lq, ld]
            out[b, c] = S.max(dim=-1).values.sum()
    return out
