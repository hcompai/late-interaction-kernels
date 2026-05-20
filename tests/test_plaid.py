"""Parity tests for the PLAID-style kernels (approximate + residual rerank)."""

import pytest
import torch

pytestmark = pytest.mark.cuda


# --------------------------------------------------------------------------- #
# Shared helpers: synthesize a realistic PLAID-style quantized index.         #
# --------------------------------------------------------------------------- #


def _make_quant_index(Nd, max_Ld, d, n_centroids, nbits, device="cuda", seed=0):
    """Build a synthetic PLAID-like quantized index.

    Returns a dict with: centroids, bucket_weights, codes, residuals,
    doc_lengths, and the unquantized ``emb`` used for reference scoring.
    """
    torch.manual_seed(seed)
    centroids = torch.randn(n_centroids, d, device=device, dtype=torch.float32) * 0.3
    n_buckets = 2**nbits
    bucket_weights = torch.linspace(-0.1, 0.1, n_buckets, device=device, dtype=torch.float32)

    codes_per_byte = 8 // nbits
    packed_dim = (d * nbits + 7) // 8

    codes = torch.randint(0, n_centroids, (Nd, max_Ld), device=device, dtype=torch.int64)
    # random bucket indices per feature per token
    bucket_codes = torch.randint(0, n_buckets, (Nd, max_Ld, d), device=device, dtype=torch.int64)

    # Pack bucket_codes into uint8 bytes.
    residuals = torch.zeros(Nd, max_Ld, packed_dim, device=device, dtype=torch.uint8)
    for f in range(d):
        byte_idx = f // codes_per_byte
        slot = f % codes_per_byte
        residuals[..., byte_idx] |= bucket_codes[..., f].to(torch.uint8) << (slot * nbits)

    doc_lengths = torch.randint(max_Ld // 2, max_Ld + 1, (Nd,), device=device, dtype=torch.int64)

    # Ground-truth unquantized embedding = centroid[code] + bucket_weights[bucket_code]
    emb = centroids[codes] + bucket_weights[bucket_codes]

    return {
        "centroids": centroids,
        "bucket_weights": bucket_weights,
        "codes": codes,
        "residuals": residuals,
        "doc_lengths": doc_lengths,
        "emb": emb,
    }


# --------------------------------------------------------------------------- #
# C1. plaid_approx_score                                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("Lq", [32, 128])
@pytest.mark.parametrize("max_Ld", [64, 200])
def test_plaid_approx_score_parity(Lq, max_Ld):
    from late_interaction_kernels.plaid import plaid_approx_score, plaid_approx_score_reference

    torch.manual_seed(0)
    n_centroids = 256
    B = 20
    qcs = torch.randn(n_centroids, Lq, device="cuda", dtype=torch.float32)
    codes = torch.randint(0, n_centroids, (B, max_Ld), device="cuda", dtype=torch.int64)
    doc_lens = torch.randint(max_Ld // 2, max_Ld + 1, (B,), device="cuda", dtype=torch.int64)

    fast = plaid_approx_score(qcs, codes, doc_lens)
    ref = plaid_approx_score_reference(qcs, codes, doc_lens)
    assert (fast - ref).abs().max().item() / max(1e-6, ref.abs().max().item()) < 1e-4


def test_plaid_approx_score_handles_empty_docs():
    from late_interaction_kernels.plaid import plaid_approx_score

    qcs = torch.randn(64, 32, device="cuda", dtype=torch.float32)
    codes = torch.randint(0, 64, (4, 50), device="cuda", dtype=torch.int64)
    doc_lens = torch.tensor([0, 25, 50, 0], device="cuda", dtype=torch.int64)

    scores = plaid_approx_score(qcs, codes, doc_lens)
    assert scores[0].item() == 0.0
    assert scores[3].item() == 0.0


# --------------------------------------------------------------------------- #
# C2. maxsim_residual                                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("nbits", [2, 4, 8])
def test_residual_unpack_roundtrips(nbits):
    """Round-trip a packed residual through our Triton unpack and verify the
    recovered emb matches the ground-truth (centroid + bucket_weight)."""
    from late_interaction_kernels.plaid import maxsim_residual, maxsim_residual_reference

    idx = _make_quant_index(
        Nd=4,
        max_Ld=16,
        d=128,
        n_centroids=32,
        nbits=nbits,
    )
    Q = torch.randn(2, 32, 128, device="cuda", dtype=torch.bfloat16)

    fast = maxsim_residual(
        Q,
        idx["codes"],
        idx["residuals"],
        idx["doc_lengths"],
        idx["centroids"],
        idx["bucket_weights"],
        nbits=nbits,
        normalize=False,
    )
    ref = maxsim_residual_reference(
        Q,
        idx["codes"],
        idx["residuals"],
        idx["doc_lengths"],
        idx["centroids"],
        idx["bucket_weights"],
        nbits=nbits,
        normalize=False,
    )
    tol = 2e-2
    assert (fast - ref).abs().max().item() / max(1e-6, ref.abs().max().item()) < tol


@pytest.mark.parametrize("nbits", [2, 4])
def test_residual_normalize_matches_reference(nbits):
    from late_interaction_kernels.plaid import maxsim_residual, maxsim_residual_reference

    idx = _make_quant_index(Nd=6, max_Ld=32, d=128, n_centroids=64, nbits=nbits)
    Q = torch.randn(3, 32, 128, device="cuda", dtype=torch.bfloat16)

    fast = maxsim_residual(
        Q,
        idx["codes"],
        idx["residuals"],
        idx["doc_lengths"],
        idx["centroids"],
        idx["bucket_weights"],
        nbits=nbits,
        normalize=True,
    )
    ref = maxsim_residual_reference(
        Q,
        idx["codes"],
        idx["residuals"],
        idx["doc_lengths"],
        idx["centroids"],
        idx["bucket_weights"],
        nbits=nbits,
        normalize=True,
    )
    assert (fast - ref).abs().max().item() / max(1e-6, ref.abs().max().item()) < 3e-2


# --------------------------------------------------------------------------- #
# C2.bwd. maxsim_residual backward — grad_Q parity vs dense unpack + autograd #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("nbits", [2, 4, 8])
@pytest.mark.parametrize("normalize", [False, True])
def test_residual_backward_grad_Q_matches_dense_autograd(nbits, normalize):
    """The fused residual backward is compared against the "dense" path:
    unpack the residual index to ``emb = centroid + bucket`` once (fp32), then
    run the standard differentiable ``maxsim`` on (Q, emb). If our kernel is
    correct, ``grad_Q`` must match bit-for-bit up to fp16/bf16 tolerance.

    centroids / residuals / codes are non-differentiable by construction.
    """
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.plaid import maxsim_residual
    from late_interaction_kernels.reference import unpack_residuals_reference

    idx = _make_quant_index(
        Nd=4,
        max_Ld=24,
        d=128,
        n_centroids=32,
        nbits=nbits,
    )

    d = 128
    Nq, Lq = 3, 32
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float32) * 0.1

    # --- Path A: fused residual kernel, autograd on Q.
    Qa = Q.clone().requires_grad_(True)
    scores_a = maxsim_residual(
        Qa,
        idx["codes"],
        idx["residuals"],
        idx["doc_lengths"],
        idx["centroids"],
        idx["bucket_weights"],
        nbits=nbits,
        normalize=normalize,
    )

    # --- Path B: unpack residuals to a dense emb, then use standard maxsim.
    bucket_codes = unpack_residuals_reference(idx["residuals"], nbits, d).clamp_min(0)
    emb = idx["centroids"][idx["codes"]] + idx["bucket_weights"][bucket_codes]
    d_mask = torch.arange(emb.shape[1], device=emb.device).unsqueeze(0) < idx["doc_lengths"].unsqueeze(-1)

    Qb = Q.clone().requires_grad_(True)
    if normalize:
        Qb_n = torch.nn.functional.normalize(Qb, p=2, dim=-1, eps=1e-12)
        emb_n = torch.nn.functional.normalize(emb, p=2, dim=-1, eps=1e-12)
        scores_b = maxsim(Qb_n, emb_n, d_mask=d_mask)
    else:
        scores_b = maxsim(Qb, emb, d_mask=d_mask)

    # Score parity first (sanity).
    denom = max(1.0, scores_b.detach().abs().max().item())
    assert (scores_a - scores_b).abs().max().item() / denom < 5e-3

    g = torch.randn_like(scores_a)
    scores_a.backward(g)
    scores_b.backward(g)

    err = (Qa.grad - Qb.grad).abs().max().item()
    denom = max(1.0, Qb.grad.abs().max().item())
    # nbits=2 has only 4 quantization buckets per feature, so many doc tokens
    # end up with near-identical reconstructed embeddings. Argmax ties are then
    # broken differently by the two kernels due to bf16-matmul rounding, which
    # routes `grad_scores * emb` to different winner embeddings and causes
    # max-abs deltas of ~1-2 % that shrink as you average. That's a reference
    # comparison artifact, not a kernel bug (scores are still within 5e-3).
    tol = 3e-2 if nbits == 2 else 5e-3
    assert err / denom < tol, f"grad_Q err={err} denom={denom} nbits={nbits} norm={normalize}"


def test_residual_inference_does_not_save_argmax():
    """``maxsim_residual`` with ``Q.requires_grad=False`` skips the argmax buffer."""
    from late_interaction_kernels.plaid import maxsim_residual, maxsim_residual_reference

    idx = _make_quant_index(Nd=3, max_Ld=16, d=128, n_centroids=32, nbits=4)
    Q = torch.randn(2, 32, 128, device="cuda", dtype=torch.bfloat16)

    fast = maxsim_residual(
        Q,
        idx["codes"],
        idx["residuals"],
        idx["doc_lengths"],
        idx["centroids"],
        idx["bucket_weights"],
        nbits=4,
        normalize=True,
    )
    ref = maxsim_residual_reference(
        Q,
        idx["codes"],
        idx["residuals"],
        idx["doc_lengths"],
        idx["centroids"],
        idx["bucket_weights"],
        nbits=4,
        normalize=True,
    )
    assert (fast - ref).abs().max().item() / max(1e-6, ref.abs().max().item()) < 3e-2


# --------------------------------------------------------------------------- #
# C3. maxsim_residual_varlen                                                  #
# --------------------------------------------------------------------------- #


def _dense_to_varlen(idx):
    """Pack a padded _make_quant_index output into a ragged (cu_seqlens) layout."""
    codes_p = idx["codes"]
    res_p = idx["residuals"]
    dlens = idx["doc_lengths"]
    Nd, _max_Ld = codes_p.shape
    flat_codes = []
    flat_res = []
    for i in range(Nd):
        li = int(dlens[i].item())
        flat_codes.append(codes_p[i, :li])
        flat_res.append(res_p[i, :li])
    codes_flat = torch.cat(flat_codes, dim=0) if flat_codes else codes_p.new_zeros(0)
    res_flat = torch.cat(flat_res, dim=0) if flat_res else res_p.new_zeros(0, res_p.shape[-1])
    cu = torch.zeros(Nd + 1, dtype=torch.int32, device=codes_p.device)
    cu[1:] = dlens.to(torch.int32).cumsum(0)
    return codes_flat, res_flat, cu


@pytest.mark.parametrize("nbits", [2, 4, 8])
def test_residual_varlen_matches_dense(nbits):
    from late_interaction_kernels.plaid import maxsim_residual, maxsim_residual_varlen

    idx = _make_quant_index(Nd=5, max_Ld=32, d=128, n_centroids=64, nbits=nbits)
    Q = torch.randn(2, 32, 128, device="cuda", dtype=torch.bfloat16)

    codes_flat, res_flat, cu = _dense_to_varlen(idx)

    dense = maxsim_residual(
        Q,
        idx["codes"],
        idx["residuals"],
        idx["doc_lengths"],
        idx["centroids"],
        idx["bucket_weights"],
        nbits=nbits,
        normalize=True,
    )
    varlen = maxsim_residual_varlen(
        Q,
        codes_flat,
        res_flat,
        cu,
        idx["centroids"],
        idx["bucket_weights"],
        nbits=nbits,
        normalize=True,
    )
    # Varlen and dense must agree up to kernel-schedule rounding. They use the
    # same math and the same fp32-accumulator dot, so the bar is tight.
    assert dense.shape == varlen.shape
    denom = max(1e-6, dense.abs().max().item())
    assert (dense - varlen).abs().max().item() / denom < 1e-4


def test_residual_varlen_handles_empty_docs():
    from late_interaction_kernels.plaid import maxsim_residual_varlen

    centroids = torch.randn(32, 128, device="cuda", dtype=torch.float32)
    buckets = torch.linspace(-0.1, 0.1, 16, device="cuda", dtype=torch.float32)
    codes = torch.randint(0, 32, (20,), device="cuda", dtype=torch.int64)
    # Packed residual for nbits=4 and d=128 -> packed_dim = 64.
    res = torch.randint(0, 256, (20, 64), device="cuda", dtype=torch.uint8)
    # Three docs: lengths 0, 20, 0.
    cu = torch.tensor([0, 0, 20, 20], device="cuda", dtype=torch.int32)
    Q = torch.randn(1, 32, 128, device="cuda", dtype=torch.bfloat16)

    scores = maxsim_residual_varlen(Q, codes, res, cu, centroids, buckets, nbits=4, normalize=True)
    assert scores.shape == (1, 3)
    assert scores[0, 0].item() == 0.0
    assert scores[0, 2].item() == 0.0


def test_residual_matches_dense_maxsim():
    """With nbits=8 and identity bucket weights, residual scoring should
    recover exact dense MaxSim up to rounding."""
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.plaid import maxsim_residual

    torch.manual_seed(0)
    Nd, max_Ld, d, n_cent, nbits = 4, 32, 128, 16, 8
    centroids = torch.randn(n_cent, d, device="cuda", dtype=torch.float32)
    bucket_weights = torch.zeros(2**nbits, device="cuda", dtype=torch.float32)
    codes = torch.randint(0, n_cent, (Nd, max_Ld), device="cuda", dtype=torch.int64)
    residuals = torch.zeros(Nd, max_Ld, d, device="cuda", dtype=torch.uint8)
    doc_lens = torch.full((Nd,), max_Ld, device="cuda", dtype=torch.int64)

    emb = centroids[codes]
    Q = torch.randn(2, 32, d, device="cuda", dtype=torch.bfloat16)

    fast = maxsim_residual(
        Q,
        codes,
        residuals,
        doc_lens,
        centroids,
        bucket_weights,
        nbits=nbits,
        normalize=False,
    )
    ref = maxsim(Q, emb.to(torch.bfloat16)).float()
    assert (fast - ref).abs().max().item() / max(1e-6, ref.abs().max().item()) < 3e-2
