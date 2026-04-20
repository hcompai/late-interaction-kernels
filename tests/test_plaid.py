"""Parity tests for the PLAID-style kernels (approximate + residual rerank)."""

from __future__ import annotations

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
    from late_interaction_kernels import plaid_approx_score
    from late_interaction_kernels.plaid import plaid_approx_score_reference

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
    from late_interaction_kernels import plaid_approx_score

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
    from late_interaction_kernels import maxsim_residual
    from late_interaction_kernels.plaid import maxsim_residual_reference

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
    from late_interaction_kernels import maxsim_residual
    from late_interaction_kernels.plaid import maxsim_residual_reference

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


def test_residual_matches_dense_maxsim():
    """With nbits=8 and identity bucket weights, residual scoring should
    recover exact dense MaxSim up to rounding."""
    from late_interaction_kernels import maxsim, maxsim_residual

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
