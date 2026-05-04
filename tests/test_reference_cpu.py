"""CPU-only sanity tests for the PyTorch reference implementations.

These tests don't exercise the Triton kernels — they validate that the
references themselves agree with known identities (e.g. XTR top_k=1 equals
plain MaxSim, PLAID residual reconstruction round-trips through the bit
packer). This lets us catch logical bugs in the references long before we
need a GPU.
"""

import pytest
import torch

from late_interaction_kernels.reference import (
    maxsim_reference,
    maxsim_residual_reference,
    plaid_approx_score_reference,
    unpack_residuals_reference,
    xtr_reference,
)

# --------------------------------------------------------------------------- #
# XTR                                                                         #
# --------------------------------------------------------------------------- #


def test_xtr_topk1_matches_maxsim():
    torch.manual_seed(0)
    Q = torch.randn(2, 8, 32)
    D = torch.randn(3, 20, 32)
    maxsim_score = maxsim_reference(Q, D)
    xtr_score = xtr_reference(Q, D, top_k=1, normalize_by_k=False)
    assert torch.allclose(maxsim_score, xtr_score, atol=1e-5)


def test_xtr_topk_increases_monotonically():
    torch.manual_seed(0)
    Q = torch.randn(1, 4, 16)
    D = torch.randn(1, 12, 16)
    s1 = xtr_reference(Q, D, top_k=1, normalize_by_k=False).item()
    s3 = xtr_reference(Q, D, top_k=3, normalize_by_k=False).item()
    s6 = xtr_reference(Q, D, top_k=6, normalize_by_k=False).item()
    # Sum of top-k grows with k (scores can be negative, but the argmax is
    # always included in larger k sets).
    assert s3 >= s1 - 1e-5
    assert s6 >= s3 - 1e-5


def test_xtr_respects_doc_mask():
    torch.manual_seed(0)
    Q = torch.randn(1, 4, 16)
    D = torch.randn(1, 12, 16)
    mask = torch.ones(1, 12, dtype=torch.bool)
    mask[0, 8:] = False

    D_trunc = D[:, :8]
    full = xtr_reference(Q, D, top_k=2, d_mask=mask)
    truncated = xtr_reference(Q, D_trunc, top_k=2)
    assert torch.allclose(full, truncated, atol=1e-5)


# --------------------------------------------------------------------------- #
# PLAID approx score                                                          #
# --------------------------------------------------------------------------- #


def test_plaid_approx_empty_docs_zero():
    torch.manual_seed(0)
    qcs = torch.randn(64, 16)
    codes = torch.randint(0, 64, (3, 10))
    lens = torch.tensor([0, 5, 0])
    out = plaid_approx_score_reference(qcs, codes, lens)
    assert out[0].item() == 0.0
    assert out[2].item() == 0.0
    assert out[1].item() != 0.0  # non-empty doc should produce something


def test_plaid_approx_masking_consistent():
    """Padding beyond doc_length must not affect the score."""
    torch.manual_seed(0)
    qcs = torch.randn(32, 8)
    codes_short = torch.randint(0, 32, (2, 5))
    # Pad with arbitrary garbage; real doc length stays 5.
    codes_padded = torch.cat([codes_short, torch.randint(0, 32, (2, 10))], dim=1)
    lens = torch.tensor([5, 5])

    full = plaid_approx_score_reference(qcs, codes_short, torch.tensor([5, 5]))
    padded = plaid_approx_score_reference(qcs, codes_padded, lens)
    assert torch.allclose(full, padded, atol=1e-5)


# --------------------------------------------------------------------------- #
# Residual bit-pack round-trip                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("nbits", [2, 4, 8])
def test_unpack_residuals_roundtrip(nbits):
    """Pack random bucket codes, unpack with the reference, recover exactly."""
    torch.manual_seed(0)
    Nd, Ld, d = 3, 8, 64
    codes_per_byte = 8 // nbits
    packed_dim = (d * nbits + 7) // 8
    n_buckets = 2**nbits

    bucket_codes = torch.randint(0, n_buckets, (Nd, Ld, d), dtype=torch.int64)
    residuals = torch.zeros(Nd, Ld, packed_dim, dtype=torch.uint8)
    for f in range(d):
        byte_idx = f // codes_per_byte
        slot = f % codes_per_byte
        residuals[..., byte_idx] |= bucket_codes[..., f].to(torch.uint8) << (slot * nbits)

    recovered = unpack_residuals_reference(residuals, nbits, d).long()
    assert torch.equal(recovered, bucket_codes)


# --------------------------------------------------------------------------- #
# Residual MaxSim reference: with nbits=8 and identity buckets,               #
# residual scoring should recover the plain MaxSim of centroid[codes].        #
# --------------------------------------------------------------------------- #


def test_maxsim_residual_matches_dense_maxsim_identity_buckets():
    torch.manual_seed(0)
    Nd, Ld, d, n_cent = 3, 10, 64, 16
    nbits = 8
    centroids = torch.randn(n_cent, d)
    bucket_weights = torch.zeros(2**nbits)
    codes = torch.randint(0, n_cent, (Nd, Ld), dtype=torch.int64)
    residuals = torch.zeros(Nd, Ld, d, dtype=torch.uint8)
    doc_lens = torch.full((Nd,), Ld, dtype=torch.int64)
    Q = torch.randn(2, 6, d)

    scored = maxsim_residual_reference(
        Q,
        codes,
        residuals,
        doc_lens,
        centroids,
        bucket_weights,
        nbits=nbits,
        normalize=False,
    )
    emb = centroids[codes]
    direct = maxsim_reference(Q, emb)
    assert torch.allclose(scored, direct, atol=1e-4, rtol=1e-4)
