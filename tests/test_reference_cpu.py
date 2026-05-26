"""CPU-only sanity tests for the PyTorch reference implementations.

These tests don't exercise the Triton kernels — they validate that the
references themselves agree with known identities (e.g. PLAID residual
reconstruction round-trips through the bit packer). This lets us catch
logical bugs in the references long before we need a GPU.
"""

import pytest
import torch

from late_interaction_kernels.reference import (
    maxsim_reference,
    maxsim_residual_reference,
    plaid_approx_score_reference,
    unpack_residuals_reference,
)

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
