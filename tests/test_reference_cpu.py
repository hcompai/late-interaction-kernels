"""CPU-only sanity tests for the PyTorch reference implementations.

These tests don't exercise the Triton kernels — they validate that the
references themselves agree with known identities (e.g. PLAID residual
reconstruction round-trips through the bit packer). This lets us catch
logical bugs in the references long before we need a GPU.
"""

import pytest
import torch

from late_interaction_kernels.reference import (
    NEG_INF,
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


# --------------------------------------------------------------------------- #
# maxsim_reference mask handling                                              #
#                                                                             #
# maxsim_reference is the ground truth every Triton parity test compares      #
# against, yet its q_mask / d_mask handling is only exercised indirectly on   #
# GPU. These cross-check it against an independent brute-force loop on CPU so  #
# a regression in the reference itself is caught without a GPU.               #
# --------------------------------------------------------------------------- #


def _maxsim_bruteforce(
    Q: torch.Tensor,
    D: torch.Tensor,
    q_mask: torch.Tensor | None = None,
    d_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Triple-nested-loop MaxSim — slow but obviously correct.

    score[i, j] = sum over kept query tokens s of
                  max over kept doc tokens t of Q[i, s] . D[j, t]
    A query row with no kept doc token contributes 0 (not -inf).
    """
    Nq, Lq, _ = Q.shape
    Nd, Ld, _ = D.shape
    out = torch.zeros(Nq, Nd, dtype=torch.float32)
    for i in range(Nq):
        for j in range(Nd):
            total = 0.0
            for s in range(Lq):
                if q_mask is not None and not bool(q_mask[i, s]):
                    continue
                best = NEG_INF
                for t in range(Ld):
                    if d_mask is not None and not bool(d_mask[j, t]):
                        continue
                    best = max(best, float(Q[i, s] @ D[j, t]))
                total += best if best != NEG_INF else 0.0
            out[i, j] = total
    return out


def test_maxsim_reference_q_mask_matches_bruteforce():
    torch.manual_seed(0)
    Q = torch.randn(2, 5, 8)
    D = torch.randn(3, 7, 8)
    q_mask = torch.tensor([[1, 1, 0, 1, 0], [0, 1, 1, 1, 1]], dtype=torch.bool)
    got = maxsim_reference(Q, D, q_mask=q_mask)
    assert torch.allclose(got, _maxsim_bruteforce(Q, D, q_mask=q_mask), atol=1e-5)


def test_maxsim_reference_d_mask_matches_bruteforce():
    torch.manual_seed(1)
    Q = torch.randn(2, 5, 8)
    D = torch.randn(3, 7, 8)
    d_mask = torch.ones(3, 7, dtype=torch.bool)
    d_mask[0, 3:] = False
    d_mask[2, :2] = False
    got = maxsim_reference(Q, D, d_mask=d_mask)
    assert torch.allclose(got, _maxsim_bruteforce(Q, D, d_mask=d_mask), atol=1e-5)


def test_maxsim_reference_both_masks_matches_bruteforce():
    torch.manual_seed(2)
    Q = torch.randn(3, 6, 16)
    D = torch.randn(4, 9, 16)
    q_mask = torch.rand(3, 6) > 0.3
    d_mask = torch.rand(4, 9) > 0.3
    got = maxsim_reference(Q, D, q_mask=q_mask, d_mask=d_mask)
    assert torch.allclose(got, _maxsim_bruteforce(Q, D, q_mask=q_mask, d_mask=d_mask), atol=1e-5)


def test_maxsim_reference_fully_masked_query_row_is_zero():
    """A query whose tokens are all dropped scores 0 against every doc."""
    torch.manual_seed(3)
    Q = torch.randn(2, 4, 8)
    D = torch.randn(3, 5, 8)
    q_mask = torch.ones(2, 4, dtype=torch.bool)
    q_mask[1] = False
    got = maxsim_reference(Q, D, q_mask=q_mask)
    assert torch.all(got[1] == 0.0)
    assert torch.any(got[0] != 0.0)


def test_maxsim_reference_fully_masked_doc_is_zero():
    """A doc whose tokens are all masked from the max scores 0 for every query.

    This is the edge case the kernels guard with a -inf -> 0 clamp; the
    reference must agree (max over an empty set -> 0, never -inf).
    """
    torch.manual_seed(4)
    Q = torch.randn(2, 4, 8)
    D = torch.randn(3, 5, 8)
    d_mask = torch.ones(3, 5, dtype=torch.bool)
    d_mask[1] = False
    got = maxsim_reference(Q, D, d_mask=d_mask)
    assert torch.all(got[:, 1] == 0.0)
    assert torch.isfinite(got).all()


def test_maxsim_reference_1d_mask_broadcast_matches_2d():
    """A 1-D mask is promoted to a leading batch of 1 and broadcast."""
    torch.manual_seed(5)
    Q = torch.randn(1, 5, 8)
    D = torch.randn(1, 6, 8)
    q_mask_1d = torch.tensor([1, 0, 1, 1, 0], dtype=torch.bool)
    d_mask_1d = torch.tensor([1, 1, 0, 1, 0, 1], dtype=torch.bool)
    got_1d = maxsim_reference(Q, D, q_mask=q_mask_1d, d_mask=d_mask_1d)
    got_2d = maxsim_reference(Q, D, q_mask=q_mask_1d.unsqueeze(0), d_mask=d_mask_1d.unsqueeze(0))
    assert torch.allclose(got_1d, got_2d, atol=1e-6)


def test_maxsim_reference_normalize_with_masks_matches_bruteforce():
    """normalize=True must compose with masking (L2-normalize, then MaxSim)."""
    torch.manual_seed(6)
    Q = torch.randn(2, 5, 8)
    D = torch.randn(3, 7, 8)
    q_mask = torch.rand(2, 5) > 0.3
    d_mask = torch.rand(3, 7) > 0.3
    Qn = torch.nn.functional.normalize(Q, p=2, dim=-1, eps=1e-12)
    Dn = torch.nn.functional.normalize(D, p=2, dim=-1, eps=1e-12)
    got = maxsim_reference(Q, D, q_mask=q_mask, d_mask=d_mask, normalize=True)
    expected = _maxsim_bruteforce(Qn, Dn, q_mask=q_mask, d_mask=d_mask)
    assert torch.allclose(got, expected, atol=1e-5)
