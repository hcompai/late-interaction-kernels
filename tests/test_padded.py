"""Tests for pack_padded and maxsim_padded."""

import pytest
import torch

from late_interaction_kernels.padded import maxsim_padded, pack_padded
from late_interaction_kernels.reference import maxsim_padded_reference

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(
    B: int = 3,
    C: int = 4,
    Lq: int = 16,
    Ld: int = 32,
    d: int = 64,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random padded batch with variable-length queries and documents."""
    torch.manual_seed(seed)
    queries = torch.randn(B, Lq, d, device=device, dtype=dtype)
    documents = torch.randn(B, C, Ld, d, device=device, dtype=dtype)
    qlen = torch.randint(1, Lq + 1, (B,), device=device, dtype=torch.int32)
    dlen = torch.randint(1, Ld + 1, (B, C), device=device, dtype=torch.int32)
    return queries, documents, qlen, dlen


# ---------------------------------------------------------------------------
# pack_padded — shape and structural correctness
# ---------------------------------------------------------------------------


def test_pack_padded_shapes() -> None:
    B, C, Lq, Ld, d = 3, 5, 16, 24, 64
    queries, documents, qlen, dlen = _make_batch(B=B, C=C, Lq=Lq, Ld=Ld, d=d)
    batch = pack_padded(queries, documents, qlen, dlen)

    total_q = int(qlen.sum().item())
    total_d = int(dlen.sum().item())

    assert batch.Q_packed.shape == (total_q, d)
    assert batch.D_packed.shape == (total_d, d)
    assert batch.cu_seqlens_q.shape == (B + 1,)
    assert batch.cu_seqlens_d.shape == (B * C + 1,)
    assert batch.pair_q_idx.shape == (B * C,)
    assert batch.pair_d_idx.shape == (B * C,)
    assert isinstance(batch.max_seqlen_q, int)
    assert batch.max_seqlen_q == int(qlen.max().item())


def test_pack_padded_cu_seqlens_start_at_zero() -> None:
    queries, documents, qlen, dlen = _make_batch()
    batch = pack_padded(queries, documents, qlen, dlen)

    assert int(batch.cu_seqlens_q[0].item()) == 0
    assert int(batch.cu_seqlens_d[0].item()) == 0


def test_pack_padded_cu_seqlens_end_at_total() -> None:
    queries, documents, qlen, dlen = _make_batch(B=4, C=3)
    batch = pack_padded(queries, documents, qlen, dlen)

    assert int(batch.cu_seqlens_q[-1].item()) == int(qlen.sum().item())
    assert int(batch.cu_seqlens_d[-1].item()) == int(dlen.sum().item())


def test_pack_padded_pair_indices_row_major() -> None:
    B, C = 3, 4
    queries, documents, qlen, dlen = _make_batch(B=B, C=C)
    batch = pack_padded(queries, documents, qlen, dlen)

    expected_q = torch.arange(B, dtype=torch.int32).repeat_interleave(C)
    expected_d = torch.arange(B * C, dtype=torch.int32)
    torch.testing.assert_close(batch.pair_q_idx.cpu(), expected_q)
    torch.testing.assert_close(batch.pair_d_idx.cpu(), expected_d)


def test_pack_padded_tokens_match_originals() -> None:
    """Tokens in the packed buffer must exactly match valid positions in the padded input."""
    B, C, Lq, Ld, d = 2, 3, 8, 12, 16
    queries, documents, qlen, dlen = _make_batch(B=B, C=C, Lq=Lq, Ld=Ld, d=d)
    batch = pack_padded(queries, documents, qlen, dlen)

    # Unpack Q and compare token-by-token.
    for b in range(B):
        q_start = int(batch.cu_seqlens_q[b].item())
        q_end = int(batch.cu_seqlens_q[b + 1].item())
        lq = int(qlen[b].item())
        torch.testing.assert_close(batch.Q_packed[q_start:q_end], queries[b, :lq])

    # Unpack D.
    for b in range(B):
        for c in range(C):
            flat = b * C + c
            d_start = int(batch.cu_seqlens_d[flat].item())
            d_end = int(batch.cu_seqlens_d[flat + 1].item())
            ld = int(dlen[b, c].item())
            torch.testing.assert_close(batch.D_packed[d_start:d_end], documents[b, c, :ld])


def test_pack_padded_ignores_pad_tokens() -> None:
    """Tokens beyond declared lengths must not appear in the packed buffer."""
    B, C, Lq, Ld, d = 2, 2, 8, 10, 16
    queries, documents, qlen, dlen = _make_batch(B=B, C=C, Lq=Lq, Ld=Ld, d=d)
    batch_clean = pack_padded(queries, documents, qlen, dlen)

    # Corrupt everything past declared lengths with large values.
    q_corrupt = queries.clone()
    d_corrupt = documents.clone()
    for b in range(B):
        q_corrupt[b, int(qlen[b].item()) :] = 1e6
        for c in range(C):
            d_corrupt[b, c, int(dlen[b, c].item()) :] = 1e6

    batch_corrupt = pack_padded(q_corrupt, d_corrupt, qlen, dlen)

    torch.testing.assert_close(batch_clean.Q_packed, batch_corrupt.Q_packed)
    torch.testing.assert_close(batch_clean.D_packed, batch_corrupt.D_packed)


def test_pack_padded_iter_unpacking() -> None:
    """PackedBatch can be unpacked as a 7-tuple via __iter__."""
    queries, documents, qlen, dlen = _make_batch()
    batch = pack_padded(queries, documents, qlen, dlen)
    Q, cu_q, D, cu_d, pq, pd, max_lq = batch
    assert Q is batch.Q_packed
    assert cu_q is batch.cu_seqlens_q
    assert D is batch.D_packed
    assert cu_d is batch.cu_seqlens_d
    assert pq is batch.pair_q_idx
    assert pd is batch.pair_d_idx
    assert max_lq == batch.max_seqlen_q


def test_pack_padded_validate_flag_catches_zero_length() -> None:
    B, C = 2, 2
    queries, documents, qlen, dlen = _make_batch(B=B, C=C)
    qlen_bad = qlen.clone()
    qlen_bad[0] = 0
    with pytest.raises(ValueError, match="query_lengths must all be > 0"):
        pack_padded(queries, documents, qlen_bad, dlen, validate=True)


# ---------------------------------------------------------------------------
# maxsim_padded — correctness vs reference (CPU, no CUDA required)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_maxsim_padded_matches_reference_cpu(dtype: torch.dtype) -> None:
    queries, documents, qlen, dlen = _make_batch(B=3, C=5, Lq=12, Ld=20, d=32, dtype=dtype)
    scores = maxsim_padded(queries, documents, qlen, dlen)
    expected = maxsim_padded_reference(queries, documents, qlen, dlen)

    assert scores.shape == expected.shape == (3, 5)
    assert scores.dtype == torch.float32

    tol = {"rtol": 1e-3, "atol": 1e-3} if dtype != torch.float32 else {"rtol": 1e-5, "atol": 1e-5}
    torch.testing.assert_close(scores, expected, **tol)


def test_maxsim_padded_ignores_pad_positions_cpu() -> None:
    """Garbage beyond declared lengths must not affect results."""
    B, C, Lq, Ld, d = 2, 3, 8, 16, 32
    queries, documents, qlen, dlen = _make_batch(B=B, C=C, Lq=Lq, Ld=Ld, d=d)

    baseline = maxsim_padded(queries, documents, qlen, dlen)

    q_corrupt = queries.clone()
    d_corrupt = documents.clone()
    for b in range(B):
        q_corrupt[b, int(qlen[b].item()) :] = 1e4
        for c in range(C):
            d_corrupt[b, c, int(dlen[b, c].item()) :] = -1e4

    out = maxsim_padded(q_corrupt, d_corrupt, qlen, dlen)
    torch.testing.assert_close(out, baseline, rtol=1e-5, atol=1e-5)


def test_maxsim_padded_single_token_queries_and_docs_cpu() -> None:
    """Each query/doc has exactly one token — MaxSim reduces to a dot product."""
    B, C, d = 2, 3, 16
    torch.manual_seed(7)
    queries = torch.randn(B, 1, d)
    documents = torch.randn(B, C, 1, d)
    qlen = torch.ones(B, dtype=torch.int32)
    dlen = torch.ones(B, C, dtype=torch.int32)

    scores = maxsim_padded(queries, documents, qlen, dlen)
    expected = maxsim_padded_reference(queries, documents, qlen, dlen)
    torch.testing.assert_close(scores, expected, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# maxsim_padded — CUDA parity (skipped without a GPU)
# ---------------------------------------------------------------------------


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_maxsim_padded_matches_reference_cuda(dtype: torch.dtype) -> None:
    queries, documents, qlen, dlen = _make_batch(B=4, C=8, Lq=32, Ld=128, d=64, device="cuda", dtype=dtype)
    scores = maxsim_padded(queries, documents, qlen, dlen)
    expected = maxsim_padded_reference(queries, documents, qlen, dlen)

    assert scores.device.type == "cuda"
    assert scores.shape == (4, 8)
    # Kernel accumulates in fp32; reference is fp32 throughout — tight tol.
    torch.testing.assert_close(scores, expected, rtol=5e-3, atol=5e-3)


@pytest.mark.cuda
def test_maxsim_padded_cuda_matches_cpu() -> None:
    """CUDA kernel output matches reference computed on CPU."""
    B, C, Lq, Ld, d = 2, 5, 24, 64, 64
    queries, documents, qlen, dlen = _make_batch(B=B, C=C, Lq=Lq, Ld=Ld, d=d, dtype=torch.float16)
    scores_cpu = maxsim_padded(queries, documents, qlen, dlen)
    scores_cuda = maxsim_padded(queries.cuda(), documents.cuda(), qlen.cuda(), dlen.cuda())
    torch.testing.assert_close(scores_cpu, scores_cuda.cpu(), rtol=5e-3, atol=5e-3)
