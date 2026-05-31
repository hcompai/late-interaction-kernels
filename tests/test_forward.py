"""Forward-pass numerical parity against the pure-PyTorch reference.

These are the first-line correctness tests: they pin the kernel's output to
within the expected tensor-core ULP drift of the reference implementation
across the realistic shape grid for ColBERT (text), ColPali (visual), and
ModernColBERT (long document) workloads.
"""

import pytest
import torch

from tests.conftest import needs_large_smem

pytestmark = pytest.mark.cuda


# (Nq, Nd, Lq, Ld, d) — representative ColBERT / ColPali / ModernColBERT shapes.
SHAPES = [
    (1, 4, 32, 64, 128),
    (8, 16, 32, 128, 128),
    (16, 8, 32, 300, 128),
    (4, 4, 128, 1024, 128),  # long doc
    (2, 2, 256, 256, 128),  # long-seq parity (shrunk from 1024² for CI speed)
    (4, 8, 32, 200, 256),
    (2, 4, 32, 128, 512),
    (2, 2, 32, 128, 1024),  # large d
    (1, 1, 3, 5, 128),  # tiny edge
]
SHAPE_IDS = [f"Nq{s[0]}_Nd{s[1]}_Lq{s[2]}_Ld{s[3]}_d{s[4]}" for s in SHAPES]


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_forward_parity_no_mask(shape, dtype, rel):
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.reference import maxsim_reference

    Nq, Nd, Lq, Ld, d = shape
    if needs_large_smem(d):
        pytest.skip(f"d={d} overflows sm_75 shared memory; runs on sm_80+")
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=dtype)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=dtype)

    fast = maxsim(Q, D).float()
    ref = maxsim_reference(Q.float(), D.float())

    tol = 5e-3 if dtype == torch.float16 else 2e-2
    assert rel(fast, ref) < tol


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_forward_parity_with_masks(shape, rel):
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.reference import maxsim_reference

    Nq, Nd, Lq, Ld, d = shape
    if needs_large_smem(d):
        pytest.skip(f"d={d} overflows sm_75 shared memory; runs on sm_80+")
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float16)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float16)
    q_mask = torch.rand(Nq, Lq, device="cuda") > 0.2
    d_mask = torch.rand(Nd, Ld, device="cuda") > 0.2
    q_mask[:, 0] = True  # guarantee at least one active token per row
    d_mask[:, 0] = True

    fast = maxsim(Q, D, q_mask=q_mask, d_mask=d_mask).float()
    ref = maxsim_reference(Q.float(), D.float(), q_mask, d_mask)

    assert rel(fast, ref) < 5e-3


def test_fully_masked_doc_scores_zero():
    """A doc whose mask is all-False must score 0 against every query (semantic
    contract: masked tokens are -inf, their max is -inf → clamped to 0)."""
    from late_interaction_kernels import maxsim

    Q = torch.randn(2, 8, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(3, 16, 128, device="cuda", dtype=torch.float16)
    d_mask = torch.ones(3, 16, device="cuda", dtype=torch.bool)
    d_mask[1] = False

    scores = maxsim(Q, D, d_mask=d_mask)
    assert torch.equal(scores[:, 1], torch.zeros(2, device="cuda"))


def test_fully_masked_query_scores_zero():
    """Dual of the above: an all-False query mask gives 0 across the corpus."""
    from late_interaction_kernels import maxsim

    Q = torch.randn(3, 8, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(2, 16, 128, device="cuda", dtype=torch.float16)
    q_mask = torch.ones(3, 8, device="cuda", dtype=torch.bool)
    q_mask[1] = False

    scores = maxsim(Q, D, q_mask=q_mask)
    assert torch.equal(scores[1, :], torch.zeros(2, device="cuda"))


def test_no_grad_dispatch_bit_equals_autograd_path():
    """``maxsim`` must produce a bit-identical forward whether the autograd
    path is taken or not. Pins the ``requires_grad``-driven dispatch that
    auto-skips the saved argmax when neither input needs a backward."""
    from late_interaction_kernels import maxsim

    Nq, Nd, Lq, Ld, d = 4, 8, 32, 128, 128
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float16)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float16)

    no_grad = maxsim(Q, D)
    with_grad = maxsim(Q.clone().requires_grad_(True), D.clone().requires_grad_(True)).detach()
    torch.testing.assert_close(no_grad, with_grad, atol=0, rtol=0)


def test_2d_inputs_return_scalar():
    """Single-query, single-doc call returns a scalar, not a [1, 1] tensor."""
    from late_interaction_kernels import maxsim

    Q = torch.randn(8, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(16, 128, device="cuda", dtype=torch.float16)
    s = maxsim(Q, D)
    assert s.dim() == 0, f"expected scalar, got {s.shape}"


@pytest.mark.parametrize("lq", [3, 9, 17, 33, 63, 127])
def test_non_pow2_lq_parity(lq):
    """Bucketing Lq to the next power of two must not change scores.

    Variable-length training has ``Lq`` floating around with the tokenizer
    output. The internal bucket-pad-mask must be transparent to the caller.
    """
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.reference import maxsim_reference

    Nq, Nd, Ld, d = 2, 3, 64, 128
    Q = torch.randn(Nq, lq, d, device="cuda", dtype=torch.float16)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float16)

    fast = maxsim(Q, D).float()
    ref = maxsim_reference(Q.float(), D.float())

    assert fast.shape == (Nq, Nd), f"output shape leaked the bucketed Lq: {fast.shape}"
    torch.testing.assert_close(fast, ref, rtol=5e-3, atol=5e-3)


def test_colpali_range_lq_buckets(rel):
    """Realistic ColPali Lq (~1030 visual patches) is still in the bucketed range.

    Was the motivating case for raising the bucket ceiling above 1024 — a
    fixed-shape ColPali workload would otherwise re-trigger autotune on
    every distinct token count seen during eval.
    """
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.reference import maxsim_reference

    Q = torch.randn(1, 1030, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(2, 256, 128, device="cuda", dtype=torch.float16)

    fast = maxsim(Q, D).float()
    ref = maxsim_reference(Q.float(), D.float())
    assert fast.shape == (1, 2)
    assert rel(fast, ref) < 5e-3


def test_very_long_lq_chunks_without_autotune_blowup(rel):
    """A query far above the old bucket ceiling is chunked, not warned about.

    Query-token chunking splits any ``Lq > _LQ_CHUNK`` into 128-token blocks,
    so the kernel only ever sees ``Lq == 128`` on the cross-product path. That
    both returns correct scores and pins the autotune cache to a single entry
    — no per-Lq sweep, no long-context fallback warning.
    """
    import warnings as _warnings

    from late_interaction_kernels import maxsim
    from late_interaction_kernels.reference import maxsim_reference

    Q = torch.randn(1, 5000, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(2, 256, 128, device="cuda", dtype=torch.float16)

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        fast = maxsim(Q, D).float()
    ref = maxsim_reference(Q.float(), D.float())

    assert fast.shape == (1, 2)
    assert rel(fast, ref) < 5e-3
    assert not any(
        "maxsim_varlen" in str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)
    ), "chunking should handle very long Lq without the long-context fallback warning"


def test_non_pow2_lq_gradient_shape():
    """``Q.grad`` must come back at the user's Lq, not the bucketed one.

    ``F.pad`` is autograd-aware, so the grad scattered onto the padded Q
    gets sliced back to the original shape on the way out. Pinning this
    so a future refactor doesn't accidentally break the contract.
    """
    from late_interaction_kernels import maxsim

    Q = torch.randn(2, 17, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    D = torch.randn(3, 64, 128, device="cuda", dtype=torch.float16, requires_grad=True)

    maxsim(Q, D).sum().backward()

    assert Q.grad is not None and Q.grad.shape == Q.shape
    assert D.grad is not None and D.grad.shape == D.shape
    assert torch.all(torch.isfinite(Q.grad))
    assert torch.all(torch.isfinite(D.grad))


# --- query-token chunking (large Lq) ---------------------------------------
# When Lq exceeds the chunk size the forward splits the query into 128-token
# blocks and sums the per-block MaxSim. These pin the chunked path to the
# reference across the ColPali-scale Lq grid, including tail padding (Lq not a
# multiple of 128), masks, normalize, and the backward.

# (Nq, Nd, Lq, Ld, d). All Lq > _LQ_CHUNK_MIN so chunking actually fires, and
# the grid spans exact multiples (1024 = 8·128) and non-multiples needing tail
# padding (700, 1030 ≈ ColPali visual patches) of the 128 chunk.
CHUNK_SHAPES = [
    (2, 4, 700, 256, 128),
    (4, 6, 1024, 320, 128),
    (1, 8, 1024, 1024, 128),
    (1, 2, 1030, 256, 128),
]
CHUNK_IDS = [f"Nq{s[0]}_Nd{s[1]}_Lq{s[2]}_Ld{s[3]}" for s in CHUNK_SHAPES]


@pytest.mark.parametrize("shape", CHUNK_SHAPES, ids=CHUNK_IDS)
@pytest.mark.parametrize("normalize", [False, True])
def test_chunked_forward_parity(shape, normalize, rel):
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.reference import maxsim_reference

    Nq, Nd, Lq, Ld, d = shape
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float16)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float16)

    fast = maxsim(Q, D, normalize=normalize).float()
    ref = maxsim_reference(Q.float(), D.float(), normalize=normalize)

    assert fast.shape == (Nq, Nd), f"chunking leaked into the output shape: {fast.shape}"
    assert rel(fast, ref) < 5e-3


def test_chunked_forward_parity_with_masks(rel):
    """Tail padding must compose with a caller-supplied q_mask + d_mask."""
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.reference import maxsim_reference

    Nq, Nd, Lq, Ld, d = 3, 5, 1030, 384, 128
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float16)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float16)
    q_mask = torch.rand(Nq, Lq, device="cuda") > 0.2
    d_mask = torch.rand(Nd, Ld, device="cuda") > 0.2
    q_mask[:, 0] = True
    d_mask[:, 0] = True

    fast = maxsim(Q, D, q_mask=q_mask, d_mask=d_mask, normalize=True).float()
    ref = maxsim_reference(Q.float(), D.float(), q_mask, d_mask, normalize=True)
    assert rel(fast, ref) < 5e-3


@pytest.mark.parametrize("Lq", [768, 1030])
@pytest.mark.parametrize("normalize", [False, True])
def test_chunked_backward_matches_unchunked(Lq, normalize, rel):
    """Chunked forward+backward == the un-chunked kernel on the same inputs.

    Comparing against the un-chunked *kernel* core (not the dense fp32
    reference) isolates the chunking algebra: both paths run the identical
    Triton kernel, so they make the identical argmax decisions and any
    residual difference is just fp32 accumulation order in the chunk sum and
    the atomic ``grad_D`` scatter.
    """
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.autograd import _maxsim_cross

    Nq, Nd, Ld, d = 2, 4, 256, 128
    torch.manual_seed(0)
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float16)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float16)
    g = torch.randn(Nq, Nd, device="cuda", dtype=torch.float32)

    Qk = Q.clone().requires_grad_(True)
    Dk = D.clone().requires_grad_(True)
    sk = maxsim(Qk, Dk, normalize=normalize)
    sk.backward(g)

    Qb = Q.clone().requires_grad_(True)
    Db = D.clone().requires_grad_(True)
    sb = _maxsim_cross(Qb, Db, None, None, normalize, "auto")
    sb.backward(g)

    assert rel(sk, sb) < 5e-3
    assert Qk.grad.shape == Q.shape and Dk.grad.shape == D.shape
    assert rel(Qk.grad.float(), Qb.grad.float()) < 1e-2
    assert rel(Dk.grad.float(), Db.grad.float()) < 1e-2
