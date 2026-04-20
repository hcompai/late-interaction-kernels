"""Tests for the CSR (scatter-free) grad_D path.

Covers:

* Parity with the atomic path (both should agree to fp32 accumulator noise).
* Parity with PyTorch autograd reference.
* Hot-bucket stress (all queries collapsing to the same ``t``).
* Empty-bucket handling (no ``(i, s)`` lands on some ``t``).
* Fully-masked rows and columns.
* Non-power-of-two ``d``.
* Cross-run determinism (same input → same output).
* CSR builder correctness on tiny hand-computed cases.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.cuda


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = max(1e-6, b.abs().max().item())
    return (a - b).abs().max().item() / denom


# ---------------------------------------------------------------------------
# Unit test for the CSR builder itself (CPU-correctness style but on GPU)
# ---------------------------------------------------------------------------

def test_build_csr_small_known():
    """Hand-computable CSR for a 2x2 x Lq=3, Ld=4 argmax."""
    from flash_colbert.backward_csr import _build_csr

    Nq, Nd, Lq, Ld = 2, 2, 3, 4
    # argmax[i, j, s] with shape [Nq*Nd, Lq] == [4, 3].
    # Fill deterministically:
    #   (i=0,j=0): s -> [0, 2, 2]
    #   (i=0,j=1): s -> [1, 1, 3]
    #   (i=1,j=0): s -> [0, 0, 3]
    #   (i=1,j=1): s -> [2, 2, 2]
    argmax = torch.tensor(
        [[0, 2, 2], [1, 1, 3], [0, 0, 3], [2, 2, 2]],
        device="cuda", dtype=torch.int32,
    )
    row_ptr, perm = _build_csr(argmax, Nq, Nd, Lq, Ld)
    assert row_ptr.shape == (Nd, Ld + 1)
    assert perm.shape == (Nd, Nq * Lq)

    # For j=0: entries are
    #   (i=0): t=[0,2,2]  at flat indices 0,1,2
    #   (i=1): t=[0,0,3]  at flat indices 3,4,5
    # Bucketed: t=0 -> {0, 3, 4}, t=1 -> {}, t=2 -> {1, 2}, t=3 -> {5}
    # row_ptr[0] = [0, 3, 3, 5, 6]
    assert row_ptr[0].tolist() == [0, 3, 3, 5, 6]

    # For j=1:
    #   (i=0): t=[1,1,3]  flat 0,1,2
    #   (i=1): t=[2,2,2]  flat 3,4,5
    # t=0 -> {}, t=1 -> {0,1}, t=2 -> {3,4,5}, t=3 -> {2}
    # row_ptr[1] = [0, 0, 2, 5, 6]
    assert row_ptr[1].tolist() == [0, 0, 2, 5, 6]

    # Verify perm indices form valid buckets by sorted t
    # (torch.sort is not stable-guaranteed but the counts above must match).
    for j in range(Nd):
        for t in range(Ld):
            s_start = row_ptr[j, t].item()
            s_end = row_ptr[j, t + 1].item()
            for off in range(s_start, s_end):
                flat = perm[j, off].item()
                i_flat, s_flat = flat // Lq, flat % Lq
                assert argmax[i_flat * Nd + j, s_flat].item() == t


# ---------------------------------------------------------------------------
# CSR vs atomic equivalence — both paths consume the same argmax buffer,
# so they must agree up to fp32 reduction-order noise.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "Nq,Nd,Lq,Ld,d",
    [
        (2, 4, 16, 32, 128),
        (4, 8, 32, 128, 128),
        (2, 2, 32, 300, 128),
        (1, 1, 8, 16, 256),
        (8, 8, 32, 200, 128),
    ],
)
def test_csr_matches_atomic(Nq, Nd, Lq, Ld, d):
    from flash_colbert import maxsim, set_backward_method

    torch.manual_seed(0)
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float32)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float32)
    grad_out = torch.randn(Nq, Nd, device="cuda", dtype=torch.float32)

    # Atomic path
    set_backward_method("atomic")
    Qa = Q.clone().requires_grad_(True)
    Da = D.clone().requires_grad_(True)
    maxsim(Qa, Da).backward(grad_out)
    gQa, gDa = Qa.grad.clone(), Da.grad.clone()

    # CSR path
    set_backward_method("csr")
    Qc = Q.clone().requires_grad_(True)
    Dc = D.clone().requires_grad_(True)
    maxsim(Qc, Dc).backward(grad_out)
    gQc, gDc = Qc.grad.clone(), Dc.grad.clone()

    # grad_Q is bit-identical (same kernel in both paths).
    assert torch.equal(gQa, gQc), "grad_Q should be bitwise identical between paths"

    # grad_D differs only in reduction order (fp32). fp32 reassociation error
    # on ~Nq*Lq adds per bucket is well below 1e-4 relative.
    assert _rel_err(gDa, gDc) < 1e-5, f"rel_err(grad_D, csr vs atomic)={_rel_err(gDa, gDc)}"


# ---------------------------------------------------------------------------
# CSR vs PyTorch autograd
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "Nq,Nd,Lq,Ld,d",
    [
        (2, 4, 16, 32, 128),
        (4, 8, 32, 128, 128),
        (8, 16, 32, 200, 128),
    ],
)
def test_csr_parity_vs_pytorch_fp32(Nq, Nd, Lq, Ld, d):
    from flash_colbert import maxsim, set_backward_method
    from flash_colbert.reference import maxsim_reference

    set_backward_method("csr")
    torch.manual_seed(1)
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float32, requires_grad=True)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float32, requires_grad=True)
    go = torch.randn(Nq, Nd, device="cuda", dtype=torch.float32)

    maxsim(Q, D).backward(go)
    gQf, gDf = Q.grad.clone(), D.grad.clone()

    Q.grad = None
    D.grad = None
    maxsim_reference(Q, D).backward(go)
    gQr, gDr = Q.grad.clone(), D.grad.clone()

    # Triton's tl.dot promotes fp32 inputs to TF32 on Hopper tensor cores
    # (mantissa truncated to 10 bits), so "fp32" compute here has ~5e-4
    # relative error vs the pure-PyTorch fp32 reference. That's expected and
    # matches the tolerance used elsewhere in the repo.
    assert _rel_err(gQf, gQr) < 2e-3, _rel_err(gQf, gQr)
    assert _rel_err(gDf, gDr) < 2e-3, _rel_err(gDf, gDr)


# ---------------------------------------------------------------------------
# Hot-bucket stress: construct inputs where every query-token's argmax
# collapses to a single doc-token. This is the worst case for the atomic
# path (maximum contention) and a stress test for the CSR kernel's
# single-bucket dynamic loop.
# ---------------------------------------------------------------------------

def test_hot_bucket_all_queries_win_on_same_t():
    """Stress the CSR path with maximum bucket contention: every ``(i, j, s)``
    argmax equals 0, so bucket ``(j, 0)`` has ``Nq * Lq`` entries and every
    other bucket is empty. This is the worst case for the atomic path (all
    threads hammering a single cache line) and a correctness stress for CSR
    (one program does a long reduction, all others write zeros).
    """
    from flash_colbert import maxsim, set_backward_method

    Nq, Nd, Lq, Ld, d = 8, 4, 32, 64, 128
    # Q all ones, D[:, 0, :] = +1 (dot = d), D[:, t>0, :] = -1 (dot = -d).
    # argmax is always t=0.
    Q = torch.ones(Nq, Lq, d, device="cuda", dtype=torch.float32)
    D = -torch.ones(Nd, Ld, d, device="cuda", dtype=torch.float32)
    D[:, 0, :] = 1.0
    go = torch.randn(Nq, Nd, device="cuda", dtype=torch.float32)

    for method in ("csr", "atomic"):
        set_backward_method(method)
        Qv = Q.clone().requires_grad_(True)
        Dv = D.clone().requires_grad_(True)
        maxsim(Qv, Dv).backward(go)
        gD = Dv.grad
        other = gD[:, 1:, :].abs().max().item()
        assert other == 0.0, f"hot-bucket leak ({method}): {other}"
        assert gD[:, 0, :].abs().max().item() > 0.0


# ---------------------------------------------------------------------------
# Empty bucket: make the argmax live on a few t values only.
# ---------------------------------------------------------------------------

def test_empty_buckets_zero_grad():
    """Most buckets are empty (only t=3 wins). CSR kernel must still write
    zeros to every non-winning ``(j, t)`` output."""
    from flash_colbert import maxsim, set_backward_method

    Nq, Nd, Lq, Ld, d = 2, 2, 8, 64, 128
    Q = torch.ones(Nq, Lq, d, device="cuda", dtype=torch.float32)
    D = -torch.ones(Nd, Ld, d, device="cuda", dtype=torch.float32)
    D[:, 3, :] = 1.0  # only token 3 has positive alignment with Q
    go = torch.randn(Nq, Nd, device="cuda", dtype=torch.float32)

    set_backward_method("csr")
    Qv = Q.clone().requires_grad_(True)
    Dv = D.clone().requires_grad_(True)
    maxsim(Qv, Dv).backward(go)
    mask = torch.ones(Ld, dtype=torch.bool, device="cuda")
    mask[3] = False
    assert Dv.grad[:, mask, :].abs().max().item() == 0.0
    assert Dv.grad[:, 3, :].abs().max().item() > 0.0


# ---------------------------------------------------------------------------
# Masks: both q_mask and d_mask interacting with CSR.
# ---------------------------------------------------------------------------

def test_csr_grad_with_masks():
    from flash_colbert import maxsim, set_backward_method
    from flash_colbert.reference import maxsim_reference

    set_backward_method("csr")
    Nq, Nd, Lq, Ld, d = 4, 4, 32, 128, 128
    torch.manual_seed(2)
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float32, requires_grad=True)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float32, requires_grad=True)
    q_mask = torch.rand(Nq, Lq, device="cuda") > 0.25
    d_mask = torch.rand(Nd, Ld, device="cuda") > 0.25
    q_mask[:, 0] = True  # guarantee at least one active token per row
    d_mask[:, 0] = True
    go = torch.randn(Nq, Nd, device="cuda", dtype=torch.float32)

    maxsim(Q, D, q_mask=q_mask, d_mask=d_mask).backward(go)
    gQf, gDf = Q.grad.clone(), D.grad.clone()

    Q.grad = None
    D.grad = None
    maxsim_reference(Q, D, q_mask=q_mask, d_mask=d_mask).backward(go)
    gQr, gDr = Q.grad.clone(), D.grad.clone()

    assert _rel_err(gQf, gQr) < 1e-3, _rel_err(gQf, gQr)
    assert _rel_err(gDf, gDr) < 1e-3, _rel_err(gDf, gDr)


def test_csr_masked_query_gets_zero_grad():
    from flash_colbert import maxsim, set_backward_method

    set_backward_method("csr")
    Q = torch.randn(2, 8, 128, device="cuda", requires_grad=True)
    D = torch.randn(3, 16, 128, device="cuda", requires_grad=True)
    q_mask = torch.ones(2, 8, device="cuda", dtype=torch.bool)
    q_mask[0, 3] = False

    maxsim(Q, D, q_mask=q_mask).sum().backward()
    assert Q.grad[0, 3].abs().max().item() == 0.0


def test_csr_masked_doc_token_gets_zero_grad():
    """d_mask[j, t] == False means token t was forced to -inf in the forward,
    so it can't be an argmax winner and must receive no gradient from the CSR
    reduction."""
    from flash_colbert import maxsim, set_backward_method

    set_backward_method("csr")
    Nq, Nd, Lq, Ld, d = 2, 3, 8, 16, 128
    Q = torch.randn(Nq, Lq, d, device="cuda", requires_grad=True)
    D = torch.randn(Nd, Ld, d, device="cuda", requires_grad=True)
    d_mask = torch.ones(Nd, Ld, device="cuda", dtype=torch.bool)
    d_mask[1, 5] = False  # mask out one token
    d_mask[1, 11] = False

    maxsim(Q, D, d_mask=d_mask).sum().backward()
    assert D.grad[1, 5, :].abs().max().item() == 0.0
    assert D.grad[1, 11, :].abs().max().item() == 0.0


# ---------------------------------------------------------------------------
# Non-power-of-two d (tests the `d_pad` masking path in the kernel).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d", [96, 192, 320])
def test_csr_non_pow2_d(d):
    from flash_colbert import maxsim, set_backward_method
    from flash_colbert.reference import maxsim_reference

    set_backward_method("csr")
    Nq, Nd, Lq, Ld = 2, 3, 16, 32
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float32, requires_grad=True)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float32, requires_grad=True)
    go = torch.randn(Nq, Nd, device="cuda", dtype=torch.float32)

    maxsim(Q, D).backward(go)
    gQf, gDf = Q.grad.clone(), D.grad.clone()

    Q.grad = None
    D.grad = None
    maxsim_reference(Q, D).backward(go)
    gQr, gDr = Q.grad.clone(), D.grad.clone()

    # tf32 tensor-core tolerance, same as fp32 parity tests above.
    assert _rel_err(gQf, gQr) < 2e-3
    assert _rel_err(gDf, gDr) < 2e-3


# ---------------------------------------------------------------------------
# Determinism: CSR should be bit-reproducible across runs.
# ---------------------------------------------------------------------------

def test_csr_determinism_across_runs():
    from flash_colbert import maxsim, set_backward_method

    set_backward_method("csr")
    torch.manual_seed(3)
    Q0 = torch.randn(4, 32, 128, device="cuda", dtype=torch.float32)
    D0 = torch.randn(8, 128, 128, device="cuda", dtype=torch.float32)
    go = torch.randn(4, 8, device="cuda", dtype=torch.float32)

    grads = []
    for _ in range(3):
        Q = Q0.clone().requires_grad_(True)
        D = D0.clone().requires_grad_(True)
        maxsim(Q, D).backward(go)
        grads.append((Q.grad.clone(), D.grad.clone()))

    for k in range(1, 3):
        assert torch.equal(grads[0][0], grads[k][0]), "grad_Q drift across runs"
        assert torch.equal(grads[0][1], grads[k][1]), "grad_D drift across runs"


# ---------------------------------------------------------------------------
# bf16 inputs: CSR should still match atomic on the fp32 grad output.
# ---------------------------------------------------------------------------

def test_auto_matches_both_explicit_methods():
    """``set_backward_method("auto")`` must produce the same grad_Q as any
    explicit path, and a grad_D consistent with whichever it chose."""
    from flash_colbert import maxsim, set_backward_method

    shapes = [
        (2, 4, 16, 32, 128),   # small — auto should pick atomic
        (64, 64, 32, 128, 128),  # medium — auto picks atomic
    ]
    for Nq, Nd, Lq, Ld, d in shapes:
        torch.manual_seed(10)
        Q0 = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float32)
        D0 = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float32)
        go = torch.randn(Nq, Nd, device="cuda", dtype=torch.float32)

        grads = {}
        for m in ("auto", "csr", "atomic"):
            set_backward_method(m)
            Q = Q0.clone().requires_grad_(True)
            D = D0.clone().requires_grad_(True)
            maxsim(Q, D).backward(go)
            grads[m] = (Q.grad.clone(), D.grad.clone())

        # grad_Q: all three paths use the same kernel, bitwise identical.
        assert torch.equal(grads["auto"][0], grads["csr"][0])
        assert torch.equal(grads["auto"][0], grads["atomic"][0])
        # grad_D: auto must match one of the two (it picks one of them).
        auto_d = grads["auto"][1]
        near_csr = _rel_err(auto_d, grads["csr"][1]) < 1e-5
        near_atom = _rel_err(auto_d, grads["atomic"][1]) < 1e-5
        assert near_csr or near_atom, "auto grad_D matches neither path"


def test_csr_bf16_matches_atomic():
    from flash_colbert import maxsim, set_backward_method

    torch.manual_seed(4)
    Q0 = torch.randn(4, 32, 128, device="cuda", dtype=torch.bfloat16)
    D0 = torch.randn(8, 128, 128, device="cuda", dtype=torch.bfloat16)
    go = torch.randn(4, 8, device="cuda", dtype=torch.float32)

    set_backward_method("atomic")
    Qa = Q0.clone().requires_grad_(True)
    Da = D0.clone().requires_grad_(True)
    maxsim(Qa, Da).backward(go)

    set_backward_method("csr")
    Qc = Q0.clone().requires_grad_(True)
    Dc = D0.clone().requires_grad_(True)
    maxsim(Qc, Dc).backward(go)

    # bf16 output dtype — tolerate bf16's ~3e-3 relative noise on grad_D
    # while requiring grad_Q (same kernel both paths) to match exactly.
    assert torch.equal(Qa.grad, Qc.grad)
    rd = _rel_err(Da.grad.float(), Dc.grad.float())
    assert rd < 5e-3, rd
