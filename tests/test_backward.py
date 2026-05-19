"""Backward-pass correctness and path equivalence.

Covers three things:

1. **Gradient parity vs PyTorch autograd** — the kernel's backward must match the
   autograd of the pure-PyTorch reference within the expected fp16 / tf32 ULP drift,
   with and without masks.
2. **Path equivalence** — ``set_backward_method("atomic" | "csr" | "auto")`` must
   produce the same gradients up to fp32-reduction-order noise (grad_Q is bitwise
   identical across paths because the Q-grad kernel is the same).
3. **Stress cases for grad_D** — hot buckets (one doc token wins for every query),
   empty buckets (most doc tokens never win), and non-power-of-two ``d``.

Plus a unit test for the CSR builder (``_build_csr``) on a hand-computed 2×2 / Lq=3
/ Ld=4 example.
"""

import pytest
import torch

pytestmark = pytest.mark.cuda


# --------------------------------------------------------------------------- #
# 1. Gradient parity vs PyTorch autograd                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "Nq,Nd,Lq,Ld,d",
    [
        (2, 4, 16, 32, 128),
        (4, 8, 32, 128, 128),
        (2, 2, 32, 300, 128),
        (1, 1, 8, 16, 256),
    ],
)
def test_grad_parity_fp16(Nq, Nd, Lq, Ld, d, rel):
    """fp16 inputs, fp32 accumulator — grads within ~1% of the fp32 reference."""
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.reference import maxsim_reference

    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float32, requires_grad=True)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float32, requires_grad=True)
    grad_out = torch.randn(Nq, Nd, device="cuda", dtype=torch.float32)

    maxsim(Q.half(), D.half()).backward(grad_out)
    gQf, gDf = Q.grad.clone(), D.grad.clone()

    Q.grad = None
    D.grad = None
    maxsim_reference(Q, D).backward(grad_out)
    gQr, gDr = Q.grad.clone(), D.grad.clone()

    assert rel(gQf, gQr) < 1e-2
    assert rel(gDf, gDr) < 1e-2


def test_grad_parity_with_masks(rel):
    """Both q_mask and d_mask active; fp32 compute tolerance (tf32 tensor cores)."""
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.reference import maxsim_reference

    Nq, Nd, Lq, Ld, d = 4, 4, 32, 128, 128
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float32, requires_grad=True)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float32, requires_grad=True)
    q_mask = torch.rand(Nq, Lq, device="cuda") > 0.2
    d_mask = torch.rand(Nd, Ld, device="cuda") > 0.2
    q_mask[:, 0] = True
    d_mask[:, 0] = True
    grad_out = torch.randn(Nq, Nd, device="cuda", dtype=torch.float32)

    maxsim(Q, D, q_mask=q_mask, d_mask=d_mask).backward(grad_out)
    gQf, gDf = Q.grad.clone(), D.grad.clone()

    Q.grad = None
    D.grad = None
    maxsim_reference(Q, D, q_mask, d_mask).backward(grad_out)
    gQr, gDr = Q.grad.clone(), D.grad.clone()

    # tf32 tensor-core drift on fp32 inputs is ~5e-4; allow a 5× margin.
    assert rel(gQf, gQr) < 3e-3
    assert rel(gDf, gDr) < 3e-3


@pytest.mark.parametrize("d", [96, 192, 320])
def test_grad_parity_non_pow2_d(d, rel):
    """Backward must also handle non-power-of-two embedding dims (the ``d_pad``
    masking path in the kernel)."""
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.reference import maxsim_reference

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

    assert rel(gQf, gQr) < 3e-3
    assert rel(gDf, gDr) < 3e-3


# --------------------------------------------------------------------------- #
# 2. Mask semantics: masked positions get zero gradient                       #
# --------------------------------------------------------------------------- #


def test_masked_query_token_gets_zero_grad():
    """``q_mask[i, s] == False`` ⇒ Q[i, s] has no influence and must get 0 grad."""
    from late_interaction_kernels import maxsim

    Q = torch.randn(2, 8, 128, device="cuda", requires_grad=True)
    D = torch.randn(3, 16, 128, device="cuda", requires_grad=True)
    q_mask = torch.ones(2, 8, device="cuda", dtype=torch.bool)
    q_mask[0, 3] = False

    maxsim(Q, D, q_mask=q_mask).sum().backward()
    assert Q.grad[0, 3].abs().max().item() == 0.0


def test_masked_doc_token_gets_zero_grad():
    """``d_mask[j, t] == False`` ⇒ token t is ``-inf`` in the forward, can't win
    the argmax, and must receive no gradient."""
    from late_interaction_kernels import maxsim

    Nq, Nd, Lq, Ld, d = 2, 3, 8, 16, 128
    Q = torch.randn(Nq, Lq, d, device="cuda", requires_grad=True)
    D = torch.randn(Nd, Ld, d, device="cuda", requires_grad=True)
    d_mask = torch.ones(Nd, Ld, device="cuda", dtype=torch.bool)
    d_mask[1, 5] = False
    d_mask[1, 11] = False

    maxsim(Q, D, d_mask=d_mask).sum().backward()
    assert D.grad[1, 5, :].abs().max().item() == 0.0
    assert D.grad[1, 11, :].abs().max().item() == 0.0


# --------------------------------------------------------------------------- #
# 3. Atomic vs CSR path equivalence                                           #
# --------------------------------------------------------------------------- #


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
def test_atomic_csr_equivalence(Nq, Nd, Lq, Ld, d, rel):
    """Both backward paths consume the *same* argmax buffer, so they must agree
    up to fp32 reduction-order noise. grad_Q comes from the same kernel either
    way and must be bitwise identical."""
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.autograd import set_backward_method

    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float32)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float32)
    grad_out = torch.randn(Nq, Nd, device="cuda", dtype=torch.float32)

    set_backward_method("atomic")
    Qa = Q.clone().requires_grad_(True)
    Da = D.clone().requires_grad_(True)
    maxsim(Qa, Da).backward(grad_out)

    set_backward_method("csr")
    Qc = Q.clone().requires_grad_(True)
    Dc = D.clone().requires_grad_(True)
    maxsim(Qc, Dc).backward(grad_out)

    set_backward_method("auto")  # restore default

    assert torch.equal(Qa.grad, Qc.grad), "grad_Q must be bitwise identical"
    assert rel(Dc.grad, Da.grad) < 1e-5


def test_atomic_csr_equivalence_bf16(rel):
    """bf16 inputs: grad_Q identical, grad_D within bf16 ULP drift."""
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.autograd import set_backward_method

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

    set_backward_method("auto")

    assert torch.equal(Qa.grad, Qc.grad)
    assert rel(Dc.grad.float(), Da.grad.float()) < 5e-3


def test_auto_selects_a_valid_path(rel):
    """``auto`` must produce results matching *one of* the two explicit paths
    (whichever its heuristic selected). Run across shapes that exercise both
    branches of the heuristic."""
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.autograd import set_backward_method

    for Nq, Nd, Lq, Ld, d in [(2, 4, 16, 32, 128), (64, 64, 32, 128, 128)]:
        Q0 = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float32)
        D0 = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float32)
        go = torch.randn(Nq, Nd, device="cuda", dtype=torch.float32)

        grads = {}
        for m in ("auto", "csr", "atomic", "unified"):
            set_backward_method(m)
            Q = Q0.clone().requires_grad_(True)
            D = D0.clone().requires_grad_(True)
            maxsim(Q, D).backward(go)
            grads[m] = (Q.grad.clone(), D.grad.clone())

        set_backward_method("auto")

        # grad_Q is row-owned in every path — all four must agree bitwise.
        for m in ("csr", "atomic", "unified"):
            assert torch.equal(grads["auto"][0], grads[m][0]), f"grad_Q diverges for {m=}"

        # grad_D: auto must match whichever explicit path its heuristic picked.
        # Reorder drift across atomic variants is ~1e-3 in fp32, so we pick the
        # closest path as ground truth and check that match.
        dists = {m: rel(grads["auto"][1], grads[m][1]) for m in ("csr", "atomic", "unified")}
        closest = min(dists, key=dists.get)
        assert dists[closest] < 1e-4, (
            f"auto grad_D matches neither path at tight tol ({Nq=}, {Nd=}); dists={dists}"
        )


# --------------------------------------------------------------------------- #
# 4. Stress cases for grad_D                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["atomic", "csr"])
def test_hot_bucket_single_winner(method):
    """Worst case for both paths: every ``(i, j, s)`` argmax collapses to t=0.
    Atomic path: maximum cache-line contention. CSR path: one bucket holds
    ``Nq * Lq`` entries, all other buckets are empty. Both must produce nonzero
    grad on token 0 and exactly zero grad everywhere else."""
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.autograd import set_backward_method

    Nq, Nd, Lq, Ld, d = 8, 4, 32, 64, 128
    # Deterministic: Q = +1, D[:, 0, :] = +1 (dot = d), D[:, t>0, :] = -1.
    Q = torch.ones(Nq, Lq, d, device="cuda", dtype=torch.float32)
    D = -torch.ones(Nd, Ld, d, device="cuda", dtype=torch.float32)
    D[:, 0, :] = 1.0
    go = torch.randn(Nq, Nd, device="cuda", dtype=torch.float32)

    set_backward_method(method)
    try:
        Qv = Q.clone().requires_grad_(True)
        Dv = D.clone().requires_grad_(True)
        maxsim(Qv, Dv).backward(go)
        assert Dv.grad[:, 1:, :].abs().max().item() == 0.0
        assert Dv.grad[:, 0, :].abs().max().item() > 0.0
    finally:
        set_backward_method("auto")


def test_empty_buckets_write_zero():
    """Most CSR buckets are empty (only t=3 ever wins). The kernel must still
    write zeros to every non-winning ``(j, t)`` output cell."""
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.autograd import set_backward_method

    Nq, Nd, Lq, Ld, d = 2, 2, 8, 64, 128
    Q = torch.ones(Nq, Lq, d, device="cuda", dtype=torch.float32)
    D = -torch.ones(Nd, Ld, d, device="cuda", dtype=torch.float32)
    D[:, 3, :] = 1.0
    go = torch.randn(Nq, Nd, device="cuda", dtype=torch.float32)

    set_backward_method("csr")
    try:
        Qv = Q.clone().requires_grad_(True)
        Dv = D.clone().requires_grad_(True)
        maxsim(Qv, Dv).backward(go)
        mask = torch.ones(Ld, dtype=torch.bool, device="cuda")
        mask[3] = False
        assert Dv.grad[:, mask, :].abs().max().item() == 0.0
        assert Dv.grad[:, 3, :].abs().max().item() > 0.0
    finally:
        set_backward_method("auto")


# --------------------------------------------------------------------------- #
# 5. Determinism                                                              #
# --------------------------------------------------------------------------- #


def test_csr_is_bitwise_deterministic():
    """CSR is genuinely bitwise-deterministic: every bucket reduces in a fixed
    order inside one program, no atomics. Running backward three times on the
    same input must produce the exact same grad tensors."""
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.autograd import set_backward_method

    Q0 = torch.randn(4, 32, 128, device="cuda", dtype=torch.float32)
    D0 = torch.randn(8, 128, 128, device="cuda", dtype=torch.float32)
    go = torch.randn(4, 8, device="cuda", dtype=torch.float32)

    set_backward_method("csr")
    try:
        grads = []
        for _ in range(3):
            Q = Q0.clone().requires_grad_(True)
            D = D0.clone().requires_grad_(True)
            maxsim(Q, D).backward(go)
            grads.append((Q.grad.clone(), D.grad.clone()))
        for k in range(1, 3):
            assert torch.equal(grads[0][0], grads[k][0])
            assert torch.equal(grads[0][1], grads[k][1])
    finally:
        set_backward_method("auto")


def test_atomic_is_numerically_stable_across_runs(rel):
    """fp32 ``atomic_add`` reduction order depends on thread scheduling, so the
    atomic path is *not* strictly bitwise-reproducible across runs — but the
    drift is bounded by fp32 ULP (~1e-6 relative). grad_Q (no atomics) still
    matches exactly."""
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.autograd import set_backward_method

    Q0 = torch.randn(4, 32, 128, device="cuda", dtype=torch.float32)
    D0 = torch.randn(8, 128, 128, device="cuda", dtype=torch.float32)
    go = torch.randn(4, 8, device="cuda", dtype=torch.float32)

    set_backward_method("atomic")
    try:
        grads = []
        for _ in range(3):
            Q = Q0.clone().requires_grad_(True)
            D = D0.clone().requires_grad_(True)
            maxsim(Q, D).backward(go)
            grads.append((Q.grad.clone(), D.grad.clone()))
        for k in range(1, 3):
            assert torch.equal(grads[0][0], grads[k][0])  # grad_Q is scatter-free
            assert rel(grads[0][1], grads[k][1]) < 1e-5
    finally:
        set_backward_method("auto")


# --------------------------------------------------------------------------- #
# 6. CSR builder unit test                                                    #
# --------------------------------------------------------------------------- #


def test_build_csr_hand_computed():
    """Hand-checkable CSR for a ``Nq=Nd=2, Lq=3, Ld=4`` argmax.

    With the argmax below, bucket ``(j=0, t=0)`` gets 3 entries, ``(j=0, t=2)``
    gets 2, ``(j=0, t=3)`` gets 1, ``(j=0, t=1)`` is empty, etc. We verify both
    ``row_ptr`` and that every entry in ``perm`` lands in the correct bucket.
    """
    from late_interaction_kernels.backward.csr import _build_csr

    Nq, Nd, Lq, Ld = 2, 2, 3, 4
    #   (i=0, j=0): s -> [0, 2, 2]
    #   (i=0, j=1): s -> [1, 1, 3]
    #   (i=1, j=0): s -> [0, 0, 3]
    #   (i=1, j=1): s -> [2, 2, 2]
    argmax = torch.tensor(
        [[0, 2, 2], [1, 1, 3], [0, 0, 3], [2, 2, 2]],
        device="cuda",
        dtype=torch.int32,
    )
    row_ptr, perm = _build_csr(argmax, Nq, Nd, Lq, Ld)
    assert row_ptr.shape == (Nd, Ld + 1)
    assert perm.shape == (Nd, Nq * Lq)

    # j=0: t-counts = [3, 0, 2, 1]  → row_ptr = cumsum([0, 3, 0, 2, 1]) = [0, 3, 3, 5, 6]
    # j=1: t-counts = [0, 2, 3, 1]  → row_ptr = [0, 0, 2, 5, 6]
    assert row_ptr[0].tolist() == [0, 3, 3, 5, 6]
    assert row_ptr[1].tolist() == [0, 0, 2, 5, 6]

    for j in range(Nd):
        for t in range(Ld):
            for off in range(row_ptr[j, t].item(), row_ptr[j, t + 1].item()):
                flat = perm[j, off].item()
                i, s = flat // Lq, flat % Lq
                assert argmax[i * Nd + j, s].item() == t
