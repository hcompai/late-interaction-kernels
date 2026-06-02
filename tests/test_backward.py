"""Backward-pass correctness and path equivalence.

Covers three things:

1. **Gradient parity vs PyTorch autograd** — the kernel's backward must match the
   autograd of the pure-PyTorch reference within the expected fp16 / tf32 ULP drift,
   with and without masks.
2. **Path equivalence** — the ``backward=`` kwarg (``"unified" | "lowmem" | "auto"``)
   must produce the same gradients up to fp32-reduction-order noise.
3. **Stress cases for grad_D** — hot buckets (one doc token wins for every query),
   empty buckets (most doc tokens never win), and non-power-of-two ``d``.
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
# 3. unified vs lowmem path equivalence                                       #
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
def test_unified_lowmem_equivalence(Nq, Nd, Lq, Ld, d, rel):
    """Both backward paths consume the *same* argmax buffer. grad_Q matches to
    fp32 (both accumulate in fp32 registers). grad_D drifts more: lowmem reduces
    it with a bf16/fp16 one-hot matmul (its compute dtype is never fp32), so vs
    unified's fp32 atomics it carries bf16-matmul precision even for fp32 inputs."""
    from late_interaction_kernels import maxsim

    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float32)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float32)
    grad_out = torch.randn(Nq, Nd, device="cuda", dtype=torch.float32)

    Qu = Q.clone().requires_grad_(True)
    Du = D.clone().requires_grad_(True)
    maxsim(Qu, Du, backward="unified").backward(grad_out)

    Ql = Q.clone().requires_grad_(True)
    Dl = D.clone().requires_grad_(True)
    maxsim(Ql, Dl, backward="lowmem").backward(grad_out)

    assert rel(Ql.grad, Qu.grad) < 1e-5, "grad_Q must match (row-owned, same math)"
    assert rel(Dl.grad, Du.grad) < 2e-3


def test_unified_lowmem_equivalence_bf16(rel):
    """bf16 inputs: lowmem rounds each program's grad_D to bf16, unified rounds
    once after an fp32 accumulate — they must agree within bf16 ULP drift."""
    from late_interaction_kernels import maxsim

    Q0 = torch.randn(4, 32, 128, device="cuda", dtype=torch.bfloat16)
    D0 = torch.randn(8, 128, 128, device="cuda", dtype=torch.bfloat16)
    go = torch.randn(4, 8, device="cuda", dtype=torch.float32)

    Qu = Q0.clone().requires_grad_(True)
    Du = D0.clone().requires_grad_(True)
    maxsim(Qu, Du, backward="unified").backward(go)

    Ql = Q0.clone().requires_grad_(True)
    Dl = D0.clone().requires_grad_(True)
    maxsim(Ql, Dl, backward="lowmem").backward(go)

    assert rel(Ql.grad.float(), Qu.grad.float()) < 5e-3
    assert rel(Dl.grad.float(), Du.grad.float()) < 5e-3


def test_auto_selects_a_valid_path(rel):
    """``auto`` must produce results matching *one of* the two explicit paths
    (whichever its heuristic selected). Run across shapes that exercise both
    branches of the heuristic."""
    from late_interaction_kernels import maxsim

    for Nq, Nd, Lq, Ld, d in [(2, 4, 16, 32, 128), (64, 64, 32, 128, 128)]:
        Q0 = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float32)
        D0 = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float32)
        go = torch.randn(Nq, Nd, device="cuda", dtype=torch.float32)

        grads = {}
        for m in ("auto", "unified", "lowmem"):
            Q = Q0.clone().requires_grad_(True)
            D = D0.clone().requires_grad_(True)
            maxsim(Q, D, backward=m).backward(go)
            grads[m] = (Q.grad.clone(), D.grad.clone())

        # grad_D: auto must match whichever explicit path its heuristic picked.
        dists = {m: rel(grads["auto"][1], grads[m][1]) for m in ("unified", "lowmem")}
        closest = min(dists, key=dists.get)
        assert dists[closest] < 1e-4, (
            f"auto grad_D matches neither path at tight tol ({Nq=}, {Nd=}); dists={dists}"
        )


# --------------------------------------------------------------------------- #
# 4. Stress cases for grad_D                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["unified", "lowmem"])
def test_hot_bucket_single_winner(method):
    """Worst case for grad_D: every ``(i, j, s)`` argmax collapses to t=0.
    Unified: maximum atomic cache-line contention. Lowmem: one doc-token row
    accumulates ``Nq * Lq`` entries. Both must produce nonzero grad on token 0
    and exactly zero grad everywhere else."""
    from late_interaction_kernels import maxsim

    Nq, Nd, Lq, Ld, d = 8, 4, 32, 64, 128
    # Deterministic: Q = +1, D[:, 0, :] = +1 (dot = d), D[:, t>0, :] = -1.
    Q = torch.ones(Nq, Lq, d, device="cuda", dtype=torch.float32)
    D = -torch.ones(Nd, Ld, d, device="cuda", dtype=torch.float32)
    D[:, 0, :] = 1.0
    go = torch.randn(Nq, Nd, device="cuda", dtype=torch.float32)

    Qv = Q.clone().requires_grad_(True)
    Dv = D.clone().requires_grad_(True)
    maxsim(Qv, Dv, backward=method).backward(go)
    assert Dv.grad[:, 1:, :].abs().max().item() == 0.0
    assert Dv.grad[:, 0, :].abs().max().item() > 0.0


@pytest.mark.parametrize("method", ["unified", "lowmem"])
def test_empty_buckets_write_zero(method):
    """Most doc tokens never win (only t=3 ever does). The backward must still
    write zeros to every non-winning ``(j, t)`` output cell."""
    from late_interaction_kernels import maxsim

    Nq, Nd, Lq, Ld, d = 2, 2, 8, 64, 128
    Q = torch.ones(Nq, Lq, d, device="cuda", dtype=torch.float32)
    D = -torch.ones(Nd, Ld, d, device="cuda", dtype=torch.float32)
    D[:, 3, :] = 1.0
    go = torch.randn(Nq, Nd, device="cuda", dtype=torch.float32)

    Qv = Q.clone().requires_grad_(True)
    Dv = D.clone().requires_grad_(True)
    maxsim(Qv, Dv, backward=method).backward(go)
    mask = torch.ones(Ld, dtype=torch.bool, device="cuda")
    mask[3] = False
    assert Dv.grad[:, mask, :].abs().max().item() == 0.0
    assert Dv.grad[:, 3, :].abs().max().item() > 0.0


# --------------------------------------------------------------------------- #
# 5. Determinism                                                              #
# --------------------------------------------------------------------------- #


def test_lowmem_is_bitwise_deterministic():
    """lowmem is genuinely bitwise-deterministic: each doc-token row reduces in
    a fixed order inside one program, no atomics. Running backward three times on
    the same input must produce the exact same grad tensors."""
    from late_interaction_kernels import maxsim

    Q0 = torch.randn(4, 32, 128, device="cuda", dtype=torch.float32)
    D0 = torch.randn(8, 128, 128, device="cuda", dtype=torch.float32)
    go = torch.randn(4, 8, device="cuda", dtype=torch.float32)

    grads = []
    for _ in range(3):
        Q = Q0.clone().requires_grad_(True)
        D = D0.clone().requires_grad_(True)
        maxsim(Q, D, backward="lowmem").backward(go)
        grads.append((Q.grad.clone(), D.grad.clone()))
    for k in range(1, 3):
        assert torch.equal(grads[0][0], grads[k][0])
        assert torch.equal(grads[0][1], grads[k][1])


@pytest.mark.parametrize("backward", ["unified", "lowmem"])
def test_fully_d_masked_row_does_not_poison_grad_d(backward):
    """Argmax sentinel: a query row whose docs are all masked must not write
    a spurious atomic-add into ``grad_D[d_global, 0, :]``.

    The forward saves ``argmax = -1`` (sentinel) for that row; the backward
    must skip the load + atomic-add for ``t < 0``. Regression for the bug
    where ``m_idx`` defaulted to ``0`` in the forward.
    """
    from late_interaction_kernels import maxsim

    torch.manual_seed(0)
    Nq, Nd, Lq, Ld, d = 2, 3, 8, 16, 64
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float32, requires_grad=True)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float32, requires_grad=True)
    # Mask every doc token of doc j=1. The (i, j=1) pairs have no valid winner
    # for any query token → argmax stays at the sentinel for those slots.
    d_mask = torch.ones(Nd, Ld, dtype=torch.bool, device="cuda")
    d_mask[1, :] = False

    grad_out = torch.randn(Nq, Nd, device="cuda", dtype=torch.float32)
    maxsim(Q, D, d_mask=d_mask, backward=backward).backward(grad_out)

    # The whole D[1, :, :] slab must receive *zero* gradient — every (i, j=1, s)
    # row was fully masked in the forward.
    assert torch.equal(D.grad[1], torch.zeros_like(D.grad[1])), (
        f"grad_D[1] has non-zero entries on the {backward} path; the argmax sentinel guard is missing."
    )


def test_unified_is_numerically_stable_across_runs(rel):
    """fp32 ``atomic_add`` reduction order depends on thread scheduling, so the
    unified path is *not* strictly bitwise-reproducible across runs — but the
    drift is bounded by fp32 ULP (~1e-6 relative). grad_Q (no atomics) still
    matches exactly."""
    from late_interaction_kernels import maxsim

    Q0 = torch.randn(4, 32, 128, device="cuda", dtype=torch.float32)
    D0 = torch.randn(8, 128, 128, device="cuda", dtype=torch.float32)
    go = torch.randn(4, 8, device="cuda", dtype=torch.float32)

    grads = []
    for _ in range(3):
        Q = Q0.clone().requires_grad_(True)
        D = D0.clone().requires_grad_(True)
        maxsim(Q, D, backward="unified").backward(go)
        grads.append((Q.grad.clone(), D.grad.clone()))
    for k in range(1, 3):
        assert torch.equal(grads[0][0], grads[k][0])  # grad_Q is scatter-free
        assert rel(grads[0][1], grads[k][1]) < 1e-5
