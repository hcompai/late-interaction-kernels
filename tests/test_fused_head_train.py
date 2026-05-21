"""Backward parity tests for :func:`maxsim_from_hidden`.

The fused head is autograd-aware (auto-dispatched on ``requires_grad``).
The backward must:

* flow gradients to ``Q``, ``H_d``, ``W``, ``b`` matching the unfused
  ``F.linear + F.normalize + maxsim`` path (bf16 tolerance),
* allocate gradient buffers only for tensors that actually requested
  them.

Forward parity is covered by :mod:`test_fused_head` — the fused head
runs the same kernel whether ``requires_grad`` is set or not, so we
don't duplicate that here.
"""

import pytest
import torch


def _unfused_reference(Q, H_d, W, b, *, normalize, d_mask=None):
    """Canonical unfused path: linear + normalize + maxsim, all autograd-tracked.

    Harmonizes dtypes so mixed fp32 / bf16 leaves (caller decides which ones
    need gradients) don't blow up ``F.linear``.
    """
    from late_interaction_kernels.reference import maxsim_reference

    target = torch.promote_types(torch.promote_types(Q.dtype, H_d.dtype), W.dtype)
    if b is not None:
        target = torch.promote_types(target, b.dtype)
    Q_c = Q.to(target)
    H_c = H_d.to(target)
    W_c = W.to(target)
    b_c = b.to(target) if b is not None else None

    D_proj = torch.nn.functional.linear(H_c, W_c, b_c)
    if normalize:
        D_proj = torch.nn.functional.normalize(D_proj, p=2, dim=-1, eps=1e-12)
    return maxsim_reference(Q_c, D_proj, d_mask=d_mask)


@pytest.mark.cuda
@pytest.mark.parametrize("need_grads", [["Q"], ["H_d"], ["W"], ["Q", "H_d", "W", "b"]])
def test_fused_head_backward_matches_unfused(need_grads):
    """All requested gradients must match the unfused autograd path.

    Uses fp32 inputs on purpose: bf16 ties on the inner argmax flip the
    winning doc-token between the kernel (fp32 D_proj accumulator) and
    the eager path (bf16 ``F.linear`` → bf16 D_proj), which injects 100 %
    relative error on a small fraction of ``Q`` rows. With fp32 inputs
    the winner is deterministic and gradients match tightly.
    """
    from late_interaction_kernels.fused_head import maxsim_from_hidden

    Nq, Nd, Lq, Ld, d_model, d_out = 1, 4, 16, 64, 256, 64
    dtype = torch.float32
    torch.manual_seed(0)

    def _make():
        H_d = torch.randn(Nd, Ld, d_model, device="cuda", dtype=dtype)
        W = torch.randn(d_out, d_model, device="cuda", dtype=dtype) * (1.0 / (d_model**0.5))
        b = torch.randn(d_out, device="cuda", dtype=dtype) * 0.01
        Q = torch.nn.functional.normalize(
            torch.randn(Nq, Lq, d_out, device="cuda", dtype=dtype).float(), dim=-1
        ).to(dtype)
        for name, t in [("Q", Q), ("H_d", H_d), ("W", W), ("b", b)]:
            if name in need_grads:
                t.requires_grad_(True)
        return Q, H_d, W, b

    torch.manual_seed(0)
    Q, H_d, W, b = _make()
    out = maxsim_from_hidden(Q, H_d, W, b=b, normalize=True)
    out.sum().backward()
    g_fused = {"Q": Q.grad, "H_d": H_d.grad, "W": W.grad, "b": b.grad}

    torch.manual_seed(0)
    Q2, H_d2, W2, b2 = _make()
    ref = _unfused_reference(
        Q2.float() if not Q2.requires_grad else Q2,
        H_d2.float() if not H_d2.requires_grad else H_d2,
        W2.float() if not W2.requires_grad else W2,
        b2.float() if not b2.requires_grad else b2,
        normalize=True,
    )
    ref.sum().backward()
    g_ref = {"Q": Q2.grad, "H_d": H_d2.grad, "W": W2.grad, "b": b2.grad}

    for name in need_grads:
        a, r = g_fused[name], g_ref[name]
        assert a is not None, f"missing {name} gradient"
        assert r is not None, f"missing reference {name} gradient"
        # bf16 MaxSim has argmax ties where a single doc-token slot can
        # bit-equal its neighbor, which flips the winner between kernel
        # and reference and makes ``max(|Δgrad|) / max(|grad|)`` jump to
        # ~1.0 at that one slot. The training signal is unaffected —
        # we compare in RMS (Frobenius) space instead.
        a_f = a.float()
        r_f = r.float()
        rmse = (a_f - r_f).pow(2).mean().sqrt().item()
        rms_ref = r_f.pow(2).mean().sqrt().clamp_min(1e-6).item()
        rel = rmse / rms_ref
        assert rel < 2e-2, f"{name} rel_rms={rel:.3e}"


@pytest.mark.cuda
def test_fused_head_only_active_grads_filled():
    """If a tensor doesn't require grad we must not silently allocate for it."""
    from late_interaction_kernels.fused_head import maxsim_from_hidden

    Nq, Nd, Lq, Ld, d_model, d_out = 1, 4, 8, 16, 128, 64
    H_d = torch.randn(Nd, Ld, d_model, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(d_out, d_model, device="cuda", dtype=torch.bfloat16) * (1.0 / (d_model**0.5))
    Q = (
        torch.nn.functional.normalize(
            torch.randn(Nq, Lq, d_out, device="cuda", dtype=torch.bfloat16).float(), dim=-1
        )
        .to(torch.bfloat16)
        .requires_grad_(True)
    )
    # Only Q needs grad.
    out = maxsim_from_hidden(Q, H_d, W, normalize=True)
    out.sum().backward()
    assert Q.grad is not None
    assert W.grad is None
    assert H_d.grad is None


@pytest.mark.cuda
def test_fused_head_no_grad_dispatch_matches_grad_path():
    """Forward result must be identical whether the autograd path ran or not.

    Pins the ``requires_grad`` dispatch in :func:`maxsim_from_hidden`: the
    no-grad path skips the argmax save but otherwise runs the same kernel.
    """
    from late_interaction_kernels.fused_head import maxsim_from_hidden

    Nq, Nd, Lq, Ld, d_model, d_out = 2, 8, 32, 128, 768, 128
    H_d = torch.randn(Nd, Ld, d_model, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(d_out, d_model, device="cuda", dtype=torch.bfloat16) * (1.0 / (d_model**0.5))
    b = torch.randn(d_out, device="cuda", dtype=torch.bfloat16) * 0.01
    Q = torch.nn.functional.normalize(
        torch.randn(Nq, Lq, d_out, device="cuda", dtype=torch.bfloat16).float(), dim=-1
    ).to(torch.bfloat16)

    no_grad = maxsim_from_hidden(Q, H_d, W, b=b, normalize=True)
    with_grad = maxsim_from_hidden(Q.clone().requires_grad_(True), H_d, W, b=b, normalize=True).detach()
    torch.testing.assert_close(no_grad, with_grad, atol=0, rtol=0)
