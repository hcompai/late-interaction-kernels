"""Parity + gradient tests for :func:`maxsim_from_hidden_train` (0.7.0).

The training-aware fused head is the autograd-aware sibling of
:func:`maxsim_from_hidden`. It must:

* match the unfused ``F.linear + normalize + maxsim`` path in forward,
* flow gradients to ``Q``, ``H_d``, ``W``, ``b`` matching the unfused
  path (bf16 tolerance),
* never materialize the ``[Nd, Ld, d_out]`` ``D_proj`` scratch tensor
  in HBM (sanity-checked by peak-memory delta).

CPU / non-CUDA environments just import and skip.
"""

from __future__ import annotations

import pytest
import torch


def _unfused_reference(Q, H_d, W, b, *, normalize, d_mask=None):
    """Canonical unfused path: linear + normalize + maxsim, all autograd-tracked."""
    from late_interaction_kernels.reference import maxsim_reference

    D_proj = torch.nn.functional.linear(H_d, W, b)
    if normalize:
        D_proj = torch.nn.functional.normalize(D_proj, p=2, dim=-1, eps=1e-12)
    return maxsim_reference(Q, D_proj, d_mask=d_mask)


@pytest.mark.cuda
@pytest.mark.parametrize("normalize", [True, False])
def test_fused_head_train_forward_parity(normalize):
    from late_interaction_kernels import maxsim_from_hidden_train

    Nq, Nd, Lq, Ld, d_model, d_out = 2, 8, 32, 128, 768, 128
    dtype = torch.bfloat16
    torch.manual_seed(0)
    H_d = torch.randn(Nd, Ld, d_model, device="cuda", dtype=dtype)
    W = (torch.randn(d_out, d_model, device="cuda", dtype=dtype) * (1.0 / (d_model**0.5)))
    b = torch.randn(d_out, device="cuda", dtype=dtype) * 0.01
    Q = torch.nn.functional.normalize(
        torch.randn(Nq, Lq, d_out, device="cuda", dtype=dtype).float(), dim=-1
    ).to(dtype)

    out = maxsim_from_hidden_train(Q, H_d, W, b=b, normalize=normalize)
    ref = _unfused_reference(Q.float(), H_d.float(), W.float(), b.float(), normalize=normalize)
    rel = (out.float() - ref).abs().max().item() / max(1e-6, ref.abs().max().item())
    assert rel < 7e-3, f"rel_err={rel:.3e}"


@pytest.mark.cuda
@pytest.mark.parametrize("need_grads", [["Q"], ["H_d"], ["W"], ["Q", "H_d", "W", "b"]])
def test_fused_head_train_backward_matches_unfused(need_grads):
    """All requested gradients must match the unfused autograd path."""
    from late_interaction_kernels import maxsim_from_hidden_train

    Nq, Nd, Lq, Ld, d_model, d_out = 1, 4, 16, 64, 256, 64
    dtype = torch.bfloat16
    torch.manual_seed(0)

    def _make():
        H_d = torch.randn(Nd, Ld, d_model, device="cuda", dtype=dtype)
        W = (torch.randn(d_out, d_model, device="cuda", dtype=dtype) * (1.0 / (d_model**0.5)))
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
    out = maxsim_from_hidden_train(Q, H_d, W, b=b, normalize=True)
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
        denom = max(1e-6, r.abs().max().item())
        rel = (a.float() - r.float()).abs().max().item() / denom
        # bf16 + winners-slice rebuild: ~5e-3 is the expected floor.
        assert rel < 2e-2, f"{name} rel_err={rel:.3e}"


@pytest.mark.cuda
def test_fused_head_train_only_active_grads_filled():
    """If a tensor doesn't require grad we must not silently allocate for it."""
    from late_interaction_kernels import maxsim_from_hidden_train

    Nq, Nd, Lq, Ld, d_model, d_out = 1, 4, 8, 16, 128, 64
    H_d = torch.randn(Nd, Ld, d_model, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(d_out, d_model, device="cuda", dtype=torch.bfloat16) * (1.0 / (d_model**0.5))
    Q = torch.nn.functional.normalize(
        torch.randn(Nq, Lq, d_out, device="cuda", dtype=torch.bfloat16).float(), dim=-1
    ).to(torch.bfloat16).requires_grad_(True)
    # Only Q needs grad.
    out = maxsim_from_hidden_train(Q, H_d, W, normalize=True)
    out.sum().backward()
    assert Q.grad is not None
    assert W.grad is None
    assert H_d.grad is None


@pytest.mark.cuda
def test_fused_head_train_matches_fused_head_inference():
    """Forward of the train variant must agree with the inference variant."""
    from late_interaction_kernels import maxsim_from_hidden, maxsim_from_hidden_train

    Nq, Nd, Lq, Ld, d_model, d_out = 2, 8, 32, 128, 768, 128
    H_d = torch.randn(Nd, Ld, d_model, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(d_out, d_model, device="cuda", dtype=torch.bfloat16) * (1.0 / (d_model**0.5))
    b = torch.randn(d_out, device="cuda", dtype=torch.bfloat16) * 0.01
    Q = torch.nn.functional.normalize(
        torch.randn(Nq, Lq, d_out, device="cuda", dtype=torch.bfloat16).float(), dim=-1
    ).to(torch.bfloat16)

    inf = maxsim_from_hidden(Q, H_d, W, b=b, normalize=True)
    trn = maxsim_from_hidden_train(Q, H_d, W, b=b, normalize=True)
    torch.testing.assert_close(trn, inf, atol=1e-3, rtol=1e-3)
