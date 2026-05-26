"""Unified ``maxsim()`` API: ``D.dim() == 4`` → packed KD path.

Pins the dispatch contract that lets PyLate / ColPali drop their per-query
Python ``for`` loop and call ``lik.maxsim(Q, D_kd)`` directly with a 4-D
candidate tensor (PyLate's ``colbert_kd_scores`` layout).
"""

from __future__ import annotations

import pytest
import torch

from late_interaction_kernels import maxsim
from late_interaction_kernels.padded import maxsim_padded
from late_interaction_kernels.reference import maxsim_padded_reference

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _make_kd(
    Nq: int = 4,
    K: int = 6,
    Lq: int = 17,
    Ld: int = 91,
    d: int = 128,
    *,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)
    Q = torch.randn(Nq, Lq, d, generator=g, device=device, dtype=dtype)
    D = torch.randn(Nq, K, Ld, d, generator=g, device=device, dtype=dtype)
    return Q, D


# --------------------------------------------------------------------------- #
# Dispatch correctness                                                         #
# --------------------------------------------------------------------------- #


@cuda_only
def test_kd_dispatch_shape() -> None:
    """``D.dim() == 4`` returns ``[Nq, K]``, not the in-batch ``[Nq, Nq*K]``."""
    Nq, K = 5, 8
    Q, D = _make_kd(Nq=Nq, K=K)
    out = maxsim(Q, D)
    assert out.shape == (Nq, K)
    assert out.dtype == torch.float32


@cuda_only
def test_kd_dispatch_matches_maxsim_padded() -> None:
    """``maxsim(Q, D_kd)`` (fast path) and ``maxsim_padded(Q, D_kd, full_lens)``
    (packed pair path) agree within fp16 accumulation noise.

    They use the same math and same fp32 accumulator but different kernels:
    the fast path runs ``_maxsim_fwd_kernel`` with ``kd_layout=True`` (full
    ``tl.static_range`` over Lq); the packed path runs ``_scatter_fwd_kernel``
    with a dynamic Lq loop. Different tile orderings + different autotune
    configs mean the sum of ``Lq`` per-token maxes lands on slightly
    different fp32 representations — within ``5e-3`` for fp16 inputs.
    """
    Q, D = _make_kd()
    Nq, K, Ld, _ = D.shape
    Lq = Q.shape[1]

    qlen = torch.full((Nq,), Lq, device=Q.device, dtype=torch.int32)
    dlen = torch.full((Nq, K), Ld, device=D.device, dtype=torch.int32)

    out_dispatch = maxsim(Q, D)
    out_padded = maxsim_padded(Q, D, qlen, dlen)

    torch.testing.assert_close(out_dispatch, out_padded, rtol=5e-3, atol=5e-3)


@cuda_only
def test_kd_dispatch_matches_reference() -> None:
    """Fused kernel output matches the pure-PyTorch reference within fp16 noise."""
    Q, D = _make_kd()
    Nq, K, Ld, _ = D.shape
    Lq = Q.shape[1]

    qlen = torch.full((Nq,), Lq, device=Q.device, dtype=torch.int32)
    dlen = torch.full((Nq, K), Ld, device=D.device, dtype=torch.int32)

    out_fast = maxsim(Q, D)
    out_ref = maxsim_padded_reference(Q.float(), D.float(), qlen, dlen)
    torch.testing.assert_close(out_fast, out_ref, rtol=5e-3, atol=5e-3)


@cuda_only
def test_kd_dispatch_handles_masks_as_lengths() -> None:
    """Contiguous-prefix masks convert to lengths and reproduce the lengths path."""
    Nq, K, Lq, Ld, d = 3, 4, 20, 50, 128
    Q, D = _make_kd(Nq=Nq, K=K, Lq=Lq, Ld=Ld, d=d)

    qlen_int = torch.tensor([Lq, Lq - 3, Lq - 7], device=Q.device, dtype=torch.int32)
    dlen_int = torch.randint(1, Ld + 1, (Nq, K), device=D.device, dtype=torch.int32)

    q_mask = torch.arange(Lq, device=Q.device)[None, :] < qlen_int[:, None]
    d_mask = torch.arange(Ld, device=D.device)[None, None, :] < dlen_int[:, :, None]

    out_via_mask = maxsim(Q, D, q_mask=q_mask, d_mask=d_mask)
    out_via_len = maxsim_padded(Q, D, qlen_int, dlen_int)
    # Same parity bar as test_kd_dispatch_matches_maxsim_padded — the two
    # kernels share the math but differ in tile order / autotune choice.
    torch.testing.assert_close(out_via_mask, out_via_len, rtol=5e-3, atol=5e-3)


@cuda_only
def test_kd_dispatch_normalize_matches_pre_normalized() -> None:
    """``normalize=True`` on the KD path matches feeding pre-normalized tensors."""
    Q, D = _make_kd()
    Nq, K, Ld, _ = D.shape
    Lq = Q.shape[1]

    Q_hat = torch.nn.functional.normalize(Q.float(), p=2, dim=-1)
    D_hat = torch.nn.functional.normalize(D.float(), p=2, dim=-1)
    qlen = torch.full((Nq,), Lq, device=Q.device, dtype=torch.int32)
    dlen = torch.full((Nq, K), Ld, device=D.device, dtype=torch.int32)

    out_norm_flag = maxsim(Q, D, normalize=True)
    out_pre_norm = maxsim_padded(Q_hat.to(Q.dtype), D_hat.to(D.dtype), qlen, dlen)
    torch.testing.assert_close(out_norm_flag, out_pre_norm, rtol=2e-3, atol=2e-3)


# --------------------------------------------------------------------------- #
# Gradient correctness                                                         #
# --------------------------------------------------------------------------- #


@cuda_only
def test_kd_dispatch_grad_shape_and_finite() -> None:
    """Gradients flow back to Q ``[Nq, Lq, d]`` and D ``[Nq, K, Ld, d]``."""
    Q, D = _make_kd()
    Q = Q.detach().requires_grad_(True)
    D = D.detach().requires_grad_(True)

    maxsim(Q, D).sum().backward()

    assert Q.grad is not None and Q.grad.shape == Q.shape
    assert D.grad is not None and D.grad.shape == D.shape
    assert torch.all(torch.isfinite(Q.grad))
    assert torch.all(torch.isfinite(D.grad))


@cuda_only
def test_kd_dispatch_grad_matches_padded_path() -> None:
    """grad_Q / grad_D from the wrapper agree with calling maxsim_padded directly."""
    Q, D = _make_kd()
    Nq, K, Ld, _ = D.shape
    Lq = Q.shape[1]
    qlen = torch.full((Nq,), Lq, device=Q.device, dtype=torch.int32)
    dlen = torch.full((Nq, K), Ld, device=D.device, dtype=torch.int32)

    Q_a, D_a = Q.detach().clone().requires_grad_(True), D.detach().clone().requires_grad_(True)
    Q_b, D_b = Q.detach().clone().requires_grad_(True), D.detach().clone().requires_grad_(True)

    maxsim(Q_a, D_a).sum().backward()
    maxsim_padded(Q_b, D_b, qlen, dlen).sum().backward()

    # See test_kd_dispatch_matches_maxsim_padded: same math, different tile
    # ordering between the two kernels → fp16-scale noise on grads.
    torch.testing.assert_close(Q_a.grad, Q_b.grad, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(D_a.grad, D_b.grad, rtol=5e-3, atol=5e-3)


# --------------------------------------------------------------------------- #
# Error paths                                                                  #
# --------------------------------------------------------------------------- #


@cuda_only
def test_kd_dispatch_rejects_2d_Q() -> None:
    """KD layout needs batched Q — calling with a single query is a contract bug."""
    Q = torch.randn(7, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(2, 3, 64, 128, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="needs Q to be"):
        maxsim(Q, D)


@cuda_only
def test_kd_dispatch_rejects_mismatched_Nq() -> None:
    """``Q.shape[0]`` and ``D.shape[0]`` must agree on the KD path."""
    Q = torch.randn(3, 16, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(5, 4, 32, 128, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="D.shape\\[0\\] == Q.shape\\[0\\]"):
        maxsim(Q, D)


@cuda_only
def test_kd_dispatch_rejects_bad_qmask_shape() -> None:
    """q_mask shape mismatch on the KD path errors with a clear message."""
    Q, D = _make_kd(Nq=4, Lq=16)
    bad_mask = torch.ones(4, 17, dtype=torch.bool, device="cuda")  # off-by-one
    with pytest.raises(ValueError, match="q_mask must be"):
        maxsim(Q, D, q_mask=bad_mask)
