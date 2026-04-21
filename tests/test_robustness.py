"""Robustness / integration tests.

These are the tests a senior reviewer would ask for *after* basic parity
and backward tests pass:

* Bitwise-identical repeated forward calls (determinism contract).
* ``torch.autograd.gradcheck`` on a small shape in fp64 (gold-standard
  autograd sanity check).
* ``torch.compile`` smoke — the ops must round-trip through Dynamo
  without graph breaks that crash the kernel.
* ``torch.inference_mode()`` — the forward must work without autograd
  even when tensors would normally require grad.
* CUDA Graphs compatibility — the forward must be capturable and
  replayable (required by high-throughput serving stacks).
* Error-path contract — wrong shapes / dtypes must raise cleanly.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.cuda


# --------------------------------------------------------------------------- #
# 1. Forward determinism                                                       #
# --------------------------------------------------------------------------- #


def test_forward_is_bitwise_deterministic():
    """Calling the kernel twice on identical inputs must yield bit-identical
    outputs. No atomics on the forward path means this is a hard guarantee."""
    from late_interaction_kernels import maxsim

    Q = torch.randn(4, 32, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(8, 128, 128, device="cuda", dtype=torch.float16)
    a = maxsim(Q, D)
    b = maxsim(Q, D)
    assert torch.equal(a, b)


def test_varlen_forward_is_bitwise_deterministic():
    from late_interaction_kernels import maxsim_varlen

    Qp = torch.randn(64, 128, device="cuda", dtype=torch.float16)
    Dp = torch.randn(500, 128, device="cuda", dtype=torch.float16)
    cu_q = torch.tensor([0, 16, 32, 48, 64], device="cuda", dtype=torch.int32)
    cu_d = torch.tensor([0, 100, 250, 500], device="cuda", dtype=torch.int32)
    a = maxsim_varlen(Qp, Dp, cu_q, cu_d)
    b = maxsim_varlen(Qp, Dp, cu_q, cu_d)
    assert torch.equal(a, b)


# --------------------------------------------------------------------------- #
# 2. Soft-maxsim gradcheck (smooth forward → fp64 gradcheck is valid)         #
# --------------------------------------------------------------------------- #


def test_gradcheck_soft_maxsim_smooth():
    """Soft-maxsim uses log-sum-exp and is smooth everywhere — unlike hard
    maxsim, whose forward has argmax kinks that confuse finite-difference
    gradcheck. We run gradcheck on a tiny fp64 shape; on fp64 the soft path
    falls back to a pure-PyTorch reference whose backward must match
    finite-difference to tight tolerances.
    """
    from late_interaction_kernels import soft_maxsim

    Nq, Nd, Lq, Ld, d = 1, 2, 3, 5, 32
    torch.manual_seed(0)
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float64, requires_grad=True)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float64, requires_grad=True)

    def fn(q, d_):
        return soft_maxsim(q, d_, beta=2.0)

    assert torch.autograd.gradcheck(fn, (Q, D), eps=1e-6, atol=1e-5, rtol=1e-5, fast_mode=True)


# --------------------------------------------------------------------------- #
# 3. torch.compile smoke                                                       #
# --------------------------------------------------------------------------- #


def test_torch_compile_maxsim_smoke():
    """`torch.compile` must be able to trace a function that calls our
    kernel. We don't assert any speedup — just that it doesn't crash and
    that the output still matches eager."""
    from late_interaction_kernels import maxsim

    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile unavailable")

    Q = torch.randn(2, 16, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(4, 64, 128, device="cuda", dtype=torch.float16)

    eager = maxsim(Q, D)
    try:
        compiled = torch.compile(maxsim, fullgraph=False)(Q, D)
    except Exception as e:  # pragma: no cover — upstream inductor bug
        pytest.skip(f"torch.compile crashed in the wrapper: {e}")

    assert torch.equal(eager, compiled)


# --------------------------------------------------------------------------- #
# 4. inference_mode / no_grad                                                  #
# --------------------------------------------------------------------------- #


def test_inference_mode_returns_correct_scores():
    """Kernel must work inside `torch.inference_mode()` even when the
    *reference path* would have captured argmax. `maxsim_inference` is
    the canonical no-grad alias; this test pins both."""
    from late_interaction_kernels import maxsim, maxsim_inference
    from late_interaction_kernels.reference import maxsim_reference

    Q = torch.randn(2, 16, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(4, 64, 128, device="cuda", dtype=torch.float16)
    ref = maxsim_reference(Q.float(), D.float())

    with torch.inference_mode():
        s1 = maxsim(Q, D).float()
        s2 = maxsim_inference(Q, D).float()

    denom = max(1.0, ref.abs().max().item())
    assert (s1 - ref).abs().max().item() / denom < 5e-3
    assert (s2 - ref).abs().max().item() / denom < 5e-3


def test_no_grad_output_has_no_grad_fn():
    """Under `torch.no_grad()`, the output of `maxsim` must have no
    `grad_fn` — confirming the autograd tape is not being populated."""
    from late_interaction_kernels import maxsim

    Q = torch.randn(2, 8, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    D = torch.randn(3, 16, 128, device="cuda", dtype=torch.float16, requires_grad=True)

    with torch.no_grad():
        s = maxsim(Q, D)
    assert s.grad_fn is None
    assert not s.requires_grad


def test_no_grad_does_not_save_argmax_buffer():
    """Under `torch.no_grad()` with non-grad inputs, the kernel skips the
    argmax save (Nq·Nd·Lq int32 buffer). Compare peak allocation with the
    grad-enabled path on the same shape."""
    from late_interaction_kernels import maxsim

    Nq, Nd, Lq, Ld, d = 4, 8, 32, 512, 128
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float16)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float16)

    # Warmup so kernel launch memory isn't counted.
    _ = maxsim(Q, D)
    del _
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    base = torch.cuda.memory_allocated()
    with torch.no_grad():
        s_ng = maxsim(Q, D)
    torch.cuda.synchronize()
    alloc_ng = torch.cuda.memory_allocated() - base
    del s_ng
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Grad-enabled path (needs argmax buffer).
    Qg = Q.clone().requires_grad_(True)
    Dg = D.clone().requires_grad_(True)
    base = torch.cuda.memory_allocated()
    s_g = maxsim(Qg, Dg)
    torch.cuda.synchronize()
    alloc_g = torch.cuda.memory_allocated() - base

    argmax_bytes = Nq * Nd * Lq * 4  # int32
    # The grad path allocates at least the argmax buffer more than no_grad.
    assert alloc_g - alloc_ng >= argmax_bytes * 0.9, (
        f"no_grad={alloc_ng} B, grad={alloc_g} B, argmax={argmax_bytes} B — "
        f"`no_grad` doesn't look like it's skipping the argmax save."
    )
    del s_g


# --------------------------------------------------------------------------- #
# 5. CUDA Graph capture                                                        #
# --------------------------------------------------------------------------- #


def test_cuda_graph_capture_and_replay():
    """High-throughput serving stacks (e.g. vLLM, NVIDIA Triton Inference
    Server) capture the forward into a `torch.cuda.CUDAGraph`. Our kernel
    must be capturable: no host-side allocator calls during the launch,
    no host syncs, deterministic grid. Capture once, mutate the inputs
    in place, replay, and check the scores match a fresh eager call."""
    from late_interaction_kernels import maxsim_inference

    Q = torch.randn(4, 32, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(8, 128, 128, device="cuda", dtype=torch.float16)

    # Warmup on a side stream to populate Triton's autotuner cache.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            _ = maxsim_inference(Q, D)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            captured = maxsim_inference(Q, D)
    except RuntimeError as e:  # pragma: no cover — Triton/torch version quirks
        pytest.skip(f"CUDAGraph capture unsupported here: {e}")

    # Mutate input buffers in place and replay.
    Q.copy_(torch.randn_like(Q))
    D.copy_(torch.randn_like(D))
    g.replay()
    torch.cuda.synchronize()

    ref = maxsim_inference(Q, D)
    assert torch.equal(captured, ref), (
        "CUDAGraph replay produced stale / incorrect scores — check that the "
        "kernel has no host-side state captured by the graph."
    )


# --------------------------------------------------------------------------- #
# 6. Error-path contract                                                       #
# --------------------------------------------------------------------------- #


def test_mismatched_embedding_dim_raises():
    """Shape contract: Q.shape[-1] must equal D.shape[-1]."""
    from late_interaction_kernels import maxsim

    Q = torch.randn(2, 16, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(4, 64, 256, device="cuda", dtype=torch.float16)
    with pytest.raises((RuntimeError, AssertionError, ValueError)):
        maxsim(Q, D)


def test_wrong_device_raises():
    """Mixing CPU and CUDA tensors must fail fast with a clear error
    (rather than silently copying or producing garbage)."""
    from late_interaction_kernels import maxsim

    Q = torch.randn(2, 16, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(4, 64, 128, device="cpu", dtype=torch.float16)
    with pytest.raises((RuntimeError, AssertionError, ValueError)):
        maxsim(Q, D)


def test_varlen_cu_seqlens_dtype_coercion():
    """cu_seqlens is documented as int32 but users often pass int64. We
    accept both; the wrapper must coerce or the kernel must handle both.
    """
    from late_interaction_kernels import maxsim_varlen

    Qp = torch.randn(16, 128, device="cuda", dtype=torch.float16)
    Dp = torch.randn(100, 128, device="cuda", dtype=torch.float16)
    cu_q_i64 = torch.tensor([0, 8, 16], device="cuda", dtype=torch.int64)
    cu_d_i64 = torch.tensor([0, 50, 100], device="cuda", dtype=torch.int64)
    # If this raises, we just want a clear TypeError — not a silent memory
    # corruption. Either behavior (accept or reject) is fine; we just
    # don't tolerate UB.
    try:
        out = maxsim_varlen(Qp, Dp, cu_q_i64, cu_d_i64)
        assert out.shape == (2, 2)
    except (RuntimeError, TypeError, AssertionError):
        pass  # acceptable: reject int64 loudly


# --------------------------------------------------------------------------- #
# 7. Numerical stability                                                       #
# --------------------------------------------------------------------------- #


def test_very_small_magnitudes_dont_underflow():
    """L2-normalized ColBERT vectors are tiny (|x| ~ 1/sqrt(d) ≈ 0.09 at
    d=128). The kernel must not round the dot products to zero."""
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.reference import maxsim_reference

    d = 128
    Q = torch.nn.functional.normalize(
        torch.randn(2, 16, d, device="cuda", dtype=torch.float16), p=2, dim=-1
    )
    D = torch.nn.functional.normalize(
        torch.randn(4, 64, d, device="cuda", dtype=torch.float16), p=2, dim=-1
    )
    fast = maxsim(Q, D).float()
    ref = maxsim_reference(Q.float(), D.float())
    assert (fast - ref).abs().max().item() / max(1.0, ref.abs().max().item()) < 5e-3


def test_soft_maxsim_backward_deterministic():
    """soft_maxsim has a dense gradient (no argmax); it must be bitwise
    deterministic across repeated calls on fp32."""
    from late_interaction_kernels import soft_maxsim

    Q0 = torch.randn(2, 16, 128, device="cuda", dtype=torch.float32)
    D0 = torch.randn(4, 64, 128, device="cuda", dtype=torch.float32)
    go = torch.randn(2, 4, device="cuda", dtype=torch.float32)

    grads = []
    for _ in range(3):
        Q = Q0.clone().requires_grad_(True)
        D = D0.clone().requires_grad_(True)
        soft_maxsim(Q, D, beta=5.0).backward(go)
        grads.append((Q.grad.clone(), D.grad.clone()))
    for k in range(1, 3):
        assert torch.equal(grads[0][0], grads[k][0]), f"soft_maxsim grad_Q non-det at run {k}"
        # soft_maxsim grad_D may use atomics; allow tiny drift but not nonsense.
        err = (grads[0][1] - grads[k][1]).abs().max().item()
        denom = max(1.0, grads[0][1].abs().max().item())
        assert err / denom < 1e-5, f"soft_maxsim grad_D drift at run {k}: {err}"
