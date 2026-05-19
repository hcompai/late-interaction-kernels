"""Performance regression suite — runs in GPU CI.

Times each non-experimental kernel on a small set of canonical shapes and
compares the median (p50) latency against the baseline JSON committed in
``benchmarks/baselines/{gpu_class}.json``. A test fails if the measured median
exceeds the baseline by more than 5 %.

Generate / refresh the baseline with::

    pytest tests/regression/ -m perf --update-perf-baseline

See :mod:`tests.regression._perf_utils` for the timing primitive
(``triton.testing.do_bench`` median) and the baseline I/O.
"""

from __future__ import annotations

import pytest
import torch

from tests.regression._perf_utils import check_regression, time_kernel

pytestmark = [pytest.mark.cuda, pytest.mark.perf]


# --------------------------------------------------------------------------- #
# Shared helpers                                                              #
# --------------------------------------------------------------------------- #


def _seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_varlen(seqlens: list[int], d: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    total = sum(seqlens)
    data = torch.randn(total, d, device="cuda", dtype=dtype)
    cu = torch.zeros(len(seqlens) + 1, device="cuda", dtype=torch.int32)
    cu[1:] = torch.tensor(seqlens, device="cuda", dtype=torch.int32).cumsum(0)
    return data, cu


# --------------------------------------------------------------------------- #
# maxsim — dense forward                                                      #
# --------------------------------------------------------------------------- #


_MAXSIM_FWD_SHAPES: list[tuple[str, dict, str]] = [
    # (shape_id, dict of dims, dtype string)
    ("rerank_1x256x32x200x128", {"Nq": 1, "Nd": 256, "Lq": 32, "Ld": 200, "d": 128}, "fp16"),
    ("rerank_1x256x32x200x128", {"Nq": 1, "Nd": 256, "Lq": 32, "Ld": 200, "d": 128}, "bf16"),
    ("train_16x16x32x200x128", {"Nq": 16, "Nd": 16, "Lq": 32, "Ld": 200, "d": 128}, "bf16"),
]


@pytest.mark.parametrize("shape_id,dims,dtype_str", _MAXSIM_FWD_SHAPES)
def test_perf_maxsim_forward(shape_id, dims, dtype_str, request):
    from late_interaction_kernels import maxsim_inference

    dtype = torch.float16 if dtype_str == "fp16" else torch.bfloat16
    _seed()
    Q = torch.randn(dims["Nq"], dims["Lq"], dims["d"], device="cuda", dtype=dtype)
    D = torch.randn(dims["Nd"], dims["Ld"], dims["d"], device="cuda", dtype=dtype)

    p20, p50, p80 = time_kernel(lambda: maxsim_inference(Q, D, normalize=True))
    check_regression(f"maxsim_forward_{dtype_str}", shape_id, p20, p50, p80, request=request)


# --------------------------------------------------------------------------- #
# maxsim — backward (unified + csr paths)                                     #
# --------------------------------------------------------------------------- #


_MAXSIM_BWD_SHAPES: list[tuple[str, dict]] = [
    ("train_16x16x32x200x128", {"Nq": 16, "Nd": 16, "Lq": 32, "Ld": 200, "d": 128}),
]


@pytest.mark.parametrize("method", ["unified", "csr"])
@pytest.mark.parametrize("shape_id,dims", _MAXSIM_BWD_SHAPES)
def test_perf_maxsim_backward(shape_id, dims, method, request):
    from late_interaction_kernels import maxsim

    _seed()
    Q = torch.randn(
        dims["Nq"], dims["Lq"], dims["d"], device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    D = torch.randn(
        dims["Nd"], dims["Ld"], dims["d"], device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    grad_out = torch.randn(dims["Nq"], dims["Nd"], device="cuda", dtype=torch.float32)

    # Each call: forward + backward. ``do_bench`` resets timing per call but
    # gradients accumulate, so zero them inside the closure.
    def step() -> None:
        Q.grad = None
        D.grad = None
        scores = maxsim(Q, D, normalize=True, backward=method)
        scores.backward(grad_out)

    p20, p50, p80 = time_kernel(step)
    check_regression(f"maxsim_backward_{method}", shape_id, p20, p50, p80, request=request)


# --------------------------------------------------------------------------- #
# maxsim_varlen — packed forward + backward                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("shape_id", ["varlen_8x32_mean32x200_d128"])
def test_perf_maxsim_varlen_forward(shape_id, request):
    from late_interaction_kernels import maxsim_varlen

    _seed()
    q_lens = [32] * 8
    d_lens = [200] * 32
    Qp, cu_q = _build_varlen(q_lens, 128, torch.bfloat16)
    Dp, cu_d = _build_varlen(d_lens, 128, torch.bfloat16)

    p20, p50, p80 = time_kernel(lambda: maxsim_varlen(Qp, Dp, cu_q, cu_d))
    check_regression("maxsim_varlen_forward", shape_id, p20, p50, p80, request=request)


@pytest.mark.parametrize("shape_id", ["varlen_8x32_mean32x200_d128"])
def test_perf_maxsim_varlen_backward(shape_id, request):
    from late_interaction_kernels import maxsim_varlen

    _seed()
    q_lens = [32] * 8
    d_lens = [200] * 32
    Qp, cu_q = _build_varlen(q_lens, 128, torch.bfloat16)
    Dp, cu_d = _build_varlen(d_lens, 128, torch.bfloat16)
    Qp.requires_grad_(True)
    Dp.requires_grad_(True)
    grad_out = torch.randn(8, 32, device="cuda", dtype=torch.float32)

    def step() -> None:
        Qp.grad = None
        Dp.grad = None
        scores = maxsim_varlen(Qp, Dp, cu_q, cu_d)
        scores.backward(grad_out)

    p20, p50, p80 = time_kernel(step)
    check_regression("maxsim_varlen_backward", shape_id, p20, p50, p80, request=request)


# --------------------------------------------------------------------------- #
# maxsim_padded — padded → packed wrapper                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("shape_id", ["padded_8x8_32x200_d128"])
def test_perf_maxsim_padded(shape_id, request):
    from late_interaction_kernels import maxsim_padded

    _seed()
    B, C, Lq, Ld, d = 8, 8, 32, 200, 128
    queries = torch.randn(B, Lq, d, device="cuda", dtype=torch.bfloat16)
    documents = torch.randn(B, C, Ld, d, device="cuda", dtype=torch.bfloat16)
    qlen = torch.full((B,), Lq, device="cuda", dtype=torch.int32)
    dlen = torch.full((B, C), Ld, device="cuda", dtype=torch.int32)

    p20, p50, p80 = time_kernel(lambda: maxsim_padded(queries, documents, qlen, dlen))
    check_regression("maxsim_padded", shape_id, p20, p50, p80, request=request)


# --------------------------------------------------------------------------- #
# score_pairs_packed — sparse pair scheduling                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("shape_id", ["pairs_64_q8d32_32x200_d128"])
def test_perf_score_pairs_packed(shape_id, request):
    from late_interaction_kernels.score_pairs import score_pairs_packed

    _seed()
    Nq, Nd, Lq, Ld, d = 8, 32, 32, 200, 128
    Qp, cu_q = _build_varlen([Lq] * Nq, d, torch.bfloat16)
    Dp, cu_d = _build_varlen([Ld] * Nd, d, torch.bfloat16)
    # 64 random (q, d) pairs.
    num_pairs = 64
    pair_q = torch.randint(0, Nq, (num_pairs,), device="cuda", dtype=torch.int32)
    pair_d = torch.randint(0, Nd, (num_pairs,), device="cuda", dtype=torch.int32)

    p20, p50, p80 = time_kernel(
        lambda: score_pairs_packed(Qp, Dp, cu_q, cu_d, pair_q, pair_d, max_seqlen_q=Lq, max_seqlen_d=Ld)
    )
    check_regression("score_pairs_packed", shape_id, p20, p50, p80, request=request)


# --------------------------------------------------------------------------- #
# retrieve — top-k                                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("shape_id", ["retrieve_4x4096_32x200_d128_k100"])
def test_perf_retrieve(shape_id, request):
    from late_interaction_kernels import retrieve

    _seed()
    Nq, Nd, Lq, Ld, d, top_k = 4, 4096, 32, 200, 128, 100
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.bfloat16)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.bfloat16)

    p20, p50, p80 = time_kernel(lambda: retrieve(Q, D, top_k=top_k, normalize=True, chunk=1024))
    check_regression("retrieve", shape_id, p20, p50, p80, request=request)


# --------------------------------------------------------------------------- #
# fused_head — D-side Linear+Normalize+MaxSim                                 #
# --------------------------------------------------------------------------- #


_FUSED_HEAD_SHAPE = {"Nq": 8, "Nd": 64, "Lq": 32, "Ld": 200, "d_model": 1024, "d_out": 128}


@pytest.mark.parametrize("shape_id,dims", [("fused_8x64_32x200_dm1024_dout128", _FUSED_HEAD_SHAPE)])
def test_perf_maxsim_from_hidden(shape_id, dims, request):
    from late_interaction_kernels.fused_head import maxsim_from_hidden

    _seed()
    Q = torch.randn(dims["Nq"], dims["Lq"], dims["d_out"], device="cuda", dtype=torch.bfloat16)
    H_d = torch.randn(dims["Nd"], dims["Ld"], dims["d_model"], device="cuda", dtype=torch.bfloat16)
    W = torch.randn(dims["d_out"], dims["d_model"], device="cuda", dtype=torch.bfloat16)
    b = torch.randn(dims["d_out"], device="cuda", dtype=torch.bfloat16)

    p20, p50, p80 = time_kernel(lambda: maxsim_from_hidden(Q, H_d, W, b, normalize=True))
    check_regression("maxsim_from_hidden", shape_id, p20, p50, p80, request=request)


@pytest.mark.parametrize("shape_id,dims", [("fused_8x64_32x200_dm1024_dout128", _FUSED_HEAD_SHAPE)])
def test_perf_maxsim_from_hidden_train(shape_id, dims, request):
    from late_interaction_kernels.fused_head import maxsim_from_hidden_train

    _seed()
    Q = torch.randn(
        dims["Nq"], dims["Lq"], dims["d_out"], device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    H_d = torch.randn(
        dims["Nd"], dims["Ld"], dims["d_model"], device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    W = torch.randn(dims["d_out"], dims["d_model"], device="cuda", dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn(dims["d_out"], device="cuda", dtype=torch.bfloat16, requires_grad=True)
    grad_out = torch.randn(dims["Nq"], dims["Nd"], device="cuda", dtype=torch.float32)

    def step() -> None:
        for t in (Q, H_d, W, b):
            t.grad = None
        scores = maxsim_from_hidden_train(Q, H_d, W, b, normalize=True)
        scores.backward(grad_out)

    p20, p50, p80 = time_kernel(step)
    check_regression("maxsim_from_hidden_train", shape_id, p20, p50, p80, request=request)


# --------------------------------------------------------------------------- #
# PLAID — approx (IVF prune), residual rerank, varlen residual                #
# --------------------------------------------------------------------------- #


def _make_quant_index(Nd: int, max_Ld: int, d: int, n_centroids: int, nbits: int, seed: int = 0):
    """Synthetic PLAID-style index, mirrors tests/test_plaid.py:_make_quant_index."""
    torch.manual_seed(seed)
    centroids = torch.randn(n_centroids, d, device="cuda", dtype=torch.float32) * 0.3
    n_buckets = 2**nbits
    bucket_weights = torch.linspace(-0.1, 0.1, n_buckets, device="cuda", dtype=torch.float32)
    codes_per_byte = 8 // nbits
    packed_dim = (d * nbits + 7) // 8
    codes = torch.randint(0, n_centroids, (Nd, max_Ld), device="cuda", dtype=torch.int64)
    bucket_codes = torch.randint(0, n_buckets, (Nd, max_Ld, d), device="cuda", dtype=torch.int64)
    residuals = torch.zeros(Nd, max_Ld, packed_dim, device="cuda", dtype=torch.uint8)
    for f in range(d):
        byte_idx = f // codes_per_byte
        slot = f % codes_per_byte
        residuals[..., byte_idx] |= bucket_codes[..., f].to(torch.uint8) << (slot * nbits)
    doc_lengths = torch.full((Nd,), max_Ld, device="cuda", dtype=torch.int64)
    return centroids, bucket_weights, codes, residuals, doc_lengths


@pytest.mark.parametrize("shape_id", ["approx_B512_c2048_Lq32_Ld200"])
def test_perf_plaid_approx_score(shape_id, request):
    from late_interaction_kernels.plaid import plaid_approx_score

    _seed()
    n_centroids, Lq, B, max_Ld = 2048, 32, 512, 200
    qcs = torch.randn(n_centroids, Lq, device="cuda", dtype=torch.float32)
    codes = torch.randint(0, n_centroids, (B, max_Ld), device="cuda", dtype=torch.int64)
    doc_lens = torch.full((B,), max_Ld, device="cuda", dtype=torch.int64)

    p20, p50, p80 = time_kernel(lambda: plaid_approx_score(qcs, codes, doc_lens))
    check_regression("plaid_approx_score", shape_id, p20, p50, p80, request=request)


_PLAID_RESIDUAL_SHAPES = [
    ("residual_nbits2_4x64_32x200_d128", {"Nq": 4, "Nd": 64, "Lq": 32, "max_Ld": 200, "d": 128, "nbits": 2}),
]


@pytest.mark.parametrize("shape_id,dims", _PLAID_RESIDUAL_SHAPES)
def test_perf_maxsim_residual(shape_id, dims, request):
    from late_interaction_kernels.plaid import maxsim_residual

    centroids, bucket_weights, codes, residuals, doc_lengths = _make_quant_index(
        Nd=dims["Nd"], max_Ld=dims["max_Ld"], d=dims["d"], n_centroids=256, nbits=dims["nbits"]
    )
    Q = torch.randn(dims["Nq"], dims["Lq"], dims["d"], device="cuda", dtype=torch.bfloat16)

    p20, p50, p80 = time_kernel(
        lambda: maxsim_residual(
            Q,
            codes,
            residuals,
            doc_lengths,
            centroids,
            bucket_weights,
            nbits=dims["nbits"],
            normalize=True,
        )
    )
    check_regression("maxsim_residual", shape_id, p20, p50, p80, request=request)


@pytest.mark.parametrize("shape_id,dims", _PLAID_RESIDUAL_SHAPES)
def test_perf_maxsim_residual_varlen(shape_id, dims, request):
    from late_interaction_kernels.plaid import maxsim_residual_varlen

    centroids, bucket_weights, codes, residuals, _ = _make_quant_index(
        Nd=dims["Nd"], max_Ld=dims["max_Ld"], d=dims["d"], n_centroids=256, nbits=dims["nbits"]
    )
    # Flatten to varlen layout (every doc has the same length here).
    codes_flat = codes.reshape(-1).contiguous()
    residuals_flat = residuals.reshape(-1, residuals.shape[-1]).contiguous()
    cu_seqlens_d = torch.arange(
        0, (dims["Nd"] + 1) * dims["max_Ld"], dims["max_Ld"], device="cuda", dtype=torch.int32
    )
    Q = torch.randn(dims["Nq"], dims["Lq"], dims["d"], device="cuda", dtype=torch.bfloat16)

    p20, p50, p80 = time_kernel(
        lambda: maxsim_residual_varlen(
            Q,
            codes_flat,
            residuals_flat,
            cu_seqlens_d,
            centroids,
            bucket_weights,
            nbits=dims["nbits"],
            max_seqlen_d=dims["max_Ld"],
            normalize=True,
        )
    )
    check_regression("maxsim_residual_varlen", shape_id, p20, p50, p80, request=request)


# --------------------------------------------------------------------------- #
# FP8 inference — Hopper+ only                                                #
# --------------------------------------------------------------------------- #


def _hopper_or_newer() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 9


@pytest.mark.skipif(not _hopper_or_newer(), reason="FP8 path requires Hopper SM90+")
@pytest.mark.parametrize("shape_id", ["fp8_rerank_1x256_32x200_d128"])
def test_perf_maxsim_inference_fp8(shape_id, request):
    from late_interaction_kernels.fp8 import maxsim_inference_fp8, quantize_fp8_per_tensor

    _seed()
    Nq, Nd, Lq, Ld, d = 1, 256, 32, 200, 128
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float32)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float32)
    Q_fp8, sQ = quantize_fp8_per_tensor(Q)
    D_fp8, sD = quantize_fp8_per_tensor(D)

    p20, p50, p80 = time_kernel(lambda: maxsim_inference_fp8(Q_fp8, D_fp8, scale_Q=sQ, scale_D=sD))
    check_regression("maxsim_inference_fp8", shape_id, p20, p50, p80, request=request)
