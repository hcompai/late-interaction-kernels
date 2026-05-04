"""Fused MaxSim forward on MPS via Metal ``simdgroup_matrix`` MMA.

Direct Metal-side analogue of :mod:`late_interaction_kernels.forward`:
the same FlashAttention-style outer-product tiling, but the inner GEMM
runs on Apple's ``simdgroup_matrix<T, 8, 8>`` MMA instead of CUDA tensor
cores. The ``[Nq · Nd · Lq · Ld]`` similarity tensor never reaches HBM.

Each threadgroup serves ``J_PER_TG`` consecutive ``j`` values, loading
Q exactly once and hoisting it into a register-resident
``simdgroup_matrix`` cache that survives every ``(j, d-chunk)`` pair —
this is the single largest win versus the naive one-(i,j)-per-thread-
group layout. Two persistence levels are dispatched by ``Nd``:

* ``J_PER_TG = 1`` for tiny corpora where the launch grid needs to be
  wide to keep the GPU saturated;
* ``J_PER_TG = 8`` for the typical inference regime, amortising the Q
  load and the register-cache build 8× per threadgroup.

D streams in 32-row tiles through threadgroup memory with the optional
L2-normalize folded into the cooperative load.

Constraints (everything outside falls back to ``torch.compile``):

* ``d`` is a multiple of 8 and ≤ ``D_MAX = 128`` (covers ColBERT,
  ColPali, LateOn, LateOn-Code, edge-d48 / d64);
* fp16 / bf16 inputs only — Apple's MMA accepts those; fp32 inputs
  go through the compile path.
"""

from __future__ import annotations

import struct
import threading
from typing import Any

import torch

# Tile sizes. One simdgroup per threadgroup so each of the 32 lanes
# owns exactly one query / doc row during the cooperative load.
_BLOCK_Q = 32
_BLOCK_D = 32
_BLOCK_K = 8  # Apple ``simdgroup_matrix`` is 8 × 8
_M_TILES = _BLOCK_Q // _BLOCK_K
_N_TILES = _BLOCK_D // _BLOCK_K

# ``D_MAX`` caps the threadgroup-memory footprint AND the depth of the
# register-resident Q cache. 128 covers every standard late-interaction
# model dim (48 / 64 / 96 / 128) and keeps ``K_TILES_MAX = 16`` so the
# Q cache is ``4 × 16 = 64`` simdgroup matrices ≈ 8 KiB of register
# state per simdgroup — comfortably inside the M-series register file.
_D_MAX = 128
_K_TILES_MAX = _D_MAX // _BLOCK_K
_THREADS_PER_GROUP = 32

# Persistence levels dispatched by ``Nd``.
_J_PER_TG_BIG = 8
_J_THRESHOLD = 256  # Nd above which the persistent variant wins

_FLAG_HAS_Q_MASK = 1 << 0
_FLAG_HAS_D_MASK = 1 << 1
_FLAG_NORMALIZE = 1 << 2

_PARAMS_FORMAT = "6I"  # (Nq, Nd, Lq, Ld, d, flags), all u32


_KERNEL_SOURCE = f"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#define BLOCK_Q      {_BLOCK_Q}
#define BLOCK_D      {_BLOCK_D}
#define BLOCK_K      {_BLOCK_K}
#define M_TILES      {_M_TILES}
#define N_TILES      {_N_TILES}
#define D_MAX        {_D_MAX}
#define K_TILES_MAX  {_K_TILES_MAX}
#define THREADS      {_THREADS_PER_GROUP}

struct MaxSimParams {{
    uint Nq;
    uint Nd;
    uint Lq;
    uint Ld;
    uint d;
    uint flags;        // bit 0: has_q_mask, bit 1: has_d_mask, bit 2: normalize
}};

template <typename T>
inline void cooperative_load_row(
    device const vec<T, 4>* src,
    threadgroup vec<T, 4>* dst,
    uint src_row_off,
    uint dst_row_off,
    uint d4,
    bool valid,
    bool normalize)
{{
    // Stage the row through per-thread registers so the optional
    // L2-normalize doesn't bounce values through LDS twice. Without
    // this, ``normalize=True`` paid 3 LDS ops / element (write, read,
    // write); the register-staged version pays 1.
    constexpr uint D4_MAX = D_MAX >> 2;
    vec<T, 4> regs[D4_MAX];

    if (valid) {{
        float sum2 = 0.0f;
        for (uint k = 0; k < d4; ++k) {{
            regs[k] = src[src_row_off + k];
            float4 vf = float4(regs[k]);
            sum2 += dot(vf, vf);
        }}
        if (normalize) {{
            const float inv = rsqrt(max(sum2, 1e-24f));
            for (uint k = 0; k < d4; ++k) {{
                regs[k] = vec<T, 4>(float4(regs[k]) * inv);
            }}
        }}
        for (uint k = 0; k < d4; ++k) {{
            dst[dst_row_off + k] = regs[k];
        }}
    }} else {{
        for (uint k = 0; k < d4; ++k) {{
            dst[dst_row_off + k] = vec<T, 4>(0);
        }}
    }}
}}

template <typename T, int J_PER_TG>
inline void maxsim_inference_impl(
    device const T*        Q,
    device const T*        D,
    device const char*     q_mask,
    device const char*     d_mask,
    device       float*    scores,
    constant MaxSimParams& p,
    threadgroup T*         Q_tile,        // [BLOCK_Q, D_MAX]
    threadgroup T*         D_tile,        // [BLOCK_D, D_MAX]
    threadgroup float*     S_tile,        // [BLOCK_Q, BLOCK_D]
    threadgroup uchar*     d_active_tile, // [BLOCK_D]
    threadgroup uchar*     q_active_tile, // [BLOCK_Q]
    uint tid,
    uint i,
    uint j_block)
{{
    const bool has_q_mask = (p.flags & {_FLAG_HAS_Q_MASK}u) != 0u;
    const bool has_d_mask = (p.flags & {_FLAG_HAS_D_MASK}u) != 0u;
    const bool normalize  = (p.flags & {_FLAG_NORMALIZE}u) != 0u;

    const uint Nd = p.Nd;
    const uint Lq = p.Lq;
    const uint Ld = p.Ld;
    const uint d  = p.d;
    const uint d4 = d >> 2;
    const uint K_TILES = d / BLOCK_K;
    const uint j_start = j_block * J_PER_TG;

    device   const vec<T, 4>* Q4  = reinterpret_cast<device const vec<T, 4>*>(Q);
    device   const vec<T, 4>* D4  = reinterpret_cast<device const vec<T, 4>*>(D);
    threadgroup vec<T, 4>*    QT4 = reinterpret_cast<threadgroup vec<T, 4>*>(Q_tile);
    threadgroup vec<T, 4>*    DT4 = reinterpret_cast<threadgroup vec<T, 4>*>(D_tile);

    // Per-(j_idx) score accumulators. Only thread 0's copies matter
    // because we land each ``j`` total via ``simd_sum``.
    float score_acc[J_PER_TG];
    for (int j_idx = 0; j_idx < J_PER_TG; ++j_idx) {{
        score_acc[j_idx] = 0.0f;
    }}

    for (uint q_start = 0; q_start < Lq; q_start += BLOCK_Q) {{
        const uint s = q_start + tid;
        const bool s_valid = s < Lq;

        bool q_active = s_valid;
        if (has_q_mask && s_valid) {{
            q_active = q_mask[i * Lq + s] != 0;
        }}
        q_active_tile[tid] = q_active ? (uchar)1 : (uchar)0;

        cooperative_load_row<T>(
            Q4, QT4,
            (i * Lq + s) * d4,
            tid * (D_MAX >> 2),
            d4, s_valid, normalize);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Hoist Q into a register-resident simdgroup_matrix cache.
        // Reused for every (j, d-chunk) below.
        simdgroup_matrix<T, 8, 8> Q_cache[M_TILES][K_TILES_MAX];
        for (uint k = 0; k < K_TILES; ++k) {{
            for (int m = 0; m < M_TILES; ++m) {{
                simdgroup_load(
                    Q_cache[m][k],
                    &Q_tile[m * BLOCK_K * D_MAX + k * BLOCK_K],
                    D_MAX);
            }}
        }}

        for (uint j_idx = 0; j_idx < J_PER_TG; ++j_idx) {{
            const uint j = j_start + j_idx;
            if (j >= Nd) break;

            float running_max = -INFINITY;

            for (uint d_start = 0; d_start < Ld; d_start += BLOCK_D) {{
                const uint t = d_start + tid;
                const bool t_valid = t < Ld;

                bool d_active = t_valid;
                if (has_d_mask && t_valid) {{
                    d_active = d_mask[j * Ld + t] != 0;
                }}
                d_active_tile[tid] = d_active ? (uchar)1 : (uchar)0;

                cooperative_load_row<T>(
                    D4, DT4,
                    (j * Ld + t) * d4,
                    tid * (D_MAX >> 2),
                    d4, t_valid, normalize);

                threadgroup_barrier(mem_flags::mem_threadgroup);

                simdgroup_float8x8 acc[M_TILES][N_TILES];
                for (int m = 0; m < M_TILES; ++m) {{
                    for (int n = 0; n < N_TILES; ++n) {{
                        acc[m][n] = simdgroup_float8x8(0);
                    }}
                }}

                // Inner GEMM: only B (D sub-tile) is loaded fresh per
                // K-step; A comes from the register-resident Q_cache.
                for (uint k = 0; k < K_TILES; ++k) {{
                    simdgroup_matrix<T, 8, 8> B[N_TILES];
                    for (int n = 0; n < N_TILES; ++n) {{
                        simdgroup_load(
                            B[n],
                            &D_tile[n * BLOCK_K * D_MAX + k * BLOCK_K],
                            D_MAX,
                            ulong2(0, 0),
                            /* transpose */ true);
                    }}
                    for (int m = 0; m < M_TILES; ++m) {{
                        for (int n = 0; n < N_TILES; ++n) {{
                            simdgroup_multiply_accumulate(
                                acc[m][n], Q_cache[m][k], B[n], acc[m][n]);
                        }}
                    }}
                }}

                for (int m = 0; m < M_TILES; ++m) {{
                    for (int n = 0; n < N_TILES; ++n) {{
                        simdgroup_store(
                            acc[m][n],
                            &S_tile[(m * BLOCK_K) * BLOCK_D + n * BLOCK_K],
                            BLOCK_D);
                    }}
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (s_valid) {{
                    float local_max = -INFINITY;
                    const uint row_base = tid * BLOCK_D;
                    for (uint n = 0; n < BLOCK_D; ++n) {{
                        if (d_active_tile[n] != 0) {{
                            local_max = fmax(local_max, S_tile[row_base + n]);
                        }}
                    }}
                    running_max = fmax(running_max, local_max);
                }}
                // No trailing barrier: the next iteration's first writes
                // (cooperative D load + d_active fill) already pair with
                // the post-load barrier, which orders them after this
                // iteration's reads on the lockstepped simdgroup.
            }}

            // q-mask + clamp -inf rows to 0, sum across the simdgroup.
            float m = 0.0f;
            if (q_active_tile[tid] != 0 && isfinite(running_max)) {{
                m = running_max;
            }}
            float partial = simd_sum(m);
            if (tid == 0) {{
                score_acc[j_idx] += partial;
            }}
        }}
    }}

    if (tid == 0) {{
        for (uint j_idx = 0; j_idx < J_PER_TG; ++j_idx) {{
            const uint j = j_start + j_idx;
            if (j < Nd) {{
                scores[i * Nd + j] = score_acc[j_idx];
            }}
        }}
    }}
}}

#define MAXSIM_KERNEL(NAME, T, J)                                                \\
kernel void NAME(                                                                \\
    device const T*        Q       [[buffer(0)]],                                \\
    device const T*        D       [[buffer(1)]],                                \\
    device const char*     q_mask  [[buffer(2)]],                                \\
    device const char*     d_mask  [[buffer(3)]],                                \\
    device       float*    scores  [[buffer(4)]],                                \\
    constant MaxSimParams& p       [[buffer(5)]],                                \\
    uint3 tg                       [[threadgroup_position_in_grid]],             \\
    uint3 tid3                     [[thread_position_in_threadgroup]])           \\
{{                                                                               \\
    threadgroup T     Q_tile[BLOCK_Q * D_MAX];                                   \\
    threadgroup T     D_tile[BLOCK_D * D_MAX];                                   \\
    threadgroup float S_tile[BLOCK_Q * BLOCK_D];                                 \\
    threadgroup uchar d_active_tile[BLOCK_D];                                    \\
    threadgroup uchar q_active_tile[BLOCK_Q];                                    \\
    maxsim_inference_impl<T, J>(Q, D, q_mask, d_mask, scores, p,                 \\
                                Q_tile, D_tile, S_tile,                          \\
                                d_active_tile, q_active_tile,                    \\
                                tid3.z, tg.x, tg.y);                             \\
}}

MAXSIM_KERNEL(maxsim_half_small,   half,   1)
MAXSIM_KERNEL(maxsim_half_big,     half,   {_J_PER_TG_BIG})
MAXSIM_KERNEL(maxsim_bfloat_small, bfloat, 1)
MAXSIM_KERNEL(maxsim_bfloat_big,   bfloat, {_J_PER_TG_BIG})
"""


_lib_lock = threading.Lock()
_lib: Any | None = None
_lib_compile_error: BaseException | None = None


def _get_lib() -> Any:
    """JIT-compile the Metal source on first call; cache thereafter."""
    global _lib, _lib_compile_error
    if _lib is not None:
        return _lib
    if _lib_compile_error is not None:
        raise _lib_compile_error
    with _lib_lock:
        if _lib is None and _lib_compile_error is None:
            try:
                _lib = torch.mps.compile_shader(_KERNEL_SOURCE)
            except BaseException as exc:  # noqa: BLE001
                _lib_compile_error = exc
                raise
    if _lib is None:
        assert _lib_compile_error is not None
        raise _lib_compile_error
    return _lib


_DTYPE_TO_KERNEL = {
    (torch.float16, 1): "maxsim_half_small",
    (torch.float16, _J_PER_TG_BIG): "maxsim_half_big",
    (torch.bfloat16, 1): "maxsim_bfloat_small",
    (torch.bfloat16, _J_PER_TG_BIG): "maxsim_bfloat_big",
}


def _pick_j_per_tg(Nd: int) -> int:
    """Persist 8 j-values when Nd is large enough that the smaller
    launch grid still saturates the GPU."""
    return _J_PER_TG_BIG if Nd >= _J_THRESHOLD else 1


def is_available() -> bool:
    """Whether the Metal MaxSim path can run on this machine.

    Requires an MPS-capable PyTorch build with ``torch.mps.compile_shader``
    (PyTorch ≥ 2.10).
    """
    return (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and hasattr(torch.mps, "compile_shader")
    )


def supports(Q: torch.Tensor, D: torch.Tensor) -> bool:
    """Whether the Metal kernel handles this dtype + shape combination.

    Falls back when ``d`` is not a multiple of 8 or exceeds ``D_MAX``,
    and when the dtype is not fp16 / bf16.
    """
    if Q.dtype != D.dtype:
        return False
    if Q.dtype not in (torch.float16, torch.bfloat16):
        return False
    if Q.shape[-1] != D.shape[-1]:
        return False
    d = Q.shape[-1]
    return 0 < d <= _D_MAX and d % _BLOCK_K == 0


def _pack_params(Nq: int, Nd: int, Lq: int, Ld: int, d: int, flags: int) -> torch.Tensor:
    """Pack the kernel's ``MaxSimParams`` struct into a 6-int32 MPS tensor."""
    raw = struct.pack(_PARAMS_FORMAT, Nq, Nd, Lq, Ld, d, flags)
    return torch.frombuffer(bytearray(raw), dtype=torch.int32).clone().to("mps")


def _empty_mask() -> torch.Tensor:
    """1-element placeholder for the ``has_*_mask=False`` cases."""
    return torch.zeros(1, dtype=torch.int8, device="mps")


def maxsim_inference_metal(
    Q: torch.Tensor,
    D: torch.Tensor,
    q_mask: torch.Tensor | None = None,
    d_mask: torch.Tensor | None = None,
    *,
    normalize: bool = False,
) -> torch.Tensor:
    """Forward-only fused MaxSim on MPS via the ``simdgroup_matrix`` kernel.

    Args:
        Q: ``[Nq, Lq, d]`` or ``[Lq, d]`` query embeddings on ``mps``,
            fp16 or bf16.
        D: ``[Nd, Ld, d]`` or ``[Ld, d]`` document embeddings on ``mps``,
            same dtype as ``Q``.
        q_mask: optional ``[Nq, Lq]`` bool / int8 mask (``True`` = valid).
        d_mask: optional ``[Nd, Ld]`` bool / int8 mask.
        normalize: L2-normalize Q and D per-token inside the kernel.

    Returns:
        ``[Nq, Nd]`` fp32 scores (squeezed to match 2-D inputs).

    Raises:
        RuntimeError: if MPS or :func:`torch.mps.compile_shader` is missing,
            or the shape / dtype falls outside :func:`supports`.
        ValueError: on shape / device contract violations.
    """
    if not is_available():
        raise RuntimeError(
            "MPS Metal path unavailable: requires PyTorch ≥ 2.10 with MPS "
            "support and `torch.mps.compile_shader`."
        )

    q_was_2d = Q.dim() == 2
    d_was_2d = D.dim() == 2
    if q_was_2d:
        Q = Q.unsqueeze(0)
    if d_was_2d:
        D = D.unsqueeze(0)
    if q_mask is not None and q_mask.dim() == 1:
        q_mask = q_mask.unsqueeze(0)
    if d_mask is not None and d_mask.dim() == 1:
        d_mask = d_mask.unsqueeze(0)

    if Q.shape[-1] != D.shape[-1]:
        raise ValueError(
            f"Q and D must share the embedding dim; got Q.shape[-1]={Q.shape[-1]} "
            f"vs D.shape[-1]={D.shape[-1]}."
        )
    if Q.device.type != "mps" or D.device.type != "mps":
        raise ValueError(
            f"maxsim_inference_metal requires MPS tensors; got Q.device={Q.device}, D.device={D.device}."
        )
    if Q.dtype != D.dtype:
        raise ValueError(f"Q and D must share dtype; got {Q.dtype} vs {D.dtype}.")
    if Q.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(f"Unsupported dtype {Q.dtype}; the Metal path handles fp16 and bf16.")

    Nq, Lq, d = Q.shape
    Nd, Ld, _ = D.shape
    if d > _D_MAX or d % _BLOCK_K != 0:
        raise RuntimeError(f"Metal path supports d ≤ {_D_MAX} and d %% {_BLOCK_K} == 0; got d={d}.")

    j_per_tg = _pick_j_per_tg(Nd)
    kernel_name = _DTYPE_TO_KERNEL[(Q.dtype, j_per_tg)]

    Q = Q.contiguous()
    D = D.contiguous()

    flags = 0
    if q_mask is not None:
        flags |= _FLAG_HAS_Q_MASK
        q_mask_i8 = q_mask.contiguous().to(torch.int8)
    else:
        q_mask_i8 = _empty_mask()
    if d_mask is not None:
        flags |= _FLAG_HAS_D_MASK
        d_mask_i8 = d_mask.contiguous().to(torch.int8)
    else:
        d_mask_i8 = _empty_mask()
    if normalize:
        flags |= _FLAG_NORMALIZE

    out = torch.empty(Nq, Nd, device="mps", dtype=torch.float32)
    params = _pack_params(Nq, Nd, Lq, Ld, d, flags)

    j_blocks = (Nd + j_per_tg - 1) // j_per_tg

    fn = getattr(_get_lib(), kernel_name)
    fn(
        Q,
        D,
        q_mask_i8,
        d_mask_i8,
        out,
        params,
        threads=(Nq, j_blocks, _THREADS_PER_GROUP),
        group_size=(1, 1, _THREADS_PER_GROUP),
    )

    if q_was_2d and d_was_2d:
        return out.reshape(())
    if q_was_2d:
        return out.squeeze(0)
    if d_was_2d:
        return out.squeeze(-1)
    return out


__all__ = ["is_available", "supports", "maxsim_inference_metal"]
