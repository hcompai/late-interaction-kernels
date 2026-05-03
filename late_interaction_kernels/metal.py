"""Fused MaxSim forward on MPS via Metal ``simdgroup_matrix`` MMA.

Direct Metal-side analogue of :mod:`late_interaction_kernels.forward`:
the same FlashAttention-style outer-product tiling, but the inner GEMM
runs on Apple's ``simdgroup_matrix<T, 8, 8>`` MMA instead of CUDA tensor
cores. The ``[Nq · Nd · Lq · Ld]`` similarity tensor never reaches HBM —
each ``(q_batch, d_batch)`` threadgroup streams ``D`` in 32-row tiles
through threadgroup memory while accumulating a per-row running max.

Layout per threadgroup (one ``(i, j)`` pair, 32 threads = 1 simdgroup):

* ``Q_tile [32, d]`` and ``D_tile [32, d]`` in threadgroup memory; one
  thread owns one query / doc row, fuses the L2-normalize during the
  cooperative load.
* ``S_tile [32, 32]`` is computed by a 4×4 grid of 8×8
  ``simdgroup_multiply_accumulate`` calls reducing along ``d`` in steps
  of 8. Accumulators are fp32; inputs are fp16 / bf16 (the two dtypes
  Apple's MMA accepts on M2+ / M3+ silicon).
* The ``S_tile`` is stored back to threadgroup memory; each thread reads
  its own row, applies ``d_mask``, computes a row max, and folds it
  into a per-thread running max.
* After the doc-tile loop, ``q_mask`` is applied and the per-thread row
  maxes are summed across the simdgroup with ``simd_sum`` to produce
  ``scores[i, j]``.

The kernel is forward-only (training-time backward stays on the
``torch.compile`` reference path; autograd flows through the dispatch
in :mod:`._mps`). Constraints:

* ``d`` must be a multiple of 8 and ≤ ``D_MAX = 192``;
* fp16 / bf16 inputs only — fp32 falls back to ``torch.compile`` since
  the threadgroup-memory budget can't fit two ``32 × 192`` fp32 tiles
  alongside ``S_tile``;
* Apple Silicon GPU (M2 family for fp16, M3+ for bf16).

Outside those bounds the dispatch in :mod:`._mps` transparently picks
the compiled-reference path.
"""

from __future__ import annotations

import struct
import threading
from typing import Any

import torch

# 32×32 output tile so each of the 32 threads in the simdgroup owns one
# query and one doc row. ``D_MAX`` caps the static threadgroup-memory
# allocation: 32 · 192 · 2 (half) · 2 buffers + 32 · 32 · 4 (S_tile)
# ≈ 28 KiB, comfortably inside Apple's 32 KiB / threadgroup ceiling.
_BLOCK_Q = 32
_BLOCK_D = 32
_BLOCK_K = 8  # Apple ``simdgroup_matrix`` is 8 × 8
_M_TILES = _BLOCK_Q // _BLOCK_K
_N_TILES = _BLOCK_D // _BLOCK_K
_D_MAX = 192
_THREADS_PER_GROUP = 32

_FLAG_HAS_Q_MASK = 1 << 0
_FLAG_HAS_D_MASK = 1 << 1
_FLAG_NORMALIZE = 1 << 2

_PARAMS_FORMAT = "6I"  # (Nq, Nd, Lq, Ld, d, flags), all u32


_KERNEL_SOURCE = f"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#define BLOCK_Q  {_BLOCK_Q}
#define BLOCK_D  {_BLOCK_D}
#define BLOCK_K  {_BLOCK_K}
#define M_TILES  {_M_TILES}
#define N_TILES  {_N_TILES}
#define D_MAX    {_D_MAX}
#define THREADS  {_THREADS_PER_GROUP}

struct MaxSimParams {{
    uint Nq;
    uint Nd;
    uint Lq;
    uint Ld;
    uint d;
    uint flags;        // bit 0: has_q_mask, bit 1: has_d_mask, bit 2: normalize
}};

// One threadgroup = one simdgroup = 32 threads. Each thread owns one
// query position s = q_start + tid and one doc position t = d_start + tid
// during the cooperative loads. The simdgroup_matrix MMA then operates
// across the full 32 threads to compute a 32×32 score tile per d_start.
template <typename T>
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
    uint j)
{{
    const bool has_q_mask = (p.flags & {_FLAG_HAS_Q_MASK}u) != 0u;
    const bool has_d_mask = (p.flags & {_FLAG_HAS_D_MASK}u) != 0u;
    const bool normalize  = (p.flags & {_FLAG_NORMALIZE}u) != 0u;

    const uint Lq = p.Lq;
    const uint Ld = p.Ld;
    const uint d  = p.d;

    float score_acc = 0.0f;

    for (uint q_start = 0; q_start < Lq; q_start += BLOCK_Q) {{
        // Cooperative Q load: one row per thread, fold L2-normalize in registers.
        const uint s = q_start + tid;
        const bool s_valid = s < Lq;

        bool q_active = s_valid;
        if (has_q_mask && s_valid) {{
            q_active = q_mask[i * Lq + s] != 0;
        }}
        q_active_tile[tid] = q_active ? (uchar)1 : (uchar)0;

        // Vectorise the load + L2-norm in 4-wide chunks; both buffers are
        // contiguous and ``d`` is a multiple of 8 so the cast is well-defined.
        {{
            const uint d4 = d >> 2;
            device const vec<T, 4>* Q4 = reinterpret_cast<device const vec<T, 4>*>(Q);
            threadgroup vec<T, 4>* QT4 = reinterpret_cast<threadgroup vec<T, 4>*>(Q_tile);
            const uint q_row_off = (i * Lq + s) * d4;
            const uint t_row_off = tid * (D_MAX >> 2);
            if (s_valid) {{
                float sum2 = 0.0f;
                for (uint k = 0; k < d4; ++k) {{
                    vec<T, 4> v = Q4[q_row_off + k];
                    QT4[t_row_off + k] = v;
                    float4 vf = float4(v);
                    sum2 += dot(vf, vf);
                }}
                if (normalize) {{
                    float inv = rsqrt(max(sum2, 1e-24f));
                    for (uint k = 0; k < d4; ++k) {{
                        QT4[t_row_off + k] = vec<T, 4>(float4(QT4[t_row_off + k]) * inv);
                    }}
                }}
            }} else {{
                for (uint k = 0; k < d4; ++k) {{
                    QT4[t_row_off + k] = vec<T, 4>(0);
                }}
            }}
        }}

        // Each thread tracks the running max for its own query row.
        float running_max = -INFINITY;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint d_start = 0; d_start < Ld; d_start += BLOCK_D) {{
            // Cooperative D load: one row per thread, fold L2-normalize.
            const uint t = d_start + tid;
            const bool t_valid = t < Ld;

            bool d_active = t_valid;
            if (has_d_mask && t_valid) {{
                d_active = d_mask[j * Ld + t] != 0;
            }}
            d_active_tile[tid] = d_active ? (uchar)1 : (uchar)0;

            {{
                const uint d4 = d >> 2;
                device const vec<T, 4>* D4 = reinterpret_cast<device const vec<T, 4>*>(D);
                threadgroup vec<T, 4>* DT4 = reinterpret_cast<threadgroup vec<T, 4>*>(D_tile);
                const uint d_row_off = (j * Ld + t) * d4;
                const uint t_row_off = tid * (D_MAX >> 2);
                if (t_valid) {{
                    float sum2 = 0.0f;
                    for (uint k = 0; k < d4; ++k) {{
                        vec<T, 4> v = D4[d_row_off + k];
                        DT4[t_row_off + k] = v;
                        float4 vf = float4(v);
                        sum2 += dot(vf, vf);
                    }}
                    if (normalize) {{
                        float inv = rsqrt(max(sum2, 1e-24f));
                        for (uint k = 0; k < d4; ++k) {{
                            DT4[t_row_off + k] = vec<T, 4>(float4(DT4[t_row_off + k]) * inv);
                        }}
                    }}
                }} else {{
                    for (uint k = 0; k < d4; ++k) {{
                        DT4[t_row_off + k] = vec<T, 4>(0);
                    }}
                }}
            }}

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // 32×32 score tile via 4×4 grid of 8×8 simdgroup_matrix MMAs,
            // reducing along d in BLOCK_K=8 steps.
            simdgroup_float8x8 acc[M_TILES][N_TILES];
            for (int m = 0; m < M_TILES; ++m) {{
                for (int n = 0; n < N_TILES; ++n) {{
                    acc[m][n] = simdgroup_float8x8(0);
                }}
            }}

            for (uint k_start = 0; k_start < d; k_start += BLOCK_K) {{
                simdgroup_matrix<T, 8, 8> A[M_TILES];
                simdgroup_matrix<T, 8, 8> B[N_TILES];

                // Q sub-tiles: rows [m*8, m*8+8), cols [k_start, k_start+8).
                for (int m = 0; m < M_TILES; ++m) {{
                    simdgroup_load(A[m], &Q_tile[m * BLOCK_K * D_MAX + k_start], D_MAX);
                }}
                // D sub-tiles, transposed in-place: we want B[k, t] = D_tile[t, k].
                for (int n = 0; n < N_TILES; ++n) {{
                    simdgroup_load(
                        B[n],
                        &D_tile[n * BLOCK_K * D_MAX + k_start],
                        D_MAX,
                        ulong2(0, 0),
                        /* transpose */ true);
                }}
                for (int m = 0; m < M_TILES; ++m) {{
                    for (int n = 0; n < N_TILES; ++n) {{
                        simdgroup_multiply_accumulate(acc[m][n], A[m], B[n], acc[m][n]);
                    }}
                }}
            }}

            // Land the accumulators in threadgroup memory so the per-row
            // max reduction below can read them as plain floats.
            for (int m = 0; m < M_TILES; ++m) {{
                for (int n = 0; n < N_TILES; ++n) {{
                    simdgroup_store(
                        acc[m][n],
                        &S_tile[(m * BLOCK_K) * BLOCK_D + n * BLOCK_K],
                        BLOCK_D);
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Each thread maxes its own row across the BLOCK_D doc tile,
            // honouring d_active for masked / past-end positions.
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
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        // Apply q_mask and clamp -inf rows (whole-doc-masked) to 0, then
        // sum across the 32 threads via the simdgroup intrinsic.
        float m = 0.0f;
        if (q_active_tile[tid] != 0 && isfinite(running_max)) {{
            m = running_max;
        }}
        score_acc += simd_sum(m);
    }}

    if (tid == 0) {{
        scores[i * p.Nd + j] = score_acc;
    }}
}}

#define MAXSIM_KERNEL(NAME, T)                                                   \\
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
    maxsim_inference_impl<T>(Q, D, q_mask, d_mask, scores, p,                    \\
                             Q_tile, D_tile, S_tile,                             \\
                             d_active_tile, q_active_tile,                       \\
                             tid3.z, tg.x, tg.y);                                \\
}}

MAXSIM_KERNEL(maxsim_inference_half,   half)
MAXSIM_KERNEL(maxsim_inference_bfloat, bfloat)
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
        # Compile failed in another thread; ``_lib_compile_error`` was set.
        assert _lib_compile_error is not None
        raise _lib_compile_error
    return _lib


_DTYPE_TO_KERNEL = {
    torch.float16: "maxsim_inference_half",
    torch.bfloat16: "maxsim_inference_bfloat",
}


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
    and when the dtype is not fp16 / bf16. The dispatch in
    :mod:`._mps` consults this before launching.
    """
    if Q.dtype != D.dtype or Q.dtype not in _DTYPE_TO_KERNEL:
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

    kernel_name = _DTYPE_TO_KERNEL.get(Q.dtype)
    if kernel_name is None:
        raise RuntimeError(f"Unsupported dtype {Q.dtype}; the Metal path handles fp16 and bf16.")

    Nq, Lq, d = Q.shape
    Nd, Ld, _ = D.shape
    if d > _D_MAX or d % _BLOCK_K != 0:
        raise RuntimeError(f"Metal path supports d ≤ {_D_MAX} and d %% {_BLOCK_K} == 0; got d={d}.")

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

    fn = getattr(_get_lib(), kernel_name)
    fn(
        Q,
        D,
        q_mask_i8,
        d_mask_i8,
        out,
        params,
        threads=(Nq, Nd, _THREADS_PER_GROUP),
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
