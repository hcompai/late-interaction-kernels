// SPDX: same as the parent package.
//
// Fused MaxSim forward kernel (Apple `simdgroup_matrix` MMA).
//
// Direct Metal-side analogue of `late_interaction_kernels/forward.py`:
// FlashAttention-style outer-product tiling, Q hoisted into a
// register-resident simdgroup-matrix cache that survives every
// (j, d-chunk) pair, [Lq, Ld] similarities never reach HBM.
//
// This file is loaded by `late_interaction_kernels.mps.metal` via
// `Path.read_text().format(...)` so Python-side constants and bit-flag
// values stay the single source of truth. The host packs
// `MaxSimParams` with `struct.pack(_PARAMS_FORMAT, ...)` using those
// same constants.
//
// Two persistence levels are dispatched by `Nd`:
//   * J_PER_TG = 1 for tiny corpora (wide launch grid keeps the GPU saturated);
//   * J_PER_TG = 8 for the typical inference regime (Q-cache build amortised 8x).
//
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#define BLOCK_Q      {BLOCK_Q}
#define BLOCK_D      {BLOCK_D}
#define BLOCK_K      {BLOCK_K}
#define M_TILES      {M_TILES}
#define N_TILES      {N_TILES}
#define D_MAX        {D_MAX}
#define K_TILES_MAX  {K_TILES_MAX}
#define THREADS      {THREADS_PER_GROUP}

struct MaxSimParams {{
    uint Nq;
    uint Nd;             // cross-product: doc count;  KD/pairs: K per query
    uint Lq;
    uint Ld;
    uint d;
    uint flags;          // bit 0: has_q_mask, bit 1: has_d_mask,
                         // bit 2: normalize,  bit 3: kd_layout
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
    // this, `normalize=True` paid 3 LDS ops / element (write, read,
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
            // eps = 1e-12 matches the Triton forward.py path
            // (`q_norm_sq = max(_, 1e-12)`) and the dense reference
            // (`F.normalize(..., eps=1e-12)`) so a vector with norm
            // ``< 1e-6`` produces the same scaled output on CUDA and MPS.
            // The old 1e-24 floor was effectively never reached on real
            // data — the change only affects truly degenerate inputs.
            const float inv = rsqrt(max(sum2, 1e-12f));
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
    const bool has_q_mask = (p.flags & {FLAG_HAS_Q_MASK}u) != 0u;
    const bool has_d_mask = (p.flags & {FLAG_HAS_D_MASK}u) != 0u;
    const bool normalize  = (p.flags & {FLAG_NORMALIZE}u) != 0u;
    // KD / pairs layout: every query owns its own [Nd, Ld, d] slab in a
    // flattened D[Nq * Nd, Ld, d] view, mirroring the Triton path in
    // ``forward.py`` (`d_global = pid`). Cross-product (default) has all
    // queries share the same D[j, t]. Unlike Triton — which sets
    // ``kd_layout`` as a ``tl.constexpr`` and emits two specialised
    // binaries — Metal keeps the two modes in a single kernel and pays
    // one ``select`` instruction per ``j_idx`` for the index pick.
    // That's effectively free on Apple's ALU and saves us 4 extra
    // kernel variants (half×big/small × bfloat×big/small × kd).
    const bool kd_layout  = (p.flags & {FLAG_KD_LAYOUT}u) != 0u;

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
    // because we land each `j` total via `simd_sum`.
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
            // KD: D[i, j, t]  ->  flat index (i*Nd + j)*Ld + t.
            // X-prod: D[j, t] ->  flat index j*Ld + t (every query reads
            // the same D, so the per-query offset is zero).
            const uint j_global = kd_layout ? (i * Nd + j) : j;

            float running_max = -INFINITY;

            for (uint d_start = 0; d_start < Ld; d_start += BLOCK_D) {{
                const uint t = d_start + tid;
                const bool t_valid = t < Ld;

                bool d_active = t_valid;
                if (has_d_mask && t_valid) {{
                    d_active = d_mask[j_global * Ld + t] != 0;
                }}
                d_active_tile[tid] = d_active ? (uchar)1 : (uchar)0;

                cooperative_load_row<T>(
                    D4, DT4,
                    (j_global * Ld + t) * d4,
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

#define MAXSIM_KERNEL(NAME, T, J)                                                \
kernel void NAME(                                                                \
    device const T*        Q       [[buffer(0)]],                                \
    device const T*        D       [[buffer(1)]],                                \
    device const char*     q_mask  [[buffer(2)]],                                \
    device const char*     d_mask  [[buffer(3)]],                                \
    device       float*    scores  [[buffer(4)]],                                \
    constant MaxSimParams& p       [[buffer(5)]],                                \
    uint3 tg                       [[threadgroup_position_in_grid]],             \
    uint3 tid3                     [[thread_position_in_threadgroup]])           \
{{                                                                               \
    threadgroup T     Q_tile[BLOCK_Q * D_MAX];                                   \
    threadgroup T     D_tile[BLOCK_D * D_MAX];                                   \
    threadgroup float S_tile[BLOCK_Q * BLOCK_D];                                 \
    threadgroup uchar d_active_tile[BLOCK_D];                                    \
    threadgroup uchar q_active_tile[BLOCK_Q];                                    \
    maxsim_inference_impl<T, J>(Q, D, q_mask, d_mask, scores, p,                 \
                                Q_tile, D_tile, S_tile,                          \
                                d_active_tile, q_active_tile,                    \
                                tid3.z, tg.x, tg.y);                             \
}}

MAXSIM_KERNEL(maxsim_half_small,   half,   1)
MAXSIM_KERNEL(maxsim_half_big,     half,   {J_PER_TG_BIG})
MAXSIM_KERNEL(maxsim_bfloat_small, bfloat, 1)
MAXSIM_KERNEL(maxsim_bfloat_big,   bfloat, {J_PER_TG_BIG})
