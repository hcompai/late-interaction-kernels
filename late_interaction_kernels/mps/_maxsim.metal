// SPDX: same as the parent package.
//
// Fused MaxSim forward kernel (Apple `simdgroup_matrix` MMA) — Metal
// analogue of `forward.py`. FlashAttention-style tiling: Q hoisted into
// a register-resident simdgroup-matrix cache, [Lq, Ld] similarities
// never reach HBM. Loaded by `mps/metal.py` via
// `Path.read_text().format(...)` with the Python-side constants
// (tile sizes, flag bits) substituted in.
//
// Two persistence levels dispatched by `Nd`:
//   * J_PER_TG = 1 for tiny corpora (wide grid, keeps the GPU busy);
//   * J_PER_TG = 8 otherwise (amortises the Q-cache build).

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
            // eps = 1e-12 to match `forward.py` and `F.normalize`.
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

template <typename T, int J_PER_TG, bool SaveArgmax>
inline void maxsim_inference_impl(
    device const T*        Q,
    device const T*        D,
    device const char*     q_mask,
    device const char*     d_mask,
    device       float*    scores,
    device       int*      argmax,        // [Nq * Nd, Lq]; unused when !SaveArgmax
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
    // KD / pairs: each query owns its own slab in a flattened
    // D[Nq * Nd, Ld, d] view (`d_global = i * Nd + j`); cross-product
    // shares one D across all queries (`d_global = j`). Runtime flag —
    // not a constexpr — to avoid doubling kernel variants for one
    // ``select`` op (Apple ALU handles it for free).
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
            const uint j_global = kd_layout ? (i * Nd + j) : j;

            float running_max = -INFINITY;
            int running_argmax = 0;

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
                    int local_argmax = 0;
                    const uint row_base = tid * BLOCK_D;
                    for (uint n = 0; n < BLOCK_D; ++n) {{
                        if (d_active_tile[n] != 0) {{
                            float v = S_tile[row_base + n];
                            if (v > local_max) {{
                                local_max = v;
                                local_argmax = (int)(d_start + n);
                            }}
                        }}
                    }}
                    if (local_max > running_max) {{
                        running_max = local_max;
                        running_argmax = local_argmax;
                    }}
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

            // Per-(i, s, j) argmax write. Each lane owns its own s.
            // Lanes whose s ≥ Lq skipped via s_valid; masked rows still
            // get a write (value=0) so the backward sees a defined slot.
            if (SaveArgmax && s_valid) {{
                argmax[(i * Nd + j) * Lq + s] = running_argmax;
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

// Inference (no argmax save). Buffer 6 is a placeholder.
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
    /* argmax is unused (SaveArgmax=false) — reuse `scores` as placeholder. */\
    maxsim_inference_impl<T, J, false>(                                          \
        Q, D, q_mask, d_mask, scores,                                            \
        reinterpret_cast<device int*>(scores), p,                                \
        Q_tile, D_tile, S_tile, d_active_tile, q_active_tile,                    \
        tid3.z, tg.x, tg.y);                                                     \
}}

// Training (save argmax to buffer 6). One kernel per (dtype, J_PER_TG).
#define MAXSIM_TRAIN_KERNEL(NAME, T, J)                                          \
kernel void NAME(                                                                \
    device const T*        Q       [[buffer(0)]],                                \
    device const T*        D       [[buffer(1)]],                                \
    device const char*     q_mask  [[buffer(2)]],                                \
    device const char*     d_mask  [[buffer(3)]],                                \
    device       float*    scores  [[buffer(4)]],                                \
    constant MaxSimParams& p       [[buffer(5)]],                                \
    device       int*      argmax  [[buffer(6)]],                                \
    uint3 tg                       [[threadgroup_position_in_grid]],             \
    uint3 tid3                     [[thread_position_in_threadgroup]])           \
{{                                                                               \
    threadgroup T     Q_tile[BLOCK_Q * D_MAX];                                   \
    threadgroup T     D_tile[BLOCK_D * D_MAX];                                   \
    threadgroup float S_tile[BLOCK_Q * BLOCK_D];                                 \
    threadgroup uchar d_active_tile[BLOCK_D];                                    \
    threadgroup uchar q_active_tile[BLOCK_Q];                                    \
    maxsim_inference_impl<T, J, true>(                                           \
        Q, D, q_mask, d_mask, scores, argmax, p,                                 \
        Q_tile, D_tile, S_tile, d_active_tile, q_active_tile,                    \
        tid3.z, tg.x, tg.y);                                                     \
}}

MAXSIM_KERNEL(maxsim_half_small,   half,   1)
MAXSIM_KERNEL(maxsim_half_big,     half,   {J_PER_TG_BIG})
MAXSIM_KERNEL(maxsim_bfloat_small, bfloat, 1)
MAXSIM_KERNEL(maxsim_bfloat_big,   bfloat, {J_PER_TG_BIG})

MAXSIM_TRAIN_KERNEL(maxsim_half_small_train,   half,   1)
MAXSIM_TRAIN_KERNEL(maxsim_half_big_train,     half,   {J_PER_TG_BIG})
MAXSIM_TRAIN_KERNEL(maxsim_bfloat_small_train, bfloat, 1)
MAXSIM_TRAIN_KERNEL(maxsim_bfloat_big_train,   bfloat, {J_PER_TG_BIG})


// ===== Backward =====
//
// Triton analogue: `maxsim_backward_unified` in `backward/unified.py`.
// One threadgroup per (i, s); `THREADS` lanes cooperate on the d-axis.
// `acc_Q` is row-private (no atomics on grad_Q); `grad_D` is scattered
// via `atomic_fetch_add_explicit` because multiple (i, j) pairs can
// land on the same (d_global, t).
//
// `kd_layout` switches `d_global` from `j` (cross-product) to `i*Nd+j`
// (each query owns its own slab).
template <typename T>
inline void maxsim_bwd_impl(
    device const T*           Q,
    device const T*           D,
    device const int*         argmax,        // [Nq*Nd, Lq] int32
    device const float*       grad_scores,   // [Nq, Nd] fp32
    device const char*        q_mask,        // [Nq, Lq] int8 or unused
    device       float*       grad_Q,        // [Nq, Lq, d] fp32
    device       atomic_uint* grad_D_atomic, // [Nd_total, Ld, d] aliased fp32 storage
    constant MaxSimParams&    p,
    uint tid,                                // 0 ≤ tid < d
    uint i,
    uint s)
{{
    const uint Nd = p.Nd;
    const uint Lq = p.Lq;
    const uint Ld = p.Ld;
    const uint d  = p.d;
    const bool has_q_mask = (p.flags & {FLAG_HAS_Q_MASK}u) != 0u;
    const bool kd_layout  = (p.flags & {FLAG_KD_LAYOUT}u) != 0u;

    if (s >= Lq || tid >= d) return;

    bool q_active = true;
    if (has_q_mask) {{
        q_active = q_mask[i * Lq + s] != 0;
    }}

    const uint q_off = (i * Lq + s) * d + tid;

    if (!q_active) {{
        grad_Q[q_off] = 0.0f;
        return;
    }}

    const float qv = float(Q[q_off]);
    float acc_Q = 0.0f;

    for (uint j = 0; j < Nd; ++j) {{
        const float gs = grad_scores[i * Nd + j];
        const uint d_global = kd_layout ? (i * Nd + j) : j;
        const int t = argmax[(i * Nd + j) * Lq + s];
        if (t < 0 || (uint)t >= Ld) continue;
        const uint d_idx = (d_global * Ld + (uint)t) * d + tid;
        const float dv = float(D[d_idx]);
        acc_Q += gs * dv;
        // Apple Silicon supports fp32 atomic add on device memory via
        // the bitcast-to-uint dance below (Metal3 has no direct
        // atomic_float; ``atomic_fetch_add_explicit`` on a uint that
        // aliases a float is the documented workaround).
        const float contrib = gs * qv;
        uint old_bits = atomic_load_explicit(&grad_D_atomic[d_idx], memory_order_relaxed);
        uint new_bits;
        do {{
            const float old_val = as_type<float>(old_bits);
            new_bits = as_type<uint>(old_val + contrib);
        }} while (!atomic_compare_exchange_weak_explicit(
            &grad_D_atomic[d_idx], &old_bits, new_bits,
            memory_order_relaxed, memory_order_relaxed));
    }}

    grad_Q[q_off] = acc_Q;
}}

#define MAXSIM_BWD_KERNEL(NAME, T)                                               \
kernel void NAME(                                                                \
    device const T*           Q           [[buffer(0)]],                         \
    device const T*           D           [[buffer(1)]],                         \
    device const int*         argmax      [[buffer(2)]],                         \
    device const float*       grad_scores [[buffer(3)]],                         \
    device const char*        q_mask      [[buffer(4)]],                         \
    device       float*       grad_Q      [[buffer(5)]],                         \
    device       atomic_uint* grad_D      [[buffer(6)]],                         \
    constant MaxSimParams&    p           [[buffer(7)]],                         \
    uint3 tg                              [[threadgroup_position_in_grid]],      \
    uint3 tid3                            [[thread_position_in_threadgroup]])    \
{{                                                                               \
    maxsim_bwd_impl<T>(Q, D, argmax, grad_scores, q_mask,                        \
                       grad_Q, grad_D, p,                                        \
                       tid3.z, tg.x, tg.y);                                      \
}}

MAXSIM_BWD_KERNEL(maxsim_bwd_half,   half)
MAXSIM_BWD_KERNEL(maxsim_bwd_bfloat, bfloat)
