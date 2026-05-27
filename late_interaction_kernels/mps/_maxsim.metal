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
    /* argmax slot is unused (SaveArgmax=false); reuse `scores`. */              \
    maxsim_inference_impl<T, J, false>(                                          \
        Q, D, q_mask, d_mask, scores,                                            \
        reinterpret_cast<device int*>(scores), p,                                \
        Q_tile, D_tile, S_tile, d_active_tile, q_active_tile,                    \
        tid3.z, tg.x, tg.y);                                                     \
}}

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


// Backward — Triton analogue: `maxsim_backward_unified`. One
// threadgroup per (i, s); `d` lanes cooperate on the d-axis. `acc_Q`
// is row-private (no atomics on grad_Q); `grad_D` scatters via
// `atomic_uint` CAS since multiple (i, s) can land on the same
// (d_global, t).
//
// When `Normalize` is set, the kernel normalizes Q/D on the fly and
// folds the L2-normalize Jacobian into the writes — saves the
// otherwise-dominant host-side launch overhead (full-tensor norms +
// projections were ~3 ms on a 32x32x32x200x128 step versus ~0.3 ms
// for the kernel itself).

// Reduce `val` across all `num_threads` lanes of the threadgroup.
// `scratch` must be sized for one float per simdgroup; unused when
// `num_threads ≤ 32`.
inline float threadgroup_reduce_sum(
    float val,
    uint tid,
    uint num_threads,
    threadgroup float* scratch)
{{
    const uint SIMD_W = 32u;
    float simd_partial = simd_sum(val);
    if (num_threads <= SIMD_W) {{
        return simd_partial;
    }}
    const uint simd_id = tid / SIMD_W;
    if ((tid % SIMD_W) == 0u) {{
        scratch[simd_id] = simd_partial;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint num_simds = (num_threads + SIMD_W - 1u) / SIMD_W;
    float total = 0.0f;
    for (uint k = 0; k < num_simds; ++k) total += scratch[k];
    // Trailing barrier so successive reductions sharing the same
    // `scratch` don't race (next write would otherwise outrun this
    // read on simdgroups that finish quickly).
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}}

template <typename T, bool Normalize>
inline void maxsim_bwd_impl(
    device const T*           Q,
    device const T*           D,
    device const int*         argmax,
    device const float*       grad_scores,
    device const char*        q_mask,
    device       float*       grad_Q,
    device       atomic_uint* grad_D_atomic,
    constant MaxSimParams&    p,
    threadgroup float*        scratch,    // sized for one float per simdgroup
    uint tid,                             // 0 ≤ tid < d
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

    const float qv_raw = float(Q[q_off]);

    // q_hat and 1 / ||Q||. In the un-normalized path q_hat == qv_raw
    // and inv_q_norm == 1, so the same formula reduces to the original
    // contributions.
    float q_norm = 1.0f;
    float inv_q_norm = 1.0f;
    float qv = qv_raw;
    if (Normalize) {{
        const float qq = threadgroup_reduce_sum(qv_raw * qv_raw, tid, d, scratch);
        q_norm = sqrt(max(qq, 1e-12f));
        inv_q_norm = 1.0f / q_norm;
        qv = qv_raw * inv_q_norm;
    }}

    float acc_Q = 0.0f;

    for (uint j = 0; j < Nd; ++j) {{
        const float gs = grad_scores[i * Nd + j];
        const uint d_global = kd_layout ? (i * Nd + j) : j;
        const int t = argmax[(i * Nd + j) * Lq + s];
        if (t < 0 || (uint)t >= Ld) continue;
        const uint d_idx = (d_global * Ld + (uint)t) * d + tid;
        const float dv_raw = float(D[d_idx]);

        float dv = dv_raw;
        float inv_d_norm = 1.0f;
        float dot_qh_dh = 0.0f;
        if (Normalize) {{
            const float dd = threadgroup_reduce_sum(dv_raw * dv_raw, tid, d, scratch);
            const float d_norm = sqrt(max(dd, 1e-12f));
            inv_d_norm = 1.0f / d_norm;
            dv = dv_raw * inv_d_norm;
            dot_qh_dh = threadgroup_reduce_sum(qv * dv, tid, d, scratch);
        }}

        acc_Q += gs * dv;
        // grad_D contribution. In the normalize path we fold the
        // D-side Jacobian here: (gs * Q_hat - <gs * Q_hat, D_hat> * D_hat) / ||D||.
        const float contrib = Normalize
            ? gs * (qv - dot_qh_dh * dv) * inv_d_norm
            : gs * qv;
        uint old_bits = atomic_load_explicit(&grad_D_atomic[d_idx], memory_order_relaxed);
        uint new_bits;
        do {{
            new_bits = as_type<uint>(as_type<float>(old_bits) + contrib);
        }} while (!atomic_compare_exchange_weak_explicit(
            &grad_D_atomic[d_idx], &old_bits, new_bits,
            memory_order_relaxed, memory_order_relaxed));
    }}

    if (Normalize) {{
        // Q-side Jacobian: (acc_Q - <acc_Q, Q_hat> * Q_hat) / ||Q||.
        const float dot_aQ_qh = threadgroup_reduce_sum(acc_Q * qv, tid, d, scratch);
        grad_Q[q_off] = (acc_Q - dot_aQ_qh * qv) * inv_q_norm;
    }} else {{
        grad_Q[q_off] = acc_Q;
    }}
}}

#define MAXSIM_BWD_KERNEL(NAME, T, NORM)                                         \
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
    /* One slot per possible simdgroup at d=128 (max 4 simdgroups). */           \
    threadgroup float scratch[4];                                                \
    maxsim_bwd_impl<T, NORM>(Q, D, argmax, grad_scores, q_mask,                  \
                             grad_Q, grad_D, p, scratch,                         \
                             tid3.z, tg.x, tg.y);                                \
}}

MAXSIM_BWD_KERNEL(maxsim_bwd_half,         half,   false)
MAXSIM_BWD_KERNEL(maxsim_bwd_half_norm,    half,   true)
MAXSIM_BWD_KERNEL(maxsim_bwd_bfloat,       bfloat, false)
MAXSIM_BWD_KERNEL(maxsim_bwd_bfloat_norm,  bfloat, true)
