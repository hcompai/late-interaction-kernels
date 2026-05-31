"""Shared autotune configs for the backward kernels.

Every backward kernel is one program per output row that streams a single
``d_pad`` vector through a doc/bucket loop — there is no block tiling to
sweep, so the only useful knobs are ``num_warps`` and ``num_stages``. The
stock launch (``num_warps=4``) badly over-subscribes these narrow programs;
on H100 the optimum sits at 1–2 warps and tuning is worth 1.3–1.7x on the
unified path and up to 1.6x on the CSR ``grad_D`` reduction.

The key mirrors the forward autotuner: ``Lq`` (already power-of-two bucketed
upstream, and the dimension that separates the ColBERT ``Lq=32`` regime from
ColPali ``Lq=1024``), ``d_pad``, and the layout flags. ``Nd`` / ``Ld`` stay
out of the key so the cache holds one entry per regime instead of one per
batch size — the chosen ``num_warps`` is stable across ``Nd`` within a
regime.

Caveat: this shared key is the weakest fit for the CSR ``grad_D`` kernel,
whose per-program work is the bucket size ``≈ Nq·Lq/Ld`` (which ``Lq`` alone
doesn't capture), so a single config spanning a range of ``Ld`` may leave a
few percent on the table there. Correctness is unaffected — every config in
``BWD_CONFIGS`` computes the same gradient — so a stale-but-suboptimal config
is only ever a perf question, never a correctness one.

``reset_to_zero`` (used by the atomic-accumulating kernels at their decoration
sites) is needed so the autotuner's *benchmark trials* don't atomic-add onto
each other; in steady state every launcher allocates a fresh ``torch.zeros``
grad_D, so correctness on cache hits never depends on Triton's reset behaviour.
"""

try:
    import triton

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False


if _HAS_TRITON:
    # Curated (num_warps, num_stages) spread covering every per-shape optimum
    # measured on H100, plus the old (4, 2) default as a safety fallback.
    _BWD_LAUNCH = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (4, 2)]
    BWD_CONFIGS = [triton.Config({}, num_warps=w, num_stages=s) for (w, s) in _BWD_LAUNCH]

    # Layout-aware kernels (grad_Q, unified, atomic grad_D) — kd_layout flips
    # the doc-index math and shifts the optimum, so it stays in the key.
    BWD_KEY = ["Lq", "d_pad", "has_q_mask", "kd_layout"]
    # The CSR grad_D kernel is cross-product only (no kd_layout argument).
    BWD_KEY_CSR = ["Lq", "d_pad", "has_q_mask"]
else:  # pragma: no cover
    BWD_CONFIGS = []
    BWD_KEY = []
    BWD_KEY_CSR = []
