"""Autotune-cache regressions across kernels.

Pins the invariants for what stays in / out of each kernel's autotune
key:

* Dense forward: ``Ld`` is OUT of the key — distinct doc lengths share
  one cache entry. ``Lq`` is IN the key (drives ``tl.static_range``
  unrolling).
* Scatter pair-list kernel: ``max_lq`` and ``max_ld`` are OUT of the
  key — the kernel keys only on ``d_pad``, so variable-seqlen pair
  lists share a single autotune entry.
"""

import pytest
import torch

pytestmark = pytest.mark.cuda


def test_forward_kernel_compiles_once_for_varying_ld():
    """Many distinct ``Ld`` values share one autotune-cache entry."""
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.forward import _maxsim_fwd_kernel

    _maxsim_fwd_kernel.cache.clear()
    Q = torch.randn(2, 32, 128, device="cuda", dtype=torch.float16)
    for ld in (192, 256, 384, 512, 768, 1024, 1100, 1280):
        D = torch.randn(4, ld, 128, device="cuda", dtype=torch.float16)
        _ = maxsim(Q, D)

    cache = _maxsim_fwd_kernel.cache
    assert len(cache) == 1, f"forward autotune cache exploded across Ld: {len(cache)} entries (expected 1)"


def test_forward_kernel_keys_on_lq():
    """``Lq`` stays in the key; distinct ``Lq`` values get distinct entries."""
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.forward import _maxsim_fwd_kernel

    _maxsim_fwd_kernel.cache.clear()
    for lq in (32, 128):
        Q = torch.randn(2, lq, 128, device="cuda", dtype=torch.float16)
        D = torch.randn(4, 256, 128, device="cuda", dtype=torch.float16)
        _ = maxsim(Q, D)

    cache = _maxsim_fwd_kernel.cache
    assert len(cache) == 2, f"Lq must stay in the autotune key; got {len(cache)} entries for 2 distinct Lq"


def test_scatter_kernel_compiles_once_for_varying_max_ld():
    """Many distinct ``max_ld`` values share one autotune-cache entry."""
    from late_interaction_kernels.score_pairs import _scatter_fwd_kernel, score_pairs_packed

    _scatter_fwd_kernel.cache.clear()

    d_emb = 64
    lq = 8
    Q = torch.randn(lq, d_emb, device="cuda", dtype=torch.float16)
    cu_q = torch.tensor([0, lq], device="cuda", dtype=torch.int32)
    pair_q = torch.zeros(1, device="cuda", dtype=torch.int32)
    pair_d = torch.zeros(1, device="cuda", dtype=torch.int32)

    for max_ld in (64, 128, 256, 512, 1024):
        D = torch.randn(max_ld, d_emb, device="cuda", dtype=torch.float16)
        cu_d = torch.tensor([0, max_ld], device="cuda", dtype=torch.int32)
        _ = score_pairs_packed(Q, D, cu_q, cu_d, pair_q, pair_d, max_seqlen_q=lq, max_seqlen_d=max_ld)

    cache = _scatter_fwd_kernel.cache
    assert len(cache) == 1, (
        f"scatter autotune cache exploded across max_ld: {len(cache)} entries (expected 1)"
    )


def test_scatter_kernel_compiles_once_for_varying_max_lq():
    """Many distinct ``max_lq`` values share one autotune-cache entry."""
    from late_interaction_kernels.score_pairs import _scatter_fwd_kernel, score_pairs_packed

    _scatter_fwd_kernel.cache.clear()

    d_emb = 64
    ld = 64
    D = torch.randn(ld, d_emb, device="cuda", dtype=torch.float16)
    cu_d = torch.tensor([0, ld], device="cuda", dtype=torch.int32)
    pair_q = torch.zeros(1, device="cuda", dtype=torch.int32)
    pair_d = torch.zeros(1, device="cuda", dtype=torch.int32)

    for lq in (8, 16, 32, 64, 128):
        Q = torch.randn(lq, d_emb, device="cuda", dtype=torch.float16)
        cu_q = torch.tensor([0, lq], device="cuda", dtype=torch.int32)
        _ = score_pairs_packed(Q, D, cu_q, cu_d, pair_q, pair_d, max_seqlen_q=lq, max_seqlen_d=ld)

    cache = _scatter_fwd_kernel.cache
    assert len(cache) == 1, (
        f"scatter autotune cache exploded across max_lq: {len(cache)} entries (expected 1)"
    )
