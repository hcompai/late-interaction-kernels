"""Autotune-cache regression: variable ``Ld`` must not thrash the cache.

Pins the invariant that ``Ld`` is no longer in the autotune key on the
dense forward kernel: many distinct doc lengths share a single cache
entry. ``Lq`` does stay in the key (it drives ``tl.static_range``
unrolling), which the second test guards against an over-eager future
refactor.
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
