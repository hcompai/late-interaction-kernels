"""Shared pytest fixtures and helpers."""

import pytest
import torch


def pytest_collection_modifyitems(config, items):
    """Auto-skip CUDA-marked tests when no GPU is visible."""
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _set_seed():
    """Every test starts from the same RNG state."""
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def rel_err(actual: torch.Tensor, reference: torch.Tensor) -> float:
    """``max |a - b| / max(1e-6, |b|_∞)`` — the standard Triton-kernel tolerance."""
    denom = max(1e-6, reference.abs().max().item())
    return (actual - reference).abs().max().item() / denom


@pytest.fixture
def rel():
    return rel_err


def needs_large_smem(d: int) -> bool:
    """``d >= 513`` overflows the 64 KB shared-mem ceiling on sm_75 (T4) with
    the kernel's current tiling. sm_80+ (A100/H100) have 100+ KB and run fine.
    Tests that parametrize over such shapes wrap the offending entry in
    ``pytest.mark.skipif(needs_large_smem(d), reason=...)``."""
    if not torch.cuda.is_available():
        return False
    return d >= 513 and torch.cuda.get_device_capability()[0] < 8
