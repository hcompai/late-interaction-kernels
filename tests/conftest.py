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
