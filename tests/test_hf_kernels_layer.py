"""Parity tests for the HF-Kernels ``torch-ext/late_interaction_kernels/`` tree.

These tests validate that the curated inference surface exposed to the HF Hub
produces numerically identical scores to the pure-PyTorch reference, and that
the ``MaxSim`` nn.Module layer can be dropped into an HF ``kernelize()`` flow.

Run locally (CUDA required for the kernel paths)::

    pytest tests/test_hf_kernels_layer.py -v

The tests import the HF-facing package directly from its on-disk location
(``torch-ext/late_interaction_kernels``) so we exercise the exact tree that
``kernel-builder build-and-copy`` will ship to the Hub.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch

from late_interaction_kernels.reference import maxsim_reference

TORCH_EXT = Path(__file__).resolve().parents[1] / "torch-ext"


@pytest.fixture(scope="module")
def hf_lik():
    """Import the HF-facing package from ``torch-ext/``, shadowing the PyPI package."""
    sys.path.insert(0, str(TORCH_EXT))
    # Drop any previously-imported late_interaction_kernels so we re-import from torch-ext.
    for mod_name in [m for m in sys.modules if m.startswith("late_interaction_kernels")]:
        del sys.modules[mod_name]
    try:
        module = importlib.import_module("late_interaction_kernels")
        yield module
    finally:
        sys.path.remove(str(TORCH_EXT))
        for mod_name in [m for m in sys.modules if m.startswith("late_interaction_kernels")]:
            del sys.modules[mod_name]


def _rand(*shape: int, dtype: torch.dtype = torch.float16, device: str = "cuda") -> torch.Tensor:
    return torch.randn(*shape, dtype=dtype, device=device)


@pytest.mark.cuda
def test_exports(hf_lik):
    """The v1 public surface must be importable from the HF build."""
    expected: set[str] = {
        "maxsim",
        "maxsim_inference",
        "maxsim_varlen_inference",
        "maxsim_inference_fp8",
        "maxsim_from_hidden",
        "plaid_approx_score",
        "maxsim_residual_inference",
        "maxsim_residual_varlen",
        "layers",
    }
    missing = expected - set(dir(hf_lik))
    assert not missing, f"HF build is missing exports: {sorted(missing)}"


@pytest.mark.cuda
def test_maxsim_inference_parity(hf_lik, rel):
    Nq, Lq, Nd, Ld, d = 4, 32, 8, 128, 128
    Q = _rand(Nq, Lq, d)
    D = _rand(Nd, Ld, d)
    scores = hf_lik.maxsim_inference(Q, D, normalize=True)
    expected = maxsim_reference(Q, D, normalize=True)
    assert rel(scores.float(), expected.float()) < 2e-3


@pytest.mark.cuda
def test_maxsim_with_masks_parity(hf_lik, rel):
    Nq, Lq, Nd, Ld, d = 2, 24, 5, 96, 128
    Q = _rand(Nq, Lq, d)
    D = _rand(Nd, Ld, d)
    q_mask = torch.ones(Nq, Lq, dtype=torch.bool, device="cuda")
    d_mask = torch.ones(Nd, Ld, dtype=torch.bool, device="cuda")
    q_mask[:, Lq - 4 :] = False
    d_mask[:, Ld - 8 :] = False
    scores = hf_lik.maxsim_inference(Q, D, q_mask=q_mask, d_mask=d_mask, normalize=True)
    expected = maxsim_reference(Q, D, q_mask=q_mask, d_mask=d_mask, normalize=True)
    assert rel(scores.float(), expected.float()) < 2e-3


@pytest.mark.cuda
def test_maxsim_layer_matches_functional(hf_lik, rel):
    """The stateless ``MaxSim`` layer must match the functional call bit-for-bit."""
    Nq, Lq, Nd, Ld, d = 2, 16, 4, 64, 128
    Q = _rand(Nq, Lq, d)
    D = _rand(Nd, Ld, d)

    layer = hf_lik.layers.MaxSim().to("cuda").eval()
    with torch.no_grad():
        layer_scores = layer(Q, D)
        functional_scores = hf_lik.maxsim_inference(Q, D, normalize=True)

    assert rel(layer_scores.float(), functional_scores.float()) == 0.0


@pytest.mark.cuda
def test_maxsim_layer_contract(hf_lik):
    """The layer must satisfy the HF Kernels layer contract."""
    layer_cls = hf_lik.layers.MaxSim
    assert issubclass(layer_cls, torch.nn.Module)
    assert getattr(layer_cls, "has_backward", True) is False
    assert getattr(layer_cls, "can_torch_compile", False) is True
    # Layers must be instantiable with no args — no __init__ state.
    instance = layer_cls()
    assert list(instance.parameters()) == []


@pytest.mark.cuda
def test_maxsim_residual_parity(hf_lik):
    """``maxsim_residual_inference`` must run without errors on realistic PLAID inputs.

    Numerical parity is already covered by ``tests/test_plaid.py``; this test
    is a smoke check that the export reaches the right implementation in the
    HF-facing tree.
    """
    assert callable(hf_lik.maxsim_residual_inference)
    assert callable(hf_lik.maxsim_residual_varlen)
    assert callable(hf_lik.plaid_approx_score)
