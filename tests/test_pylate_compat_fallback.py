"""CPU-safe regression tests for the patched scorers' fallback path.

On CPU ``_device_path`` returns ``None``, so the patched wrappers defer to
PyLate's original implementation. A legacy ``mask=`` (single document mask)
must be forwarded as ``documents_mask`` on that path — it used to be
silently dropped. We inject a minimal fake ``pylate`` (just
``convert_to_tensor``) and stub ``_ORIGINAL`` directly; no GPU, no real
PyLate required.
"""

import sys
import types

import pytest
import torch


@pytest.fixture
def fake_pylate_tensor(monkeypatch):
    """Provide ``pylate.utils.tensor.convert_to_tensor`` for the wrappers."""
    tensor_mod = types.ModuleType("pylate.utils.tensor")
    tensor_mod.convert_to_tensor = lambda x: x
    utils_mod = types.ModuleType("pylate.utils")
    utils_mod.tensor = tensor_mod
    pylate_mod = types.ModuleType("pylate")
    pylate_mod.utils = utils_mod
    monkeypatch.setitem(sys.modules, "pylate", pylate_mod)
    monkeypatch.setitem(sys.modules, "pylate.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "pylate.utils.tensor", tensor_mod)


def _vanilla_colbert_scores(q, d, queries_mask=None, documents_mask=None):
    """Minimal PyLate-style scorer: doc mask multiplied in before the max."""
    sim = torch.einsum("ald,bsd->abls", q, d)
    if documents_mask is not None:
        sim = sim * documents_mask[None, :, None, :].to(sim.dtype)
    scores = sim.max(dim=-1).values
    if queries_mask is not None:
        scores = scores * queries_mask[:, None, :].to(scores.dtype)
    return scores.sum(dim=-1)


def test_fallback_forwards_legacy_mask(fake_pylate_tensor, monkeypatch):
    from late_interaction_kernels import pylate_compat

    received: dict[str, torch.Tensor | None] = {}

    def recording_original(q, d, queries_mask=None, documents_mask=None):
        received["documents_mask"] = documents_mask
        return _vanilla_colbert_scores(q, d, queries_mask=queries_mask, documents_mask=documents_mask)

    monkeypatch.setattr(pylate_compat, "_ORIGINAL", {"colbert_scores": recording_original})

    torch.manual_seed(0)
    Q = torch.nn.functional.normalize(torch.randn(2, 6, 16), dim=-1)
    D = torch.nn.functional.normalize(torch.randn(3, 8, 16), dim=-1)
    d_mask = torch.ones(3, 8)
    d_mask[:, -3:] = 0

    out = pylate_compat.patched_colbert_scores(Q, D, mask=d_mask)

    assert received["documents_mask"] is d_mask
    expected = _vanilla_colbert_scores(Q, D, documents_mask=d_mask)
    torch.testing.assert_close(out, expected)


def test_kd_fallback_forwards_legacy_mask(fake_pylate_tensor, monkeypatch):
    from late_interaction_kernels import pylate_compat

    received: dict[str, torch.Tensor | None] = {}

    def recording_original(q, d, queries_mask=None, documents_mask=None):
        received["documents_mask"] = documents_mask
        return torch.zeros(q.shape[0], d.shape[1])

    monkeypatch.setattr(pylate_compat, "_ORIGINAL", {"colbert_kd_scores": recording_original})

    Q = torch.randn(2, 6, 16)
    D = torch.randn(2, 3, 8, 16)
    d_mask = torch.ones(2, 3, 8)

    pylate_compat.patched_colbert_kd_scores(Q, D, mask=d_mask)
    assert received["documents_mask"] is d_mask
