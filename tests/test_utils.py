"""Unit tests for the small shared helpers in ``_utils``.

These are pure-Python and need neither a GPU nor any optional downstream
(pylate / colpali-engine), so they run on every CI host. ``package_at_least``
gates whether ``patch_pylate()`` / ``patch_colpali_engine()`` defer to the
native LIK backends, so its numeric-prefix parsing is pinned here.
"""

import importlib.metadata as metadata

import pytest
import torch

from late_interaction_kernels._utils import assert_max_seqlen_covers, bucket_seqlen, package_at_least


@pytest.mark.parametrize(
    ("installed", "minimum", "expected"),
    [
        ("1.5.1", "1.5.1", True),
        ("1.5.0", "1.5.1", False),
        ("1.10.0", "1.5.1", True),  # numeric, not lexicographic: 10 > 5
        ("1.5", "1.5.1", False),  # shorter release tuple sorts below
        ("2.0.0", "1.5.1", True),
        ("1.5.1.dev0", "1.5.1", True),  # dev/rc suffixes floor to the release
        ("1.5.1rc1", "1.5.1", True),
        ("0.3.16", "0.3.17", False),
        ("0.3.17", "0.3.17", True),
    ],
)
def test_package_at_least_release_comparison(monkeypatch, installed: str, minimum: str, expected: bool):
    monkeypatch.setattr(metadata, "version", lambda name: installed)
    assert package_at_least("any-dist", minimum) is expected


def test_package_at_least_absent_package_is_false(monkeypatch):
    def _raise(name: str):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", _raise)
    assert package_at_least("definitely-not-installed", "1.0.0") is False


@pytest.mark.parametrize(
    ("max_len", "expected"),
    [(0, 0), (1, 16), (5, 16), (16, 16), (17, 32), (1030, 2048)],
)
def test_bucket_seqlen(max_len: int, expected: int):
    assert bucket_seqlen(max_len) == expected


def test_assert_max_seqlen_covers_accepts_exact_and_generous():
    cu = torch.tensor([0, 4, 12], dtype=torch.int32)
    assert_max_seqlen_covers(cu, 8, "max_seqlen_q")  # longest is 8
    assert_max_seqlen_covers(cu, 64, "max_seqlen_q")
    assert_max_seqlen_covers(torch.zeros(1, dtype=torch.int32), 0, "max_seqlen_q")  # no sequences


def test_assert_max_seqlen_covers_rejects_too_small():
    """On CPU ``torch._assert_async`` evaluates eagerly; on CUDA the same
    violation surfaces as a device-side assert (see test_varlen.py)."""
    cu = torch.tensor([0, 4, 12], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="max_seqlen_q=4 is smaller"):
        assert_max_seqlen_covers(cu, 4, "max_seqlen_q")
