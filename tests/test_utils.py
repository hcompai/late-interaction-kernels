"""Unit tests for the small shared helpers in ``_utils``.

These are pure-Python and need neither a GPU nor any optional downstream
(pylate / colpali-engine), so they run on every CI host. ``package_at_least``
gates whether ``patch_pylate()`` / ``patch_colpali_engine()`` defer to the
native LIK backends, so its numeric-prefix parsing is pinned here.
"""

import importlib.metadata as metadata

import pytest

from late_interaction_kernels._utils import package_at_least


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
