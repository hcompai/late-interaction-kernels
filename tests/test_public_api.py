"""Freeze the public MaxSim signatures that downstreams call by keyword.

PyLate's native ``_lik_backend`` (lightonai/pylate#222) and colpali-engine's
(illuin-tech/colpali#412) call ``maxsim`` / ``maxsim_pairs`` / ``maxsim_mps``
with keyword args (``q_mask``, ``d_mask``, ``normalize``). Those kwarg names are
now an external contract, so a rename has to fail here loudly.

CPU-safe and downstream-free, so it runs on every CI host — unlike the e2e
patch tests in ``test_pylate_compat.py``, which need CUDA and a live PyLate.
``autograd`` transitively imports Triton (bundled with torch on Linux, absent
on macOS), so skip when Triton is missing.
"""

import inspect

import pytest

pytest.importorskip("triton")

from late_interaction_kernels.autograd import maxsim, maxsim_pairs  # noqa: E402
from late_interaction_kernels.mps import maxsim_mps  # noqa: E402


def test_public_api_signatures_match_native_backends():
    for fn in (maxsim, maxsim_pairs):
        params = inspect.signature(fn).parameters
        assert {"Q", "D", "q_mask", "d_mask", "normalize"} <= set(params)
    mps_params = inspect.signature(maxsim_mps).parameters
    assert {"Q", "D", "q_mask", "d_mask", "normalize"} <= set(mps_params)
