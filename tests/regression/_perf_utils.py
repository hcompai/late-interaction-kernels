"""Helpers for the GPU perf regression suite.

The suite times each non-experimental kernel on a small set of canonical shapes
and compares the median latency against a baseline JSON committed under
``benchmarks/baselines/{gpu_class}.json``. A test fails if the measured median
exceeds ``baseline * (1 + tolerance)`` (default 5 %).

When pytest is invoked with ``--update-perf-baseline``, ``check_regression``
records the measurement into a session-scoped dict instead of asserting; the
conftest fixture flushes that dict back to disk at session end.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BASELINES_DIR = _REPO_ROOT / "benchmarks" / "baselines"

# Default tolerance: 5 % slower than baseline fails CI.
DEFAULT_TOLERANCE: float = 0.05

# do_bench tuning. ms=50 / rep=200 keeps the suite well under 2 min on H100
# while still giving stable medians (CV typically < 2 %).
_BENCH_WARMUP_MS: int = 50
_BENCH_REP_MS: int = 200


def gpu_class() -> str:
    """Map ``torch.cuda.get_device_name()`` to a short, stable filename token.

    Examples:
        ``"NVIDIA H100 80GB HBM3"`` → ``"H100"``
        ``"NVIDIA A100-SXM4-40GB"`` → ``"A100"``
        ``"NVIDIA GeForce RTX 4090"`` → ``"RTX4090"``
    """
    if not torch.cuda.is_available():
        return "cpu"
    name = torch.cuda.get_device_name(0)
    # Known datacenter GPUs: pull the model token out directly.
    for token in ("H200", "H100", "A100", "L40S", "L40", "L4", "A10", "V100", "T4"):
        if token in name:
            return token
    # GeForce / Ada / generic — strip vendor + spaces.
    cleaned = re.sub(r"NVIDIA|GeForce|RTX|GTX", "", name)
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", cleaned)
    return cleaned or "unknown"


def baseline_path() -> Path:
    return _BASELINES_DIR / f"{gpu_class()}.json"


def load_baseline() -> dict[str, Any] | None:
    """Read the baseline file for the current GPU, or ``None`` if missing."""
    path = baseline_path()
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def time_kernel(fn) -> tuple[float, float, float]:
    """Return ``(p20, p50, p80)`` latency in ms via ``triton.testing.do_bench``.

    ``fn`` must be a no-arg callable that launches the kernel; warmup and CUDA
    graph capture are handled by ``do_bench``.
    """
    import triton.testing

    torch.cuda.synchronize()
    p20, p50, p80 = triton.testing.do_bench(
        fn,
        warmup=_BENCH_WARMUP_MS,
        rep=_BENCH_REP_MS,
        quantiles=[0.2, 0.5, 0.8],
    )
    return float(p20), float(p50), float(p80)


def check_regression(
    kernel_id: str,
    shape_id: str,
    p20_ms: float,
    p50_ms: float,
    p80_ms: float,
    *,
    request: pytest.FixtureRequest,
    tolerance: float = DEFAULT_TOLERANCE,
) -> None:
    """Compare ``p50_ms`` against the committed baseline.

    Behavior:
        * ``--update-perf-baseline`` active → record the measurement and return.
        * No baseline file at all → ``pytest.skip`` with an actionable message.
        * No entry for ``(kernel_id, shape_id)`` → ``pytest.skip``.
        * Otherwise assert ``p50_ms <= baseline_p50 * (1 + tolerance)``.
    """
    measurement = {"p50_ms": p50_ms, "p20_ms": p20_ms, "p80_ms": p80_ms}

    if request.config.getoption("--update-perf-baseline", default=False):
        store = request.config._perf_measurements  # set in conftest
        store.setdefault(kernel_id, {})[shape_id] = measurement
        # Print so the runner sees what was captured.
        print(f"[perf-baseline] {kernel_id}/{shape_id}: p50={p50_ms:.3f}ms")
        return

    baseline = load_baseline()
    if baseline is None:
        pytest.skip(
            f"No perf baseline for GPU class {gpu_class()!r} at {baseline_path()}. "
            "Generate one with: pytest tests/regression/ -m perf --update-perf-baseline"
        )

    entry = baseline.get("measurements", {}).get(kernel_id, {}).get(shape_id)
    if entry is None:
        pytest.skip(
            f"No baseline entry for {kernel_id}/{shape_id} in {baseline_path()}. "
            "Regenerate with --update-perf-baseline."
        )

    baseline_p50 = float(entry["p50_ms"])
    ratio = p50_ms / baseline_p50
    limit = 1.0 + tolerance
    assert ratio <= limit, (
        f"Perf regression in {kernel_id}/{shape_id}: "
        f"measured p50={p50_ms:.3f}ms vs baseline p50={baseline_p50:.3f}ms "
        f"(ratio={ratio:.3f}, limit={limit:.3f}, tolerance={tolerance:.0%}). "
        f"If this slow-down is intentional, regenerate with --update-perf-baseline."
    )


def write_baseline(measurements: dict[str, dict[str, dict[str, float]]]) -> Path:
    """Persist measurements as the new baseline for the current GPU class."""
    _BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    path = baseline_path()

    import triton

    payload = {
        "gpu": gpu_class(),
        "gpu_full_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tolerance": DEFAULT_TOLERANCE,
        "measurements": measurements,
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return path
