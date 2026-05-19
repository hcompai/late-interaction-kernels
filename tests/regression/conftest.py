"""Pytest plumbing for the perf regression suite."""

from __future__ import annotations

import pytest

from tests.regression._perf_utils import write_baseline


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-perf-baseline",
        action="store_true",
        default=False,
        help=(
            "Record measured kernel latencies as the new perf baseline for the "
            "current GPU class (writes benchmarks/baselines/{GPU}.json). Use "
            "after an intentional perf change."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    # Session-scoped measurement store. Populated by check_regression() when
    # --update-perf-baseline is active; flushed in pytest_sessionfinish.
    config._perf_measurements = {}  # type: ignore[attr-defined]


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    config = session.config
    if not config.getoption("--update-perf-baseline", default=False):
        return
    measurements = getattr(config, "_perf_measurements", {})
    if not measurements:
        return
    path = write_baseline(measurements)
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_sep("=", f"perf baseline written: {path}")
