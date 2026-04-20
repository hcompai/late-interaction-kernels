#!/usr/bin/env bash
# Run every benchmark in the repo and dump artifacts into ${OUTDIR:-benchmarks/results}.
#
# Used both locally (single GPU) and by the SkyPilot 8xH100 job. Individual
# benchmarks are independent; a failure in one won't stop the rest.

set -u
OUTDIR="${OUTDIR:-benchmarks/results}"
mkdir -p "${OUTDIR}"

run() {
  echo ""
  echo "=================================================================="
  echo "Running: $*"
  echo "=================================================================="
  ( "$@" ) || echo "WARN: $* failed (exit $?) — continuing"
}

run python benchmarks/bench_forward.py          --outdir "${OUTDIR}"
run python benchmarks/bench_backward_method.py  --outdir "${OUTDIR}"
run python benchmarks/bench_normalize.py        --outdir "${OUTDIR}"
run python benchmarks/bench_new_kernels.py      --outdir "${OUTDIR}"
run python benchmarks/bench_moderncolbert.py    --outdir "${OUTDIR}"
run python benchmarks/bench_cached_maxsim.py    --outdir "${OUTDIR}"

# bench_fastplaid installs lightonai/fastplaid; skip if the package isn't
# available (we don't want to hard-depend on Rust toolchain at bench time).
if python -c "import fast_plaid" >/dev/null 2>&1; then
  run python benchmarks/bench_fastplaid.py      --outdir "${OUTDIR}"
else
  echo "INFO: fast_plaid not installed, skipping bench_fastplaid.py"
fi

echo ""
echo "All benchmarks done. Artifacts:"
ls -la "${OUTDIR}"
