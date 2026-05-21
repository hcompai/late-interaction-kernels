# Prebuilt CI image: NVIDIA PyTorch base + our test-only deps preinstalled.
# Cuts the GPU CI cold start from ~7.5 min container pull + 30 s pip install
# down to a ~30 s pull from ghcr.io and a near-instant `pip install -e . --no-deps`.
#
# Pylate is intentionally omitted: it has unresolvable dependency conflicts
# against this image's pinned torch. The CI workflow auto-skips pylate tests
# when the import fails, matching prior behavior.
FROM nvcr.io/nvidia/pytorch:25.06-py3

# Versions mirror pyproject.toml `[project.optional-dependencies].dev`.
# Keep these in lockstep — the build workflow rebuilds when pyproject.toml
# changes, so any drift here triggers a fresh image.
RUN pip install --no-cache-dir \
    "numpy>=1.21,<3" \
    "pandas>=1.5,<4" \
    "pytest>=7,<10" \
    "pytest-xdist>=3,<4" \
    "tabulate>=0.9,<1"
