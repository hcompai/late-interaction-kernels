## What & why

<!-- One or two sentences. Link any related issue. -->

## How

<!-- Brief notes on the approach. -->

## Test plan

<!-- How did you verify correctness? Shapes, GPU, numbers. -->

- [ ] `ruff check . && ruff format --check .` and `pytest -q` pass
- [ ] Parity vs `reference.maxsim_reference` holds for new numerical paths
- [ ] Benchmarks included for performance-motivated changes
- [ ] `CHANGELOG.md` and README updated for public API changes
