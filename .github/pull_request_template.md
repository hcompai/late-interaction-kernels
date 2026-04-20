<!--
Thanks for the PR! Please fill in the sections that apply. Delete the rest.
-->

## What

<!-- One or two sentences. -->

## Why

<!-- Motivation. Link to any related issue. -->

## How

<!-- Brief notes on the approach, any alternatives considered. -->

## Test plan

<!-- How did you verify correctness? What shapes, what GPU, what numbers? -->

- [ ] `ruff check . && ruff format --check .` passes
- [ ] `pytest -q` passes (state: CPU / `<GPU name>`)
- [ ] Parity vs `reference.maxsim_reference` holds for new numerical paths
- [ ] Benchmarks included for performance-motivated changes
- [ ] `CHANGELOG.md` updated under **Unreleased**
- [ ] Public API changes documented in the README
