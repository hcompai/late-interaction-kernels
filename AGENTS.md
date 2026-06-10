# AGENTS.md

Notes for AI coding agents. Humans: read [`CONTRIBUTING.md`](CONTRIBUTING.md)
first — this extends it. The repo is uv-managed: always `uv run <tool>`,
never bare `ruff`/`pytest`.

## Before committing

1. `uv run ruff check . && uv run ruff format --check .` (format + stage if
   needed); `uv run ty check late_interaction_kernels/` if types changed.
2. `uv run pytest -q` — no regressions. CUDA tests auto-skip without a GPU,
   so local green does **not** validate kernel changes.

## Before opening a PR

1. Update `CHANGELOG.md` under `## [Unreleased]`
   ([Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/)): one
   concrete bullet per user-visible change, API names in backticks.
2. Sweep `docs/how-it-works.html`, `docs/design.md`, and the README tables
   for claims your change invalidates — most doc drift comes from skipping this.
3. Kernel or dispatch change? Validate perf on an H100
   (`sky launch scripts/sky_run_all_benchmarks.yaml`, then `sky down`) and
   request GPU CI (`run-gpu-tests` label, triage+, or `workflow_dispatch`).
   Post numbers as a PR comment; only edit `docs/benchmarks.md` beyond run noise.
4. Never commit `benchmarks/results/` (gitignored).

## Style, commits, PRs

- Python 3.10+, native `X | Y` / `list[X]` typing — no
  `from __future__ import annotations`, no `typing.List`.
- Comments explain *why*, never *what*. Docstrings short, no marketing.
  No emoji. No drive-by style sweeps.
- Commits follow [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/)
  (`type(scope): subject`, e.g. `fix(plaid): …`, `bench(mps): …`); bodies say why.
- No AI attribution in commits, PR descriptions, or PR comments.
