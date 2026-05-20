# AGENTS.md

Notes for AI coding agents (Claude, Codex, Cursor, etc.) working in this repo.
Humans should read [`CONTRIBUTING.md`](CONTRIBUTING.md) first — this file
strictly extends it for non-human contributors.

## Per-task checklist

After **every** task that touches Python — code, tests, comments,
docstrings — and **before** committing, every agent MUST:

1. Run `ruff check . && ruff format --check .` from the repo root, and
   `ty check late_interaction_kernels/` if types changed. Both must be
   clean (no new diagnostics). If `ruff format --check` reports a
   reformat, run `ruff format` and stage the result.
2. Run `pytest -q` (or the targeted test file when iterating) and
   confirm no regressions vs the pre-change baseline. Pre-existing
   flaky/skipped tests are fine.

These steps run **after every task**, not just at PR time — CI runs the
same checks and a failure caught locally is a free fix. Do not commit
without the linter being clean.

## Pre-merge checklist

Before opening a PR for review or merging an existing one, every agent
MUST (in addition to the per-task checklist above):

1. **Update [`CHANGELOG.md`](CHANGELOG.md)** under the `## [Unreleased]`
   section, following [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/).
   - Group entries under `### Added`, `### Changed`, `### Deprecated`,
     `### Removed`, `### Fixed`, or `### Security` (omit empty groups).
   - One bullet per user-visible change. Reference public API names
     in backticks. Keep the tone of existing entries — concrete, no marketing.
   - If the `## [Unreleased]` heading does not exist (because the previous
     release just shipped), insert it directly above the latest version.
2. Never commit anything in `benchmarks/results/` — it is `.gitignore`d.

## Style

- Python 3.10+; type hints on public APIs. Use the native PEP 604 / PEP
  585 syntax (`X | Y`, `list[X]`, `dict[K, V]`) directly — no
  `from __future__ import annotations` and no `typing.List` /
  `typing.Optional`.
- Comments explain *why*, never narrate *what* the code does.
- No emoji in code, comments, or commit messages unless the user explicitly
  asks for them.
- Match existing docstring tone — short, concrete, no marketing.
- Keep diffs tight: no drive-by style sweeps in unrelated PRs.

## Commits

- Conventional-style prefixes are used loosely (`bench(mps): …`, `mps: …`,
  `fix: …`). Match the surrounding history; don't invent a new convention.
- Each commit message body should explain *why* the change is needed, not
  restate the diff.

## When in doubt

Read `CONTRIBUTING.md` for kernel-authoring conventions, the autotune
workflow, and the release process.
