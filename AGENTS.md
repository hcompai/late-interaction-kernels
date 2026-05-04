# AGENTS.md

Notes for AI coding agents (Claude, Codex, Cursor, etc.) working in this repo.
Humans should read [`CONTRIBUTING.md`](CONTRIBUTING.md) first — this file
strictly extends it for non-human contributors.

## Pre-merge checklist

Before opening a PR for review or merging an existing one, every agent MUST:

1. **Update [`CHANGELOG.md`](CHANGELOG.md)** under the `## [Unreleased]`
   section, following [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/).
   - Group entries under `### Added`, `### Changed`, `### Deprecated`,
     `### Removed`, `### Fixed`, or `### Security` (omit empty groups).
   - One bullet per user-visible change. Reference public API names
     in backticks. Keep the tone of existing entries — concrete, no marketing.
   - If the `## [Unreleased]` heading does not exist (because the previous
     release just shipped), insert it directly above the latest version.
2. Run `ruff check . && ruff format --check .` and `ty check
   late_interaction_kernels/` — both must be clean (no new diagnostics).
3. Run `pytest -q` (or the targeted test file when iterating) and confirm
   no regressions vs the pre-change baseline. Pre-existing flaky/skipped
   tests are fine; document them in the PR body if relevant.
4. Never commit anything in `benchmarks/results/` — it is `.gitignore`d.

## Style

- Python 3.9+; type hints on public APIs.
- Comments explain *why*, never narrate *what* the code does.
- No emoji in code, comments, or commit messages unless the user explicitly
  asks for them.
- Match existing docstring tone — short, concrete, no marketing.
- Don't add a `from __future__ import annotations` or other style sweeps in
  unrelated PRs; keep diffs tight.

## Commits

- Conventional-style prefixes are used loosely (`bench(mps): …`, `mps: …`,
  `fix: …`). Match the surrounding history; don't invent a new convention.
- Each commit message body should explain *why* the change is needed, not
  restate the diff.

## When in doubt

Read `CONTRIBUTING.md` for kernel-authoring conventions, the autotune
workflow, and the release process.
