# Contributing

## Reporting issues

Use the **Bug report** / **Feature request** templates under
[Issues → New issue](https://github.com/hcompai/late-interaction-kernels/issues/new/choose).

## Development setup

```bash
git clone https://github.com/hcompai/late-interaction-kernels
cd late-interaction-kernels
pip install -e ".[dev,pylate]"

ruff check . && ruff format --check .
pytest -q
```

## Style

- Python 3.10+; type hints on public APIs.
- Comments explain *why*, not *what*.
- Docstrings: short, concrete, no marketing.

## Autotune for a new GPU

Run the relevant `benchmarks/bench_*.py`, add a shortlist in
`late_interaction_kernels/_autotune.py` keyed on the device-name prefix,
re-run and include before / after in the PR.

## New kernel variant

Keep it in its own module under `late_interaction_kernels/` and follow
the existing split: internal `_forward` returning `(scores, argmax)`,
`torch.autograd.Function` wrapper, pure-PyTorch reference in
`reference.py`, parity tests in `tests/`.

## Publishing a release

1. `main` is green and `CHANGELOG.md` `Unreleased` is filled in.
2. **Releases → Draft a new release** on GitHub, tag `vX.Y.Z` off `main`.
3. Paste the matching `CHANGELOG.md` section as the body, then **Publish**.

[`publish.yml`](.github/workflows/publish.yml) builds and uploads to PyPI
via OIDC — no token needed.

## License

By contributing you agree your work is licensed under Apache 2.0
(see [`LICENSE`](LICENSE)).
