# Contributing

## Reporting issues

Use the **Bug report** or **Feature request** templates under
[Issues → New issue](https://github.com/hcompai/late-interaction-kernels/issues/new/choose).

## Autotune for a new GPU

If performance is poor on a GPU we don't have a shortlist for:

1. Run the benchmark for the shape you care about
   (`benchmarks/bench_forward.py`, `benchmarks/bench_inference_edge.py`,
   `benchmarks/bench_backward_method.py`).
2. Add a shortlist in `late_interaction_kernels/_autotune.py` keyed on
   the device-name prefix.
3. Re-run the benchmark and include before / after in the PR.

## New kernel variant

For a new reduction flavor (e.g. top-K, soft variants), keep it in a
separate module under `late_interaction_kernels/` and follow the
existing split:

- internal `_forward` returning `(scores, argmax)` without autograd;
- `torch.autograd.Function` wrapper that saves minimal state;
- pure-PyTorch reference in `late_interaction_kernels/reference.py`;
- parity tests in `tests/`.

Research kernels with no production user yet land under
`late_interaction_kernels/experimental/`.

## Development setup

```bash
git clone https://github.com/hcompai/late-interaction-kernels
cd late-interaction-kernels
pip install -e ".[dev,pylate]"

ruff check . && ruff format --check .
pytest -q
```

## PR checklist

- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] `pytest -q` passes (state: CPU / `<GPU name>`)
- [ ] New behavior has tests; bug fixes have a regression test
- [ ] Numerical changes include a parity test vs `reference.maxsim_reference`
- [ ] Benchmarks included if the change is performance-motivated
- [ ] `CHANGELOG.md` updated under the in-flight `Unreleased` block
- [ ] Public API changes mentioned in the README

## Style

- Python 3.9+; type hints on public APIs.
- Comments explain *why*, not *what*. Don't narrate trivial code.
- Match the existing docstring tone — short, concrete, no marketing.

## Publishing a release to PyPI

Releases are published by the
[`Publish Python Package`](.github/workflows/publish.yml) workflow, which
runs on `release: published` and uploads to PyPI via OIDC trusted
publishing (no token kept in repo secrets). The package version is
derived from the git tag by `hatch-vcs`, so there is no version literal
to bump.

To cut a release:

1. Make sure `main` is green and `CHANGELOG.md` has the
   `Unreleased` block filled in for the version you're about to ship.
2. On GitHub: **Releases → Draft a new release**.
3. Under **Choose a tag**, type `vX.Y.Z` and pick *Create new tag on
   publish*. Target `main`.
4. Title the release `vX.Y.Z` and paste the matching `CHANGELOG.md`
   section into the body.
5. Click **Publish release**. The workflow builds the sdist + wheel and
   uploads them to PyPI; watch it under **Actions → Publish Python
   Package**.

One-time PyPI setup (already done for this project, kept here for
reference):
[pypi.org/manage/account/publishing/](https://pypi.org/manage/account/publishing/)
→ *Add a new pending publisher* with project `late-interaction-kernels`,
owner `hcompai`, repo `late-interaction-kernels`, workflow
`publish.yml`, environment `pypi`. Then in this repo:
**Settings → Environments → New environment** named `pypi`.

## License

By contributing you agree your work is licensed under Apache 2.0
(see [`LICENSE`](LICENSE)).
