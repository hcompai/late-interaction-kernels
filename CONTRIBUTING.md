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

## Style

- Python 3.9+; type hints on public APIs.
- Comments explain *why*, not *what*. Don't narrate trivial code.
- Match the existing docstring tone — short, concrete, no marketing.

## Publishing a release

1. Ensure `main` is green and `CHANGELOG.md` has the `Unreleased` block filled in.
2. On GitHub: **Releases → Draft a new release**, tag `vX.Y.Z` off `main`.
3. Paste the matching `CHANGELOG.md` section as the release body, then **Publish**.

The [`publish.yml`](.github/workflows/publish.yml) workflow builds and uploads
to PyPI automatically via OIDC trusted publishing. No token needed.

## License

By contributing you agree your work is licensed under Apache 2.0
(see [`LICENSE`](LICENSE)).
