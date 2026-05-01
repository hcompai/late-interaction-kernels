# Contributing

## Bugs and performance regressions

Open an issue with:

- GPU, CUDA, Triton and PyTorch versions
  (`python -c "import torch; print(torch.__version__, torch.version.cuda)"`)
- Shape that triggers it: `(Nq, Nd, Lq, Ld, d)`, dtype, mask usage
- Minimal reproducer (< 20 lines) and expected vs observed.

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

## License

By contributing you agree your work is licensed under Apache 2.0
(see [`LICENSE`](LICENSE)).
