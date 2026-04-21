# Contributing

Thanks for your interest! This is a performance library — most contributions
fall into one of four buckets.

## 1. Report a bug or performance regression

Open an issue with:

- GPU (e.g. `H100 80 GB SXM`, `A100 40 GB PCIe`), CUDA version, Triton version,
  PyTorch version — `python -c "import torch; print(torch.__version__, torch.version.cuda)"`
- Shape that triggers it: `(Nq, Nd, Lq, Ld, d)`, dtype, mask usage
- Minimal reproducer (ideally < 20 lines)
- Expected vs observed (numbers / traceback)

## 2. Add an autotune config for your hardware

If you see slow performance on a GPU we don't have a shortlist for:

1. Run the benchmark matching your shape of interest:
   - `python benchmarks/bench_forward.py` — PyLate training shapes (d=128).
   - `python benchmarks/bench_inference_edge.py` — small-d rerankers
     (LateOn-Code-edge d=48, mxbai-edge d=64) at long context & high BS.
   - `python benchmarks/bench_backward_0_5.py` — fused backward
     (`maxsim_residual`, `maxsim_varlen`) vs the reference autograd path.
2. Add a shortlist entry to `late_interaction_kernels/_autotune.py` keyed
   on the device name prefix.
3. Re-run the benchmark and include before / after in the PR.

## 3. New kernel variant

For a new reduction flavor (e.g. top-k instead of max, soft variants, a
different score aggregator), keep it in a separate module under
`late_interaction_kernels/` and expose via `__init__.py` only if it's
generally useful. Match the `maxsim_forward` / `maxsim` split:

- Internal `_forward` returning `(scores, argmax)` without autograd
- `torch.autograd.Function` wrapper that saves minimal state
- Reference implementation in `late_interaction_kernels/reference.py`
- Parity tests in `tests/`

## 4. Docs / examples

Always welcome. Keep examples runnable and under ~30 lines.

## Development setup

```bash
git clone https://github.com/hcompai/late-interaction-kernels
cd late-interaction-kernels
pip install -e ".[dev,pylate]"

ruff check . && ruff format --check .
pytest -q
```

## Pull-request checklist

- [ ] `ruff check . && ruff format --check .` passes
- [ ] `pytest -q` passes on your machine (state: CPU / `<GPU name>`)
- [ ] New behavior has tests; bug fixes have a regression test
- [ ] Numerical changes include a parity test vs `reference.maxsim_reference`
- [ ] Benchmarks included if the change is performance-motivated
- [ ] CHANGELOG.md updated (add a new `## <next-version> — Unreleased` block
      at the top if one doesn't exist, or append to the existing in-flight one)
- [ ] Public API changes documented in the README

## Style

- Python 3.9+ idioms, type hints on public APIs, no implicit `Any`.
- Follow the existing file layout and docstring tone — short,
  non-redundant, pointers to the mechanism not the syntax.
- Comments explain *why*, not *what*. Don't narrate trivial code.

## License

By contributing, you agree your contributions will be licensed under
Apache 2.0 (see [`LICENSE`](LICENSE)).
