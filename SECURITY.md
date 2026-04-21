# Security Policy

`late-interaction-kernels` is a numerical-kernels library — it has no
network or file I/O of its own, and no persistent state. Even so, we
take security reports seriously. This document explains what to do if
you find something.

## Supported versions

We support the latest minor release on PyPI, and the `main` branch.
Older minor releases receive fixes at our discretion.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for suspected
vulnerabilities.

Email [opensource@hcompany.ai](mailto:opensource@hcompany.ai) with:

- A description of the issue and its impact (crash / UB / incorrect
  numerics under adversarial input / etc.).
- A minimal reproducer: PyTorch version, Triton version, CUDA version,
  GPU, and the shape / dtype / mask that triggers it.
- Any suggested fix or patch if you have one.

We will acknowledge receipt within 3 business days, and aim to issue a
fix or disclosure plan within 30 days of confirmed reports.

## Scope

In scope:

- Kernel bugs that produce incorrect numerics for valid inputs.
- Out-of-bounds reads / writes on the GPU that leak data or corrupt
  adjacent allocations.
- Host-side crashes or hangs triggered by well-typed tensor inputs.
- Memory-safety issues in the Python wrapper layer.

Out of scope:

- CUDA / Triton / PyTorch upstream bugs — please report those upstream.
- Numerical drift within documented tolerances (see the parity tests in
  `tests/`).
- Denial-of-service via obviously oversized inputs (we validate shapes;
  we do not cap VRAM usage).

## Disclosure

Once a fix is available on `main` and a release is tagged, we will
publish an advisory describing the issue and credit the reporter (with
permission).
