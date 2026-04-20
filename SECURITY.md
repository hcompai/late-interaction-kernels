# Security policy

## Reporting a vulnerability

Please **do not** open a public issue. Email the maintainers directly:

- Aurélien Lac — aurelien@h-company.ai
- Tony Wu — tony@h-company.ai

Include:

- A description of the issue and its impact
- Steps to reproduce (or a small PoC)
- Affected versions
- Any suggested mitigations

We'll acknowledge within **3 business days** and aim to ship a fix or
mitigation within **14 days** for confirmed issues.

## Scope

This project is a pure-compute GPU kernel library — it does not open
network sockets, read/write user files, or execute user code. The most
realistic concern is **out-of-bounds memory access** in the Triton
kernels that could leak adjacent GPU memory to the caller. Reports of
such issues are prioritized.

## Out of scope

- Issues in upstream dependencies (report to PyTorch / Triton / NVIDIA).
- Performance regressions without a correctness or safety angle — please
  file those as regular issues.
