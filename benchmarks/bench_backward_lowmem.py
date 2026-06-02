"""Realistic PyLate / ColPali memory + speed: inference and training.

Reports forward-only (inference) and forward+backward (training) peak memory
and time for the shapes that actually stress memory: ColPali in-batch visual,
PyLate text in-batch, and the 4-D hard-negative layout (where grad_D is
n_neg-inflated). Compares LIK against flash-maxsim 0.2.1.
"""

import argparse
import statistics

import torch

from late_interaction_kernels import maxsim


def _time_ms(fn, iters=20, warmup=6):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def _measure(make_inputs, run, train):
    inp = make_inputs()

    def step():
        for t in inp:
            if t.grad is not None:
                t.grad = None
        out = run(inp)
        if train:
            out.sum().backward()

    for _ in range(6):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1e6
    t = _time_ms(step)
    for t_ in inp:
        t_.grad = None
    torch.cuda.empty_cache()
    return t, peak


def lik_cross(Nq, Nd, Lq, Ld, d, dtype, backend, train):
    def make():
        return [
            torch.randn(Nq, Lq, d, device="cuda", dtype=dtype, requires_grad=train),
            torch.randn(Nd, Ld, d, device="cuda", dtype=dtype, requires_grad=train),
        ]

    return _measure(make, lambda inp: maxsim(inp[0], inp[1], backward=backend), train)


def lik_neg(B, nn, Lq, Ld, d, dtype, backend, train):
    def make():
        return [
            torch.randn(B, Lq, d, device="cuda", dtype=dtype, requires_grad=train),
            torch.randn(B, nn, Ld, d, device="cuda", dtype=dtype, requires_grad=train),
        ]

    return _measure(make, lambda inp: maxsim(inp[0], inp[1], backward=backend), train)


def flash_cross(Nq, Nd, Lq, Ld, d, dtype, train):
    from flash_maxsim import flash_maxsim_batched, flash_maxsim_batched_train

    def make():
        return [
            torch.randn(Nq, Lq, d, device="cuda", dtype=dtype, requires_grad=train),
            torch.randn(Nd, Ld, d, device="cuda", dtype=dtype, requires_grad=train),
        ]

    if train:
        fn = lambda inp: flash_maxsim_batched_train(inp[0], inp[1], shared_docs=True)  # noqa: E731
    else:
        fn = lambda inp: flash_maxsim_batched(inp[0], inp[1], shared_docs=True)  # noqa: E731
    return _measure(make, fn, train)


def flash_neg(B, nn, Lq, Ld, d, dtype, train):
    from flash_maxsim import flash_maxsim_batched, flash_maxsim_batched_train

    def make():
        return [
            torch.randn(B, Lq, d, device="cuda", dtype=dtype, requires_grad=train),
            torch.randn(B, nn, Ld, d, device="cuda", dtype=dtype, requires_grad=train),
        ]

    # shared_docs=False (KD/non-shared) wants D as [Nq, B, Ld, d]; our [B, nn, Ld, d]
    # is exactly that with Nq=B, B=nn.
    if train:
        fn = lambda inp: flash_maxsim_batched_train(inp[0], inp[1], shared_docs=False)  # noqa: E731
    else:
        fn = lambda inp: flash_maxsim_batched(inp[0], inp[1], shared_docs=False)  # noqa: E731
    return _measure(make, fn, train)


def pylate_cross(Nq, Nd, Lq, Ld, d, dtype, train):
    """PyLate's colbert_scores: materializes [Nq, Nd, Lq, Ld] via einsum."""

    def make():
        return [
            torch.randn(Nq, Lq, d, device="cuda", dtype=dtype, requires_grad=train),
            torch.randn(Nd, Ld, d, device="cuda", dtype=dtype, requires_grad=train),
        ]

    def run(inp):
        s = torch.einsum("ash,bth->abst", inp[0], inp[1])
        return s.max(axis=-1).values.sum(axis=-1)

    return _measure(make, run, train)


def pylate_neg(B, nn, Lq, Ld, d, dtype, train):
    """PyLate's colbert_kd_scores: materializes [B, nn, Lq, Ld] via einsum."""

    def make():
        return [
            torch.randn(B, Lq, d, device="cuda", dtype=dtype, requires_grad=train),
            torch.randn(B, nn, Ld, d, device="cuda", dtype=dtype, requires_grad=train),
        ]

    def run(inp):
        s = torch.einsum("ash,abth->abst", inp[0], inp[1])
        return s.max(axis=-1).values.sum(axis=-1)

    return _measure(make, run, train)


CROSS = [
    # realistic retrieval: short text query vs long doc, in-batch [B,B]
    ("pylate-text B256 Lq32 Ld300", 256, 256, 32, 300, 128),
    ("colpali B64 Lq32 Ld1030", 64, 64, 32, 1030, 128),
    ("colpali B128 Lq32 Ld1030", 128, 128, 32, 1030, 128),
    # stress: long query AND long doc (doc-doc / visual-as-query)
    ("longq B64 Lq1030 Ld1030", 64, 64, 1030, 1030, 128),
]
NEG = [
    ("colpali-neg B128 n8 Ld1030", 128, 8, 32, 1030, 128),
    ("colpali-neg B256 n16 Ld1030", 256, 16, 32, 1030, 128),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    args = ap.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    def fmt(t, p):
        return f"{t:.3f}ms/{p:.0f}MB"

    print(f"# Realistic LIK vs flash-maxsim 0.2.1, {args.dtype}, H100\n")
    print("| shape | mode | LIK inf | flash inf | LIK train(unified) | LIK train(lowmem) | flash train |")
    print("|---|---|---|---|---|---|---|")
    for tag, Nq, Nd, Lq, Ld, d in CROSS:
        li = lik_cross(Nq, Nd, Lq, Ld, d, dtype, "unified", train=False)
        lt = lik_cross(Nq, Nd, Lq, Ld, d, dtype, "unified", train=True)
        llm = lik_cross(Nq, Nd, Lq, Ld, d, dtype, "lowmem", train=True)
        try:
            fi = flash_cross(Nq, Nd, Lq, Ld, d, dtype, train=False)
        except Exception as e:  # noqa: BLE001
            fi = (float("nan"), float("nan"))
            print(f"  ! {tag} flash inf: {e}")
        try:
            ft = flash_cross(Nq, Nd, Lq, Ld, d, dtype, train=True)
        except Exception as e:  # noqa: BLE001
            ft = (float("nan"), float("nan"))
            print(f"  ! {tag} flash train: {e}")
        print(f"| {tag} | cross | {fmt(*li)} | {fmt(*fi)} | {fmt(*lt)} | {fmt(*llm)} | {fmt(*ft)} |")

    for tag, B, nn, Lq, Ld, d in NEG:
        li = lik_neg(B, nn, Lq, Ld, d, dtype, "unified", train=False)
        lt = lik_neg(B, nn, Lq, Ld, d, dtype, "unified", train=True)
        llm = lik_neg(B, nn, Lq, Ld, d, dtype, "lowmem", train=True)
        try:
            fi = flash_neg(B, nn, Lq, Ld, d, dtype, train=False)
        except Exception as e:  # noqa: BLE001
            fi = (float("nan"), float("nan"))
            print(f"  ! {tag} flash inf: {e}")
        try:
            ft = flash_neg(B, nn, Lq, Ld, d, dtype, train=True)
        except Exception as e:  # noqa: BLE001
            ft = (float("nan"), float("nan"))
            print(f"  ! {tag} flash train: {e}")
        print(f"| {tag} | neg4d | {fmt(*li)} | {fmt(*fi)} | {fmt(*lt)} | {fmt(*llm)} | {fmt(*ft)} |")

    # PyLate-naive (materialized einsum) training peak — OOM => caught.
    print("\n## vs PyLate-naive (materialized) training peak\n")
    print("| shape | LIK train(best) | PyLate-naive train |")
    print("|---|---|---|")
    for tag, Nq, Nd, Lq, Ld, d in CROSS:
        lt = lik_cross(Nq, Nd, Lq, Ld, d, dtype, "unified", train=True)
        try:
            pt = pylate_cross(Nq, Nd, Lq, Ld, d, dtype, train=True)
            ps = fmt(*pt)
        except Exception as e:  # noqa: BLE001
            ps = f"OOM/{type(e).__name__}"
            torch.cuda.empty_cache()
        print(f"| {tag} | {fmt(*lt)} | {ps} |")
    for tag, B, nn, Lq, Ld, d in NEG:
        lt = lik_neg(B, nn, Lq, Ld, d, dtype, "lowmem", train=True)
        try:
            pt = pylate_neg(B, nn, Lq, Ld, d, dtype, train=True)
            ps = fmt(*pt)
        except Exception as e:  # noqa: BLE001
            ps = f"OOM/{type(e).__name__}"
            torch.cuda.empty_cache()
        print(f"| {tag} | {fmt(*lt)} | {ps} |")


if __name__ == "__main__":
    main()
