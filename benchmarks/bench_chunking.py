"""Large-Lq query-chunking: speedup + regression guard.

Compares the chunked public ``maxsim`` against the un-chunked kernel core
(``_maxsim_cross``) on the same inputs, across the realistic large-Lq grid
(ColPali visual patches, long-doc rerank, in-batch contrastive at long Lq).
flash-maxsim is shown only as an external reference point.

  sky exec lik-invest --gpus H100:1 \
    "cd ~/sky_workdir && source .venv/bin/activate && python benchmarks/bench_chunking.py"
"""

import statistics

import torch

from late_interaction_kernels import maxsim
from late_interaction_kernels.autograd import _maxsim_cross

try:
    from flash_maxsim import flash_maxsim_batched

    HAS_FM = True
except Exception:
    HAS_FM = False


# (name, Nq, Nd, Lq, Ld, d). Mix of the chunked regime (Lq > 512) and the
# boundary cases that must stay un-chunked (Lq <= 512 → speedup == 1.00x by
# construction, since maxsim falls through to the same core).
SHAPES = [
    ("colpali-rerank   Nq=1   Nd=500 Lq=1024", 1, 500, 1024, 1024, 128),
    ("colpali-inbatch  Nq=Nd=16     Lq=1024", 16, 16, 1024, 1024, 128),
    ("colpali-inbatch  Nq=Nd=32     Lq=1024", 32, 32, 1024, 1024, 128),
    ("colpali-tail     Nq=Nd=16     Lq=1030", 16, 16, 1030, 1024, 128),
    ("longq-inbatch    Nq=Nd=16     Lq=768 ", 16, 16, 768, 512, 128),
    ("longq-inbatch    Nq=Nd=64     Lq=768 ", 64, 64, 768, 512, 128),
    ("boundary-nochunk Nq=Nd=256    Lq=256 ", 256, 256, 256, 180, 128),
    ("boundary-nochunk Nq=Nd=128    Lq=512 ", 128, 128, 512, 512, 128),
]


def cuda_time(fn, warmup=20, iters=60):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def main():
    dtype = torch.bfloat16
    print(f"GPU {torch.cuda.get_device_name()}  dtype=bf16  flash={HAS_FM}\n")
    print(
        f"{'shape':40s} {'unchunked':>10s} {'chunked':>10s} {'speedup':>8s} {'flash':>9s} {'chk/flash':>9s}"
    )
    worst = 1e9
    for name, Nq, Nd, Lq, Ld, d in SHAPES:
        Q = torch.randn(Nq, Lq, d, device="cuda", dtype=dtype)
        D = torch.randn(Nd, Ld, d, device="cuda", dtype=dtype)

        # correctness: chunked == unchunked core
        ref = _maxsim_cross(Q, D, None, None, False, "auto")
        chk = maxsim(Q, D)
        err = (ref - chk).abs().max().item()

        t_unchunk = cuda_time(lambda: _maxsim_cross(Q, D, None, None, False, "auto"))
        t_chunk = cuda_time(lambda: maxsim(Q, D))
        sp = t_unchunk / t_chunk
        worst = min(worst, sp)
        t_flash = cuda_time(lambda: flash_maxsim_batched(Q, D, shared_docs=True)) if HAS_FM else float("nan")
        cf = (t_flash / t_chunk) if HAS_FM else float("nan")
        flag = "  <-- REGRESSION" if sp < 0.97 else ""
        print(
            f"{name:40s} {t_unchunk:9.3f}m {t_chunk:9.3f}m {sp:7.2f}x {t_flash:8.3f}m {cf:8.2f}x"
            f"  err={err:.1e}{flag}"
        )
    print(f"\nworst chunked/unchunked speedup: {worst:.2f}x (>=0.97 = no regression)")

    # --- training (forward + backward) on the chunked regime --------------
    # The grad path chunks too (Q.reshape fans D into nc autograd nodes); the
    # grad_D atomic scatter does the same total atomic adds either way, so the
    # expectation is neutral-to-better — large batches win big (Nq=64: +30%),
    # while at tiny batches (Nq=Nd=16) the step is dominated by the un-chunked
    # backward and the result sits in the noise (the same 16x16 batch flips
    # sign between Lq=1024 and Lq=1030 run to run), so the guard is looser here.
    _TRAIN_REGRESSION = 0.90
    print(f"\n{'shape (train fwd+bwd)':40s} {'unchunked':>10s} {'chunked':>10s} {'speedup':>8s}")
    worst_tr = 1e9
    for name, Nq, Nd, Lq, Ld, d in SHAPES:
        if Lq <= 512:  # only the chunked regime is interesting for training
            continue
        Q = torch.randn(Nq, Lq, d, device="cuda", dtype=dtype, requires_grad=True)
        D = torch.randn(Nd, Ld, d, device="cuda", dtype=dtype, requires_grad=True)
        g = torch.randn(Nq, Nd, device="cuda", dtype=dtype)

        def _unchunk():
            Q.grad = D.grad = None
            _maxsim_cross(Q, D, None, None, False, "auto").backward(g)

        def _chunk():
            Q.grad = D.grad = None
            maxsim(Q, D).backward(g)

        t_un = cuda_time(_unchunk, warmup=10, iters=30)
        t_ch = cuda_time(_chunk, warmup=10, iters=30)
        sp = t_un / t_ch
        worst_tr = min(worst_tr, sp)
        flag = "  <-- REGRESSION" if sp < _TRAIN_REGRESSION else ""
        print(f"{name:40s} {t_un:9.3f}m {t_ch:9.3f}m {sp:7.2f}x{flag}")
    print(f"\nworst chunked/unchunked train speedup: {worst_tr:.2f}x (>={_TRAIN_REGRESSION} = no regression)")


if __name__ == "__main__":
    main()
