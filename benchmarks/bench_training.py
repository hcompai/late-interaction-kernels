"""Training step (forward + backward) head-to-head: speed AND peak memory.

Targets the in-batch contrastive (ColBERT) and ColPali regimes. Measures a
full ``maxsim`` forward + ``.backward()`` with a ones upstream gradient, then
peak allocated memory for the same step. flash-maxsim is the external
reference point.

  sky exec lik-invest --gpus H100:1 \
    "cd ~/sky_workdir && source .venv/bin/activate && python benchmarks/bench_training.py"
"""

import statistics

import torch

from late_interaction_kernels import maxsim

try:
    from flash_maxsim.flash_maxsim_batched_train import flash_maxsim_batched_train

    HAS_FM = True
except Exception:
    HAS_FM = False


# (name, Nq, Nd, Lq, Ld, d)
SHAPES = [
    ("colbert-B128", 128, 128, 32, 180, 128),
    ("colbert-B256", 256, 256, 32, 180, 128),
    ("colbert-B512", 512, 512, 32, 180, 128),
    ("colpali-B8", 8, 8, 1024, 1024, 128),
    ("colpali-B16", 16, 16, 1024, 1024, 128),
    ("colpali-B32", 32, 32, 1024, 1024, 128),
]


def cuda_time(fn, warmup=10, iters=40):
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


def peak_mb(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - before) / 1024 / 1024


def main():
    dtype = torch.bfloat16
    print(f"GPU {torch.cuda.get_device_name()}  dtype=bf16  flash={HAS_FM}\n")
    for name, Nq, Nd, Lq, Ld, d in SHAPES:
        Q = torch.randn(Nq, Lq, d, device="cuda", dtype=dtype, requires_grad=True)
        D = torch.randn(Nd, Ld, d, device="cuda", dtype=dtype, requires_grad=True)
        g = torch.ones(Nq, Nd, device="cuda", dtype=torch.float32)

        def _lik():
            maxsim(Q, D).backward(g)
            Q.grad = None
            D.grad = None

        t_lik = cuda_time(_lik)
        m_lik = peak_mb(_lik)

        t_fm = m_fm = float("nan")
        if HAS_FM:

            def _fm():
                flash_maxsim_batched_train(Q, D).backward(g)
                Q.grad = None
                D.grad = None

            try:
                t_fm = cuda_time(_fm)
                m_fm = peak_mb(_fm)
            except Exception as ex:
                print(f"{name}: flash error {type(ex).__name__}: {ex}")

        ratio = (t_fm / t_lik) if t_fm == t_fm else float("nan")
        mratio = (m_lik / m_fm) if m_fm == m_fm and m_fm > 0 else float("nan")
        print(
            f"{name:14s} lik={t_lik:8.3f}ms/{m_lik:8.1f}MB  flash={t_fm:8.3f}ms/{m_fm:8.1f}MB  "
            f"| speed lik/flash {ratio:.2f}x  | mem lik/flash {mratio:.2f}x"
        )


if __name__ == "__main__":
    main()
