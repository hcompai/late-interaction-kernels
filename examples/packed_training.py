"""Runnable packed / varlen training demo.

Compares a padded MaxSim training step to a packed MaxSim training step on a
synthetic heterogeneous corpus (short queries, long-tail document lengths).
Reports per-step loss, wall-clock, and peak memory for both paths so the
padding-waste vs packing-win tradeoff is visible on your hardware.

Usage::

    python examples/packed_training.py
    python examples/packed_training.py --batch-size 32 --steps 50

Requires CUDA + Triton (the kernels are GPU-only). The docs/packed_training.md
cookbook walks through the pieces in more detail.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from late_interaction_kernels import maxsim, maxsim_varlen

# ---------- synthetic heterogeneous batch --------------------------------


def sample_lengths(batch_size: int, short: int, long: int, generator: torch.Generator) -> list[int]:
    """Half the batch is short, half has a power-law-ish long tail."""
    half = batch_size // 2
    short_lens = [short] * half
    # Long tail: uniform in [short, long], with a heavy bias toward `long`.
    u = torch.rand(batch_size - half, generator=generator)
    long_lens = (short + (long - short) * (u**0.3)).round().clamp_min(1).to(torch.int64).tolist()
    out = short_lens + long_lens
    # Shuffle so the "long" sequences are not all at the end.
    idx = torch.randperm(len(out), generator=generator).tolist()
    return [out[i] for i in idx]


def sample_batch(
    batch_size: int,
    d: int,
    q_short: int,
    q_long: int,
    d_short: int,
    d_long: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Return lists of [L_i, d] tensors for queries / positives / negatives."""
    q_lens = sample_lengths(batch_size, q_short, q_long, generator)
    p_lens = sample_lengths(batch_size, d_short, d_long, generator)
    n_lens = sample_lengths(batch_size, d_short, d_long, generator)
    Q = [torch.randn(L, d, device=device, dtype=dtype, generator=generator) for L in q_lens]
    P = [torch.randn(L, d, device=device, dtype=dtype, generator=generator) for L in p_lens]
    N = [torch.randn(L, d, device=device, dtype=dtype, generator=generator) for L in n_lens]
    return Q, P, N


# ---------- pack / pad helpers -------------------------------------------


def pack(seqs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    packed = torch.cat(seqs, dim=0).contiguous()
    lens = torch.tensor([s.shape[0] for s in seqs], dtype=torch.int32, device=packed.device)
    cu = torch.zeros(len(seqs) + 1, dtype=torch.int32, device=packed.device)
    cu[1:] = lens.cumsum(0)
    return packed, cu


def pad(seqs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    B = len(seqs)
    L = max(s.shape[0] for s in seqs)
    d = seqs[0].shape[1]
    out = torch.zeros(B, L, d, device=seqs[0].device, dtype=seqs[0].dtype)
    mask = torch.zeros(B, L, device=seqs[0].device, dtype=torch.bool)
    for i, s in enumerate(seqs):
        out[i, : s.shape[0]] = s
        mask[i, : s.shape[0]] = True
    return out, mask


# ---------- training steps -----------------------------------------------


def padded_step(Q, P, N):
    """One contrastive step via the padded MaxSim kernel."""
    Qp, Qm = pad(Q)
    Pp, Pm = pad(P)
    Np, Nm = pad(N)
    Qn = F.normalize(Qp, dim=-1)
    Pn = F.normalize(Pp, dim=-1)
    Nn = F.normalize(Np, dim=-1)
    s_pos = maxsim(Qn, Pn, q_mask=Qm, d_mask=Pm)
    s_neg = maxsim(Qn, Nn, q_mask=Qm, d_mask=Nm)
    scores = torch.cat([s_pos, s_neg], dim=1)
    labels = torch.arange(Qn.shape[0], device=scores.device)
    return F.cross_entropy(scores, labels)


def packed_step(Q, P, N):
    """One contrastive step via the packed MaxSim kernel."""
    Qn = [F.normalize(x, dim=-1) for x in Q]
    Pn = [F.normalize(x, dim=-1) for x in P]
    Nn = [F.normalize(x, dim=-1) for x in N]
    Qp, cu_q = pack(Qn)
    Pp, cu_p = pack(Pn)
    Np, cu_n = pack(Nn)

    Dp = torch.cat([Pp, Np], dim=0)
    cu_d = torch.cat([cu_p, cu_p[-1] + cu_n[1:]], dim=0)

    scores = maxsim_varlen(Qp, Dp, cu_q, cu_d)
    labels = torch.arange(cu_q.numel() - 1, device=scores.device)
    return F.cross_entropy(scores, labels)


# ---------- bench driver -------------------------------------------------


def bench(path_name: str, step_fn, batches, device: torch.device, warmup: int):
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    for i, batch in enumerate(batches):
        with torch.enable_grad():
            Q = [x.clone().requires_grad_(True) for x in batch[0]]
            P = [x.clone().requires_grad_(True) for x in batch[1]]
            N = [x.clone().requires_grad_(True) for x in batch[2]]
            loss = step_fn(Q, P, N)
            loss.backward()
        if i + 1 == warmup and device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
        elif i + 1 == warmup:
            t0 = time.perf_counter()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    n_timed = len(batches) - warmup
    per_step_ms = 1000.0 * (t1 - t0) / max(n_timed, 1)
    peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else float("nan")
    print(f"  {path_name:8s}  loss={loss.item():7.4f}  {per_step_ms:7.2f} ms/step  peak={peak_mb:7.1f} MB")
    return per_step_ms, peak_mb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--q-short", type=int, default=16)
    p.add_argument("--q-long", type=int, default=64)
    p.add_argument("--d-short", type=int, default=64)
    p.add_argument("--d-long", type=int, default=2048)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA device required for late-interaction-kernels kernels.")

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    print(
        f"device={device}  dtype={dtype}  B={args.batch_size}  d={args.d}  "
        f"q_len in [{args.q_short}, {args.q_long}]  d_len in [{args.d_short}, {args.d_long}]"
    )

    batches = [
        sample_batch(
            args.batch_size,
            args.d,
            args.q_short,
            args.q_long,
            args.d_short,
            args.d_long,
            device,
            dtype,
            gen,
        )
        for _ in range(args.steps)
    ]

    def total_real_tokens(seqs):
        return sum(s.shape[0] for s in seqs)

    first = batches[0]
    real = sum(total_real_tokens(x) for x in first)
    padded = sum(len(x) * max(s.shape[0] for s in x) for x in first)
    waste = 1.0 - real / padded
    print(f"batch-0 padding waste: {100 * waste:5.1f} %  (real={real}, padded={padded} tokens)")

    print()
    ms_pad, mem_pad = bench("padded", padded_step, batches, device, args.warmup)
    ms_pak, mem_pak = bench("packed", packed_step, batches, device, args.warmup)

    if ms_pad > 0:
        print(f"\n  speedup   packed / padded = {ms_pad / ms_pak:.2f}x")
    if mem_pad == mem_pad and mem_pak == mem_pak and mem_pad > 0:
        print(f"  memory    packed / padded = {mem_pak / mem_pad:.2f}x  ({mem_pad - mem_pak:+.1f} MB saved)")


if __name__ == "__main__":
    main()
