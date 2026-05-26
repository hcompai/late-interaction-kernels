"""Measures the three autotune-cost reductions added in this PR:

1. **Persistent disk cache** (Triton 3.4+ ``cache_results=True``):
   first run pays the usual sweep, second python process loads the JSON
   and skips the bench. The script drives this by clearing the Triton
   cache, timing a cold ``maxsim`` call, then re-running in a second
   subprocess against the warm cache.

2. **Small-input bypass**: forward shapes with ``Nq*Nd ≤ 500`` and
   ``d ≤ 256`` route through ``_maxsim_fwd_kernel.fn[grid](...)``
   directly, with fixed ``(BLOCK_Q=32, BLOCK_D=64, num_warps=4,
   num_stages=2)``. The autotune cache stays empty and the first call is
   sub-millisecond. The KD shapes from ``lightonai/pylate#224`` §2 all
   fall in this regime (Nq=1, Nd=K=8-32, d=128).

3. **``normalize`` out of the autotune key**: toggling normalize doesn't
   spawn a second cache entry.

Reproducer for the perf claims in the PR description. Run via
``scripts/sky_bench_autotune_persistence.yaml`` on an H100 with NGC 25.06
(PyTorch 2.8 + Triton 3.4+).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

os.environ.setdefault("LIK_SUPPRESS_NORM_WARN", "1")


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _triton_cache_dir() -> Path:
    """Where Triton stores compiled binaries and (3.4+) autotune JSONs."""
    return Path(os.environ.get("TRITON_CACHE_DIR", Path.home() / ".triton" / "cache"))


def _clear_triton_cache() -> None:
    """Wipe the entire Triton cache so the next call is genuinely cold.

    Also clears the in-process autotune dict so a same-process second call
    doesn't short-circuit.
    """
    cache_dir = _triton_cache_dir()
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)


def _time_first_call(fn) -> float:
    """Time a single call, sync at end. No warmup — that's the whole point."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def _time_median(fn, *, warmup: int = 5, iters: int = 30) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    return sorted(samples)[len(samples) // 2]


def _autotune_supports_disk_cache() -> bool:
    import inspect

    import triton

    return "cache_results" in inspect.signature(triton.autotune).parameters


# --------------------------------------------------------------------------- #
# Scenarios                                                                    #
# --------------------------------------------------------------------------- #


def scenario_cold_vs_warm_in_process() -> None:
    """One process. Cold first call eats autotune; second call is cache hit."""
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.forward import _maxsim_fwd_kernel

    print("=== Scenario A: in-process cold vs warm ===")
    print("(autotune cost = first call; cache hit = second call)")
    _maxsim_fwd_kernel.cache.clear()

    # Big enough to hit the autotune path (>500 Nq*Nd).
    Nq, Nd, Lq, Ld, d = 64, 64, 32, 180, 128
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float16)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float16)

    cold = _time_first_call(lambda: maxsim(Q, D))
    warm = _time_median(lambda: maxsim(Q, D))
    print(f"  cold:           {cold * 1e3:7.1f} ms  (includes autotune sweep + JIT)")
    print(f"  warm (median):  {warm * 1e3:7.3f} ms  (in-memory cache hit)")
    print(f"  cache entries:  {len(_maxsim_fwd_kernel.cache)}")
    print()


def scenario_disk_cache_across_processes() -> None:
    """Wipe the Triton cache. First subprocess pays autotune. Second
    subprocess (cache_results=True) should load the JSON and run cold-fast.
    """
    print("=== Scenario B: persistent disk cache across processes ===")
    if not _autotune_supports_disk_cache():
        print("  Triton lacks ``cache_results`` (need 3.4+). Skipping.")
        print()
        return

    cache_dir = _triton_cache_dir()
    print(f"  Triton cache dir: {cache_dir}")

    code = (
        "import os, time, torch; os.environ['LIK_SUPPRESS_NORM_WARN']='1';\n"
        "from late_interaction_kernels import maxsim;\n"
        "Q = torch.randn(64,32,128, device='cuda', dtype=torch.float16);\n"
        "D = torch.randn(64,180,128, device='cuda', dtype=torch.float16);\n"
        "torch.cuda.synchronize(); t0=time.perf_counter();\n"
        "maxsim(Q, D); torch.cuda.synchronize();\n"
        "print(f'__FIRST_CALL_MS={(time.perf_counter()-t0)*1e3:.1f}')\n"
    )

    # Run 1: cache wiped → must pay sweep.
    _clear_triton_cache()
    out1 = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    line1 = [_ for _ in out1.stdout.splitlines() if _.startswith("__FIRST_CALL_MS=")][0]
    cold_first = float(line1.split("=")[1])

    # Run 2: cache intact → should hit disk cache, skip sweep.
    out2 = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    line2 = [_ for _ in out2.stdout.splitlines() if _.startswith("__FIRST_CALL_MS=")][0]
    warm_first = float(line2.split("=")[1])

    speedup = cold_first / warm_first
    print(f"  process 1 (cache wiped): {cold_first:7.1f} ms first call")
    print(f"  process 2 (cache hit):   {warm_first:7.1f} ms first call")
    print(f"  speedup:                 {speedup:.1f}×")
    if speedup < 2.0:
        print("  WARNING: persistent cache didn't kick in (expected ≥2× speedup).")
    print()


def scenario_small_input_bypass() -> None:
    """Issue #224 §2 shapes — exactly the KD shapes flash-maxsim showed
    catastrophic vs LIK with the for-loop. With the small-input bypass
    these are sub-millisecond and never touch autotune.
    """
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.forward import _maxsim_fwd_kernel

    print("=== Scenario C: small-input bypass (issue #224-style shapes) ===")
    print("(Nq*Nd ≤ 500, d ≤ 256 → no autotune, no JIT bench)")

    shapes = [
        # (Nq, Nd, Lq, Ld, d) — small inference-like
        (1, 100, 32, 180, 128),
        (1, 200, 32, 180, 128),
        (1, 500, 32, 180, 128),
        (2, 100, 32, 180, 128),
        (4, 100, 128, 512, 128),
    ]
    _maxsim_fwd_kernel.cache.clear()
    for Nq, Nd, Lq, Ld, d in shapes:
        Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float16)
        D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float16)
        cold = _time_first_call(lambda Q=Q, D=D: maxsim(Q, D))
        warm = _time_median(lambda Q=Q, D=D: maxsim(Q, D))
        cache_size = len(_maxsim_fwd_kernel.cache)
        bypassed = "✓" if cache_size == 0 else "✗"
        print(
            f"  Nq={Nq:2} Nd={Nd:3} Lq={Lq:3} Ld={Ld:3} d={d}  "
            f"cold={cold * 1e3:6.1f} ms  warm={warm * 1e3:6.3f} ms  bypass={bypassed}"
        )
    print()


def scenario_normalize_collapses_to_one_entry() -> None:
    """``normalize=True/False`` must share a single autotune entry."""
    from late_interaction_kernels import maxsim
    from late_interaction_kernels.forward import _maxsim_fwd_kernel

    print("=== Scenario D: normalize is no longer in the autotune key ===")
    _maxsim_fwd_kernel.cache.clear()
    Nq, Nd, Lq, Ld, d = 64, 64, 32, 180, 128
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float16)
    D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float16)

    _ = maxsim(Q, D, normalize=False)
    n_after_first = len(_maxsim_fwd_kernel.cache)
    _ = maxsim(Q, D, normalize=True)
    n_after_second = len(_maxsim_fwd_kernel.cache)

    print(f"  cache entries after normalize=False: {n_after_first}")
    print(f"  cache entries after normalize=True:  {n_after_second}")
    if n_after_second == n_after_first:
        print("  ✓ shared entry — autotune key shrunk correctly")
    else:
        print("  ✗ second normalize value spawned a new entry — key didn't shrink!")
    print()


def scenario_inference_like_shapes() -> None:
    """Issue lightonai/pylate#224 §1: ``colbert_scores`` in-batch inference
    shapes (single query × many docs). These hit the autotune path before
    the small-input bypass (Nq*Nd > 500) so they're where ``cache_results``
    pays off — first run on a fresh machine, all subsequent runs free.

    We just time them once warm (in-memory cache) to confirm we're in the
    expected ballpark vs eager einsum.
    """
    from late_interaction_kernels import maxsim

    print("=== Scenario E: in-batch inference shapes (warm) ===")

    def _einsum(Q, D):
        return torch.einsum("nqd,mkd->nmqk", Q, D).max(axis=-1).values.sum(axis=-1)

    shapes = [
        # (label, Nq, Nd, Lq, Ld, d) — inference: 1 query × many docs.
        ("Nq=1   Nd=1000 Lq=32  Ld=180", 1, 1000, 32, 180, 128),
        ("Nq=1   Nd=5000 Lq=32  Ld=180", 1, 5000, 32, 180, 128),
        ("Nq=1   Nd=1000 Lq=128 Ld=512", 1, 1000, 128, 512, 128),
        # training-shaped: above the bypass threshold
        ("Nq=64  Nd=64   Lq=32  Ld=180", 64, 64, 32, 180, 128),
        ("Nq=128 Nd=128  Lq=32  Ld=180", 128, 128, 32, 180, 128),
    ]
    print("shape                              base(fp16)  LIK         ratio   parity")
    print("-" * 75)
    for label, Nq, Nd, Lq, Ld, d in shapes:
        torch.manual_seed(0)
        Q = torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float16)
        D = torch.randn(Nd, Ld, d, device="cuda", dtype=torch.float16)

        t_base = _time_median(lambda Q=Q, D=D: _einsum(Q, D))
        t_lik = _time_median(lambda Q=Q, D=D: maxsim(Q, D))

        ref = _einsum(Q.float(), D.float())
        fast = maxsim(Q, D).float()
        err = (fast - ref).abs().max().item()
        ok = err < 5e-2

        verdict = "✓" if ok else f"✗ (err={err:.2e})"
        print(
            f"{label:35}  {t_base * 1e3:7.3f} ms   {t_lik * 1e3:7.3f} ms   "
            f"{t_base / t_lik:5.2f}×  {verdict}"
        )
    print()


# --------------------------------------------------------------------------- #
# Entrypoint                                                                   #
# --------------------------------------------------------------------------- #


def main() -> None:
    assert torch.cuda.is_available(), "needs CUDA"
    print(f"# device          : {torch.cuda.get_device_name(0)}")
    print(f"# torch           : {torch.__version__}")
    import triton

    print(f"# triton          : {triton.__version__}")
    print(
        f"# cache_results   : {'supported' if _autotune_supports_disk_cache() else 'NOT SUPPORTED (pre-3.4)'}"
    )
    print()

    scenario_cold_vs_warm_in_process()
    scenario_disk_cache_across_processes()
    scenario_small_input_bypass()
    scenario_normalize_collapses_to_one_entry()
    scenario_inference_like_shapes()


if __name__ == "__main__":
    main()
