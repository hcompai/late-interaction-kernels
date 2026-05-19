"""MPS dispatch for the high-level :func:`retrieve` and :class:`MaxSimScorer`.

Apple Silicon GPUs run PyTorch through the MPS backend, which has no
Triton support. The kernels in :mod:`late_interaction_kernels.forward`
and friends therefore can't run on ``mps:0`` tensors.

Two MaxSim implementations cover the gap:

* :mod:`.compile_dispatch` (this module) ships a ``torch.compile``-fused reference.
  Inductor lowers the einsum + max + sum chain to a single MPSGraph,
  typically ≈2× faster than eager. The compile path is autograd-aware,
  so it carries every training-time call.
* :mod:`.metal` ships a fused forward kernel built on Apple's
  ``simdgroup_matrix`` MMA. It saves the ``[Lq, Ld]`` similarity tensor
  and beats the compiled path by ≈1.2-1.6× on inference shapes with
  realistic doc batches.

For inference we route to the Metal kernel when its assumptions hold
(fp16 / bf16 inputs, ``d`` ≤ 128 and divisible by 8) and the workload
is large enough that the kernel's launch overhead amortises. The
heuristic is shape-only — measured on M-series silicon — and falls
back to the compile path for everything else, including all training
calls. ``LIK_DISABLE_COMPILE=1`` and ``LIK_FORCE_MPS_BACKEND={metal,
compile,reference}`` give explicit overrides.
"""

import os
import threading
from collections.abc import Callable

import torch

from late_interaction_kernels.mps import metal as _metal
from late_interaction_kernels.reference import maxsim_reference

_compile_lock = threading.Lock()
_compiled_cache: dict[tuple, Callable] = {}

# Crossover thresholds measured on M-series silicon: below these, the
# compile path's lower launch overhead beats the Metal kernel's bigger
# per-threadgroup work. Tweak via ``LIK_MPS_METAL_MIN_*`` env vars if
# you're benchmarking on different hardware.
_DEFAULT_MIN_BATCH = 64
_DEFAULT_MIN_LD = 192


def _disable_compile() -> bool:
    """Honour ``LIK_DISABLE_COMPILE=1`` so users can opt out."""
    return os.environ.get("LIK_DISABLE_COMPILE", "0") == "1"


def _forced_backend() -> str | None:
    """Return ``"metal"`` / ``"compile"`` / ``"reference"`` if forced, else ``None``."""
    value = os.environ.get("LIK_FORCE_MPS_BACKEND")
    if value is None:
        return None
    value = value.strip().lower()
    if value in {"metal", "compile", "reference"}:
        return value
    return None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _compile_key(
    Q: torch.Tensor,
    normalize: bool,
    q_mask: torch.Tensor | None,
    d_mask: torch.Tensor | None,
) -> tuple:
    return (Q.dtype, bool(normalize), q_mask is not None, d_mask is not None)


def _get_compiled(key: tuple) -> Callable:
    fn = _compiled_cache.get(key)
    if fn is not None:
        return fn
    with _compile_lock:
        fn = _compiled_cache.get(key)
        if fn is None:
            # ``dynamic=False`` (was ``dynamic=True``): MPS inductor on torch 2.8
            # cannot lower the ``S.max(dim=-1)`` reduction when ``Ld`` is symbolic
            # (`cannot determine truth value of Relational: s12 <= 1024` from
            # ``codegen_iteration_ranges_entry``). With static shapes, PyTorch's
            # dynamo cache transparently recompiles per (Nq, Nd, Lq, Ld) tuple
            # up to ``torch._dynamo.config.cache_size_limit`` (default 8) then
            # gracefully falls back to eager — fine for typical inference where
            # shapes are stable, and for shape-varying workloads users should
            # reach for the Metal kernel anyway.
            fn = torch.compile(
                maxsim_reference,
                mode="reduce-overhead",
                dynamic=False,
                fullgraph=False,
            )
            _compiled_cache[key] = fn
    return fn


def is_mps_tensor(x: torch.Tensor) -> bool:
    return x.device.type == "mps"


def _metal_is_worthwhile(Q: torch.Tensor, D: torch.Tensor) -> bool:
    """Heuristic: only launch the Metal kernel when the work amortises.

    Below ``Nq * Nd ≥ MIN_BATCH`` and ``Ld ≥ MIN_LD`` the compile path
    has lower launch overhead and tends to win.
    """
    if Q.dim() == 2:
        Nq, Lq = 1, Q.shape[0]
    else:
        Nq, Lq = Q.shape[0], Q.shape[1]
    if D.dim() == 2:
        Nd, Ld = 1, D.shape[0]
    else:
        Nd, Ld = D.shape[0], D.shape[1]
    del Lq
    min_batch = _env_int("LIK_MPS_METAL_MIN_BATCH", _DEFAULT_MIN_BATCH)
    min_ld = _env_int("LIK_MPS_METAL_MIN_LD", _DEFAULT_MIN_LD)
    return Nq * Nd >= min_batch and Ld >= min_ld


def _compile_path(
    Q: torch.Tensor,
    D: torch.Tensor,
    q_mask: torch.Tensor | None,
    d_mask: torch.Tensor | None,
    normalize: bool,
) -> torch.Tensor:
    if _disable_compile():
        return maxsim_reference(Q, D, q_mask=q_mask, d_mask=d_mask, normalize=normalize)
    fn = _get_compiled(_compile_key(Q, normalize, q_mask, d_mask))
    return fn(Q, D, q_mask=q_mask, d_mask=d_mask, normalize=normalize)


def maxsim_mps(
    Q: torch.Tensor,
    D: torch.Tensor,
    q_mask: torch.Tensor | None = None,
    d_mask: torch.Tensor | None = None,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """``torch.compile``-fused MaxSim on MPS. Autograd-aware.

    Always uses the compile path; the Metal kernel is forward-only so
    it can't carry gradients.
    """
    forced = _forced_backend()
    if forced == "reference":
        return maxsim_reference(Q, D, q_mask=q_mask, d_mask=d_mask, normalize=normalize)
    return _compile_path(Q, D, q_mask, d_mask, normalize)


def maxsim_inference_mps(
    Q: torch.Tensor,
    D: torch.Tensor,
    q_mask: torch.Tensor | None = None,
    d_mask: torch.Tensor | None = None,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Inference-only MaxSim on MPS, picking the faster of Metal / compile.

    Routes to :func:`late_interaction_kernels.metal.maxsim_inference_metal`
    when the dtype, embedding dim, and batch size suit the Metal path;
    falls back to the compile path otherwise.
    """
    with torch.no_grad():
        forced = _forced_backend()
        if forced == "reference":
            return maxsim_reference(Q, D, q_mask=q_mask, d_mask=d_mask, normalize=normalize)
        if forced == "compile":
            return _compile_path(Q, D, q_mask, d_mask, normalize)

        use_metal = (
            _metal.is_available()
            and _metal.supports(Q, D)
            and (forced == "metal" or _metal_is_worthwhile(Q, D))
        )
        if use_metal:
            try:
                return _metal.maxsim_inference_metal(Q, D, q_mask=q_mask, d_mask=d_mask, normalize=normalize)
            except RuntimeError:
                # Compile-time MSL errors or device-side faults: fall back
                # transparently rather than punish the caller.
                pass
        return _compile_path(Q, D, q_mask, d_mask, normalize)


__all__ = ["is_mps_tensor", "maxsim_mps", "maxsim_inference_mps"]
