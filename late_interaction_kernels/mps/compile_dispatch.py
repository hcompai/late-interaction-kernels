"""MPS dispatch for the high-level :func:`retrieve` and :class:`MaxSimScorer`.

Apple Silicon GPUs run PyTorch through the MPS backend, which has no
Triton support. The kernels in :mod:`late_interaction_kernels.forward`
and friends therefore can't run on ``mps:0`` tensors.

Two MaxSim implementations cover the gap:

* :mod:`.compile_dispatch` (this module) ships a ``torch.compile``-fused reference.
  Inductor lowers the einsum + max + sum chain to a single MPSGraph,
  typically ≈2× faster than eager. The compile path is autograd-aware
  and accepts any dtype, so it covers everything the Metal kernel
  can't.
* :mod:`.metal` ships fused forward + backward kernels built on Apple's
  ``simdgroup_matrix`` MMA. The forward never materialises the
  ``[Lq, Ld]`` similarity tile; the backward is single-pass
  argmax-conditioned (``grad_Q`` row-owned, ``grad_D`` scattered via
  ``atomic_uint`` CAS) with the L2-normalize Jacobian folded in.

We route to the Metal kernel when its assumptions hold (fp16 / bf16
inputs, ``d`` ≤ 128 and divisible by 8) and the workload is large
enough that the launch overhead amortises — both for inference and
for autograd-tracking training calls (via ``_MaxSimFnMetal``). The
heuristic is shape-only, measured on M-series silicon, and falls back
to the compile path for everything else (fp32 inputs, tiny batches,
``d > 128``, etc.). ``LIK_DISABLE_COMPILE=1`` and
``LIK_FORCE_MPS_BACKEND={metal,compile,reference}`` give explicit
overrides.
"""

import os
import threading
from collections.abc import Callable

import torch

from late_interaction_kernels.mps import metal as _metal
from late_interaction_kernels.reference import maxsim_reference

_compile_lock = threading.Lock()
_compiled_cache: dict[tuple, Callable] = {}
_compiled_kd_cache: dict[tuple, Callable] = {}

# Crossover thresholds measured on M-series silicon: below these, the
# compile path's lower launch overhead beats the Metal kernel's bigger
# per-threadgroup work. Tweak via ``LIK_MPS_METAL_MIN_*`` env vars if
# you're benchmarking on different hardware.
_DEFAULT_MIN_BATCH = 64
_DEFAULT_MIN_LD = 192

# KD / pairs: the compile fallback materialises a dense [Nq, K, Lq, Ld]
# einsum, so it scales worse than the cross-product compile path and
# the Metal kernel wins from a much smaller workload. Looser thresholds.
_DEFAULT_KD_MIN_BATCH = 8
_DEFAULT_KD_MIN_LD = 64


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


def _get_compiled_kd(key: tuple) -> Callable:
    """``_get_compiled`` for the 4-D KD reference — same compile flags."""
    fn = _compiled_kd_cache.get(key)
    if fn is not None:
        return fn
    with _compile_lock:
        fn = _compiled_kd_cache.get(key)
        if fn is None:
            fn = torch.compile(_kd_reference, mode="reduce-overhead", dynamic=False, fullgraph=False)
            _compiled_kd_cache[key] = fn
    return fn


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


def _metal_is_worthwhile(Q: torch.Tensor, D: torch.Tensor) -> bool:
    """Heuristic: only launch the Metal kernel when the work amortises.

    Cross-product: ``Nq * Nd ≥ MIN_BATCH`` and ``Ld ≥ MIN_LD``.
    KD / pairs (4-D ``D``): a separate, looser pair of thresholds. The
    compile path on KD goes through the dense ``[Nq, K, Lq, Ld]``
    einsum materialise, which scales much worse than the cross-product
    compile path — so the Metal kernel wins from a much smaller
    workload. Crossover measured on M4 fp16; matches the marquee
    ``kd-rerank-top10`` shape (Nq=1, K=10, Ld=300).
    """
    if Q.dim() == 2:
        Nq = 1
    else:
        Nq = Q.shape[0]
    kd_layout = D.dim() == 4
    if kd_layout:
        _, K, Ld, _ = D.shape
        Nd = K
    elif D.dim() == 2:
        Nd, Ld = 1, D.shape[0]
    else:
        Nd, Ld = D.shape[0], D.shape[1]
    if kd_layout:
        min_batch = _env_int("LIK_MPS_METAL_KD_MIN_BATCH", _DEFAULT_KD_MIN_BATCH)
        min_ld = _env_int("LIK_MPS_METAL_KD_MIN_LD", _DEFAULT_KD_MIN_LD)
    else:
        min_batch = _env_int("LIK_MPS_METAL_MIN_BATCH", _DEFAULT_MIN_BATCH)
        min_ld = _env_int("LIK_MPS_METAL_MIN_LD", _DEFAULT_MIN_LD)
    return Nq * Nd >= min_batch and Ld >= min_ld


def _kd_reference(
    Q: torch.Tensor,
    D: torch.Tensor,
    q_mask: torch.Tensor | None = None,
    d_mask: torch.Tensor | None = None,
    *,
    normalize: bool = False,
) -> torch.Tensor:
    """Reference KD MaxSim: ``Q[Nq, Lq, d] × D[Nq, K, Ld, d] -> [Nq, K]``.

    fp32 accumulator with ``-inf`` clamp on fully-masked rows, same
    contract as :func:`maxsim_reference` for the cross-product layout.
    """
    import torch.nn.functional as F

    from late_interaction_kernels.reference import NEG_INF

    Qf = Q.float()
    Df = D.float()
    if normalize:
        Qf = F.normalize(Qf, p=2, dim=-1, eps=1e-12)
        Df = F.normalize(Df, p=2, dim=-1, eps=1e-12)
    S = torch.einsum("ild,iktd->iklt", Qf, Df)  # [Nq, K, Lq, Ld]
    if d_mask is not None:
        S = S.masked_fill(~d_mask.bool().unsqueeze(2), NEG_INF)
    row_max = S.max(dim=-1).values
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    if q_mask is not None:
        row_max = row_max * q_mask.to(row_max.dtype).unsqueeze(1)
    return row_max.sum(dim=-1)


def _compile_path(
    Q: torch.Tensor,
    D: torch.Tensor,
    q_mask: torch.Tensor | None,
    d_mask: torch.Tensor | None,
    normalize: bool,
) -> torch.Tensor:
    # KD (4-D D) wraps ``_kd_reference`` via ``torch.compile`` — same
    # policy as the cross-product branch, opt-out via ``LIK_DISABLE_COMPILE``.
    if D.dim() == 4:
        if _disable_compile():
            return _kd_reference(Q, D, q_mask=q_mask, d_mask=d_mask, normalize=normalize)
        fn = _get_compiled_kd(_compile_key(Q, normalize, q_mask, d_mask))
        return fn(Q, D, q_mask=q_mask, d_mask=d_mask, normalize=normalize)
    if _disable_compile():
        return maxsim_reference(Q, D, q_mask=q_mask, d_mask=d_mask, normalize=normalize)
    fn = _get_compiled(_compile_key(Q, normalize, q_mask, d_mask))
    return fn(Q, D, q_mask=q_mask, d_mask=d_mask, normalize=normalize)


class _MaxSimFnMetal(torch.autograd.Function):
    """Fused MaxSim forward + Metal backward (mirrors Triton's ``_MaxSimFn``).

    Both halves run on the GPU. When ``normalize=True`` the backward
    kernel folds the L2-normalize Jacobian into its writes (the
    host-side variant was ~10x more expensive than the kernel proper).
    """

    @staticmethod
    def forward(ctx, Q, D, q_mask, d_mask, normalize, kd_layout):
        scores, argmax, fwd_ctx = _metal.maxsim_train_metal(
            Q, D, q_mask=q_mask, d_mask=d_mask, normalize=normalize
        )
        ctx.save_for_backward(fwd_ctx.Q, fwd_ctx.D, argmax, fwd_ctx.q_mask_i8)
        ctx.normalize = normalize
        ctx.kd_layout = kd_layout
        ctx.d_input_shape = D.shape
        return scores

    @staticmethod
    def backward(ctx, grad_scores):
        Q, D, argmax, q_mask_i8 = ctx.saved_tensors
        grad_Q, grad_D = _metal.maxsim_backward_metal(
            grad_scores,
            Q,
            D,
            argmax,
            q_mask=q_mask_i8,
            kd_layout=ctx.kd_layout,
            normalize=ctx.normalize,
        )
        if ctx.kd_layout and grad_D.shape != ctx.d_input_shape:
            grad_D = grad_D.view(ctx.d_input_shape)
        return grad_Q, grad_D, None, None, None, None


def _metal_train_supported(Q: torch.Tensor, D: torch.Tensor) -> bool:
    """Metal forward+backward applies to fp16/bf16, ``d`` ≤ 128, 3-D Q,
    3-D or 4-D D, and large-enough workloads (same threshold as inference)."""
    if not _metal.is_available() or not _metal.supports(Q, D):
        return False
    if Q.dim() != 3 or D.dim() not in (3, 4):
        return False
    return _metal_is_worthwhile(Q, D)


def maxsim_mps(
    Q: torch.Tensor,
    D: torch.Tensor,
    q_mask: torch.Tensor | None = None,
    d_mask: torch.Tensor | None = None,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Autograd-aware MaxSim on MPS.

    When ``Q``/``D`` need grads and the shape/dtype suits the Metal
    kernel, routes through the fused Metal forward + Metal backward
    via :class:`_MaxSimFnMetal`. Otherwise (no grads needed, unsupported
    dtype, or below the size threshold) falls back to the
    ``torch.compile`` path, which is itself autograd-aware.
    """
    forced = _forced_backend()
    if forced == "reference":
        if D.dim() == 4:
            return _kd_reference(Q, D, q_mask=q_mask, d_mask=d_mask, normalize=normalize)
        return maxsim_reference(Q, D, q_mask=q_mask, d_mask=d_mask, normalize=normalize)
    if forced == "compile":
        return _compile_path(Q, D, q_mask, d_mask, normalize)

    needs_grad = Q.requires_grad or D.requires_grad
    if needs_grad and (forced == "metal" or _metal_train_supported(Q, D)):
        kd_layout = D.dim() == 4
        return _MaxSimFnMetal.apply(Q, D, q_mask, d_mask, normalize, kd_layout)
    # `forced == "metal"` with no grad: the autograd path above doesn't
    # apply, so route to the inference kernel directly rather than
    # silently falling through to compile.
    if forced == "metal" and _metal.is_available() and _metal.supports(Q, D):
        with torch.no_grad():
            return _metal.maxsim_inference_metal(Q, D, q_mask=q_mask, d_mask=d_mask, normalize=normalize)
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
            if D.dim() == 4:
                return _kd_reference(Q, D, q_mask=q_mask, d_mask=d_mask, normalize=normalize)
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


__all__ = ["maxsim_mps", "maxsim_inference_mps"]
