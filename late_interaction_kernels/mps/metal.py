"""Fused MaxSim forward + backward on MPS via Metal ``simdgroup_matrix`` MMA.

Direct Metal-side analogue of :mod:`late_interaction_kernels.forward`
and :func:`maxsim_backward_unified`: the same FlashAttention-style
outer-product tiling on the forward, the same argmax-conditioned
single-pass backward (``grad_Q`` row-owned, ``grad_D`` scattered via
``atomic_uint`` CAS). The inner GEMM runs on Apple's
``simdgroup_matrix<T, 8, 8>`` MMA instead of CUDA tensor cores; the
``[Nq · Nd · Lq · Ld]`` similarity tensor never reaches HBM.

Each threadgroup serves ``J_PER_TG`` consecutive ``j`` values, loading
Q exactly once and hoisting it into a register-resident
``simdgroup_matrix`` cache that survives every ``(j, d-chunk)`` pair —
this is the single largest win versus the naive one-(i,j)-per-thread-
group layout. Two persistence levels are dispatched by ``Nd``:

* ``J_PER_TG = 1`` for tiny corpora where the launch grid needs to be
  wide to keep the GPU saturated;
* ``J_PER_TG = 8`` for the typical inference / training regime,
  amortising the Q load and the register-cache build 8× per
  threadgroup.

D streams in 32-row tiles through threadgroup memory with the optional
L2-normalize folded into the cooperative load. The training forward
additionally writes the per-(i, s, j) argmax; the matching backward
optionally folds the L2-normalize Jacobian into its writes.

Constraints (everything outside falls back to ``torch.compile``):

* ``d`` is a multiple of 8 and ≤ ``D_MAX = 128`` (covers ColBERT,
  ColPali, LateOn, LateOn-Code, edge-d48 / d64);
* fp16 / bf16 inputs only — Apple's MMA accepts those; fp32 inputs
  go through the compile path.

The MSL source lives in :file:`_maxsim.metal` and is loaded via
``str.format(...)`` at JIT-compile time, with host-side constants
(tile sizes, flag bits, struct format) substituted in so host and
device can't drift.
"""

import struct
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, NamedTuple

import torch

# Tile sizes. One simdgroup per threadgroup so each of the 32 lanes
# owns exactly one query / doc row during the cooperative load.
_BLOCK_Q = 32
_BLOCK_D = 32
_BLOCK_K = 8  # Apple ``simdgroup_matrix`` is 8 × 8
_M_TILES = _BLOCK_Q // _BLOCK_K
_N_TILES = _BLOCK_D // _BLOCK_K

# ``D_MAX`` caps the threadgroup-memory footprint AND the depth of the
# register-resident Q cache. 128 covers every standard late-interaction
# model dim (48 / 64 / 96 / 128) and keeps ``K_TILES_MAX = 16`` so the
# Q cache is ``4 × 16 = 64`` simdgroup matrices ≈ 8 KiB of register
# state per simdgroup — comfortably inside the M-series register file.
_D_MAX = 128
_K_TILES_MAX = _D_MAX // _BLOCK_K
_THREADS_PER_GROUP = 32

# Persistence levels dispatched by ``Nd``.
_J_PER_TG_BIG = 8
_J_THRESHOLD = 256  # Nd above which the persistent variant wins

_FLAG_HAS_Q_MASK = 1 << 0
_FLAG_HAS_D_MASK = 1 << 1
_FLAG_NORMALIZE = 1 << 2
_FLAG_KD_LAYOUT = 1 << 3

_PARAMS_FORMAT = "6I"  # (Nq, Nd, Lq, Ld, d, flags), all u32


_MSL_TEMPLATE_PATH = Path(__file__).with_name("_maxsim.metal")


def _build_kernel_source() -> str:
    """Read the MSL template and substitute the Python-side constants.

    Literal MSL braces in the template are doubled (``{{`` / ``}}``);
    substitutions like ``{BLOCK_Q}`` are single.
    """
    template = _MSL_TEMPLATE_PATH.read_text()
    return template.format(
        BLOCK_Q=_BLOCK_Q,
        BLOCK_D=_BLOCK_D,
        BLOCK_K=_BLOCK_K,
        M_TILES=_M_TILES,
        N_TILES=_N_TILES,
        D_MAX=_D_MAX,
        K_TILES_MAX=_K_TILES_MAX,
        THREADS_PER_GROUP=_THREADS_PER_GROUP,
        J_PER_TG_BIG=_J_PER_TG_BIG,
        FLAG_HAS_Q_MASK=_FLAG_HAS_Q_MASK,
        FLAG_HAS_D_MASK=_FLAG_HAS_D_MASK,
        FLAG_NORMALIZE=_FLAG_NORMALIZE,
        FLAG_KD_LAYOUT=_FLAG_KD_LAYOUT,
    )


_lib_lock = threading.Lock()
_lib: Any | None = None
_lib_compile_error: BaseException | None = None


def _get_lib() -> Any:
    """JIT-compile the Metal source on first call; cache thereafter."""
    global _lib, _lib_compile_error
    if _lib is not None:
        return _lib
    if _lib_compile_error is not None:
        raise _lib_compile_error
    with _lib_lock:
        if _lib is None and _lib_compile_error is None:
            try:
                _lib = torch.mps.compile_shader(_build_kernel_source())
            except BaseException as exc:  # noqa: BLE001
                _lib_compile_error = exc
                raise
    if _lib is None:
        assert _lib_compile_error is not None
        raise _lib_compile_error
    return _lib


_DTYPE_TO_KERNEL = {
    (torch.float16, 1): "maxsim_half_small",
    (torch.float16, _J_PER_TG_BIG): "maxsim_half_big",
    (torch.bfloat16, 1): "maxsim_bfloat_small",
    (torch.bfloat16, _J_PER_TG_BIG): "maxsim_bfloat_big",
}

_DTYPE_TO_TRAIN_KERNEL = {
    (torch.float16, 1): "maxsim_half_small_train",
    (torch.float16, _J_PER_TG_BIG): "maxsim_half_big_train",
    (torch.bfloat16, 1): "maxsim_bfloat_small_train",
    (torch.bfloat16, _J_PER_TG_BIG): "maxsim_bfloat_big_train",
}

_DTYPE_TO_BWD_KERNEL = {
    (torch.float16, False): "maxsim_bwd_half",
    (torch.float16, True): "maxsim_bwd_half_norm",
    (torch.bfloat16, False): "maxsim_bwd_bfloat",
    (torch.bfloat16, True): "maxsim_bwd_bfloat_norm",
}


def _pick_j_per_tg(Nd: int) -> int:
    """Persist 8 j-values when Nd is large enough that the smaller
    launch grid still saturates the GPU."""
    return _J_PER_TG_BIG if Nd >= _J_THRESHOLD else 1


def is_available() -> bool:
    """Whether the Metal MaxSim path can run on this machine.

    Requires an MPS-capable PyTorch build with ``torch.mps.compile_shader``
    (PyTorch ≥ 2.10).
    """
    return (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and hasattr(torch.mps, "compile_shader")
    )


def supports(Q: torch.Tensor, D: torch.Tensor) -> bool:
    """Whether the Metal kernel handles this dtype + shape combination.

    Falls back when ``d`` is not a multiple of 8 or exceeds ``D_MAX``,
    and when the dtype is not fp16 / bf16. ``D`` may be 3-D
    (cross-product, default) or 4-D ``[Nq, K, Ld, d]`` (KD / pairs); in
    the 4-D case ``Q`` must be 3-D and ``Q.shape[0] == D.shape[0]``.
    """
    if Q.dtype != D.dtype:
        return False
    if Q.dtype not in (torch.float16, torch.bfloat16):
        return False
    if Q.dim() not in (2, 3) or D.dim() not in (2, 3, 4):
        return False
    if Q.shape[-1] != D.shape[-1]:
        return False
    if D.dim() == 4:
        # KD / pairs layout: Q must be [Nq, Lq, d] and D must be [Nq, K, Ld, d].
        if Q.dim() != 3 or Q.shape[0] != D.shape[0]:
            return False
    d = Q.shape[-1]
    return 0 < d <= _D_MAX and d % _BLOCK_K == 0


_PARAMS_CACHE_MAX = 64
_params_cache: "OrderedDict[tuple[int, int, int, int, int, int], torch.Tensor]" = OrderedDict()
_params_cache_lock = threading.Lock()


def _pack_params(Nq: int, Nd: int, Lq: int, Ld: int, d: int, flags: int) -> torch.Tensor:
    """Pack ``MaxSimParams`` into a 6-int32 MPS tensor; LRU-cached by key."""
    key = (Nq, Nd, Lq, Ld, d, flags)
    with _params_cache_lock:
        cached = _params_cache.get(key)
        if cached is not None:
            _params_cache.move_to_end(key)
            return cached
    raw = struct.pack(_PARAMS_FORMAT, Nq, Nd, Lq, Ld, d, flags)
    tensor = torch.frombuffer(bytearray(raw), dtype=torch.int32).clone().to("mps")
    with _params_cache_lock:
        _params_cache[key] = tensor
        _params_cache.move_to_end(key)
        while len(_params_cache) > _PARAMS_CACHE_MAX:
            _params_cache.popitem(last=False)
    return tensor


def _empty_mask() -> torch.Tensor:
    """1-element placeholder for the ``has_*_mask=False`` cases."""
    return torch.zeros(1, dtype=torch.int8, device="mps")


def _prepare_inputs(
    Q: torch.Tensor,
    D: torch.Tensor,
    q_mask: torch.Tensor | None,
    d_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, bool, bool, bool]:
    """Shared shape/dtype/device validation for forward and training paths.

    Returns the contiguous, broadcast-normalised
    ``(Q, D, q_mask, d_mask, kd_layout, q_was_2d, d_was_2d)`` tuple.
    """
    kd_layout = D.dim() == 4
    if kd_layout:
        if Q.dim() != 3:
            raise ValueError(
                f"KD layout (D.dim()==4) needs Q to be [Nq, Lq, d]; got Q.shape={tuple(Q.shape)}"
            )
        if Q.shape[0] != D.shape[0]:
            raise ValueError(f"KD layout needs Q.shape[0] == D.shape[0]; got {Q.shape[0]} vs {D.shape[0]}")
        Nq, K, Ld, d = D.shape
        Lq = Q.shape[1]
        if q_mask is not None and (q_mask.dim() != 2 or tuple(q_mask.shape) != (Nq, Lq)):
            raise ValueError(
                "KD layout needs q_mask.shape == (Nq, Lq); got "
                f"{tuple(q_mask.shape)} for Q.shape={tuple(Q.shape)}"
            )
        # Flatten into the [Nq * K, Ld, d] view the kernel already
        # indexes via `d_global = i * Nd + j` (with Nd = K).
        D = D.contiguous().view(Nq * K, Ld, d)
        if d_mask is not None:
            if d_mask.dim() != 3 or d_mask.shape[:3] != (Nq, K, Ld):
                raise ValueError(
                    "KD layout needs d_mask.shape == (Nq, K, Ld); got "
                    f"{tuple(d_mask.shape)} for D.shape={Nq, K, Ld, d}"
                )
            d_mask = d_mask.contiguous().view(Nq * K, Ld)
        q_was_2d = False
        d_was_2d = False
    else:
        if Q.dim() not in (2, 3) or D.dim() not in (2, 3):
            raise ValueError(
                "Metal path needs Q.dim() in (2, 3) and D.dim() in (2, 3, 4); "
                f"got Q.shape={tuple(Q.shape)}, D.shape={tuple(D.shape)}"
            )
        q_was_2d = Q.dim() == 2
        d_was_2d = D.dim() == 2
        if q_was_2d:
            Q = Q.unsqueeze(0)
        if d_was_2d:
            D = D.unsqueeze(0)
    if q_mask is not None and q_mask.dim() == 1:
        q_mask = q_mask.unsqueeze(0)
    if d_mask is not None and d_mask.dim() == 1:
        d_mask = d_mask.unsqueeze(0)

    if Q.shape[-1] != D.shape[-1]:
        raise ValueError(
            f"Q and D must share the embedding dim; got Q.shape[-1]={Q.shape[-1]} "
            f"vs D.shape[-1]={D.shape[-1]}."
        )
    if Q.device.type != "mps" or D.device.type != "mps":
        raise ValueError(f"Metal path requires MPS tensors; got Q.device={Q.device}, D.device={D.device}.")
    if Q.dtype != D.dtype:
        raise ValueError(f"Q and D must share dtype; got {Q.dtype} vs {D.dtype}.")
    if Q.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(f"Unsupported dtype {Q.dtype}; the Metal path handles fp16 and bf16.")

    return Q.contiguous(), D.contiguous(), q_mask, d_mask, kd_layout, q_was_2d, d_was_2d


def _prepare_flags_and_masks(
    q_mask: torch.Tensor | None,
    d_mask: torch.Tensor | None,
    normalize: bool,
    kd_layout: bool,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Build the kernel flag word + ensure masks are contiguous int8."""
    flags = 0
    if q_mask is not None:
        flags |= _FLAG_HAS_Q_MASK
        q_mask_i8 = q_mask.contiguous().to(torch.int8)
    else:
        q_mask_i8 = _empty_mask()
    if d_mask is not None:
        flags |= _FLAG_HAS_D_MASK
        d_mask_i8 = d_mask.contiguous().to(torch.int8)
    else:
        d_mask_i8 = _empty_mask()
    if normalize:
        flags |= _FLAG_NORMALIZE
    if kd_layout:
        flags |= _FLAG_KD_LAYOUT
    return flags, q_mask_i8, d_mask_i8


def maxsim_inference_metal(
    Q: torch.Tensor,
    D: torch.Tensor,
    q_mask: torch.Tensor | None = None,
    d_mask: torch.Tensor | None = None,
    *,
    normalize: bool = False,
) -> torch.Tensor:
    """Forward-only fused MaxSim on MPS via the ``simdgroup_matrix`` kernel.

    Args:
        Q: ``[Nq, Lq, d]`` or ``[Lq, d]``, fp16 or bf16.
        D: same dtype as ``Q``. Either ``[Nd, Ld, d]`` / ``[Ld, d]``
            (cross-product) or ``[Nq, K, Ld, d]`` (KD / pairs).
        q_mask: optional ``[Nq, Lq]`` mask (``True`` = valid).
        d_mask: optional mask matching ``D`` — ``[Nd, Ld]`` for
            cross-product, ``[Nq, K, Ld]`` for KD.
        normalize: L2-normalize Q and D per-token inside the kernel.

    Returns:
        Cross-product: ``[Nq, Nd]`` fp32 scores (squeezed to match 2-D inputs).
        KD: ``[Nq, K]`` fp32 scores.

    Raises:
        RuntimeError: if MPS or :func:`torch.mps.compile_shader` is missing,
            or the shape / dtype falls outside :func:`supports`.
        ValueError: on shape / device contract violations.
    """
    if not is_available():
        raise RuntimeError(
            "MPS Metal path unavailable: requires PyTorch ≥ 2.10 with MPS "
            "support and `torch.mps.compile_shader`."
        )

    Q, D, q_mask, d_mask, kd_layout, q_was_2d, d_was_2d = _prepare_inputs(Q, D, q_mask, d_mask)
    Nq, Lq, d = Q.shape
    n_axis, Ld, _ = D.shape
    Nd = n_axis // Nq if kd_layout else n_axis
    if d > _D_MAX or d % _BLOCK_K != 0:
        raise RuntimeError(f"Metal path supports d ≤ {_D_MAX} and d %% {_BLOCK_K} == 0; got d={d}.")

    j_per_tg = _pick_j_per_tg(Nd)
    kernel_name = _DTYPE_TO_KERNEL[(Q.dtype, j_per_tg)]

    flags, q_mask_i8, d_mask_i8 = _prepare_flags_and_masks(q_mask, d_mask, normalize, kd_layout)

    scores = torch.empty(Nq, Nd, device="mps", dtype=torch.float32)
    params = _pack_params(Nq, Nd, Lq, Ld, d, flags)

    j_blocks = (Nd + j_per_tg - 1) // j_per_tg

    fn = getattr(_get_lib(), kernel_name)
    fn(
        Q,
        D,
        q_mask_i8,
        d_mask_i8,
        scores,
        params,
        threads=(Nq, j_blocks, _THREADS_PER_GROUP),
        group_size=(1, 1, _THREADS_PER_GROUP),
    )

    if kd_layout:
        return scores
    if q_was_2d and d_was_2d:
        return scores.reshape(())
    if q_was_2d:
        return scores.squeeze(0)
    if d_was_2d:
        return scores.squeeze(-1)
    return scores


class MaxSimFwdCtx(NamedTuple):
    """Forward-pass tensors that the backward needs verbatim.

    ``Q`` / ``D`` are the contiguous, kernel-ready buffers (D flattened
    to ``[Nq * K, Ld, d]`` in KD mode); ``q_mask_i8`` is ``None`` when
    no mask was supplied, else the int8 form the kernel saw.
    """

    Q: torch.Tensor
    D: torch.Tensor
    q_mask_i8: torch.Tensor | None
    kd_layout: bool


def maxsim_train_metal(
    Q: torch.Tensor,
    D: torch.Tensor,
    q_mask: torch.Tensor | None = None,
    d_mask: torch.Tensor | None = None,
    *,
    normalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, MaxSimFwdCtx]:
    """Forward + per-(i, s, j) argmax save. Returns ``(scores, argmax, ctx)``.

    ``argmax`` is ``[Nq * Nd, Lq]`` int32 (Nd = K in KD mode), matching
    the Triton ``maxsim_backward_unified`` contract. ``ctx`` carries
    the contiguous Q / D / q_mask the kernel saw so the backward
    doesn't redo broadcasts.
    """
    if not is_available():
        raise RuntimeError(
            "MPS Metal path unavailable: requires PyTorch ≥ 2.10 with MPS "
            "support and `torch.mps.compile_shader`."
        )

    Q, D, q_mask, d_mask, kd_layout, q_was_2d, d_was_2d = _prepare_inputs(Q, D, q_mask, d_mask)
    if q_was_2d or d_was_2d:
        raise ValueError("maxsim_train_metal needs batched (3-D Q, 3-D or 4-D D); reshape upstream.")
    Nq, Lq, d = Q.shape
    n_axis, Ld, _ = D.shape
    Nd = n_axis // Nq if kd_layout else n_axis
    if d > _D_MAX or d % _BLOCK_K != 0:
        raise RuntimeError(f"Metal path supports d ≤ {_D_MAX} and d %% {_BLOCK_K} == 0; got d={d}.")

    j_per_tg = _pick_j_per_tg(Nd)
    kernel_name = _DTYPE_TO_TRAIN_KERNEL[(Q.dtype, j_per_tg)]

    flags, q_mask_i8, d_mask_i8 = _prepare_flags_and_masks(q_mask, d_mask, normalize, kd_layout)

    scores = torch.empty(Nq, Nd, device="mps", dtype=torch.float32)
    # -1 sentinel: any (i, s, j) row the kernel doesn't touch (e.g. when
    # the whole tile is d-masked) keeps -1, and the backward skips it.
    argmax = torch.full((Nq * Nd, Lq), -1, device="mps", dtype=torch.int32)
    params = _pack_params(Nq, Nd, Lq, Ld, d, flags)

    j_blocks = (Nd + j_per_tg - 1) // j_per_tg

    fn = getattr(_get_lib(), kernel_name)
    fn(
        Q,
        D,
        q_mask_i8,
        d_mask_i8,
        scores,
        params,
        argmax,
        threads=(Nq, j_blocks, _THREADS_PER_GROUP),
        group_size=(1, 1, _THREADS_PER_GROUP),
    )

    ctx = MaxSimFwdCtx(
        Q=Q,
        D=D,
        q_mask_i8=q_mask_i8 if q_mask is not None else None,
        kd_layout=kd_layout,
    )
    return scores, argmax, ctx


def maxsim_backward_metal(
    grad_scores: torch.Tensor,
    Q: torch.Tensor,
    D: torch.Tensor,
    argmax: torch.Tensor,
    q_mask: torch.Tensor | None = None,
    *,
    kd_layout: bool | None = None,
    normalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``grad_Q``, ``grad_D`` for the fused MaxSim. Mirrors Triton's
    :func:`maxsim_backward_unified` API.

    Args:
        grad_scores: ``[Nq, Nd]`` fp32 upstream gradient.
        Q: ``[Nq, Lq, d]`` contiguous, fp16 / bf16. Pass the *raw* Q
            when ``normalize=True`` (kernel normalizes internally).
        D: ``[Nd, Ld, d]`` (cross-product) or ``[Nq, K, Ld, d]`` /
            ``[Nq * K, Ld, d]`` (KD: either the natural 4-D or the
            flat view the forward saw — both accepted). Same
            raw/normalized rule as ``Q``.
        argmax: ``[Nq * Nd, Lq]`` int32 winner buffer from
            :func:`maxsim_train_metal`.
        q_mask: optional ``[Nq, Lq]`` int8 mask (``1`` = valid).
        kd_layout: ``True`` if ``D`` is the flat KD view. Auto-detected
            (`D.dim() == 4`) when left as ``None``.
        normalize: when ``True``, fold the L2-normalize Jacobian into
            the writes. Eliminates the host-side norm + projection
            ops that otherwise dominate the backward step.

    Returns:
        ``(grad_Q, grad_D)`` cast back to ``Q.dtype`` / ``D.dtype``.
        ``grad_D`` matches the original ``D`` shape (4-D in -> 4-D out).
    """
    if not is_available():
        raise RuntimeError("MPS Metal path unavailable.")
    if Q.dtype != D.dtype or Q.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(f"maxsim_backward_metal needs matching fp16/bf16 Q/D; got {Q.dtype}, {D.dtype}.")
    if grad_scores.dtype != torch.float32:
        grad_scores = grad_scores.to(torch.float32)
    if argmax.dtype != torch.int32:
        argmax = argmax.to(torch.int32)
    if q_mask is not None and q_mask.dtype != torch.int8:
        q_mask = q_mask.to(torch.int8)

    d_input_shape = D.shape
    if D.dim() == 4:
        if kd_layout is False:
            raise ValueError("Got 4-D D but kd_layout=False; pass kd_layout=True or omit the flag.")
        kd_layout = True
        Nq_d, K, Ld_4d, d_4d = D.shape
        D = D.contiguous().view(Nq_d * K, Ld_4d, d_4d)
    elif kd_layout is None:
        kd_layout = False

    Q = Q.contiguous()
    D = D.contiguous()
    grad_scores = grad_scores.contiguous()
    argmax = argmax.contiguous()

    Nq, Lq, d = Q.shape
    n_axis, Ld, d_d = D.shape
    if d_d != d:
        raise ValueError(f"Q and D must share the embedding dim; got {d} vs {d_d}.")
    if d > _D_MAX or d % _BLOCK_K != 0:
        raise RuntimeError(f"Metal backward supports d ≤ {_D_MAX} and d %% {_BLOCK_K} == 0; got d={d}.")
    Nd = n_axis // Nq if kd_layout else n_axis
    if argmax.shape != (Nq * Nd, Lq):
        raise ValueError(f"argmax must be [{Nq * Nd}, {Lq}]; got {tuple(argmax.shape)}.")
    if grad_scores.shape != (Nq, Nd):
        raise ValueError(f"grad_scores must be [{Nq}, {Nd}]; got {tuple(grad_scores.shape)}.")

    flags = 0
    if q_mask is not None:
        flags |= _FLAG_HAS_Q_MASK
        q_mask_i8 = q_mask.contiguous()
    else:
        q_mask_i8 = _empty_mask()
    if kd_layout:
        flags |= _FLAG_KD_LAYOUT

    grad_Q = torch.empty(Nq, Lq, d, device="mps", dtype=torch.float32)
    grad_D = torch.zeros(n_axis, Ld, d, device="mps", dtype=torch.float32)

    params = _pack_params(Nq, Nd, Lq, Ld, d, flags)
    kernel_name = _DTYPE_TO_BWD_KERNEL[(Q.dtype, bool(normalize))]
    fn = getattr(_get_lib(), kernel_name)
    fn(
        Q,
        D,
        argmax,
        grad_scores,
        q_mask_i8,
        grad_Q,
        grad_D,
        params,
        threads=(Nq, Lq, d),
        group_size=(1, 1, d),
    )

    grad_D = grad_D.to(D.dtype)
    if len(d_input_shape) == 4:
        grad_D = grad_D.view(d_input_shape)
    return grad_Q.to(Q.dtype), grad_D


__all__ = [
    "is_available",
    "supports",
    "maxsim_inference_metal",
    "maxsim_train_metal",
    "maxsim_backward_metal",
]
