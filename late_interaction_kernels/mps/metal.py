"""Fused MaxSim forward on MPS via Metal ``simdgroup_matrix`` MMA.

Direct Metal-side analogue of :mod:`late_interaction_kernels.forward`:
the same FlashAttention-style outer-product tiling, but the inner GEMM
runs on Apple's ``simdgroup_matrix<T, 8, 8>`` MMA instead of CUDA tensor
cores. The ``[Nq · Nd · Lq · Ld]`` similarity tensor never reaches HBM.

Each threadgroup serves ``J_PER_TG`` consecutive ``j`` values, loading
Q exactly once and hoisting it into a register-resident
``simdgroup_matrix`` cache that survives every ``(j, d-chunk)`` pair —
this is the single largest win versus the naive one-(i,j)-per-thread-
group layout. Two persistence levels are dispatched by ``Nd``:

* ``J_PER_TG = 1`` for tiny corpora where the launch grid needs to be
  wide to keep the GPU saturated;
* ``J_PER_TG = 8`` for the typical inference regime, amortising the Q
  load and the register-cache build 8× per threadgroup.

D streams in 32-row tiles through threadgroup memory with the optional
L2-normalize folded into the cooperative load.

Constraints (everything outside falls back to ``torch.compile``):

* ``d`` is a multiple of 8 and ≤ ``D_MAX = 128`` (covers ColBERT,
  ColPali, LateOn, LateOn-Code, edge-d48 / d64);
* fp16 / bf16 inputs only — Apple's MMA accepts those; fp32 inputs
  go through the compile path.

The MSL source lives in :file:`_maxsim.metal` so editors give us
syntax highlighting / navigation. This module is the single
source-of-truth for the host-side constants (tile sizes, flag bits,
struct format) — they're substituted into the source via
``str.format(...)`` at JIT-compile time, so host and device never drift.
"""

import struct
import threading
from pathlib import Path
from typing import Any

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

    The ``.metal`` file is a ``str.format``-style template — literal MSL
    braces are doubled (``{{`` / ``}}``), substitutions are single (e.g.
    ``{BLOCK_Q}``). Keeping the template external is purely a developer-
    experience win (editor syntax highlighting); the bytes handed to
    ``torch.mps.compile_shader`` are identical to the previous inline
    f-string.
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
    if Q.shape[-1] != D.shape[-1]:
        return False
    if D.dim() == 4:
        # KD / pairs layout: Q must be [Nq, Lq, d] and D must be [Nq, K, Ld, d].
        if Q.dim() != 3 or Q.shape[0] != D.shape[0]:
            return False
    d = Q.shape[-1]
    return 0 < d <= _D_MAX and d % _BLOCK_K == 0


def _pack_params(Nq: int, Nd: int, Lq: int, Ld: int, d: int, flags: int) -> torch.Tensor:
    """Pack the kernel's ``MaxSimParams`` struct into a 6-int32 MPS tensor."""
    raw = struct.pack(_PARAMS_FORMAT, Nq, Nd, Lq, Ld, d, flags)
    return torch.frombuffer(bytearray(raw), dtype=torch.int32).clone().to("mps")


def _empty_mask() -> torch.Tensor:
    """1-element placeholder for the ``has_*_mask=False`` cases."""
    return torch.zeros(1, dtype=torch.int8, device="mps")


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
        Q: ``[Nq, Lq, d]`` or ``[Lq, d]`` query embeddings on ``mps``,
            fp16 or bf16.
        D: document embeddings on ``mps``, same dtype as ``Q``. Two
            layouts are accepted:
              * cross-product (default): ``[Nd, Ld, d]`` or ``[Ld, d]``,
                every query scores against the same ``Nd`` docs;
              * KD / pairs: ``[Nq, K, Ld, d]`` — each query owns its own
                ``K``-slab of candidates. Mirrors the public
                :func:`maxsim` 4-D dispatch on CUDA.
        q_mask: optional ``[Nq, Lq]`` bool / int8 mask (``True`` = valid).
        d_mask: optional mask matching ``D``: ``[Nd, Ld]`` for
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

    kd_layout = D.dim() == 4
    if kd_layout:
        if Q.dim() != 3:
            raise ValueError(
                f"KD layout (D.dim()==4) needs Q to be [Nq, Lq, d]; got Q.shape={tuple(Q.shape)}"
            )
        if Q.shape[0] != D.shape[0]:
            raise ValueError(
                f"KD layout needs Q.shape[0] == D.shape[0]; got {Q.shape[0]} vs {D.shape[0]}"
            )
        Nq, K, Ld, d = D.shape
        # Flatten D and d_mask into the [Nq * K, Ld, d] view the kernel
        # already knows how to index (`d_global = i * Nd + j`). The kernel
        # treats `params.Nd = K` and the per-query slab is selected by `i`.
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
        raise ValueError(
            f"maxsim_inference_metal requires MPS tensors; got Q.device={Q.device}, D.device={D.device}."
        )
    if Q.dtype != D.dtype:
        raise ValueError(f"Q and D must share dtype; got {Q.dtype} vs {D.dtype}.")
    if Q.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(f"Unsupported dtype {Q.dtype}; the Metal path handles fp16 and bf16.")

    Nq, Lq, d = Q.shape
    # In cross-product mode `params.Nd` is the corpus size. In KD mode
    # it's `K` (the per-query candidate count); the kernel then derives
    # the flat D index as `(i * Nd + j) * Ld + t`.
    n_axis, Ld, _ = D.shape
    if kd_layout:
        Nd = n_axis // Nq  # = K
    else:
        Nd = n_axis
    if d > _D_MAX or d % _BLOCK_K != 0:
        raise RuntimeError(f"Metal path supports d ≤ {_D_MAX} and d %% {_BLOCK_K} == 0; got d={d}.")

    j_per_tg = _pick_j_per_tg(Nd)
    kernel_name = _DTYPE_TO_KERNEL[(Q.dtype, j_per_tg)]

    Q = Q.contiguous()
    D = D.contiguous()

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

    out = torch.empty(Nq, Nd, device="mps", dtype=torch.float32)
    params = _pack_params(Nq, Nd, Lq, Ld, d, flags)

    j_blocks = (Nd + j_per_tg - 1) // j_per_tg

    fn = getattr(_get_lib(), kernel_name)
    fn(
        Q,
        D,
        q_mask_i8,
        d_mask_i8,
        out,
        params,
        threads=(Nq, j_blocks, _THREADS_PER_GROUP),
        group_size=(1, 1, _THREADS_PER_GROUP),
    )

    if kd_layout:
        # Output is already shaped [Nq, K] — exactly what KD callers want.
        return out
    if q_was_2d and d_was_2d:
        return out.reshape(())
    if q_was_2d:
        return out.squeeze(0)
    if d_was_2d:
        return out.squeeze(-1)
    return out


__all__ = ["is_available", "supports", "maxsim_inference_metal"]
