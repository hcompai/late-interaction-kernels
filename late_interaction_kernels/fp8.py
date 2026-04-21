"""FP8 MaxSim inference (Hopper SM90+ / Blackwell).

Motivation
----------
For large-corpus reranking the forward is HBM-bound: every ``D`` tile
round-trips through L2 + HBM and tensor cores idle. FP8 halves the
footprint and doubles the tensor-core throughput on Hopper (``WGMMA``)
and Blackwell.

This module ships the inference-only FP8 path. Training-side FP8 (with
per-output scaling + master-weight maintenance) is a separate RFC.

API
---
::

    Q_fp8, sq = quantize_fp8_per_tensor(Q)       # or quantize_fp8_per_token(...)
    D_fp8, sd = quantize_fp8_per_tensor(D)
    scores = maxsim_inference_fp8(Q_fp8, D_fp8, scale_Q=sq, scale_D=sd)

The kernel assumes the standard ``torch.float8_e4m3fn`` dtype (safer
dynamic range than ``e5m2`` for normalized embeddings). Scores are
returned in fp32.

Fallback
--------
When the running Triton build doesn't support the ``tl.dot`` FP8 path
— typically pre-Hopper or Triton < 3.0 — we reconstitute the kernel
automatically in bf16 and warn once. The API contract is preserved so
calling code never has to branch.
"""

from __future__ import annotations

import warnings

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

from ._autotune import forward_configs, prune_forward
from ._utils import next_pow2

_FP8_E4M3_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max — hard-coded for old torch compat.


def quantize_fp8_per_tensor(
    X: torch.Tensor, dtype: torch.dtype = torch.float8_e4m3fn
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``X`` to FP8 with a single fp32 scale such that the max
    absolute value maps to ``finfo(dtype).max``. Returns ``(X_fp8, scale)``
    where reading back is ``X_fp8.to(fp32) * scale``.
    """
    if X.numel() == 0:
        return X.to(dtype), torch.ones((), device=X.device, dtype=torch.float32)
    amax = X.abs().amax().clamp_min(1e-6).to(torch.float32)
    scale = amax / _FP8_E4M3_MAX
    X_fp8 = (X.to(torch.float32) / scale).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(dtype)
    return X_fp8.contiguous(), scale


def quantize_fp8_per_token(
    X: torch.Tensor, dtype: torch.dtype = torch.float8_e4m3fn
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize per-row along the last axis. ``X`` is 2-D or 3-D; scale
    has shape ``X.shape[:-1]`` (fp32).
    """
    amax = X.abs().amax(dim=-1, keepdim=False).clamp_min(1e-6).to(torch.float32)
    scale = amax / _FP8_E4M3_MAX
    # Broadcast along the last axis
    X_fp8 = (X.to(torch.float32) / scale.unsqueeze(-1)).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(dtype)
    return X_fp8.contiguous(), scale


def dequantize_fp8_per_tensor(X_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return X_fp8.to(torch.float32) * scale


def dequantize_fp8_per_token(X_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return X_fp8.to(torch.float32) * scale.unsqueeze(-1)


if _HAS_TRITON:

    @triton.autotune(
        configs=forward_configs(),
        key=["Lq", "Ld", "d_pad", "has_q_mask", "has_d_mask", "SCALE_Q_PER_TOKEN", "SCALE_D_PER_TOKEN"],
        prune_configs_by={"early_config_prune": prune_forward},
    )
    @triton.jit
    def _maxsim_fp8_fwd_kernel(
        Q_ptr,
        D_ptr,
        q_scale_ptr,
        d_scale_ptr,
        q_mask_ptr,
        d_mask_ptr,
        scores_ptr,
        Nq: tl.constexpr,
        Nd: tl.constexpr,
        Lq: tl.constexpr,
        Ld: tl.constexpr,
        d: tl.constexpr,
        d_pad: tl.constexpr,
        stride_q_n,
        stride_q_l,
        stride_q_d,
        stride_d_n,
        stride_d_l,
        stride_d_d,
        stride_s_n,
        stride_s_d,
        stride_qm_n,
        stride_qm_l,
        stride_dm_n,
        stride_dm_l,
        stride_qs_n,
        stride_qs_l,
        stride_ds_n,
        stride_ds_l,
        has_q_mask: tl.constexpr,
        has_d_mask: tl.constexpr,
        SCALE_Q_PER_TOKEN: tl.constexpr,
        SCALE_D_PER_TOKEN: tl.constexpr,
        scale_qd_tensor,  # fp32 — used only when both scales are per-tensor
        BLOCK_Q: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        q_idx = pid // Nd
        d_idx = pid % Nd

        k_off = tl.arange(0, d_pad)
        k_mask = k_off < d

        score_acc = tl.zeros([], dtype=tl.float32)

        for q_start in tl.static_range(0, Lq, BLOCK_Q):
            q_off = q_start + tl.arange(0, BLOCK_Q)
            q_valid = q_off < Lq

            if has_q_mask:
                qm = tl.load(
                    q_mask_ptr + q_idx * stride_qm_n + q_off * stride_qm_l,
                    mask=q_valid,
                    other=0,
                ).to(tl.int1)
                q_active = q_valid & qm
            else:
                q_active = q_valid

            # Load Q tile in fp8. Triton's tl.dot natively supports fp8
            # operands on Hopper+ with an fp32 accumulator.
            Q_block = tl.load(
                Q_ptr + q_idx * stride_q_n + q_off[:, None] * stride_q_l + k_off[None, :] * stride_q_d,
                mask=q_valid[:, None] & k_mask[None, :],
                other=0.0,
            )

            # Per-Q-token scale if requested (one fp32 per row).
            if SCALE_Q_PER_TOKEN:
                q_scale_vec = tl.load(
                    q_scale_ptr + q_idx * stride_qs_n + q_off * stride_qs_l,
                    mask=q_valid,
                    other=1.0,
                ).to(tl.float32)
            else:
                q_scale_vec = tl.zeros([BLOCK_Q], dtype=tl.float32)  # unused

            m = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)

            for d_start in range(0, Ld, BLOCK_D):
                d_off = d_start + tl.arange(0, BLOCK_D)
                d_valid = d_off < Ld

                if has_d_mask:
                    dm = tl.load(
                        d_mask_ptr + d_idx * stride_dm_n + d_off * stride_dm_l,
                        mask=d_valid,
                        other=0,
                    ).to(tl.int1)
                    d_active = d_valid & dm
                else:
                    d_active = d_valid

                D_block = tl.load(
                    D_ptr + d_idx * stride_d_n + d_off[:, None] * stride_d_l + k_off[None, :] * stride_d_d,
                    mask=d_valid[:, None] & k_mask[None, :],
                    other=0.0,
                )

                # Tensor-core fp8 dot: inputs fp8, accumulator fp32.
                S = tl.dot(Q_block, tl.trans(D_block), out_dtype=tl.float32)

                # Apply per-tensor scale at tile level (cheap broadcast).
                if SCALE_Q_PER_TOKEN and SCALE_D_PER_TOKEN:
                    d_scale_vec = tl.load(
                        d_scale_ptr + d_idx * stride_ds_n + d_off * stride_ds_l,
                        mask=d_valid,
                        other=1.0,
                    ).to(tl.float32)
                    S = S * q_scale_vec[:, None] * d_scale_vec[None, :]
                elif SCALE_Q_PER_TOKEN:
                    S = S * q_scale_vec[:, None] * scale_qd_tensor
                elif SCALE_D_PER_TOKEN:
                    d_scale_vec = tl.load(
                        d_scale_ptr + d_idx * stride_ds_n + d_off * stride_ds_l,
                        mask=d_valid,
                        other=1.0,
                    ).to(tl.float32)
                    S = S * d_scale_vec[None, :] * scale_qd_tensor
                else:
                    S = S * scale_qd_tensor

                S = tl.where(d_active[None, :], S, float("-inf"))
                tile_max = tl.max(S, axis=1)
                m = tl.maximum(m, tile_max)

            m_finite = m != float("-inf")
            m = tl.where(m_finite & q_active, m, 0.0)
            score_acc += tl.sum(m)

        tl.store(scores_ptr + q_idx * stride_s_n + d_idx * stride_s_d, score_acc)


def _hopper_or_newer() -> bool:
    if not torch.cuda.is_available():
        return False
    maj, _ = torch.cuda.get_device_capability()
    return maj >= 9


def _fp8_dtype_supported(dtype: torch.dtype) -> bool:
    return dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
    )


_WARNED_FP8_FALLBACK = False


def maxsim_inference_fp8(
    Q: torch.Tensor,
    D: torch.Tensor,
    *,
    scale_Q: torch.Tensor,
    scale_D: torch.Tensor,
    q_mask: torch.Tensor | None = None,
    d_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Inference MaxSim on FP8 inputs with per-tensor or per-token scales.

    Args:
        Q: ``[Nq, Lq, d]`` or ``[Lq, d]`` in ``torch.float8_e4m3fn``.
        D: ``[Nd, Ld, d]`` or ``[Ld, d]`` in the same FP8 dtype.
        scale_Q: ``()`` (per-tensor) or ``[Nq, Lq]`` / ``[Lq]`` (per-token) fp32 scale.
        scale_D: same rules for the doc side.
        q_mask, d_mask: bool masks (same convention as :func:`maxsim`).

    Returns:
        ``scores [Nq, Nd]`` fp32, squeezed for 2-D inputs.

    Notes:
        - Hopper SM90+ or Blackwell required. On older GPUs (or old
          Triton) the function transparently falls back to a dequantized
          bf16 path with a one-time warning.
        - Ties with argmax are not tie-broken — the first fp8-equal
          doc token wins. Use the bf16 rerank path if ties matter.
    """
    global _WARNED_FP8_FALLBACK

    # --------------------------------------------------------------- shape norm
    q_was_2d = Q.dim() == 2
    d_was_2d = D.dim() == 2
    if q_was_2d:
        Q = Q.unsqueeze(0)
        if scale_Q.dim() > 0:
            scale_Q = scale_Q.unsqueeze(0)
    if d_was_2d:
        D = D.unsqueeze(0)
        if scale_D.dim() > 0:
            scale_D = scale_D.unsqueeze(0)
    if q_mask is not None and q_mask.dim() == 1:
        q_mask = q_mask.unsqueeze(0)
    if d_mask is not None and d_mask.dim() == 1:
        d_mask = d_mask.unsqueeze(0)

    if Q.shape[-1] != D.shape[-1]:
        raise ValueError(f"Q and D must share the embedding dim; got {Q.shape[-1]} vs {D.shape[-1]}.")
    Q = Q.contiguous()
    D = D.contiguous()

    # Per-tensor scales are 0-d fp32; per-token scales are 2-D fp32 [N, L].
    scale_q_per_token = scale_Q.dim() >= 2
    scale_d_per_token = scale_D.dim() >= 2
    if not (scale_q_per_token or scale_d_per_token) and (scale_Q.numel() == 1 and scale_D.numel() == 1):
        scale_qd_tensor = float((scale_Q * scale_D).item())
    else:
        # Always pass the per-tensor product (used as a multiplier for the
        # non-per-token dimension when only one side is per-token).
        if scale_q_per_token and scale_d_per_token:
            scale_qd_tensor = 1.0
        elif scale_q_per_token:
            scale_qd_tensor = float(scale_D.item())
        elif scale_d_per_token:
            scale_qd_tensor = float(scale_Q.item())
        else:
            scale_qd_tensor = float((scale_Q * scale_D).item())

    use_triton_fp8 = (
        _HAS_TRITON
        and Q.is_cuda
        and _fp8_dtype_supported(Q.dtype)
        and Q.dtype == D.dtype
        and _hopper_or_newer()
    )
    if not use_triton_fp8:
        if not _WARNED_FP8_FALLBACK:
            warnings.warn(
                "maxsim_inference_fp8: FP8 tensor-core path unavailable on this "
                "GPU/Triton; falling back to dequantized bf16. Inference still "
                "works but without the FP8 speedup.",
                RuntimeWarning,
                stacklevel=2,
            )
            _WARNED_FP8_FALLBACK = True
        # Dequantize and fall back to the regular kernel.
        from .autograd import maxsim_inference

        if scale_q_per_token:
            Qd = Q.to(torch.bfloat16) * scale_Q.unsqueeze(-1).to(torch.bfloat16)
        else:
            Qd = Q.to(torch.bfloat16) * scale_Q.to(torch.bfloat16)
        if scale_d_per_token:
            Dd = D.to(torch.bfloat16) * scale_D.unsqueeze(-1).to(torch.bfloat16)
        else:
            Dd = D.to(torch.bfloat16) * scale_D.to(torch.bfloat16)
        scores = maxsim_inference(Qd, Dd, q_mask=q_mask, d_mask=d_mask)
        if q_was_2d and d_was_2d:
            return scores.reshape(())
        if q_was_2d:
            return scores.squeeze(0)
        if d_was_2d:
            return scores.squeeze(-1)
        return scores

    Nq, Lq, d = Q.shape
    Nd, Ld, _ = D.shape
    d_pad = next_pow2(d)

    has_q_mask = q_mask is not None
    has_d_mask = d_mask is not None
    q_mask_i8 = q_mask.contiguous().to(torch.int8) if has_q_mask else Q
    d_mask_i8 = d_mask.contiguous().to(torch.int8) if has_d_mask else D
    qm_strides = (q_mask_i8.stride(0), q_mask_i8.stride(1)) if has_q_mask else (0, 0)
    dm_strides = (d_mask_i8.stride(0), d_mask_i8.stride(1)) if has_d_mask else (0, 0)

    if scale_q_per_token:
        q_scale_ptr = scale_Q.contiguous().to(torch.float32)
        qs_strides = (q_scale_ptr.stride(0), q_scale_ptr.stride(1))
    else:
        q_scale_ptr = Q  # unused but Triton needs a pointer
        qs_strides = (0, 0)
    if scale_d_per_token:
        d_scale_ptr = scale_D.contiguous().to(torch.float32)
        ds_strides = (d_scale_ptr.stride(0), d_scale_ptr.stride(1))
    else:
        d_scale_ptr = D
        ds_strides = (0, 0)

    scores = torch.empty(Nq, Nd, device=Q.device, dtype=torch.float32)
    _maxsim_fp8_fwd_kernel[(Nq * Nd,)](
        Q,
        D,
        q_scale_ptr,
        d_scale_ptr,
        q_mask_i8,
        d_mask_i8,
        scores,
        Nq,
        Nd,
        Lq,
        Ld,
        d,
        d_pad,
        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        D.stride(0),
        D.stride(1),
        D.stride(2),
        scores.stride(0),
        scores.stride(1),
        qm_strides[0],
        qm_strides[1],
        dm_strides[0],
        dm_strides[1],
        qs_strides[0],
        qs_strides[1],
        ds_strides[0],
        ds_strides[1],
        has_q_mask,
        has_d_mask,
        scale_q_per_token,
        scale_d_per_token,
        float(scale_qd_tensor),
    )

    if q_was_2d and d_was_2d:
        return scores.reshape(())
    if q_was_2d:
        return scores.squeeze(0)
    if d_was_2d:
        return scores.squeeze(-1)
    return scores


__all__ = [
    "maxsim_inference_fp8",
    "quantize_fp8_per_tensor",
    "quantize_fp8_per_token",
    "dequantize_fp8_per_tensor",
    "dequantize_fp8_per_token",
]
