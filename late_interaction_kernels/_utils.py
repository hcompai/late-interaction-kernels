"""Small helpers shared across kernels."""

import functools

import torch


def next_pow2(x: int) -> int:
    """Smallest power of two >= x. `next_pow2(0)` returns 1."""
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def package_at_least(name: str, minimum: str) -> bool:
    """True when installed distribution ``name`` is >= ``minimum`` (absent → False).

    Compares leading numeric release parts (``"1.5.1"`` → ``(1, 5, 1)``), so
    dev/rc/post suffixes don't matter for a floor check.
    """
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _dist_version

    def _release(v: str) -> tuple[int, ...]:
        parts: list[int] = []
        for chunk in v.split("."):
            head = ""
            for ch in chunk:
                if not ch.isdigit():
                    break
                head += ch
            if not head:
                break
            parts.append(int(head))
        return tuple(parts)

    try:
        return _release(_dist_version(name)) >= _release(minimum)
    except PackageNotFoundError:
        return False


@functools.lru_cache(maxsize=1)
def detect_gpu() -> str:
    """Return a short GPU family string: 'hopper' | 'a100' | 'ada' | 'ampere' | 'generic'."""
    if not torch.cuda.is_available():
        return "generic"
    name = torch.cuda.get_device_name().lower()
    if "h100" in name or "h200" in name:
        return "hopper"
    if "a100" in name:
        return "a100"
    if "l4" in name or "l40" in name or "rtx 40" in name:
        return "ada"
    if "3090" in name or "a10" in name or "a40" in name:
        return "ampere"
    return "generic"


def ensure_contiguous_last(x: torch.Tensor) -> torch.Tensor:
    """Make sure the last dim is contiguous — cheap path for most inputs."""
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


@functools.cache
def _cached_placeholder(device: str, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(1, device=device, dtype=dtype)


def autotune_placeholder(ref: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """A cached 1-element ``dtype`` tensor on ``ref``'s device, to stand in for
    an absent optional kernel arg.

    Triton's autotuner appends ``str(arg.dtype)`` of every tensor argument to
    its cache key, so an optional arg that flips between ``None`` and a real
    tensor must be backed by a placeholder of the *real* arg's dtype — otherwise
    present-vs-absent changes the key and re-triggers the (5–10 s) sweep on
    every variable-length batch. The kernel never reads it: its ``has_*``
    constexpr is ``False`` and the matching strides are zero.
    """
    return _cached_placeholder(str(ref.device), dtype)


def pick_compute_dtype(Q: torch.Tensor, D: torch.Tensor) -> torch.dtype:
    """Pick the compute dtype for `tl.dot`.

    We honor user intent: if both tensors are fp16/bf16, dot runs in that dtype
    with fp32 accumulator. If either is fp32 we fall back to fp16 on the tile
    (fp32 GEMM doesn't go through tensor cores on H100 anyway).
    """
    if Q.dtype == torch.bfloat16 or D.dtype == torch.bfloat16:
        return torch.bfloat16
    return torch.float16
