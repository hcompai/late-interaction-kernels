"""Small helpers shared across kernels."""

import functools

import torch


def next_pow2(x: int) -> int:
    """Smallest power of two >= x. `next_pow2(0)` returns 1."""
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def bucket_seqlen(max_len: int, floor: int = 16) -> int:
    """Round a max sequence length up to the next power of two for use as an
    autotune cache key.

    Keying the autotune cache on an exact max length re-triggers the (5-10 s)
    sweep for every distinct value; bucketing to {floor, 2*floor, ...} caps
    the cache at a handful of entries. Only the *key* is bucketed — kernels
    keep the exact value as their runtime loop bound, so no masked iterations
    are added (bucketing the bound itself measured 30-50% throughput loss on
    the PLAID kernels). The default floor matches the smallest block size in
    the autotune pools.
    """
    if max_len <= 0:
        return 0
    return max(floor, next_pow2(max_len))


def assert_max_seqlen_covers(cu_seqlens: torch.Tensor, max_seqlen: int, arg_name: str) -> None:
    """Check a caller-supplied ``max_seqlen`` against the real ``cu_seqlens`` maxima.

    ``max_seqlen_*`` arguments are hard kernel loop bounds, not hints: a
    too-small value silently truncates tokens and returns wrong scores.
    ``torch._assert_async`` runs the check on-device with no D2H sync (the
    whole point of letting callers pass ``max_seqlen`` is to skip that sync),
    so a violation surfaces as an async device-side assert rather than an
    eager ``ValueError`` — same trade-off as the length checks in
    :func:`late_interaction_kernels.padded.pack_padded`.
    """
    if cu_seqlens.numel() <= 1:
        return
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    torch._assert_async(
        seqlens.max().le(max_seqlen),
        f"{arg_name}={max_seqlen} is smaller than the longest sequence in cu_seqlens; "
        f"it is a hard loop bound and would silently truncate tokens.",
    )


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
    """Return a short GPU family string:
    'hopper' | 'blackwell' | 'a100' | 'ada' | 'ampere' | 'generic'.

    Keyed on compute capability rather than the marketing name — names are
    open-ended (an "RTX PRO 6000 Blackwell" or "A30" never matched) while the
    capability cleanly identifies the SMEM-per-SM class the autotune configs
    care about.
    """
    if not torch.cuda.is_available():
        return "generic"
    major, minor = torch.cuda.get_device_capability()
    if major == 12:
        # Consumer Blackwell (RTX 50): 100 KiB-class SMEM, Ada-like configs.
        return "ada"
    if major >= 10:
        # Datacenter Blackwell (B100/B200/GB200): Hopper-class 228 KiB SMEM.
        return "blackwell"
    if major == 9:
        return "hopper"
    if (major, minor) == (8, 0):
        return "a100"
    if (major, minor) == (8, 9):
        return "ada"
    if major == 8:
        return "ampere"  # 8.6 / 8.7: 3090, A10, A40, Orin
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
    with fp32 accumulator. If either is fp32 we downcast the tile to fp16:
    same 10-bit mantissa as TF32 (the fp32 tensor-core path) but half the
    operand bandwidth, and the fp16 range is a non-issue on the unit-norm
    embeddings these kernels score.
    """
    if Q.dtype == torch.bfloat16 or D.dtype == torch.bfloat16:
        return torch.bfloat16
    return torch.float16
