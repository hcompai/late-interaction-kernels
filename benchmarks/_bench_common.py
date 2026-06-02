"""Shared benchmark infra: timing, the result record, and DDP helpers.

``benchmarks/`` isn't an installable package, but the directory of the script
being run is always on ``sys.path[0]``, so sibling benches import this directly
(``from _bench_common import Measurement, _log, _timed_step``) — no
``importlib.util.spec_from_file_location`` dance needed.
"""

import os
from dataclasses import dataclass

import torch


def _is_dist() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _rank() -> int:
    return torch.distributed.get_rank() if _is_dist() else 0


def _world_size() -> int:
    return torch.distributed.get_world_size() if _is_dist() else 1


def _log(msg: str) -> None:
    if _rank() == 0:
        print(msg, flush=True)


def _init_ddp() -> torch.device:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    return torch.device(f"cuda:{local_rank}")


@dataclass
class Measurement:
    step_ms: float
    peak_mb: float
    err: str | None = None


def _timed_step(closure, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        closure()
    torch.cuda.synchronize()
    if _is_dist():
        torch.distributed.barrier()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        closure()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters
