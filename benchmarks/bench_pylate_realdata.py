"""End-to-end PyLate training on *real* query–passage triplets.

Loads ``sentence-transformers/msmarco-bm25`` (triplet split) — the same
public benchmark set PyLate / sentence-transformers use for ColBERT-style
contrastive training — tokenizes actual queries and passages, and times
full ``ColBERT`` encoder + ``Contrastive`` / ``CachedContrastive`` steps
with and without ``patch_pylate()``.

This is the closest thing to a real training loop we can run in a single
GPU job without standing up a multi-hour fine-tune: real text statistics,
real tokenizer lengths, real backward through the encoder.

    python benchmarks/bench_pylate_realdata.py \\
        --model lightonai/LateOn-Code-edge \\
        --recipe reason --batch-size 64 --mini-batch-size 16 \\
        --max-samples 512 --steps 30

Requires: ``pip install pylate datasets`` (see ``.[dev,pylate]`` extra).
"""
# ruff: noqa: F821  -- ruff loses ``loss_fn`` / ``optim`` across the trailing
# ``del`` in ``run_realdata`` and false-positives on the ``step`` closure.

import argparse
import gc
import json
import os
import random
import time
from dataclasses import dataclass

import torch
from _bench_common import Measurement, _log, _timed_step

DATASET_DEFAULT = "sentence-transformers/msmarco-bm25"
DATASET_CONFIG = "triplet"
MODEL_DEFAULT = "lightonai/LateOn-Code-edge"


@dataclass(frozen=True)
class TripletRow:
    query: str
    positive: str
    negative: str


def _load_triplets(
    dataset: str,
    config: str,
    split: str,
    max_samples: int,
    seed: int,
) -> list[TripletRow]:
    from datasets import load_dataset

    # Slice at load time — msmarco-bm25 train is ~500k rows; we only need a
    # few hundred for a timing sweep.
    n_fetch = max_samples + 64  # headroom for shuffling
    ds = load_dataset(dataset, config, split=f"{split}[:{n_fetch}]")

    cols = set(ds.column_names)
    _log(f"  dataset columns: {sorted(cols)}")

    def _pick(ex: dict, *keys: str) -> str:
        for k in keys:
            if k in ex and ex[k] is not None:
                return str(ex[k])
        raise KeyError(f"need one of {keys} in {cols}")

    pool: list[TripletRow] = []
    for ex in ds:
        pool.append(
            TripletRow(
                query=_pick(ex, "anchor", "query"),
                positive=_pick(ex, "positive", "positive_passage"),
                negative=_pick(ex, "negative", "negative_passage"),
            )
        )
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:max_samples]


def _encode_texts(tokenizer, texts: list[str], max_length: int, device: torch.device):
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in enc.items()}


def _batch_from_rows(
    rows: list[TripletRow],
    tokenizer,
    batch_size: int,
    Lq: int,
    Ld: int,
    device: torch.device,
) -> list:
    """PyLate ``Contrastive`` batch: ``[query, positive, negative]``."""
    chunk = rows[:batch_size]
    q_texts = [r.query for r in chunk]
    p_texts = [r.positive for r in chunk]
    n_texts = [r.negative for r in chunk]
    return [
        _encode_texts(tokenizer, q_texts, Lq, device),
        _encode_texts(tokenizer, p_texts, Ld, device),
        _encode_texts(tokenizer, n_texts, Ld, device),
    ]


def run_realdata(
    variant: str,
    model_name: str,
    rows: list[TripletRow],
    batch_size: int,
    Lq: int,
    Ld: int,
    steps: int,
    warmup: int,
    device: torch.device,
    recipe: str,
    mini_batch_size: int,
    grad_checkpoint: bool,
) -> Measurement:
    """Train for ``steps`` optimizer steps, cycling real triplets."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    import importlib
    import sys

    for mod in list(sys.modules):
        if mod.startswith("pylate"):
            del sys.modules[mod]
    if variant == "flash":
        os.environ.pop("LIK_DISABLE", None)
        from late_interaction_kernels import patch_pylate

        patch_pylate()
    else:
        os.environ["LIK_DISABLE"] = "1"

    import pylate.losses  # noqa: F401
    import pylate.models

    importlib.reload(pylate.models)
    importlib.reload(pylate.losses)

    from pylate import losses, models
    from pylate.losses import contrastive as _contrastive_mod
    from pylate.scores import colbert_scores as active_cbs

    from late_interaction_kernels.pylate_compat import patched_colbert_scores as _flash_fn

    is_flash = active_cbs is _flash_fn and _contrastive_mod.colbert_scores is _flash_fn
    _log(f"    [{variant}] patch active={is_flash}")
    if variant == "flash" and not is_flash:
        return Measurement(step_ms=float("nan"), peak_mb=float("nan"), err="patch not active")
    if variant == "vanilla" and is_flash:
        return Measurement(step_ms=float("nan"), peak_mb=float("nan"), err="patch leaked")

    try:
        model = models.ColBERT(
            model_name_or_path=model_name,
            document_length=Ld,
            query_length=Lq,
        ).to(device)
    except Exception as e:  # noqa: BLE001
        return Measurement(step_ms=float("nan"), peak_mb=float("nan"), err=f"model load: {e}")

    if grad_checkpoint:
        try:
            transformer = model[0].auto_model
            transformer.gradient_checkpointing_enable()
            if hasattr(transformer.config, "use_cache"):
                transformer.config.use_cache = False
        except Exception as e:  # noqa: BLE001
            return Measurement(step_ms=float("nan"), peak_mb=float("nan"), err=f"grad ckpt: {e}")

    if recipe == "reason":
        loss_fn = losses.CachedContrastive(
            model=model,
            mini_batch_size=mini_batch_size,
            temperature=1.0,
            show_progress_bar=False,
        )
    else:
        loss_fn = losses.Contrastive(model=model)
    tokenizer = model.tokenizer

    if batch_size > len(rows):
        return Measurement(
            step_ms=float("nan"),
            peak_mb=float("nan"),
            err=f"batch_size={batch_size} > dataset={len(rows)}",
        )

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)

    # Cycle windows over the shuffled real triplets so every timed step sees
    # different text (not one batch repeated).
    windows: list[list[TripletRow]] = []
    for start in range(0, len(rows) - batch_size + 1, batch_size):
        windows.append(rows[start : start + batch_size])
    if not windows:
        return Measurement(step_ms=float("nan"), peak_mb=float("nan"), err="no windows")

    step_idx = 0

    def step():
        nonlocal step_idx
        window = windows[step_idx % len(windows)]
        step_idx += 1
        batch = _batch_from_rows(window, tokenizer, batch_size, Lq, Ld, device)
        optim.zero_grad(set_to_none=True)
        with autocast:
            loss_value = loss_fn(batch, labels=None)
        loss_value.backward()
        optim.step()

    try:
        step_ms = _timed_step(step, iters=steps, warmup=warmup)
    except torch.cuda.OutOfMemoryError:
        return Measurement(step_ms=float("nan"), peak_mb=float("nan"), err="OOM")

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    del model, loss_fn, optim
    gc.collect()
    torch.cuda.empty_cache()
    return Measurement(step_ms=step_ms, peak_mb=peak_mb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--dataset", default=DATASET_DEFAULT)
    ap.add_argument("--dataset-config", default=DATASET_CONFIG)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-samples", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--Lq", type=int, default=32)
    ap.add_argument("--Ld", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20, help="timed training steps per variant")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--recipe", choices=["contrastive", "reason"], default="contrastive")
    ap.add_argument("--mini-batch-size", type=int, default=16)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--variants", choices=["both", "vanilla", "flash"], default="both")
    ap.add_argument("--outdir", default="benchmarks/results")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    device = torch.device("cuda:0")
    _log(f"GPU: {torch.cuda.get_device_name()}")
    _log(f"Loading {args.dataset} ({args.dataset_config}) split={args.split}, max={args.max_samples}")
    rows = _load_triplets(args.dataset, args.dataset_config, args.split, args.max_samples, args.seed)
    _log(f"  -> {len(rows)} triplets")
    _log(f"model={args.model}  recipe={args.recipe}  bs={args.batch_size}  Lq={args.Lq}  Ld={args.Ld}")
    _log(f"steps={args.steps}  warmup={args.warmup}  grad_ckpt={args.grad_checkpoint}")
    _log("-" * 72)

    results: dict[str, Measurement] = {}
    variants = ["vanilla", "flash"] if args.variants == "both" else [args.variants]
    for v in variants:
        _log(f"[{v}] running ...")
        m = run_realdata(
            variant=v,
            model_name=args.model,
            rows=rows,
            batch_size=args.batch_size,
            Lq=args.Lq,
            Ld=args.Ld,
            steps=args.steps,
            warmup=args.warmup,
            device=device,
            recipe=args.recipe,
            mini_batch_size=args.mini_batch_size,
            grad_checkpoint=args.grad_checkpoint,
        )
        results[v] = m
        if m.err:
            _log(f"  {v:>8}: FAILED ({m.err})")
        else:
            _log(f"  {v:>8}: {m.step_ms:8.2f} ms/step   peak {m.peak_mb / 1024:5.2f} GB")

    if "vanilla" in results and "flash" in results:
        vr, fr = results["vanilla"], results["flash"]
        if vr.err is None and fr.err is None:
            _log("-" * 72)
            _log(
                f"  speedup:    {vr.step_ms / fr.step_ms:.2f}x  ({vr.step_ms:.2f} -> {fr.step_ms:.2f} ms/step)"
            )
            _log(
                f"  mem delta:  {(vr.peak_mb - fr.peak_mb) / 1024:+.2f} GB  "
                f"({vr.peak_mb / 1024:.2f} -> {fr.peak_mb / 1024:.2f} GB)"
            )

    os.makedirs(args.outdir, exist_ok=True)
    gpu = torch.cuda.get_device_name().replace(" ", "_")
    fn = os.path.join(
        args.outdir,
        f"pylate_realdata_{args.recipe}_{gpu}_bs{args.batch_size}_Lq{args.Lq}_Ld{args.Ld}.json",
    )
    with open(fn, "w") as f:
        json.dump(
            {
                "config": vars(args),
                "n_triplets": len(rows),
                "time_s": time.time(),
                "results": {
                    k: {"step_ms": v.step_ms, "peak_mb": v.peak_mb, "err": v.err} for k, v in results.items()
                },
            },
            f,
            indent=2,
        )
    _log(f"\nwrote {fn}")


if __name__ == "__main__":
    main()
