"""End-to-end ColQwen2 training-step benchmark on *real* vidore data.

Loads ``vidore/docvqa_test_subsampled`` (the small public DocVQA subset
used in the ViDoRe leaderboard's smoke tests — a few hundred real
document-image / query pairs), runs the colpali_engine
``VisualRetrieverCollator`` to build proper batches, and times full
``ColQwen2`` encoder + ``ColbertLoss`` (or pairwise CE / sigmoid) steps
with and without :func:`patch_colpali_engine`.

This is the closest a single GPU job can get to a real ColQwen2 fine-
tune without paying the full training-set download: real document
images, real OCR-targeted queries, real attention/vision tower work
on the backward pass.

Usage
-----
    python benchmarks/bench_colpali_realdata.py
    python benchmarks/bench_colpali_realdata.py --batch-size 8 --loss colbert
    python benchmarks/bench_colpali_realdata.py --loss pairwise_ce --grad-checkpoint

Requires: ``pip install colpali-engine datasets``.
"""
# ruff: noqa: F821  -- ruff loses ``model`` / ``loss_fn`` / ``optim`` across the
# trailing ``del`` in ``run_realdata`` and false-positives on the ``step`` closure.

import argparse
import gc
import json
import os
import random
import time
from dataclasses import dataclass

import torch
from _bench_common import Measurement, _log, _timed_step

# Loss resolution + patch detection + LoRA setup live in the synthetic bench.
from bench_colpali_training import _apply_lora, _is_patched, _resolve_loss_cls

MODEL_DEFAULT = "vidore/colqwen2-v1.0"
DATASET_DEFAULT = "vidore/docvqa_test_subsampled"


@dataclass(frozen=True)
class Sample:
    query: str
    image: object  # PIL.Image — typed loosely so we don't import PIL at module load


def _load_samples(dataset: str, split: str, max_samples: int, seed: int) -> list[Sample]:
    from datasets import load_dataset

    n_fetch = max_samples + 32
    ds = load_dataset(dataset, split=f"{split}[:{n_fetch}]")
    cols = set(ds.column_names)
    _log(f"  dataset columns: {sorted(cols)}")

    def _pick(ex: dict, *keys: str):
        for k in keys:
            if k in ex and ex[k] is not None:
                return ex[k]
        raise KeyError(f"need one of {keys} in {cols}")

    pool: list[Sample] = []
    for ex in ds:
        pool.append(
            Sample(
                query=str(_pick(ex, "query", "question")),
                image=_pick(ex, "image"),
            )
        )
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:max_samples]


def _to_collator_samples(rows: list[Sample]) -> list[dict]:
    """colpali_engine's VisualRetrieverCollator expects a list of dicts."""
    return [{"query": r.query, "pos_target": [r.image], "neg_target": None} for r in rows]


def _split_collated(batch: dict, device: torch.device) -> tuple[dict, dict]:
    """Split the collator's ``query_*`` / ``doc_*`` keys into model kwargs."""
    query_kwargs: dict = {}
    doc_kwargs: dict = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            v = v.to(device)
        if k.startswith("query_"):
            query_kwargs[k.removeprefix("query_")] = v
        elif k.startswith("doc_"):
            doc_kwargs[k.removeprefix("doc_")] = v
        # neg_target_* keys (only present with hard negatives) are ignored.
    return query_kwargs, doc_kwargs


def run_realdata(
    variant: str,
    model_name: str,
    rows: list[Sample],
    batch_size: int,
    loss_kind: str,
    steps: int,
    warmup: int,
    grad_checkpoint: bool,
    full_finetune: bool,
    device: torch.device,
    seed: int = 0,
) -> Measurement:
    """Train for ``steps`` optimizer steps, cycling over real DocVQA pairs."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Reseed every variant so vanilla and lik see identical param init.
    # Batch contents are deterministic (real DocVQA windows), so once init
    # is fixed both variants run on the same data.
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    from late_interaction_kernels import patch_colpali_engine, unpatch_colpali_engine

    unpatch_colpali_engine()
    if variant == "lik":
        os.environ.pop("LIK_DISABLE", None)
        import colpali_engine.loss.late_interaction_losses  # noqa: F401

        patch_colpali_engine()
    else:
        os.environ["LIK_DISABLE"] = "1"

    loss_cls = _resolve_loss_cls(loss_kind)
    is_patched = _is_patched(loss_cls)
    _log(f"    [{variant}] {loss_cls.__name__}.forward patched={is_patched}")
    if variant == "lik" and not is_patched:
        return Measurement(step_ms=float("nan"), peak_mb=float("nan"), err="patch not active")
    if variant == "vanilla" and is_patched:
        return Measurement(step_ms=float("nan"), peak_mb=float("nan"), err="patch leaked")

    from colpali_engine.collators import VisualRetrieverCollator
    from colpali_engine.models import ColQwen2, ColQwen2Processor

    try:
        from transformers.utils.import_utils import is_flash_attn_2_available

        attn_impl = "flash_attention_2" if is_flash_attn_2_available() else None
        model = ColQwen2.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        ).to(device)
        if full_finetune:
            model.requires_grad_(True)
        else:
            model = _apply_lora(model)
        model.train()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        _log(f"    [{variant}] trainable params: {trainable / 1e6:.1f}M / {total / 1e6:.1f}M")
    except Exception as e:  # noqa: BLE001
        return Measurement(step_ms=float("nan"), peak_mb=float("nan"), err=f"model load: {e}")

    if grad_checkpoint:
        try:
            model.gradient_checkpointing_enable()
            if hasattr(model, "config") and hasattr(model.config, "use_cache"):
                model.config.use_cache = False
        except Exception as e:  # noqa: BLE001
            return Measurement(step_ms=float("nan"), peak_mb=float("nan"), err=f"grad ckpt: {e}")

    try:
        processor = ColQwen2Processor.from_pretrained(model_name)
    except Exception as e:  # noqa: BLE001
        return Measurement(step_ms=float("nan"), peak_mb=float("nan"), err=f"processor load: {e}")

    collator = VisualRetrieverCollator(processor=processor)
    loss_fn = loss_cls(temperature=0.02, normalize_scores=True)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    if batch_size > len(rows):
        return Measurement(
            step_ms=float("nan"),
            peak_mb=float("nan"),
            err=f"batch_size={batch_size} > dataset={len(rows)}",
        )

    # Pre-build sliding windows so every timed step sees a different batch.
    windows: list[list[Sample]] = []
    for start in range(0, len(rows) - batch_size + 1, batch_size):
        windows.append(rows[start : start + batch_size])
    if not windows:
        return Measurement(step_ms=float("nan"), peak_mb=float("nan"), err="no windows")

    step_idx = 0

    def step():
        nonlocal step_idx
        window = windows[step_idx % len(windows)]
        step_idx += 1
        batch = collator(_to_collator_samples(window))
        query_kwargs, doc_kwargs = _split_collated(batch, device)
        optim.zero_grad(set_to_none=True)
        query_embeddings = model(**query_kwargs)
        doc_embeddings = model(**doc_kwargs)
        loss_value = loss_fn(
            query_embeddings=query_embeddings,
            doc_embeddings=doc_embeddings,
        )
        loss_value.backward()
        optim.step()

    try:
        step_ms = _timed_step(step, iters=steps, warmup=warmup)
    except torch.cuda.OutOfMemoryError:
        return Measurement(step_ms=float("nan"), peak_mb=float("nan"), err="OOM")

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2

    del model, loss_fn, optim, collator
    gc.collect()
    torch.cuda.empty_cache()
    return Measurement(step_ms=step_ms, peak_mb=peak_mb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--dataset", default=DATASET_DEFAULT)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-samples", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--loss", choices=("colbert", "pairwise_ce", "sigmoid"), default="colbert")
    ap.add_argument("--steps", type=int, default=10, help="timed optimizer steps per variant")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument(
        "--full-finetune",
        action="store_true",
        help=(
            "make every parameter trainable (no LoRA). Default is LoRA-only, "
            "which matches the real ColPali training recipe; --full-finetune "
            "is for measuring an upper-bound encoder-dominated regime."
        ),
    )
    ap.add_argument("--variants", choices=["both", "vanilla", "lik"], default="both")
    ap.add_argument("--outdir", default="benchmarks/results")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    device = torch.device("cuda:0")
    _log(f"GPU: {torch.cuda.get_device_name()}")
    _log(f"Loading {args.dataset} split={args.split}, max={args.max_samples}")
    rows = _load_samples(args.dataset, args.split, args.max_samples, args.seed)
    _log(f"  -> {len(rows)} samples")
    train_mode = "full-finetune" if args.full_finetune else "lora"
    _log(
        f"model={args.model}  loss={args.loss}  bs={args.batch_size}  "
        f"grad_ckpt={args.grad_checkpoint}  train_mode={train_mode}"
    )
    _log(f"steps={args.steps}  warmup={args.warmup}  seed={args.seed}")
    _log("-" * 72)

    results: dict[str, Measurement] = {}
    variants = ["vanilla", "lik"] if args.variants == "both" else [args.variants]
    for v in variants:
        _log(f"[{v}] running ...")
        m = run_realdata(
            variant=v,
            model_name=args.model,
            rows=rows,
            batch_size=args.batch_size,
            loss_kind=args.loss,
            steps=args.steps,
            warmup=args.warmup,
            grad_checkpoint=args.grad_checkpoint,
            full_finetune=args.full_finetune,
            device=device,
            seed=args.seed,
        )
        results[v] = m
        if m.err:
            _log(f"  {v:>8}: FAILED ({m.err})")
        else:
            _log(f"  {v:>8}: {m.step_ms:8.2f} ms/step   peak {m.peak_mb / 1024:5.2f} GB")

    if "vanilla" in results and "lik" in results:
        vr, fr = results["vanilla"], results["lik"]
        if vr.err is None and fr.err is None:
            _log("-" * 72)
            _log(
                f"  speedup:    {vr.step_ms / fr.step_ms:.2f}x  "
                f"({vr.step_ms:.2f} -> {fr.step_ms:.2f} ms/step)"
            )
            _log(
                f"  mem delta:  {(vr.peak_mb - fr.peak_mb) / 1024:+.2f} GB  "
                f"({vr.peak_mb / 1024:.2f} -> {fr.peak_mb / 1024:.2f} GB)"
            )

    os.makedirs(args.outdir, exist_ok=True)
    gpu = torch.cuda.get_device_name().replace(" ", "_")
    ckpt_tag = "_ckpt" if args.grad_checkpoint else ""
    mode_tag = "_full" if args.full_finetune else "_lora"
    fn = os.path.join(
        args.outdir,
        f"colpali_realdata_{args.loss}_{gpu}_bs{args.batch_size}{mode_tag}{ckpt_tag}.json",
    )
    with open(fn, "w") as f:
        json.dump(
            {
                "config": vars(args),
                "n_samples": len(rows),
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
