"""End-to-end PyLate training-step benchmark on the LateOn family.

Drives a *real* PyLate ``models.ColBERT`` forward + (``Contrastive`` or
``CachedContrastive``) backward on the full LateOn line-up:

  * ``lightonai/LateOn`` (default) — SOTA ColBERT on BEIR, 149 M params
  * ``lightonai/LateOn-Code`` — same backbone, long-context code retrieval
  * ``lightonai/LateOn-Code-edge`` — 17 M edge variant, d=48
  * ``lightonai/GTE-ModernColBERT-v1`` — predecessor, same ModernBERT-base

Measures:

  * step time (ms)
  * peak GPU memory (GB)
  * (optionally) DDP across all visible GPUs
  * (optionally) gradient checkpointing on the encoder

Two recipes are baked in:

``--recipe contrastive``  (default)
    Plain ``losses.Contrastive``. The simplest PyLate loss. Transformer
    forward dominates the step here, so late-interaction-kernels is ~1×
    end-to-end. Useful as a baseline.

``--recipe reason``
    Mirrors LightOn's own cached-contrastive training recipe:
    ``CachedContrastive``, ``batch_size=256``, ``mini_batch_size=32``,
    ``Lq=128``, ``Ld=8192``, grad-checkpointing on. PyLate's
    ``CachedContrastive`` manually chunks MaxSim into ``(bs/mini)**2``
    Python-level calls to avoid OOM — that is exactly what
    late-interaction-kernels lets you skip.

Usage
-----
    # Simple contrastive step on one GPU (default model: lightonai/LateOn)
    python benchmarks/bench_pylate_lateon.py --recipe contrastive \\
        --batch-size 4 --Ld 8192

    # LateOn-Code long-context training, one GPU
    python benchmarks/bench_pylate_lateon.py --recipe reason \\
        --model lightonai/LateOn-Code --batch-size 64 \\
        --mini-batch-size 32 --Ld 8192 --grad-checkpoint

    # DDP 8× H100 at the full recipe
    torchrun --standalone --nproc_per_node=8 \\
        benchmarks/bench_pylate_lateon.py --recipe reason \\
        --batch-size 64 --mini-batch-size 32 --Ld 8192 --grad-checkpoint --ddp
"""
# ruff: noqa: F821  -- closures over `loss_fn` / `optim` confuse ruff's scope analysis

import argparse
import gc
import json
import os
import random
import string
import time

import torch
from _bench_common import (
    Measurement,
    _init_ddp,
    _is_dist,
    _log,
    _rank,
    _timed_step,
    _world_size,
)

MODEL_NAME_DEFAULT = "lightonai/LateOn"


# --------------------------------------------------------------------------- #
# Fake training batch                                                          #
# --------------------------------------------------------------------------- #


def _random_text(n_words: int) -> str:
    """Pseudo-text of ~n_words tokens (roughly; the tokenizer decides)."""
    vocab = ["".join(random.choices(string.ascii_lowercase, k=random.randint(3, 8))) for _ in range(512)]
    return " ".join(random.choice(vocab) for _ in range(n_words))


def _build_batch(model, tokenizer, batch_size: int, Lq: int, Ld: int, device: torch.device):
    """Return a PyLate-style batch: one query set + positive docs + negative docs.

    We build it by tokenizing fake text to the exact lengths we want, so
    `Lq`/`Ld` control the actual sequence length (not an upper bound).
    """
    # Roughly 3 chars / token for random ascii → pad to ``Lq``/``Ld`` exact.
    q_texts = [_random_text(Lq) for _ in range(batch_size)]
    pos_texts = [_random_text(Ld) for _ in range(batch_size)]
    neg_texts = [_random_text(Ld) for _ in range(batch_size)]

    def encode(texts, length):
        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=length,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in enc.items()}

    q = encode(q_texts, Lq)
    p = encode(pos_texts, Ld)
    n = encode(neg_texts, Ld)
    return [q, p, n]  # PyLate `Contrastive` expects [query, positive, negative1, ...]


# --------------------------------------------------------------------------- #
# Benchmark core                                                              #
# --------------------------------------------------------------------------- #


def run_one(
    variant: str,  # "vanilla" | "flash"
    model_name: str,
    batch_size: int,
    Lq: int,
    Ld: int,
    iters: int,
    warmup: int,
    device: torch.device,
    use_ddp: bool,
    recipe: str = "contrastive",  # "contrastive" | "reason"
    mini_batch_size: int = 32,
    grad_checkpoint: bool = False,
) -> Measurement:
    """Run ``iters`` training steps of the selected loss and return timing + peak mem."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Always start from a clean module table — patch state must NOT leak between
    # variants. We re-import pylate after (un)patching so the loss module picks
    # up the right ``colbert_scores``.
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
        # Force the vanilla code path even if patch_pylate was called earlier.
        os.environ["LIK_DISABLE"] = "1"
        import late_interaction_kernels.pylate_compat as _pc  # noqa: F401

    import pylate.losses  # noqa: F401
    import pylate.models

    importlib.reload(pylate.models)
    importlib.reload(pylate.losses)

    from pylate import losses, models
    from pylate.losses import contrastive as _contrastive_mod
    from pylate.scores import colbert_scores as active_cbs

    # Sanity check that the variant actually bound the right function everywhere.
    from late_interaction_kernels.pylate_compat import patched_colbert_scores as _flash_fn

    is_flash_top = active_cbs is _flash_fn
    is_flash_loss = _contrastive_mod.colbert_scores is _flash_fn
    _log(
        f"    [{variant}] pylate.scores.colbert_scores is flash={is_flash_top}  "
        f"pylate.losses.contrastive.colbert_scores is flash={is_flash_loss}"
    )
    if variant == "flash" and not (is_flash_top and is_flash_loss):
        return Measurement(step_ms=float("nan"), peak_mb=float("nan"), err="patch not active")
    if variant == "vanilla" and (is_flash_top or is_flash_loss):
        return Measurement(step_ms=float("nan"), peak_mb=float("nan"), err="patch leaked into vanilla")

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
            # pylate.models.ColBERT wraps a transformers model; enable checkpointing on it.
            transformer = model[0].auto_model  # ST module 0 is the Transformer wrapper
            transformer.gradient_checkpointing_enable()
            # Some models (e.g. ModernBert) don't expose `use_cache`; only set it when present.
            if hasattr(transformer.config, "use_cache"):
                transformer.config.use_cache = False
            assert transformer.is_gradient_checkpointing, "gradient checkpointing did not enable"
        except Exception as e:  # noqa: BLE001
            return Measurement(step_ms=float("nan"), peak_mb=float("nan"), err=f"grad ckpt: {e}")

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index],
            find_unused_parameters=False,
            static_graph=not grad_checkpoint,
        )

    inner_model = model.module if use_ddp else model
    if recipe == "reason":
        loss_fn = losses.CachedContrastive(
            model=inner_model,
            mini_batch_size=mini_batch_size,
            gather_across_devices=use_ddp,
            temperature=1.0,
            show_progress_bar=False,
        )
    else:
        loss_fn = losses.Contrastive(model=inner_model)
    tokenizer = inner_model.tokenizer

    try:
        batch = _build_batch(model, tokenizer, batch_size, Lq, Ld, device)
    except Exception as e:  # noqa: BLE001
        return Measurement(step_ms=float("nan"), peak_mb=float("nan"), err=f"build batch: {e}")

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # Match LightOn's training: bf16=True in SentenceTransformerTrainingArguments runs the
    # forward under bf16 autocast. Without it we'd pay 4× activation memory.
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)

    def step():
        optim.zero_grad(set_to_none=True)
        with autocast:
            loss_value = loss_fn(batch, labels=None)
        loss_value.backward()
        optim.step()

    try:
        step_ms = _timed_step(step, iters=iters, warmup=warmup)
    except torch.cuda.OutOfMemoryError:
        return Measurement(step_ms=float("nan"), peak_mb=float("nan"), err="OOM")

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2

    del model, loss_fn, optim, batch
    gc.collect()
    torch.cuda.empty_cache()

    return Measurement(step_ms=step_ms, peak_mb=peak_mb)


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_NAME_DEFAULT)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--Lq", type=int, default=32)
    ap.add_argument("--Ld", type=int, default=8192)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument(
        "--recipe",
        choices=["contrastive", "reason"],
        default="contrastive",
        help="'contrastive' = plain losses.Contrastive, 'reason' = LightOn's cached-contrastive recipe (CachedContrastive)",
    )
    ap.add_argument(
        "--mini-batch-size",
        type=int,
        default=32,
        help="CachedContrastive mini_batch_size (only for --recipe reason)",
    )
    ap.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="enable gradient checkpointing on the ModernBERT encoder",
    )
    ap.add_argument("--ddp", action="store_true", help="initialize torch.distributed (torchrun required)")
    ap.add_argument(
        "--variants",
        choices=["both", "vanilla", "flash"],
        default="both",
        help="run a subset of variants (useful when vanilla OOMs)",
    )
    ap.add_argument("--outdir", default="benchmarks/results")
    args = ap.parse_args()

    if args.ddp:
        device = _init_ddp()
    else:
        device = torch.device("cuda:0")

    world = _world_size()
    total_batch = args.batch_size * world

    _log(f"device={device}  world={world}  per_rank_batch={args.batch_size}  total_batch={total_batch}")
    _log(f"recipe={args.recipe}  mini_bs={args.mini_batch_size}  grad_ckpt={args.grad_checkpoint}")
    _log(f"shape: Lq={args.Lq}  Ld={args.Ld}  model={args.model}")
    _log(f"iters={args.iters}  warmup={args.warmup}")
    _log("-" * 70)

    results = {}
    variants = ["vanilla", "flash"] if args.variants == "both" else [args.variants]
    for v in variants:
        _log(f"[{v}] running ...")
        m = run_one(
            variant=v,
            model_name=args.model,
            batch_size=args.batch_size,
            Lq=args.Lq,
            Ld=args.Ld,
            iters=args.iters,
            warmup=args.warmup,
            device=device,
            use_ddp=args.ddp,
            recipe=args.recipe,
            mini_batch_size=args.mini_batch_size,
            grad_checkpoint=args.grad_checkpoint,
        )
        results[v] = m
        tag = f"{v:>8}"
        if m.err:
            _log(f"  {tag}: FAILED ({m.err})")
        else:
            _log(f"  {tag}: {m.step_ms:8.2f} ms/step   peak {m.peak_mb / 1024:5.2f} GB")

    if "vanilla" in results and "flash" in results and all(r.err is None for r in results.values()):
        vr, fr = results["vanilla"], results["flash"]
        speed_ratio = vr.step_ms / fr.step_ms
        mem_delta_gb = (vr.peak_mb - fr.peak_mb) / 1024
        _log("-" * 70)
        _log(f"  speedup:     {speed_ratio:.2f}x  ({vr.step_ms:.2f} -> {fr.step_ms:.2f} ms)")
        _log(
            f"  memory won:  {mem_delta_gb:+.2f} GB  ({vr.peak_mb / 1024:.2f} -> {fr.peak_mb / 1024:.2f} GB)"
        )

    if _rank() == 0:
        os.makedirs(args.outdir, exist_ok=True)
        gpu = torch.cuda.get_device_name().replace(" ", "_")
        ckpt_tag = "_ckpt" if args.grad_checkpoint else ""
        fn = os.path.join(
            args.outdir,
            f"pylate_{args.recipe}_{gpu}_bs{args.batch_size}_mini{args.mini_batch_size}_Lq{args.Lq}_Ld{args.Ld}_ws{world}{ckpt_tag}.json",
        )
        payload = {
            "config": vars(args),
            "world_size": world,
            "total_batch_size": total_batch,
            "time_s": time.time(),
            "results": {
                k: {"step_ms": v.step_ms, "peak_mb": v.peak_mb, "err": v.err} for k, v in results.items()
            },
        }
        with open(fn, "w") as f:
            json.dump(payload, f, indent=2)
        _log(f"\nwrote {fn}")

    if _is_dist():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
