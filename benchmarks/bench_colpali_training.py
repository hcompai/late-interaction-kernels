"""End-to-end ColQwen2 training-step benchmark via colpali_engine.

Drives a *real* ``colpali_engine.models.ColQwen2`` forward + backward
through one of the three in-batch late-interaction losses, comparing
vanilla colpali_engine against :func:`patch_colpali_engine` on
synthetic image + query batches (no dataset download — see
``bench_colpali_realdata.py`` for the real-data sibling).

Three loss heads are reachable through ``--loss``:

* ``colbert``       — :class:`colpali_engine.loss.late_interaction_losses.ColbertLoss`
* ``pairwise_ce``   — :class:`ColbertPairwiseCELoss`
* ``sigmoid``       — :class:`ColbertSigmoidLoss`

All three materialize the same ``einsum("bnd,csd->bcns") -> amax -> sum``
in-batch similarity tile, which is exactly what
``patch_colpali_engine`` replaces with the fused kernel.

Usage
-----
    python benchmarks/bench_colpali_training.py
    python benchmarks/bench_colpali_training.py --batch-size 16 --loss colbert
    python benchmarks/bench_colpali_training.py --loss pairwise_ce --grad-checkpoint
"""
# ruff: noqa: F821  -- ruff loses ``model`` / ``loss_fn`` / ``optim`` across the
# trailing ``del`` in ``run_one`` and false-positives on the ``step`` closure.

import argparse
import gc
import importlib.util
import json
import os
import random
import string
import time
from pathlib import Path

import torch

# ``benchmarks/`` is not a package; reuse Measurement + _timed_step + _log
# from bench_pylate_lateon by loading it directly.
_LATEON = Path(__file__).resolve().parent / "bench_pylate_lateon.py"
_spec = importlib.util.spec_from_file_location("_bench_pylate_lateon", _LATEON)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
Measurement = _mod.Measurement
_log = _mod._log
_timed_step = _mod._timed_step


MODEL_NAME_DEFAULT = "vidore/colqwen2-v1.0"
LOSS_CHOICES = ("colbert", "pairwise_ce", "sigmoid")


def _random_query(n_words: int) -> str:
    """Pseudo-query of ~``n_words`` random ascii words."""
    vocab = ["".join(random.choices(string.ascii_lowercase, k=random.randint(3, 8))) for _ in range(256)]
    return " ".join(random.choice(vocab) for _ in range(n_words))


def _random_image(size: int):
    """Random RGB ``size x size`` PIL image — exercises the full vision tower."""
    import numpy as np
    from PIL import Image

    arr = (np.random.rand(size, size, 3) * 255).astype("uint8")
    return Image.fromarray(arr, mode="RGB")


def _resolve_loss_cls(kind: str):
    from colpali_engine.loss.late_interaction_losses import (
        ColbertLoss,
        ColbertPairwiseCELoss,
        ColbertSigmoidLoss,
    )

    return {
        "colbert": ColbertLoss,
        "pairwise_ce": ColbertPairwiseCELoss,
        "sigmoid": ColbertSigmoidLoss,
    }[kind]


def _is_patched(loss_cls) -> bool:
    """True iff ``loss_cls.forward`` is the replacement from colpali_compat."""
    from late_interaction_kernels import colpali_compat

    cls_name = loss_cls.__name__
    expected = {
        "ColbertLoss": colpali_compat.patched_colbert_loss_forward,
        "ColbertPairwiseCELoss": colpali_compat.patched_colbert_pairwise_ce_forward,
        "ColbertSigmoidLoss": colpali_compat.patched_colbert_sigmoid_forward,
    }[cls_name]
    return loss_cls.forward is expected


def _apply_lora(model):
    """Make the LoRA adapters the only trainable params (realistic ColPali training).

    ColQwen2 ships with PEFT/LoRA already wired in: ``vidore/colqwen2-v1.0``
    has ``peft_config`` baked into the checkpoint, with LoRA adapters on
    every q/k/v/o/gate/up/down/custom_text_proj. We just need to freeze
    the base and unfreeze the adapters, mirroring what
    ``ColModelTraining`` does on top of a pre-LoRA'd model.

    If the model has no peft_config we fall back to wrapping it with the
    official ColPali LoraConfig (still ~7 M trainable params).

    Either way: encoder backward computes activation gradients only (no
    weight grads through the frozen base), AdamW state is ~30× smaller
    than full fine-tune, and the MaxSim share of step time is realistic.
    """
    if getattr(model, "peft_config", None):
        # Already adapted — just toggle requires_grad to LoRA-only.
        for name, param in model.named_parameters():
            param.requires_grad = "lora_" in name
        return model

    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="FEATURE_EXTRACTION",
        target_modules=(
            r"(.*(model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$"
            r"|.*(custom_text_proj).*$)"
        ),
    )
    return get_peft_model(model, lora_config)


def run_one(
    variant: str,
    model_name: str,
    batch_size: int,
    image_size: int,
    query_words: int,
    loss_kind: str,
    iters: int,
    warmup: int,
    grad_checkpoint: bool,
    full_finetune: bool,
    device: torch.device,
    seed: int = 0,
) -> Measurement:
    """Run ``iters`` training steps and report ms/step + peak GB."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Reseed every variant so vanilla and flash see the exact same synthetic
    # images / queries / param init — otherwise step time wobbles from input
    # variance, not from the patch.
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make sure no prior patch state leaks across variants.
    from late_interaction_kernels import patch_colpali_engine, unpatch_colpali_engine

    unpatch_colpali_engine()
    if variant == "flash":
        os.environ.pop("LIK_DISABLE", None)
        # Importing colpali_engine first lets the patcher find the classes.
        import colpali_engine.loss.late_interaction_losses  # noqa: F401

        patch_colpali_engine()
    else:
        os.environ["LIK_DISABLE"] = "1"

    loss_cls = _resolve_loss_cls(loss_kind)
    is_patched = _is_patched(loss_cls)
    _log(f"    [{variant}] {loss_cls.__name__}.forward patched={is_patched}")
    if variant == "flash" and not is_patched:
        return Measurement(step_ms=float("nan"), peak_gb=float("nan"), err="patch not active")
    if variant == "vanilla" and is_patched:
        return Measurement(step_ms=float("nan"), peak_gb=float("nan"), err="patch leaked")

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
            # All params trainable — what naive backward looks like, and what
            # nobody actually trains for ColPali. Useful as an upper bound only.
            model.requires_grad_(True)
        else:
            # Realistic training mode: only LoRA adapters (~7 M params) update.
            model = _apply_lora(model)
        model.train()
        # Sanity log: confirms the trainable-params count matches expectation
        # (~7 M for LoRA, ~2 B for full fine-tune) so the headline numbers
        # can't silently regress to the wrong regime.
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        _log(f"    [{variant}] trainable params: {trainable / 1e6:.1f}M / {total / 1e6:.1f}M")
    except Exception as e:  # noqa: BLE001
        return Measurement(step_ms=float("nan"), peak_gb=float("nan"), err=f"model load: {e}")

    if grad_checkpoint:
        try:
            model.gradient_checkpointing_enable()
            if hasattr(model, "config") and hasattr(model.config, "use_cache"):
                model.config.use_cache = False
        except Exception as e:  # noqa: BLE001
            return Measurement(step_ms=float("nan"), peak_gb=float("nan"), err=f"grad ckpt: {e}")

    try:
        processor = ColQwen2Processor.from_pretrained(model_name)
    except Exception as e:  # noqa: BLE001
        return Measurement(step_ms=float("nan"), peak_gb=float("nan"), err=f"processor load: {e}")

    loss_fn = loss_cls(temperature=0.02, normalize_scores=True)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    images = [_random_image(image_size) for _ in range(batch_size)]
    queries = [_random_query(query_words) for _ in range(batch_size)]

    try:
        batch_images = processor.process_images(images).to(device)
        batch_queries = processor.process_queries(queries).to(device)
    except Exception as e:  # noqa: BLE001
        return Measurement(step_ms=float("nan"), peak_gb=float("nan"), err=f"process: {e}")

    def step():
        optim.zero_grad(set_to_none=True)
        query_embeddings = model(**batch_queries)
        doc_embeddings = model(**batch_images)
        loss_value = loss_fn(
            query_embeddings=query_embeddings,
            doc_embeddings=doc_embeddings,
        )
        loss_value.backward()
        optim.step()

    try:
        step_ms = _timed_step(step, iters=iters, warmup=warmup)
    except torch.cuda.OutOfMemoryError:
        return Measurement(step_ms=float("nan"), peak_gb=float("nan"), err="OOM")

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3

    del model, loss_fn, optim, batch_images, batch_queries
    gc.collect()
    torch.cuda.empty_cache()
    return Measurement(step_ms=step_ms, peak_gb=peak_gb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_NAME_DEFAULT)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument(
        "--image-size",
        type=int,
        default=448,
        help="square edge of the synthetic PIL image fed to ColQwen2Processor",
    )
    ap.add_argument(
        "--query-words",
        type=int,
        default=16,
        help="number of random ascii words per query (controls Lq after tokenization)",
    )
    ap.add_argument(
        "--loss",
        choices=LOSS_CHOICES,
        default="colbert",
        help="which in-batch late-interaction loss to drive",
    )
    ap.add_argument("--iters", type=int, default=10)
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--only",
        choices=["both", "vanilla", "flash"],
        default="both",
        help="run a subset of variants (useful when vanilla OOMs)",
    )
    ap.add_argument("--outdir", default="benchmarks/results")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    device = torch.device("cuda:0")
    _log(f"GPU: {torch.cuda.get_device_name()}")
    _log(f"model={args.model}  loss={args.loss}")
    train_mode = "full-finetune" if args.full_finetune else "lora"
    _log(
        f"bs={args.batch_size}  image_size={args.image_size}  "
        f"query_words={args.query_words}  grad_ckpt={args.grad_checkpoint}  "
        f"train_mode={train_mode}"
    )
    _log(f"iters={args.iters}  warmup={args.warmup}  seed={args.seed}")
    _log("-" * 72)

    results: dict[str, Measurement] = {}
    variants = ["vanilla", "flash"] if args.only == "both" else [args.only]
    for v in variants:
        _log(f"[{v}] running ...")
        m = run_one(
            variant=v,
            model_name=args.model,
            batch_size=args.batch_size,
            image_size=args.image_size,
            query_words=args.query_words,
            loss_kind=args.loss,
            iters=args.iters,
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
            _log(f"  {v:>8}: {m.step_ms:8.2f} ms/step   peak {m.peak_gb:5.2f} GB")

    if "vanilla" in results and "flash" in results:
        vr, fr = results["vanilla"], results["flash"]
        if vr.err is None and fr.err is None:
            _log("-" * 72)
            _log(
                f"  speedup:    {vr.step_ms / fr.step_ms:.2f}x  "
                f"({vr.step_ms:.2f} -> {fr.step_ms:.2f} ms/step)"
            )
            _log(
                f"  mem delta:  {vr.peak_gb - fr.peak_gb:+.2f} GB  ({vr.peak_gb:.2f} -> {fr.peak_gb:.2f} GB)"
            )

    os.makedirs(args.outdir, exist_ok=True)
    gpu = torch.cuda.get_device_name().replace(" ", "_")
    ckpt_tag = "_ckpt" if args.grad_checkpoint else ""
    mode_tag = "_full" if args.full_finetune else "_lora"
    fn = os.path.join(
        args.outdir,
        f"colpali_{args.loss}_{gpu}_bs{args.batch_size}_img{args.image_size}{mode_tag}{ckpt_tag}.json",
    )
    with open(fn, "w") as f:
        json.dump(
            {
                "config": vars(args),
                "time_s": time.time(),
                "results": {
                    k: {"step_ms": v.step_ms, "peak_gb": v.peak_gb, "err": v.err} for k, v in results.items()
                },
            },
            f,
            indent=2,
        )
    _log(f"\nwrote {fn}")


if __name__ == "__main__":
    main()
