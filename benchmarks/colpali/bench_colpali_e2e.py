"""Run a few real ColQwen2 training steps and record the VRAM used by every MaxSim call.

Adapted from the harness in colpali PR #412 (`bench_lik/bench_train.py` at commit
``0f289e4``). Recent colpali-engine ships that PR's native
``COLPALI_SCORES_BACKEND`` dispatcher; here vanilla vs LIK is toggled with
``--variant`` (``lik`` sets ``COLPALI_SCORES_BACKEND=lik`` and calls
:func:`patch_colpali_engine`, which covers releases without native support and is
a no-op on the native build) and the instrumentation wraps the loss head's
``forward`` instead of a ``maxsim_inbatch`` dispatcher. Each wrapped call records:

- ``forward_transient_peak_mib``: extra memory while the loss forward runs, freed after.
- ``saved_for_backward_mib``: memory held from forward until backward (vanilla keeps the
  ``[B, B, Lq, Ld]`` score tensor; LIK keeps the ``[B, B]`` output).

The in-train bracket is loss-level, so it includes the CE term's ``O(B^2)`` tensors on
top of the bare MaxSim op — the score grid dominates, but the *exact* op numbers come
from the isolated replay. If the loss itself OOMs, that is recorded too: it pins the OOM
inside the score computation rather than the model.

The op's backward cannot be bracketed in-train: a tensor grad hook is a pre-hook on the
*producing* node, so a "close bracket" on ``query`` only fires when the query tower's
backward is scheduled — after the whole doc-tower backward ran. Instead, each recorded
(shape, dtype) is replayed after training on fresh random embeddings whose graph contains
only the MaxSim op (vanilla: the exact ``einsum + amax + sum`` from
``ColbertPairwiseCELoss``; lik: the fused kernel), where forward/saved/backward peaks
bracket exactly.

A whole-run OOM is an expected sweep outcome: it is recorded in the JSON and the process
exits 0 so the sweep driver keeps going (see ``scripts/sky_colpali_e2e.yaml``).

Usage:
    python benchmarks/colpali/bench_colpali_e2e.py --variant vanilla \\
        --batch-size 64 --max-steps 4 \\
        --output benchmarks/results/colpali_e2e_b64_vanilla.json
"""

import argparse
import json
import math
import os
import time
from collections.abc import Callable
from pathlib import Path

import torch

LossForwardFn = Callable[..., torch.Tensor]
MaxsimFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

# Verbatim from colpali PR #412's bench config (ColQwen2 + LoRA r=32 recipe).
_LORA_TARGET_MODULES = (
    r"(.*(model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)"
)


def _vanilla_maxsim(query: torch.Tensor, doc: torch.Tensor) -> torch.Tensor:
    """The exact in-batch MaxSim from colpali-engine v0.3.16's ``ColbertPairwiseCELoss``."""
    return torch.einsum("bnd,csd->bcns", query, doc).amax(dim=3).sum(dim=2)


def _lik_maxsim(query: torch.Tensor, doc: torch.Tensor) -> torch.Tensor:
    """The fused-kernel in-batch MaxSim that ``patch_colpali_engine()`` routes to."""
    from late_interaction_kernels.autograd import maxsim

    return maxsim(query, doc)


def _lik_version() -> str:
    import late_interaction_kernels as lik

    return getattr(lik, "__version__", "unknown")


def _load_colpali_train_subset(n_samples: int):
    """Load a subset of ``vidore/colpali_train_set`` from just enough parquet shards.

    The full train set is 82 shards (~52 GB, ~1442 rows each). Shape / throughput / memory
    benchmarks only need a few thousand examples, so this downloads just the first few shards
    that cover ``n_samples`` rows (via ``data_files``) and keeps that many — never the full
    set. ``verification_mode="no_checks"`` skips the split-completeness check (the repo
    declares ``train``+``test``, but we deliberately load only part of ``train``).
    """
    from colpali_engine.data.dataset import ColPaliEngineDataset
    from datasets import load_dataset

    rows_per_shard = 1442  # approximate; +1 shard of headroom guards against variance
    num_shards = min(82, math.ceil(n_samples / rows_per_shard) + 1)
    data_files = [f"data/train-{i:05d}-of-00082.parquet" for i in range(num_shards)]
    dataset = load_dataset(
        "vidore/colpali_train_set",
        data_files=data_files,
        split="train",
        verification_mode="no_checks",
    )
    if n_samples < len(dataset):
        dataset = dataset.select(range(n_samples))

    return ColPaliEngineDataset(dataset, pos_target_column_name="image")


def _build_training_config(model_name: str, batch_size: int, max_steps: int, n_samples: int):
    """ColQwen2 + LoRA r=32 + ColbertPairwiseCELoss + grad-checkpointing + bf16 — the
    recipe from colpali PR #412's batch sweep (``bench_config_subset.yaml``), built
    programmatically against released colpali-engine."""
    from colpali_engine.loss.late_interaction_losses import ColbertPairwiseCELoss
    from colpali_engine.models import ColQwen2, ColQwen2Processor
    from colpali_engine.trainer.colmodel_training import ColModelTrainingConfig
    from peft import LoraConfig
    from transformers import TrainingArguments

    output_dir = "/tmp/colpali_e2e_trainer_output"
    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        warmup_steps=0,
        learning_rate=5e-5,
        report_to="none",
        bf16=True,
        seed=42,
        data_seed=42,
    )
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.1,
        init_lora_weights="gaussian",
        bias="none",
        task_type="FEATURE_EXTRACTION",
        target_modules=_LORA_TARGET_MODULES,
    )
    # ColModelTrainingConfig.__post_init__ applies the LoRA adapter to the model.
    return ColModelTrainingConfig(
        model=ColQwen2.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="sdpa"),
        processor=ColQwen2Processor.from_pretrained(model_name, max_num_visual_tokens=768),
        train_dataset=_load_colpali_train_subset(n_samples),
        tr_args=training_args,
        output_dir=output_dir,
        run_eval=False,
        loss_func=ColbertPairwiseCELoss(),
        peft_config=lora_config,
    )


def _make_fixed_trainer_cls():
    """A ``ContrastiveTrainer`` subclass fixing two v0.3.16 bugs under transformers 5.x.

    Kept for the pinned 0.3.16 bench env; on colpali-engine with native LIK the
    ``COLPALI_SCORES_BACKEND`` dispatch supersedes the LIK-side patches (which
    retire to a deprecated no-op).
    """
    from colpali_engine.trainer.contrastive_trainer import ContrastiveTrainer
    from transformers import Trainer

    class FixedContrastiveTrainer(ContrastiveTrainer):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            # v0.3.16 only sets these in the multi-dataset branch of
            # get_train_dataloader, but compute_loss reads them on every step.
            collator = kwargs.get("data_collator")
            self.query_prefix: str = getattr(collator, "query_prefix", "query_")
            self.pos_prefix: str = getattr(collator, "pos_doc_prefix", "doc_")
            self.neg_prefix: str = getattr(collator, "neg_doc_prefix", "neg_doc_")

        def _get_train_sampler(self, dataset=None):
            # transformers 5.x passes the dataset positionally; v0.3.16's override
            # accepts no argument and would raise TypeError, so bypass it.
            if dataset is None:
                return super()._get_train_sampler()
            return Trainer._get_train_sampler(self, dataset)

    return FixedContrastiveTrainer


def _make_step_timer():
    """A ``TrainerCallback`` recording wall time for each optimizer step."""
    from transformers import TrainerCallback

    class StepTimerCallback(TrainerCallback):
        def __init__(self) -> None:
            self.step_times: list[float] = []
            self._step_start: float | None = None

        def on_step_begin(self, args, state, control, **kwargs) -> None:
            torch.cuda.synchronize()
            self._step_start = time.perf_counter()

        def on_step_end(self, args, state, control, **kwargs) -> None:
            torch.cuda.synchronize()
            if self._step_start is not None:
                self.step_times.append(time.perf_counter() - self._step_start)
                self._step_start = None

    return StepTimerCallback()


def _replay_op_isolated(fn: MaxsimFn, record: dict) -> dict:
    """Replay one recorded call on fresh random embeddings whose graph contains only the op,
    so the forward/saved/backward peaks bracket exactly. Returns MiB deltas, or an OOM marker."""
    dtype = getattr(torch, record["dtype"].removeprefix("torch."))
    query = torch.randn(record["query_shape"], dtype=dtype, device="cuda", requires_grad=True)
    doc = torch.randn(record["doc_shape"], dtype=dtype, device="cuda", requires_grad=True)

    torch.cuda.synchronize()
    before_forward_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    try:
        out = fn(query, doc)
        torch.cuda.synchronize()
        forward_peak_mib = (torch.cuda.max_memory_allocated() - before_forward_bytes) / 2**20
        saved_mib = (torch.cuda.memory_allocated() - before_forward_bytes) / 2**20

        before_backward_bytes = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        out.sum().backward()
        torch.cuda.synchronize()
        backward_peak_mib = (torch.cuda.max_memory_allocated() - before_backward_bytes) / 2**20
    except torch.cuda.OutOfMemoryError:
        return {**record, "oom_in_replay": True}

    return {
        **record,
        "forward_transient_peak_mib": forward_peak_mib,
        "saved_for_backward_mib": saved_mib,
        "backward_transient_peak_mib": backward_peak_mib,
    }


def _summarize_calls(calls: list[dict]) -> dict:
    """Max per-call numbers — memory is shape-deterministic, so max is the story."""
    metric_names = ["forward_transient_peak_mib", "saved_for_backward_mib", "backward_transient_peak_mib"]
    summary: dict = {
        "num_calls": len(calls),
        "num_oom": sum(1 for call in calls if call.get("oom_in_forward") or call.get("oom_in_replay")),
    }
    for metric in metric_names:
        summary[f"max_{metric}"] = max((call[metric] for call in calls if metric in call), default=None)
    return summary


class MaxsimVramRecorder:
    """Per-call VRAM for the loss head, plus a run-level peak that survives the per-op stat resets."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.run_peak_alloc_bytes: int = 0
        self.run_peak_reserved_bytes: int = 0

    def fold_run_peak(self) -> None:
        """Capture the global peak before a reset wipes it; call once more after training."""
        self.run_peak_alloc_bytes = max(self.run_peak_alloc_bytes, torch.cuda.max_memory_allocated())
        self.run_peak_reserved_bytes = max(self.run_peak_reserved_bytes, torch.cuda.max_memory_reserved())

    def wrap(self, loss_forward: LossForwardFn, op_name: str) -> LossForwardFn:
        # Parameter names must match the loss's forward: the trainer calls it with
        # keyword arguments (query_embeddings=..., doc_embeddings=..., offset=...).
        def wrapped(
            query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor, offset: int = 0
        ) -> torch.Tensor:
            torch.cuda.synchronize()
            self.fold_run_peak()
            before_forward_bytes = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            record: dict = {
                "op": op_name,
                "query_shape": list(query_embeddings.shape),
                "doc_shape": list(doc_embeddings.shape),
                "dtype": str(query_embeddings.dtype),
            }
            try:
                out = loss_forward(query_embeddings, doc_embeddings, offset)
            except torch.cuda.OutOfMemoryError:
                record["oom_in_forward"] = True
                self.calls.append(record)
                raise
            torch.cuda.synchronize()
            record["forward_transient_peak_mib"] = (
                torch.cuda.max_memory_allocated() - before_forward_bytes
            ) / 2**20
            record["saved_for_backward_mib"] = (torch.cuda.memory_allocated() - before_forward_bytes) / 2**20
            self.calls.append(record)
            return out

        return wrapped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--variant",
        choices=["vanilla", "lik"],
        required=True,
        help="'lik' applies patch_colpali_engine() before training; 'vanilla' leaves colpali-engine untouched.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Where to write the JSON metrics.")
    parser.add_argument(
        "--max-steps", type=int, default=4, help="Training steps; memory peaks settle by step 1."
    )
    parser.add_argument("--batch-size", type=int, default=16, help="per_device_train_batch_size.")
    parser.add_argument("--n-samples", type=int, default=2048, help="colpali_train_set subset size.")
    parser.add_argument("--model", default="vidore/colqwen2-base", help="Base ColQwen2 checkpoint.")
    args = parser.parse_args()

    if args.variant == "lik":
        # Native-LIK colpali-engine routes through this env var (ignored by older
        # releases); patch_colpali_engine() covers older releases and is a no-op
        # on the native build.
        os.environ.setdefault("COLPALI_SCORES_BACKEND", "lik")
        from late_interaction_kernels import patch_colpali_engine

        patch_colpali_engine()
    print(f"variant={args.variant} · LIK version: {_lik_version()}")

    from colpali_engine.trainer.colmodel_training import ColModelTraining

    config = _build_training_config(args.model, args.batch_size, args.max_steps, args.n_samples)
    training_app = ColModelTraining(config)

    # Wrap the loss head's forward (the patched one when --variant lik, since
    # patch_colpali_engine() already swapped the class method) to record per-call VRAM.
    recorder = MaxsimVramRecorder()
    loss_func = config.loss_func
    unwrapped_loss_forward: LossForwardFn = loss_func.forward
    loss_func.forward = recorder.wrap(unwrapped_loss_forward, "pairwise_ce_loss")

    # Inline ColModelTraining.train() so the timer callback and trainer fixes can be attached.
    timer_cb = _make_step_timer()
    trainer = _make_fixed_trainer_cls()(
        model=training_app.model,
        train_dataset=training_app.train_dataset,
        eval_dataset=None,
        args=config.tr_args,
        data_collator=training_app.collator,
        loss_func=loss_func,
        is_vision_model=config.processor is not None,
        callbacks=[timer_cb],
    )
    trainer.args.remove_unused_columns = False

    torch.cuda.reset_peak_memory_stats()
    base_payload: dict = {
        "variant": args.variant,
        "lik_version": _lik_version(),
        "device_name": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "batch_size": config.tr_args.per_device_train_batch_size,
    }

    # OOM is an expected sweep outcome: record it and exit 0 so the driver keeps going.
    oom = False
    oom_message: str | None = None
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError as error:
        # The first line names the failed allocation ("Tried to allocate X GiB") — which term crossed.
        oom = True
        oom_message = str(error).split("\n")[0]
    recorder.fold_run_peak()

    # Free what an OOMed run left behind so the isolated replays start from a clean allocator.
    torch.cuda.empty_cache()
    op_under_test: MaxsimFn = _lik_maxsim if args.variant == "lik" else _vanilla_maxsim
    isolated_calls = [_replay_op_isolated(op_under_test, record) for record in recorder.calls]

    payload = {
        **base_payload,
        "oom": oom,
        "oom_message": oom_message,
        "step_peak_alloc_mib": recorder.run_peak_alloc_bytes / 2**20,
        "step_peak_reserved_mib": recorder.run_peak_reserved_bytes / 2**20,
        "step_times_sec": timer_cb.step_times,
        "maxsim_in_train_summary": _summarize_calls(recorder.calls),
        "maxsim_isolated_summary": _summarize_calls(isolated_calls),
        "maxsim_calls_in_train": recorder.calls,
        "maxsim_calls_isolated": isolated_calls,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Wrote metrics to {args.output}")


if __name__ == "__main__":
    main()
