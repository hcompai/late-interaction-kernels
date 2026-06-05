"""Run a few real PyLate training steps and record the VRAM used by every MaxSim call.

The PyLate sibling of ``bench_colpali_e2e.py`` — same design (per-call recording, exact
isolated replay, OOM as a recorded outcome; see that module for the methodology), with
real ``SentenceTransformerTrainer`` steps on MS MARCO triplets and the ``Contrastive``
loss's ``score_metric`` as the wrap point.

Two PyLate-specific mechanics shape what to expect here, both narrowing the LIK gap vs
the ColQwen2 results: ``ColBERTScores`` materializes the ``[A, B, Lq, Ld]`` grid one
document-slot at a time and reduces with ``.max(dim)`` — which saves int64 *indices*
for backward, not the grid (colpali's ``amax`` keeps the grid alive) — so vanilla's B²
cost is a forward/backward *transient* rather than held bytes. And PyLate ships its own
mitigation knob, exposed here as ``--score-mini-batch-size``, so the sweep can compare
LIK against it.

Usage:
    python benchmarks/bench_pylate_e2e.py --variant vanilla \\
        --batch-size 256 --max-steps 4 \\
        --output benchmarks/results/pylate_e2e_b256_vanilla.json
"""

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import torch

ScoreMetricFn = Callable[..., torch.Tensor]
GroupScoresFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor], torch.Tensor]


def _vanilla_group_scores(
    queries: torch.Tensor,
    documents: torch.Tensor,
    queries_mask: torch.Tensor | None,
    documents_mask: torch.Tensor,
) -> torch.Tensor:
    """Verbatim PyLate 1.5 ``ColBERTScores`` math, so the isolated replay allocates
    exactly what vanilla training does."""
    batch, n_slots = documents.shape[0], documents.shape[1]
    per_group: list[torch.Tensor] = []
    for slot in range(n_slots):
        scores = torch.einsum("ash,bth->abst", queries, documents[:, slot])
        if queries_mask is not None:
            scores = scores * queries_mask.unsqueeze(1).unsqueeze(3)
        scores = scores * documents_mask[:, slot].unsqueeze(0).unsqueeze(2)
        per_group.append(scores.max(axis=-1).values.sum(axis=-1))
    return torch.stack(per_group, dim=2).reshape(-1, batch * n_slots)


def _lik_group_scores(
    queries: torch.Tensor,
    documents: torch.Tensor,
    queries_mask: torch.Tensor | None,
    documents_mask: torch.Tensor,
) -> torch.Tensor:
    """The same per-slot loop with each slot fused: what ``patch_pylate()`` routes
    ``colbert_scores`` to inside ``ColBERTScores``."""
    from late_interaction_kernels.autograd import maxsim

    batch, n_slots = documents.shape[0], documents.shape[1]
    q_mask = None if queries_mask is None else queries_mask != 0
    per_group: list[torch.Tensor] = []
    for slot in range(n_slots):
        per_group.append(
            maxsim(queries, documents[:, slot], q_mask=q_mask, d_mask=documents_mask[:, slot] != 0)
        )
    return torch.stack(per_group, dim=2).reshape(-1, batch * n_slots)


def _lik_version() -> str:
    import late_interaction_kernels as lik

    return getattr(lik, "__version__", "unknown")


def _load_msmarco_triplets(n_samples: int):
    """The canonical PyLate contrastive train set (`examples/train/contrastive.py`):
    `sentence-transformers/msmarco-bm25` "triplet" — (query, positive, negative) text
    columns, a single ~220 MB parquet file."""
    from datasets import load_dataset

    dataset = load_dataset("sentence-transformers/msmarco-bm25", "triplet", split="train")
    if n_samples < len(dataset):
        dataset = dataset.select(range(n_samples))
    return dataset


def _build_training_args(output_dir: str, batch_size: int, max_steps: int):
    """The PyLate contrastive example recipe (bf16 autocast, fp32 weights, no grad
    checkpointing), pinned seeds — mirrors `_build_training_config` in the colpali bench."""
    from sentence_transformers import SentenceTransformerTrainingArguments

    return SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        dataloader_num_workers=4,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        warmup_steps=0,
        learning_rate=3e-6,
        report_to="none",
        bf16=True,
        seed=42,
        data_seed=42,
    )


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


def _replay_op_isolated(fn: GroupScoresFn, record: dict) -> dict:
    """Replay one recorded ``score_metric`` call on fresh random embeddings whose graph
    contains only the scoring op, so the forward/saved/backward peaks bracket exactly.
    Masks are replayed as all-ones of the recorded shapes (allocation-identical to the
    real skiplist masks). Returns MiB deltas, or an OOM marker."""
    dtype = getattr(torch, record["dtype"].removeprefix("torch."))
    queries = torch.randn(record["queries_shape"], dtype=dtype, device="cuda", requires_grad=True)
    documents = torch.randn(record["documents_shape"], dtype=dtype, device="cuda", requires_grad=True)
    queries_mask = (
        torch.ones(record["queries_mask_shape"], dtype=dtype, device="cuda")
        if record.get("queries_mask_shape")
        else None
    )
    documents_mask = torch.ones(record["documents_mask_shape"], dtype=dtype, device="cuda")

    torch.cuda.synchronize()
    before_forward_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    try:
        out = fn(queries, documents, queries_mask, documents_mask)
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
    """Per-call VRAM for the scoring op, plus a run-level peak that survives the per-op stat resets."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.run_peak_alloc_bytes: int = 0
        self.run_peak_reserved_bytes: int = 0

    def fold_run_peak(self) -> None:
        """Capture the global peak before a reset wipes it; call once more after training."""
        self.run_peak_alloc_bytes = max(self.run_peak_alloc_bytes, torch.cuda.max_memory_allocated())
        self.run_peak_reserved_bytes = max(self.run_peak_reserved_bytes, torch.cuda.max_memory_reserved())

    def wrap(self, score_metric: ScoreMetricFn, op_name: str) -> ScoreMetricFn:
        # Parameter names must match ColBERTScores.__call__: Contrastive calls the
        # score_metric with queries_mask=/documents_mask= keywords.
        def wrapped(
            queries_embeddings: torch.Tensor,
            documents_embeddings: torch.Tensor,
            queries_mask: torch.Tensor | None = None,
            documents_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            torch.cuda.synchronize()
            self.fold_run_peak()
            before_forward_bytes = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            record: dict = {
                "op": op_name,
                "queries_shape": list(queries_embeddings.shape),
                "documents_shape": list(documents_embeddings.shape),
                "queries_mask_shape": list(queries_mask.shape) if queries_mask is not None else None,
                "documents_mask_shape": list(documents_mask.shape),
                "dtype": str(queries_embeddings.dtype),
            }
            try:
                out = score_metric(
                    queries_embeddings,
                    documents_embeddings,
                    queries_mask=queries_mask,
                    documents_mask=documents_mask,
                )
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
        help="'lik' applies patch_pylate() before training; 'vanilla' leaves PyLate untouched.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Where to write the JSON metrics.")
    parser.add_argument(
        "--max-steps", type=int, default=4, help="Training steps; memory peaks settle by step 1."
    )
    parser.add_argument("--batch-size", type=int, default=64, help="per_device_train_batch_size.")
    parser.add_argument("--n-samples", type=int, default=8192, help="MS MARCO triplet subset size.")
    parser.add_argument(
        "--model", default="lightonai/GTE-ModernColBERT-v1", help="PyLate ColBERT checkpoint."
    )
    parser.add_argument(
        "--score-mini-batch-size",
        type=int,
        default=0,
        help="PyLate's own mitigation knob: chunk Contrastive's query axis (0 = off, the default recipe).",
    )
    parser.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="Checkpoint the encoder. Off in PyLate's canonical recipe — but without it the encoder"
        " activations cap the batch at ~B=128 on 80 GB before the score grid matters; this regime"
        " matches the ColQwen2 bench (which checkpoints) and is where the MaxSim term binds.",
    )
    args = parser.parse_args()

    if args.variant == "lik":
        from late_interaction_kernels import patch_pylate

        patch_pylate()
    print(f"variant={args.variant} · LIK version: {_lik_version()}")

    from pylate import losses, models, utils
    from sentence_transformers import SentenceTransformerTrainer

    model = models.ColBERT(model_name_or_path=args.model)
    if args.grad_checkpoint:
        # SentenceTransformer doesn't expose this; reach the HF encoder directly.
        model[0].auto_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    loss_func = losses.Contrastive(
        model=model,
        score_mini_batch_size=args.score_mini_batch_size or None,
    )

    # Wrap the loss's score_metric (ColBERTScores — already routing through the patched
    # colbert_scores when --variant lik) to record per-call VRAM.
    recorder = MaxsimVramRecorder()
    loss_func.score_metric = recorder.wrap(loss_func.score_metric, "colbert_scores")

    timer_cb = _make_step_timer()
    trainer = SentenceTransformerTrainer(
        model=model,
        args=_build_training_args("/tmp/pylate_e2e_trainer_output", args.batch_size, args.max_steps),
        train_dataset=_load_msmarco_triplets(args.n_samples),
        loss=loss_func,
        data_collator=utils.ColBERTCollator(model.tokenize),
        callbacks=[timer_cb],
    )

    torch.cuda.reset_peak_memory_stats()
    base_payload: dict = {
        "variant": args.variant,
        "lik_version": _lik_version(),
        "device_name": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "batch_size": args.batch_size,
        "model": args.model,
        "query_length": model.query_length,
        "document_length": model.document_length,
        "score_mini_batch_size": args.score_mini_batch_size or None,
        "grad_checkpointing": args.grad_checkpoint,
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
    op_under_test: GroupScoresFn = _lik_group_scores if args.variant == "lik" else _vanilla_group_scores
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
