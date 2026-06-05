"""Summarize the PyLate e2e MaxSim VRAM sweep: markdown tables + log-log plot.

The PyLate sibling of ``summarize_colpali_e2e.py``: op-VRAM table, batch-ceiling table,
optional log-log plot from the ``pylate_e2e_b<B>_<label>.json`` cells, plus a
supplementary listing for the ``vanilla-chunk<N>`` cells (PyLate's own mitigation).
Unlike the colpali table, the op-VRAM table leads with the *forward transient*: PyLate's
``.max(dim)`` saves int64 indices rather than the grid, so vanilla's "held" column stays
small and the B² cost lives in the transients.

Usage:
    python benchmarks/summarize_pylate_e2e.py --results-dir benchmarks/results \\
        --plot benchmarks/results/pylate_e2e_vram.png
"""

import argparse
import glob
import json
from pathlib import Path

VARIANTS = ("vanilla", "lik")


def _fmt_mib(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value >= 1024:
        return f"{value / 1024:.2f} GiB"
    return f"{value:.0f} MiB"


def _label(data: dict) -> str:
    """Ckpt / chunked cells get their own column families so regimes don't collide."""
    label: str = data["variant"]
    if data.get("grad_checkpointing"):
        label += "-ckpt"
    chunk = data.get("score_mini_batch_size")
    if chunk:
        label += f"-chunk{chunk}"
    return label


def _load_cells(results_dir: Path) -> dict[tuple[int, str], dict]:
    """One cell per (batch_size, label); crashed cells keep only the marker."""
    cells: dict[tuple[int, str], dict] = {}
    for path in glob.glob(str(results_dir / "pylate_e2e_b*_*.json")):
        data = json.load(open(path))
        key = (data["batch_size"], _label(data))
        if data.get("crashed"):
            cells[key] = {"crashed": True}
            continue
        summary = data["maxsim_isolated_summary"]
        cells[key] = {
            "fwd": summary["max_forward_transient_peak_mib"],
            "saved": summary["max_saved_for_backward_mib"],
            "bwd": summary["max_backward_transient_peak_mib"],
            "step_peak": data["step_peak_alloc_mib"],
            "step_reserved": data["step_peak_reserved_mib"],
            "oom": data["oom"],
            "oom_message": data.get("oom_message"),
        }
    return cells


def _op_total(cell: dict | None) -> float | None:
    """Peak op footprint: the worse of the forward and (held + backward) phases."""
    if cell is None or cell.get("crashed"):
        return None
    if any(cell[k] is None for k in ("fwd", "saved", "bwd")):
        return None
    return max(cell["fwd"], cell["saved"] + cell["bwd"])


def _print_op_vram_table(batch_sizes: list[int], cells: dict[tuple[int, str], dict]) -> None:
    print("## VRAM attributable to the MaxSim op (isolated replay)\n")
    print(
        "| batch size | vanilla: fwd peak | vanilla: held | vanilla: bwd spike "
        "| LIK: fwd peak | LIK: held | LIK: bwd spike |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for batch in batch_sizes:
        row: list[str] = [str(batch)]
        for variant in VARIANTS:
            cell = cells.get((batch, variant))
            if cell is None or cell.get("crashed"):
                row += ["n/a", "n/a", "n/a"]
                continue
            row += [_fmt_mib(cell["fwd"]), _fmt_mib(cell["saved"]), _fmt_mib(cell["bwd"])]
        print("| " + " | ".join(row) + " |")


def _print_batch_ceiling_table(batch_sizes: list[int], cells: dict[tuple[int, str], dict]) -> None:
    # Fragmentation is a reserved-vs-alloc story (the colpali B=128 OOM is a small
    # request failing with ~25 GiB reserved-but-unallocated), so show both.
    print("\n## Batch-size ceiling (whole-step peak alloc / reserved)\n")
    print("| batch size | vanilla | LIK |")
    print("| --- | --- | --- |")
    for batch in batch_sizes:
        row: list[str] = [str(batch)]
        for variant in VARIANTS:
            cell = cells.get((batch, variant))
            if cell is None or cell.get("crashed"):
                row.append("crashed" if cell else "n/a")
            elif cell["oom"]:
                row.append(f"OOM ({_fmt_mib(cell['step_peak'])} / {_fmt_mib(cell['step_reserved'])} pre-OOM)")
            else:
                row.append(f"{_fmt_mib(cell['step_peak'])} / {_fmt_mib(cell['step_reserved'])}")
        print("| " + " | ".join(row) + " |")

    for batch in batch_sizes:
        for variant in VARIANTS:
            message = (cells.get((batch, variant)) or {}).get("oom_message")
            if message:
                print(f"\nOOM message ({variant} B={batch}): {message}")


def _print_chunked_cells(cells: dict[tuple[int, str], dict]) -> None:
    # Main vanilla/lik families (either regime) are covered by the tables above.
    main_labels = {v for base in ("vanilla", "lik") for v in (base, f"{base}-ckpt")}
    extra = sorted(key for key in cells if key[1] not in main_labels)
    if not extra:
        return
    print("\n## PyLate's own mitigation: score_mini_batch_size cells\n")
    print("| batch size | label | op fwd peak | op held | op bwd spike | step peak | outcome |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for batch, label in extra:
        cell = cells[(batch, label)]
        if cell.get("crashed"):
            print(f"| {batch} | {label} | n/a | n/a | n/a | n/a | crashed |")
            continue
        outcome = "OOM" if cell["oom"] else "fits"
        print(
            f"| {batch} | {label} | {_fmt_mib(cell['fwd'])} | {_fmt_mib(cell['saved'])} "
            f"| {_fmt_mib(cell['bwd'])} | {_fmt_mib(cell['step_peak'])} | {outcome} |"
        )


def _write_plot(plot_path: Path, batch_sizes: list[int], cells: dict[tuple[int, str], dict]) -> None:
    # Imported here so the tables work in a venv without matplotlib.
    import matplotlib.pyplot as plt

    # The plot needs both variants' op totals; keep only fully-populated batch sizes.
    plotted = [b for b in batch_sizes if all(_op_total(cells.get((b, v))) is not None for v in VARIANTS)]
    vanilla_totals = [_op_total(cells[(b, VARIANTS[0])]) for b in plotted]
    lik_totals = [_op_total(cells[(b, VARIANTS[1])]) for b in plotted]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(plotted, vanilla_totals, "o-", color="tab:red", label="vanilla (PyLate einsum)")
    ax.plot(plotted, lik_totals, "s-", color="tab:green", label="LIK (fused kernel)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(plotted, [str(b) for b in plotted])
    tick_values = [8, 32, 128, 512, 2048, 8192, 32768]
    ax.set_yticks(tick_values, [_fmt_mib(v) for v in tick_values])
    ax.minorticks_off()
    ax.set_xlabel("per-device batch size (log scale)")
    ax.set_ylabel("MaxSim op VRAM: peak phase footprint\n(log scale)")
    ax.set_title("VRAM attributable to the MaxSim op (PyLate Contrastive)")
    ax.grid(True, which="both", alpha=0.3)
    ratio = vanilla_totals[-1] / lik_totals[-1]
    ax.annotate(
        f"{ratio:.0f}× at B={plotted[-1]}",
        xy=(plotted[-1], vanilla_totals[-1]),
        xytext=(-60, -8),
        textcoords="offset points",
        color="black",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    print(f"\nWrote plot to {plot_path}")


def _write_ceiling_plot(plot_path: Path, batch_sizes: list[int], cells: dict[tuple[int, str], dict]) -> None:
    """The batch-sweep figure from colpali PR #412: whole-step peak per variant,
    identical while both fit, hatched where a variant OOMs."""
    # Imported here so the tables work in a venv without matplotlib.
    import matplotlib.pyplot as plt

    plotted = [
        b for b in batch_sizes if all((b, v) in cells and not cells[(b, v)].get("crashed") for v in VARIANTS)
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    positions = range(len(plotted))
    width = 0.38
    colors = {"vanilla": "tab:red", "lik": "tab:green"}
    for offset, variant in zip((-width / 2, width / 2), VARIANTS, strict=True):
        peaks = [cells[(b, variant)]["step_peak"] / 1024 for b in plotted]
        ooms = [cells[(b, variant)]["oom"] for b in plotted]
        bars = ax.bar(
            [p + offset for p in positions],
            peaks,
            width,
            color=colors[variant.split("-")[0]],
            label=f"{variant} (hatched = OOM)",
        )
        for bar, oom in zip(bars, ooms, strict=True):
            if oom:
                bar.set_hatch("//")
                bar.set_alpha(0.45)
                ax.annotate(
                    "OOM",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    color="black",
                )
    ax.set_xticks(list(positions), [str(b) for b in plotted])
    ax.set_xlabel("per-device batch size")
    ax.set_ylabel("whole-step peak allocated VRAM (GiB)")
    ax.set_title("Batch-size ceiling (OOM bars show the pre-OOM peak)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    print(f"\nWrote ceiling plot to {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--ceiling-plot", type=Path, default=None)
    parser.add_argument(
        "--regime",
        choices=["plain", "ckpt"],
        default="plain",
        help="Which cells fill the main vanilla/LIK columns: the canonical recipe or --grad-checkpoint.",
    )
    args = parser.parse_args()

    global VARIANTS
    if args.regime == "ckpt":
        VARIANTS = tuple(f"{v}-ckpt" for v in VARIANTS)

    cells = _load_cells(args.results_dir)
    batch_sizes: list[int] = sorted({batch for batch, label in cells if label in VARIANTS})

    _print_op_vram_table(batch_sizes, cells)
    _print_batch_ceiling_table(batch_sizes, cells)
    _print_chunked_cells(cells)

    if args.plot is not None:
        _write_plot(args.plot, batch_sizes, cells)
    if args.ceiling_plot is not None:
        _write_ceiling_plot(args.ceiling_plot, batch_sizes, cells)


if __name__ == "__main__":
    main()
