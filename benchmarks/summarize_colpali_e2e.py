"""Summarize the ColQwen2 e2e MaxSim VRAM sweep: markdown tables + log-log plot.

Reads the ``colpali_e2e_b<B>_<variant>.json`` files written by
``bench_colpali_e2e.py`` (driven by ``scripts/sky_colpali_e2e.yaml``) and emits:

1. the op-attributable VRAM table (held for backward + backward spike, from the
   isolated replays),
2. the batch-ceiling table (whole-step peak alloc + OOM flag per variant — the
   "vanilla OOMs at B=128, LIK trains it" story),
3. optionally a log-log plot of the op totals.

Usage:
    python benchmarks/summarize_colpali_e2e.py --results-dir benchmarks/results \\
        --plot benchmarks/results/colpali_e2e_vram.png
"""

import argparse
import glob
import json
import math
from pathlib import Path

VARIANTS = ("vanilla", "lik")


def _fmt_mib(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value >= 1024:
        return f"{value / 1024:.2f} GiB"
    return f"{value:.0f} MiB"


def _load_cells(results_dir: Path) -> dict[tuple[int, str], dict]:
    """One cell per (batch_size, variant); crashed cells keep only the marker."""
    cells: dict[tuple[int, str], dict] = {}
    for path in glob.glob(str(results_dir / "colpali_e2e_b*_*.json")):
        data = json.load(open(path))
        key = (data["batch_size"], data["variant"])
        if data.get("crashed"):
            cells[key] = {"crashed": True}
            continue
        summary = data["maxsim_isolated_summary"]
        cells[key] = {
            "saved": summary["max_saved_for_backward_mib"],
            "bwd": summary["max_backward_transient_peak_mib"],
            "step_peak": data["step_peak_alloc_mib"],
            "step_reserved": data["step_peak_reserved_mib"],
            "oom": data["oom"],
            "oom_message": data.get("oom_message"),
        }
    return cells


def _op_total(cell: dict | None) -> float | None:
    if cell is None or cell.get("crashed"):
        return None
    if cell["saved"] is None or cell["bwd"] is None:
        return None
    return cell["saved"] + cell["bwd"]


def _print_op_vram_table(batch_sizes: list[int], cells: dict[tuple[int, str], dict]) -> None:
    print("## VRAM attributable to the MaxSim op (isolated replay)\n")
    print(
        "| batch size | vanilla: held | vanilla: bwd spike | vanilla: total "
        "| LIK: held | LIK: bwd spike | LIK: total |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for batch in batch_sizes:
        row: list[str] = [str(batch)]
        for variant in VARIANTS:
            cell = cells.get((batch, variant))
            if cell is None or cell.get("crashed"):
                row += ["n/a", "n/a", "n/a"]
                continue
            row += [_fmt_mib(cell["saved"]), _fmt_mib(cell["bwd"]), _fmt_mib(_op_total(cell))]
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


def _write_plot(plot_path: Path, batch_sizes: list[int], cells: dict[tuple[int, str], dict]) -> None:
    # Imported here so the tables work in a venv without matplotlib.
    import matplotlib.pyplot as plt

    # The plot needs both variants' op totals; keep only fully-populated batch sizes.
    plotted = [b for b in batch_sizes if all(_op_total(cells.get((b, v))) is not None for v in VARIANTS)]
    vanilla_totals = [_op_total(cells[(b, "vanilla")]) for b in plotted]
    lik_totals = [_op_total(cells[(b, "lik")]) for b in plotted]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(plotted, vanilla_totals, "o-", color="tab:red", label="vanilla (torch einsum)")
    ax.plot(plotted, lik_totals, "s-", color="tab:green", label="LIK (fused kernel)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(plotted, [str(b) for b in plotted])
    tick_values = [8, 32, 128, 512, 2048, 8192]
    ax.set_yticks(tick_values, [_fmt_mib(v) for v in tick_values])
    ax.minorticks_off()
    ax.set_xlabel("per-device batch size (log scale)")
    ax.set_ylabel("MaxSim op VRAM: held + backward spike\n(log scale)")
    ax.set_title("VRAM attributable to the MaxSim op")
    ax.grid(True, which="both", alpha=0.3)
    # Slope annotations derived from the data so a drifting run can't end up with a
    # figure asserting a scaling the points don't show. Expected: vanilla ~×4 per
    # doubling (B² score grid), LIK ~×2 (grad_D, linear).
    doublings = math.log2(plotted[-1] / plotted[-2])
    vanilla_rate = (vanilla_totals[-1] / vanilla_totals[-2]) ** (1 / doublings)
    lik_rate = (lik_totals[-1] / lik_totals[-2]) ** (1 / doublings)
    ax.annotate(
        f"×{vanilla_rate:.1f} per doubling (B² grid)",
        xy=(plotted[-2], vanilla_totals[-2]),
        xytext=(-10, 14),
        textcoords="offset points",
        color="tab:red",
        ha="right",
    )
    ax.annotate(
        f"×{lik_rate:.1f} per doubling (grad_D, linear)",
        xy=(plotted[-2], lik_totals[-2]),
        xytext=(-10, 14),
        textcoords="offset points",
        color="tab:green",
        ha="right",
    )
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
    args = parser.parse_args()

    cells = _load_cells(args.results_dir)
    batch_sizes: list[int] = sorted({batch for batch, _ in cells})

    _print_op_vram_table(batch_sizes, cells)
    _print_batch_ceiling_table(batch_sizes, cells)

    if args.plot is not None:
        _write_plot(args.plot, batch_sizes, cells)
    if args.ceiling_plot is not None:
        _write_ceiling_plot(args.ceiling_plot, batch_sizes, cells)


if __name__ == "__main__":
    main()
