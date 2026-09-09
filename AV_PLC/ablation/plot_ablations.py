r"""
Create publication-ready 3 x 2 result grids for:

    Experiment 1  : audio-gap duration sweep
    Experiment 2A : uniform global video-frame reduction
    Experiment 2B : local video-frame reduction

Each experiment produces one figure:
    rows    = GRID, LRS2, VoxCeleb2
    columns = PESQ, STOI

Each figure has one shared legend at the bottom and is saved as:
    high-resolution PNG
    vector PDF
    vector SVG

The script uses LaTeX rendering with:
    amsmath, amssymb, and amsfonts-style notation

A working LaTeX installation is therefore required.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# Fixed experiment definitions
# ---------------------------------------------------------------------

DATASETS = ["grid", "lrs2", "voxceleb2"]

DATASET_TITLES = {
    "grid": "GRID",
    "lrs2": "LRS2",
    "voxceleb2": "VoxCeleb2",
}

METRICS = ["pesq", "stoi"]

METRIC_TITLES = {
    "pesq": "PESQ",
    "stoi": "STOI",
}

GAP_LENGTHS_MS = [
    10, 20, 40, 80, 160, 320, 500, 750, 1000, 1250, 1500
]

EXP2_GAPS_MS = [160, 500, 1000]

GLOBAL_KEEP_COUNTS = [0, 3, 6, 12, 24, 48, 75]

LOCAL_KEEP_ORDER = ["0", "2", "4", "8", "12", "16", "all"]


# ---------------------------------------------------------------------
# I/O and common utilities
# ---------------------------------------------------------------------

def load_json(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Result file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a list in {path}, got {type(data).__name__}"
        )

    return data


def normalize_dataset_name(name: str) -> str:
    name = str(name).strip().lower()

    aliases = {
        "vox2": "voxceleb2",
        "vox2_short": "voxceleb2",
        "voxceleb": "voxceleb2",
    }

    return aliases.get(name, name)


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    return value if np.isfinite(value) else None


def confidence_interval_95(
    std_value: Any,
    num_samples: Any,
) -> Optional[float]:
    """
    Approximate 95% confidence interval:

        1.96 * std / sqrt(n)
    """
    std = safe_float(std_value)
    n = safe_float(num_samples)

    if std is None or n is None or n <= 0:
        return None

    return 1.96 * std / math.sqrt(n)


def metric_from_summary(
    summary: Dict[str, Any],
    metric: str,
) -> Tuple[Optional[float], Optional[float]]:
    mean = safe_float(summary.get(f"{metric}_mean"))

    ci95 = confidence_interval_95(
        summary.get(f"{metric}_std"),
        summary.get(f"num_{metric}"),
    )

    return mean, ci95


def configure_matplotlib(font_size: float = 12.0) -> None:
    """
    Use Matplotlib's built-in Computer Modern-style fonts.

    No external LaTeX installation is required.
    """
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": [
            "Computer Modern Roman",
            "CMU Serif",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "cm",
        "mathtext.rm": "serif",
        "mathtext.it": "serif:italic",
        "mathtext.bf": "serif:bold",

        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size + 1,
        "legend.fontsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "figure.titlesize": font_size + 2,

        "figure.dpi": 200,
        "savefig.dpi": 1200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,

        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "lines.linewidth": 2.2,
        "lines.markersize": 6.5,
        "legend.frameon": False,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_base.with_suffix(".png"), dpi=1200)
    fig.savefig(output_base.with_suffix(".pdf"))
    fig.savefig(output_base.with_suffix(".svg"))

    plt.close(fig)


def add_ci_band(
    ax: plt.Axes,
    x: Sequence[float],
    y: Sequence[Optional[float]],
    ci: Sequence[Optional[float]],
) -> None:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(
        [np.nan if value is None else value for value in y],
        dtype=float,
    )
    ci_arr = np.asarray(
        [np.nan if value is None else value for value in ci],
        dtype=float,
    )

    valid = (
        np.isfinite(x_arr)
        & np.isfinite(y_arr)
        & np.isfinite(ci_arr)
    )

    if valid.sum() >= 2:
        ax.fill_between(
            x_arr[valid],
            y_arr[valid] - ci_arr[valid],
            y_arr[valid] + ci_arr[valid],
            alpha=0.14,
            linewidth=0,
        )


def plot_line_with_ci(
    ax: plt.Axes,
    x: Sequence[float],
    y: Sequence[Optional[float]],
    ci: Sequence[Optional[float]],
    *,
    label: str,
    marker: str,
    linestyle: str,
) -> None:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(
        [np.nan if value is None else value for value in y],
        dtype=float,
    )

    ax.plot(
        x_arr,
        y_arr,
        marker=marker,
        linestyle=linestyle,
        label=label,
    )

    add_ci_band(ax, x_arr, y, ci)


def apply_grid_layout(
    fig: plt.Figure,
    axes: np.ndarray,
    experiment_title: str,
    legend_title: Optional[str],
    *,
    bottom_annotation: Optional[str] = None,
) -> None:
    """
    Apply one common title and one common legend below the entire 3 x 2 grid.
    """
    fig.suptitle(
        experiment_title,
        y=0.995,
        fontweight="bold",
    )

    # Column headings.
    for col, metric in enumerate(METRICS):
        axes[0, col].set_title(
            METRIC_TITLES[metric],
            pad=10,
        )

    # Dataset row labels placed once at the far left of each row.
    for row, dataset in enumerate(DATASETS):
        axes[row, 0].annotate(
            DATASET_TITLES[dataset],
            xy=(-0.24, 0.5),
            xycoords="axes fraction",
            ha="center",
            va="center",
            rotation=90,
            fontsize=13,
            fontweight="bold",
        )

    # Collect legend handles from the first panel that contains curves.
    handles, labels = [], []

    for ax in axes.flat:
        candidate_handles, candidate_labels = ax.get_legend_handles_labels()
        if candidate_handles:
            handles = candidate_handles
            labels = candidate_labels
            break

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.025),
            ncol=len(labels),
            title=legend_title,
            columnspacing=2.0,
            handlelength=2.8,
        )

    if bottom_annotation:
        fig.text(
            0.5,
            0.008,
            bottom_annotation,
            ha="center",
            va="bottom",
            fontsize=11,
            fontstyle="italic",
        )

    # Reserve space for shared legend/annotation.
    fig.subplots_adjust(
        left=0.12,
        right=0.985,
        top=0.94,
        bottom=0.13,
        hspace=0.36,
        wspace=0.22,
    )


def write_combined_csv(
    rows: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    if not rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({
        key
        for row in rows
        for key in row.keys()
    })

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# Experiment 1
# ---------------------------------------------------------------------

def plot_experiment1(
    records: List[Dict[str, Any]],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    exported_rows: List[Dict[str, Any]] = []

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(15.5, 16.5),
        squeeze=False,
    )

    for row, dataset in enumerate(DATASETS):
        dataset_records = [
            record
            for record in records
            if normalize_dataset_name(
                record.get("dataset", "")
            ) == dataset
        ]

        by_gap = {
            int(record["gap_ms"]): record
            for record in dataset_records
        }

        for col, metric in enumerate(METRICS):
            ax = axes[row, col]
            x = [
                gap
                for gap in GAP_LENGTHS_MS
                if gap in by_gap
            ]

            audio_y, audio_ci = [], []
            av_y, av_ci = [], []

            for gap_ms in x:
                record = by_gap[gap_ms]

                audio_mean, audio_err = metric_from_summary(
                    record.get("audio_only", {}),
                    metric,
                )
                av_mean, av_err = metric_from_summary(
                    record.get("audio_visual", {}),
                    metric,
                )

                audio_y.append(audio_mean)
                audio_ci.append(audio_err)
                av_y.append(av_mean)
                av_ci.append(av_err)

                exported_rows.extend([
                    {
                        "experiment": "Experiment 1",
                        "dataset": dataset,
                        "metric": metric,
                        "gap_ms": gap_ms,
                        "condition": "Audio-only",
                        "mean": audio_mean,
                        "ci95": audio_err,
                    },
                    {
                        "experiment": "Experiment 1",
                        "dataset": dataset,
                        "metric": metric,
                        "gap_ms": gap_ms,
                        "condition": "Audio-visual",
                        "mean": av_mean,
                        "ci95": av_err,
                    },
                ])

            if not x:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )
                continue

            plot_line_with_ci(
                ax,
                x,
                audio_y,
                audio_ci,
                label="Audio-only",
                marker="o",
                linestyle="--",
            )
            plot_line_with_ci(
                ax,
                x,
                av_y,
                av_ci,
                label="Audio-visual",
                marker="s",
                linestyle="-",
            )

            ax.set_xscale("log", base=2)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [str(value) for value in x],
                rotation=35,
                ha="right",
            )

            ax.set_xlabel(
                "Audio gap duration (ms)"
            )
            ax.set_ylabel(
                METRIC_TITLES[metric]
            )

    apply_grid_layout(
        fig,
        axes,
        experiment_title=(
            "Experiment 1: Effect of Missing-Audio Duration"
        ),
        legend_title=None,
        bottom_annotation=(
            "Shaded regions indicate approximate 95% confidence intervals."
        ),
    )

    save_figure(
        fig,
        output_dir / "experiment1_gap_sweep",
    )

    return exported_rows


# ---------------------------------------------------------------------
# Experiment 2A
# ---------------------------------------------------------------------

def plot_experiment2a(
    records: List[Dict[str, Any]],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    exported_rows: List[Dict[str, Any]] = []

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(15.5, 16.5),
        squeeze=False,
    )

    markers = ["o", "s", "^"]
    linestyles = ["-", "--", "-."]

    for row, dataset in enumerate(DATASETS):
        dataset_records = [
            record
            for record in records
            if normalize_dataset_name(
                record.get("dataset", "")
            ) == dataset
        ]

        lookup = {
            (
                int(record["gap_ms"]),
                int(record["video_keep_count"]),
            ): record
            for record in dataset_records
        }

        for col, metric in enumerate(METRICS):
            ax = axes[row, col]

            for gap_ms, marker, linestyle in zip(
                EXP2_GAPS_MS,
                markers,
                linestyles,
            ):
                x, y, ci = [], [], []

                for keep_count in GLOBAL_KEEP_COUNTS:
                    record = lookup.get(
                        (gap_ms, keep_count)
                    )

                    if record is None:
                        continue

                    mean, err = metric_from_summary(
                        record.get("summary", {}),
                        metric,
                    )

                    x.append(keep_count)
                    y.append(mean)
                    ci.append(err)

                    exported_rows.append({
                        "experiment": "Experiment 2A",
                        "dataset": dataset,
                        "metric": metric,
                        "gap_ms": gap_ms,
                        "video_keep_count": keep_count,
                        "condition": f"{gap_ms} ms gap",
                        "mean": mean,
                        "ci95": err,
                    })

                if x:
                    plot_line_with_ci(
                        ax,
                        x,
                        y,
                        ci,
                        label=rf"${gap_ms}\,\mathrm{{ms}}$",
                        marker=marker,
                        linestyle=linestyle,
                    )

            ax.set_xticks(GLOBAL_KEEP_COUNTS)
            ax.set_xticklabels([
                "0\nAO",
                "3\n1 fps",
                "6\n2 fps",
                "12\n4 fps",
                "24\n8 fps",
                "48\n16 fps",
                "75\n25 fps",
            ])

            ax.set_xlabel(
                "Frames retained across the 3-s clip"
            )
            ax.set_ylabel(
                METRIC_TITLES[metric]
            )

    apply_grid_layout(
        fig,
        axes,
        experiment_title=(
            "Experiment 2A: Uniform Global Video-Frame Reduction"
        ),
        legend_title="Audio-gap duration",
        bottom_annotation=(
            "AO denotes the audio-only model; shaded regions indicate approximate 95% confidence intervals."
        ),
    )

    save_figure(
        fig,
        output_dir / "experiment2a_uniform_video_reduction",
    )

    return exported_rows


# ---------------------------------------------------------------------
# Experiment 2B
# ---------------------------------------------------------------------

def normalize_local_keep(value: Any) -> str:
    text = str(value).strip().lower()

    if text in {"0", "0.0"}:
        return "0"

    if text == "all":
        return "all"

    try:
        return str(int(float(text)))
    except ValueError:
        return text


def plot_experiment2b(
    records: List[Dict[str, Any]],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    exported_rows: List[Dict[str, Any]] = []

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(15.5, 16.5),
        squeeze=False,
    )

    markers = ["o", "s", "^"]
    linestyles = ["-", "--", "-."]

    x_positions = np.arange(
        len(LOCAL_KEEP_ORDER),
        dtype=float,
    )

    x_labels = [
        "0\nAO",
        "2",
        "4",
        "8",
        "12",
        "16",
        "All",
    ]

    for row, dataset in enumerate(DATASETS):
        dataset_records = [
            record
            for record in records
            if normalize_dataset_name(
                record.get("dataset", "")
            ) == dataset
        ]

        lookup = {
            (
                int(record["gap_ms"]),
                normalize_local_keep(
                    record.get("local_keep_count")
                ),
            ): record
            for record in dataset_records
        }

        for col, metric in enumerate(METRICS):
            ax = axes[row, col]

            for gap_ms, marker, linestyle in zip(
                EXP2_GAPS_MS,
                markers,
                linestyles,
            ):
                y, ci = [], []

                for keep_label in LOCAL_KEEP_ORDER:
                    record = lookup.get(
                        (gap_ms, keep_label)
                    )

                    if record is None:
                        y.append(None)
                        ci.append(None)
                        continue

                    mean, err = metric_from_summary(
                        record.get("summary", {}),
                        metric,
                    )

                    y.append(mean)
                    ci.append(err)

                    exported_rows.append({
                        "experiment": "Experiment 2B",
                        "dataset": dataset,
                        "metric": metric,
                        "gap_ms": gap_ms,
                        "local_keep_count": keep_label,
                        "condition": f"{gap_ms} ms gap",
                        "mean": mean,
                        "ci95": err,
                    })

                plot_line_with_ci(
                    ax,
                    x_positions,
                    y,
                    ci,
                    label=rf"${gap_ms}\,\mathrm{{ms}}$",
                    marker=marker,
                    linestyle=linestyle,
                )

            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels)

            ax.set_xlabel(
                "Frames retained inside the local window"
            )
            ax.set_ylabel(
                METRIC_TITLES[metric]
            )

    apply_grid_layout(
        fig,
        axes,
        experiment_title=(
            "Experiment 2B: Local Video-Frame Reduction Around the Gap"
        ),
        legend_title="Audio-gap duration",
        bottom_annotation=(
            "AO denotes the audio-only model; All retains every frame in the local window; shaded regions indicate approximate 95% confidence intervals."
        ),
    )

    save_figure(
        fig,
        output_dir / "experiment2b_local_video_reduction",
    )

    return exported_rows


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create 3 x 2 publication figures for Experiments 1, 2A, and 2B."
        )
    )

    parser.add_argument(
        "--exp1",
        type=Path,
        default=Path(
            "results_exp1_gap_sweep/all_results_summary.json"
        ),
    )

    parser.add_argument(
        "--exp2a",
        type=Path,
        default=Path(
            "results_exp2a_uniform_video_rate/"
            "all_results_summary.json"
        ),
    )

    parser.add_argument(
        "--exp2b",
        type=Path,
        default=Path(
            "results_exp2b_local_video_reduction/"
            "all_results_summary.json"
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("paper_figures"),
    )

    parser.add_argument(
        "--font_size",
        type=float,
        default=12.0,
    )

    args = parser.parse_args()

    configure_matplotlib(
        font_size=args.font_size
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    exp1_records = load_json(args.exp1)
    exp2a_records = load_json(args.exp2a)
    exp2b_records = load_json(args.exp2b)

    combined_rows: List[Dict[str, Any]] = []

    combined_rows.extend(
        plot_experiment1(
            exp1_records,
            args.output_dir,
        )
    )

    combined_rows.extend(
        plot_experiment2a(
            exp2a_records,
            args.output_dir,
        )
    )

    combined_rows.extend(
        plot_experiment2b(
            exp2b_records,
            args.output_dir,
        )
    )

    write_combined_csv(
        combined_rows,
        args.output_dir / "combined_plot_data.csv",
    )

    print(
        f"Figures saved under: "
        f"{args.output_dir.resolve()}"
    )
    print("PNG resolution: 1200 dpi")
    print("Vector PDF and SVG files were also generated.")


if __name__ == "__main__":
    main()
