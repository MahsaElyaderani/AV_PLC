#!/usr/bin/env python3
"""Run selected evaluations and create qualitative comparison figures.

Typical workflow
----------------
1. Run inference and create one figure per saved utterance:

   python run_qualitative_evaluations.py preview \
       --run-evaluation \
       --datasets grid lrs2 voxceleb2 \
       --loss-rates 30 40

2. After reviewing the preview figures, create a final combined figure:

   python run_qualitative_evaluations.py final \
       --loss-rate 30 \
       --samples \
         grid/test/s30/bbai9p.mpg \
         lrs2/lrs2_v1/mvlrs_v1/main/6330311066473698535/00018.mp4 \
         vox2_short/vox2_test_mp4/id03789/go_QOzc79Uc/00313.mp4 \
       --models av_plc av_lstm av_transformer \
       --output final_examples.png

The script matches samples by the portable path after ``datasets/``. Absolute
paths from different servers are therefore not required.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import random
import h5py
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from evaluations.runtime_config import DATA_ROOT


DATASETS_ROOT = Path(DATA_ROOT)
RANDOM_SEED = 42
SAMPLES_PER_DATASET = 100


def relative_dataset_path(path: str) -> str:
    normalized = str(path).replace("\\", "/")

    if "datasets/" in normalized:
        normalized = normalized.split("datasets/", 1)[1]

    return normalized.lstrip("/")


def read_video_paths(h5_paths: list[Path]) -> list[str]:
    paths: set[str] = set()

    for h5_path in h5_paths:
        if not h5_path.is_file():
            print(f"[warning] H5 file not found: {h5_path}")
            continue

        with h5py.File(h5_path, "r") as h5f:
            for sample_key in h5f.keys():
                video_path = h5f.attrs.get(
                    f"{sample_key}/video_path",
                    None,
                )

                if video_path is None:
                    continue

                if isinstance(video_path, bytes):
                    video_path = video_path.decode("utf-8")

                paths.add(relative_dataset_path(video_path))

    return sorted(paths)


def select_random_samples(
    h5_paths: list[Path],
    count: int,
    seed: int,
) -> list[str]:
    paths = read_video_paths(h5_paths)

    if len(paths) < count:
        raise ValueError(
            f"Requested {count} samples, but only {len(paths)} were found."
        )

    rng = random.Random(seed)
    return rng.sample(paths, count)


DEFAULT_SAMPLES: dict[str, list[str]] = {
    "grid": select_random_samples(
        [
            DATASETS_ROOT / "grid" / f"grid_test_features_chunk{i}.h5"
            for i in range(1, 5)
        ],
        SAMPLES_PER_DATASET,
        RANDOM_SEED,
    ),
    "lrs2": select_random_samples(
        [
            DATASETS_ROOT / "lrs2" / f"lrs2_test_features_chunk{i}.h5"
            for i in range(1, 3)
        ],
        SAMPLES_PER_DATASET,
        RANDOM_SEED + 1,
    ),
    "voxceleb2": select_random_samples(
        [
            DATASETS_ROOT
            / "vox2_short"
            / f"vox2_short_test_features_chunk{i}.h5"
            for i in range(1, 9)
        ],
        SAMPLES_PER_DATASET,
        RANDOM_SEED + 2,
    ),
}

# CLI name -> (figure label, project directory, proposed model flag)
MODEL_INFO: dict[str, tuple[str, str, bool]] = {
    "av_plc": ("Proposed", "AV_PLC", True),
    "av_lstm": ("Morrone et al.", "AV_LSTM", False),
    "av_s2s": ("Elyaderani et al.", "AV_S2S", False),
    "av_transformer": ("Montesinos et al.", "AV_Transformer", False),
}
DEFAULT_MODELS = list(MODEL_INFO)


def normalize_rel_path(path: str) -> str:
    """Return a portable forward-slash path relative to ``datasets/``."""
    normalized = str(path).replace("\\", "/")
    if "datasets/" in normalized:
        normalized = normalized.split("datasets/", 1)[1]
    return normalized.lstrip("/")


def scalar_string(value: Any) -> str:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    return str(value)


def dataset_for_sample(sample_path: str) -> str:
    path = normalize_rel_path(sample_path)
    if path.startswith("grid/"):
        return "grid"
    if path.startswith("lrs2/"):
        return "lrs2"
    if path.startswith("vox2_short/") or path.startswith("voxceleb2/"):
        return "voxceleb2"
    raise ValueError(f"Cannot infer dataset from sample path: {sample_path}")


def safe_name(sample_path: str) -> str:
    path = Path(normalize_rel_path(sample_path))
    return "__".join(path.with_suffix("").parts)


def condition_name(loss_rate: int) -> str:
    return f"test_ge_{loss_rate}"


def find_npz(model_dir: Path, condition: str, sample_path: str) -> Path:
    """Find the newest saved archive matching one portable sample path."""
    target = normalize_rel_path(sample_path)
    matches: list[Path] = []

    # Search recursively because each project may place condition folders under
    # a run/checkpoint-specific directory.
    for spec_root in model_dir.rglob("spectrograms"):
        if spec_root.parent.name != condition:
            continue
        for npz_path in spec_root.rglob("*.npz"):
            try:
                with np.load(npz_path, allow_pickle=False) as data:
                    path_key = next(
                        (
                            key
                            for key in ("relative_video_path", "video_path")
                            if key in data.files
                        ),
                        None,
                    )
                    if path_key is None:
                        continue
                    stored = normalize_rel_path(scalar_string(data[path_key]))
                    if stored == target:
                        matches.append(npz_path)
            except (OSError, ValueError):
                continue

    if not matches:
        raise FileNotFoundError(
            f"No output for '{target}' in {model_dir} under condition '{condition}'."
        )
    return max(matches, key=lambda path: path.stat().st_mtime)


def first_existing(data: np.lib.npyio.NpzFile, keys: Iterable[str]) -> str:
    for key in keys:
        if key in data.files:
            return key
    raise KeyError(f"None of these keys were found: {', '.join(keys)}")


def load_model_output(
    model_dir: Path,
    condition: str,
    sample_path: str,
    proposed: bool,
) -> dict[str, np.ndarray]:
    npz_path = find_npz(model_dir, condition, sample_path)
    with np.load(npz_path, allow_pickle=False) as data:
        if proposed:
            spec_key = first_existing(
                data,
                ("fused_merged_spec", "fused_spec", "merged_spec"),
            )
            audio_key = first_existing(
                data,
                ("fused_audio", "reconstructed_audio"),
            )
        else:
            spec_key = first_existing(
                data,
                ("merged_spec", "reconstructed_spec", "predicted_spec"),
            )
            audio_key = first_existing(data, ("reconstructed_audio",))

        required = (
            "original_spec",
            "masked_spec",
            "original_audio",
            "masked_audio",
            spec_key,
            audio_key,
        )
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"{npz_path} is missing: {', '.join(missing)}")

        return {
            "original_spec": np.asarray(data["original_spec"]).squeeze(),
            "masked_spec": np.asarray(data["masked_spec"]).squeeze(),
            "original_audio": np.asarray(data["original_audio"]).squeeze(),
            "masked_audio": np.asarray(data["masked_audio"]).squeeze(),
            "reconstructed_spec": np.asarray(data[spec_key]).squeeze(),
            "reconstructed_audio": np.asarray(data[audio_key]).squeeze(),
        }


def load_av_plc_heads(
    model_dir: Path,
    condition: str,
    sample_path: str,
) -> dict[str, np.ndarray]:
    """Load the fused, audio-only, and video-only AV-PLC outputs."""
    npz_path = find_npz(model_dir, condition, sample_path)
    with np.load(npz_path, allow_pickle=False) as data:
        fused_spec_key = first_existing(
            data,
            ("fused_merged_spec", "fused_predicted_spec", "fused_spec"),
        )
        audio_spec_key = first_existing(
            data,
            ("audio_merged_spec", "audio_predicted_spec", "rec_spec"),
        )
        video_spec_key = first_existing(
            data,
            ("video_merged_spec", "video_predicted_spec", "synth_spec"),
        )
        fused_audio_key = first_existing(
            data,
            ("fused_audio", "reconstructed_audio"),
        )
        audio_audio_key = first_existing(
            data,
            ("audio_only_audio", "rec_audio"),
        )
        video_audio_key = first_existing(
            data,
            ("video_only_audio", "synth_audio"),
        )

        required = (
            "original_spec",
            "masked_spec",
            "original_audio",
            "masked_audio",
            fused_spec_key,
            audio_spec_key,
            video_spec_key,
            fused_audio_key,
            audio_audio_key,
            video_audio_key,
        )
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"{npz_path} is missing: {', '.join(missing)}")

        return {
            "original_spec": np.asarray(data["original_spec"]).squeeze(),
            "masked_spec": np.asarray(data["masked_spec"]).squeeze(),
            "original_audio": np.asarray(data["original_audio"]).squeeze(),
            "masked_audio": np.asarray(data["masked_audio"]).squeeze(),
            "fused_spec": np.asarray(data[fused_spec_key]).squeeze(),
            "fused_audio": np.asarray(data[fused_audio_key]).squeeze(),
            "audio_only_spec": np.asarray(data[audio_spec_key]).squeeze(),
            "audio_only_audio": np.asarray(data[audio_audio_key]).squeeze(),
            "video_only_spec": np.asarray(data[video_spec_key]).squeeze(),
            "video_only_audio": np.asarray(data[video_audio_key]).squeeze(),
        }


def load_panels(
    models_root: Path,
    condition: str,
    sample_path: str,
    models: list[str],
    view: str,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    if view == "av_plc_heads":
        output = load_av_plc_heads(
            models_root / "AV_PLC",
            condition,
            sample_path,
        )
        return [
            ("Masked Input", output["masked_spec"], output["masked_audio"]),
            ("Fused", output["fused_spec"], output["fused_audio"]),
            (
                "Audio-only",
                output["audio_only_spec"],
                output["audio_only_audio"],
            ),
            (
                "Video-only",
                output["video_only_spec"],
                output["video_only_audio"],
            ),
            ("Ground Truth", output["original_spec"], output["original_audio"]),
        ]

    loaded: dict[str, dict[str, np.ndarray]] = {}
    for model in models:
        label, folder, proposed = MODEL_INFO[model]
        loaded[model] = load_model_output(
            models_root / folder,
            condition,
            sample_path,
            proposed,
        )

    # Any model archive contains the identical input and ground truth. Prefer
    # AV-PLC when included, otherwise use the first selected model.
    reference_key = "av_plc" if "av_plc" in loaded else models[0]
    reference = loaded[reference_key]

    panels: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("Input", reference["masked_spec"], reference["masked_audio"]),
    ]
    for model in models:
        label, _, _ = MODEL_INFO[model]
        panels.append(
            (
                label,
                loaded[model]["reconstructed_spec"],
                loaded[model]["reconstructed_audio"],
            )
        )
    panels.append(
        ("Ground Truth", reference["original_spec"], reference["original_audio"])
    )
    return panels


def plot_comparison(
    sample_groups: list[tuple[str, list[tuple[str, np.ndarray, np.ndarray]]]],
    output: Path,
    *,
    sample_rate: int,
    hop_length: int,
    title: str | None = None,
) -> None:
    """Plot one two-row block (Mel + waveform) per selected utterance."""
    if not sample_groups:
        raise ValueError("No samples were supplied for plotting.")

    column_count = len(sample_groups[0][1])
    if any(len(panels) != column_count for _, panels in sample_groups):
        raise ValueError("All samples must use the same selected model list.")

    row_count = 2 * len(sample_groups)
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(2.75 * column_count, 3.7 * len(sample_groups)),
        squeeze=False,
        constrained_layout=True,
    )

    for sample_index, (sample_label, panels) in enumerate(sample_groups):
        spec_row = 2 * sample_index
        wave_row = spec_row + 1

        specs = [panel[1] for panel in panels]
        audios = [panel[2] for panel in panels]
        vmin = min(float(np.nanmin(spec)) for spec in specs)
        vmax = max(float(np.nanmax(spec)) for spec in specs)
        peak = max(float(np.nanmax(np.abs(audio))) for audio in audios)
        peak = max(peak, 1e-6)

        for col, (panel_title, spec, audio) in enumerate(panels):
            spec_ax = axes[spec_row, col]
            wave_ax = axes[wave_row, col]

            duration = spec.shape[-1] * hop_length / sample_rate
            spec_ax.imshow(
                spec,
                origin="lower",
                aspect="auto",
                extent=(0.0, duration, 0, spec.shape[-2]),
                vmin=vmin,
                vmax=vmax,
            )
            if sample_index == 0:
                spec_ax.set_title(panel_title)
            spec_ax.set_xticks([])
            if col == 0:
                spec_ax.set_ylabel(f"{sample_label}\nMel bin")
            else:
                spec_ax.set_yticklabels([])

            time = np.arange(audio.size, dtype=np.float64) / sample_rate
            wave_ax.plot(time, audio, linewidth=0.65)
            wave_ax.set_xlim(0.0, time[-1] if time.size else 0.0)
            wave_ax.set_ylim(-peak, peak)
            wave_ax.set_xlabel("Time (s)")
            if col == 0:
                wave_ax.set_ylabel("Amplitude")
            else:
                wave_ax.set_yticklabels([])

    if title:
        fig.suptitle(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


def run_evaluations(
    models_root: Path,
    datasets: list[str],
    models: list[str],
    loss_rates: list[int],
    view: str,
) -> None:
    runner = models_root / "run_all_evaluations.py"
    if not runner.is_file():
        raise FileNotFoundError(f"Cannot find: {runner}")

    selected_models = ["av_plc"] if view == "av_plc_heads" else models

    for dataset in datasets:
        samples = DEFAULT_SAMPLES[dataset]
        cmd = [
            sys.executable,
            str(runner),
            "--models",
            *selected_models,
            "--datasets",
            dataset,
            "--mask-types",
            "gilbert",
            "--loss-rates",
            *map(str, loss_rates),
            "--save-output",
            "--sample-paths",
            *samples,
        ]
        print("\nRunning:")
        print(" ".join(cmd))
        subprocess.run(cmd, cwd=models_root, check=True)


def create_previews(args: argparse.Namespace) -> None:
    if args.run_evaluation:
        run_evaluations(
            args.models_root,
            args.datasets,
            args.models,
            args.loss_rates,
            args.view,
        )

    for dataset in args.datasets:
        for loss_rate in args.loss_rates:
            condition = condition_name(loss_rate)
            for sample in DEFAULT_SAMPLES[dataset]:
                try:
                    panels = load_panels(
                        args.models_root,
                        condition,
                        sample,
                        args.models,
                        args.view,
                    )
                except (FileNotFoundError, KeyError) as exc:
                    print(f"[skip] {sample}: {exc}", file=sys.stderr)
                    continue

                output = (
                    args.output_dir
                    / dataset
                    / f"ge_{loss_rate}"
                    / f"{safe_name(sample)}.png"
                )
                plot_comparison(
                    [(dataset.upper(), panels)],
                    output,
                    sample_rate=args.sample_rate,
                    hop_length=args.hop_length,
                    title=f"{normalize_rel_path(sample)} | GE {loss_rate}%",
                )


def create_final(args: argparse.Namespace) -> None:
    condition = condition_name(args.loss_rate)
    groups = []
    for sample in args.samples:
        panels = load_panels(
            args.models_root,
            condition,
            sample,
            args.models,
            args.view,
        )
        dataset = dataset_for_sample(sample)
        short_id = Path(normalize_rel_path(sample)).stem
        groups.append((f"{dataset.upper()}\n{short_id}", panels))

    plot_comparison(
        groups,
        args.output,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        title=f"Gilbert–Elliott loss rate: {args.loss_rate}%",
    )


def add_common_plot_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--models-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help=(
            "Directory containing run_all_evaluations.py and all model projects. "
            "Defaults to the directory containing this script."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_INFO),
        default=DEFAULT_MODELS,
        help="Models to include, in the desired figure order.",
    )
    parser.add_argument(
        "--view",
        choices=("all_models", "av_plc_heads"),
        default="all_models",
        help=(
            "Use 'all_models' for the selected model comparison, or "
            "'av_plc_heads' for Masked Input, Fused, Audio-only, "
            "Video-only, and Ground Truth."
        ),
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--hop-length", type=int, default=160)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    preview = subparsers.add_parser(
        "preview",
        help="Optionally run inference, then make one comparison per utterance.",
    )
    add_common_plot_args(preview)
    preview.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DEFAULT_SAMPLES),
        default=list(DEFAULT_SAMPLES),
    )
    preview.add_argument("--loss-rates", nargs="+", type=int, default=[30, 40])
    preview.add_argument(
        "--run-evaluation",
        action="store_true",
        help="Run run_all_evaluations.py before creating previews.",
    )
    preview.add_argument(
        "--output-dir",
        type=Path,
        default=Path("qualitative_previews"),
    )
    preview.set_defaults(func=create_previews)

    final = subparsers.add_parser(
        "final",
        help="Combine user-selected utterances into one publication figure.",
    )
    add_common_plot_args(final)
    final.add_argument("--loss-rate", type=int, required=True)
    final.add_argument("--samples", nargs="+", required=True)
    final.add_argument(
        "--output",
        type=Path,
        default=Path("qualitative_final.png"),
    )
    final.set_defaults(func=create_final)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

# python run_qualitative_evaluations.py preview \
# --run-evaluation \
# --datasets grid lrs2 voxceleb2 \
# --loss-rates 40 50 \
# --view all_models \
# --models av_plc av_lstm av_s2s av_transformer

# python run_qualitative_evaluations.py final \
#   --view all_models \
#   --loss-rates 40 \
#   --samples \
#     grid/test/s30/bbai9p.mpg \
#     lrs2/lrs2_v1/mvlrs_v1/main/6330311066473698535/00018.mp4 \
#     vox2_short/vox2_test_mp4/id03789/go_QOzc79Uc/00313.mp4 \
#   --models av_plc av_lstm av_transformer \
#   --output comparisons.png

# python run_qualitative_evaluations.py preview \
#   --run-evaluation \
#   --datasets grid lrs2 voxceleb2 \
#   --loss-rates 30 \
#   --view av_plc_heads

# python run_qualitative_evaluations.py final \
#   --view av_plc_heads \
#   --loss-rate 30 \
#   --samples \
#     grid/test/s30/bbai9p.mpg \
#     lrs2/lrs2_v1/mvlrs_v1/main/6330311066473698535/00018.mp4 \
#     vox2_short/vox2_test_mp4/id03789/go_QOzc79Uc/00313.mp4 \
#   --output av_plc_heads_final.png