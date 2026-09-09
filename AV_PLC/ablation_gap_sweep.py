import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from evaluations.runtime_config import project_log_dir, project_checkpoint_dir, set_global_seed,  DATA_ROOT, SEED

import os
import json
import csv
import argparse
import hashlib

import matplotlib as mpl
mpl.use('Agg')  # Headless backend - no GUI
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import numpy as np
from scipy.stats import t
import torch
from torch.utils.data import DataLoader, Dataset

from audio_encoder import Audio_Encoder
from multimodal_decoder import AV_PLC
from av_dataloader import AVDataloader
from shared.metrics import calculate_pesq, calculate_stoi
from shared.audio_processing import load_audio_ffmpeg, librosa_mel2audio


mpl.rcParams.update({
    "font.family":               "serif",
    "font.serif":                ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset":          "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus":        False,
    # ── sizes ──────────────────────────────────
    "font.size":                 9,
    "axes.titlesize":            9,
    "axes.labelsize":            9,
    "xtick.labelsize":           9,
    "ytick.labelsize":           9,
    "legend.fontsize":           9,
    "legend.title_fontsize":     9,
    # ── lines ──────────────────────────────────
    "lines.linewidth":           1.5,
    "axes.linewidth":            0.6,
    "patch.linewidth":           0.5,
    # ── ticks ──────────────────────────────────
    "xtick.major.width":         0.6,
    "ytick.major.width":         0.6,
    "xtick.minor.width":         0.4,
    "ytick.minor.width":         0.4,
    "xtick.major.size":          3.0,
    "ytick.major.size":          3.0,
    "xtick.minor.size":          1.5,
    "ytick.minor.size":          1.5,
    "xtick.direction":           "in",
    "ytick.direction":           "in",
    "xtick.minor.visible":       False,
    # ── legend ─────────────────────────────────
    "legend.frameon":            False,
    "legend.borderpad":          0.4,
    "legend.labelspacing":       0.3,
    "legend.handlelength":       1.8,
    "legend.handletextpad":      0.5,
    # ── saving ─────────────────────────────────
    "savefig.dpi":               300,
    "savefig.bbox":              "tight",
})
# ============================================================================
# PLOT STYLE CONSTANTS (configurable for paper/screen output)
# ============================================================================

# Data line widths
DATA_LINE_WIDTH = 1.5                    # Main data lines (recommended for 300 dpi)
AUDIO_ONLY_LINE_WIDTH = 1.5              # Reference/baseline lines
AV_LINE_WIDTH = 1.5                      # AV method lines

# Confidence interval bands
CI_BAND_LINEWIDTH = 0                    # Edgeless bands look cleaner in print

# Grid lines
GRID_MAJOR_LINEWIDTH = 0.6               # Slightly lighter than data
GRID_MINOR_LINEWIDTH = 0.4               # Proportional to major

# Reference lines (axes, zero-lines)
REFERENCE_LINE_WIDTH = 0.8               # Context, not primary data

# Markers
MARKER_SIZE = 4.5                        # Filled markers (cleaner at 9pt base)
MARKER_EDGE_WIDTH = 0.5                  # Thin white ring for separation

# Legend
LEGEND_LINE_WIDTH = 1.5                  # Match data line width

# ── Figure dimensions (inches) ─────────────────────────────────────────────
# IEEE / Elsevier / Springer column widths: 88 mm single / 180 mm double.
FIG_WIDTH_1COL  = 3.46   # 88 mm  – single-column figure
FIG_WIDTH_2COL  = 7.09   # 180 mm – two-column / full-width figure
FIG_HEIGHT_1ROW = 2.60   # single panel row (≈ 0.75 × single-col width)
FIG_HEIGHT_ROW  = 2.20   # per-row height for multi-row stacked figures

# Detailed single-gap sweep. All masks come from AVDataset.
GAP_LENGTHS_MS = [20, 40, 80, 160, 320, 500, 750, 1000, 1250, 1500]

# ---------------------------
# Dataset single-gap masking
# ---------------------------

def gap_bounds_from_mask(mask):
    """Return the inclusive-exclusive missing-frame bounds from a dataset mask."""
    time_mask = np.asarray(mask[0], dtype=np.float32)
    missing = np.flatnonzero(time_mask < 0.5)
    if missing.size == 0:
        raise ValueError("Single-gap mask contains no missing frames.")
    if not np.all(np.diff(missing) == 1):
        raise ValueError("Expected one contiguous single gap, but mask has multiple gaps.")
    return int(missing[0]), int(missing[-1] + 1)


def build_single_gap_loader(
    dataset_name: str,
    mode: str,
    gap_ms: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    test_subset=None,
):
    """Build the project's dataset-native deterministic single-gap loader."""
    builder = AVDataloader(
        dataset_name=dataset_name,
        mode=mode,
        batch_size=batch_size,
        num_workers=num_workers,
        video_aug=False,
        dropout_modality=False,
        test_subset=test_subset,
    )
    return builder.test_dataloader(
        mask_range="10",
        mask_type="single_gap",
        gap_ms=int(gap_ms),
        seed=int(seed),
    )


# ---------------------------
# Model loading
# ---------------------------

def load_checkpoint_flexible(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        state = (
            ckpt.get("model_state_dict")
            or ckpt.get("model_state")
            or ckpt.get("state_dict")
            or ckpt
        )
    else:
        state = ckpt

    # Remove common wrappers if present.
    clean_state = {}
    for k, v in state.items():
        k = k.replace("module.", "")
        clean_state[k] = v

    model.load_state_dict(clean_state, strict=True)

    print(f"Loaded checkpoint strictly: {ckpt_path}")

    return model


def build_audio_model(dataset_name: str, device):
    """
    from  audio_encoder.py:
    grid uses fewer conformer blocks, non-grid uses deeper audio encoder.
    """
    conformer_blocks = 4 if dataset_name == "grid" else 8
    num_heads = 4

    model = Audio_Encoder(
        conformer_block=conformer_blocks,
        num_heads=num_heads,
    ).to(device)

    return model


def build_av_model(dataset_name: str, device):
    """
    Same configuration style as infer.py.
    """
    model = AV_PLC(
        video_depth=6,
        video_heads=4,
        audio_depth=4,
        audio_heads=4,
        video_hidden_size=256,
        audio_hidden_size=256,
        feat_dim=256,
        audio_ckpt_path=None,
        freeze_audio_enc=False,
    ).to(device)

    return model


def default_audio_model_name(dataset_name: str):
    # Based on audio_encoder.py when l2s_flag=False and losses are disabled.
    return f"audio_bursty_plc({dataset_name})"


def default_av_model_name(dataset_name: str):
    # Based on infer.py.
    return (
        f"av_wide_masking_mlp_av_only_fusion_5loss_bursty2"
        f"_plc_a0.05_v0.1"
        f"_pesq_0.01"
        f"_asr_0.1"
        f"({dataset_name})"
    )


def get_checkpoint_path(checkpoint_dir, model_name):
    return os.path.join(checkpoint_dir, model_name, "best_model.pt")


# ---------------------------
# Audio path handling
# ---------------------------

def resolve_audio_path(video_path: str, dataset_root: str):
    """
    Your metrics.py reconstructs the audio path from video_path.
    This version keeps it explicit and configurable.

    If video_path already exists, use it.
    Otherwise, replace the prefix before 'datasets' with dataset_root.
    """
    if isinstance(video_path, bytes):
        video_path = video_path.decode("utf-8")

    if os.path.exists(video_path):
        return video_path

    if "datasets" in video_path:
        rel_path = video_path.split("datasets", 1)[-1].lstrip(os.sep)
        candidate = os.path.join(dataset_root, rel_path)
        if os.path.exists(candidate):
            return candidate

    return video_path


# ---------------------------
# Reconstruction and plotting helpers
# ---------------------------

def _get_dataset_stats(loader):
    dataset = loader.dataset
    while hasattr(dataset, "base_dataset"):
        dataset = dataset.base_dataset
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return (
        float(getattr(dataset, "mel_mean", -56.775)),
        float(getattr(dataset, "mel_std", 19.707)),
    )


def insert_reconstructed_gap(original_audio, reconstructed_audio, mask, hop_length=160):
    """Insert only the reconstructed mel-gap region into the original waveform."""
    original_audio = np.asarray(original_audio, dtype=np.float32).squeeze()
    reconstructed_audio = np.asarray(reconstructed_audio, dtype=np.float32).squeeze()

    if torch.is_tensor(mask):
        time_keep = mask[0].detach().cpu().numpy().astype(np.float32)
    else:
        time_keep = np.asarray(mask[0], dtype=np.float32)

    waveform_mask = np.repeat(time_keep, hop_length)
    n = min(len(original_audio), len(reconstructed_audio), len(waveform_mask))
    return (
        original_audio[:n] * waveform_mask[:n]
        + reconstructed_audio[:n] * (1.0 - waveform_mask[:n])
    )


def _configure_gap_axis(ax):
    ax.xaxis.set_major_locator(MaxNLocator(nbins=9, integer=True))
    # ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(which="major", alpha=0.45, linewidth=GRID_MAJOR_LINEWIDTH)
    ax.grid(which="minor", alpha=0.20, linewidth=GRID_MINOR_LINEWIDTH, linestyle=":")
    ax.tick_params(which="both", direction="in")


def save_gap_summary_plots(dataset_results, output_dir, dataset_name):
    """Plot dataset summaries using 95% confidence intervals."""
    if not dataset_results:
        return

    dataset_results = sorted(dataset_results, key=lambda row: int(row["gap_ms"]))
    gap_ms = np.asarray([row["gap_ms"] for row in dataset_results], dtype=np.float64)

    figure_dir = os.path.join(output_dir, dataset_name, "figures")
    os.makedirs(figure_dir, exist_ok=True)

    for metric, label in (("pesq", "PESQ"), ("stoi", "STOI")):
        audio_mean = np.asarray(
            [row["audio_only"][f"{metric}_mean"] for row in dataset_results],
            dtype=np.float64,
        )
        audio_ci95 = np.asarray(
            [row["audio_only"][f"{metric}_ci95"] for row in dataset_results],
            dtype=np.float64,
        )
        av_mean = np.asarray(
            [row["audio_visual"][f"{metric}_mean"] for row in dataset_results],
            dtype=np.float64,
        )
        av_ci95 = np.asarray(
            [row["audio_visual"][f"{metric}_ci95"] for row in dataset_results],
            dtype=np.float64,
        )

        fig, ax = plt.subplots(figsize=(FIG_WIDTH_1COL, FIG_HEIGHT_1ROW))

        # Audio-only: dashed mean line with a shaded 95% CI band.
        audio_line, = ax.plot(
            gap_ms,
            audio_mean,
            marker="o",
            linestyle="--",
            color="#808080",   # gray,
            linewidth=AUDIO_ONLY_LINE_WIDTH,
            markersize=MARKER_SIZE,
            markeredgewidth=MARKER_EDGE_WIDTH,
            label="Audio-only",
        )
        ax.fill_between(
            gap_ms,
            audio_mean - audio_ci95,
            audio_mean + audio_ci95,
            alpha=0.35,
            color=audio_line.get_color(),
            linewidth=CI_BAND_LINEWIDTH,
            zorder=1,
        )

        # ax.errorbar(
        #     gap_ms,
        #     av_mean,
        #     yerr=av_ci95,
        #     marker="s",
        #     linestyle="-",
        #     capsize=5,
        #     elinewidth=1.8,
        #     capthick=1.8,
        #     zorder=3,
        #     label="Audio-visual",
        # )

        # Audio-visual: solid mean line with shaded 95% CI band.
        av_line, = ax.plot(
            gap_ms,
            av_mean,
            marker="s",
            linestyle="-",
            color="#0072B2", # blue
            linewidth=DATA_LINE_WIDTH,
            markersize=MARKER_SIZE,
            markeredgewidth=MARKER_EDGE_WIDTH,
            label="Audio-visual",
        )
        ax.fill_between(
            gap_ms,
            av_mean - av_ci95,
            av_mean + av_ci95,
            alpha=0.25,
            color=av_line.get_color(),
            linewidth=0,
        )

        ax.set_xlabel("Audio gap duration (ms)")
        ax.set_ylabel(rf"{label} ($\uparrow$)")
        _configure_gap_axis(ax)

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=2,
            frameon=False,
        )
        fig.tight_layout()
        fig.savefig(
            os.path.join(figure_dir, f"{metric}_vs_gap_duration_ci95.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

def save_combined_datasets_plot(all_results, output_dir):
    """
    Create one combined N x 2 figure:

        rows    = datasets
        column 0 = PESQ
        column 1 = STOI

    A single shared legend is placed below the whole figure.
    """
    preferred_order = ["grid", "lrs2", "voxceleb2"]
    available_datasets = {
        str(row["dataset"]).lower()
        for row in all_results
    }

    datasets = [
        name for name in preferred_order
        if name in available_datasets
    ]
    datasets.extend(
        sorted(available_datasets - set(datasets))
    )

    if not datasets:
        return

    metrics = [
        ("pesq", r"PESQ ($\uparrow$)"),
        ("stoi", r"STOI ($\uparrow$)"),
    ]

    fig, axes = plt.subplots(
        nrows=len(datasets),
        ncols=2,
        figsize=(FIG_WIDTH_2COL, FIG_HEIGHT_ROW * len(datasets)),
        squeeze=False,
        sharex="col",
        sharey="col",
    )

    legend_handles = None
    legend_labels = None

    display_names = {
        "grid": "GRID",
        "lrs2": "LRS2",
        "voxceleb2": "VoxCeleb2",
    }

    for row_index, dataset_name in enumerate(datasets):
        dataset_rows = sorted(
            [
                row for row in all_results
                if str(row["dataset"]).lower() == dataset_name
            ],
            key=lambda row: int(row["gap_ms"]),
        )

        if not dataset_rows:
            continue

        gap_ms = np.asarray(
            [row["gap_ms"] for row in dataset_rows],
            dtype=np.float64,
        )

        for column_index, (metric, ylabel) in enumerate(metrics):
            ax = axes[row_index, column_index]

            audio_mean = np.asarray(
                [
                    row["audio_only"][f"{metric}_mean"]
                    for row in dataset_rows
                ],
                dtype=np.float64,
            )
            audio_ci95 = np.asarray(
                [
                    row["audio_only"][f"{metric}_ci95"]
                    for row in dataset_rows
                ],
                dtype=np.float64,
            )

            av_mean = np.asarray(
                [
                    row["audio_visual"][f"{metric}_mean"]
                    for row in dataset_rows
                ],
                dtype=np.float64,
            )
            av_ci95 = np.asarray(
                [
                    row["audio_visual"][f"{metric}_ci95"]
                    for row in dataset_rows
                ],
                dtype=np.float64,
            )

            # Audio-only: dashed mean line and shaded 95% CI.
            audio_line, = ax.plot(
                gap_ms,
                audio_mean,
                marker="o",
                linestyle="--",
                color="#808080", # gray
                linewidth=AUDIO_ONLY_LINE_WIDTH,
                markersize=MARKER_SIZE,
                markeredgewidth=MARKER_EDGE_WIDTH,
                label="Audio-only",
                zorder=3,
            )
            ax.fill_between(
                gap_ms,
                audio_mean - audio_ci95,
                audio_mean + audio_ci95,
                color=audio_line.get_color(),
                alpha=0.25,
                linewidth=CI_BAND_LINEWIDTH,
                zorder=1,
            )

            # Audio-visual: solid mean line and shaded 95% CI.
            av_line, = ax.plot(
                gap_ms,
                av_mean,
                marker="s",
                linestyle="-",
                color="#0072B2", # blue
                linewidth=DATA_LINE_WIDTH,
                markersize=MARKER_SIZE,
                markeredgewidth=MARKER_EDGE_WIDTH,
                label="Audio-visual",
                zorder=3,
            )
            ax.fill_between(
                gap_ms,
                av_mean - av_ci95,
                av_mean + av_ci95,
                color=av_line.get_color(),
                alpha=0.25,
                linewidth=0,
                zorder=1,
            )

            _configure_gap_axis(ax)

            # One major vertical grid line at every tested gap.
            ax.set_xticks(gap_ms)
            ax.set_xticklabels(
                ["" if int(x) == 40 else str(int(x)) for x in gap_ms],
                rotation=90,
                ha="right",
            )

            # More horizontal grid levels.
            ax.yaxis.set_major_locator(MaxNLocator(nbins=12))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))

            ax.grid(
                which="major",
                axis="both",
                alpha=0.45,
                linewidth=GRID_MAJOR_LINEWIDTH,
            )

            ax.grid(
                which="minor",
                axis="y",
                alpha=0.20,
                linewidth=GRID_MINOR_LINEWIDTH,
                linestyle=":",
            )
            # Metric names above the two columns.
            if row_index == 0:
                ax.set_title(ylabel)

            # Dataset name on the left side of each row.
            if column_index == 0:
                ax.set_ylabel(
                    display_names.get(dataset_name, dataset_name),
                    fontweight="bold",
                )

            # X-axis labels only on the last row.
            if row_index == len(datasets) - 1:
                ax.set_xlabel("Audio gap duration (ms)")

            if legend_handles is None:
                legend_handles, legend_labels = (
                    ax.get_legend_handles_labels()
                )

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
    )

    fig.tight_layout(rect=(0, 0.06, 1, 1))

    fig.savefig(
        os.path.join(
            output_dir,
            "combined_datasets_pesq_stoi_ci95.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
# ---------------------------
# Evaluation
# ---------------------------

@torch.no_grad()
def evaluate_audio_model(
    model,
    loader,
    #vocoder,
    device,
    dataset_root,
    sample_rate=16000,
):
    model.eval()
    mel_mean, mel_std = _get_dataset_stats(loader)

    all_rows = []
    pesq_values = []
    stoi_values = []

    for batch_idx, batch in enumerate(loader):
        masked_spec, mel_spec, audio_length, text, mask, video_path = batch

        masked_spec = masked_spec.to(device).float()
        mel_spec = mel_spec.to(device).float()

        pred_mel, _ = model(masked_spec)

        B = pred_mel.size(0)

        for i in range(B):
            path_i = resolve_audio_path(video_path[i], dataset_root)
            ref_audio = load_audio_ffmpeg(path_i, sr=sample_rate, fixlen_sec=3)

            reconstructed_audio = librosa_mel2audio(
                pred_mel[i].detach().cpu(), sr=sample_rate, mel_mean=mel_mean, mel_std=mel_std)
            if torch.is_tensor(reconstructed_audio):
                reconstructed_audio = reconstructed_audio.detach().cpu().numpy()

            pred_audio = insert_reconstructed_gap(
                original_audio=ref_audio,
                reconstructed_audio=reconstructed_audio,
                mask=mask[i],
                hop_length=160,
            )
            n = min(len(ref_audio), len(pred_audio))
            pesq_score = calculate_pesq(ref_audio[:n], pred_audio[:n], sr=sample_rate)
            stoi_score = calculate_stoi(ref_audio[:n], pred_audio[:n], sr=sample_rate)

            gap_start_frame, gap_end_frame = gap_bounds_from_mask(mask[i])
            row = {
                "batch_idx": int(batch_idx),
                "sample_in_batch": int(i),
                "video_path": str(video_path[i]),
                "gap_start_frame": gap_start_frame,
                "gap_end_frame": gap_end_frame,
                "pesq": None if pesq_score is None else float(pesq_score),
                "stoi": None if stoi_score is None else float(stoi_score),
            }
            all_rows.append(row)

            if pesq_score is not None and np.isfinite(pesq_score):
                pesq_values.append(float(pesq_score))
            if stoi_score is not None and np.isfinite(stoi_score):
                stoi_values.append(float(stoi_score))

    summary = summarize_per_sample_rows(all_rows)

    return summary, all_rows


@torch.no_grad()
def evaluate_av_model(
    model,
    loader,
    #vocoder,
    device,
    dataset_root,
    sample_rate=16000,
):
    model.eval()
    mel_mean, mel_std = _get_dataset_stats(loader)

    all_rows = []
    pesq_values = []
    stoi_values = []

    for batch_idx, batch in enumerate(loader):
        (
            frames,
            spk_emb,
            masked_spec,
            mel_spec,
            audio_length,
            text,
            mask,
            video_path,
            avail,
        ) = batch

        frames = frames.to(device).float()
        spk_emb = spk_emb.to(device).float()
        masked_spec = masked_spec.to(device).float()
        mel_spec = mel_spec.to(device).float()
        avail = avail.to(device).bool()

        fused_mel, _, _ = model(
            masked_spec,
            frames,
            spk_emb,
            audio_length,
            avail=avail,
        )

        B = fused_mel.size(0)

        for i in range(B):
            path_i = resolve_audio_path(video_path[i], dataset_root)
            ref_audio = load_audio_ffmpeg(path_i, sr=sample_rate, fixlen_sec=3)

            reconstructed_audio = librosa_mel2audio(
                fused_mel[i].detach().cpu(), sr=sample_rate, mel_mean=mel_mean, mel_std=mel_std)
            if torch.is_tensor(reconstructed_audio):
                reconstructed_audio = reconstructed_audio.detach().cpu().numpy()

            pred_audio = insert_reconstructed_gap(
                original_audio=ref_audio,
                reconstructed_audio=reconstructed_audio,
                mask=mask[i],
                hop_length=160,
            )
            n = min(len(ref_audio), len(pred_audio))
            pesq_score = calculate_pesq(ref_audio[:n], pred_audio[:n], sr=sample_rate)
            stoi_score = calculate_stoi(ref_audio[:n], pred_audio[:n], sr=sample_rate)

            gap_start_frame, gap_end_frame = gap_bounds_from_mask(mask[i])
            row = {
                "batch_idx": int(batch_idx),
                "sample_in_batch": int(i),
                "video_path": str(video_path[i]),
                "gap_start_frame": gap_start_frame,
                "gap_end_frame": gap_end_frame,
                "pesq": None if pesq_score is None else float(pesq_score),
                "stoi": None if stoi_score is None else float(stoi_score),
            }
            all_rows.append(row)

            if pesq_score is not None and np.isfinite(pesq_score):
                pesq_values.append(float(pesq_score))
            if stoi_score is not None and np.isfinite(stoi_score):
                stoi_values.append(float(stoi_score))

    summary = summarize_per_sample_rows(all_rows)

    return summary, all_rows


def sample_identity(rows):
    """Return a stable hash for ordered sample paths and gap boundaries."""
    identities = [
        {
            "video_path": str(row["video_path"]),
            "gap_start_frame": int(row["gap_start_frame"]),
            "gap_end_frame": int(row["gap_end_frame"]),
        }
        for row in rows
    ]
    payload = json.dumps(identities, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), identities


def validate_paired_rows(audio_rows, av_rows):
    """Ensure audio-only and AV evaluations used identical ordered samples/masks."""
    if len(audio_rows) != len(av_rows):
        raise ValueError(
            f"Audio/AV sample-count mismatch: {len(audio_rows)} vs {len(av_rows)}"
        )
    for index, (audio_row, av_row) in enumerate(zip(audio_rows, av_rows)):
        audio_identity = (
            str(audio_row["video_path"]),
            int(audio_row["gap_start_frame"]),
            int(audio_row["gap_end_frame"]),
        )
        av_identity = (
            str(av_row["video_path"]),
            int(av_row["gap_start_frame"]),
            int(av_row["gap_end_frame"]),
        )
        if audio_identity != av_identity:
            raise ValueError(
                "Audio-only and AV loaders are not sample/mask aligned at "
                f"index {index}: {audio_identity!r} != {av_identity!r}"
            )


def mean_std_ci95(values):
    """Return mean, population std, 95% CI half-width, and valid sample count."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = int(values.size)

    if n == 0:
        return None, None, None, 0

    mean = float(np.mean(values))
    if n == 1:
        return mean, 0.0, 0.0, 1

    std = float(np.std(values, ddof=0))
    sample_std = float(np.std(values, ddof=1))
    sem = sample_std / np.sqrt(n)
    ci95 = float(t.ppf(0.975, df=n - 1) * sem)
    return mean, std, ci95, n


def _optional_float(value):
    if value in (None, "", "None", "nan", "NaN"):
        return None
    parsed = float(value)
    return parsed if np.isfinite(parsed) else None


def load_rows_csv(path):
    """Load a per-sample result CSV and restore numeric fields used downstream."""
    with open(path, "r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    for row in rows:
        for key in ("batch_idx", "sample_in_batch", "gap_start_frame", "gap_end_frame"):
            row[key] = int(round(float(row[key])))
        row["pesq"] = _optional_float(row.get("pesq"))
        row["stoi"] = _optional_float(row.get("stoi"))
    return rows


def summarize_per_sample_rows(rows):
    """Compute PESQ/STOI mean, std, and 95% CI from per-sample rows."""
    pesq_values = [row["pesq"] for row in rows if row.get("pesq") is not None]
    stoi_values = [row["stoi"] for row in rows if row.get("stoi") is not None]

    pesq_mean, pesq_std, pesq_ci95, num_pesq = mean_std_ci95(pesq_values)
    stoi_mean, stoi_std, stoi_ci95, num_stoi = mean_std_ci95(stoi_values)

    return {
        "pesq_mean": pesq_mean,
        "pesq_std": pesq_std,
        "pesq_ci95": pesq_ci95,
        "stoi_mean": stoi_mean,
        "stoi_std": stoi_std,
        "stoi_ci95": stoi_ci95,
        "num_pesq": num_pesq,
        "num_stoi": num_stoi,
    }


def build_gap_result_from_rows(
    dataset_name,
    gap_ms,
    audio_rows,
    av_rows,
    audio_model_name,
    av_model_name,
    test_subset,
    sample_rate,
):
    """Validate paired rows and build one complete per-gap summary."""
    validate_paired_rows(audio_rows, av_rows)
    identity_hash, ordered_identities = sample_identity(audio_rows)
    audio_summary = summarize_per_sample_rows(audio_rows)
    av_summary = summarize_per_sample_rows(av_rows)

    result = {
        "dataset": dataset_name,
        "gap_ms": int(gap_ms),
        "mask_type": "single_gap",
        "mask_source": "AVDataset.generate_single_gap_mask",
        "mask_seed": int(SEED),
        "test_subset": test_subset,
        "sample_rate": int(sample_rate),
        "hop_length": 160,
        "num_samples": len(audio_rows),
        "sample_identity_sha256": identity_hash,
        "ordered_sample_identities": ordered_identities,
        "audio_model": audio_model_name,
        "av_model": av_model_name,
        "audio_only": audio_summary,
        "audio_visual": av_summary,
        "delta_pesq_av_minus_audio": None,
        "delta_stoi_av_minus_audio": None,
    }

    if audio_summary["pesq_mean"] is not None and av_summary["pesq_mean"] is not None:
        result["delta_pesq_av_minus_audio"] = (
            av_summary["pesq_mean"] - audio_summary["pesq_mean"]
        )
    if audio_summary["stoi_mean"] is not None and av_summary["stoi_mean"] is not None:
        result["delta_stoi_av_minus_audio"] = (
            av_summary["stoi_mean"] - audio_summary["stoi_mean"]
        )
    return result



def saved_gap_is_compatible(gap_dir, expected_test_subset, expected_sample_rate):
    """Return True when both CSVs exist and saved metadata matches this run."""
    audio_csv = os.path.join(gap_dir, "audio_only_per_sample.csv")
    av_csv = os.path.join(gap_dir, "audio_visual_per_sample.csv")
    if not (os.path.isfile(audio_csv) and os.path.isfile(av_csv)):
        return False

    summary_path = os.path.join(gap_dir, "summary.json")
    if not os.path.isfile(summary_path):
        # Older saved runs may have CSVs but no summary metadata. The CSVs can
        # still be summarized, but compatibility cannot be fully verified.
        print(f"Warning: no summary.json in {gap_dir}; reusing existing paired CSVs.")
        return True

    with open(summary_path, "r", encoding="utf-8") as file:
        saved = json.load(file)

    return (
        saved.get("mask_type") == "single_gap"
        and int(saved.get("mask_seed", -1)) == int(SEED)
        and saved.get("test_subset") == expected_test_subset
        and int(saved.get("sample_rate", expected_sample_rate)) == int(expected_sample_rate)
        and int(saved.get("hop_length", 160)) == 160
    )

def flatten_dataset_summary(dataset_results):
    """Convert nested per-gap summaries into rows for dataset-level summary.csv."""
    rows = []
    for result in sorted(dataset_results, key=lambda item: int(item["gap_ms"])):
        row = {
            "dataset": result["dataset"],
            "gap_ms": result["gap_ms"],
            "num_samples": result["num_samples"],
            "audio_model": result["audio_model"],
            "av_model": result["av_model"],
        }
        for model_key in ("audio_only", "audio_visual"):
            for stat_key, value in result[model_key].items():
                row[f"{model_key}_{stat_key}"] = value
        row["delta_pesq_av_minus_audio"] = result["delta_pesq_av_minus_audio"]
        row["delta_stoi_av_minus_audio"] = result["delta_stoi_av_minus_audio"]
        rows.append(row)
    return rows


# ---------------------------
# Saving
# ---------------------------

def save_rows_csv(rows, out_path):
    if not rows:
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    keys = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_csv(rows, out_path):
    if not rows:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(obj, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(obj, f, indent=2)


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets", nargs="+", default=["grid", "lrs2", "voxceleb2"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--checkpoint_dir", type=str, default=project_checkpoint_dir("AV_PLC"),)
    parser.add_argument("--results_dir", type=str, default=os.path.join(project_log_dir("AV_PLC"), "ablation_gap_sweep"),)

    parser.add_argument(
        "--vocoder_path",
        type=str,
        default=None #"/home/ai/Projects/Mahsa/sources/AV_PLC/hifigan/checkpoints/model-best.pt",
    )

    parser.add_argument(
        "--dataset_root",
        type=str,
        default=str(DATA_ROOT),
        help="Root used to resolve video/audio paths.",
    )

    parser.add_argument("--sample_rate", type=int, default=16000)

    # Useful for debugging before running full test set.
    parser.add_argument("--test_subset", type=int, default=None)

    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    set_global_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #vocoder = Vocoder(args.vocoder_path, device=device)

    final_summary = []

    for dataset_name in args.datasets:
        print("\n==============================")
        print(f"Dataset: {dataset_name}")
        print("==============================")

        audio_model_name = default_audio_model_name(dataset_name)
        av_model_name = default_av_model_name(dataset_name)

        missing_gaps = []
        for gap_ms in GAP_LENGTHS_MS:
            gap_dir = os.path.join(args.results_dir, dataset_name, f"gap_{gap_ms}ms")
            if not saved_gap_is_compatible(
                gap_dir=gap_dir,
                expected_test_subset=args.test_subset,
                expected_sample_rate=args.sample_rate,
            ):
                missing_gaps.append(gap_ms)

        audio_model = None
        av_model = None
        if missing_gaps:
            print(f"Inference required for gaps: {missing_gaps}")

            audio_ckpt = get_checkpoint_path(args.checkpoint_dir, audio_model_name)
            audio_model = build_audio_model(dataset_name, device)
            audio_model = load_checkpoint_flexible(audio_model, audio_ckpt, device)

            av_ckpt = get_checkpoint_path(args.checkpoint_dir, av_model_name)
            av_model = build_av_model(dataset_name, device)
            av_model = load_checkpoint_flexible(av_model, av_ckpt, device)
        else:
            print("All per-sample CSV files exist; skipping model loading and inference.")

        # Run only missing conditions. If either CSV is missing, recompute both
        # branches for that gap to preserve paired samples and masks.
        for gap_ms in missing_gaps:
            print(f"\n--- Gap length: {gap_ms} ms ---")
            gap_dir = os.path.join(args.results_dir, dataset_name, f"gap_{gap_ms}ms")

            audio_loader = build_single_gap_loader(
                dataset_name=dataset_name,
                mode="a",
                gap_ms=gap_ms,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=SEED,
                test_subset=args.test_subset,
            )
            av_loader = build_single_gap_loader(
                dataset_name=dataset_name,
                mode="av",
                gap_ms=gap_ms,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=SEED,
                test_subset=args.test_subset,
            )

            _, audio_rows = evaluate_audio_model(
                model=audio_model,
                loader=audio_loader,
                device=device,
                dataset_root=args.dataset_root,
                sample_rate=args.sample_rate,
            )
            _, av_rows = evaluate_av_model(
                model=av_model,
                loader=av_loader,
                device=device,
                dataset_root=args.dataset_root,
                sample_rate=args.sample_rate,
            )

            validate_paired_rows(audio_rows, av_rows)
            save_rows_csv(audio_rows, os.path.join(gap_dir, "audio_only_per_sample.csv"))
            save_rows_csv(av_rows, os.path.join(gap_dir, "audio_visual_per_sample.csv"))

        # The saved per-sample CSV files are the source of truth for summaries
        # and plots, regardless of whether inference ran in this invocation.
        dataset_results = []
        for gap_ms in GAP_LENGTHS_MS:
            gap_dir = os.path.join(args.results_dir, dataset_name, f"gap_{gap_ms}ms")
            audio_csv = os.path.join(gap_dir, "audio_only_per_sample.csv")
            av_csv = os.path.join(gap_dir, "audio_visual_per_sample.csv")

            if not os.path.isfile(audio_csv) or not os.path.isfile(av_csv):
                raise FileNotFoundError(
                    f"Missing per-sample CSV files for {dataset_name}, gap={gap_ms} ms"
                )

            audio_rows = load_rows_csv(audio_csv)
            av_rows = load_rows_csv(av_csv)
            gap_result = build_gap_result_from_rows(
                dataset_name=dataset_name,
                gap_ms=gap_ms,
                audio_rows=audio_rows,
                av_rows=av_rows,
                audio_model_name=audio_model_name,
                av_model_name=av_model_name,
                test_subset=args.test_subset,
                sample_rate=args.sample_rate,
            )
            dataset_results.append(gap_result)
            final_summary.append(gap_result)

            # Refresh the existing per-gap summary using CSV-derived statistics.
            save_json(gap_result, os.path.join(gap_dir, "summary.json"))

            print(f"Gap {gap_ms} ms")
            print("Audio-only:", gap_result["audio_only"])
            print("Audio-visual:", gap_result["audio_visual"])

        dataset_dir = os.path.join(args.results_dir, dataset_name)
        save_summary_csv(
            flatten_dataset_summary(dataset_results),
            os.path.join(dataset_dir, "summary.csv"),
        )
        save_gap_summary_plots(
            dataset_results=dataset_results,
            output_dir=args.results_dir,
            dataset_name=dataset_name,
        )

        # Release checkpoints before moving to the next dataset.
        del audio_model, av_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_json(
        final_summary,
        os.path.join(args.results_dir, "all_results_summary.json"),
    )

    save_combined_datasets_plot(
        all_results=final_summary,
        output_dir=args.results_dir,
    )
    print(f"\nSaved all results to: {args.results_dir}")


if __name__ == "__main__":
    main()

#Debug:

# python ablation_gap_sweep.py \
#  --datasets grid \
#  --batch_size 2 \
#  --num_workers 0 \
#  --test_subset 10

# Main:

# python ablation_gap_sweep.py \
#  --datasets grid lrs2 voxceleb2 \
#  --batch_size 8 \
#  --num_workers 4 \