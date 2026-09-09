"""
Combined uniform and local video-frame reduction ablation for AV_PLC.

This script evaluates the same AV model and the same deterministic single audio
 gap under two video reduction strategies:

1. Uniform reduction across the complete video clip.
2. Local reduction only inside a gap-centred temporal window.

Both strategies use the same retention fractions. The 100% full-video condition
is evaluated only once per dataset/gap and reused as the endpoint of both curves.

The AV model is not re-evaluated as audio-only. Instead, matching audio-only
results from ablation_gap_sweep.py are loaded and shown as horizontal baselines.
"""

import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from evaluations.runtime_config import (
    DATA_ROOT,
    SEED,
    project_checkpoint_dir,
    project_log_dir,
    set_global_seed,
)

import argparse
import hashlib
import csv
import json
import os
from typing import Dict, List, Sequence, Tuple

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

import numpy as np
from scipy.stats import t
import torch
from torch.utils.data import DataLoader

from av_dataloader import AVDataloader
from multimodal_decoder import AV_PLC
from shared.audio_processing import librosa_mel2audio, load_audio_ffmpeg
from shared.metrics import calculate_pesq, calculate_stoi


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

# -----------------------------------------------------------------------------
# Experiment settings
# -----------------------------------------------------------------------------

GAP_LENGTHS_MS = [160, 500, 1000]
RETENTION_FRACTIONS = [0.25, 0.50, 0.75, 1.00]

VIDEO_FPS = 25
CLIP_SEC = 3.0
LOCAL_CONTEXT_SEC = 0.320

# Fixed metric ranges make all figures directly comparable.
METRIC_LIMITS = {
    "pesq": (1.0, 4.5),
    "stoi": (0.0, 1.0),
}

METHOD_LABELS = {
    "uniform": "Uniform reduction",
    "local": "Local reduction",
}

# Color identifies the AV method; audio-only is gray (reference baseline).
METHOD_COLORS = {
    "uniform": "#0072B2",   # blue
    "local":   "#D55E00",   # vermilion (color-blind safe)
    "audio":   "#808080",   # gray
}

# Dash pattern identifies the gap duration (same for all methods).
GAP_LINESTYLES = {
    160:  "-",    # solid
    500:  "--",   # dashed
    1000: ":",    # dotted
}

# Shared style properties for the two AV methods.
METHOD_STYLES = {
    "uniform": {
        "marker": "s",             # square
        "linewidth": AV_LINE_WIDTH,
        "markersize": MARKER_SIZE,
        "markeredgecolor": "white",
        "markeredgewidth": MARKER_EDGE_WIDTH,
        "band_alpha": 0.12,
        "zorder": 4,
    },
    "local": {
        "marker": "D",             # diamond
        "linewidth": AV_LINE_WIDTH,
        "markersize": MARKER_SIZE,
        "markeredgecolor": "white",
        "markeredgewidth": MARKER_EDGE_WIDTH,
        "band_alpha": 0.08,
        "zorder": 5,
    },
}

AUDIO_ONLY_STYLE = {
    "marker": "o",                 # circle
    "linewidth": AUDIO_ONLY_LINE_WIDTH,
    "markersize": MARKER_SIZE,
    "markeredgecolor": "white",
    "markeredgewidth": MARKER_EDGE_WIDTH,
    "band_alpha": 0.10,
    "zorder": 3,
}

# -----------------------------------------------------------------------------
# Dataset single-gap mask helpers
# -----------------------------------------------------------------------------

def gap_bounds_from_mask(mask: np.ndarray) -> Tuple[int, int, float, float]:
    """Read one contiguous gap from the dataset-provided mel mask."""
    time_mask = np.asarray(mask[0], dtype=np.float32)
    missing = np.flatnonzero(time_mask < 0.5)
    if missing.size == 0:
        raise ValueError("Single-gap mask contains no missing frames.")
    if not np.all(np.diff(missing) == 1):
        raise ValueError("Expected one contiguous single gap, but mask has multiple gaps.")

    start = int(missing[0])
    end = int(missing[-1] + 1)
    mel_fps = time_mask.shape[0] / CLIP_SEC
    return start, end, start / mel_fps, end / mel_fps


# -----------------------------------------------------------------------------
# Video reduction helpers
# -----------------------------------------------------------------------------

def count_from_fraction(total_frames: int, fraction: float) -> int:
    """Convert a retention fraction to a safe integer frame count."""
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, received {total_frames}.")
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], received {fraction}.")
    if fraction >= 1.0:
        return total_frames
    return max(1, min(total_frames, int(round(total_frames * fraction))))


def uniformly_spaced_indices(start: int, end: int, keep_count: int) -> np.ndarray:
    """Select unique, approximately uniform indices in inclusive [start, end]."""
    if end < start:
        raise ValueError(f"Invalid index interval [{start}, {end}].")

    available = end - start + 1
    if keep_count >= available:
        return np.arange(start, end + 1, dtype=np.int64)
    if keep_count <= 0:
        raise ValueError("keep_count must be positive.")

    indices = np.linspace(start, end, keep_count).round().astype(np.int64)
    indices = np.unique(indices)

    # np.linspace over an integer interval should already produce the requested
    # number for keep_count <= available. This guard catches unexpected cases.
    if len(indices) != keep_count:
        candidates = np.arange(start, end + 1, dtype=np.int64)
        target = np.linspace(0, available - 1, keep_count)
        indices = candidates[np.round(target).astype(np.int64)]
        indices = np.unique(indices)

    if len(indices) != keep_count:
        raise RuntimeError(
            f"Could not select {keep_count} unique indices from {available} frames."
        )

    return indices


def replace_with_nearest_kept(
    frames: np.ndarray,
    region_start: int,
    region_end: int,
    keep_indices: np.ndarray,
) -> np.ndarray:
    """Replace dropped frames in an inclusive region by the nearest kept frame."""
    if len(keep_indices) == 0:
        raise ValueError("keep_indices cannot be empty.")

    output = frames.astype(np.float32, copy=True)
    for frame_idx in range(region_start, region_end + 1):
        nearest = keep_indices[np.argmin(np.abs(keep_indices - frame_idx))]
        output[frame_idx] = frames[nearest]
    return output


def reduce_uniform_video(
    frames: np.ndarray,
    retention_fraction: float,
) -> Tuple[np.ndarray, int, int]:
    """Reduce frames uniformly over the complete video."""
    total_frames = int(frames.shape[0])
    keep_count = count_from_fraction(total_frames, retention_fraction)

    if keep_count >= total_frames:
        return frames.astype(np.float32, copy=True), keep_count, total_frames

    keep_indices = uniformly_spaced_indices(0, total_frames - 1, keep_count)
    reduced = replace_with_nearest_kept(
        frames=frames,
        region_start=0,
        region_end=total_frames - 1,
        keep_indices=keep_indices,
    )
    return reduced, keep_count, total_frames


def get_local_video_window(
    gap_start_sec: float,
    gap_end_sec: float,
    total_video_frames: int,
    video_fps: int = VIDEO_FPS,
    context_sec: float = LOCAL_CONTEXT_SEC,
) -> Tuple[int, int]:
    """Return inclusive bounds of context + gap + context in video frames."""
    local_start = int(np.floor((gap_start_sec - context_sec) * video_fps))
    local_end = int(np.ceil((gap_end_sec + context_sec) * video_fps)) - 1

    local_start = max(0, local_start)
    local_end = min(total_video_frames - 1, local_end)
    if local_end < local_start:
        local_end = local_start

    return local_start, local_end


def reduce_local_video(
    frames: np.ndarray,
    gap_start_sec: float,
    gap_end_sec: float,
    retention_fraction: float,
) -> Tuple[np.ndarray, int, int, int, int]:
    """Reduce frames only inside the gap-centred local window."""
    total_frames = int(frames.shape[0])
    local_start, local_end = get_local_video_window(
        gap_start_sec=gap_start_sec,
        gap_end_sec=gap_end_sec,
        total_video_frames=total_frames,
    )
    local_window_frames = local_end - local_start + 1
    keep_count = count_from_fraction(local_window_frames, retention_fraction)

    if keep_count >= local_window_frames:
        return (
            frames.astype(np.float32, copy=True),
            keep_count,
            local_window_frames,
            local_start,
            local_end,
        )

    keep_indices = uniformly_spaced_indices(local_start, local_end, keep_count)
    reduced = replace_with_nearest_kept(
        frames=frames,
        region_start=local_start,
        region_end=local_end,
        keep_indices=keep_indices,
    )
    return reduced, keep_count, local_window_frames, local_start, local_end


# -----------------------------------------------------------------------------
# Dataset-native loader
# -----------------------------------------------------------------------------

def build_loader(
    dataset_name: str,
    gap_ms: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    test_subset=None,
) -> DataLoader:
    """Build one dataset-native single-gap loader per gap duration."""
    builder = AVDataloader(
        dataset_name=dataset_name,
        mode="av",
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


# -----------------------------------------------------------------------------
# Model and audio helpers
# -----------------------------------------------------------------------------

def load_checkpoint_flexible(model, checkpoint_file: str, device):
    checkpoint = torch.load(
        checkpoint_file,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        state = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("model_state")
            or checkpoint.get("state_dict")
            or checkpoint
        )
    else:
        state = checkpoint

    clean_state = {key.replace("module.", ""): value for key, value in state.items()}
    model.load_state_dict(clean_state, strict=True)
    print(f"Loaded checkpoint strictly: {checkpoint_file}")
    return model


def build_av_model(device):
    return AV_PLC(
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


def default_av_model_name(dataset_name: str) -> str:
    return (
        "av_wide_masking_mlp_av_only_fusion_5loss_bursty2"
        "_plc_a0.05_v0.1"
        "_pesq_0.01"
        "_asr_0.1"
        f"({dataset_name})"
    )


def checkpoint_path(checkpoint_dir: str, model_name: str) -> str:
    return os.path.join(checkpoint_dir, model_name, "best_model.pt")


def resolve_audio_path(video_path: str, dataset_root: str) -> str:
    if isinstance(video_path, bytes):
        video_path = video_path.decode("utf-8")
    video_path = str(video_path)

    if os.path.exists(video_path):
        return video_path

    if "datasets" in video_path:
        relative_path = video_path.split("datasets", 1)[-1].lstrip(os.sep)
        candidate = os.path.join(dataset_root, relative_path)
        if os.path.exists(candidate):
            return candidate

    return video_path


def get_dataset_stats(loader: DataLoader) -> Tuple[float, float]:
    dataset = loader.dataset
    while hasattr(dataset, "base_dataset"):
        dataset = dataset.base_dataset
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset

    return (
        float(getattr(dataset, "mel_mean", -56.775)),
        float(getattr(dataset, "mel_std", 19.707)),
    )


def insert_reconstructed_gap(
    original_audio,
    reconstructed_audio,
    mask,
    hop_length: int = 160,
) -> np.ndarray:
    """Keep original audio outside the missing region."""
    original = np.asarray(original_audio, dtype=np.float32).squeeze()
    reconstructed = np.asarray(reconstructed_audio, dtype=np.float32).squeeze()

    if torch.is_tensor(mask):
        time_keep = mask[0].detach().cpu().numpy().astype(np.float32)
    else:
        time_keep = np.asarray(mask[0], dtype=np.float32)

    waveform_mask = np.repeat(time_keep, hop_length)
    length = min(len(original), len(reconstructed), len(waveform_mask))

    return (
        original[:length] * waveform_mask[:length]
        + reconstructed[:length] * (1.0 - waveform_mask[:length])
    )


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

CONDITIONS = [
    ("uniform", 0.25),
    ("uniform", 0.50),
    ("uniform", 0.75),
    ("local", 0.25),
    ("local", 0.50),
    ("local", 0.75),
    ("full", 1.00),
]


def _repeat_batch_value(value, repeats: int):
    """Repeat a batch-aligned tensor/list for sample-major expanded conditions."""
    if torch.is_tensor(value):
        return value.repeat_interleave(repeats, dim=0)
    if isinstance(value, np.ndarray):
        return np.repeat(value, repeats, axis=0)
    if isinstance(value, (list, tuple)):
        return [item for item in value for _ in range(repeats)]
    raise TypeError(f"Unsupported batch value type: {type(value)!r}")


def _forward_in_chunks(
    model,
    masked_spec,
    frames,
    spk_emb,
    audio_length,
    avail,
    chunk_size: int,
):
    """Forward expanded conditions in bounded chunks to avoid GPU OOM."""
    outputs = []
    total = int(frames.size(0))
    chunk_size = max(1, int(chunk_size))
    for start in range(0, total, chunk_size):
        end = min(total, start + chunk_size)
        length_chunk = (
            audio_length[start:end]
            if torch.is_tensor(audio_length)
            else audio_length[start:end]
        )
        fused_mel, _, _ = model(
            masked_spec[start:end],
            frames[start:end],
            spk_emb[start:end],
            length_chunk,
            avail=avail[start:end],
        )
        outputs.append(fused_mel.detach().cpu())
    return torch.cat(outputs, dim=0)


def _identity_hash(identities: List[Dict]) -> str:
    payload = json.dumps(identities, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@torch.no_grad()
def evaluate_all_conditions(
    model,
    loader: DataLoader,
    device,
    dataset_root: str,
    dataset_name: str,
    gap_ms: int,
    forward_chunk_size: int,
    sample_rate: int = 16000,
) -> Tuple[List[Dict], List[Dict], str, List[Dict]]:
    """Evaluate all reduction conditions in one dataset pass for one gap."""
    model.eval()
    mel_mean, mel_std = get_dataset_stats(loader)

    rows: List[Dict] = []
    identities: List[Dict] = []
    metric_values = {
        (method, fraction): {"pesq": [], "stoi": []}
        for method, fraction in CONDITIONS
    }
    frame_metadata = {
        (method, fraction): {"kept": [], "region": [], "effective": []}
        for method, fraction in CONDITIONS
    }

    condition_count = len(CONDITIONS)

    for batch_idx, batch in enumerate(loader):
        (
            frames,
            spk_emb,
            masked_spec,
            _mel_spec,
            audio_length,
            _text,
            mask,
            video_path,
            avail,
        ) = batch

        batch_size = int(frames.size(0))
        expanded_frames = []
        expanded_meta = []

        for sample_idx in range(batch_size):
            original_frames = frames[sample_idx].detach().cpu().numpy()
            gap_start_mel, gap_end_mel, gap_start_sec, gap_end_sec = (
                gap_bounds_from_mask(mask[sample_idx].detach().cpu().numpy())
            )
            identity = {
                "video_path": str(video_path[sample_idx]),
                "gap_start_frame": int(gap_start_mel),
                "gap_end_frame": int(gap_end_mel),
            }
            identities.append(identity)

            total_frames = int(original_frames.shape[0])
            for method, fraction in CONDITIONS:
                if method == "uniform":
                    reduced, kept, region = reduce_uniform_video(
                        original_frames, fraction
                    )
                    local_start = -1
                    local_end = -1
                    effective = kept
                elif method == "local":
                    reduced, kept, region, local_start, local_end = (
                        reduce_local_video(
                            original_frames,
                            gap_start_sec,
                            gap_end_sec,
                            fraction,
                        )
                    )
                    effective = total_frames - region + kept
                else:
                    reduced = original_frames.astype(np.float32, copy=True)
                    kept = total_frames
                    region = total_frames
                    effective = total_frames
                    local_start = -1
                    local_end = -1

                expanded_frames.append(torch.from_numpy(reduced).float())
                expanded_meta.append(
                    {
                        "sample_idx": sample_idx,
                        "method": method,
                        "fraction": float(fraction),
                        "gap_start_mel": gap_start_mel,
                        "gap_end_mel": gap_end_mel,
                        "gap_start_sec": gap_start_sec,
                        "gap_end_sec": gap_end_sec,
                        "kept": int(kept),
                        "region": int(region),
                        "effective": int(effective),
                        "local_start": int(local_start),
                        "local_end": int(local_end),
                    }
                )

        expanded_frames = torch.stack(expanded_frames, dim=0).to(
            device, non_blocking=True
        )
        expanded_masked = _repeat_batch_value(masked_spec, condition_count).to(
            device, non_blocking=True
        ).float()
        expanded_spk = _repeat_batch_value(spk_emb, condition_count).to(
            device, non_blocking=True
        ).float()
        expanded_avail = _repeat_batch_value(avail, condition_count).to(
            device, non_blocking=True
        ).bool()
        expanded_length = _repeat_batch_value(audio_length, condition_count)
        if torch.is_tensor(expanded_length):
            expanded_length = expanded_length.to(device, non_blocking=True)

        fused_mels = _forward_in_chunks(
            model=model,
            masked_spec=expanded_masked,
            frames=expanded_frames,
            spk_emb=expanded_spk,
            audio_length=expanded_length,
            avail=expanded_avail,
            chunk_size=forward_chunk_size,
        )

        # Decode the reference waveform once per original sample, then evaluate
        # every condition for that sample against the same reference.
        reference_cache = {}
        for expanded_idx, meta in enumerate(expanded_meta):
            sample_idx = meta["sample_idx"]
            if sample_idx not in reference_cache:
                path = resolve_audio_path(video_path[sample_idx], dataset_root)
                reference_cache[sample_idx] = load_audio_ffmpeg(
                    path, sr=sample_rate, fixlen_sec=CLIP_SEC
                )
            reference_audio = reference_cache[sample_idx]

            reconstructed_audio = librosa_mel2audio(
                fused_mels[expanded_idx],
                sr=sample_rate,
                mel_mean=mel_mean,
                mel_std=mel_std,
            )
            if torch.is_tensor(reconstructed_audio):
                reconstructed_audio = reconstructed_audio.detach().cpu().numpy()

            predicted_audio = insert_reconstructed_gap(
                original_audio=reference_audio,
                reconstructed_audio=reconstructed_audio,
                mask=mask[sample_idx],
                hop_length=160,
            )
            valid_length = min(len(reference_audio), len(predicted_audio))
            pesq_score = calculate_pesq(
                reference_audio[:valid_length],
                predicted_audio[:valid_length],
                sr=sample_rate,
            )
            stoi_score = calculate_stoi(
                reference_audio[:valid_length],
                predicted_audio[:valid_length],
                sr=sample_rate,
            )

            key = (meta["method"], meta["fraction"])
            frame_metadata[key]["kept"].append(meta["kept"])
            frame_metadata[key]["region"].append(meta["region"])
            frame_metadata[key]["effective"].append(meta["effective"])
            if pesq_score is not None and np.isfinite(pesq_score):
                metric_values[key]["pesq"].append(float(pesq_score))
            if stoi_score is not None and np.isfinite(stoi_score):
                metric_values[key]["stoi"].append(float(stoi_score))

            rows.append(
                {
                    "dataset": dataset_name,
                    "batch_idx": int(batch_idx),
                    "sample_in_batch": int(sample_idx),
                    "video_path": str(video_path[sample_idx]),
                    "gap_ms": int(gap_ms),
                    "method": meta["method"],
                    "retention_fraction": meta["fraction"],
                    "retention_percent": meta["fraction"] * 100.0,
                    "gap_start_mel": int(meta["gap_start_mel"]),
                    "gap_end_mel": int(meta["gap_end_mel"]),
                    "gap_start_sec": float(meta["gap_start_sec"]),
                    "gap_end_sec": float(meta["gap_end_sec"]),
                    "kept_frames_in_reduced_region": meta["kept"],
                    "reduced_region_frames": meta["region"],
                    "effective_total_unique_frames": meta["effective"],
                    "local_start_video_frame": meta["local_start"],
                    "local_end_video_frame": meta["local_end"],
                    "pesq": None if pesq_score is None else float(pesq_score),
                    "stoi": None if stoi_score is None else float(stoi_score),
                }
            )

    summaries: List[Dict] = []
    for method, fraction in CONDITIONS:
        key = (method, fraction)
        pesq_values = metric_values[key]["pesq"]
        stoi_values = metric_values[key]["stoi"]
        metadata = frame_metadata[key]
        summaries.append(
            {
                "dataset": dataset_name,
                "gap_ms": int(gap_ms),
                "method": method,
                "retention_fraction": float(fraction),
                "retention_percent": float(fraction * 100.0),
                "pesq_mean": float(np.mean(pesq_values)) if pesq_values else None,
                "pesq_std": float(np.std(pesq_values)) if pesq_values else None,
                "stoi_mean": float(np.mean(stoi_values)) if stoi_values else None,
                "stoi_std": float(np.std(stoi_values)) if stoi_values else None,
                "num_pesq": len(pesq_values),
                "num_stoi": len(stoi_values),
                "kept_frames_mean": float(np.mean(metadata["kept"])),
                "reduced_region_frames_mean": float(np.mean(metadata["region"])),
                "effective_total_unique_frames_mean": float(
                    np.mean(metadata["effective"])
                ),
            }
        )

    # Full video is the common 100% endpoint for both plotted methods.
    full_summary = next(item for item in summaries if item["method"] == "full")
    for method in ("uniform", "local"):
        endpoint = dict(full_summary)
        endpoint["method"] = method
        if method == "local":
            # The local window is independent of retention fraction, so reuse
            # its measured size from an evaluated local condition.
            local_regions = frame_metadata[("local", 0.25)]["region"]
            endpoint["kept_frames_mean"] = float(np.mean(local_regions))
            endpoint["reduced_region_frames_mean"] = float(np.mean(local_regions))
            endpoint["effective_total_unique_frames_mean"] = float(
                np.mean(frame_metadata[("full", 1.0)]["effective"])
            )
        summaries.append(endpoint)

    summaries = [item for item in summaries if item["method"] != "full"]
    return summaries, rows, _identity_hash(identities), identities



# -----------------------------------------------------------------------------
# Saved-result loading and statistical summaries
# -----------------------------------------------------------------------------

def mean_std_ci95(values) -> Tuple[float, float, float, int]:
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


def _as_float(value):
    if value in (None, "", "None", "nan", "NaN"):
        return None
    return float(value)


def _as_int(value):
    return int(round(float(value)))


def load_per_sample_rows(path: str) -> List[Dict]:
    with open(path, "r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    normalized = []
    for row in rows:
        item = dict(row)
        for key in (
            "batch_idx", "sample_in_batch", "gap_ms", "gap_start_mel",
            "gap_end_mel", "kept_frames_in_reduced_region",
            "reduced_region_frames", "effective_total_unique_frames",
            "local_start_video_frame", "local_end_video_frame",
        ):
            item[key] = _as_int(item[key])
        for key in (
            "retention_fraction", "retention_percent", "gap_start_sec",
            "gap_end_sec", "pesq", "stoi",
        ):
            item[key] = _as_float(item.get(key))
        normalized.append(item)
    return normalized


def _condition_rows(rows: List[Dict], dataset_name: str, gap_ms: int,
                    method: str, fraction: float) -> List[Dict]:
    selected = [
        row for row in rows
        if str(row.get("dataset")) == str(dataset_name)
        and int(row["gap_ms"]) == int(gap_ms)
        and str(row["method"]) == method
        and np.isclose(float(row["retention_fraction"]), float(fraction))
    ]
    return sorted(selected, key=lambda row: (row["batch_idx"], row["sample_in_batch"]))


def validate_complete_stored_rows(
    rows: List[Dict],
    dataset_name: str,
    audio_only_baselines: Dict[int, Dict[str, float]],
) -> None:
    """Validate every raw condition and its ordered sample/mask identity."""
    for gap_ms in GAP_LENGTHS_MS:
        baseline = audio_only_baselines[gap_ms]
        expected_n = int(baseline["num_samples"])
        expected_hash = baseline["sample_identity_sha256"]

        for method, fraction in CONDITIONS:
            selected = _condition_rows(
                rows, dataset_name, gap_ms, method, fraction
            )
            if len(selected) != expected_n:
                raise ValueError(
                    f"Incomplete saved condition for {dataset_name}, {gap_ms} ms, "
                    f"{method}, {fraction}: expected {expected_n} rows, "
                    f"found {len(selected)}."
                )

            identities = [
                {
                    "video_path": str(row["video_path"]),
                    "gap_start_frame": int(row["gap_start_mel"]),
                    "gap_end_frame": int(row["gap_end_mel"]),
                }
                for row in selected
            ]
            if _identity_hash(identities) != expected_hash:
                raise ValueError(
                    f"Saved sample/mask identities do not match the gap sweep for "
                    f"{dataset_name}, {gap_ms} ms, {method}, {fraction}."
                )


def summarize_per_sample_rows(dataset_name: str, rows: List[Dict]) -> List[Dict]:
    """Rebuild all summaries, including CI95, from per-sample results."""
    raw_summaries: List[Dict] = []

    for gap_ms in GAP_LENGTHS_MS:
        for method, fraction in CONDITIONS:
            selected = _condition_rows(
                rows, dataset_name, gap_ms, method, fraction
            )
            if not selected:
                raise ValueError(
                    f"Missing condition: {dataset_name}, {gap_ms} ms, "
                    f"{method}, {fraction}."
                )

            pesq_mean, pesq_std, pesq_ci95, num_pesq = mean_std_ci95(
                [row["pesq"] for row in selected if row["pesq"] is not None]
            )
            stoi_mean, stoi_std, stoi_ci95, num_stoi = mean_std_ci95(
                [row["stoi"] for row in selected if row["stoi"] is not None]
            )

            raw_summaries.append(
                {
                    "dataset": dataset_name,
                    "gap_ms": int(gap_ms),
                    "method": method,
                    "retention_fraction": float(fraction),
                    "retention_percent": float(fraction * 100.0),
                    "pesq_mean": pesq_mean,
                    "pesq_std": pesq_std,
                    "pesq_ci95": pesq_ci95,
                    "stoi_mean": stoi_mean,
                    "stoi_std": stoi_std,
                    "stoi_ci95": stoi_ci95,
                    "num_pesq": num_pesq,
                    "num_stoi": num_stoi,
                    "kept_frames_mean": float(np.mean([
                        row["kept_frames_in_reduced_region"] for row in selected
                    ])),
                    "reduced_region_frames_mean": float(np.mean([
                        row["reduced_region_frames"] for row in selected
                    ])),
                    "effective_total_unique_frames_mean": float(np.mean([
                        row["effective_total_unique_frames"] for row in selected
                    ])),
                }
            )

    # The full-video condition is evaluated once. Reuse it as the 100% endpoint
    # of both uniform and local curves, matching the original experiment logic.
    summaries: List[Dict] = []
    for gap_ms in GAP_LENGTHS_MS:
        gap_rows = [row for row in raw_summaries if row["gap_ms"] == gap_ms]
        full_row = next(row for row in gap_rows if row["method"] == "full")
        summaries.extend(row for row in gap_rows if row["method"] != "full")

        for method in ("uniform", "local"):
            endpoint = dict(full_row)
            endpoint["method"] = method
            if method == "local":
                local_reference = next(
                    row for row in gap_rows
                    if row["method"] == "local"
                    and np.isclose(row["retention_fraction"], 0.25)
                )
                endpoint["kept_frames_mean"] = local_reference[
                    "reduced_region_frames_mean"
                ]
                endpoint["reduced_region_frames_mean"] = local_reference[
                    "reduced_region_frames_mean"
                ]
            summaries.append(endpoint)

    return summaries

# -----------------------------------------------------------------------------
# Reusable audio-only baseline from the gap sweep
# -----------------------------------------------------------------------------

def load_audio_only_baselines(
    baseline_dir: str,
    dataset_name: str,
    expected_seed: int,
    expected_test_subset,
) -> Dict[int, Dict[str, float]]:
    """Load audio-only baselines and ensure mean/std/CI95 are available."""
    baselines: Dict[int, Dict[str, float]] = {}
    for gap_ms in GAP_LENGTHS_MS:
        gap_dir = os.path.join(baseline_dir, dataset_name, f"gap_{gap_ms}ms")
        summary_path = os.path.join(gap_dir, "summary.json")
        rows_path = os.path.join(gap_dir, "audio_only_per_sample.csv")

        if not os.path.isfile(summary_path):
            raise FileNotFoundError(
                f"Missing gap-sweep baseline for {dataset_name}, {gap_ms} ms: "
                f"{summary_path}"
            )
        with open(summary_path, "r", encoding="utf-8") as file:
            payload = json.load(file)

        if payload.get("mask_type") != "single_gap":
            raise ValueError(
                f"Baseline was not produced with single-gap masking: {summary_path}"
            )
        if int(payload.get("mask_seed", -1)) != int(expected_seed):
            raise ValueError(
                f"Mask seed mismatch in {summary_path}: expected {expected_seed}, "
                f"found {payload.get('mask_seed')}"
            )
        if payload.get("test_subset") != expected_test_subset:
            raise ValueError(
                f"test_subset mismatch in {summary_path}: expected "
                f"{expected_test_subset}, found {payload.get('test_subset')}"
            )

        audio = payload.get("audio_only")
        if not isinstance(audio, dict):
            raise KeyError(f"Missing 'audio_only' summary in {summary_path}")

        # Older gap-sweep summaries may lack CI95. Recompute all statistics from
        # the saved per-sample CSV so the baseline uses the same convention.
        needs_recompute = any(
            audio.get(f"{metric}_{stat}") is None
            for metric in ("pesq", "stoi")
            for stat in ("mean", "std", "ci95")
        )
        if needs_recompute:
            if not os.path.isfile(rows_path):
                raise FileNotFoundError(
                    f"Baseline summary lacks CI95 and per-sample rows are missing: "
                    f"{rows_path}"
                )
            with open(rows_path, "r", newline="", encoding="utf-8") as file:
                baseline_rows = list(csv.DictReader(file))

            pesq_mean, pesq_std, pesq_ci95, num_pesq = mean_std_ci95(
                [_as_float(row.get("pesq")) for row in baseline_rows
                 if _as_float(row.get("pesq")) is not None]
            )
            stoi_mean, stoi_std, stoi_ci95, num_stoi = mean_std_ci95(
                [_as_float(row.get("stoi")) for row in baseline_rows
                 if _as_float(row.get("stoi")) is not None]
            )
            audio = {
                **audio,
                "pesq_mean": pesq_mean,
                "pesq_std": pesq_std,
                "pesq_ci95": pesq_ci95,
                "stoi_mean": stoi_mean,
                "stoi_std": stoi_std,
                "stoi_ci95": stoi_ci95,
                "num_pesq": num_pesq,
                "num_stoi": num_stoi,
            }

        for metric in ("pesq", "stoi"):
            for stat in ("mean", "std", "ci95"):
                value = audio.get(f"{metric}_{stat}")
                if value is None or not np.isfinite(value):
                    raise ValueError(
                        f"Missing/non-finite {metric}_{stat} in {summary_path}"
                    )

        identity_hash = payload.get("sample_identity_sha256")
        identities = payload.get("ordered_sample_identities")
        if not identity_hash or not isinstance(identities, list):
            raise ValueError(
                f"Baseline lacks sample-identity metadata; rerun gap sweep: "
                f"{summary_path}"
            )

        baselines[int(gap_ms)] = {
            **audio,
            "sample_identity_sha256": identity_hash,
            "ordered_sample_identities": identities,
            "num_samples": int(payload.get("num_samples", len(identities))),
            "audio_model": payload.get("audio_model"),
            "sample_rate": payload.get("sample_rate"),
            "hop_length": payload.get("hop_length"),
        }
    return baselines


# -----------------------------------------------------------------------------
# Saving and plotting
# -----------------------------------------------------------------------------

def save_json(data, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def save_rows_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_csv(results: List[Dict], path: str) -> None:
    if not results:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = list(results[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def configure_axis(ax, metric: str) -> None:
    ax.set_xlim(20.0, 105.0)
    ax.set_ylim(*METRIC_LIMITS[metric])
    ax.set_xticks([25, 50, 75, 100])
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(which="major", alpha=0.45, linewidth=GRID_MAJOR_LINEWIDTH)
    ax.grid(which="minor", alpha=0.20, linewidth=GRID_MINOR_LINEWIDTH, linestyle=":")
    ax.tick_params(which="both", direction="in", length=4)
    ax.tick_params(which="minor", length=2)


# Frame counts per retention level derived from experiment constants.
# Uniform: fraction * total_clip_frames (3 s × 25 fps = 75 frames).
# Local:   fraction * local_window_frames (0.320 s context each side).
# These are the mean expected values; exact per-sample means are dataset-
# dependent but these round numbers match what is shown in the uploaded image.
_TOTAL_CLIP_FRAMES = int(round(CLIP_SEC * VIDEO_FPS))          # 75
_LOCAL_WINDOW_FRAMES = int(round(
    (2 * LOCAL_CONTEXT_SEC + 1.0 / VIDEO_FPS) * VIDEO_FPS      # ≈ 17 (320 ms each side + 1 gap frame)
))

def _xtick_labels(dataset_results: List[Dict]) -> List[str]:
    """Build tick labels with mean U/L frame counts from summary data."""
    labels = []
    for pct in [25, 50, 75, 100]:
        fraction = pct / 100.0

        # Pull mean kept-frames from the summaries when available.
        u_rows = [
            r for r in dataset_results
            if r.get("method") == "uniform"
            and np.isclose(float(r.get("retention_fraction", -1)), fraction)
        ]
        l_rows = [
            r for r in dataset_results
            if r.get("method") == "local"
            and np.isclose(float(r.get("retention_fraction", -1)), fraction)
        ]

        if u_rows:
            u_val = int(round(np.mean([r["kept_frames_mean"] for r in u_rows])))
        else:
            u_val = int(round(fraction * _TOTAL_CLIP_FRAMES))

        if l_rows:
            l_val = int(round(np.mean([r["kept_frames_mean"] for r in l_rows])))
        else:
            l_val = int(round(fraction * _LOCAL_WINDOW_FRAMES))

        labels.append(f"{pct}\nU:{u_val} / L:{l_val}")
    return labels


def plot_reduction_panel(
    ax,
    dataset_results: List[Dict],
    audio_only_baselines: Dict[int, Dict[str, float]],
    metric: str,
) -> None:
    """Plot all gaps: audio-only (gray), uniform AV (blue), local AV (vermilion).
    Color encodes AV method; dash pattern encodes gap duration.
    """
    x_baseline = np.asarray([25.0, 50.0, 75.0, 100.0])

    # Draw uniform first so local (higher zorder) renders on top.
    for method in ("uniform", "local"):
        style = METHOD_STYLES[method]
        color = METHOD_COLORS[method]

        for gap_ms in GAP_LENGTHS_MS:
            linestyle = GAP_LINESTYLES[gap_ms]
            selected = sorted(
                [
                    row for row in dataset_results
                    if int(row["gap_ms"]) == int(gap_ms)
                    and row["method"] == method
                ],
                key=lambda row: row["retention_fraction"],
            )
            x_values = np.asarray(
                [row["retention_percent"] for row in selected], dtype=np.float64
            )
            means = np.asarray(
                [row[f"{metric}_mean"] for row in selected], dtype=np.float64
            )
            ci95 = np.asarray(
                [row[f"{metric}_ci95"] for row in selected], dtype=np.float64
            )
            valid = (
                np.isfinite(x_values)
                & np.isfinite(means)
                & np.isfinite(ci95)
            )
            if not np.any(valid):
                continue

            ax.plot(
                x_values[valid],
                means[valid],
                color=color,
                linestyle=linestyle,
                linewidth=style["linewidth"],
                marker=style["marker"],
                markersize=style["markersize"],
                markerfacecolor=color,
                markeredgecolor=style["markeredgecolor"],
                markeredgewidth=style["markeredgewidth"],
                zorder=style["zorder"],
            )
            ax.fill_between(
                x_values[valid],
                means[valid] - ci95[valid],
                means[valid] + ci95[valid],
                color=color,
                alpha=style["band_alpha"],
                linewidth=0,
                zorder=1,
            )

    # Audio-only baselines: gray, same dash-per-gap convention.
    audio_color = METHOD_COLORS["audio"]
    for gap_ms in GAP_LENGTHS_MS:
        linestyle = GAP_LINESTYLES[gap_ms]
        baseline = audio_only_baselines[gap_ms]
        baseline_mean = float(baseline[f"{metric}_mean"])
        baseline_ci95 = float(baseline[f"{metric}_ci95"])
        ax.plot(
            x_baseline,
            np.full_like(x_baseline, baseline_mean),
            color=audio_color,
            linestyle=linestyle,
            linewidth=AUDIO_ONLY_STYLE["linewidth"],
            marker=AUDIO_ONLY_STYLE["marker"],
            markersize=AUDIO_ONLY_STYLE["markersize"],
            markerfacecolor=audio_color,
            markeredgecolor=AUDIO_ONLY_STYLE["markeredgecolor"],
            markeredgewidth=AUDIO_ONLY_STYLE["markeredgewidth"],
            zorder=AUDIO_ONLY_STYLE["zorder"],
        )
        ax.fill_between(
            x_baseline,
            baseline_mean - baseline_ci95,
            baseline_mean + baseline_ci95,
            color=audio_color,
            alpha=AUDIO_ONLY_STYLE["band_alpha"],
            linewidth=0,
            zorder=1,
        )

    configure_axis(ax, metric)
    ax.set_xlabel("Retained frames in reduced region (%)")
    ax.set_xticklabels(_xtick_labels(dataset_results), fontsize=8)


def _shared_legend_handles() -> List[Line2D]:
    # Method entries: color identifies the condition, solid line for clarity.
    handles = [
        Line2D(
            [0], [0],
            color=METHOD_COLORS["audio"],
            linewidth=LEGEND_LINE_WIDTH,
            linestyle="-",
            marker=AUDIO_ONLY_STYLE["marker"],
            markersize=AUDIO_ONLY_STYLE["markersize"],
            markerfacecolor=METHOD_COLORS["audio"],
            markeredgecolor="white",
            markeredgewidth=MARKER_EDGE_WIDTH,
            label="Audio-only",
        ),
        Line2D(
            [0], [0],
            color=METHOD_COLORS["uniform"],
            linewidth=LEGEND_LINE_WIDTH,
            linestyle="-",
            marker=METHOD_STYLES["uniform"]["marker"],
            markersize=METHOD_STYLES["uniform"]["markersize"],
            markerfacecolor=METHOD_COLORS["uniform"],
            markeredgecolor="white",
            markeredgewidth=MARKER_EDGE_WIDTH,
            label="AV uniform",
        ),
        Line2D(
            [0], [0],
            color=METHOD_COLORS["local"],
            linewidth=LEGEND_LINE_WIDTH,
            linestyle="-",
            marker=METHOD_STYLES["local"]["marker"],
            markersize=METHOD_STYLES["local"]["markersize"],
            markerfacecolor=METHOD_COLORS["local"],
            markeredgecolor="white",
            markeredgewidth=MARKER_EDGE_WIDTH,
            label="AV local",
        ),
    ]

    # Gap-duration entries: black lines with the matching dash pattern.
    for gap_ms in GAP_LENGTHS_MS:
        handles.append(
            Line2D(
                [0], [0],
                color="black",
                linestyle=GAP_LINESTYLES[gap_ms],
                linewidth=LEGEND_LINE_WIDTH,
                marker="none",
                label=f"{gap_ms} ms",
            )
        )
    return handles


def save_dataset_pair_plot(
    dataset_results: List[Dict],
    output_dir: str,
    dataset_name: str,
    audio_only_baselines: Dict[int, Dict[str, float]],
) -> None:
    """Save one dataset figure with PESQ and STOI columns."""
    if not dataset_results:
        return

    figure_dir = os.path.join(output_dir, dataset_name, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    fig, axes = plt.subplots(
        1, 2,
        figsize=(FIG_WIDTH_2COL, FIG_HEIGHT_1ROW),
        squeeze=False,
    )
    axes = axes[0]

    for ax, metric in zip(axes, ("pesq", "stoi")):
        plot_reduction_panel(
            ax, dataset_results, audio_only_baselines, metric
        )
        ax.set_title(metric.upper())
        ax.set_ylabel(rf"{metric.upper()} ($\uparrow$)")

    fig.legend(
        handles=_shared_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=6,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(
        os.path.join(figure_dir, "pesq_stoi_video_reduction_ci95.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_all_datasets_plot(
    all_summaries: List[Dict],
    all_audio_baselines: Dict[str, Dict[int, Dict[str, float]]],
    output_dir: str,
) -> None:
    """Save a rows=datasets, columns=PESQ/STOI combined figure."""
    preferred_order = ["grid", "lrs2", "voxceleb2"]
    datasets = [name for name in preferred_order if name in all_audio_baselines]
    if not datasets:
        return

    fig, axes = plt.subplots(
        len(datasets), 2,
        figsize=(FIG_WIDTH_2COL, FIG_HEIGHT_ROW * len(datasets)),
        squeeze=False,
        sharex="col",
        sharey="col",
    )
    display_names = {
        "grid": "GRID",
        "lrs2": "LRS2",
        "voxceleb2": "VoxCeleb2",
    }

    for row_index, dataset_name in enumerate(datasets):
        dataset_results = [
            row for row in all_summaries
            if str(row["dataset"]).lower() == dataset_name
        ]
        for column_index, metric in enumerate(("pesq", "stoi")):
            ax = axes[row_index, column_index]
            plot_reduction_panel(
                ax,
                dataset_results,
                all_audio_baselines[dataset_name],
                metric,
            )
            if row_index == 0:
                ax.set_title(metric.upper())
            if column_index == 0:
                ax.set_ylabel(display_names.get(dataset_name, dataset_name))
            if row_index != len(datasets) - 1:
                ax.set_xlabel("")

    fig.legend(
        handles=_shared_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=6,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(
        os.path.join(output_dir, "combined_datasets_video_reduction_ci95.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="uniform/local video-frame reduction ablation."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["grid", "lrs2", "voxceleb2"],
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=project_checkpoint_dir("AV_PLC"),
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(
            project_log_dir("AV_PLC"),
            "ablation_video_reduction",
        ),
    )
    parser.add_argument(
        "--gap_sweep_results_dir",
        type=str,
        default=os.path.join(
            project_log_dir("AV_PLC"),
            "ablation_gap_sweep",
        ),
        help="Folder containing reusable audio-only gap-sweep summaries.",
    )
    parser.add_argument("--dataset_root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--test_subset", type=int, default=None)
    parser.add_argument(
        "--forward_chunk_size",
        type=int,
        default=16,
        help="Maximum expanded AV samples per model forward.",
    )
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    set_global_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_summaries: List[Dict] = []
    all_rows: List[Dict] = []
    all_audio_baselines: Dict[str, Dict[int, Dict[str, float]]] = {}

    for dataset_name_raw in args.datasets:
        dataset_name = str(dataset_name_raw).lower()
        print("\n==============================")
        print(f"Dataset: {dataset_name}")
        print("==============================")

        audio_only_baselines = load_audio_only_baselines(
            args.gap_sweep_results_dir,
            dataset_name,
            expected_seed=SEED,
            expected_test_subset=args.test_subset,
        )
        all_audio_baselines[dataset_name] = audio_only_baselines

        for gap_ms, baseline in audio_only_baselines.items():
            if baseline.get("sample_rate") not in (None, args.sample_rate):
                raise ValueError(
                    f"Sample-rate mismatch for baseline {dataset_name}, {gap_ms} ms."
                )
            if baseline.get("hop_length") not in (None, 160):
                raise ValueError(
                    f"Hop-length mismatch for baseline {dataset_name}, {gap_ms} ms."
                )

        dataset_dir = os.path.join(args.results_dir, dataset_name)
        per_sample_path = os.path.join(dataset_dir, "per_sample_results.csv")
        dataset_rows = None

        if os.path.isfile(per_sample_path):
            try:
                candidate_rows = load_per_sample_rows(per_sample_path)
                validate_complete_stored_rows(
                    candidate_rows,
                    dataset_name,
                    audio_only_baselines,
                )
                dataset_rows = candidate_rows
                print("Complete per-sample results found; skipping inference.")
            except Exception as error:
                print(
                    "Saved per-sample results are incomplete or incompatible; "
                    f"rerunning this dataset. Reason: {error}"
                )

        model = None
        if dataset_rows is None:
            model_name = default_av_model_name(dataset_name)
            model_file = checkpoint_path(args.checkpoint_dir, model_name)
            model = load_checkpoint_flexible(
                build_av_model(device),
                model_file,
                device,
            )

            generated_rows: List[Dict] = []
            for gap_ms in GAP_LENGTHS_MS:
                print(f"\n--- Audio gap: {gap_ms} ms ---")
                loader = build_loader(
                    dataset_name=dataset_name,
                    gap_ms=gap_ms,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    seed=SEED,
                    test_subset=args.test_subset,
                )
                _summaries, rows, identity_hash, identities = evaluate_all_conditions(
                    model=model,
                    loader=loader,
                    device=device,
                    dataset_root=args.dataset_root,
                    dataset_name=dataset_name,
                    gap_ms=gap_ms,
                    forward_chunk_size=args.forward_chunk_size,
                    sample_rate=args.sample_rate,
                )

                baseline = audio_only_baselines[gap_ms]
                if identity_hash != baseline["sample_identity_sha256"]:
                    expected = baseline["ordered_sample_identities"]
                    mismatch_index = next(
                        (
                            index
                            for index, (current, saved) in enumerate(
                                zip(identities, expected)
                            )
                            if current != saved
                        ),
                        min(len(identities), len(expected)),
                    )
                    raise ValueError(
                        f"Sample/mask identity mismatch for {dataset_name}, "
                        f"{gap_ms} ms at ordered index {mismatch_index}."
                    )
                if len(identities) != baseline["num_samples"]:
                    raise ValueError(
                        f"Sample-count mismatch for {dataset_name}, {gap_ms} ms: "
                        f"{len(identities)} vs {baseline['num_samples']}"
                    )
                generated_rows.extend(rows)

            validate_complete_stored_rows(
                generated_rows,
                dataset_name,
                audio_only_baselines,
            )
            dataset_rows = generated_rows
            save_rows_csv(dataset_rows, per_sample_path)

        # Always rebuild summaries from per-sample rows, regardless of whether
        # inference was run now or results were already stored.
        dataset_summaries = summarize_per_sample_rows(
            dataset_name,
            dataset_rows,
        )
        save_json(
            dataset_summaries,
            os.path.join(dataset_dir, "summary.json"),
        )
        save_summary_csv(
            dataset_summaries,
            os.path.join(dataset_dir, "summary.csv"),
        )
        save_dataset_pair_plot(
            dataset_results=dataset_summaries,
            output_dir=args.results_dir,
            dataset_name=dataset_name,
            audio_only_baselines=audio_only_baselines,
        )

        all_summaries.extend(dataset_summaries)
        all_rows.extend(dataset_rows)

        if model is not None:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    save_json(all_summaries, os.path.join(args.results_dir, "all_summary.json"))
    save_summary_csv(
        all_summaries,
        os.path.join(args.results_dir, "all_summary.csv"),
    )
    save_rows_csv(
        all_rows,
        os.path.join(args.results_dir, "all_per_sample_results.csv"),
    )

    required_datasets = {"grid", "lrs2", "voxceleb2"}
    if required_datasets.issubset(all_audio_baselines.keys()):
        save_all_datasets_plot(
            all_summaries=all_summaries,
            all_audio_baselines=all_audio_baselines,
            output_dir=args.results_dir,
        )
    else:
        missing = sorted(required_datasets - set(all_audio_baselines.keys()))
        print(
            "Skipping combined figure because these datasets were not processed "
            f"in this run: {missing}"
        )

    print(f"\nFinished. Results saved to: {args.results_dir}")


if __name__ == "__main__":
    main()