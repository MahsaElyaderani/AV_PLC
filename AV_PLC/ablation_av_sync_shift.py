"""
Audio-video synchronization tolerance ablation for AV_PLC.

This script evaluates how much visual assistance remains useful when the video
stream is shifted relative to the masked audio. It keeps the same deterministic
single audio gap and shifts only the video frames.

Main question:
    How large can audio-video synchronization error become before AV
    reconstruction stops helping over the audio-only baseline?

It uses audio-only gap-sweep results from ablation_gap_sweep.py as the baseline.
If this script already has complete per-sample results for a dataset, it reuses
those rows and only regenerates summaries/plots.
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
import csv
import hashlib
import json
import os
from typing import Dict, List, Tuple

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
SYNC_OFFSETS_MS = [-320, -160, -80, -40, 0, 40, 80, 160, 320]

VIDEO_FPS = 25
FRAME_MS = 1000.0 / VIDEO_FPS
CLIP_SEC = 3.0
HOP_LENGTH = 160

METRIC_LIMITS = {
    "pesq": (1.0, 4.5),
    "stoi": (0.6, 1.0),
}

# Color identifies the model; line style and marker identify gap duration.
MODEL_COLORS = {
    "audio": "#7F7F7F",  # gray baseline
    "av": "#0072B2",     # Okabe-Ito blue
}


GAP_STYLES = {
    160: {"linestyle": "-"},
    500: {"linestyle": "--"},
    1000: {"linestyle": ":"},
}
# style properties for AV method.
AV_STYLE = {
    "marker": "s",             # square
    "linewidth": AV_LINE_WIDTH,
    "markersize": MARKER_SIZE,
    "markeredgecolor": "white",
    "markeredgewidth": MARKER_EDGE_WIDTH,
    "band_alpha": 0.12,
    "zorder": 4,
}

A_STYLE = {
    "marker": "o",                 # circle
    "linewidth": AUDIO_ONLY_LINE_WIDTH,
    "markersize": MARKER_SIZE,
    "markeredgecolor": "white",
    "markeredgewidth": MARKER_EDGE_WIDTH,
    "band_alpha": 0.10,
    "zorder": 3,
}

# -----------------------------------------------------------------------------
# Mask, model, and audio helpers
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


def build_loader(
    dataset_name: str,
    gap_ms: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    test_subset=None,
) -> DataLoader:
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


def load_checkpoint_flexible(model, checkpoint_file: str, device):
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
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


def insert_reconstructed_gap(original_audio, reconstructed_audio, mask, hop_length: int = HOP_LENGTH):
    """Keep original audio outside the missing mel region."""
    original = np.asarray(original_audio, dtype=np.float32).squeeze()
    reconstructed = np.asarray(reconstructed_audio, dtype=np.float32).squeeze()

    if torch.is_tensor(mask):
        time_keep = mask[0].detach().cpu().numpy().astype(np.float32)
    else:
        time_keep = np.asarray(mask[0], dtype=np.float32)

    waveform_mask = np.repeat(time_keep, hop_length)
    length = min(len(original), len(reconstructed), len(waveform_mask))
    return original[:length] * waveform_mask[:length] + reconstructed[:length] * (1.0 - waveform_mask[:length])


# -----------------------------------------------------------------------------
# Synchronization shift helper
# -----------------------------------------------------------------------------

def shift_video_frames(frames: np.ndarray, offset_ms: int) -> Tuple[np.ndarray, int]:
    """
    Shift video relative to audio, preserving sequence length.

    Sign convention:
        positive offset: video lags audio. Frames are delayed.
        negative offset: video leads audio. Frames are advanced.

    Boundary positions are filled by repeating the nearest valid edge frame.
    No circular wraparound is used.
    """
    shift = int(round(float(offset_ms) / FRAME_MS))
    if abs(float(offset_ms) - shift * FRAME_MS) > 1e-6:
        raise ValueError(f"Offset {offset_ms} ms is not an integer number of {FRAME_MS:g} ms frames.")

    total_frames = int(frames.shape[0])
    if shift == 0:
        return frames.astype(np.float32, copy=True), 0

    if abs(shift) >= total_frames:
        raise ValueError(f"Shift of {shift} frames is too large for {total_frames} video frames.")

    output = frames.astype(np.float32, copy=True)
    if shift > 0:
        # Video is delayed: audio time t receives an earlier video frame.
        output[:shift] = frames[0]
        output[shift:] = frames[:-shift]
    else:
        # Video is advanced: audio time t receives a later video frame.
        s = -shift
        output[:-s] = frames[s:]
        output[-s:] = frames[-1]

    return output, shift


# -----------------------------------------------------------------------------
# Statistical helpers
# -----------------------------------------------------------------------------

def mean_std_ci95(values) -> Tuple[float, float, float, int]:
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


def _identity_key_from_row(row: Dict) -> Tuple[str, int, int]:
    return (
        str(row["video_path"]),
        int(row.get("gap_start_mel", row.get("gap_start_frame"))),
        int(row.get("gap_end_mel", row.get("gap_end_frame"))),
    )


def _identity_hash(identities: List[Dict]) -> str:
    payload = json.dumps(identities, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def _repeat_batch_value(value, repeats: int):
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
    outputs = []
    total = int(frames.size(0))
    chunk_size = max(1, int(chunk_size))
    for start in range(0, total, chunk_size):
        end = min(total, start + chunk_size)
        length_chunk = audio_length[start:end] if torch.is_tensor(audio_length) else audio_length[start:end]
        fused_mel, _, _ = model(
            masked_spec[start:end],
            frames[start:end],
            spk_emb[start:end],
            length_chunk,
            avail=avail[start:end],
        )
        outputs.append(fused_mel.detach().cpu())
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def evaluate_sync_offsets(
    model,
    loader: DataLoader,
    device,
    dataset_root: str,
    dataset_name: str,
    gap_ms: int,
    offsets_ms: List[int],
    forward_chunk_size: int,
    sample_rate: int = 16000,
) -> Tuple[List[Dict], str, List[Dict]]:
    """Evaluate every synchronization offset in one dataset pass for one gap."""
    model.eval()
    mel_mean, mel_std = get_dataset_stats(loader)

    rows: List[Dict] = []
    identities: List[Dict] = []
    condition_count = len(offsets_ms)

    for batch_idx, batch in enumerate(loader):
        (
            frames,
            spk_emb,
            masked_spec,
            _mel_spec,
            _video_aligned_spec,
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
            gap_start_mel, gap_end_mel, gap_start_sec, gap_end_sec = gap_bounds_from_mask(
                mask[sample_idx].detach().cpu().numpy()
            )
            identity = {
                "video_path": str(video_path[sample_idx]),
                "gap_start_frame": int(gap_start_mel),
                "gap_end_frame": int(gap_end_mel),
            }
            identities.append(identity)

            for offset_ms in offsets_ms:
                shifted_frames, shift_frames = shift_video_frames(original_frames, offset_ms)
                expanded_frames.append(torch.from_numpy(shifted_frames).float())
                expanded_meta.append(
                    {
                        "sample_idx": int(sample_idx),
                        "offset_ms": int(offset_ms),
                        "shift_frames": int(shift_frames),
                        "gap_start_mel": int(gap_start_mel),
                        "gap_end_mel": int(gap_end_mel),
                        "gap_start_sec": float(gap_start_sec),
                        "gap_end_sec": float(gap_end_sec),
                    }
                )

        expanded_frames = torch.stack(expanded_frames, dim=0).to(device, non_blocking=True)
        expanded_masked = _repeat_batch_value(masked_spec, condition_count).to(device, non_blocking=True).float()
        expanded_spk = _repeat_batch_value(spk_emb, condition_count).to(device, non_blocking=True).float()
        expanded_avail = _repeat_batch_value(avail, condition_count).to(device, non_blocking=True).bool()
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
                hop_length=HOP_LENGTH,
            )
            valid_length = min(len(reference_audio), len(predicted_audio))
            pesq_score = calculate_pesq(
                reference_audio[:valid_length], predicted_audio[:valid_length], sr=sample_rate
            )
            stoi_score = calculate_stoi(
                reference_audio[:valid_length], predicted_audio[:valid_length], sr=sample_rate
            )

            rows.append(
                {
                    "dataset": dataset_name,
                    "batch_idx": int(batch_idx),
                    "sample_in_batch": int(sample_idx),
                    "video_path": str(video_path[sample_idx]),
                    "gap_ms": int(gap_ms),
                    "offset_ms": int(meta["offset_ms"]),
                    "shift_frames": int(meta["shift_frames"]),
                    "gap_start_mel": int(meta["gap_start_mel"]),
                    "gap_end_mel": int(meta["gap_end_mel"]),
                    "gap_start_sec": float(meta["gap_start_sec"]),
                    "gap_end_sec": float(meta["gap_end_sec"]),
                    "pesq": None if pesq_score is None else float(pesq_score),
                    "stoi": None if stoi_score is None else float(stoi_score),
                }
            )

    return rows, _identity_hash(identities), identities


# -----------------------------------------------------------------------------
# Gap-sweep audio-only baseline loading
# -----------------------------------------------------------------------------

def _read_csv_rows(path: str) -> List[Dict]:
    with open(path, "r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _as_float(value):
    if value in (None, "", "None", "nan", "NaN"):
        return None
    return float(value)


def _as_int(value):
    return int(round(float(value)))


def _normalize_baseline_row(row: Dict) -> Dict:
    normalized = dict(row)
    for key in ("batch_idx", "sample_in_batch", "gap_start_frame", "gap_end_frame"):
        normalized[key] = _as_int(row[key])
    for key in ("pesq", "stoi"):
        normalized[key] = _as_float(row.get(key))
    return normalized


def load_audio_only_baselines(
    baseline_dir: str,
    dataset_name: str,
    expected_seed: int,
    expected_test_subset,
) -> Dict[int, Dict]:
    """Load audio-only baseline summaries and per-sample rows from gap sweep."""
    baselines: Dict[int, Dict] = {}
    for gap_ms in GAP_LENGTHS_MS:
        gap_dir = os.path.join(baseline_dir, dataset_name, f"gap_{gap_ms}ms")
        summary_path = os.path.join(gap_dir, "summary.json")
        rows_path = os.path.join(gap_dir, "audio_only_per_sample.csv")
        if not os.path.isfile(summary_path):
            raise FileNotFoundError(f"Missing gap-sweep summary: {summary_path}")
        if not os.path.isfile(rows_path):
            raise FileNotFoundError(f"Missing gap-sweep audio-only rows: {rows_path}")

        with open(summary_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        if payload.get("mask_type") != "single_gap":
            raise ValueError(f"Baseline was not single-gap: {summary_path}")
        if int(payload.get("mask_seed", -1)) != int(expected_seed):
            raise ValueError(f"Mask seed mismatch in {summary_path}")
        if payload.get("test_subset") != expected_test_subset:
            raise ValueError(f"test_subset mismatch in {summary_path}")
        if payload.get("sample_rate") not in (None, 16000):
            raise ValueError(f"Unexpected sample rate in {summary_path}")
        if payload.get("hop_length") not in (None, HOP_LENGTH):
            raise ValueError(f"Unexpected hop length in {summary_path}")

        rows = [_normalize_baseline_row(row) for row in _read_csv_rows(rows_path)]
        if len(rows) != int(payload.get("num_samples", len(rows))):
            raise ValueError(f"Audio-only row count mismatch in {rows_path}")

        identities = [
            {
                "video_path": str(row["video_path"]),
                "gap_start_frame": int(row["gap_start_frame"]),
                "gap_end_frame": int(row["gap_end_frame"]),
            }
            for row in rows
        ]
        identity_hash = _identity_hash(identities)
        if payload.get("sample_identity_sha256") and identity_hash != payload["sample_identity_sha256"]:
            raise ValueError(f"Audio-only identity hash mismatch in {rows_path}")

        audio_values = payload.get("audio_only", {})
        pesq_mean, pesq_std, pesq_ci95, num_pesq = mean_std_ci95(
            [row["pesq"] for row in rows if row["pesq"] is not None]
        )
        stoi_mean, stoi_std, stoi_ci95, num_stoi = mean_std_ci95(
            [row["stoi"] for row in rows if row["stoi"] is not None]
        )

        baselines[int(gap_ms)] = {
            "summary": {
                **audio_values,
                "pesq_mean": pesq_mean,
                "pesq_std": pesq_std,
                "pesq_ci95": pesq_ci95,
                "stoi_mean": stoi_mean,
                "stoi_std": stoi_std,
                "stoi_ci95": stoi_ci95,
                "num_pesq": num_pesq,
                "num_stoi": num_stoi,
            },
            "rows": rows,
            "rows_by_identity": {_identity_key_from_row(row): row for row in rows},
            "sample_identity_sha256": identity_hash,
            "ordered_sample_identities": identities,
            "num_samples": len(rows),
        }
    return baselines


# -----------------------------------------------------------------------------
# Saved-result reuse and summarization
# -----------------------------------------------------------------------------

def _normalize_sync_row(row: Dict) -> Dict:
    normalized = dict(row)
    for key in (
        "batch_idx", "sample_in_batch", "gap_ms", "offset_ms", "shift_frames",
        "gap_start_mel", "gap_end_mel",
    ):
        normalized[key] = _as_int(row[key])
    for key in ("gap_start_sec", "gap_end_sec", "pesq", "stoi"):
        normalized[key] = _as_float(row.get(key))
    return normalized


def summarize_rows(dataset_name: str, rows: List[Dict], audio_only_baselines: Dict[int, Dict]) -> List[Dict]:
    summaries: List[Dict] = []
    for gap_ms in GAP_LENGTHS_MS:
        baseline_rows = audio_only_baselines[gap_ms]["rows_by_identity"]
        for offset_ms in SYNC_OFFSETS_MS:
            selected = [
                row for row in rows
                if row["dataset"] == dataset_name
                and row["gap_ms"] == gap_ms
                and row["offset_ms"] == offset_ms
            ]
            if not selected:
                raise ValueError(f"Missing condition: {dataset_name}, {gap_ms} ms, {offset_ms} ms offset")

            pesq_mean, pesq_std, pesq_ci95, num_pesq = mean_std_ci95(
                [row["pesq"] for row in selected if row["pesq"] is not None]
            )
            stoi_mean, stoi_std, stoi_ci95, num_stoi = mean_std_ci95(
                [row["stoi"] for row in selected if row["stoi"] is not None]
            )

            delta_pesq = []
            delta_stoi = []
            for row in selected:
                base = baseline_rows.get(_identity_key_from_row(row))
                if base is None:
                    raise ValueError(
                        f"No matching audio-only baseline row for {dataset_name}, "
                        f"{gap_ms} ms, offset {offset_ms} ms, {row['video_path']}"
                    )
                if row["pesq"] is not None and base["pesq"] is not None:
                    delta_pesq.append(float(row["pesq"]) - float(base["pesq"]))
                if row["stoi"] is not None and base["stoi"] is not None:
                    delta_stoi.append(float(row["stoi"]) - float(base["stoi"]))

            d_pesq_mean, d_pesq_std, d_pesq_ci95, d_num_pesq = mean_std_ci95(delta_pesq)
            d_stoi_mean, d_stoi_std, d_stoi_ci95, d_num_stoi = mean_std_ci95(delta_stoi)

            summaries.append(
                {
                    "dataset": dataset_name,
                    "gap_ms": int(gap_ms),
                    "offset_ms": int(offset_ms),
                    "shift_frames": int(round(offset_ms / FRAME_MS)),
                    "pesq_mean": pesq_mean,
                    "pesq_std": pesq_std,
                    "pesq_ci95": pesq_ci95,
                    "stoi_mean": stoi_mean,
                    "stoi_std": stoi_std,
                    "stoi_ci95": stoi_ci95,
                    "num_pesq": num_pesq,
                    "num_stoi": num_stoi,
                    "delta_pesq_av_minus_audio_mean": d_pesq_mean,
                    "delta_pesq_av_minus_audio_std": d_pesq_std,
                    "delta_pesq_av_minus_audio_ci95": d_pesq_ci95,
                    "delta_num_pesq": d_num_pesq,
                    "delta_stoi_av_minus_audio_mean": d_stoi_mean,
                    "delta_stoi_av_minus_audio_std": d_stoi_std,
                    "delta_stoi_av_minus_audio_ci95": d_stoi_ci95,
                    "delta_num_stoi": d_num_stoi,
                }
            )
    return summaries


def load_complete_stored_dataset(
    results_dir: str,
    dataset_name: str,
    audio_only_baselines: Dict[int, Dict],
) -> Tuple[List[Dict], List[Dict]]:
    path = os.path.join(results_dir, dataset_name, "per_sample_results.csv")
    if not os.path.isfile(path):
        return None, None

    try:
        rows = [_normalize_sync_row(row) for row in _read_csv_rows(path)]
        rows = [row for row in rows if row.get("dataset") == dataset_name]
        for gap_ms in GAP_LENGTHS_MS:
            expected_n = int(audio_only_baselines[gap_ms]["num_samples"])
            expected_hash = audio_only_baselines[gap_ms]["sample_identity_sha256"]
            for offset_ms in SYNC_OFFSETS_MS:
                selected = sorted(
                    [
                        row for row in rows
                        if row["gap_ms"] == gap_ms
                        and row["offset_ms"] == offset_ms
                    ],
                    key=lambda row: (row["batch_idx"], row["sample_in_batch"]),
                )
                if len(selected) != expected_n:
                    print(
                        f"Stored sync results incomplete for {dataset_name}: "
                        f"{gap_ms} ms, offset {offset_ms} ms has "
                        f"{len(selected)}/{expected_n} rows."
                    )
                    return None, None

                identities = [
                    {
                        "video_path": str(row["video_path"]),
                        "gap_start_frame": int(row["gap_start_mel"]),
                        "gap_end_frame": int(row["gap_end_mel"]),
                    }
                    for row in selected
                ]
                if _identity_hash(identities) != expected_hash:
                    print(
                        f"Stored sample identities do not match the gap sweep for "
                        f"{dataset_name}, gap={gap_ms} ms, offset={offset_ms} ms."
                    )
                    return None, None
        summaries = summarize_rows(dataset_name, rows, audio_only_baselines)
        print(
            f"Complete stored sync-shift results found for {dataset_name}; "
            "skipping inference and regenerating summaries/plots."
        )
        return summaries, rows
    except Exception as exc:
        print(f"Could not reuse stored sync-shift results for {dataset_name}: {exc}")
        return None, None


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


def configure_score_axis(ax, metric: str) -> None:
    ax.set_xlim(min(SYNC_OFFSETS_MS) - 20, max(SYNC_OFFSETS_MS) + 20)
    ax.set_ylim(*METRIC_LIMITS[metric])
    ax.set_xticks(SYNC_OFFSETS_MS)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(which="major", alpha=0.45, linewidth=GRID_MAJOR_LINEWIDTH)
    ax.grid(which="minor", alpha=0.20, linewidth=GRID_MINOR_LINEWIDTH, linestyle=":")
    ax.tick_params(which="both", direction="in", length=4)
    ax.tick_params(which="minor", length=2)
    ax.axvline(0, linestyle=":", linewidth=REFERENCE_LINE_WIDTH, alpha=0.8)


def configure_delta_axis(ax) -> None:
    ax.set_xlim(min(SYNC_OFFSETS_MS) - 20, max(SYNC_OFFSETS_MS) + 20)
    ax.set_xticks(SYNC_OFFSETS_MS)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(which="major", alpha=0.45, linewidth=GRID_MAJOR_LINEWIDTH)
    ax.grid(which="minor", alpha=0.20, linewidth=GRID_MINOR_LINEWIDTH, linestyle=":")
    ax.tick_params(which="both", direction="in", length=4)
    ax.tick_params(which="minor", length=2)
    ax.axhline(0, linestyle=":", linewidth=REFERENCE_LINE_WIDTH, alpha=0.8)
    ax.axvline(0, linestyle=":", linewidth=REFERENCE_LINE_WIDTH, alpha=0.8)


def plot_sync_score_panel(
    ax,
    dataset_results: List[Dict],
    audio_only_baselines: Dict[int, Dict],
    metric: str,
) -> None:
    """Plot AV and audio-only results for all gaps across synchronization offsets."""
    offsets = np.asarray(SYNC_OFFSETS_MS, dtype=np.float64)

    for gap_ms in GAP_LENGTHS_MS:
        style = GAP_STYLES[gap_ms]
        selected = sorted(
            [row for row in dataset_results if int(row["gap_ms"]) == int(gap_ms)],
            key=lambda row: row["offset_ms"],
        )
        if not selected:
            raise ValueError(
                f"Missing summary rows for gap {gap_ms} ms while plotting {metric}."
            )

        x_values = np.asarray([row["offset_ms"] for row in selected], dtype=np.float64)
        av_mean = np.asarray([row[f"{metric}_mean"] for row in selected], dtype=np.float64)
        av_ci95 = np.asarray([row[f"{metric}_ci95"] for row in selected], dtype=np.float64)
        valid = np.isfinite(x_values) & np.isfinite(av_mean) & np.isfinite(av_ci95)
        if not np.any(valid):
            continue

        ax.plot(
            x_values[valid],
            av_mean[valid],
            color=MODEL_COLORS["av"],
            linestyle=style["linestyle"],
            marker=AV_STYLE["marker"],
            markerfacecolor=MODEL_COLORS["av"],
            markeredgecolor="white",
            markeredgewidth=MARKER_EDGE_WIDTH,
            linewidth=AV_LINE_WIDTH,
            markersize=MARKER_SIZE,
            zorder=4,
        )
        ax.fill_between(
            x_values[valid],
            av_mean[valid] - av_ci95[valid],
            av_mean[valid] + av_ci95[valid],
            color=MODEL_COLORS["av"],
            alpha=0.10,
            linewidth=0,
            zorder=2,
        )

        baseline = audio_only_baselines[gap_ms]["summary"]
        audio_mean = baseline.get(f"{metric}_mean")
        audio_ci95 = baseline.get(f"{metric}_ci95")
        if (
            audio_mean is None
            or audio_ci95 is None
            or not np.isfinite(audio_mean)
            or not np.isfinite(audio_ci95)
        ):
            raise ValueError(
                f"Missing/non-finite audio-only {metric} statistics for gap {gap_ms} ms."
            )

        audio_mean = float(audio_mean)
        audio_ci95 = float(audio_ci95)
        ax.plot(
            offsets,
            np.full_like(offsets, audio_mean),
            color=MODEL_COLORS["audio"],
            linestyle=style["linestyle"],
            marker=A_STYLE["marker"],
            markerfacecolor=MODEL_COLORS["audio"],
            markeredgecolor="white",
            markeredgewidth=MARKER_EDGE_WIDTH,
            linewidth=AUDIO_ONLY_LINE_WIDTH,
            markersize=MARKER_SIZE,
            zorder=3,
        )
        ax.fill_between(
            offsets,
            audio_mean - audio_ci95,
            audio_mean + audio_ci95,
            color=MODEL_COLORS["audio"],
            alpha=0.08,
            linewidth=0,
            zorder=1,
        )

    configure_score_axis(ax, metric)
    ax.set_xlabel("Video offset (ms)")

def _sync_legend_handles() -> List[Line2D]:
    handles = [
        Line2D(
            [0], [0],
            color=MODEL_COLORS["audio"],
            linewidth=LEGEND_LINE_WIDTH,
            linestyle="-",
            marker=A_STYLE["marker"],
            markersize=A_STYLE["markersize"],
            markerfacecolor=MODEL_COLORS["audio"],
            markeredgecolor="white",
            markeredgewidth=MARKER_EDGE_WIDTH,
            label="Audio-only",
        ),
        Line2D(
            [0], [0],
            color=MODEL_COLORS["av"],
            linewidth=LEGEND_LINE_WIDTH,
            linestyle="-",
            marker=AV_STYLE["marker"],
            markersize=AV_STYLE["markersize"],
            markerfacecolor=MODEL_COLORS["av"],
            markeredgecolor="white",
            markeredgewidth=MARKER_EDGE_WIDTH,
            label="Audio-visual",
        ),
    ]

    # Gap-duration entries: black lines with the matching dash pattern.
    for gap_ms in GAP_LENGTHS_MS:
        handles.append(
            Line2D(
                [0], [0],
                color="black",
                linestyle=GAP_STYLES[gap_ms]["linestyle"],
                linewidth=LEGEND_LINE_WIDTH,
                marker="none",
                label=f"{gap_ms} ms",
            )
        )
    return handles

# def _sync_legend_handles() -> List[Line2D]:
#     handles = [
#         Line2D([0], [0], color=MODEL_COLORS["audio"], linewidth=LEGEND_LINE_WIDTH, label="Audio-only"),
#         Line2D([0], [0], color=MODEL_COLORS["av"], linewidth=LEGEND_LINE_WIDTH, label="Audio-visual"),
#     ]
#     for gap_ms in GAP_LENGTHS_MS:
#         style = GAP_STYLES[gap_ms]
#         handles.append(
#             Line2D(
#                 [0],
#                 [0],
#                 color="black",
#                 linestyle=style["linestyle"],
#                 #marker=style["marker"],
#                 markerfacecolor="black",
#                 markeredgecolor="white",
#                 #markeredgewidth=MARKER_EDGE_WIDTH,
#                 linewidth=LEGEND_LINE_WIDTH,
#                 label=f"{gap_ms} ms",
#             )
#         )
#     return handles


def save_dataset_sync_plot(
    dataset_results: List[Dict],
    output_dir: str,
    dataset_name: str,
    audio_only_baselines: Dict[int, Dict],
) -> None:
    if not dataset_results:
        return

    figure_dir = os.path.join(output_dir, dataset_name, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_2COL, FIG_HEIGHT_1ROW), squeeze=False)
    axes = axes[0]

    for ax, metric in zip(axes, ("pesq", "stoi")):
        plot_sync_score_panel(ax, dataset_results, audio_only_baselines, metric)
        ax.set_title(metric.upper())
        ax.set_ylabel(rf"{metric.upper()} ($\uparrow$)")

    fig.legend(
        handles=_sync_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(
        os.path.join(figure_dir, "pesq_stoi_vs_sync_offset_ci95.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_all_datasets_sync_plot(
    all_summaries: List[Dict],
    all_audio_baselines: Dict[str, Dict[int, Dict]],
    output_dir: str,
) -> None:
    dataset_order = ["grid", "lrs2", "voxceleb2"]
    if not all(dataset in all_audio_baselines for dataset in dataset_order):
        return

    fig, axes = plt.subplots(
        len(dataset_order),
        2,
        figsize=(FIG_WIDTH_2COL, FIG_HEIGHT_ROW * len(dataset_order)),
        squeeze=False,
        sharex="col",
        sharey="col",
    )
    display_names = {
        "grid": "GRID",
        "lrs2": "LRS2",
        "voxceleb2": "VoxCeleb2",
    }

    for row_index, dataset_name in enumerate(dataset_order):
        dataset_results = [
            row for row in all_summaries
            if str(row["dataset"]).lower() == dataset_name
        ]
        if not dataset_results:
            raise ValueError(f"No summaries available for required dataset {dataset_name}.")

        for column_index, metric in enumerate(("pesq", "stoi")):
            ax = axes[row_index, column_index]
            plot_sync_score_panel(
                ax,
                dataset_results,
                all_audio_baselines[dataset_name],
                metric,
            )
            if row_index == 0:
                ax.set_title(rf"{metric.upper()} ($\uparrow$)")
            if column_index == 0:
                ax.set_ylabel(
                    display_names[dataset_name]
                    #+ "\n"
                    #+ rf"{metric.upper()} ($\uparrow$)"
                )
            else:
                #ax.set_ylabel(rf"{metric.upper()} ($\uparrow$)")
                ax.set_ylabel("")
            if row_index != len(dataset_order) - 1:
                ax.set_xlabel("")

    fig.legend(
        handles=_sync_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(
        os.path.join(output_dir, "combined_datasets_sync_shift_ci95.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audio-video synchronization tolerance ablation for AV_PLC."
    )
    parser.add_argument("--datasets", nargs="+", default=["grid", "lrs2", "voxceleb2"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default=project_checkpoint_dir("AV_PLC"))
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(project_log_dir("AV_PLC"), "ablation_av_sync_shift"),
    )
    parser.add_argument(
        "--gap_sweep_results_dir",
        type=str,
        default=os.path.join(project_log_dir("AV_PLC"), "ablation_gap_sweep"),
        help="Folder containing reusable audio-only gap-sweep summaries and rows.",
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
    all_audio_baselines: Dict[str, Dict[int, Dict]] = {}

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

        dataset_summaries, dataset_rows = load_complete_stored_dataset(
            args.results_dir,
            dataset_name,
            audio_only_baselines,
        )

        if dataset_summaries is None:
            model_name = default_av_model_name(dataset_name)
            model_file = checkpoint_path(args.checkpoint_dir, model_name)
            model = load_checkpoint_flexible(build_av_model(device), model_file, device)

            dataset_rows = []
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
                rows, identity_hash, identities = evaluate_sync_offsets(
                    model=model,
                    loader=loader,
                    device=device,
                    dataset_root=args.dataset_root,
                    dataset_name=dataset_name,
                    gap_ms=gap_ms,
                    offsets_ms=SYNC_OFFSETS_MS,
                    forward_chunk_size=args.forward_chunk_size,
                    sample_rate=args.sample_rate,
                )

                baseline = audio_only_baselines[gap_ms]
                if identity_hash != baseline["sample_identity_sha256"]:
                    expected = baseline["ordered_sample_identities"]
                    mismatch_index = next(
                        (
                            index
                            for index, (current, saved) in enumerate(zip(identities, expected))
                            if current != saved
                        ),
                        min(len(identities), len(expected)),
                    )
                    raise ValueError(
                        f"Sample/mask identity mismatch for {dataset_name}, {gap_ms} ms "
                        f"at ordered index {mismatch_index}. Gap-sweep baseline and "
                        "sync-shift results cannot be paired."
                    )
                if len(identities) != baseline["num_samples"]:
                    raise ValueError(
                        f"Sample-count mismatch for {dataset_name}, {gap_ms} ms: "
                        f"{len(identities)} vs {baseline['num_samples']}"
                    )

                dataset_rows.extend(rows)

            dataset_summaries = summarize_rows(dataset_name, dataset_rows, audio_only_baselines)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        dataset_dir = os.path.join(args.results_dir, dataset_name)
        save_rows_csv(dataset_rows, os.path.join(dataset_dir, "per_sample_results.csv"))
        save_json(dataset_summaries, os.path.join(dataset_dir, "summary.json"))
        save_summary_csv(dataset_summaries, os.path.join(dataset_dir, "summary.csv"))
        save_dataset_sync_plot(
            dataset_results=dataset_summaries,
            output_dir=args.results_dir,
            dataset_name=dataset_name,
            audio_only_baselines=audio_only_baselines,
        )

        all_summaries.extend(dataset_summaries)
        all_rows.extend(dataset_rows)

    save_json(all_summaries, os.path.join(args.results_dir, "all_summary.json"))
    save_summary_csv(all_summaries, os.path.join(args.results_dir, "all_summary.csv"))
    save_rows_csv(all_rows, os.path.join(args.results_dir, "all_per_sample_results.csv"))

    required_datasets = {"grid", "lrs2", "voxceleb2"}
    if required_datasets.issubset(set(all_audio_baselines)):
        save_all_datasets_sync_plot(
            all_summaries=all_summaries,
            all_audio_baselines=all_audio_baselines,
            output_dir=args.results_dir,
        )
    else:
        missing = sorted(required_datasets - set(all_audio_baselines))
        print(
            "Skipping combined sync-shift figure because these datasets "
            f"were not processed in this run: {missing}"
        )

    print(f"\nFinished. Results saved to: {args.results_dir}")


if __name__ == "__main__":
    main()
