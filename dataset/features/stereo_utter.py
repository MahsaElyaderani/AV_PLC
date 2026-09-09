#!/usr/bin/env python3
"""
Check how many samples in train/val/test have stereo audio.

Reads video paths from H5 feature files, resolves them on the current
server, then uses ffprobe to check the audio channel count.

Usage:
    python check_stereo_audio_by_split.py

Dataset root comes from evaluations/runtime_config.py (DATA_ROOT), not a CLI flag.
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from evaluations.runtime_config import DATA_ROOT

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import h5py

H5_PATTERNS = {
    "grid": {
        "train": ["grid/grid_train_features_chunk*.h5"],
        "val":   ["grid/grid_val_features_chunk*.h5"],
        "test":  ["grid/grid_test_features_chunk*.h5"],
    },
    "lrs2": {
        "train": ["lrs2/lrs2_train_features_chunk*.h5"],
        "val":   ["lrs2/lrs2_val_features_chunk*.h5"],
        "test":  ["lrs2/lrs2_test_features_chunk*.h5"],
    },
    "voxceleb2": {
        "train": ["vox2_short/vox2_short_dev_features_chunk*.h5"],
        "val":   ["vox2_short/vox2_short_val_features_chunk*.h5"],
        "test":  ["vox2_short/vox2_short_test_features_chunk*.h5"],
    },
}

KNOWN_ROOTS = ("grid/", "lrs2/", "vox2_short/")


def dataset_relative_path(video_path: str) -> str:
    """Strip the server-specific prefix, keep the portable dataset-relative part."""
    path = video_path.replace("\\", "/")
    if "/datasets/" in path:
        return path.split("/datasets/", 1)[1].lstrip("/")
    if path.startswith("datasets/"):
        return path[len("datasets/"):].lstrip("/")
    for root in KNOWN_ROOTS:
        idx = path.find(root)
        if idx >= 0:
            return path[idx:]
    return path.lstrip("/")


def read_video_paths(h5_path: Path) -> list[str]:
    """Read every video_path stored in an H5 file (file-level or per-group attrs)."""
    paths = []
    with h5py.File(h5_path, "r") as h5f:
        for key in h5f.keys():
            value = h5f.attrs.get(f"{key}/video_path")
            if value is None:
                value = h5f[key].attrs.get("video_path")
            if value:
                paths.append(value.decode() if isinstance(value, bytes) else str(value))
    return paths


def collect_h5_files(dataset_root: Path) -> dict[tuple[str, str], list[Path]]:
    result = {}
    for dataset, splits in H5_PATTERNS.items():
        for split, patterns in splits.items():
            files = [f for pattern in patterns for f in dataset_root.glob(pattern)]
            result[(dataset, split)] = sorted(set(files))
    return result


def resolve_video_path(original_path: str, dataset_root: Path) -> Path | None:
    """Try the path exactly as stored; otherwise look it up under dataset_root."""
    original = Path(original_path)
    if original.is_file():
        return original
    candidate = dataset_root / dataset_relative_path(original_path)
    return candidate if candidate.is_file() else None


def audio_channels(video_path: Path) -> int | None:
    """Return the channel count of the first audio stream, or None if unknown."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=channels", "-of", "json", str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError("ffprobe not found. Install FFmpeg and make sure it's on PATH.")

    if result.returncode != 0:
        return None
    try:
        streams = json.loads(result.stdout).get("streams", [])
    except json.JSONDecodeError:
        return None
    return streams[0].get("channels") if streams else None


def classify(channels: int | None) -> str:
    if channels is None:
        return "audio_probe_failed"
    if channels == 1:
        return "mono"
    if channels == 2:
        return "stereo"
    if channels > 2:
        return "multi_channel"
    return "no_audio_or_unknown"


def summarize(dataset_root: Path, max_samples: int | None = None):
    summary_rows, sample_rows = [], []

    for (dataset, split), h5_files in collect_h5_files(dataset_root).items():
        video_paths = []
        for h5_path in h5_files:
            video_paths.extend(read_video_paths(h5_path))
        video_paths = list(dict.fromkeys(video_paths))  # de-dupe, keep order
        if max_samples:
            video_paths = video_paths[:max_samples]

        print(f"\n=== {dataset} / {split} === ({len(video_paths)} samples, {len(h5_files)} h5 files)")

        counts = defaultdict(int)
        for original_path in video_paths:
            counts["total"] += 1
            resolved = resolve_video_path(original_path, dataset_root)
            rel_path = dataset_relative_path(original_path)

            if resolved is None:
                counts["missing_file"] += 1
                status, channels = "missing_file", ""
            else:
                counts["resolved"] += 1
                raw_channels = audio_channels(resolved)
                status = classify(raw_channels)
                counts[status] += 1
                channels = "" if raw_channels is None else raw_channels

            sample_rows.append({
                "dataset": dataset, "split": split,
                "relative_path": rel_path, "original_path": original_path,
                "resolved_path": str(resolved) if resolved else "",
                "channels": channels, "status": status,
                "is_stereo": status == "stereo",
            })

        total, resolved_n, stereo_n = counts["total"], counts["resolved"], counts["stereo"]
        summary_rows.append({
            "dataset": dataset, "split": split,
            "total_samples": total, "resolved_files": resolved_n,
            "missing_files": counts["missing_file"],
            "audio_probe_failed": counts["audio_probe_failed"],
            "mono_count": counts["mono"], "stereo_count": stereo_n,
            "multi_channel_count": counts["multi_channel"],
            "stereo_percent_of_total": 100 * stereo_n / total if total else 0.0,
            "stereo_percent_of_resolved": 100 * stereo_n / resolved_n if resolved_n else 0.0,
        })

    return summary_rows, sample_rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("stereo_audio_summary.csv"))
    parser.add_argument("--samples-output", type=Path, default=Path("stereo_audio_per_sample.csv"))
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    args = parser.parse_args()

    summary_rows, sample_rows = summarize(DATA_ROOT, args.max_samples_per_split)
    write_csv(summary_rows, args.output)
    write_csv(sample_rows, args.samples_output)

    print("\nSummary")
    print("-" * 90)
    print(f"{'dataset':12s} {'split':8s} {'total':>8s} {'resolved':>9s} {'mono':>8s} "
          f"{'stereo':>8s} {'multi':>8s} {'stereo %':>10s}")
    for row in summary_rows:
        print(f"{row['dataset']:12s} {row['split']:8s} {row['total_samples']:8d} "
              f"{row['resolved_files']:9d} {row['mono_count']:8d} {row['stereo_count']:8d} "
              f"{row['multi_channel_count']:8d} {row['stereo_percent_of_total']:10.2f}")

    print(f"\nSaved: {args.output}, {args.samples_output}")


if __name__ == "__main__":
    main()