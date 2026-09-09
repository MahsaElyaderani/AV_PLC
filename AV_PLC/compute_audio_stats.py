"""Compute and save one fixed clean-training log-Mel mean/std per dataset."""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch

from AV_PLC.audio_frontend import AudioFrontend, AudioFrontendConfig
from evaluations.runtime_config import dataset_patterns, normalize_dataset_name

STATS_DIR = Path(__file__).resolve().parent / "mel_stats"


def decode_audio(ds) -> torch.Tensor:
    x = ds[:]
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    else:
        x = x.astype(np.float32)
    return torch.from_numpy(x)


def compute(pattern: str, lookahead_ms: float = 7.5) -> dict:
    """Compute scalar stats from clean valid train frames only."""
    frontend = AudioFrontend(AudioFrontendConfig(lookahead_ms=lookahead_ms))
    total = 0
    sum_x = 0.0
    sum_x2 = 0.0
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No HDF5 files match: {pattern}")

    for path in files:
        with h5py.File(path, "r") as h5f:
            for key in h5f.keys():
                audio = decode_audio(h5f[f"{key}/audio"])
                audio_len = int(h5f.attrs[f"{key}/audio_len"])
                audio_len = max(0, min(audio_len, audio.numel()))
                mel = frontend.logmel(audio)
                keep = torch.ones_like(audio)
                frame_valid, _, _ = frontend.frame_masks(keep, audio_len)
                values = mel[:, frame_valid]
                if values.numel() == 0:
                    continue
                v = values.double()
                sum_x += float(v.sum())
                sum_x2 += float(v.square().sum())
                total += v.numel()

    if total == 0:
        raise RuntimeError("No valid Mel values were found while computing statistics")
    mean = sum_x / total
    var = max(0.0, sum_x2 / total - mean * mean)
    std = math.sqrt(var)
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0.0:
        raise RuntimeError(f"Invalid statistics: mean={mean}, std={std}")
    return {
        "mean": mean,
        "std": std,
        "count": total,
        "reference_lookahead_ms": float(lookahead_ms),
    }


def save_dataset_stats(dataset_name: str, pattern: str | None = None,
                       output: str | None = None, lookahead_ms: float = 7.5) -> Path:
    dataset_name = normalize_dataset_name(dataset_name)
    pattern = pattern or dataset_patterns(dataset_name)["train"]
    output_path = Path(output) if output else STATS_DIR / f"{dataset_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = compute(pattern, lookahead_ms)
    stats["dataset"] = dataset_name
    stats["train_h5_pattern"] = str(pattern)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return output_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["grid", "lrs2", "voxceleb2"], required=True)
    p.add_argument("--h5", default=None,
                   help="Optional training HDF5 glob. Defaults to runtime_config dataset pattern.")
    p.add_argument("--output", default=None,
                   help="Optional output JSON. Defaults to AV_PLC/mel_stats/<dataset>.json")
    p.add_argument("--reference-lookahead-ms", type=float, default=7.5,
                   help="One fixed stats set is reused for all latency experiments.")
    args = p.parse_args()
    out = save_dataset_stats(
        args.dataset, pattern=args.h5, output=args.output,
        lookahead_ms=args.reference_lookahead_ms,
    )
    print(out)
    print(out.read_text())


if __name__ == "__main__":
    main()
