"""Append clean STFT phase to existing AV_PLC HDF5 datasets.

Run this once before training/evaluating with ``phase_reconstruction=True``.
The STFT geometry exactly mirrors the current Mel extraction in save_features.py:
16 kHz, n_fft=512, win_length=400, hop_length=160, center=False, one-sided,
Hann window, with 176 zero samples padded on both waveform boundaries.
"""

from __future__ import annotations

import argparse
import os
from glob import glob
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from evaluations.runtime_config import DATA_ROOT, dataset_patterns
from shared.audio_processing import load_audio_ffmpeg

SAMPLE_RATE = 16000
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
PAD = (N_FFT - HOP_LENGTH) // 2  # 176; matches save_features.py
TARGET_SECONDS = 3.0
PHASE_BINS = N_FFT // 2 + 1


def resolve_media_path(video_path: str) -> str:
    """Resolve stored dataset paths the same way AV_PLC evaluation does."""
    video_path = str(video_path)
    if os.path.isfile(video_path):
        return video_path
    rel_path = video_path.split("datasets", 1)[-1].lstrip(os.sep)
    candidate = os.path.join(str(DATA_ROOT), rel_path)
    if not os.path.isfile(candidate):
        raise FileNotFoundError(f"Cannot resolve media path: {video_path} -> {candidate}")
    return candidate


def waveform_phase(audio: np.ndarray) -> np.ndarray:
    wav = torch.as_tensor(audio, dtype=torch.float32)
    expected = int(round(SAMPLE_RATE * TARGET_SECONDS))
    if wav.numel() < expected:
        wav = F.pad(wav, (0, expected - wav.numel()))
    elif wav.numel() > expected:
        wav = wav[:expected]

    wav = F.pad(wav, (PAD, PAD), mode="constant", value=0.0)
    window = torch.hann_window(WIN_LENGTH, periodic=True, dtype=wav.dtype)
    stft = torch.stft(
        wav,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        center=False,
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    return torch.angle(stft).cpu().numpy().astype(np.float32, copy=False)


def expand_h5_patterns(dataset: str, splits: list[str]) -> list[str]:
    patterns = dataset_patterns(dataset)
    files: list[str] = []
    for split in splits:
        pattern = str(patterns[split])
        # Mirror AVDataset: runtime_config may provide either a base directory
        # or the complete _chunk*.h5 glob.
        if "_chunk*.h5" not in pattern:
            pattern = os.path.join(pattern, "_chunk*.h5")
        files.extend(glob(pattern))
    # dataset_patterns may overlap; preserve a deterministic unique order.
    return sorted(set(files))


def process_h5(path: str, overwrite: bool, storage_dtype: str) -> tuple[int, int]:
    added = skipped = 0
    np_dtype = np.float16 if storage_dtype == "float16" else np.float32

    with h5py.File(path, "r+") as h5f:
        video_keys = sorted(h5f.keys())
        for video_key in tqdm(video_keys, desc=Path(path).name, leave=False):
            phase_key = f"{video_key}/phase"
            if phase_key in h5f and not overwrite:
                skipped += 1
                continue

            video_path = h5f.attrs.get(f"{video_key}/video_path", None)
            if video_path is None:
                raise KeyError(f"Missing {video_key}/video_path in {path}")
            media_path = resolve_media_path(video_path)
            audio = load_audio_ffmpeg(
                media_path, sr=SAMPLE_RATE, fixlen_sec=TARGET_SECONDS
            )
            phase = waveform_phase(audio)

            if phase.shape[0] != PHASE_BINS:
                raise RuntimeError(
                    f"Unexpected phase bins for {video_key}: {phase.shape}; expected {PHASE_BINS}"
                )
            spec = h5f[f"{video_key}/spec"]
            if phase.shape[-1] != spec.shape[-1]:
                raise RuntimeError(
                    "STFT/Mel time mismatch. Refusing to crop/interpolate phase: "
                    f"{video_key}: phase T={phase.shape[-1]}, Mel T={spec.shape[-1]}"
                )

            if phase_key in h5f:
                del h5f[phase_key]
            h5f.create_dataset(
                phase_key,
                data=phase.astype(np_dtype, copy=False),
                compression="gzip",
            )
            added += 1
    return added, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append clean STFT phase to existing AV_PLC HDF5 chunks."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["grid", "lrs2", "voxceleb2"],
        default=["grid", "lrs2", "voxceleb2"],
    )
    parser.add_argument(
        "--splits", nargs="+", choices=["train", "val", "test"],
        default=["train", "val", "test"]
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--storage-dtype", choices=["float32", "float16"], default="float32"
    )
    args = parser.parse_args()

    total_added = total_skipped = 0
    for dataset in args.datasets:
        files = expand_h5_patterns(dataset, args.splits)
        if not files:
            raise FileNotFoundError(
                f"No HDF5 chunks found for dataset={dataset}, splits={args.splits}"
            )
        print(f"[{dataset}] {len(files)} HDF5 chunks")
        for path in files:
            added, skipped = process_h5(path, args.overwrite, args.storage_dtype)
            total_added += added
            total_skipped += skipped

    print(f"Done. phase added/updated={total_added}, skipped_existing={total_skipped}")


if __name__ == "__main__":
    main()
