import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from evaluations.runtime_config import project_log_dir, project_checkpoint_dir, set_global_seed, SEED

import os
import json
import csv
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from audio_encoder import Audio_Encoder
from multimodal_decoder import AV_PLC
from av_dataloader import AVDataloader
from shared.metrics import calculate_pesq, calculate_stoi
from shared.audio_processing import load_audio_ffmpeg, torch_mel2audio


mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 15,
})

# ---------------------------
# Fixed single-gap masking
# ---------------------------

GAP_LENGTHS_MS = [20, 40, 80, 160, 320, 500, 750, 1000, 1500]

def deterministic_gap_start(idx: int, T: int, gap_frames: int, seed: int = SEED) -> int:
    """
    Pick a reproducible random gap start for each sample index.

    Same idx + same gap length + same seed -> same gap location.
    This guarantees audio-only and AV models get the same gap.
    """
    max_start = max(0, T - gap_frames)
    rng = np.random.default_rng(seed + idx)
    return int(rng.integers(0, max_start + 1))


def make_single_gap_mask(
    spec_shape,
    idx: int,
    gap_ms: int,
    clip_sec: float = 3.0,
    seed: int = SEED,
):
    """
    Create a single continuous time gap mask.

    spec_shape: usually [80, T]
    returns:
        mask: [80, T], 1 = keep, 0 = missing
        gap_start_frame
        gap_end_frame
    """
    F, T = spec_shape

    # For your setup this is normally 300 mel frames / 3 sec = 100 fps.
    mel_fps = T / clip_sec

    gap_frames = int(round((gap_ms / 1000.0) * mel_fps))
    gap_frames = max(1, min(gap_frames, T))

    start = deterministic_gap_start(idx, T, gap_frames, seed=seed)
    end = min(T, start + gap_frames)

    mask = np.ones((F, T), dtype=np.float32)
    mask[:, start:end] = 0.0

    return mask, start, end


# ---------------------------
# Dataset wrapper
# ---------------------------

class FixedGapWrapper(Dataset):
    """
    Wraps your existing AVDataset output and replaces its original GE/HDF5 mask
    with our deterministic single-gap mask.

    This avoids modifying av_dataloader.py.
    """

    def __init__(self, base_dataset, mode: str, gap_ms: int, seed: int = SEED):
        assert mode in ["a", "av"]
        self.base_dataset = base_dataset
        self.mode = mode
        self.gap_ms = gap_ms
        self.seed = seed

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]

        if self.mode == "a":
            # Current audio-only item:
            # masked_spec, mel_spec, audio_length, text, mask, video_path
            _, mel_spec, audio_length, text, _, video_path = item

            mask_np, gap_start, gap_end = make_single_gap_mask(
                spec_shape=mel_spec.shape,
                idx=idx,
                gap_ms=self.gap_ms,
                seed=self.seed,
            )

            masked_spec = mel_spec * mask_np

            return (
                masked_spec.astype(np.float32),
                mel_spec.astype(np.float32),
                audio_length,
                text,
                mask_np.astype(np.float32),
                video_path,
                gap_start,
                gap_end,
            )

        else:
            # Current AV item:
            # frames, spk_emb, masked_spec, mel_spec, audio_length,
            # text, mask, video_path[, avail]
            if len(item) == 9:
                frames, spk_emb, _, mel_spec, audio_length, text, _, video_path, avail = item
            else:
                frames, spk_emb, _, mel_spec, audio_length, text, _, video_path = item
                avail = np.array([True, True], dtype=np.bool_)

            mask_np, gap_start, gap_end = make_single_gap_mask(
                spec_shape=mel_spec.shape,
                idx=idx,
                gap_ms=self.gap_ms,
                seed=self.seed,
            )

            masked_spec = mel_spec * mask_np

            return (
                frames.astype(np.float32),
                spk_emb.astype(np.float32),
                masked_spec.astype(np.float32),
                mel_spec.astype(np.float32),
                audio_length,
                text,
                mask_np.astype(np.float32),
                video_path,
                avail,
                gap_start,
                gap_end,
            )


def build_fixed_gap_loader(
    dataset_name: str,
    mode: str,
    gap_ms: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    test_subset=None,
):
    """
    Builds your normal test dataloader first, then wraps its dataset.

    We pass mask_range='10' only to satisfy your current dataloader.
    The returned mask is replaced by FixedGapWrapper.
    """
    dl_builder = AVDataloader(
        dataset_name=dataset_name,
        mode=mode,
        batch_size=batch_size,
        num_workers=num_workers,
        video_aug=False,
        dropout_modality=False,
        test_subset=test_subset,
    )

    base_loader = dl_builder.test_dataloader(mask_range="10", seed=seed)
    wrapped_dataset = FixedGapWrapper(
        base_dataset=base_loader.dataset,
        mode=mode,
        gap_ms=gap_ms,
        seed=seed,
    )

    return DataLoader(
        wrapped_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
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
    Same idea as audio_encoder.py:
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
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(which="major", alpha=0.45, linewidth=0.8)
    ax.grid(which="minor", alpha=0.20, linewidth=0.5, linestyle=":")
    ax.tick_params(which="both", direction="in")


def save_gap_summary_plots(dataset_results, output_dir, dataset_name):
    if not dataset_results:
        return

    gap_ms = [row["gap_ms"] for row in dataset_results]
    series = {
        "PESQ": (
            [row["audio_only"]["pesq_mean"] for row in dataset_results],
            [row["audio_visual"]["pesq_mean"] for row in dataset_results],
        ),
        "STOI": (
            [row["audio_only"]["stoi_mean"] for row in dataset_results],
            [row["audio_visual"]["stoi_mean"] for row in dataset_results],
        ),
    }

    figure_dir = os.path.join(output_dir, dataset_name, "figures")
    os.makedirs(figure_dir, exist_ok=True)

    for metric, (audio_values, av_values) in series.items():
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.plot(gap_ms, audio_values, "o-", label="Audio-only")
        ax.plot(gap_ms, av_values, "s-", label="Audio-Video")
        ax.set_xlabel("Audio gap duration (ms)")
        ax.set_ylabel(rf"{metric} ($\uparrow$)")
        _configure_gap_axis(ax)
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.18),
            ncol=2, frameon=False,
        )
        fig.tight_layout()
        fig.savefig(
            os.path.join(figure_dir, f"{metric.lower()}_vs_gap_duration.png"),
            dpi=300, bbox_inches="tight",
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
        (
            masked_spec, mel_spec, audio_length, text, mask, video_path,
            gap_start, gap_end,
        ) = batch

        masked_spec = masked_spec.to(device).float()
        mel_spec = mel_spec.to(device).float()

        pred_mel, _ = model(masked_spec)

        B = pred_mel.size(0)

        for i in range(B):
            path_i = resolve_audio_path(video_path[i], dataset_root)
            ref_audio = load_audio_ffmpeg(path_i, sr=sample_rate, fixlen_sec=3)

            reconstructed_audio = torch_mel2audio(
                pred_mel[i].detach().cpu(), mel_mean, mel_std
            )
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

            row = {
                "batch_idx": int(batch_idx),
                "sample_in_batch": int(i),
                "video_path": str(video_path[i]),
                "gap_start_frame": int(gap_start[i]),
                "gap_end_frame": int(gap_end[i]),
                "pesq": None if pesq_score is None else float(pesq_score),
                "stoi": None if stoi_score is None else float(stoi_score),
            }
            all_rows.append(row)

            if pesq_score is not None:
                pesq_values.append(float(pesq_score))
            if stoi_score is not None:
                stoi_values.append(float(stoi_score))

    summary = {
        "pesq_mean": float(np.mean(pesq_values)) if pesq_values else None,
        "pesq_std": float(np.std(pesq_values)) if pesq_values else None,
        "stoi_mean": float(np.mean(stoi_values)) if stoi_values else None,
        "stoi_std": float(np.std(stoi_values)) if stoi_values else None,
        "num_pesq": len(pesq_values),
        "num_stoi": len(stoi_values),
    }

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
            gap_start,
            gap_end,
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

            reconstructed_audio = torch_mel2audio(
                fused_mel[i].detach().cpu(), mel_mean, mel_std
            )
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

            row = {
                "batch_idx": int(batch_idx),
                "sample_in_batch": int(i),
                "video_path": str(video_path[i]),
                "gap_start_frame": int(gap_start[i]),
                "gap_end_frame": int(gap_end[i]),
                "pesq": None if pesq_score is None else float(pesq_score),
                "stoi": None if stoi_score is None else float(stoi_score),
            }
            all_rows.append(row)

            if pesq_score is not None:
                pesq_values.append(float(pesq_score))
            if stoi_score is not None:
                stoi_values.append(float(stoi_score))

    summary = {
        "pesq_mean": float(np.mean(pesq_values)) if pesq_values else None,
        "pesq_std": float(np.std(pesq_values)) if pesq_values else None,
        "stoi_mean": float(np.mean(stoi_values)) if stoi_values else None,
        "stoi_std": float(np.std(stoi_values)) if stoi_values else None,
        "num_pesq": len(pesq_values),
        "num_stoi": len(stoi_values),
    }

    return summary, all_rows


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

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=project_checkpoint_dir("AV_PLC"),
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(project_log_dir("AV_PLC"), "ablation_gap_sweep"),
    )

    parser.add_argument(
        "--vocoder_path",
        type=str,
        default=None #"/home/ai/Projects/Mahsa/sources/AV_PLC/hifigan/checkpoints/model-best.pt",
    )

    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/home/ai/Projects/Mahsa/datasets",
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
        print(f"\n==============================")
        print(f"Dataset: {dataset_name}")
        print(f"==============================")

        dataset_results = []

        # -----------------------
        # Load audio-only model
        # -----------------------
        audio_model_name = default_audio_model_name(dataset_name)
        audio_ckpt = get_checkpoint_path(args.checkpoint_dir, audio_model_name)

        audio_model = build_audio_model(dataset_name, device)
        audio_model = load_checkpoint_flexible(audio_model, audio_ckpt, device)

        # -----------------------
        # Load AV model
        # -----------------------
        av_model_name = default_av_model_name(dataset_name)
        av_ckpt = get_checkpoint_path(args.checkpoint_dir, av_model_name)

        av_model = build_av_model(dataset_name, device)
        av_model = load_checkpoint_flexible(av_model, av_ckpt, device)

        for gap_ms in GAP_LENGTHS_MS:
            print(f"\n--- Gap length: {gap_ms} ms ---")

            # Same seed, same idx ordering, same gap_ms
            # => same gap locations for both models.
            audio_loader = build_fixed_gap_loader(
                dataset_name=dataset_name,
                mode="a",
                gap_ms=gap_ms,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=SEED,
                test_subset=args.test_subset,
            )

            av_loader = build_fixed_gap_loader(
                dataset_name=dataset_name,
                mode="av",
                gap_ms=gap_ms,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=SEED,
                test_subset=args.test_subset,
            )

            audio_summary, audio_rows = evaluate_audio_model(
                model=audio_model,
                loader=audio_loader,
                #vocoder=vocoder,
                device=device,
                dataset_root=args.dataset_root,
                sample_rate=args.sample_rate,
            )

            av_summary, av_rows = evaluate_av_model(
                model=av_model,
                loader=av_loader,
                #vocoder=vocoder,
                device=device,
                dataset_root=args.dataset_root,
                sample_rate=args.sample_rate,
            )

            gap_result = {
                "dataset": dataset_name,
                "gap_ms": gap_ms,
                "audio_model": audio_model_name,
                "av_model": av_model_name,
                "audio_only": audio_summary,
                "audio_visual": av_summary,
                "delta_pesq_av_minus_audio": None,
                "delta_stoi_av_minus_audio": None,
            }

            if audio_summary["pesq_mean"] is not None and av_summary["pesq_mean"] is not None:
                gap_result["delta_pesq_av_minus_audio"] = (
                    av_summary["pesq_mean"] - audio_summary["pesq_mean"]
                )

            if audio_summary["stoi_mean"] is not None and av_summary["stoi_mean"] is not None:
                gap_result["delta_stoi_av_minus_audio"] = (
                    av_summary["stoi_mean"] - audio_summary["stoi_mean"]
                )

            final_summary.append(gap_result)
            dataset_results.append(gap_result)

            out_base = os.path.join(args.results_dir, dataset_name, f"gap_{gap_ms}ms")

            save_rows_csv(
                audio_rows,
                os.path.join(out_base, "audio_only_per_sample.csv"),
            )
            save_rows_csv(
                av_rows,
                os.path.join(out_base, "audio_visual_per_sample.csv"),
            )
            save_json(
                gap_result,
                os.path.join(out_base, "summary.json"),
            )

            print("Audio-only:", audio_summary)
            print("Audio-visual:", av_summary)
            print(
                "Delta PESQ:",
                gap_result["delta_pesq_av_minus_audio"],
                "| Delta STOI:",
                gap_result["delta_stoi_av_minus_audio"],
            )

        save_gap_summary_plots(
            dataset_results=dataset_results,
            output_dir=args.results_dir,
            dataset_name=dataset_name,
        )

    save_json(
        final_summary,
        os.path.join(args.results_dir, "all_results_summary.json"),
    )

    print(f"\nSaved all results to: {args.results_dir}")


if __name__ == "__main__":
    main()

#Debug:

#python experiment1_gap_sweep.py \
#  --datasets grid \
#  --batch_size 2 \
#  --num_workers 0 \
#  --test_subset 10

# Main:

# python experiment1_gap_sweep.py \
#  --datasets grid lrs2 voxceleb2 \
#  --batch_size 8 \
#  --num_workers 4 \
#  --checkpoint_dir checkpoints \
#  --results_dir results_exp1_gap_sweep \
#  --vocoder_path /home/ai/Projects/Mahsa/sources/AV_PLC/hifigan/checkpoints/model-best.pt \
#  --dataset_root /home/ai/Projects/Mahsa/datasets