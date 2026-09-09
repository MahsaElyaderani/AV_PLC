import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from evaluations.runtime_config import project_log_dir, project_checkpoint_dir, set_global_seed, SEED

import os
import csv
import json
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

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


# ------------------------------------------------------------
# Experiment settings
# ------------------------------------------------------------

GAP_LENGTHS_MS = [160, 500, 1000]

# 0 means use audio-only model.
# K > 0 means use AV model with K uniformly sampled video frames.
VIDEO_KEEP_COUNTS = [0, 3, 6, 12, 24, 48, 75]

# ------------------------------------------------------------
# Fixed single-gap audio mask
# ------------------------------------------------------------

def deterministic_gap_start(idx: int, T: int, gap_frames: int, seed: int = SEED) -> int:
    """
    Same sample index + same gap length + same seed gives the same gap location.
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
    Create one continuous gap in mel time.

    spec_shape: [mel_bins, T_mel]
    returns:
        mask: [mel_bins, T_mel]
        gap_start_frame
        gap_end_frame
    """
    F, T = spec_shape

    mel_fps = T / clip_sec
    gap_frames = int(round((gap_ms / 1000.0) * mel_fps))
    gap_frames = max(1, min(gap_frames, T))

    start = deterministic_gap_start(idx, T, gap_frames, seed=seed)
    end = min(T, start + gap_frames)

    mask = np.ones((F, T), dtype=np.float32)
    mask[:, start:end] = 0.0

    return mask, start, end


# ------------------------------------------------------------
# Uniform video frame-rate reduction
# ------------------------------------------------------------

def uniform_keep_indices(T: int, K: int):
    """
    Uniformly select K frames from T frames.
    For T=75:
        K=3  -> about 1 fps
        K=6  -> about 2 fps
        K=12 -> about 4 fps
        K=24 -> about 8 fps
        K=48 -> about 16 fps
        K=75 -> full 25 fps
    """
    if K <= 0:
        return np.array([], dtype=np.int64)

    if K >= T:
        return np.arange(T, dtype=np.int64)

    keep_idx = np.linspace(0, T - 1, K).round().astype(np.int64)
    keep_idx = np.unique(keep_idx)

    return keep_idx


def repeat_nearest_frames(frames: np.ndarray, K: int):
    """
    frames: [T, H, W] or [T, H, W, C]

    Keep K uniformly selected frames.
    Replace every dropped frame by the nearest kept frame.

    This keeps the tensor length fixed at T=75.
    """
    T = frames.shape[0]

    if K <= 0:
        raise ValueError("K=0 should use the audio-only model, not AV model.")

    keep_idx = uniform_keep_indices(T, K)

    if len(keep_idx) >= T:
        return frames.astype(np.float32), keep_idx

    out = np.empty_like(frames, dtype=np.float32)

    for t in range(T):
        nearest = keep_idx[np.argmin(np.abs(keep_idx - t))]
        out[t] = frames[nearest]

    return out.astype(np.float32), keep_idx


# ------------------------------------------------------------
# Dataset wrapper
# ------------------------------------------------------------

class Experiment2AWrapper(Dataset):
    """
    Wraps your existing AVDataset output.

    It replaces the original dataloader mask with:
        fixed single audio gap

    It also applies:
        uniform video frame-rate reduction with nearest-frame repetition

    For mode='a':
        used when K=0

    For mode='av':
        used when K > 0
    """

    def __init__(self, base_dataset, mode: str, gap_ms: int, video_keep_count: int, seed: int = SEED):
        assert mode in ["a", "av"]
        self.base_dataset = base_dataset
        self.mode = mode
        self.gap_ms = gap_ms
        self.video_keep_count = video_keep_count
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

            frames_repeated, keep_idx = repeat_nearest_frames(
                frames=frames,
                K=self.video_keep_count,
            )

            return (
                frames_repeated.astype(np.float32),
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
                np.array(keep_idx, dtype=np.int64),
            )


def build_loader(
    dataset_name: str,
    mode: str,
    gap_ms: int,
    video_keep_count: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    test_subset=None,
):
    """
    Build your normal test dataloader, then wrap its dataset.

    mask_range='10' is only used to satisfy the existing dataloader.
    The mask is replaced by Experiment2AWrapper.
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

    wrapped_dataset = Experiment2AWrapper(
        base_dataset=base_loader.dataset,
        mode=mode,
        gap_ms=gap_ms,
        video_keep_count=video_keep_count,
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


# ------------------------------------------------------------
# Model loading
# ------------------------------------------------------------

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

    clean_state = {}
    for k, v in state.items():
        clean_state[k.replace("module.", "")] = v
    model.load_state_dict(clean_state, strict=True)

    print(f"Loaded checkpoint strictly: {ckpt_path}")

    return model


def build_audio_model(dataset_name: str, device):
    conformer_blocks = 4 if dataset_name == "grid" else 8
    model = Audio_Encoder(
        conformer_block=conformer_blocks,
        num_heads=4,
    ).to(device)
    return model


def build_av_model(dataset_name: str, device):
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
    return f"audio_bursty_plc({dataset_name})"


def default_av_model_name(dataset_name: str):
    return (
        f"av_wide_masking_mlp_av_only_fusion_5loss_bursty2"
        f"_plc_a0.05_v0.1"
        f"_pesq_0.01"
        f"_asr_0.1"
        f"({dataset_name})"
    )


def checkpoint_path(checkpoint_dir: str, model_name: str):
    return os.path.join(checkpoint_dir, model_name, "best_model.pt")


# ------------------------------------------------------------
# Audio path handling
# ------------------------------------------------------------

def resolve_audio_path(video_path: str, dataset_root: str):
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



# ------------------------------------------------------------
# Reconstruction and plotting helpers
# ------------------------------------------------------------

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
    """Keep original audio outside the gap and insert reconstruction only inside it."""
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


def _configure_axis(ax):
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(which="major", alpha=0.45, linewidth=0.8)
    ax.grid(which="minor", alpha=0.20, linewidth=0.5, linestyle=":")
    ax.tick_params(which="both", direction="in", length=4)
    ax.tick_params(which="minor", length=2)


def save_summary_plots(dataset_results, output_dir, dataset_name):
    """Plot PESQ/STOI versus the number of uniformly retained frames."""
    if not dataset_results:
        return

    figure_dir = os.path.join(output_dir, dataset_name, "figures")
    os.makedirs(figure_dir, exist_ok=True)

    for metric in ("pesq", "stoi"):
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for gap_ms in GAP_LENGTHS_MS:
            selected = sorted(
                (r for r in dataset_results if r["gap_ms"] == gap_ms),
                key=lambda r: int(r["video_keep_count"]),
            )
            x = [r["video_keep_count"] for r in selected]
            y = [r["summary"].get(f"{metric}_mean") for r in selected]
            ax.plot(x, y, "o-", label=f"{gap_ms} ms")

        ax.set_xlabel("Video frames retained in 3 s")
        ax.set_ylabel(rf"{metric.upper()} ($\uparrow$)")
        _configure_axis(ax)
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.18),
            ncol=len(GAP_LENGTHS_MS), frameon=False,
        )
        fig.tight_layout()
        fig.savefig(
            os.path.join(figure_dir, f"{metric}_vs_video_frames.png"),
            dpi=300, bbox_inches="tight",
        )
        plt.close(fig)


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

@torch.no_grad()
def evaluate_audio_only(
    model,
    loader,
    #vocoder,
    device,
    dataset_root,
    sample_rate=16000,
):
    model.eval()
    mel_mean, mel_std = _get_dataset_stats(loader)

    rows = []
    pesq_values = []
    stoi_values = []

    for batch_idx, batch in enumerate(loader):
        (
            masked_spec, mel_spec, audio_length, text, mask, video_path,
            gap_start, gap_end,
        ) = batch

        masked_spec = masked_spec.to(device).float()

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
                ref_audio, reconstructed_audio, mask[i], hop_length=160
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
                "video_keep_count": 0,
                "approx_video_fps": 0.0,
                "pesq": None if pesq_score is None else float(pesq_score),
                "stoi": None if stoi_score is None else float(stoi_score),
            }
            rows.append(row)

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

    return summary, rows


@torch.no_grad()
def evaluate_av(
    model,
    loader,
    #vocoder,
    device,
    dataset_root,
    video_keep_count,
    sample_rate=16000,
):
    model.eval()
    mel_mean, mel_std = _get_dataset_stats(loader)

    rows = []
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
            keep_idx,
        ) = batch

        frames = frames.to(device).float()
        spk_emb = spk_emb.to(device).float()
        masked_spec = masked_spec.to(device).float()
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
                ref_audio, reconstructed_audio, mask[i], hop_length=160
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
                "video_keep_count": int(video_keep_count),
                "approx_video_fps": float(video_keep_count / 3.0),
                "pesq": None if pesq_score is None else float(pesq_score),
                "stoi": None if stoi_score is None else float(stoi_score),
            }
            rows.append(row)

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

    return summary, rows


# ------------------------------------------------------------
# Saving
# ------------------------------------------------------------

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_rows_csv(rows, path):
    if not rows:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

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
        default=os.path.join(project_log_dir("AV_PLC"), "ablation_uniform_video_rate"),
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
    )

    parser.add_argument("--sample_rate", type=int, default=16000)

    # Fast debugging option.
    parser.add_argument("--test_subset", type=int, default=None)

    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    set_global_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #vocoder = Vocoder(args.vocoder_path, device=device)

    all_results = []

    for dataset_name in args.datasets:
        print("\n==============================")
        print(f"Dataset: {dataset_name}")
        print("==============================")
        dataset_results = []

        # Load audio-only model once per dataset.
        audio_model_name = default_audio_model_name(dataset_name)
        audio_ckpt = checkpoint_path(args.checkpoint_dir, audio_model_name)

        audio_model = build_audio_model(dataset_name, device)
        audio_model = load_checkpoint_flexible(audio_model, audio_ckpt, device)

        # Load AV model once per dataset.
        av_model_name = default_av_model_name(dataset_name)
        av_ckpt = checkpoint_path(args.checkpoint_dir, av_model_name)

        av_model = build_av_model(dataset_name, device)
        av_model = load_checkpoint_flexible(av_model, av_ckpt, device)

        for gap_ms in GAP_LENGTHS_MS:
            print(f"\n--- Audio gap: {gap_ms} ms ---")

            for K in VIDEO_KEEP_COUNTS:
                print(f"\nVideo keep count: {K}")

                if K == 0:
                    # 0 frames means true audio-only baseline.
                    loader = build_loader(
                        dataset_name=dataset_name,
                        mode="a",
                        gap_ms=gap_ms,
                        video_keep_count=K,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        seed=SEED,
                        test_subset=args.test_subset,
                    )

                    summary, rows = evaluate_audio_only(
                        model=audio_model,
                        loader=loader,
                        #vocoder=vocoder,
                        device=device,
                        dataset_root=args.dataset_root,
                        sample_rate=args.sample_rate,
                    )

                    model_type = "audio_only"
                    model_name = audio_model_name

                else:
                    # K > 0 means AV model with uniformly repeated video.
                    loader = build_loader(
                        dataset_name=dataset_name,
                        mode="av",
                        gap_ms=gap_ms,
                        video_keep_count=K,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        seed=SEED,
                        test_subset=args.test_subset,
                    )

                    summary, rows = evaluate_av(
                        model=av_model,
                        loader=loader,
                        #vocoder=vocoder,
                        device=device,
                        dataset_root=args.dataset_root,
                        video_keep_count=K,
                        sample_rate=args.sample_rate,
                    )

                    model_type = "audio_visual"
                    model_name = av_model_name

                result = {
                    "dataset": dataset_name,
                    "gap_ms": gap_ms,
                    "video_keep_count": K,
                    "approx_video_fps": float(K / 3.0),
                    "model_type": model_type,
                    "model_name": model_name,
                    "summary": summary,
                }

                all_results.append(result)
                dataset_results.append(result)

                out_base = os.path.join(
                    args.results_dir,
                    dataset_name,
                    f"gap_{gap_ms}ms",
                    f"K_{K}",
                )

                save_rows_csv(rows, os.path.join(out_base, "per_sample.csv"))
                save_json(result, os.path.join(out_base, "summary.json"))

                print(summary)

        save_summary_plots(dataset_results, args.results_dir, dataset_name)

    save_json(all_results, os.path.join(args.results_dir, "all_results_summary.json"))

    print(f"\nSaved all Experiment 2A results to: {args.results_dir}")


if __name__ == "__main__":
    main()

#Debug:
#python experiment2a_uniform_video_rate.py \
#  --datasets grid \
#  --batch_size 2 \
#  --num_workers 0 \
#  --test_subset 10

#Full Run
# python experiment2a_uniform_video_rate.py \
#  --datasets grid lrs2 voxceleb2 \
#  --batch_size 8 \
#  --num_workers 4 \
#  --checkpoint_dir checkpoints \
#  --results_dir results_exp2a_uniform_video_rate \
#  --vocoder_path /home/ai/Projects/Mahsa/sources/AV_PLC/hifigan/checkpoints/model-best.pt \
#  --dataset_root /home/ai/Projects/Mahsa/datasets