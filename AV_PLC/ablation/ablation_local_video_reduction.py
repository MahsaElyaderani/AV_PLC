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
# "all" means keep all frames inside the local window.
LOCAL_KEEP_COUNTS = [0, 2, 4, 8, 12, 16, "all"]

VIDEO_FPS = 25
CLIP_SEC = 3.0
LOCAL_CONTEXT_SEC = 0.320

# ------------------------------------------------------------
# Fixed single-gap audio mask
# ------------------------------------------------------------

def deterministic_gap_start(
    idx: int,
    T: int,
    gap_frames: int,
    seed: int = SEED,
    clip_sec: float = CLIP_SEC,
    edge_margin_sec: float = 0.5,
) -> int:
    """
    Same sample index + same gap length + same seed gives the same gap location.

    The gap is sampled away from the clip edges:
        gap_start_sec in [edge_margin_sec, clip_sec - gap_duration - edge_margin_sec]

    This avoids clipping the local video window too much at the beginning/end of
    the clip, which otherwise can make keep_idx have different lengths across
    samples and break DataLoader batching.
    """
    mel_fps = T / clip_sec
    margin_frames = int(round(edge_margin_sec * mel_fps))

    min_start = margin_frames
    max_start = T - gap_frames - margin_frames

    # Fallback for very long gaps or very short clips: use the full valid range.
    # This keeps the function safe instead of crashing when the requested margin
    # cannot fit.
    if max_start < min_start:
        min_start = 0
        max_start = max(0, T - gap_frames)

    rng = np.random.default_rng(seed + idx)
    return int(rng.integers(min_start, max_start + 1))


def make_single_gap_mask(
    spec_shape,
    idx: int,
    gap_ms: int,
    clip_sec: float = CLIP_SEC,
    seed: int = SEED,
):
    """
    Create one continuous gap in mel time.

    spec_shape: [mel_bins, T_mel]
    returns:
        mask: [mel_bins, T_mel]
        gap_start_mel
        gap_end_mel
        gap_start_sec
        gap_end_sec
    """
    F, T = spec_shape

    mel_fps = T / clip_sec
    gap_frames = int(round((gap_ms / 1000.0) * mel_fps))
    gap_frames = max(1, min(gap_frames, T))

    start = deterministic_gap_start(
        idx=idx,
        T=T,
        gap_frames=gap_frames,
        seed=seed,
        clip_sec=clip_sec,
        edge_margin_sec=0.5,
    )
    end = min(T, start + gap_frames)

    mask = np.ones((F, T), dtype=np.float32)
    mask[:, start:end] = 0.0

    gap_start_sec = start / mel_fps
    gap_end_sec = end / mel_fps

    return mask, start, end, gap_start_sec, gap_end_sec


# ------------------------------------------------------------
# Local video window and local frame reduction
# ------------------------------------------------------------

def get_local_video_window(
    gap_start_sec: float,
    gap_end_sec: float,
    T_video: int,
    video_fps: int = VIDEO_FPS,
    context_sec: float = LOCAL_CONTEXT_SEC,
):
    """
    Local window:
        320 ms before gap + gap duration + 320 ms after gap

    Returns inclusive frame indices:
        local_start, local_end
    """
    local_start = int(np.floor((gap_start_sec - context_sec) * video_fps))
    local_end = int(np.ceil((gap_end_sec + context_sec) * video_fps)) - 1

    local_start = max(0, local_start)
    local_end = min(T_video - 1, local_end)

    if local_end < local_start:
        local_end = local_start

    return local_start, local_end


def uniform_keep_indices_in_window(local_start: int, local_end: int, K):
    """
    Uniformly select K frames inside [local_start, local_end].
    """
    local_idx = np.arange(local_start, local_end + 1, dtype=np.int64)
    local_len = len(local_idx)

    if K == "all":
        return local_idx

    if K <= 0:
        return np.array([], dtype=np.int64)

    if K >= local_len:
        return local_idx

    keep_idx = np.linspace(local_start, local_end, K).round().astype(np.int64)
    keep_idx = np.unique(keep_idx)

    return keep_idx


def reduce_local_video_frames(
    frames: np.ndarray,
    gap_start_sec: float,
    gap_end_sec: float,
    K_local,
    video_fps: int = VIDEO_FPS,
    context_sec: float = LOCAL_CONTEXT_SEC,
):
    """
    frames: [T, H, W] or [T, H, W, C]

    Frames outside local window:
        unchanged

    Frames inside local window:
        uniformly keep K_local frames
        dropped local frames are replaced by nearest kept local frame

    K_local = "all":
        keep all local frames unchanged

    K_local = 0:
        should use audio-only model, not AV model.
    """
    if K_local == 0:
        raise ValueError("K_local=0 should use the audio-only model, not AV model.")

    T_video = frames.shape[0]
    out = frames.copy().astype(np.float32)

    local_start, local_end = get_local_video_window(
        gap_start_sec=gap_start_sec,
        gap_end_sec=gap_end_sec,
        T_video=T_video,
        video_fps=video_fps,
        context_sec=context_sec,
    )

    local_idx = np.arange(local_start, local_end + 1, dtype=np.int64)
    keep_idx = uniform_keep_indices_in_window(local_start, local_end, K_local)

    if K_local == "all" or len(keep_idx) >= len(local_idx):
        return out, local_start, local_end, keep_idx

    for t in local_idx:
        nearest = keep_idx[np.argmin(np.abs(keep_idx - t))]
        out[t] = frames[nearest]

    return out.astype(np.float32), local_start, local_end, keep_idx


# ------------------------------------------------------------
# Dataset wrapper
# ------------------------------------------------------------

class Experiment2BWrapper(Dataset):
    """
    Wraps your existing AVDataset output.

    It replaces the original dataloader mask with:
        fixed single audio gap

    It applies local video-frame reduction only inside:
        320 ms before gap + gap + 320 ms after gap

    For mode='a':
        used when K_local=0

    For mode='av':
        used when K_local > 0 or K_local='all'
    """

    def __init__(self, base_dataset, mode: str, gap_ms: int, local_keep_count, seed: int = SEED):
        assert mode in ["a", "av"]
        self.base_dataset = base_dataset
        self.mode = mode
        self.gap_ms = gap_ms
        self.local_keep_count = local_keep_count
        self.seed = seed

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]

        if self.mode == "a":
            # Current audio-only item:
            # masked_spec, mel_spec, audio_length, text, mask, video_path
            _, mel_spec, audio_length, text, _, video_path = item

            mask_np, gap_start_mel, gap_end_mel, gap_start_sec, gap_end_sec = make_single_gap_mask(
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
                gap_start_mel,
                gap_end_mel,
                np.float32(gap_start_sec),
                np.float32(gap_end_sec),
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

            mask_np, gap_start_mel, gap_end_mel, gap_start_sec, gap_end_sec = make_single_gap_mask(
                spec_shape=mel_spec.shape,
                idx=idx,
                gap_ms=self.gap_ms,
                seed=self.seed,
            )

            masked_spec = mel_spec * mask_np

            frames_reduced, local_start, local_end, keep_idx = reduce_local_video_frames(
                frames=frames,
                gap_start_sec=gap_start_sec,
                gap_end_sec=gap_end_sec,
                K_local=self.local_keep_count,
                video_fps=VIDEO_FPS,
                context_sec=LOCAL_CONTEXT_SEC,
            )

            return (
                frames_reduced.astype(np.float32),
                spk_emb.astype(np.float32),
                masked_spec.astype(np.float32),
                mel_spec.astype(np.float32),
                audio_length,
                text,
                mask_np.astype(np.float32),
                video_path,
                avail,
                gap_start_mel,
                gap_end_mel,
                np.float32(gap_start_sec),
                np.float32(gap_end_sec),
                local_start,
                local_end,
            )


def build_loader(
    dataset_name: str,
    mode: str,
    gap_ms: int,
    local_keep_count,
    batch_size: int,
    num_workers: int,
    seed: int,
    test_subset=None,
):
    """
    Build your normal test dataloader, then wrap its dataset.

    mask_range='10' is only used to satisfy the existing dataloader.
    The mask is replaced by Experiment2BWrapper.
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

    base_loader = dl_builder.test_dataloader(mask_range="10", seed=SEED)

    wrapped_dataset = Experiment2BWrapper(
        base_dataset=base_loader.dataset,
        mode=mode,
        gap_ms=gap_ms,
        local_keep_count=local_keep_count,
        seed=SEED,
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
    """Plot PESQ/STOI versus the number of local frames retained."""
    if not dataset_results:
        return

    figure_dir = os.path.join(output_dir, dataset_name, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    labels = ["0", "2", "4", "8", "12", "16", "all"]

    for metric in ("pesq", "stoi"):
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for gap_ms in GAP_LENGTHS_MS:
            selected = [r for r in dataset_results if r["gap_ms"] == gap_ms]
            by_k = {str(r["local_keep_count"]): r for r in selected}
            values = [
                by_k[label]["summary"].get(f"{metric}_mean")
                if label in by_k else np.nan
                for label in labels
            ]
            ax.plot(labels, values, "o-", label=f"{gap_ms} ms")

        ax.set_xlabel("Local video frames retained")
        ax.set_ylabel(rf"{metric.upper()} ($\uparrow$)")
        _configure_axis(ax)
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.18),
            ncol=len(GAP_LENGTHS_MS), frameon=False,
        )
        fig.tight_layout()
        fig.savefig(
            os.path.join(figure_dir, f"{metric}_vs_local_frames.png"),
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
    gap_ms,
    sample_rate=16000,
):
    model.eval()
    mel_mean, mel_std = _get_dataset_stats(loader)

    rows = []
    pesq_values = []
    stoi_values = []

    for batch_idx, batch in enumerate(loader):
        (
            masked_spec,
            mel_spec,
            audio_length,
            text,
            mask,
            video_path,
            gap_start_mel,
            gap_end_mel,
            gap_start_sec,
            gap_end_sec,
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
                "gap_ms": int(gap_ms),
                "gap_start_mel": int(gap_start_mel[i]),
                "gap_end_mel": int(gap_end_mel[i]),
                "gap_start_sec": float(gap_start_sec[i]),
                "gap_end_sec": float(gap_end_sec[i]),
                "local_keep_count": 0,
                "model_type": "audio_only",
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
    gap_ms,
    local_keep_count,
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
            gap_start_mel,
            gap_end_mel,
            gap_start_sec,
            gap_end_sec,
            local_start,
            local_end,
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
                "gap_ms": int(gap_ms),
                "gap_start_mel": int(gap_start_mel[i]),
                "gap_end_mel": int(gap_end_mel[i]),
                "gap_start_sec": float(gap_start_sec[i]),
                "gap_end_sec": float(gap_end_sec[i]),
                "local_start_video_frame": int(local_start[i]),
                "local_end_video_frame": int(local_end[i]),
                "local_window_num_frames": int(local_end[i] - local_start[i] + 1),
                "local_keep_count": str(local_keep_count),
                "model_type": "audio_visual",
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
        default=os.path.join(project_log_dir("AV_PLC"), "ablation_local_video_reduction"),
    )

    parser.add_argument(
        "--vocoder_path",
        type=str,
        default="/home/ai/Projects/Mahsa/sources/AV_PLC/hifigan/checkpoints/model-best.pt",
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

            for K_local in LOCAL_KEEP_COUNTS:
                print(f"\nLocal keep count: {K_local}")

                if K_local == 0:
                    # K_local=0 means true audio-only baseline.
                    # We do NOT feed zero local video to AV model.
                    loader = build_loader(
                        dataset_name=dataset_name,
                        mode="a",
                        gap_ms=gap_ms,
                        local_keep_count=K_local,
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
                        gap_ms=gap_ms,
                        sample_rate=args.sample_rate,
                    )

                    model_type = "audio_only"
                    model_name = audio_model_name

                else:
                    # K_local > 0 or "all":
                    # AV model with local window downsampled/repeated.
                    loader = build_loader(
                        dataset_name=dataset_name,
                        mode="av",
                        gap_ms=gap_ms,
                        local_keep_count=K_local,
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
                        gap_ms=gap_ms,
                        local_keep_count=K_local,
                        sample_rate=args.sample_rate,
                    )

                    model_type = "audio_visual"
                    model_name = av_model_name

                result = {
                    "dataset": dataset_name,
                    "gap_ms": gap_ms,
                    "local_keep_count": str(K_local),
                    "local_context_ms_each_side": int(LOCAL_CONTEXT_SEC * 1000),
                    "model_type": model_type,
                    "model_name": model_name,
                    "summary": summary,
                }

                all_results.append(result)
                dataset_results.append(result)

                safe_k = str(K_local).replace("/", "_")
                out_base = os.path.join(
                    args.results_dir,
                    dataset_name,
                    f"gap_{gap_ms}ms",
                    f"Klocal_{safe_k}",
                )

                save_rows_csv(rows, os.path.join(out_base, "per_sample.csv"))
                save_json(result, os.path.join(out_base, "summary.json"))

                print(summary)

        save_summary_plots(dataset_results, args.results_dir, dataset_name)

    save_json(all_results, os.path.join(args.results_dir, "all_results_summary.json"))

    print(f"\nSaved all Experiment 2B results to: {args.results_dir}")


if __name__ == "__main__":
    main()

#Debug

#python experiment2b_local_video_reduction.py \
#  --datasets grid \
#  --batch_size 2 \
#  --num_workers 0 \
#  --test_subset 10

#Full Run
# python experiment2b_local_video_reduction.py \
#  --datasets grid lrs2 voxceleb2 \
#  --batch_size 8 \
#  --num_workers 4 \
#  --checkpoint_dir checkpoints \
#  --results_dir results_exp2b_local_video_reduction \
#  --dataset_root /home/ai/Projects/Mahsa/datasets \
# --test_subset 500
# --vocoder_path / home / ai / Projects / Mahsa / sources / AV_PLC / hifigan / checkpoints / model - best.pt \
