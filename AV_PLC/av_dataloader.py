# Portable root configuration
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import hashlib
import h5py
import json
import numpy as np
from glob import glob
import os, random, re
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from evaluations.runtime_config import dataset_patterns, SEED
from shared.av_augmentation import (
    Compose, CenterCrop, RandomCrop, RandomErase, TimeMask, TemporalJitter,
    HorizontalFlip, ModalityDropout,
)
from shared.masking import generate_ge_trace_bursty, generate_single_gap_trace, packet_count_from_audio_len
from AV_PLC.audio_frontend import AudioFrontend, AudioFrontendConfig
from AV_PLC.waveform_masking import trace_to_sample_mask

VALID_MODES = {'a', 'v', 'av', 'motion'}
VALID_MASK_RANGES = {'10', '20', '30', '40', '50', '60', '70', '80', '90', 'rand'}


def _decode_audio(ds) -> np.ndarray:
    x = ds[:]
    if x.dtype == np.int16:
        return x.astype(np.float32) / 32768.0
    return x.astype(np.float32)


def _load_stats(dataset_name: str, stats_path=None):
    if stats_path is None:
        stats_path = os.environ.get("AVPLC_MEL_STATS")
    if stats_path is None:
        stats_path = _Path(__file__).resolve().parent / "mel_stats" / f"{dataset_name}.json"
    stats_path = _Path(stats_path)
    if not stats_path.is_file():
        raise FileNotFoundError(
            f"AV_PLC Mel statistics not found: {stats_path}. Run:\n"
            f"  python -m AV_PLC.compute_audio_stats --h5 '<train_chunk_glob>' --output '{stats_path}'"
        )
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    return float(stats["mean"]), float(stats["std"]), str(stats_path)


class AVDataset(Dataset):
    def __init__(self, base_path_or_pattern, mode='v', mask_range='rand',
                 augment=True, drop_av=False, chunk_pattern="_chunk*.h5",
                 set_seed=None, online_loss_bounds=None, mask_type="gilbert", gap_ms=None,
                 mask_seed=SEED, temporal_jitter=False, jitter_p=0.5,
                 jitter_max_frames=2, phase_reconstruction=False,
                 frontend_lookahead_ms=7.5, mel_mean=None, mel_std=None):
        self._validate_args(mode, mask_range)
        self.mode = mode
        self.augment = bool(augment)
        self.drop_av = bool(drop_av)
        self.mask_range = str(mask_range)
        self.mask_type = mask_type
        self.gap_ms = gap_ms
        self.mask_seed = int(mask_seed)
        self.set_seed = set_seed
        self.online_loss_bounds = online_loss_bounds
        self.phase_reconstruction = bool(phase_reconstruction)

        pattern = (base_path_or_pattern if chunk_pattern in base_path_or_pattern
                   else os.path.join(base_path_or_pattern, chunk_pattern))
        self.chunk_files = sorted(glob(pattern), key=self.chunk_idx)
        if not self.chunk_files:
            raise FileNotFoundError(f"No HDF5 files found at pattern: {pattern}")

        self.index_map, self.chunk_sizes = [], []
        for chunk_idx, path in enumerate(self.chunk_files):
            with h5py.File(path, 'r') as h5f:
                keys = list(h5f.keys())
                self.index_map.extend((chunk_idx, k) for k in keys)
                self.chunk_sizes.append(len(keys))

        if mel_mean is None or mel_std is None:
            raise ValueError("AVDataset requires the new AV_PLC mel_mean and mel_std")
        self.mel_mean, self.mel_std = float(mel_mean), float(mel_std)
        self.frontend_lookahead_ms = float(frontend_lookahead_ms)
        self.frontend = AudioFrontend(
            AudioFrontendConfig(lookahead_ms=self.frontend_lookahead_ms),
            mel_mean=self.mel_mean, mel_std=self.mel_std,
        )

        self.jitter_transform = None
        video_transforms = []
        if temporal_jitter:
            self.jitter_transform = TemporalJitter(
                p=jitter_p, max_offset_frames=jitter_max_frames, fill_mode="edge"
            )
            video_transforms.append(self.jitter_transform)

        # Stored AV_PLC frames are aligned 96x96.  Every path outputs 88x88.
        video_transforms.append(RandomCrop((88, 88)) if self.augment else CenterCrop((88, 88)))
        if self.augment:
            video_transforms.extend([
                HorizontalFlip(0.5),
                RandomErase(0.4, replace_with_zero=True),
                TimeMask(0.4, replace_with_zero=True),
            ])
        self.video_aug = Compose(video_transforms)
        self.modality_dropout = ModalityDropout() if self.drop_av else None
        self._h5_cache = {}
        print(
            f"AVDataset initialized with {len(self.index_map)} samples; "
            f"lookahead={self.frontend_lookahead_ms:g} ms"
        )

    def __del__(self):
        try:
            self.close_h5_files()
        except Exception:
            pass

    def _validate_args(self, mode, mask_range):
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode: {mode}. Choose from {VALID_MODES}")
        if str(mask_range) not in VALID_MASK_RANGES:
            raise ValueError(f"Invalid mask range: {mask_range}. Choose from {VALID_MASK_RANGES}")

    def __len__(self):
        return len(self.index_map)

    def chunk_idx(self, s: str) -> int:
        m = re.search(r'_chunk(\d+)\.h5$', os.path.basename(s))
        return int(m.group(1)) if m else -1

    def _get_h5_file(self, chunk_idx):
        if chunk_idx not in self._h5_cache:
            if len(self._h5_cache) >= 20:
                old = next(iter(self._h5_cache))
                self._h5_cache[old].close()
                del self._h5_cache[old]
            self._h5_cache[chunk_idx] = h5py.File(self.chunk_files[chunk_idx], 'r', swmr=True)
        return self._h5_cache[chunk_idx]

    def close_h5_files(self):
        for f in list(self._h5_cache.values()):
            try:
                f.close()
            except Exception:
                pass
        self._h5_cache.clear()

    def _stable_item_seed(self, sample_id: str, condition: str) -> int:
        token = f"{self.set_seed}:{condition}:{sample_id}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(token).digest()[:4], "little")

    def _packet_trace(self, audio_len, sample_id):
        n_packets = packet_count_from_audio_len(audio_len, 160)
        if self.mask_type == "single_gap":
            if self.gap_ms is None:
                raise ValueError("gap_ms is required when mask_type=single_gap")
            return generate_single_gap_trace(
                n_packets, self.gap_ms, sample_id, seed=self.mask_seed, hop_ms=10.0
            )
        if self.mask_type != "gilbert":
            raise ValueError(f"Unsupported mask_type: {self.mask_type}")

        if self.online_loss_bounds is not None and self.mask_range == 'rand':
            if self.set_seed is None:
                loss_rate = np.random.uniform(*self.online_loss_bounds)
                return generate_ge_trace_bursty(n_packets, loss_rate)
            item_seed = self._stable_item_seed(sample_id, "val")
            state = np.random.get_state()
            try:
                np.random.seed(item_seed)
                loss_rate = np.random.uniform(*self.online_loss_bounds)
                return generate_ge_trace_bursty(n_packets, loss_rate)
            finally:
                np.random.set_state(state)

        if self.mask_range != 'rand':
            if self.set_seed is not None:
                item_seed = self._stable_item_seed(sample_id, str(self.mask_range))
                state = np.random.get_state()
                try:
                    np.random.seed(item_seed)
                    return generate_ge_trace_bursty(n_packets, float(self.mask_range) / 100.0)
                finally:
                    np.random.set_state(state)
            return generate_ge_trace_bursty(n_packets, float(self.mask_range) / 100.0)

        # A random training mask should never need HDF5-stored masks.
        loss_rate = np.random.uniform(0.3, 0.9)
        return generate_ge_trace_bursty(n_packets, loss_rate)

    def _audio_features(self, h5f, video_key, sample_id):
        audio_key = f"{video_key}/audio"
        if audio_key not in h5f:
            raise KeyError(
                f"Missing {audio_key}. Rebuild features with dataset/features/save_features.py "
                "before running waveform-domain AV_PLC."
            )
        audio = _decode_audio(h5f[audio_key])
        audio_len = int(h5f.attrs.get(f"{video_key}/audio_len", len(audio)))
        audio_len = max(0, min(audio_len, len(audio)))

        trace = self._packet_trace(audio_len, sample_id)
        sample_mask = trace_to_sample_mask(trace, len(audio), audio_len, packet_samples=160)
        masked_audio = audio * sample_mask

        clean_t = torch.from_numpy(audio)
        masked_t = torch.from_numpy(masked_audio.astype(np.float32, copy=False))
        sample_mask_t = torch.from_numpy(sample_mask)
        with torch.no_grad():
            clean_stft = self.frontend.stft(clean_t)
            clean_mel = self.frontend.normalize(self.frontend.magnitude_to_logmel(clean_stft.abs()))
            masked_mel = self.frontend.normalized_logmel(masked_t)
            frame_valid, hard_keep, soft_keep = self.frontend.frame_masks(sample_mask_t, audio_len)

        # Padding is invalid, not packet loss.  Neutralize it for the neural input;
        # the trainer separately excludes it from losses/metrics using frame_valid.
        clean_mel[:, ~frame_valid] = 0.0
        masked_mel[:, ~frame_valid] = 0.0
        mel_keep = torch.where(frame_valid, hard_keep, torch.ones_like(hard_keep))
        mel_mask = mel_keep.to(torch.float32)[None, :].repeat(clean_mel.shape[0], 1)

        return {
            "audio": audio.astype(np.float32, copy=False),
            "audio_len": audio_len,
            "sample_mask": sample_mask.astype(np.float32, copy=False),
            "spec": clean_mel.cpu().numpy().astype(np.float32),
            "masked_spec": masked_mel.cpu().numpy().astype(np.float32),
            "mask": mel_mask.cpu().numpy().astype(np.float32),
            "frame_valid": frame_valid.cpu().numpy().astype(np.bool_),
            "soft_keep": soft_keep.cpu().numpy().astype(np.float32),
            "stft_magnitude": clean_stft.abs().cpu().numpy().astype(np.float32),
            "phase": torch.angle(clean_stft).cpu().numpy().astype(np.float32),
        }

    def __getitem__(self, idx):
        chunk_idx = video_key = None
        try:
            chunk_idx, video_key = self.index_map[idx]
            h5f = self._get_h5_file(chunk_idx)
            video_path = h5f.attrs.get(f"{video_key}/video_path", None)
            text = h5f[f"{video_key}/text"][:]
            sample_id = f"{os.path.basename(self.chunk_files[chunk_idx])}:{video_key}"
            a = self._audio_features(h5f, video_key, sample_id)
            spec, masked_spec, mask = a["spec"], a["masked_spec"], a["mask"]
            stft_magnitude, phase = a["stft_magnitude"], a["phase"]
            aux = (a["audio"], a["sample_mask"], a["frame_valid"], a["soft_keep"])

            if self.mode == 'a':
                if self.phase_reconstruction:
                    return (masked_spec, spec, stft_magnitude, phase,
                            a["audio_len"], text, mask, video_path, *aux)
                return masked_spec, spec, a["audio_len"], text, mask, video_path, *aux

            if self.mode == 'motion':
                landmarks = h5f[f"{video_key}/landmarks"][:].astype(np.float32)
                valid = ~np.all(landmarks == 0, axis=(1, 2))
                if not np.any(valid):
                    raise RuntimeError("No valid landmarks (all zero).")
                motions = np.diff(landmarks[valid], axis=0)
                padded = np.zeros_like(landmarks, dtype=np.float32)
                padded[:len(motions)] = motions
                return masked_spec, padded, spec, text, mask, video_path, *aux

            spk_emb = h5f[f"{video_key}/spkr_embd"][:].astype(np.float32)
            frames_u8 = h5f[f"{video_key}/frames"][:]
            if frames_u8.shape[1:3] != (96, 96):
                raise ValueError(
                    f"AV_PLC expects aligned 96x96 stored frames, got {frames_u8.shape}. "
                    "Rebuild features with the new save_features.py."
                )
            invalid_frame = np.all(frames_u8 == 0, axis=tuple(range(1, frames_u8.ndim)))
            frames = np.squeeze(frames_u8.astype(np.float32) / 255.0)
            num_video_frames = frames.shape[0]
            frames = self._aug_video_frames(frames)
            frames = (frames - 0.421) / 0.165
            # Keep true missing/padded frames neutral after normalization.
            if invalid_frame.shape[0] == frames.shape[0]:
                frames[invalid_frame] = 0.0

            video_aligned_spec = self._video_aligned_target(spec, num_video_frames)
            video_aligned_magnitude = self._video_aligned_target(stft_magnitude, num_video_frames)
            video_aligned_phase = self._video_aligned_target(phase, num_video_frames)

            if self.mode == 'v':
                if self.phase_reconstruction:
                    return (frames, spk_emb, spec, video_aligned_spec,
                            stft_magnitude, video_aligned_magnitude, phase, video_aligned_phase,
                            a["audio_len"], text, mask, video_path, *aux)
                return frames, spk_emb, spec, video_aligned_spec, a["audio_len"], text, mask, video_path, *aux

            if self.mode == 'av':
                frames, masked_spec, audio_present, video_present = self._drop_av_modality(
                    video_path, frames, masked_spec
                )
                avail = np.array([audio_present, video_present], dtype=np.bool_)
                if self.phase_reconstruction:
                    return (frames, spk_emb, masked_spec, spec, video_aligned_spec,
                            stft_magnitude, video_aligned_magnitude, phase, video_aligned_phase,
                            a["audio_len"], text, mask, video_path, avail, *aux)
                return (frames, spk_emb, masked_spec, spec, video_aligned_spec,
                        a["audio_len"], text, mask, video_path, avail, *aux)
            raise NotImplementedError(f"Unsupported mode: {self.mode}")
        except Exception as e:
            file = self.chunk_files[chunk_idx] if chunk_idx is not None else "<?>"
            raise RuntimeError(
                f"Dataset error at idx={idx}, chunk={chunk_idx}, key={video_key}, file={file}: {e}"
            ) from e

    def _aug_video_frames(self, frames):
        return self.video_aug(frames) if self.video_aug is not None else frames

    def _video_aligned_target(self, target, num_video_frames):
        if self.jitter_transform is None:
            return target
        offset = self.jitter_transform.last_offset
        if offset == 0:
            return target
        per_video = target.shape[-1] / float(num_video_frames)
        shift = int(round(offset * per_video))
        out = target.copy()
        T = target.shape[-1]
        if shift > 0:
            out[..., shift:] = target[..., :T-shift]
            out[..., :shift] = target[..., :1]
        else:
            s = -shift
            out[..., :T-s] = target[..., s:]
            out[..., T-s:] = target[..., -1:]
        return out

    def _drop_av_modality(self, video_path, frames, masked_spec):
        audio_present, video_present = True, True
        if self.modality_dropout is not None:
            mode = self.modality_dropout.sample_mode()
            if mode == 'audio_only':
                video_present = False
            elif mode == 'video_only':
                audio_present = False
        return frames, masked_spec, audio_present, video_present


class AVDataloader:
    def __init__(self, dataset_name, mode, batch_size, num_workers, video_aug, dropout_modality,
                 train_subset=None, val_subset=None, test_subset=None,
                 temporal_jitter=False, jitter_p=0.5, jitter_max_frames=2,
                 phase_reconstruction=False, frontend_lookahead_ms=7.5,
                 mel_stats_path=None):
        assert dataset_name in ['grid', 'lrs2', 'voxceleb2'], f"Invalid dataset_name: {dataset_name}"
        self.dataset_name = dataset_name
        self.mode = mode
        self.batch_size = batch_size
        self.num_workers = int(num_workers)
        self.video_aug = bool(video_aug)
        self.dropout_modality = bool(dropout_modality)
        self.temporal_jitter = bool(temporal_jitter)
        self.jitter_p = float(jitter_p)
        self.jitter_max_frames = int(jitter_max_frames)
        self.phase_reconstruction = bool(phase_reconstruction)
        self.frontend_lookahead_ms = float(frontend_lookahead_ms)
        self.mel_mean, self.mel_std, self.mel_stats_path = _load_stats(dataset_name, mel_stats_path)
        self.train_subset, self.val_subset, self.test_subset = train_subset, val_subset, test_subset
        paths = dataset_patterns(dataset_name)
        self.train_files, self.val_files, self.test_files = paths["train"], paths["val"], paths["test"]

    @staticmethod
    def _unwrap_base_dataset(ds):
        while isinstance(ds, Subset):
            ds = ds.dataset
        return ds

    @staticmethod
    def _worker_init_fn(worker_id):
        info = torch.utils.data.get_worker_info()
        base = AVDataloader._unwrap_base_dataset(info.dataset)
        if hasattr(base, "close_h5_files"):
            base.close_h5_files()
        seed = torch.initial_seed() % (2 ** 32)
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        try:
            import cv2
            cv2.setNumThreads(0)
        except Exception:
            pass
        torch.set_num_threads(1)

    def _build_loader(self, files_glob, augment, drop_av, subset=None, drop_last=False,
                      shuffle=False, mask_range='rand', online_loss_bounds=None, set_seed=None,
                      mask_type='gilbert', gap_ms=None, mask_seed=SEED):
        ds = AVDataset(
            files_glob, self.mode, mask_range, augment=augment, drop_av=drop_av,
            online_loss_bounds=online_loss_bounds, set_seed=set_seed,
            mask_type=mask_type, gap_ms=gap_ms, mask_seed=mask_seed,
            temporal_jitter=self.temporal_jitter, jitter_p=self.jitter_p,
            jitter_max_frames=self.jitter_max_frames,
            phase_reconstruction=self.phase_reconstruction,
            frontend_lookahead_ms=self.frontend_lookahead_ms,
            mel_mean=self.mel_mean, mel_std=self.mel_std,
        )
        if subset is not None:
            idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(SEED)).tolist()
            ds = Subset(ds, idx[:subset])
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers,
            pin_memory=True, persistent_workers=False,
            worker_init_fn=self._worker_init_fn if self.num_workers > 0 else None,
            drop_last=drop_last, generator=torch.Generator().manual_seed(SEED),
        )

    def train_dataloader(self):
        return self._build_loader(
            self.train_files, augment=self.video_aug, drop_av=self.dropout_modality,
            subset=self.train_subset, shuffle=True, mask_range='rand',
            online_loss_bounds=(0.3, 0.9), set_seed=None,
        )

    def val_dataloader(self):
        return self._build_loader(
            self.val_files, augment=False, drop_av=False,
            subset=self.val_subset, shuffle=False, mask_range='rand',
            online_loss_bounds=(0.3, 0.9), set_seed=SEED,
        )

    def test_dataloader(self, mask_range='60', seed=SEED, mask_type="gilbert", gap_ms=None):
        if mask_type == "gilbert":
            if str(mask_range) not in VALID_MASK_RANGES or str(mask_range) == 'rand':
                raise ValueError(f"Invalid Gilbert-Elliott mask_range: {mask_range}")
            return self._build_loader(
                self.test_files, augment=False, drop_av=False, subset=self.test_subset,
                shuffle=False, mask_range=str(mask_range), set_seed=seed, mask_type="gilbert",
            )
        if mask_type == "single_gap":
            if gap_ms is None:
                raise ValueError("gap_ms is required for single_gap testing")
            return self._build_loader(
                self.test_files, augment=False, drop_av=False, subset=self.test_subset,
                shuffle=False, mask_range='rand', set_seed=None, mask_type="single_gap",
                gap_ms=gap_ms, mask_seed=seed,
            )
        raise ValueError(f"Unsupported mask_type: {mask_type}")
