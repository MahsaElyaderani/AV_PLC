
# Portable root configuration
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from evaluations.runtime_config import dataset_patterns, SEED

import h5py
import hashlib
import numpy as np
from glob import glob
import os, random, re
from typing import Callable

import torch
from torch.utils.data import Dataset, DataLoader, Subset

from shared.masking import generate_ge_mask, generate_ge_mask_bursty, generate_single_gap_mask
from shared.av_augmentation import (
    Compose, RandomCrop, RandomErase, TimeMask, TemporalJitter,
    HorizontalFlip, ModalityDropout,
)

# ----------------------------- DATASET -----------------------------

VALID_MODES = {'a', 'v', 'av', 'motion'}
VALID_MASK_RANGES = {'10', '20', '30', '40', '50', '60', '70', '80', '90', 'rand'}

class AVDataset(Dataset):

    def __init__(self, base_path_or_pattern, mode='v', mask_range='rand',
                 augment=True, drop_av=False, chunk_pattern="_chunk*.h5",
                 set_seed=None, online_loss_bounds=None, mask_type="gilbert", gap_ms=None, mask_seed=SEED,
                 temporal_jitter=False, jitter_p=0.5, jitter_max_frames=2,
                 phase_reconstruction=False):

        self._validate_args(mode, mask_range)
        self.mode = mode
        self.augment = augment
        self.drop_av = drop_av
        self.mask_range = mask_range
        self.mask_type = mask_type
        self.gap_ms = gap_ms
        self.mask_seed = int(mask_seed)
        self.set_seed = set_seed
        self.online_loss_bounds = online_loss_bounds
        self.phase_reconstruction = bool(phase_reconstruction)

        # Resolve file pattern
        pattern = (base_path_or_pattern if chunk_pattern in base_path_or_pattern
                   else os.path.join(base_path_or_pattern, chunk_pattern))
        #self.chunk_files = sorted(glob(pattern))
        self.chunk_files = sorted(glob(pattern), key=self.chunk_idx)
        if not self.chunk_files:
            raise FileNotFoundError(f"No HDF5 files found at pattern: {pattern}")

        # Build index map (chunk_idx, key)
        self.index_map, self.chunk_sizes = [], []
        for chunk_idx, path in enumerate(self.chunk_files):
            with h5py.File(path, 'r') as h5f:
                keys = list(h5f.keys())
                self.index_map.extend((chunk_idx, k) for k in keys)
                self.chunk_sizes.append(len(keys))

        # mel stats by dataset name
        bp = base_path_or_pattern.lower()
        if 'grid' in bp:
            self.mel_mean, self.mel_std = -56.775, 19.707
        elif 'vox' in bp:
            self.mel_mean, self.mel_std = -52.43, 17.499
        else:
            self.mel_mean, self.mel_std = -54.60, 18.60
        self.jitter_transform = None
        video_transforms = []
        if temporal_jitter:
            self.jitter_transform = TemporalJitter(
                p=jitter_p,
                max_offset_frames=jitter_max_frames,
                fill_mode="edge",
            )
            video_transforms.append(self.jitter_transform)

        if self.augment:
            video_transforms.extend([
                HorizontalFlip(0.5),
                RandomErase(0.4, replace_with_zero=True),
                TimeMask(0.4, replace_with_zero=True),
            ])

        self.video_aug = Compose(video_transforms) if video_transforms else None
        if self.drop_av:
            self.modality_dropout = ModalityDropout()
        else:
            self.modality_dropout = None

        # per-process cache of open H5 files
        self._h5_cache = {}
        print(f"AVDataset initialized with {len(self.index_map)} samples from {len(self.chunk_files)} chunks.")

    def __del__(self):
        try:
            self.close_h5_files()
        except Exception:
            pass

    def _validate_args(self, mode, mask_range):
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode: {mode}. Choose from {VALID_MODES}")
        if mask_range not in VALID_MASK_RANGES:
            raise ValueError(f"Invalid mask range: {mask_range}. Choose from {VALID_MASK_RANGES}")

    def __len__(self):
        return len(self.index_map)

    # ---------- HDF5 per-worker management ----------

    def chunk_idx(self, s: str) -> int:
        m = re.search(r'_chunk(\d+)\.h5$', os.path.basename(s))
        return int(m.group(1)) if m else -1

    def _get_h5_file(self, chunk_idx):
        MAX_OPEN_FILES = 20
        if chunk_idx not in self._h5_cache:
            if len(self._h5_cache) >= MAX_OPEN_FILES:
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

    def __getitem__(self, idx):
        chunk_idx = None
        video_key = None
        try:
            chunk_idx, video_key = self.index_map[idx]
            h5f = self._get_h5_file(chunk_idx)

            video_path = h5f.attrs.get(f"{video_key}/video_path", None)
            audio_length = 0 #h5f.attrs.get(f"{video_key}/audio_len", None)
            text = h5f[f"{video_key}/text"][:]
            mel_spec = h5f[f"{video_key}/spec"][:].astype(np.float32)
            phase = None
            if self.phase_reconstruction:
                phase_key = f"{video_key}/phase"
                if phase_key not in h5f:
                    raise KeyError(
                        f"Missing {phase_key}. Run AV_PLC/precompute_phase.py first."
                    )
                phase = h5f[phase_key][:].astype(np.float32)
                if phase.ndim != 2 or phase.shape[0] != 257:
                    raise ValueError(
                        f"Expected phase [257,T], got {phase.shape} for {video_key}"
                    )
                if phase.shape[-1] != mel_spec.shape[-1]:
                    raise ValueError(
                        f"Phase/Mel time mismatch for {video_key}: "
                        f"{phase.shape[-1]} vs {mel_spec.shape[-1]}"
                    )

            sample_id = f"{os.path.basename(self.chunk_files[chunk_idx])}:{video_key}"
            if self.mask_type == "single_gap":
                if self.gap_ms is None:
                    raise ValueError("gap_ms is required when mask_type=single_gap")
                mask = generate_single_gap_mask(
                    mel_spec.shape, self.gap_ms, sample_id, seed=self.mask_seed
                )
            elif self.mask_type != "gilbert":
                raise ValueError(f"Unsupported mask_type: {self.mask_type}")
            elif self.online_loss_bounds is not None and self.mask_range == 'rand':
                if self.set_seed is None:
                    # Training: draw a new loss rate and mask on every access.
                    loss_rate = np.random.uniform(*self.online_loss_bounds)
                    mask = generate_ge_mask_bursty(mel_spec.shape, loss_rate=loss_rate)
                else:
                    # Validation: stable mask for this sample without changing global RNG state.
                    item_seed = self._stable_item_seed(sample_id, "val")
                    rng_state = np.random.get_state()
                    try:
                        np.random.seed(item_seed)
                        loss_rate = np.random.uniform(*self.online_loss_bounds)
                        mask = generate_ge_mask_bursty(mel_spec.shape, loss_rate=loss_rate)
                    finally:
                        np.random.set_state(rng_state)
            elif self.set_seed is not None and self.mask_range != 'rand':
                # Test: stable mask for this sample and requested loss rate.
                item_seed = self._stable_item_seed(sample_id, str(self.mask_range))
                rng_state = np.random.get_state()
                try:
                    np.random.seed(item_seed)
                    mask = generate_ge_mask_bursty(
                        mel_spec.shape, loss_rate=float(self.mask_range) / 100
                    )
                finally:
                    np.random.set_state(rng_state)
            else:
                # ---- mask from HDF5 (test, or fixed %) ----
                if self.mask_range == 'rand':
                    mask = h5f[f"{video_key}/mask"][:].astype(np.float32)
                else:
                    mask = h5f[f"{video_key}/mask_{self.mask_range}"][:].astype(np.float32)

            # audio normalization
            mel_spec = (mel_spec - self.mel_mean) / self.mel_std
            masked_spec = mel_spec * mask

            if self.mode == 'a':
                if self.phase_reconstruction:
                    return masked_spec, mel_spec, phase, audio_length, text, mask, video_path
                return masked_spec, mel_spec, audio_length, text, mask, video_path

            if self.mode == 'motion':
                text = h5f[f"{video_key}/text"][:]
                landmarks = h5f[f"{video_key}/landmarks"][:].astype(np.float32)
                valid = ~np.all(landmarks == 0, axis=(1, 2))
                if not np.any(valid):
                    raise RuntimeError("No valid landmarks (all zero).")
                motions = np.diff(landmarks[valid], axis=0)
                padded = np.zeros_like(landmarks, dtype=np.float32)
                padded[:len(motions)] = motions
                return masked_spec, padded, mel_spec, text, mask, video_path

            if self.mode == 'v':
                spk_emb = h5f[f"{video_key}/spkr_embd"][:].astype(np.float32)
                frames = h5f[f"{video_key}/frames"][:]          # uint8 [T,H,W,C]
                frames = frames.astype(np.float32) / 255.  # normalize to [0,1] float32
                frames = (frames - 0.421) / 0.165
                frames = np.squeeze(frames)
                num_video_frames = frames.shape[0]
                frames = self._aug_video_frames(frames)
                video_aligned_spec = self._video_aligned_target(mel_spec, num_video_frames,)
                if self.phase_reconstruction:
                    video_aligned_phase = self._video_aligned_target(phase, num_video_frames,)
                    return (frames, spk_emb, mel_spec, video_aligned_spec, phase,
                            video_aligned_phase, audio_length, text, mask, video_path)
                return frames, spk_emb, mel_spec, video_aligned_spec, audio_length,text, mask, video_path

            if self.mode == 'av':
                spk_emb = h5f[f"{video_key}/spkr_embd"][:]  # .astype(np.float32)
                frames = h5f[f"{video_key}/frames"][:]          # uint8 [T,H,W,C]
                frames = frames.astype(np.float32) / 255.  # normalize to [0,1] float32
                frames = (frames - 0.421)/0.165
                frames = np.squeeze(frames)
                num_video_frames = frames.shape[0]
                frames = self._aug_video_frames(frames)
                video_aligned_spec = self._video_aligned_target(mel_spec, num_video_frames, )
                video_aligned_phase = (
                    self._video_aligned_target(phase, num_video_frames,)
                    if self.phase_reconstruction else None
                )

                frames, masked_spec, audio_present, video_present = self._drop_av_modality(
                    video_path, frames, masked_spec)
                avail = np.array([audio_present, video_present], dtype=np.bool_)
                #return frames, spk_emb, masked_spec, mel_spec, audio_length, text, mask, video_path, avail
                if self.phase_reconstruction:
                    return (frames, spk_emb, masked_spec, mel_spec, video_aligned_spec,
                            phase, video_aligned_phase, audio_length, text, mask, video_path, avail)
                return frames, spk_emb, masked_spec, mel_spec, video_aligned_spec, audio_length, text, mask, video_path, avail
            raise NotImplementedError(f"Unsupported mode: {self.mode}")

        except Exception as e:
            file = self.chunk_files[chunk_idx] if chunk_idx is not None else "<?>"
            raise RuntimeError(
                f"Dataset error at idx={idx}, chunk={chunk_idx}, key={video_key}, file={file}: {e}"
            ) from e

    def _aug_video_frames(self, frames):
        # augment (expects [T,H,W] numpy)
        if self.video_aug is not None:
            frames = self.video_aug(frames)
        return frames

    def _video_aligned_target(self, mel_spec, num_video_frames):
        if self.jitter_transform is None:
            return mel_spec

        offset = self.jitter_transform.last_offset
        if offset == 0:
            return mel_spec

        mel_per_video_frame = mel_spec.shape[-1] / float(num_video_frames)
        mel_shift = int(round(offset * mel_per_video_frame))
        out = mel_spec.copy()
        T = mel_spec.shape[-1]
        if mel_shift > 0:
            out[..., mel_shift:] = mel_spec[..., :T - mel_shift]
            out[..., :mel_shift] = mel_spec[..., :1]
        else:
            s = -mel_shift
            out[..., :T - s] = mel_spec[..., s:]
            out[..., T - s:] = mel_spec[..., -1:]
        return out

    def _drop_av_modality(self, video_path, frames, masked_spec):
        # audio_present, video_present
        audio_present, video_present = True, True

        if self.modality_dropout is not None:
            mode = self.modality_dropout.sample_mode()
            if mode == 'audio_only':
                video_present = False
            elif mode == 'video_only':
                audio_present = False

        # Keep tensors as-is; model uses flags to skip branches.
        return frames, masked_spec, audio_present, video_present

# ----------------------------- DATALOADER  -----------------------------

class AVDataloader:
    """
    Building train/val/test DataLoaders with safe worker init and mode-aware collate.
    """

    def __init__(self, dataset_name, mode, batch_size, num_workers, video_aug, dropout_modality,
                 train_subset=None, val_subset=None, test_subset=None,
                 temporal_jitter=False, jitter_p=0.5, jitter_max_frames=2,
                 phase_reconstruction=False):

        assert dataset_name in ['grid', 'lrs2', 'voxceleb2'], f"Invalid dataset_name: {dataset_name}"
        self.mode = mode
        self.batch_size = batch_size
        self.num_workers = int(num_workers)

        self.video_aug = video_aug
        self.dropout_modality = dropout_modality
        self.temporal_jitter = bool(temporal_jitter)
        self.jitter_p = float(jitter_p)
        self.jitter_max_frames = int(jitter_max_frames)
        self.phase_reconstruction = bool(phase_reconstruction)

        self.train_subset = train_subset
        self.val_subset = val_subset
        self.test_subset = test_subset

        paths = dataset_patterns(dataset_name)
        self.train_files = paths["train"]
        self.val_files = paths["val"]
        self.test_files = paths["test"]


    # ---------- helpers ----------

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
            temporal_jitter=self.temporal_jitter,
            jitter_p=self.jitter_p,
            jitter_max_frames=self.jitter_max_frames,
            phase_reconstruction=self.phase_reconstruction,
        )
        if subset is not None:
            idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(SEED)).tolist()
            ds = Subset(ds, idx[:subset])

        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False, #(self.num_workers > 0),
            #collate_fn=collate,                     #  heavy lifting happens here
            worker_init_fn=self._worker_init_fn if self.num_workers > 0 else None,
            drop_last=drop_last,
            generator=torch.Generator().manual_seed(SEED),
        )

    # ---------- public builders ----------

    def train_dataloader(self):
        return self._build_loader(self.train_files, augment=self.video_aug, drop_av=self.dropout_modality,
                                  subset=self.train_subset, drop_last=False, shuffle=True,
                                  mask_range='rand', online_loss_bounds=(0.3, 0.9), set_seed=None)

    def val_dataloader(self):
        return self._build_loader(self.val_files, augment=self.video_aug, drop_av=False,
                                  subset=self.val_subset, drop_last=False, shuffle=False,
                                  mask_range='rand', online_loss_bounds=(0.3, 0.9), set_seed=SEED)

    def test_dataloader(self, mask_range='60', seed=SEED, mask_type="gilbert", gap_ms=None):
        if mask_type == "gilbert":
            if str(mask_range) not in VALID_MASK_RANGES or str(mask_range) == 'rand':
                raise ValueError(f"Invalid Gilbert-Elliott mask_range: {mask_range}")
            return self._build_loader(
                self.test_files, augment=self.video_aug, drop_av=False,
                subset=self.test_subset, drop_last=False, shuffle=False,
                mask_range=str(mask_range), set_seed=seed, mask_type="gilbert",
            )
        if mask_type == "single_gap":
            if gap_ms is None:
                raise ValueError("gap_ms is required for single_gap testing")
            return self._build_loader(
                self.test_files, augment=self.video_aug, drop_av=False,
                subset=self.test_subset, drop_last=False, shuffle=False,
                mask_range='rand', set_seed=None, mask_type="single_gap",
                gap_ms=gap_ms, mask_seed=seed,
            )
        raise ValueError(f"Unsupported mask_type: {mask_type}")
