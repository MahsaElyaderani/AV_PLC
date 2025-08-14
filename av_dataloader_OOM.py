import os
from glob import glob
import random
import math
import cv2
import h5py
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T

from av_augmentation import Compose, RandomCrop, RandomErase, TimeMask, HorizontalFlip

VALID_MODES = {'a', 'v', 'av', 'motion'}
VALID_MASK_RANGES = {'20', '30', '40', '50', '60', 'rand'}


# ----------------------------- DATASET -----------------------------

class AVDataset(Dataset):
    """
    Lazy HDF5-backed Audio/Video dataset.

    Returns by mode:
      'a'      -> (masked_spec, mel_spec, text, mask)
      'motion' -> (masked_spec, motions_tensor[T,H,W?], mel_spec, text, mask)
      'v','av' -> (frames[T,1,H,W], spk_emb[...], masked_spec, mel_spec, mask)
    """

    def __init__(self, base_path_or_pattern, mode='v', mask_range='rand',
                 augment=True, chunk_pattern="_chunk*.h5"):
        self._validate_args(mode, mask_range)

        self.mode = mode
        self.augment = augment
        self.mask_range = mask_range

        # Resolve file pattern
        if chunk_pattern in base_path_or_pattern:
            pattern = base_path_or_pattern
        else:
            pattern = os.path.join(base_path_or_pattern, chunk_pattern)

        self.chunk_files = sorted(glob(pattern))
        if not self.chunk_files:
            raise FileNotFoundError(f"No HDF5 files found at pattern: {pattern}")

        # Build index map (chunk_idx, key)
        self.index_map = []
        self.chunk_sizes = []
        for chunk_idx, path in enumerate(self.chunk_files):
            with h5py.File(path, 'r') as h5f:
                keys = list(h5f.keys())
                self.index_map.extend((chunk_idx, k) for k in keys)
                self.chunk_sizes.append(len(keys))

        print(f"AVDataset initialized with {len(self.index_map)} samples from {len(self.chunk_files)} chunks.")

        # Augment + transforms
        if self.augment:
            self.video_aug = Compose([
                # RandomCrop((88, 88)),
                HorizontalFlip(0.5),
                RandomErase(0.5),
                TimeMask()
            ])
        else:
            self.video_aug = None

        self.video_transform = T.Compose([
            T.Resize((112, 112)),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),                      # -> [1,H,W] float[0,1]
            T.Normalize(mean=0.421, std=0.165),
        ])

        # Dataset-specific mel stats (fallback default)
        bp = base_path_or_pattern.lower()
        if 'grid' in bp:
            self.mel_mean, self.mel_std = -56.775, 19.707
        elif 'vox' in bp:
            self.mel_mean, self.mel_std = -52.43, 17.499
        else:
            self.mel_mean, self.mel_std = -54.60, 18.60

        # per-process cache of open H5 files
        self._h5_cache = {}

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

    # -------------- HDF5 management per worker --------------

    def _get_h5_file(self, chunk_idx):
        MAX_OPEN_FILES = 20
        if chunk_idx not in self._h5_cache:
            if len(self._h5_cache) >= MAX_OPEN_FILES:
                # LRU-ish: close an arbitrary old one
                old = next(iter(self._h5_cache))
                self._h5_cache[old].close()
                del self._h5_cache[old]
            # Each worker opens its own handle
            self._h5_cache[chunk_idx] = h5py.File(self.chunk_files[chunk_idx], 'r', swmr=True)
        return self._h5_cache[chunk_idx]

    def close_h5_files(self):
        for f in list(self._h5_cache.values()):
            try:
                f.close()
            except Exception:
                pass
        self._h5_cache.clear()

    # -------------------- item access --------------------

    def __getitem__(self, idx):
        chunk_idx = None
        video_key = None
        try:
            chunk_idx, video_key = self.index_map[idx]
            h5f = self._get_h5_file(chunk_idx)

            mel_spec = h5f[f"{video_key}/spec"][:].astype(np.float32)
            text = h5f[f"{video_key}/text"][:]  # keep as-is
            if self.mask_range == 'rand':
                mask = h5f[f"{video_key}/mask"][:].astype(np.float32)
            else:
                mask = h5f[f"{video_key}/mask_{self.mask_range}"][:].astype(np.float32)

            mel_spec = (mel_spec - self.mel_mean) / self.mel_std
            masked_spec = mel_spec * mask

            if self.mode == 'a':
                return (
                    masked_spec.astype(np.float32),
                    mel_spec.astype(np.float32),
                    text,
                    mask.astype(np.float32),
                )

            if self.mode == 'motion':
                landmarks = h5f[f"{video_key}/landmarks"][:].astype(np.float32)
                valid = ~np.all(landmarks == 0, axis=(1, 2))
                if not np.any(valid):
                    raise RuntimeError("No valid landmarks (all zero).")
                motions = np.diff(landmarks[valid], axis=0)
                padded = np.zeros_like(landmarks, dtype=np.float32)
                padded[:len(motions)] = motions
                return (
                    masked_spec.astype(np.float32),
                    torch.from_numpy(padded),  # torch for convenience downstream
                    mel_spec.astype(np.float32),
                    text,
                    mask.astype(np.float32),
                )

            if self.mode in ('v', 'av'):
                frames = h5f[f"{video_key}/frames"][:]               # [T,H,W,C]
                spk_emb = h5f[f"{video_key}/spkr_embd"][:].astype(np.float32)
                frames_t = self._process_video_frames(frames)        # torch [T,1,H,W]
                return (
                    frames_t,
                    torch.from_numpy(spk_emb),
                    masked_spec.astype(np.float32),
                    mel_spec.astype(np.float32),
                    mask.astype(np.float32),
                )

            raise NotImplementedError(f"Unsupported mode: {self.mode}")

        except Exception as e:
            file = self.chunk_files[chunk_idx] if chunk_idx is not None else "<?>"
            raise RuntimeError(
                f"Dataset error at idx={idx}, chunk={chunk_idx}, key={video_key}, file={file}: {e}"
            ) from e

    def _process_video_frames(self, frames_np):
        """
        frames_np: [T, H, W, C] (RGB). Returns torch [T, 1, H, W] float32
        """
        if frames_np.ndim != 4 or frames_np.shape[-1] not in (1, 3, 4):
            raise ValueError(f"Bad frames shape {frames_np.shape}")

        valid = np.any(frames_np != 0, axis=(1, 2, 3))
        frames_np = frames_np[valid]
        if frames_np.size == 0:
            raise RuntimeError("All-zero frames after validity filter")

        # grayscale
        if frames_np.shape[-1] == 1:
            gray = frames_np[..., 0]
        else:
            gray = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames_np], axis=0)

        # augment (expects [T,H,W] numpy)
        if self.video_aug is not None:
            gray = self.video_aug(gray)

        # per-frame torchvision pipeline -> [1,H,W] each
        processed = []
        for f in gray:
            img = Image.fromarray(f.astype(np.uint8), mode='L')
            processed.append(self.video_transform(img))  # [1,H,W] float32

        if not processed:
            raise RuntimeError("No processed frames produced")

        return torch.stack(processed, dim=0)  # [T,1,H,W]


# ----------------------------- DATALOADER -----------------------------

class AVDataloader:
    """
    Wrapper building train/val/test DataLoaders with safe worker init and mode-aware collate.
    """

    def __init__(self, dataset_name, mode, batch_size, num_workers,
                 train_subset=None, val_subset=None, test_subset=None):

        assert dataset_name in ['grid', 'voxceleb2'], f"Invalid dataset_name: {dataset_name}"
        self.mode = mode
        self.batch_size = batch_size
        self.num_workers = int(num_workers)
        self.train_subset = train_subset
        self.val_subset = val_subset
        self.test_subset = test_subset

        if dataset_name == 'grid':
            base_path = '/home/ai/Projects/Mahsa/datasets/grid/'
            self.train_files = base_path + 'grid_train_features_chunk*.h5'
            self.val_files   = base_path + 'grid_val_features_chunk*.h5'
            self.test_files  = base_path + 'grid_test_features_chunk*.h5'
        else:  # 'voxceleb2'
            base_path = '/home/ai/Projects/Mahsa/datasets/vox2_short/'
            self.train_files = base_path + 'vox2_short_dev_features_chunk*.h5'
            self.val_files   = base_path + 'vox2_short_val_features_chunk*.h5'
            self.test_files  = base_path + 'vox2_short_test_features_chunk*.h5'

    # ---------- worker utils ----------

    @staticmethod
    def _unwrap_base_dataset(ds):
        while isinstance(ds, Subset):
            ds = ds.dataset
        return ds

    @staticmethod
    def _worker_init_fn(worker_id):
        info = torch.utils.data.get_worker_info()
        base = AVDataloader._unwrap_base_dataset(info.dataset)
        # Close inherited H5 handles (important for multiprocessing)
        if hasattr(base, "close_h5_files"):
            base.close_h5_files()
        # Seed everything per worker (good for random augs)
        seed = (torch.initial_seed() % 2**32) + worker_id
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))

    # ---------- public builders ----------

    def train_dataloader(self):
        ds = AVDataset(self.train_files, self.mode, 'rand', augment=True)
        if self.train_subset is not None:
            idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0)).tolist()
            ds = Subset(ds, idx[:self.train_subset])

        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            collate_fn=self._av_collate_fn if self.mode in ('v', 'av') else None,
            worker_init_fn=self._worker_init_fn if self.num_workers > 0 else None,
            drop_last=False,
        )

    def val_dataloader(self):
        ds = AVDataset(self.val_files, self.mode, 'rand', augment=False)
        if self.val_subset is not None:
            idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0)).tolist()
            ds = Subset(ds, idx[:self.val_subset])

        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            collate_fn=self._av_collate_fn if self.mode in ('v', 'av') else None,
            worker_init_fn=self._worker_init_fn if self.num_workers > 0 else None,
            drop_last=True,
        )

    def test_dataloader(self, mask_range='rand'):
        if mask_range not in VALID_MASK_RANGES:
            raise ValueError(f"Invalid mask_range: {mask_range}")

        ds = AVDataset(self.test_files, self.mode, mask_range, augment=False)
        if self.test_subset is not None:
            idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0)).tolist()
            ds = Subset(ds, idx[:self.test_subset])

        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            collate_fn=self._av_collate_fn if self.mode in ('v', 'av') else None,
            worker_init_fn=self._worker_init_fn if self.num_workers > 0 else None,
            drop_last=True,
        )

    # ---------- collate for video modes ----------

    @staticmethod
    def _av_collate_fn(batch):
        """
        Expects items from 'v' or 'av' modes:
           (frames[T,1,H,W], spk_emb, masked_spec, mel_spec, mask)
        Produces:
           frames[B,75,1,112,112], spk_embs[B,...], masked_specs[B,F,T], mel_specs[B,F,T], masks[B,F,T]
        """
        frames, spk_embs, masked_specs, mel_specs, masks = zip(*batch)

        # normalize frame length to T=75 via trilinear up/down sample
        processed_frames = []
        for f in frames:  # f: [T,1,H,W], torch.float32
            f = f.permute(1, 0, 2, 3).unsqueeze(0)  # [1,C,T,H,W]
            _, C, T, H, W = f.shape
            if T != 75:
                f = torch.nn.functional.interpolate(
                    f, size=(75, H, W), mode='trilinear', align_corners=False
                )
            f = f.squeeze(0).permute(1, 0, 2, 3)  # [75, C, H, W]
            processed_frames.append(f)

        frames = torch.stack(processed_frames, dim=0)  # [B,75,1,112,112]
        spk_embs = torch.stack([torch.as_tensor(s, dtype=torch.float32) for s in spk_embs])
        masked_specs = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in masked_specs])
        mel_specs = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in mel_specs])
        masks = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in masks])

        return frames, spk_embs, masked_specs, mel_specs, masks

    # (optional) audio→video masking helper retained from your code
    @staticmethod
    def apply_audio_mask_to_video(videos: torch.Tensor, audio_masks: torch.Tensor, threshold: float = 0.5):
        """
        videos: [B, 75, 1, H, W]
        audio_masks: [B, F, T_audio] (e.g., T_audio=300, group of 4 -> 75)
        """
        masked_videos = []
        for audio_mask, video in zip(audio_masks, videos):
            T_v = video.shape[0]
            time_mask = audio_mask.min(dim=0).values.float().view(1, 1, -1)  # [1,1,300]
            inverted = 1.0 - time_mask
            pooled = torch.nn.functional.max_pool1d(inverted, kernel_size=4, stride=4)  # [1,1,75]
            downsampled_mask = 1.0 - pooled.view(T_v)  # [75]
            video_mask = downsampled_mask[:, None, None, None].expand_as(video)
            masked_videos.append(video * video_mask)
        return torch.stack(masked_videos)
