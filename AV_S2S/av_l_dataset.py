"""Eager conversion to Torch means converting to
 Torch Tensor in __getitem__ that is
 not good for large data.
 Lazy conversion returns numpy then batch and convert → much safer."""

import os
import hashlib
import h5py
import re
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from shared.masking import (generate_ge_trace_bursty, generate_single_gap_trace,
                            trace_to_spec_mask)
from evaluations.runtime_config import SEED
from shared.av_augmentation import Compose, RandomCrop, RandomErase, TimeMask, HorizontalFlip, ModalityDropout

VALID_MODES = {'a', 'av'}
VALID_MASK_RANGES = {'10','20','30','40','50','60','70','80','90','rand'}

class AVDataset(Dataset):
    def __init__(self, base_path, mode='av', augment=False, online_loss_bounds=None,
                 set_seed=None, mask_range='rand', chunk_pattern="_chunk*.h5",
                 mask_type="gilbert", gap_ms=None, mask_seed=SEED):
        self._validate_args(mode, mask_range)

        self.set_seed = set_seed
        self.online_loss_bounds = online_loss_bounds
        self.mode = mode
        self.augment = augment
        self.mask_range = mask_range
        self.mask_type = mask_type
        self.gap_ms = gap_ms
        self.mask_seed = int(mask_seed)

        self.chunk_files = sorted(glob(base_path if chunk_pattern in base_path else os.path.join(base_path, chunk_pattern)), key=self.chunk_idx)
        if not self.chunk_files:
            raise FileNotFoundError(f"No HDF5 files found at pattern: {base_path}{chunk_pattern}")

        self.index_map = []
        self.chunk_sizes = []
        for chunk_idx, path in enumerate(self.chunk_files):
            with h5py.File(path, 'r') as h5f:
                keys = list(h5f.keys())
                self.index_map.extend((chunk_idx, k) for k in keys)
                self.chunk_sizes.append(len(keys))

        print(f"AVDataset initialized with {len(self.index_map)} samples from {len(self.chunk_files)} chunks.")


        if 'grid' in base_path:
            self.mel_mean, self.mel_std = -56.775, 19.707
        elif 'vox' in base_path:
            self.mel_mean, self.mel_std = -52.43, 17.499
        else:
            self.mel_mean, self.mel_std = -54.60, 18.60

        if self.augment:
            self.video_aug = Compose([
                # RandomCrop((88, 88)),
                HorizontalFlip(0.5),
                RandomErase(0.5),
                TimeMask()
            ])
        else:
            self.video_aug = None

        self._h5_cache = {}

    def chunk_idx(self, s: str) -> int:
        m = re.search(r'_chunk(\d+)\.h5$', os.path.basename(s))
        return int(m.group(1)) if m else -1

    def _validate_args(self, mode, mask_range):
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode: {mode}. Choose from {VALID_MODES}")
        if mask_range not in VALID_MASK_RANGES:
            raise ValueError(f"Invalid mask range: {mask_range}. Choose from {VALID_MASK_RANGES}")

    def __len__(self):
        return len(self.index_map)

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
        for f in self._h5_cache.values():
            f.close()
        self._h5_cache.clear()

    def _aug_video_frames(self, frames):
        # augment (expects [T,H,W] numpy)
        if self.video_aug is not None:
            frames = self.video_aug(frames)
        return frames


    def _stable_item_seed(self, sample_id: str, condition: str) -> int:
        token = f"{self.set_seed}:{condition}:{sample_id}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(token).digest()[:4], "little")

    def __getitem__(self, idx):
        chunk_idx, video_key = self.index_map[idx]
        h5f = self._get_h5_file(chunk_idx)

        video_path = h5f.attrs.get(f"{video_key}/video_path", None)
        mel_spec = h5f[f"{video_key}/spec"][:]
        audio_length = int(h5f.attrs.get(f"{video_key}/audio_len", mel_spec.shape[-1] * 160))
        valid_t = min(mel_spec.shape[-1], (audio_length + 159) // 160)
        text = h5f[f"{video_key}/text"][:]

        sample_id = f"{os.path.basename(self.chunk_files[chunk_idx])}:{video_key}"
        if self.mask_type == "single_gap":
            if self.gap_ms is None:
                raise ValueError("gap_ms is required when mask_type=single_gap")
            trace = generate_single_gap_trace(
                valid_t, self.gap_ms, sample_id, seed=self.mask_seed, hop_ms=10.0
            )
            mask = trace_to_spec_mask(trace, mel_spec.shape)
        elif self.mask_type != "gilbert":
            raise ValueError(f"Unsupported mask_type: {self.mask_type}")
        elif self.online_loss_bounds is not None and self.mask_range == 'rand':
            if self.set_seed is None:
                # Training: draw a new loss rate and mask on every access.
                loss_rate = np.random.uniform(*self.online_loss_bounds)
                mask = trace_to_spec_mask(generate_ge_trace_bursty(valid_t, loss_rate), mel_spec.shape)
            else:
                # Validation: stable mask for this sample without changing global RNG state.
                item_seed = self._stable_item_seed(sample_id, "val")
                rng_state = np.random.get_state()
                try:
                    np.random.seed(item_seed)
                    loss_rate = np.random.uniform(*self.online_loss_bounds)
                    mask = trace_to_spec_mask(generate_ge_trace_bursty(valid_t, loss_rate), mel_spec.shape)
                finally:
                    np.random.set_state(rng_state)
        elif self.set_seed is not None and self.mask_range != 'rand':
            # Test: stable mask for this sample and requested loss rate.
            item_seed = self._stable_item_seed(sample_id, str(self.mask_range))
            rng_state = np.random.get_state()
            try:
                np.random.seed(item_seed)
                mask = trace_to_spec_mask(generate_ge_trace_bursty(valid_t, float(self.mask_range) / 100), mel_spec.shape)
            finally:
                np.random.set_state(rng_state)
        elif self.mask_range == 'rand':
            mask = h5f[f"{video_key}/mask"][:]
        else:
            mask = h5f[f"{video_key}/mask_{self.mask_range}"][:]

        # Padding is invalid audio, never packet loss.
        mask = np.asarray(mask, dtype=np.float32)
        mask[:, valid_t:] = 1.0

        mel_spec = (mel_spec - self.mel_mean) / self.mel_std
        masked_spec = mel_spec * mask

        if self.mode == 'a':
            return masked_spec, mel_spec, text, mask, video_path

        elif self.mode == 'av':
            landmarks = h5f[f"{video_key}/landmarks"][:]
            t, f, c = landmarks.shape
            valid = ~np.all(landmarks == 0, axis=(1, 2))
            if not np.any(valid):
                print(f"{video_key}: no valid landmarks; returning zeros")
                motions = np.zeros((75, f * c), dtype=np.float32)
            else:
                motions = np.diff(landmarks[valid], axis=0)
                t, f, c = motions.shape
                if t == 0:
                    print(f"{video_key}: no valid motion; returning zeros")
                    motions = np.zeros((75, f * c), dtype=np.float32)
                else:
                    motions = motions.reshape((t, f * c))
            return masked_spec, motions, mel_spec, text, mask, video_path

        raise NotImplementedError(f"Unsupported mode: {self.mode}")


from tqdm import tqdm
import  multiprocessing

def process_single_chunk_motions(chunk_file):

    #print(f"Processing: {chunk_file}")
    with h5py.File(chunk_file, 'r+') as h5f:
        video_keys = list(h5f.keys())
        for video_key in tqdm(video_keys, desc=f"{os.path.basename(chunk_file)}"):
            video_path = h5f.attrs.get(f"{video_key}/video_path", None)
            if video_path is not None:
                landmarks = h5f[f"{video_key}/landmarks"][:]
                valid = ~np.all(landmarks == 0, axis=(1, 2))
                motions = np.diff(landmarks[valid], axis=0)
                t, f, c = motions.shape
                if t <= 70:
                    print("Corrupted:", video_path)
                    del h5f[f"{video_key}"]
            else:
                print(f"Warning: No video path found for {video_key}")
    #print(f"Completed: {chunk_file}")


def update_h5(base_path, chunk_pattern="_chunk*.h5"):
    chunk_files = sorted(glob(f"{base_path}{chunk_pattern}"))

    with multiprocessing.Pool(processes=min(16, len(chunk_files))) as pool:
        pool.map(process_single_chunk_motions, chunk_files)

if __name__ == "__main__":


    base_path = '/home/ai/Projects/Mahsa/datasets/vox2_short/'
    path = base_path + 'vox2_short_test_features_chunk*.h5'

    base_path = '/home/ai/Projects/Mahsa/datasets/grid/'
    path = base_path + 'grid_train_features_chunk*.h5'

    dataset = AVDataset(path, mode='av', mask_range='rand')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(len(dataloader))
    for masked_spec, motions, mel_spec, text, mask in dataloader:
        print(motions.shape)
        print(mel_spec[0].shape)
        plt.imshow(mel_spec[0])
        #plt.imshow(frames[0, 30,0, ...])
        plt.show()

    # splits = { "train"/"dev", "val", "test"}
    #
    # for split in splits:
    #     feats_path = f'/home/ai/Projects/Mahsa/datasets/grid/grid_{split}_features'
    #     update_h5(feats_path)

