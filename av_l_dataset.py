"""Eager conversion to Torch means converting to
 Torch Tensor in __getitem__ that is
 not good for large data.
 Lazy conversion returns numpy then batch and convert → much safer."""

import os
import h5py
import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader


VALID_MODES = {'a', 'v', 'av', 'motion'}
VALID_MASK_RANGES = {'20', '30', '40', '50', '60', 'rand'}


class AVDataset(Dataset):
    def __init__(self, base_path, mode='v', mask_range='rand', chunk_pattern="_chunk*.h5"):
        self._validate_args(mode, mask_range)

        self.mode = mode
        self.mask_range = mask_range

        self.chunk_files = sorted(glob(base_path if chunk_pattern in base_path else os.path.join(base_path, chunk_pattern)))
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

        self.video_transform = T.Compose([
            T.Resize((112, 112)),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
            T.Normalize(mean=0.421, std=0.165),
        ])

        self._h5_cache = {}

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

    def close(self):
        for f in self._h5_cache.values():
            f.close()
        self._h5_cache.clear()

    def __getitem__(self, idx):
        chunk_idx, video_key = self.index_map[idx]
        h5f = self._get_h5_file(chunk_idx)

        mel_spec = torch.tensor(h5f[f"{video_key}/mel_spec"][:], dtype=torch.float32)
        text = h5f[f"{video_key}/text"][:]
        mask = h5f[f"{video_key}/mask"][:] if self.mask_range == 'rand' else h5f[f"{video_key}/mask_{self.mask_range}"][:]
        mask = torch.tensor(mask, dtype=torch.float32)
        masked_spec = torch.nn.functional.layer_norm(mel_spec, mel_spec.shape) * mask

        if self.mode == 'a':
            return masked_spec, mel_spec, text, mask

        elif self.mode == 'motion':
            landmarks = h5f[f"{video_key}/landmarks"][:]
            valid = ~np.all(landmarks == 0, axis=(1, 2))
            motions = np.diff(landmarks[valid], axis=0)
            padded = np.zeros_like(landmarks)
            padded[:len(motions)] = motions
            return masked_spec, torch.tensor(padded), mel_spec, text, mask

        elif self.mode == 'v':
            frames = h5f[f"{video_key}/frames"][:]
            spk_emb = h5f[f"{video_key}/spkr_embd"][:]
            frames = self._process_video_frames(frames)
            return frames, torch.tensor(spk_emb), masked_spec, mel_spec, mask

        raise NotImplementedError(f"Unsupported mode: {self.mode}")

    def _process_video_frames(self, frames_np):
        """
        video frames shape: [T, H, W, C].
        process video frames shape [T, C, H, W].
        """
        valid = np.any(frames_np != 0, axis=(1, 2, 3))
        frames_np = frames_np[valid]

        processed = []
        for f in frames_np:
            img = Image.fromarray(f.astype(np.uint8))
            processed.append(self.video_transform(img))

        return torch.stack(processed)

if __name__ == "__main__":


    # base_path = '/home/ai/Projects/Mahsa/datasets/vox2_short/'
    # path = base_path + 'vox2_short_test_features_chunk*.h5'

    base_path = 'datasets/grid/'
    path = base_path + 'grid_test_features_chunk*.h5'

    dataset = AVDataset(path, mode='v', mask_range='60')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(len(dataloader))
    for frames, spk_emb, masked_spec, mel_spec, mask in dataloader:
        print(frames.shape)
        print(mel_spec[0].shape)
        plt.imshow(mel_spec[0])
        plt.imshow(frames[0, 30,0, ...])
        plt.show()

