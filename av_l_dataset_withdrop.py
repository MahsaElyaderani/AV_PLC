"""Eager conversion to Torch means converting to
 Torch Tensor in __getitem__ that is
 not good for large data.
 Lazy conversion returns numpy then batch and convert → much safer."""

import os
import gc
import h5py
import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from av_augmentation import ModalityDropout

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

        if mode == 'av':
            self.modality_dropout = ModalityDropout()

        if 'grid' in base_path:
            self.mel_mean = -56.775
            self.mel_std = 19.707
        elif 'vox2' in base_path:
            self.mel_mean = -52.43
            self.mel_std = 17.499
        else:
            self.mel_mean = -54.60
            self.mel_std = 18.60

        self._h5_cache = {}

    def _validate_args(self, mode, mask_range):
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode: {mode}. Choose from {VALID_MODES}")
        if mask_range not in VALID_MASK_RANGES:
            raise ValueError(f"Invalid mask range: {mask_range}. Choose from {VALID_MASK_RANGES}")

    def __len__(self):
        return len(self.index_map)

    def _get_h5_file(self, chunk_idx):
        MAX_OPEN_FILES = 5
        if chunk_idx not in self._h5_cache:
            if len(self._h5_cache) >= MAX_OPEN_FILES:
                old = next(iter(self._h5_cache))
                self._h5_cache[old].close()
                del self._h5_cache[old]
                gc.collect()
            self._h5_cache[chunk_idx] = h5py.File(self.chunk_files[chunk_idx], 'r', swmr=True)
        return self._h5_cache[chunk_idx]

    def close_h5_files(self):
        for f in self._h5_cache.values():
            f.close()
        self._h5_cache.clear()

    def _process_video_frames(self, frames):
        # fr: (T,H,W,1) or (T,H,W) uint8
        if frames.ndim == 3: frames = frames[..., None]
        frames = frames.astype(np.float32) / 255.0  # [0,1]
        frames = (frames - 0.421) / 0.165  # normalize
        frames = np.transpose(frames, (0, 3, 1, 2))  # (T,1,H,W)
        return frames

    def __getitem__(self, idx):

        chunk_idx, video_key = self.index_map[idx]
        h5f = self._get_h5_file(chunk_idx)

        video_path = h5f.attrs.get(f"{video_key}/video_path", None)
        mel_spec = h5f[f"{video_key}/spec"][:]
        text = h5f[f"{video_key}/text"][:]
        mask = h5f[f"{video_key}/mask"][:] if self.mask_range == 'rand' else h5f[f"{video_key}/mask_{self.mask_range}"][:]
        mel_spec = (mel_spec - self.mel_mean) / self.mel_std
        masked_spec = mel_spec * mask

        if self.mode == 'a':
            return masked_spec, mel_spec, text, mask

        elif self.mode == 'motion':
            landmarks = h5f[f"{video_key}/landmarks"][:]
            valid = ~np.all(landmarks == 0, axis=(1, 2))
            motions = np.diff(landmarks[valid], axis=0)
            padded = np.zeros_like(landmarks)
            padded[:len(motions)] = motions
            return masked_spec, padded, mel_spec, text, mask

        elif self.mode == 'v':
            frames = h5f[f"{video_key}/frames"][:]
            spk_emb = h5f[f"{video_key}/spkr_embed"][:]
            frames = self._process_video_frames(frames)

            return frames, spk_emb, masked_spec, mel_spec, mask

        elif self.mode == 'av':

            spk_emb = h5f[f"{video_key}/spkr_embd"][:]

            if 'train' in video_path:
                mode = self.modality_dropout.sample_mode()

                if mode == 'audio_video':
                    frames = h5f[f"{video_key}/frames"][:]
                    frames = self._process_video_frames(frames)
                    return frames, spk_emb, masked_spec, mel_spec, mask

                elif mode == 'audio_only':
                    frames = np.zeros((75, 1, 112, 112), dtype=np.float32)  # Assuming 75 frames of size 112x112 with 1 channel
                    return frames, spk_emb, masked_spec, mel_spec, mask

                elif mode == 'video_only':
                    frames = h5f[f"{video_key}/frames"][:]
                    frames = self._process_video_frames(frames)
                    masked_spec = np.zeros_like(masked_spec)
                    return frames, spk_emb, masked_spec, mel_spec, mask
            else:
                frames = h5f[f"{video_key}/frames"][:]
                frames = self._process_video_frames(frames)

                return frames, spk_emb, masked_spec, mel_spec, mask

        raise NotImplementedError(f"Unsupported mode: {self.mode}")

    # def _process_video_frames(self, frames_np):
    #     """
    #     video frames shape: [T, H, W, C].
    #     process video frames shape [T, C, H, W].
    #     """
    #     #valid = np.any(frames_np != 0, axis=(1, 2, 3))
    #     #frames_np = frames_np[valid]
    #
    #     processed = []
    #     for f in frames_np:
    #         #img = Image.fromarray(f.astype(np.uint8))
    #         img = Image.fromarray(f[..., 0])
    #         processed.append(self.video_transform(img))
    #
    #     frames = torch.stack(processed)
    #     #aug_frames = self.video_augment(frames)
    #     return frames #aug_frames

if __name__ == "__main__":

    import math
    from av_l_dataloader_withdrop import AVDataloader
    # base_path = '/home/ai/Projects/Mahsa/datasets/vox2_short/'
    # path = base_path + 'vox2_short_test_features_chunk*.h5'

    #base_path = '/home/ai/Projects/Mahsa/datasets/grid/' #'datasets/grid/'
    #path = base_path + 'grid_train_features_chunk*.h5'

    #dataset = AVDataset(path, mode='v', mask_range='rand')
    #dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    dataset_name = 'grid'
    av_loader = AVDataloader(dataset_name, 'av', 4, 0)
    dataloader = av_loader.train_dataloader()
    print(len(dataloader))

    sum_val = 0.0
    sum_sqr_val = 0.0
    count = 0

    for frames, spk_emb, masked_spec, mel_spec, mask in dataloader:
        print(frames.shape)
        #plt.imshow(frames[0,60,0,:,:])
        #plt.show()
        # mel_spec: [B, F, T] or [B, 1, F, T]
    #     num_elements = mel_spec.numel()  # total elements in the batch
    #
    #     sum_val += mel_spec.sum().item()
    #     sum_sqr_val += (mel_spec ** 2).sum().item()
    #     count += num_elements
    #
    # global_mean = sum_val / count
    # global_var = (sum_sqr_val / count) - (global_mean ** 2)
    # #global_var = max(global_var, 0.0)  # avoid small negatives
    # global_std = math.sqrt(global_var)
    #
    # print(f"Mean: {global_mean}, std: {global_std}")
    #
    # with open("voxceleb2_mel_stats.txt", "w") as f:
    #     f.write(f"global mean of {dataset_name}: {global_mean}\n")
    #     f.write(f"global std of {dataset_name}: {global_std}\n")

