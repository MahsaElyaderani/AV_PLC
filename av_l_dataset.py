import h5py
import torch
import numpy as np
from glob import glob
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from utils import visualize_motion
from av_augmentation import VideoAugmentations, SpectrogramAugmentations, ModalityDropout

"""Eager conversion to Torch means converting to Torch Tensor in __getitem__ that is
 not good for large data. Lazy conversion returns numpy then batch and convert → much safer."""

class AVDataset(Dataset):

    def __init__(self, base_path, mode, mask_range='rand', chunk_pattern="_chunk*.h5"):

        valid_modes = ['a', 'v', 'av', 'motion']
        valid_mask_ranges = ['20', '30', '40', '50', '60', 'rand']
        self.video_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize([112, 112]),
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=0.421, std=0.165),
        ])

        # if mode == 'v':
        #     self.video_aug = VideoAugmentations(flip_p=0.3, drop_prob=0.01,
        #                                               noise_std=0.01, temporal_jitter=1)
        #     self.spec_aug = SpectrogramAugmentations(freq_mask=10, time_mask=10, num_masks=2)
        #     self.modality_dropout = ModalityDropout(mode_probs=[0.6, 0.2, 0.2])

        if mode not in valid_modes:
            raise ValueError(f"Mode must be one of {valid_modes}, got {mode}")
        self.mode = mode

        if mask_range not in valid_mask_ranges:
            raise ValueError(f"mask ranges must be one of {valid_mask_ranges}, got {mask_range}")
        self.mask_range = mask_range

        if chunk_pattern in base_path:
            self.chunk_files = sorted(glob(base_path))
        else:
            self.chunk_files = sorted(glob(f"{base_path}{chunk_pattern}"))

        if not self.chunk_files:
            raise ValueError(f"No HDF5 chunk files found with pattern {base_path}{chunk_pattern}")

        # Build an index mapping dataset indices to (chunk_idx, video_key)
        self.index_map = []
        self.chunk_sizes = []

        for chunk_idx, chunk_file in enumerate(self.chunk_files):
            with h5py.File(chunk_file, 'r') as h5f:
                video_keys = list(h5f.keys())
                for video_key in video_keys:
                    self.index_map.append((chunk_idx, video_key))
                self.chunk_sizes.append(len(video_keys))

        print(f"Found {len(self.index_map)} total videos across {len(self.chunk_files)} chunks")


    def _get_h5_file(self, chunk_idx):
        MAX_OPEN_FILES = 20

        if not hasattr(self, "_h5_cache"):
            self._h5_cache = {}

        if chunk_idx not in self._h5_cache:
            if len(self._h5_cache) > MAX_OPEN_FILES:
                old_chunk_idx = list(self._h5_cache.keys())[0] # Close least recently used
                self._h5_cache[old_chunk_idx].close()
                del self._h5_cache[old_chunk_idx]
            chunk_file = self.chunk_files[chunk_idx]
            self._h5_cache[chunk_idx] = h5py.File(chunk_file, "r", swmr=True)

        return self._h5_cache[chunk_idx]

    def close_h5_files(self):
        if hasattr(self, "_h5_cache"):
            for h5f in self._h5_cache.values():
                h5f.close()
            self._h5_cache = {}

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):

        chunk_idx, video_key = self.index_map[idx]
        h5f = self._get_h5_file(chunk_idx)

        text = h5f[f"{video_key}/text"][:]
        mel_spec = h5f[f"{video_key}/mel_spec"][:]
        video_path = h5f.attrs.get(f"{video_key}/video_path", None)

        if self.mask_range == 'rand':
            mask = h5f[f"{video_key}/mask"][:]
        else:
            mask = h5f[f"{video_key}/mask_{self.mask_range}"][:]

        mask = torch.tensor(mask)
        mel_spec = torch.nn.functional.layer_norm(torch.tensor(mel_spec),
                                                  torch.tensor(mel_spec).shape, eps=0)
        masked_spec = mel_spec * mask

        if self.mode == 'a':
            return masked_spec, mel_spec, text, mask

        elif self.mode == 'motion':
            landmarks = h5f[f"{video_key}/landmarks"][:]
            valid_landmarks = ~(np.all(landmarks == 0, axis=(1, 2)))
            #num_real_frames = np.sum(valid_landmarks)
            valid_motions = np.diff(landmarks[valid_landmarks], axis=0)
            motions = np.zeros_like(landmarks)
            motions[:len(valid_motions)] = valid_motions
            #phase = h5f[f"{video_key}/phase"][:]

            return masked_spec, motions, mel_spec, text, mask

        elif self.mode == 'v':
            #if 'train' in video_path:
            #     mode = self.modality_dropout.sample_mode()
            #
            #     if mode == 'audio_video':
            #         frames = h5f[f"{video_key}/frames"][:]
            #         #frames = self.video_transform(frames)
            #         #masked_spec = self.spec_transform(masked_spec)
            #         return masked_spec, frames, mel_spec, text, mask
            #
            #     elif mode == 'audio_only':
            #         video_is_missing = True
            #         frames = np.zeros_like(h5f[f"{video_key}/frames"][:])
            #         #frames = self.video_transform(frames)
            #         #masked_spec = self.spec_transform(masked_spec)
            #         return masked_spec, frames, mel_spec, text, mask
            #
            #     elif mode == 'video_only':
            #         video_is_missing = False
            #         frames = h5f[f"{video_key}/frames"][:]
            #         #frames = self.video_transform(frames)
            #         #masked_spec = np.zeros_like(masked_spec)
            #         #masked_spec = self.spec_transform(masked_spec)
            #         return masked_spec, frames, mel_spec, text, mask
            # else:
            #     frames = h5f[f"{video_key}/frames"][:]

            frames = h5f[f"{video_key}/frames"][:]
            spk_emb = h5f[f"{video_key}/spkr_embd"][:]

            # Discard all-zero frames: shape [T, H, W, C]
            video_mask = np.any(frames != 0, axis=(1, 2, 3))  # shape [T]
            frames = frames[video_mask]

            # Apply transform to each frame (convert from NumPy to PIL or tensor first)
            # Assuming frames shape is (T, H, W, C) — e.g., RGB video
            processed_frames = []
            for frame in frames:
                # Convert from NumPy to PIL for torchvision transforms (Resize, Grayscale, etc.)
                pil_frame = Image.fromarray(frame.astype(np.uint8))  # Use fromarray safely
                transformed = self.video_transform(pil_frame)
                processed_frames.append(transformed)

            # Stack into a Tensor: shape (T, C, H, W)
            frames_tensor = torch.stack(processed_frames)
            frames_tensor = frames_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # [c, t_v, h, w]
            if frames_tensor.shape[0] < 75:
                H, W = frames_tensor.shape[-2], frames_tensor.shape[-1]
                frames_tensor = torch.nn.functional.interpolate(frames_tensor, size=(75, H, W),
                                              mode='trilinear', align_corners=False)
                frames_tensor = frames_tensor.squeeze(0).permute(1, 0, 2, 3)


            return frames_tensor, torch.tensor(spk_emb), masked_spec, mel_spec, mask



if __name__ == "__main__":


    # base_path = '/home/ai/Projects/Mahsa/datasets/vox2_short/'
    # path = base_path + 'vox2_short_test_features_chunk*.h5'

    base_path = 'datasets/grid/'
    path = base_path + 'grid_test_features_chunk*.h5'

    dataset = AVDataset(path, mode='v', mask_range='60')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(len(dataloader))
    for frames, spk_emb, masked_spec, mel_spec, mask in dataloader:
        print(frames.shape)
        print(masked_spec[13].shape)
        plt.imshow(masked_spec[13])
        plt.imshow(frames[0, 30, ...] / 255.)
        plt.show()

