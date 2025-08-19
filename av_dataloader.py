import h5py
import torch
import os, random
import numpy as np
from glob import glob

from torch.utils.data import Dataset, DataLoader, Subset
from av_augmentation import Compose, RandomCrop, RandomErase, TimeMask, HorizontalFlip, ModalityDropout

# ----------------------------- DATASET -----------------------------

VALID_MODES = {'a', 'v', 'av', 'motion'}
VALID_MASK_RANGES = {'20', '30', '40', '50', '60', 'rand'}

class AVDataset(Dataset):

    def __init__(self, base_path_or_pattern, mode='v', mask_range='rand',
                 augment=True, drop_av=False, chunk_pattern="_chunk*.h5"):

        self._validate_args(mode, mask_range)
        self.mode = mode
        self.augment = augment
        self.drop_av = drop_av
        self.mask_range = mask_range

        # Resolve file pattern
        pattern = (base_path_or_pattern if chunk_pattern in base_path_or_pattern
                   else os.path.join(base_path_or_pattern, chunk_pattern))
        self.chunk_files = sorted(glob(pattern))
        if not self.chunk_files:
            raise FileNotFoundError(f"No HDF5 files found at pattern: {pattern}")

        # Build index map (chunk_idx, key)
        self.index_map, self.chunk_sizes = [], []
        for chunk_idx, path in enumerate(self.chunk_files):
            with h5py.File(path, 'r') as h5f:
                keys = list(h5f.keys())
                self.index_map.extend((chunk_idx, k) for k in keys)
                self.chunk_sizes.append(len(keys))

        # Default mel stats by dataset name hint
        bp = base_path_or_pattern.lower()
        if 'grid' in bp:
            self.mel_mean, self.mel_std = -56.775, 19.707
        elif 'vox' in bp:
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

    def __getitem__(self, idx):
        chunk_idx = None
        video_key = None
        try:
            chunk_idx, video_key = self.index_map[idx]
            h5f = self._get_h5_file(chunk_idx)

            video_path = h5f.attrs.get(f"{video_key}/video_path", None)
            mel_spec = h5f[f"{video_key}/spec"][:].astype(np.float32)
            if self.mask_range == 'rand':
                mask = h5f[f"{video_key}/mask"][:].astype(np.float32)
            else:
                mask = h5f[f"{video_key}/mask_{self.mask_range}"][:].astype(np.float32)

            # audio normalization
            mel_spec = (mel_spec - self.mel_mean) / self.mel_std
            masked_spec = mel_spec * mask

            if self.mode == 'a':
                return masked_spec, mel_spec  #, text, mask)

            if self.mode == 'motion':
                text = h5f[f"{video_key}/text"][:]
                landmarks = h5f[f"{video_key}/landmarks"][:].astype(np.float32)
                valid = ~np.all(landmarks == 0, axis=(1, 2))
                if not np.any(valid):
                    raise RuntimeError("No valid landmarks (all zero).")
                motions = np.diff(landmarks[valid], axis=0)
                padded = np.zeros_like(landmarks, dtype=np.float32)
                padded[:len(motions)] = motions
                return masked_spec, padded, mel_spec, text, mask

            if self.mode == 'v':
                spk_emb = h5f[f"{video_key}/spkr_embd"][:].astype(np.float32)
                frames = h5f[f"{video_key}/frames"][:]          # uint8 [T,H,W,C]
                frames = frames.astype(np.float32) / 255.  # normalize to [0,1] float32
                frames = (frames - 0.421) / 0.165
                frames = self._aug_video_frames(frames)  # apply video augmentations
                return frames, spk_emb, mel_spec

            if self.mode == 'av':
                spk_emb = h5f[f"{video_key}/spkr_embd"][:]  # .astype(np.float32)
                frames = h5f[f"{video_key}/frames"][:]          # uint8 [T,H,W,C]
                frames = frames / 255.  # normalize to [0,1] float32
                frames = (frames - 0.421)/0.165
                frames, masked_spec = self._drop_av_modality(video_path, frames, masked_spec)
                return frames, spk_emb, masked_spec, mel_spec

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

    def _drop_av_modality(self, video_path, frames, masked_spec):
        if self.modality_dropout is None:
            return frames, masked_spec
        if not video_path:
            return frames, masked_spec
        if ('train' in video_path) or ('dev' in video_path):
            mode = self.modality_dropout.sample_mode()
            if mode == 'audio_only':
                return np.zeros_like(frames, dtype=frames.dtype), masked_spec
            if mode == 'video_only':
                return frames, np.zeros_like(masked_spec, dtype=masked_spec.dtype)
        return frames, masked_spec


# ----------------------------- DATALOADER  -----------------------------

class AVDataloader:
    """
    Building train/val/test DataLoaders with safe worker init and mode-aware collate.
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
            base_path = 'datasets/grid/'
            self.train_files = base_path + 'grid_train_features_chunk*.h5'
            self.val_files   = base_path + 'grid_val_features_chunk*.h5'
            self.test_files  = base_path + 'grid_test_features_chunk*.h5'
        else:  # 'voxceleb2'
            base_path = 'datasets/vox2_short/'
            self.train_files = base_path + 'vox2_short_dev_features_chunk*.h5'
            self.val_files   = base_path + 'vox2_short_val_features_chunk*.h5'
            self.test_files  = base_path + 'vox2_short_test_features_chunk*.h5'

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
        seed = (torch.initial_seed() % 2**32) + worker_id
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        try:
            import cv2
            cv2.setNumThreads(0)
        except Exception:
            pass
        torch.set_num_threads(1)

    def _build_loader(self, files_glob, augment, drop_av, subset=None,
                      drop_last=False, shuffle=False, mask_range='rand'):
        ds = AVDataset(files_glob, self.mode, mask_range, augment=augment, drop_av=drop_av)
        if subset is not None:
            idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0)).tolist()
            ds = Subset(ds, idx[:subset])

        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            #collate_fn=collate,                     #  heavy lifting happens here
            worker_init_fn=self._worker_init_fn if self.num_workers > 0 else None,
            drop_last=drop_last,
        )

    # ---------- public builders ----------

    def train_dataloader(self):
        return self._build_loader(self.train_files, augment=(self.mode=='v'),
                                  drop_av=(self.mode=='av'),
                                  subset=self.train_subset, drop_last=False,
                                  shuffle=True, mask_range='rand')

    def val_dataloader(self):
        return self._build_loader(self.val_files, augment=False,
                                  drop_av=False,
                                  subset=self.val_subset, drop_last=False,
                                  shuffle=False, mask_range='rand')

    def test_dataloader(self, mask_range='rand'):
        if mask_range not in VALID_MASK_RANGES:
            raise ValueError(f"Invalid mask_range: {mask_range}")
        return self._build_loader(self.test_files,  augment=False, drop_av=False,
                                  subset=self.test_subset, drop_last=False,
                                  shuffle=False, mask_range=mask_range)


if __name__ == "__main__":
    from asteroid.losses.pmsqe import SingleSrcPMSQE
    from audio_processing import torch_mel2spec
    from matplotlib import pyplot as plt
    dataloader = AVDataloader('grid', 'a', batch_size=2, num_workers=4,)
    train_loader = dataloader.train_dataloader()
    for batch in train_loader:
        masked_spec, mel_spec = batch
        plt.imshow(mel_spec[0])
        plt.show()
        #print(masked_spec.shape, mel_spec.shape)
        #print(torch.max(mel_spec), torch.min(mel_spec))
        #print(SingleSrcPMSQE()(torch_mel2spec(masked_spec.float()).permute(0, 2, 1).contiguous(),
        #                       torch_mel2spec(mel_spec.float()).permute(0, 2, 1).contiguous()))
        break  # just one batch for demonstration