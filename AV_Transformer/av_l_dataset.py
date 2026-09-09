import hashlib
import h5py
import re, os
import numpy as np
from glob import glob
from torch.utils.data import Dataset

from shared.masking import generate_ge_mask_bursty, generate_single_gap_mask
from evaluations.runtime_config import SEED

"""Eager conversion to Torch means converting to Torch Tensor in __getitem__ that is
 not good for large data. Lazy conversion returns numpy then batch and convert → much safer."""

class AVDataset(Dataset):

    def __init__(self, base_path, mode, online_loss_bounds=None, set_seed=None, mask_range='rand', chunk_pattern="_chunk*.h5",
                 mask_type="gilbert", gap_ms=None, mask_seed=SEED):

        self.set_seed = set_seed
        self.online_loss_bounds = online_loss_bounds
        valid_modes = ['a', 'av']
        valid_mask_ranges = {'1','10','20','30','40','50','60','70','80','90','99','rand'}


        if 'grid' in base_path:
            self.mel_mean, self.mel_std = -56.775, 19.707
        elif 'vox' in base_path:
            self.mel_mean, self.mel_std = -52.43, 17.499
        else:
            self.mel_mean, self.mel_std = -54.60, 18.60

        if mode not in valid_modes:
            raise ValueError(f"Mode must be one of {valid_modes}, got {mode}")
        self.mode = mode

        if mask_range not in valid_mask_ranges:
            raise ValueError(f"mask ranges must be one of {valid_mask_ranges}, got {mask_range}")
        self.mask_range = mask_range
        self.mask_type = mask_type
        self.gap_ms = gap_ms
        self.mask_seed = int(mask_seed)

        if chunk_pattern in base_path:
            self.chunk_files = sorted(glob(base_path), key=self.chunk_idx)
        else:
            self.chunk_files = sorted(glob(f"{base_path}{chunk_pattern}"), key=self.chunk_idx)

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

    def chunk_idx(self, s: str) -> int:
        m = re.search(r'_chunk(\d+)\.h5$', os.path.basename(s))
        return int(m.group(1)) if m else -1

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


    def _stable_item_seed(self, sample_id: str, condition: str) -> int:
        token = f"{self.set_seed}:{condition}:{sample_id}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(token).digest()[:4], "little")

    def __getitem__(self, idx):

        chunk_idx, video_key = self.index_map[idx]
        #chunk_file = self.chunk_files[chunk_idx]
        h5f = self._get_h5_file(chunk_idx)

        #with h5py.File(chunk_file, 'r') as h5f:
        text = h5f[f"{video_key}/text"][:]
        mel_spec = h5f[f"{video_key}/spec"][:]
        video_path = h5f.attrs.get(f"{video_key}/video_path", None)

        sample_id = f"{os.path.basename(self.chunk_files[chunk_idx])}:{video_key}"
        if self.mask_type == "single_gap":
            if self.gap_ms is None:
                raise ValueError("gap_ms is required when mask_type=single_gap")
            mask = generate_single_gap_mask(mel_spec.shape, self.gap_ms,
                                            sample_id, seed=self.mask_seed)
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
                mask = generate_ge_mask_bursty( mel_spec.shape,
                                                loss_rate=float(self.mask_range) / 100)
            finally:
                np.random.set_state(rng_state)
        elif self.mask_range == 'rand':
            mask = h5f[f"{video_key}/mask"][:]
        else:
            mask = h5f[f"{video_key}/mask_{self.mask_range}"][:]

        mel_spec = (mel_spec - self.mel_mean) / self.mel_std
        masked_spec = mel_spec * mask

        if self.mode == 'a':
            return masked_spec, mel_spec, text, mask, video_path

        elif self.mode == 'av':
            visual_features = h5f[f"{video_key}/visual_features"][:]

            return masked_spec, visual_features, mel_spec, text, mask, video_path
