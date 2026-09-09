# Portable root configuration
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from evaluations.runtime_config import dataset_patterns, SEED

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from AV_Transformer.av_l_dataset import AVDataset

class AVDataloader:
    def __init__(self, dataset_name, mode, batch_size, num_workers,
                 train_subset=None, val_subset=None, test_subset=None):

        assert dataset_name in ['grid', 'lrs2', 'voxceleb2'], f"Invalid dataset_name: {dataset_name}"

        paths = dataset_patterns(dataset_name)
        self.train_files = paths["train"]
        self.val_files = paths["val"]
        self.test_files = paths["test"]


        self.mode = mode

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_subset = train_subset
        self.val_subset = val_subset
        self.test_subset = test_subset

    def worker_init_fn(self, worker_id):
        worker_seed = torch.initial_seed() % (2 ** 32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        while isinstance(dataset, Subset):
            dataset = dataset.dataset
        if hasattr(dataset, "close_h5_files"):
            dataset.close_h5_files()

    def train_dataloader(self):
        train_dataset = AVDataset(base_path=self.train_files, mode=self.mode, online_loss_bounds=(0.3, 0.9))
        if self.train_subset is not None:
            train_indices = torch.randperm(len(train_dataset),
                                           generator=torch.Generator().manual_seed(SEED)).tolist()
            train_dataset = Subset(train_dataset, train_indices[:self.train_subset])

        return DataLoader(train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          worker_init_fn=self.worker_init_fn,
                          generator=torch.Generator().manual_seed(SEED))

    def val_dataloader(self):
        val_dataset = AVDataset(base_path=self.val_files, mode=self.mode, online_loss_bounds=(0.3, 0.9), set_seed=SEED)
        if self.val_subset is not None:
            val_indices = torch.randperm(len(val_dataset),
                                         generator=torch.Generator().manual_seed(SEED)).tolist()
            val_dataset = Subset(val_dataset, val_indices[:self.val_subset])

        return DataLoader(val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          drop_last=False,
                          worker_init_fn=self.worker_init_fn,
                          generator=torch.Generator().manual_seed(SEED))

    def test_dataloader(self, mask_range=None, mask_type="gilbert", gap_ms=None, seed=SEED):
        if mask_type == "gilbert":
            valid = ['10', '20', '30', '40', '50', '60', '70', '80', '90']
            if str(mask_range) not in valid:
                raise ValueError(f"Invalid Gilbert-Elliott mask_range: {mask_range}")
            test_dataset = AVDataset(
                base_path=self.test_files, mode=self.mode, mask_range=str(mask_range),
                set_seed=seed, mask_type="gilbert",
            )
        elif mask_type == "single_gap":
            if gap_ms is None:
                raise ValueError("gap_ms is required for single_gap testing")
            test_dataset = AVDataset(
                base_path=self.test_files, mode=self.mode, mask_range='rand',
                set_seed=None, mask_type="single_gap", gap_ms=gap_ms, mask_seed=seed,
            )
        else:
            raise ValueError(f"Unsupported mask_type: {mask_type}")

        if self.test_subset is not None:
            test_indices = torch.randperm(
                len(test_dataset), generator=torch.Generator().manual_seed(SEED)
            ).tolist()
            test_dataset = Subset(test_dataset, test_indices[:self.test_subset])

        return DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, drop_last=False,
            worker_init_fn=self.worker_init_fn,
            generator=torch.Generator().manual_seed(SEED),
        )

    def __repr__(self) -> str:
        return (
            f"<Datasets: ["
            f'train: {len(self.train_files)}, '
            f'val: {len(self.val_files)}, '
            f'test: {len(self.test_files)}'
            f"]>"
        )

