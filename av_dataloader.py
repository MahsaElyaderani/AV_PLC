import os
import torch
import glob
from torch.utils.data import DataLoader, Subset
from stable_diffusion.models.plc.av_dataset import AV_Dataset

class AV_Dataloader:
    def __init__(self, dataset_name, mode, batch_size, num_workers,
                 train_subset=None, val_subset=None, test_subset=None):

        assert dataset_name in ['grid', 'voxceleb2'], f"Invalid dataset_name: {dataset_name}"

        if dataset_name == 'grid':
            base_path = '/home/ai/Projects/Mahsa/datasets/grid/'
            self.train_files = glob.glob(os.path.join(base_path, 'train', 's*/*.mpg'))
            self.val_files = glob.glob(os.path.join(base_path, 'val', 's*/*.mpg'))
            self.test_files = glob.glob(os.path.join(base_path, 'test', 's*/*.mpg'))
            # self.train_files = base_path + 'grid_train_features_chunk*.h5'
            # self.val_files = base_path + 'grid_val_features_chunk*.h5'
            # self.test_files = base_path + 'grid_test_features_chunk*.h5'

        elif dataset_name == 'voxceleb2':
            base_path = '/home/ai/Projects/Mahsa/datasets/vox2_short/'
            self.train_files = base_path + 'vox2_short_dev_features_chunk*.h5'
            self.val_files = base_path + 'vox2_short_val_features_chunk*.h5'
            self.test_files = base_path + 'vox2_short_test_features_chunk*.h5'


        self.mode = mode

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_subset = train_subset
        self.val_subset = val_subset
        self.test_subset = test_subset

    def worker_init_fn(self, worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        dataset.close_h5_files()

    def train_dataloader(self):
        train_dataset = AV_DATASET(self.train_files, 'rand')
        if self.train_subset is not None:
            train_indices = torch.randperm(len(train_dataset),
                                           generator=torch.Generator().manual_seed(0)).tolist()
            train_dataset = Subset(train_dataset, train_indices[:self.train_subset])

        return DataLoader(train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=True,)
                          #worker_init_fn=self.worker_init_fn)

    def val_dataloader(self):
        val_dataset = AV_DATASET(self.val_files, 'rand')
        if self.val_subset is not None:
            val_indices = torch.randperm(len(val_dataset),
                                         generator=torch.Generator().manual_seed(0)).tolist()
            val_dataset = Subset(val_dataset, val_indices[:self.val_subset])

        return DataLoader(val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          drop_last=True,)
                          #worker_init_fn=self.worker_init_fn)

    def test_dataloader(self, mask_range):

        if mask_range in ['20', '30', '40', '50', '60', 'rand']:
            test_dataset = AV_DATASET(self.test_files, mask_range)
        else:
            raise ValueError(f"Invalid mask_range: {mask_range}")

        if self.test_subset is not None:
            test_indices = torch.randperm(len(test_dataset),
                                         generator=torch.Generator().manual_seed(0)).tolist()
            test_dataset = Subset(test_dataset, test_indices[:self.test_subset])

        return DataLoader(test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          drop_last=True,
                          worker_init_fn=self.worker_init_fn)

    def __repr__(self) -> str:
        return (
            f"<Datasets: ["
            f'train: {len(self.train_files)}, '
            f'val: {len(self.val_files)}, '
            f'test: {len(self.test_files)}'
            f"]>"
        )

