import os
import torch
from torch.utils.data import DataLoader, Subset
from av_l_dataset_withdrop import AVDataset

class AVDataloader:
    def __init__(self, dataset_name, mode, batch_size, num_workers,
                 train_subset=None, val_subset=None, test_subset=None):

        assert dataset_name in ['grid', 'voxceleb2'], f"Invalid dataset_name: {dataset_name}"

        if dataset_name == 'grid':
            base_path = 'datasets/' #'datasets/grid/'
            self.train_files = base_path + 'grid_train_features_chunk*.h5'
            self.val_files = base_path + 'grid_val_features_chunk*.h5'
            self.test_files = base_path + 'grid_test_features_chunk*.h5'

        elif dataset_name == 'voxceleb2':
            base_path = 'datasets/vox2_short/'
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
        train_dataset = AVDataset(self.train_files, self.mode, 'rand')
        if self.train_subset is not None:
            train_indices = torch.randperm(len(train_dataset),
                                           generator=torch.Generator().manual_seed(0)).tolist()
            train_dataset = Subset(train_dataset, train_indices[:self.train_subset])

        return DataLoader(train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=True,)
                          #collate_fn=self.av_collate_fn,)
                          #worker_init_fn=self.worker_init_fn)

    def val_dataloader(self):
        val_dataset = AVDataset(self.val_files, self.mode, 'rand')
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
                          #collate_fn=self.av_collate_fn,)
                          #worker_init_fn=self.worker_init_fn)

    def test_dataloader(self, mask_range):

        if mask_range in ['20', '30', '40', '50', '60', 'rand']:
            test_dataset = AVDataset(self.test_files, self.mode, mask_range)
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
                          #collate_fn=self.av_collate_fn,
                          worker_init_fn=self.worker_init_fn)

    def av_collate_fn(self, batch):
        if self.mode == 'v' or self.mode == 'av':
            frames, spk_embs, masked_specs, mel_specs, masks = zip(*batch)

            processed_frames = []
            for f in frames:  # each f: [T, C, H, W] or [75, 1, 112, 112]
                f = f.permute(1, 0, 2, 3).unsqueeze(0)  # → [1, C, T, H, W]
                _, C, T, H, W = f.shape
                if T < 75:
                    f = torch.nn.functional.interpolate(f, size=(75, H, W),
                                                        mode='trilinear',
                                                        align_corners=False)
                f = f.squeeze(0).permute(1, 0, 2, 3)  # → [75, C, H, W]
                processed_frames.append(f)

            frames = torch.stack(processed_frames)  # [B, 75, C, H, W]
            spk_embs = torch.stack([torch.as_tensor(s) for s in spk_embs])
            masked_specs = torch.stack([torch.as_tensor(masked_spec) for masked_spec in masked_specs])
            mel_specs = torch.stack([torch.as_tensor(mel_spec) for mel_spec in mel_specs])
            masks = torch.stack([torch.as_tensor(mask) for mask in masks])
            #masked_frames = self.apply_audio_mask_to_video(frames, masks)

            return frames, spk_embs, masked_specs, mel_specs, masks
            #return masked_frames, spk_embs, masked_specs, mel_specs, masks

    def apply_audio_mask_to_video(self, videos: torch.Tensor, audio_masks: torch.Tensor, threshold: float = 0.5):

        masked_videos = []
        for audio_mask, video in zip(audio_masks, videos):

            T_v = video.shape[0]  # 75

            # 1. Collapse mel bins to temporal dimension
            time_mask = audio_mask.min(dim=0).values.float()  # [300], 1 = all OK, 0 = any lost

            # 2. Reshape to [1, 1, 300] for pooling
            time_mask = time_mask.view(1, 1, -1)  # [1, 1, 300]

            # 3. Invert mask: 1 = lost, 0 = OK
            inverted = 1.0 - time_mask

            # 4. Detect any loss in group via max_pool1d
            pooled = torch.nn.functional.max_pool1d(inverted, kernel_size=4, stride=4)  # [1, 1, 75]

            # 5. Invert again: 1 = OK, 0 = loss
            downsampled_mask = 1.0 - pooled.view(T_v)  # [75]

            # 6. Expand to video frame shape
            video_mask = downsampled_mask[:, None, None, None].expand(video.shape)  # [75, 1, 112, 112]

            masked_video = video * video_mask
            masked_videos.append(masked_video)

        return torch.stack(masked_videos)

    def __repr__(self) -> str:
        return (
            f"<Datasets: ["
            f'train: {len(self.train_files)}, '
            f'val: {len(self.val_files)}, '
            f'test: {len(self.test_files)}'
            f"]>"
        )

if __name__ == "__main__":

    import h5py
    import glob
    import numpy as np
    from tqdm import tqdm

    def process_single_chunk_filter(chunk_file):

        with h5py.File(chunk_file, 'r+') as h5f:
            video_keys = list(h5f.keys())
            for video_key in tqdm(video_keys, desc=f"{os.path.basename(chunk_file)}"):
                video_path = h5f.attrs.get(f"{video_key}/video_path", None)
                if video_path is not None:
                    try:
                        frames = h5f[f"{video_key}/frames"][:]

                        valid = np.any(frames != 0, axis=(1, 2, 3))
                        frames_valid = frames[valid]

                        if frames_valid.shape[0] == 0:
                            print(f"Corrupted Video: {video_path}")
                            del h5f[video_key]

                    except Exception as e:
                        print(f"General error for {video_key}: {e}")
                else:
                    print(f"Warning: No video path found for {video_key}")


    def update_h5(base_path, chunk_pattern="_chunk*.h5"):
        chunk_files = sorted(glob.glob(f"{base_path}{chunk_pattern}"))
        for chunk_file in chunk_files:
            process_single_chunk_filter(chunk_file)


    splits = {"dev"}  # "train"/"dev", "val", "test"

    for split in splits:
        feats_path = f'/home/ai/Projects/Mahsa/datasets/vox2_short/vox2_short_{split}_features'
        update_h5(feats_path)
