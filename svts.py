import os
import math
import h5py
import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from conformer import Conformer
from torch.utils.data import Dataset, DataLoader, Subset

#from stable_diffusion.models.l2s.conformer.encoder import ConformerEncoder
from stable_diffusion.models.modules.resnet import ResNetModel
from stable_diffusion.models.l2s.l2s_trainer import Trainer, setup_logging
from stable_diffusion.models.l2s.transforms import AudioTransform, VideoTransform


class VDataloader:
    def __init__(self, dataset_name, batch_size, num_workers,
                 train_subset=None, val_subset=None, test_subset=None):

        assert dataset_name in ['grid', 'voxceleb2'], f"Invalid dataset_name: {dataset_name}"

        if dataset_name == 'grid':
            base_path = '/home/ai/Projects/Mahsa/datasets/grid/'
            self.train_files = base_path + 'grid_train_features_chunk*.h5'
            self.val_files = base_path + 'grid_val_features_chunk*.h5'
            self.test_files = base_path + 'grid_test_features_chunk*.h5'

        elif dataset_name == 'voxceleb2':
            base_path = '/home/ai/Projects/Mahsa/datasets/vox2_short/'
            self.train_files = base_path + 'vox2_short_dev_features_chunk*.h5'
            self.val_files = base_path + 'vox2_short_val_features_chunk*.h5'
            self.test_files = base_path + 'vox2_short_test_features_chunk*.h5'

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
        train_dataset = VDataset(self.train_files)
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
        val_dataset = VDataset(self.val_files)
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

    def test_dataloader(self):

        test_dataset = VDataset(self.test_files)
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


class VDataset(Dataset):

    def __init__(self, base_path, chunk_pattern="_chunk*.h5"):

        self.video_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize([112, 112]),
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=0.421, std=0.165),
        ])
        #stats = torch.load('grid_stats.pt')
        #self.mel_mean = stats['mean']
        #self.mel_std = stats['std']

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
        #chunk_file = self.chunk_files[chunk_idx]
        h5f = self._get_h5_file(chunk_idx)

        #units = h5f[f"{video_key}/soft_units"][:]
        units = h5f[f"{video_key}/units"][:]
        mel_spec = h5f[f"{video_key}/spec"][:]
        frames = h5f[f"{video_key}/frames"][:50]
        spk_emb = h5f[f"{video_key}/spkr_embed"][:]
        video_path = h5f.attrs.get(f"{video_key}/video_path", None)

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
        #mel_spec = (torch.tensor(mel_spec) - self.mel_mean) / self.mel_std
        mel_spec = torch.nn.functional.layer_norm(torch.tensor(mel_spec),
                                        torch.tensor(mel_spec).shape, eps=0)

        #return frames, spk_emb, mel_spec, units
        return frames_tensor, torch.tensor(spk_emb), mel_spec, torch.tensor(units)


class lip2speech(nn.Module):
    def __init__(self, asr_loss, conformer_block=6,
                 hidden_size=512//2, num_heads=4, dropout=0.1):
        super(lip2speech, self).__init__()

        self.asr_loss = asr_loss
        self.video_transform = VideoTransform()

        self.frontend = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )

        self.resnet = ResNetModel(
            layers=18,
            output_dim=hidden_size,
            pretrained=False,
            large_input=False
        )

        # self.encoder = Conformer(
        #     input_dim=hidden_size+256,
        #     num_heads=num_heads,
        #     ffn_dim=hidden_size * 4,
        #     num_layers=conformer_block,
        #     depthwise_conv_kernel_size=31,
        #     dropout=dropout,
        # )
        self.encoder = Conformer(
            dim=hidden_size+256,
            depth=conformer_block,  # 12 blocks
            dim_head=64,
            heads=num_heads,
            ff_mult=4,
            conv_expansion_factor=2,
            conv_kernel_size=31,
            attn_dropout=0.,
            ff_dropout=0.,
            conv_dropout=0.
        )
        self.norm_layer = nn.LayerNorm(hidden_size+256)
        self.mel_proj = nn.Linear(hidden_size+256, 4*80)#mel bins are 80 and the output of mel_proj was 160 in the original code

    def forward(self, frame, spk_emb):

        b, t_v, c, h, w = frame.shape
        x = frame.permute(0, 2, 1, 3, 4) #[b, c=1, t_v, h, w]

        x = self.frontend(x) #[b, c, t_v, h', w']
        x = self.resnet(x)  # [b, t_v, 512]

        spk_x = torch.cat([x, spk_emb.unsqueeze(1).repeat(1, x.size(1), 1)], dim=-1)
        #lengths = torch.full((spk_x.size(0),), spk_x.size(1), dtype=torch.long, device=x.device)
        #spk_x, _ = self.encoder(spk_x, lengths) #x: (b, t_v, f)
        spk_x = self.encoder(spk_x)  # x: (b, t_v, f)
        spk_x = self.norm_layer(spk_x)

        mel = self.mel_proj(spk_x)
        b, t_v, d = mel.shape
        mel = mel.reshape(b, t_v, d//4, 4).transpose(-1, -2).reshape(b, t_v*4, d//4)

        return mel.permute(0, 2, 1)


if __name__ == "__main__":

    # seed = 42
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)

    batch_size = 32
    num_epochs = 200
    learning_rate = 0.001

    asr_flags = [False]
    dataset_names = ['grid'] #['voxceleb2']
    plc_loss_rates = ['20', '30', '40', '50', '60', 'rand']


    log_dir = 'logs'
    checkpoint_dir = 'checkpoints'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dataset_name in dataset_names:
        for asr_flag in asr_flags:
                model_name = f"svts{'_asr' if asr_flag else ''}({dataset_name})"
                logger = setup_logging(model_name, log_dir)
                logger.info(f"Using device: {device}")

                model = lip2speech(asr_loss=asr_flag)
                logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
                #model = torch.compile(model)

                logger.info("Initializing dataloaders...")
                v_dataloader = VDataloader(dataset_name=dataset_name,
                                             batch_size=batch_size, num_workers=8)
                train_loader = v_dataloader.train_dataloader()
                val_loader = v_dataloader.val_dataloader()

                # batch = next(iter(train_loader))
                # frames, embd, specs_gt, units_gt = batch

                # model.eval()
                # print(frame.shape)
                # with torch.no_grad():
                #     spec, units = model(frame)
                # print("spec shape:", spec.shape)
                # print("units shape: ", units.shape)
                #vocoder_path = '/home/ai/Projects/Mahsa/sources/stable_diffusion/dataset/hifigan/checkpoints/seg_len_4096/model-best.pt'

                trainer = Trainer(
                    model=model,
                    mode='v',
                    asr_loss=asr_flag,
                    model_name=model_name,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    learning_rate=learning_rate,
                    device=device,
                    vocoder_path=None, #vocoder_path,
                    checkpoint_dir=checkpoint_dir,
                    log_dir=log_dir
                )

                logger.info(f"Starting training for {num_epochs} epochs...")
                start_epoch = trainer.load_checkpoint(load_best=True)
                trainer.train(num_epochs=num_epochs, start_epoch=start_epoch)

                test_loader = v_dataloader.test_dataloader()
                logger.info("Evaluating model on test set...")
                test_loss = trainer.evaluate(test_loader)
                logger.info(f"Final test loss: {test_loss:.4f}")
