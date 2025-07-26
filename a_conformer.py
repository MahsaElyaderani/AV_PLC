import os
import torch
import torchaudio
import numpy as np
import torch.nn as nn

import torchvision.models as models
from  torchvision import transforms
import torch.nn.functional as F
import torchaudio.transforms as T
import torchaudio.functional as FA
from torchaudio.models import Conformer
from stable_diffusion.models.trainer import Trainer, setup_logging
from stable_diffusion.dataset.av_dataloader import AVDataloader


class PLC_Conformer(nn.Module):
    def __init__(self, audio_dim=257, hidden_dim=512, num_layers=4):
        super().__init__()
        self.audio_emb = nn.Sequential(nn.Linear(audio_dim, hidden_dim),
                                       nn.ELU(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ELU())
        self.conformer = Conformer(
            input_dim=hidden_dim,
            num_heads=4,
            ffn_dim=hidden_dim * 4,
            num_layers=num_layers,
            depthwise_conv_kernel_size=15,
            dropout=0.1,
        )

        layers = [nn.Linear(hidden_dim, hidden_dim * 4),
                  nn.ELU(),
                  nn.Linear(hidden_dim * 4, hidden_dim * 4),
                  nn.ELU(),
                  nn.Linear(hidden_dim * 4, audio_dim),
                  nn.ELU()]
        self.fc_out = nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        lengths = torch.full((x.size(0),), x.size(1), dtype=torch.long, device=x.device)
        x = self.audio_emb(x)
        x, _ = self.conformer(x, lengths)
        x = self.fc_out(x)
        return x.permute(0, 2, 1)


if __name__ == "__main__":

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


    hidden_size = 256
    num_layers = 3
    dropout = 0.1
    batch_size = 32
    num_epochs = 200
    learning_rate = 0.0001

    bi_dir_flag = True
    dataset_names = ['voxceleb2']  # ,'grid']
    asr_flags = [False]  # , True]
    pesq_flags = True
    freq_emph_flags = [False]  # , True]
    plc_loss_rates = ['20', '30', '40', '50', '60', 'rand']

    log_dir = 'logs'
    checkpoint_dir = 'checkpoints'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dataset_name in dataset_names:
        for asr_flag in asr_flags:
            for freq_emph_flag in freq_emph_flags:
                model_name = f"plc_conformer{'_asr' if asr_flag else ''}{'_pesq' if pesq_flags else ''}({dataset_name})"
                logger = setup_logging(model_name, log_dir)
                logger.info(f"Using device: {device}")

                model = PLC_Conformer()
                logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

                logger.info("Initializing dataloaders...")

                av_dataloader = AVDataloader(mode='a', dataset_name=dataset_name,
                                             batch_size=batch_size, num_workers=0)
                train_loader = av_dataloader.train_dataloader()
                val_loader = av_dataloader.val_dataloader()

                #vocoder_path = '/home/ai/Projects/Mahsa/sources/stable_diffusion/dataset/hifi_gan/cp_hifigan'

                trainer = Trainer(
                    model=model,
                    mode='a',
                    asr_loss=asr_flag,
                    pesq_loss=pesq_flags,
                    freq_emph=freq_emph_flag,
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
                start_epoch = trainer.load_checkpoint(load_best=False)
                trainer.train(num_epochs=num_epochs, start_epoch=start_epoch)

                for plc_loss_rate in plc_loss_rates:
                    test_loader = av_dataloader.test_dataloader(plc_loss_rate)
                    logger.info("Evaluating model on test set...")
                    test_loss = trainer.evaluate(test_loader, plc_loss_rate)
                    logger.info(f"Final test loss: {test_loss:.4f}")