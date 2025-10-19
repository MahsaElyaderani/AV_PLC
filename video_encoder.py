import os
import torch
import random
import numpy as np
import torch.nn as nn

from conformer import Conformer
from resnet_ import ResNetModel
from av_dataloader import AVDataloader
from trainer import Trainer, setup_logging


class Video_Encoder(nn.Module):
    def __init__(self, conformer_block=6, num_heads=4,
                 spkr_vec=256, mel_dim=80, hidden_size=256,
                 dropout=0.1, feat_dim=256):

        super(Video_Encoder, self).__init__()

        self.frontend = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7),
                      stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
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

        self.encoder = Conformer(
            dim=hidden_size,
            depth=conformer_block,
            dim_head=64,
            heads=num_heads,
            ff_mult=4,
            conv_expansion_factor=2,
            conv_kernel_size=31,
            attn_dropout=dropout,
            ff_dropout=dropout,
            conv_dropout=dropout
        )
        self.norm_layer = nn.LayerNorm(hidden_size + spkr_vec)

        self.mel_conv = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size + spkr_vec,
                      out_channels=4 * feat_dim, kernel_size=3, stride=1, padding=1),
            nn.Dropout(dropout),
            nn.ELU(),

            nn.Conv1d(in_channels=4 * feat_dim, out_channels=4 * feat_dim,
                      kernel_size=3, stride=1, padding=1),
            nn.Dropout(dropout),
            nn.ELU(),

            nn.Conv1d(in_channels=4 * feat_dim, out_channels=4 * feat_dim,
                      kernel_size=3, stride=1, padding=1),
            nn.Dropout(dropout),
            nn.ELU(),
        )
        self.feat_proj = nn.Linear(4 * feat_dim, 4 * feat_dim)
        self.mel_linear = nn.Linear(4 * feat_dim, 4 * mel_dim)

    def forward(self, frame, spk_emb, audio_lengths=None):
        if frame.dim() == 4:  # [B, T, H, W]
            frame = frame.unsqueeze(-1)  # [B,T,H,W,1]
        frame = frame.permute(0, 1, 4, 2, 3).contiguous() # [B,T,C,H,W] in [0,1]

        # resize all frames at once
        B, T, C, H, W = frame.shape
        if H != 112 or W != 112:
            x = torch.nn.functional.interpolate(frame.flatten(0, 1),
                                           size=(112, 112),
                                           mode='bilinear',
                                           align_corners=False) \
            .view(B, C, T, 112, 112)
        else:
            x = frame.permute(0, 2, 1, 3, 4)  # [b, c=1, t_v, h, w]

        x = self.frontend(x)  # [b, c, t_v, h', w']
        x = self.resnet(x)  # [b, t_v, 256]
        x = self.encoder(x)

        spk = spk_emb[:, None, :].expand(-1, x.size(1), -1)  # [B, T, Dspk]
        spk_x = torch.cat([x, spk], dim=-1)
        spk_x = self.norm_layer(spk_x) # [b, t, feat=512]
        spk_x = self.mel_conv(spk_x.permute(0, 2, 1)).permute(0, 2, 1)

        v_feat = self.feat_proj(spk_x)
        b, t_v, d = v_feat.shape
        v_feat = v_feat.reshape(b, t_v, d // 4, 4).transpose(-1, -2).reshape(b, t_v * 4, d // 4)

        mel = self.mel_linear(spk_x)
        b, t_v, d = mel.shape
        mel = mel.reshape(b, t_v, d // 4, 4).transpose(-1, -2).reshape(b, t_v * 4, d // 4)
        mel = mel.permute(0, 2, 1)  # [B,mel,T]

        return mel, v_feat



if __name__ == "__main__":


    batch_size = 8
    num_epochs = 200
    learning_rate = 0.0001

    pesq_flag = False
    l2s_flags = [False]
    dataset_names = ['grid'] #['grid', 'voxceleb2']
    plc_loss_rates = ['20', '30', '40', '50', '60', 'rand']


    log_dir = 'logs'
    checkpoint_dir = 'checkpoints'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dataset_name in dataset_names:
        for l2s_flag in l2s_flags:
                model_name = f"video_v2_reg_aug{'_sc' if l2s_flag else ''}{'_pesq' if pesq_flag else ''}({dataset_name})"
                logger = setup_logging(model_name, log_dir)
                logger.info(f"Using device: {device}")

                conformer_blocks = 6 if dataset_name == 'grid' else 8
                attn_heads = 4 if dataset_name == 'grid' else 8
                model = Video_Encoder(conformer_block=conformer_blocks, num_heads=attn_heads)

                logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
                #model = torch.compile(model)

                logger.info("Initializing dataloaders...")
                av_dataloader = AVDataloader(mode='v', dataset_name=dataset_name,
                                            batch_size=batch_size, num_workers=2)

                train_loader = av_dataloader.train_dataloader()
                val_loader = av_dataloader.val_dataloader()

                #vocoder_path = '/home/ai/Projects/Mahsa/sources/stable_diffusion/dataset/hifigan/checkpoints/seg_len_4096/model-best.pt'
                trainer = Trainer(
                    model=model,
                    mode='v',
                    drop_av=False,
                    sc_loss=False,
                    pesq_loss=pesq_flag,
                    stoi_loss=False,
                    asr_loss=False,
                    model_name=model_name,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    learning_rate=learning_rate,
                    device=device,
                    vocoder_path=None,
                    checkpoint_dir=checkpoint_dir,
                    log_dir=log_dir,
                    mixed_precision=True,
                    use_bf16=True,
                    cosine_Tmax=50
                )

                logger.info(f"Starting training for {num_epochs} epochs...")
                start_epoch = trainer.load_checkpoint(load_best=False)
                trainer.train(num_epochs=num_epochs, start_epoch=start_epoch)

                for plc_loss_rate in plc_loss_rates:
                    test_loader = av_dataloader.test_dataloader(plc_loss_rate)
                    logger.info("Evaluating model on test set...")
                    test_loss = trainer.evaluate(test_loader, plc_loss_rate)
                    logger.info(f"Final test loss: {test_loss:.4f}")
