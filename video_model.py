import os
import torch
import numpy as np
import torch.nn as nn

from conformer import Conformer

from resnet_ import ResNetModel
from av_dataloader_OOM import AVDataloader
#from trainer_OOM import Trainer, setup_logging
from trainer_cpu import Trainer, setup_logging


class Video_PLC(nn.Module):
    def __init__(self, spkr_vec=256, mel_dim=80,
                 conformer_block=6, hidden_size=512//2,
                 num_heads=4, dropout=0.1):
        super(Video_PLC, self).__init__()

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
            dim=hidden_size + spkr_vec,
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
        #self.act = nn.Tanh()
        self.mel_proj = nn.Linear(hidden_size + spkr_vec,
                                  4 * mel_dim)  # mel bins are 80 and the output of mel_proj was 160 in the original code

    def forward(self, frame, spk_emb):
        b, t_v, c, h, w = frame.shape
        x = frame.permute(0, 2, 1, 3, 4)  # [b, c=1, t_v, h, w]
        x = self.frontend(x)  # [b, c, t_v, h', w']
        x = self.resnet(x)  # [b, t_v, 512]

        spk_x = torch.cat([x, spk_emb.unsqueeze(1).repeat(1, x.size(1), 1)], dim=-1)
        spk_x = self.encoder(spk_x)  # x: (b, t_v, f)
        spk_x = self.norm_layer(spk_x)

        mel = self.mel_proj(spk_x)
        b, t_v, d = mel.shape
        mel = mel.reshape(b, t_v, d // 4, 4).transpose(-1, -2).reshape(b, t_v * 4, d // 4)
        #mel = self.act(mel)
        return mel.permute(0, 2, 1)


if __name__ == "__main__":

    batch_size = 8
    num_epochs = 200
    learning_rate = 0.0001

    conformer_blocks = 6 #4
    pesq_flag = False
    l2s_flags = [True]
    dataset_names = ['grid'] #['grid']
    plc_loss_rates = ['20', '30', '40', '50', '60', 'rand']


    log_dir = 'logs'
    checkpoint_dir = 'checkpoints'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dataset_name in dataset_names:
        for l2s_flag in l2s_flags:
                model_name = f"video_plc_reg_aug{'_sc' if l2s_flag else ''}{'_pesq' if pesq_flag else ''}({dataset_name})"
                logger = setup_logging(model_name, log_dir)
                logger.info(f"Using device: {device}")

                model = Video_PLC(conformer_block=conformer_blocks)

                logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
                #model = torch.compile(model)

                logger.info("Initializing dataloaders...")
                av_dataloader = AVDataloader(mode='v', dataset_name=dataset_name,
                                            batch_size=batch_size, num_workers=4)

                train_loader = av_dataloader.train_dataloader()
                val_loader = av_dataloader.val_dataloader()

                #vocoder_path = '/home/ai/Projects/Mahsa/sources/stable_diffusion/dataset/hifigan/checkpoints/seg_len_4096/model-best.pt'

                trainer = Trainer(
                    model=model,
                    mode='v',
                    l2s_loss=l2s_flag,
                    pesq_loss=pesq_flag,
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

                for plc_loss_rate in plc_loss_rates:
                    test_loader = av_dataloader.test_dataloader(plc_loss_rate)
                    logger.info("Evaluating model on test set...")
                    test_loss = trainer.evaluate(test_loader, plc_loss_rate)
                    logger.info(f"Final test loss: {test_loss:.4f}")
