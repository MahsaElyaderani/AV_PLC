import os
import torch
import torch.nn as nn

from conformer import Conformer

from av_l_dataloader import AVDataloader
from trainer import Trainer, setup_logging


class Audio_PLC(nn.Module):

    def __init__(self, mel_dim=80, hidden_size=512//2,
                 conformer_block=4, num_heads=4, dropout=0.1):
        super().__init__()

        self.audio_emb = nn.Linear(mel_dim, hidden_size)

        self.conformer = Conformer(
            dim=hidden_size,
            depth=conformer_block,
            dim_head=64,
            heads=num_heads,
            ff_mult=4,
            conv_expansion_factor=2,
            conv_kernel_size=31, #15
            attn_dropout=dropout,
            ff_dropout=dropout,
            conv_dropout=dropout
        )

        self.norm_layer = nn.LayerNorm(hidden_size)
        self.mel_proj = nn.Linear(hidden_size, mel_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.audio_emb(x)
        x = self.conformer(x)
        x = self.norm_layer(x)
        x = self.mel_proj(x)
        return x.permute(0, 2, 1)


if __name__ == "__main__":

    batch_size = 32
    num_epochs = 200
    learning_rate = 0.0001

    l2s_flags = False
    pesq_flags = [True]
    dataset_names = ['grid'] #['voxceleb2']
    plc_loss_rates = ['20', '30', '40', '50', '60', 'rand']


    log_dir = 'logs'
    checkpoint_dir = 'checkpoints'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dataset_name in dataset_names:
        for pesq_flag in pesq_flags:
                model_name = f"audio_plc{'_pesq_0.001' if pesq_flag else ''}({dataset_name})"
                logger = setup_logging(model_name, log_dir)
                logger.info(f"Using device: {device}")

                model = Audio_PLC()

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
                    mode='a',
                    l2s_loss=l2s_flags,
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
                start_epoch = trainer.load_checkpoint(load_best=False)
                trainer.train(num_epochs=num_epochs, start_epoch=start_epoch)

                for plc_loss_rate in plc_loss_rates:
                    test_loader = av_dataloader.test_dataloader(plc_loss_rate)
                    logger.info("Evaluating model on test set...")
                    test_loss = trainer.evaluate(test_loader, plc_loss_rate)
                    logger.info(f"Final test loss: {test_loss:.4f}")
