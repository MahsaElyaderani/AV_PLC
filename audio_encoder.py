import os
import numpy as np
import random
import torch
import torch.nn as nn

from conformer import Conformer

from av_dataloader import AVDataloader
from trainer import Trainer, setup_logging

class Audio_Encoder(nn.Module):
    def __init__(self, mel_emb=80, mel_dim=80, hidden_size=256,
                 conformer_block=4, num_heads=4, dropout=0.1,
                 feat_dim=256):

        super().__init__()

        self.audio_emb = nn.Linear(mel_emb, hidden_size)

        self.conformer = Conformer(
            dim=hidden_size, depth=conformer_block, dim_head=64,
            heads=num_heads, ff_mult=4, conv_expansion_factor=2,
            conv_kernel_size=31, attn_dropout=dropout,
            ff_dropout=dropout, conv_dropout=dropout
        )
        self.norm_layer = nn.LayerNorm(hidden_size)

        # mel head (for supervision)
        self.mel_proj = nn.Linear(hidden_size, mel_dim)

        # fusion feature head (wider, time-aligned)
        self.feat_proj = nn.Linear(hidden_size, feat_dim)

    def forward(self, x, audio_lengths=None):
        x = x.permute(0, 2, 1)          # [B,T,mel]
        x = self.audio_emb(x)
        x = self.conformer(x)
        x = self.norm_layer(x)          # [B,T,feat_dim]

        mel = self.mel_proj(x).permute(0, 2, 1)     # [B,mel,T]
        feat = self.feat_proj(x)                     # [B,T,feat_dim]
        return mel, feat


def set_seeds(seed: int = 1337, deterministic: bool = True):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Safe even on CPU-only; no-op if no CUDA devices
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    batch_size = 8
    num_epochs = 200
    learning_rate = 1e-4
    pesq_flag = False
    stoi_flag = False
    asr_flag = False
    pretrained_enc = False
    sc_flags = [False]
    l2s_flag = False
    dataset_names = ['grid']
    plc_loss_rates = ['20', '30', '40', '50', '60', 'rand']
    fusion_names = ["concat_mlp"]#,"gated_sum", "concat_time", "film", "cross_attn"]

    log_dir = 'logs'
    checkpoint_dir = 'checkpoints'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    set_seeds(1337)

    results = []  # will hold (fusion, test_loss_agg/metric)

    for dataset_name in dataset_names:
        for sc_flag in sc_flags:
            for fusion_name in fusion_names:
                model_name = (f"{'av' if l2s_flag else 'audio'}"
                              f"_plc{'_'+fusion_name if l2s_flag else ''}"
                              f"{'_pretraind' if pretrained_enc else ''}"
                              f"{'_sc' if sc_flag else ''}"
                              f"{'_pesq_0.01' if pesq_flag else ''}"
                              f"{'_stoi_0.01' if stoi_flag else ''}"
                              f"{'_asr_0.1' if asr_flag else ''}"
                              f"({dataset_name})")
                logger = setup_logging(model_name, log_dir)
                logger.info(f"Using device: {device}; Fusion: {fusion_name}")
                conformer_blocks = 4 if dataset_name == 'grid' else 8
                num_heads = 4 if dataset_name == 'grid' else 4

                model = Audio_Encoder(conformer_block=conformer_blocks, num_heads=num_heads).to(device)
                logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

                av_dataloader = AVDataloader(mode='av' if l2s_flag else 'a', dataset_name=dataset_name,
                                             batch_size=batch_size, num_workers=0,
                                             video_aug=False, dropout_modality=False)
                train_loader = av_dataloader.train_dataloader()
                val_loader = av_dataloader.val_dataloader()

                trainer = Trainer(
                    model=model,
                    mode='av' if l2s_flag else 'a',
                    drop_av=l2s_flag,
                    sc_loss=sc_flag,
                    pesq_loss=pesq_flag,
                    stoi_loss=stoi_flag,
                    asr_loss=asr_flag,
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

                start_epoch = trainer.load_checkpoint(load_best=False)
                trainer.train(num_epochs=num_epochs, start_epoch=start_epoch)

                for plc_loss_rate in plc_loss_rates:
                    test_loader = av_dataloader.test_dataloader(plc_loss_rate)
                    test_loss = trainer.evaluate(test_loader, plc_loss_rate)
                    logger.info(f"[{fusion_name}] PLC={plc_loss_rate} test: {test_loss:.4f}")
