import os
import random
import numpy as np
import torch

import torch.nn as nn
from conformer import Conformer

from audio_encoder import Audio_Encoder
from video_encoder import Video_Encoder
from av_dataloader import AVDataloader
from trainer import Trainer, setup_logging


class Fusion(nn.Module):
    def __init__(self, feat_dim=256, mel_dim=80, depth=2, dropout=0.1):
        super().__init__()
        self.norm_a = nn.LayerNorm(feat_dim)
        self.norm_v = nn.LayerNorm(feat_dim)

        self.mix = nn.Sequential(
            nn.Linear(2*feat_dim, 2*feat_dim),
            nn.GELU(),
            nn.Linear(2*feat_dim, feat_dim),
        )

        # temporal backbone runs at feat_dim (wide)
        self.temporal = Conformer(
            dim=feat_dim, depth=depth, dim_head=64, heads=4, ff_mult=4,
            conv_expansion_factor=2, conv_kernel_size=31,
            attn_dropout=dropout, ff_dropout=dropout, conv_dropout=dropout
        )

        self.out = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, mel_dim)   # final squeeze to mel bins
        )

    def forward(self, afeat=None, vfeat=None):
        """
        afeat, vfeat: [B,T,feat_dim] or None
        Returns mel_pred: [B,mel,T]
        """
        if (afeat is None) and (vfeat is None):
            raise ValueError("At least one modality must be provided.")

        if (afeat is not None) and (vfeat is not None):
            a = self.norm_a(afeat) #[b, t, feat_dim]
            v = self.norm_v(vfeat) #[b, t, feat_dim]
            x = self.mix(torch.cat([a, v,], dim=-1))  # [B,T,feat_dim]
        else:
            x = self.norm_a(afeat) if afeat is not None else self.norm_v(vfeat)

        x = self.temporal(x)                          # [B,T,feat_dim]
        mel = self.out(x).permute(0, 2, 1)            # [B,mel,T]

        return mel


class AV_PLC(nn.Module):
    def __init__(self, mel_dim=80, feat_dim=256, dropout=0.1,
                 video_depth=6, video_heads=4, audio_depth=4, audio_heads=4,
                 video_hidden_size=256, audio_hidden_size=256,):
        super().__init__()
        self.video_enc = Video_Encoder(conformer_block=video_depth, num_heads=video_heads,
                                       hidden_size=video_hidden_size, feat_dim=feat_dim)
        self.audio_enc = Audio_Encoder(conformer_block=audio_depth, num_heads=audio_heads,
                                       mel_emb=mel_dim,hidden_size=audio_hidden_size,feat_dim=feat_dim)
        self.fusion = Fusion(feat_dim=feat_dim, mel_dim=mel_dim, depth=2, dropout=dropout)

    def forward(self, dec_input=None, enc_input=None, spk_emb=None, audio_length=None):
        """
        dec_input: audio frames for audio encoder   [B,mel,T] or None
        enc_input: video frames for video encoder   [B,T,H,W,(C)] or None
        """
        amel = afeature = vmel = vfeature = None

        if dec_input is not None:
            amel, afeature = self.audio_enc(dec_input)            # [B,mel,T], [B,T,F]
        if enc_input is not None:
            vmel, vfeature = self.video_enc(enc_input, spk_emb)   # [B,mel,T], [B,T,F]

        # 3 modes in one:
        mel_pred = self.fusion(afeat=afeature if dec_input is not None else None,
                               vfeat=vfeature if enc_input is not None else None)  # [B,mel,T]

        return mel_pred, amel, vmel

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

    set_seeds(1337)
    batch_size = 32
    num_epochs = 100
    learning_rate = 1e-4
    stoi_flag, sc_flag = False, False
    pesq_flag, asr_flag, l2s_flag = True, True, True

    fusion_name = "concat_mlp"
    dataset_names = ['grid', 'lrs2', 'voxceleb2']
    plc_loss_rates = ['20', '30', '40', '50', '60']
    vocoder_path = '/home/ai/Projects/Mahsa/sources/AV_PLC/hifigan/checkpoints/model-best.pt'

    log_dir = 'logs'
    checkpoint_dir = 'checkpoints'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for dataset_name in dataset_names:
        model_name = (f"{'av' if l2s_flag else 'audio'}"
                      f"_plc_a0.05_v0.1{'_'+ fusion_name if l2s_flag else ''}"
                      f"{'_sc' if sc_flag else ''}"
                      f"{'_pesq_0.01' if pesq_flag else ''}"
                      f"{'_stoi_0.01' if stoi_flag else ''}"
                      f"{'_asr_0.1' if asr_flag else ''}"
                      f"({dataset_name})")
        logger = setup_logging(model_name, log_dir)
        logger.info(f"Using device: {device}; Fusion: {fusion_name}")

        video_depth = 6 if dataset_name == 'grid' else 6
        video_heads = 4 if dataset_name == 'grid' else 4

        audio_depth = 4 if dataset_name == 'grid' else 4
        audio_heads = 4 if dataset_name == 'grid' else 4

        video_hidden_size = 256 if dataset_name == 'grid' else 256
        audio_hidden_size = 256
        feat_dim = 256 #if dataset_name == 'grid' else 512

        model = AV_PLC(video_depth=video_depth, video_heads=video_heads,
                       audio_depth=audio_depth, audio_heads=audio_heads,
                       video_hidden_size=video_hidden_size,
                       audio_hidden_size=audio_hidden_size, feat_dim=feat_dim).to(device)
        logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

        av_dataloader = AVDataloader(mode='av' if l2s_flag else 'a',
                                     dataset_name=dataset_name,
                                     batch_size=batch_size, num_workers=4,
                                     dropout_modality=l2s_flag, video_aug=True,)
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
            vocoder_path=vocoder_path,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            mixed_precision=True,
            use_bf16=True,
            cosine_Tmax=num_epochs
        )

        start_epoch = trainer.load_checkpoint(load_best=True)
        trainer.train(num_epochs=num_epochs, start_epoch=start_epoch)

        for plc_loss_rate in plc_loss_rates:
            test_loader = av_dataloader.test_dataloader(plc_loss_rate)
            test_loss = trainer.evaluate(test_loader, plc_loss_rate)

