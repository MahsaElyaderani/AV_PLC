
# Portable root configuration
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from evaluations.runtime_config import project_log_dir, project_checkpoint_dir, VOCODER_PATH
import os
import numpy as np
import random

import torch
import torch.nn as nn
from conformer import Conformer

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

# if __name__ == "__main__":
#     import argparse
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--mode", choices=["train", "test"], required=True,
#                         help="Run mode: 'train' or 'test'")
#     parser.add_argument("--plc_loss_rates", nargs="+", default=['10', '20', '30', '40', '50', '60', '70', '80', '90'],
#                         help="PLC loss rates for test mode")
#     args = parser.parse_args()
#
#     batch_size = 16
#     num_epochs = 100
#     learning_rate = 1e-4
#     pesq_flag = False
#     stoi_flag = False
#     asr_flag = False
#     pretrained_enc = False
#     sc_flags = [False]
#     l2s_flag = False
#     dataset_names = ['voxceleb2']
#     plc_loss_rates = args.plc_loss_rates
#     fusion_names = ["concat_mlp"]
#
#     log_dir = project_log_dir('AV_PLC')
#     checkpoint_dir = project_checkpoint_dir('AV_PLC')
#     os.makedirs(log_dir, exist_ok=True)
#     os.makedirs(checkpoint_dir, exist_ok=True)
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     set_seeds(1337)
#
#     for dataset_name in dataset_names:
#         for sc_flag in sc_flags:
#             for fusion_name in fusion_names:
#                 model_name = (f"{'av' if l2s_flag else 'audio_bursty'}"
#                               f"_plc{'_'+fusion_name if l2s_flag else ''}"
#                               f"{'_pretraind' if pretrained_enc else ''}"
#                               f"{'_sc' if sc_flag else ''}"
#                               f"{'_pesq_0.01' if pesq_flag else ''}"
#                               f"{'_stoi_0.01' if stoi_flag else ''}"
#                               f"{'_asr_0.1' if asr_flag else ''}"
#                               f"({dataset_name})")
#                 logger = setup_logging(model_name, log_dir)
#                 logger.info(f"Using device: {device}; Fusion: {fusion_name}; Mode: {args.mode}")
#
#                 conformer_blocks = 4 if dataset_name == 'grid' else 8
#                 num_heads = 4 if dataset_name == 'grid' else 4
#
#                 model = Audio_Encoder(conformer_block=conformer_blocks, num_heads=num_heads).to(device)
#                 logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
#
#                 av_dataloader = AVDataloader(
#                     mode='av' if l2s_flag else 'a',
#                     dataset_name=dataset_name,
#                     batch_size=batch_size,
#                     num_workers=0,
#                     video_aug=False,
#                     dropout_modality=False
#                 )
#
#                 trainer = Trainer(
#                     model=model,
#                     mode='av' if l2s_flag else 'a',
#                     drop_av=l2s_flag,
#                     sc_loss=sc_flag,
#                     pesq_loss=pesq_flag,
#                     stoi_loss=stoi_flag,
#                     asr_loss=asr_flag,
#                     enc_loss=False,
#                     model_name=model_name,
#                     train_loader=None,
#                     val_loader=None,
#                     learning_rate=learning_rate,
#                     vocoder_path=None,
#                     checkpoint_dir=checkpoint_dir,
#                     log_dir=log_dir,
#                     mixed_precision=True,
#                     use_bf16=True,
#                     cosine_Tmax=num_epochs
#                 )
#
#                 if args.mode == "train":
#                     train_loader = av_dataloader.train_dataloader()
#                     val_loader = av_dataloader.val_dataloader()
#                     trainer.train_loader = train_loader
#                     trainer.val_loader = val_loader
#                     start_epoch = trainer.load_checkpoint(load_best=True)
#                     trainer.train(num_epochs=num_epochs, start_epoch=start_epoch)
#
#                 elif args.mode == "test":
#                     trainer.load_checkpoint(load_best=True)
#                     # Build GT cache once using any loss rate's loader (paths are identical across rates)
#                     reference_loader = av_dataloader.test_dataloader(plc_loss_rates[0], seed=42)
#                     gt_audio_cache, gt_ref_text_cache = trainer.build_gt_cache(reference_loader)
#                     logger.info(f"GT cache built: {len(gt_audio_cache)} samples")
#
#                     for plc_loss_rate in plc_loss_rates:
#                         test_loader = av_dataloader.test_dataloader(plc_loss_rate, seed=42)
#                         trainer.evaluate(test_loader, plc_loss_rate,
#                                                      gt_audio_cache=gt_audio_cache,
#                                                      gt_ref_text_cache=gt_ref_text_cache,
#                                                      save_output=False, save_metrics=True)


