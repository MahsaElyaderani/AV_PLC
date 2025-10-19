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
