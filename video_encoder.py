import torch
import torch.nn as nn

from conformer import Conformer
from resnet_ import ResNetModel


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
