import torch
import torch.nn as nn

from conformer import Conformer
from AV_PLC.resnet_ import ResNetModel


class Video_Encoder(nn.Module):
    """Speaker-free AV_PLC visual encoder.

    25-Hz frame features are expanded to 100 Hz before the temporal Conformer:
        Conv3D -> ResNet18 -> Linear(256, 4*256) -> reshape x4 -> Conformer -> heads.
    ``spk_emb`` is accepted only for old caller compatibility and is ignored.
    """

    def __init__(self, conformer_block=6, num_heads=4,
                 spkr_vec=256, mel_dim=80, hidden_size=256,
                 dropout=0.1, feat_dim=256):
        super().__init__()
        if feat_dim != hidden_size:
            raise ValueError("This encoder expects feat_dim == hidden_size for the 4x reshape")

        self.frontend = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2),
                      padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.resnet = ResNetModel(
            layers=18, output_dim=hidden_size, pretrained=False, large_input=False
        )
        self.temporal_expand = nn.Linear(hidden_size, 4 * hidden_size)
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
            conv_dropout=dropout,
        )
        self.norm_layer = nn.LayerNorm(hidden_size)
        self.feat_proj = nn.Linear(hidden_size, feat_dim)
        self.mel_linear = nn.Linear(hidden_size, mel_dim)

    def forward(self, frame, spk_emb=None, audio_lengths=None):
        if frame.dim() == 4:  # [B,T,H,W]
            frame = frame.unsqueeze(-1)
        if frame.dim() != 5:
            raise ValueError(f"Expected frames [B,T,H,W] or [B,T,H,W,C], got {tuple(frame.shape)}")
        if frame.shape[-3:-1] != (88, 88):
            # frame is [B,T,H,W,C], so H/W are positions 2/3
            if frame.shape[2:4] != (88, 88):
                raise ValueError(f"Video_Encoder expects 88x88 crops, got {tuple(frame.shape)}")

        x = frame.permute(0, 4, 1, 2, 3).contiguous()  # [B,1,T,88,88]
        x = self.frontend(x)
        x = self.resnet(x)                             # [B,Tv,256]

        b, tv, d = x.shape
        x = self.temporal_expand(x).view(b, tv, 4, d).reshape(b, tv * 4, d)
        x = self.encoder(x)
        x = self.norm_layer(x)

        v_feat = self.feat_proj(x)                    # [B,4Tv,256]
        mel = self.mel_linear(x).permute(0, 2, 1)    # [B,80,4Tv]
        return mel, v_feat
