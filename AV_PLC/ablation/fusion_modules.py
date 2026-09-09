# fusion_modules.py
from typing import Dict, Type, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FusionBase", "FUSIONS", "register_fusion",
           "ConcatMLPFusion","Concat_Time", "GatedSumFusion", "FiLMFusion", "CrossAttnFusion"]

# ---- Base & registry ----
class FusionBase(nn.Module):
    """All fusions take (audio, video) shaped [B, T, D] and return fused [B, T, D_out].
       By default D_out == D (80), but allow flexibility via kwargs."""
    def __init__(self, dim: int = 80, **kwargs):
        super().__init__()
        self.dim = dim

    def forward(self, audio: torch.Tensor, video: torch.Tensor) -> torch.Tensor:  # [B,T,D], [B,T,D]
        raise NotImplementedError

FUSIONS: Dict[str, Type[FusionBase]] = {}

def register_fusion(name: str):
    def deco(cls: Type[FusionBase]):
        FUSIONS[name] = cls
        return cls
    return deco

# ---- Implementations ----

@register_fusion("concat_mlp")  # Conformer+MLP
class ConcatMLPFusion(FusionBase):
    def __init__(self, dim: int = 80, conformer_ctor=None, **kwargs):
        super().__init__(dim)
        hid = 4 * dim
        self.mel_cat = nn.Sequential(
            nn.Linear(2 * dim, hid),
            nn.ELU(),
            nn.Linear(hid, dim),
        )
        self.backbone = conformer_ctor(dim=dim, **kwargs) if conformer_ctor else nn.Identity()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, audio, video):
        x = torch.cat([audio, video], dim=-1)           # [B,T,2D]
        x = self.mel_cat(x)                              # [B,T,D]
        x = self.backbone(x)                             # [B,T,D]
        x = self.norm(x)
        x = self.proj(x)
        return x

@register_fusion("concat_time")
class Concat_Time(nn.Module):
    def __init__(self, dim=80,  conformer_ctor=None):
        super(Concat_Time, self).__init__()

        self.backbone = conformer_ctor(dim=dim) if conformer_ctor else nn.Identity()

        self.mel_proj = nn.Linear(80, 80)
        self.frame_proj = nn.Linear(80, 80)

        self.register_parameter('audio_mod_enc', nn.Parameter(torch.rand(dim)))
        self.register_parameter('video_mod_enc', nn.Parameter(torch.rand(dim)))

    def forward(self, audio, video, training=False):

        t_a = audio.size(1)
        a_emb = audio + self.audio_mod_enc
        v_emb = video + self.video_mod_enc

        fused = torch.cat([a_emb, v_emb], dim=1)  # [b, t, dim]
        output = self.backbone(fused)

        audio_output = output[:, :t_a, :]
        audio_output = self.mel_proj(audio_output)

        video_output = output[:, t_a:, :]
        video_output = self.frame_proj(video_output)

        return audio_output, video_output



@register_fusion("gated_sum")
class GatedSumFusion(FusionBase):
    """Learn a gate in [0,1] per token:  y = g * audio + (1-g) * video, then refine with MLP."""
    def __init__(self, dim: int = 80, **kwargs):
        super().__init__(dim)
        self.gate = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        self.refine = nn.Sequential(
            nn.Linear(dim, 2*dim),
            #nn.GELU(),
            nn.ELU(),
            nn.Linear(2*dim, dim),
        )
        #self.norm = nn.LayerNorm(dim)

    def forward(self, audio, video):
        g = self.gate(torch.cat([audio, video], dim=-1))  # [B,T,1]
        #x = g * audio + (1 - g) * video                   # [B,T,D]
        x = audio + g * (video - audio)  # [B,T,D]
        x = self.refine(x)
        return x #self.norm(x)

@register_fusion("film")
class FiLMFusion(FusionBase):
    """Use video to modulate audio via FiLM: a' = gamma(video) ⊙ audio + beta(video)."""
    def __init__(self, dim: int = 80, **kwargs):
        super().__init__(dim)
        self.affine = nn.Sequential(
            nn.Linear(dim, 2*dim),
            nn.GELU(),
            nn.Linear(2*dim, 2*dim)
        )
        self.post = nn.Sequential(
            nn.Linear(dim, 2*dim),
            nn.GELU(),
            nn.Linear(2*dim, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, audio, video):
        ab = self.affine(video)             # [B,T,2D]
        gamma, beta = ab.chunk(2, dim=-1)   # [B,T,D], [B,T,D]
        x = gamma * audio + beta            # [B,T,D]
        x = self.post(x)
        return self.norm(x)

@register_fusion("cross_attn")
class CrossAttnFusion(FusionBase):
    """Cross-attend audio<-video, then fuse residual."""
    def __init__(self, dim: int = 80, num_heads: int = 4, dropout: float = 0.1, **kwargs):
        super().__init__(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ffn  = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.GELU(),
            nn.Linear(4*dim, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, audio, video):
        # Query=audio, Key=Value=video  (all [B,T,D], batch_first=True)
        z, _ = self.attn(query=audio, key=video, value=video)  # [B,T,D]
        x = self.norm1(audio + z)
        x = self.norm2(x + self.ffn(x))
        return x
