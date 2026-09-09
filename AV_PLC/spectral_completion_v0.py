"""Optional post-Mel spectral completion for AV_PLC.

The module is intentionally independent of the audio/video encoders and fusion
implementations.  It consumes the Mel prediction produced by whichever current
AV_PLC path is active and the observable STFT phase.  Temporal modeling is kept
outside this file so AV_PLC can instantiate the same Conformer implementation it
already uses in ``multimodal_decoder.py``.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class SpectralCompletion(nn.Module):
    """Encode coarse Mel + observed phase and decode Mel/phase residual outputs.

    Parameters
    ----------
    mel_dim:
        Number of Mel bins (80 in AV_PLC).
    phase_bins:
        One-sided STFT bins (257 for n_fft=512).
    feat_dim:
        Hidden size passed to the temporal Conformer in ``multimodal_decoder``.

    Notes
    -----
    Phase is represented by cosine/sine pairs.  The reliability channel is
    concatenated *after* cos/sin are computed so a missing angle never becomes
    the false observation cos(0)=1.
    """

    def __init__(self, mel_dim: int = 80, phase_bins: int = 257,
                 feat_dim: int = 256, dropout: float = 0.1,) -> None:
        super().__init__()
        self.mel_dim = int(mel_dim)
        self.phase_bins = int(phase_bins)
        self.feat_dim = int(feat_dim)
        phase_channels = 2 * self.phase_bins + 1

        # Separate projections: normalized Mel values and unit-circle phase.
        # Channels have different distributions and channel counts.

        self.mel_encoder = nn.Sequential(
            nn.Conv1d(self.mel_dim, self.feat_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(self.feat_dim, self.feat_dim, kernel_size=1),
        )
        self.phase_encoder = nn.Sequential(
            nn.Conv1d(phase_channels, self.feat_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(self.feat_dim, self.feat_dim, kernel_size=1),
        )
        self.merge = nn.Sequential(
            nn.Conv1d(2 * self.feat_dim, self.feat_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pre_temporal_norm = nn.LayerNorm(self.feat_dim)

        self.mel_head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, self.mel_dim),
        )
        self.phase_head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, 2 * self.phase_bins),
        )

        # Start as an identity refinement for Mel.  Enabling phase reconstruction
        # therefore does not randomly perturb the legacy Mel output at step zero.
        nn.init.zeros_(self.mel_head[-1].weight)
        nn.init.zeros_(self.mel_head[-1].bias)

    def encode(self, base_mel: torch.Tensor, phase_radians: torch.Tensor,
               phase_reliability: torch.Tensor,) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return temporal features plus observable phase components.

        Shapes
        ------
        base_mel:          [B, 80, T]
        phase_radians:     [B, 257, T]
        phase_reliability: [B, T], 1=observable, 0=must be predicted
        hidden:            [B, T, D]
        observed_cos/sin:  [B, 257, T]

        """

        if base_mel.dim() != 3 or base_mel.size(1) != self.mel_dim:
            raise ValueError(f"base_mel must be [B,{self.mel_dim},T], got {tuple(base_mel.shape)}")
        if phase_radians.dim() != 3 or phase_radians.size(1) != self.phase_bins:
            raise ValueError(f"phase_radians must be [B,{self.phase_bins},T], got {tuple(phase_radians.shape)}")
        if base_mel.size(0) != phase_radians.size(0) or base_mel.size(2) != phase_radians.size(2):
            raise ValueError("base_mel and phase_radians must share batch/time dimensions")
        if phase_reliability.shape != (base_mel.size(0), base_mel.size(2)):
            raise ValueError(f"phase_reliability must be [B,T], got{tuple(phase_reliability.shape)}")

        phase = phase_radians.to(device=base_mel.device, dtype=base_mel.dtype)
        reliability = phase_reliability.to(device=base_mel.device, dtype=base_mel.dtype).clamp(0.0, 1.0)

        # Compute circular representation, then mask it.
        observed_cos = torch.cos(phase) * reliability.unsqueeze(1)
        observed_sin = torch.sin(phase) * reliability.unsqueeze(1)
        phase_input = torch.cat([observed_cos, observed_sin, reliability.unsqueeze(1)], dim=1)

        mel_feat = self.mel_encoder(base_mel)
        phase_feat = self.phase_encoder(phase_input)
        hidden = self.merge(torch.cat([mel_feat, phase_feat], dim=1)).transpose(1, 2)
        hidden = self.pre_temporal_norm(hidden)

        return hidden, observed_cos, observed_sin, reliability

    def decode(self, hidden: torch.Tensor, eps: float = 1e-8
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        """
            Decode Mel residual and real-valued phase outputs.

            Returns delta_mel, p_r, p_i, pred_cos, pred_sin.
            p_r/p_i : are unconstrained network outputs
            pred_cos/pred_sin are their unit-normalized circular representation.

        """

        if hidden.dim() != 3 or hidden.size(-1) != self.feat_dim:
            raise ValueError(f"hidden must be [B,T,{self.feat_dim}], got {tuple(hidden.shape)}")

        delta_mel = self.mel_head(hidden).transpose(1, 2)
        phase_raw = self.phase_head(hidden).transpose(1, 2)
        p_r, p_i = phase_raw.chunk(2, dim=1)
        norm = torch.sqrt(p_r.square() + p_i.square() + eps)
        pred_cos = p_r / norm
        pred_sin = p_i / norm

        return delta_mel, p_r, p_i, pred_cos, pred_sin

    @staticmethod
    def merge_observed_phase(observed_cos: torch.Tensor, observed_sin: torch.Tensor,
                             pred_cos: torch.Tensor, pred_sin: torch.Tensor,
                             phase_reliability: torch.Tensor, eps: float = 1e-8,) -> tuple[torch.Tensor, torch.Tensor]:

        """Keep observed phase and insert predictions only where unavailable."""

        r = phase_reliability.unsqueeze(1).to(pred_cos.dtype)
        final_cos = observed_cos + (1.0 - r) * pred_cos
        final_sin = observed_sin + (1.0 - r) * pred_sin
        norm = torch.sqrt(final_cos.square() + final_sin.square() + eps)

        return final_cos / norm, final_sin / norm
