"""Unified latent-to-spectrum completion heads for AV_PLC.

The decoder operates directly on the 256-D latent representation selected by
``multimodal_decoder.AV_PLC``:

    audio-only -> audio encoder feature A
    video-only -> video encoder feature V
    AV         -> fusion feature F_av

A/V/F_av share dimensionality but not statistics, so each source has a small
source-specific adapter before the common temporal decoder.  When learned phase
reconstruction is enabled, one phase encoder consumes
``[R_phi*cos(phi), R_phi*sin(phi), R_phi]`` and is merged with the adapted latent
*before* the shared ``spectral_temporal`` Conformer.

This module intentionally contains no Conformer.  Temporal modeling remains in
``multimodal_decoder.py`` so there is exactly one post-encoder/fusion temporal
reconstruction stack.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpectralCompletion(nn.Module):
    """Adapt latent speech features and decode absolute Mel + optional phase.

    Source ids
    ----------
    0 : audio-only feature ``A``
    1 : video-only feature ``V``
    2 : AV fusion feature ``F_av``
    """

    SOURCE_AUDIO = 0
    SOURCE_VIDEO = 1
    SOURCE_FUSED = 2

    def __init__(self, mel_dim: int = 80, phase_bins: int = 257,
                 feat_dim: int = 256, dropout: float = 0.1,
                 phase_reconstruction: bool = False,) -> None:

        super().__init__()
        self.mel_dim = int(mel_dim)
        self.phase_bins = int(phase_bins)
        self.feat_dim = int(feat_dim)
        self.phase_reconstruction = bool(phase_reconstruction)

        # A, V and F_av are all [B,T,D], but their semantics/statistics differ.
        # Keep these adapters deliberately light so the common decoder remains
        # responsible for temporal reconstruction rather than re-fusing inputs.

        def _source_adapter():
            return nn.Sequential(nn.LayerNorm(self.feat_dim),
                                 nn.Linear(self.feat_dim, self.feat_dim),
                                 nn.GELU(),
                                 nn.Dropout(dropout),)

        self.audio_adapter = _source_adapter()
        self.video_adapter = _source_adapter()
        self.fused_adapter = _source_adapter()

        # Construct all phase-independent decoder components before optional
        # phase modules.  With the same global seed, no-phase and phase-enabled
        # runs therefore receive identical initialization for the shared source
        # adapters and Mel head.
        self.pre_temporal_norm = nn.LayerNorm(self.feat_dim)
        self.mel_head = nn.Sequential(nn.LayerNorm(self.feat_dim),
                                      nn.Linear(self.feat_dim, self.mel_dim),)

        if self.phase_reconstruction:
            phase_channels = 2 * self.phase_bins + 1
            # Do not let optional phase-module initialization advance the global
            # CPU RNG stream.  This keeps all phase-independent modules that are
            # constructed later (notably the selected fusion module) identically
            # initialized in phase/no-phase ablations under the same seed.
            with torch.random.fork_rng(devices=[]):
                # One phase encoder: cosine/sine form one circular quantity; R_phi
                # is included here only to mark observed versus missing phase.
                self.phase_encoder = nn.Sequential(nn.Conv1d(phase_channels, self.feat_dim, kernel_size=3, padding=1),
                                                   nn.GELU(),
                                                   nn.Dropout(dropout),
                                                   nn.Conv1d(self.feat_dim, self.feat_dim, kernel_size=1),)

                self.merge = nn.Sequential(nn.Linear(2 * self.feat_dim, self.feat_dim),
                                           nn.GELU(),
                                           nn.Dropout(dropout),)

                self.phase_head = nn.Sequential(nn.LayerNorm(self.feat_dim),
                                                nn.Linear(self.feat_dim, 2 * self.phase_bins),)
        else:
            self.phase_encoder = None
            self.merge = None
            self.phase_head = None

    def encode_source(self, latent: torch.Tensor, source_ids: torch.Tensor,) -> torch.Tensor:
        """Map A/V/F_av to a common hidden space.

        Parameters
        ----------
        latent:
            Selected latent sequence [B,T,D].
        source_ids:
            Integer tensor [B] using SOURCE_AUDIO/VIDEO/FUSED.
        """
        if latent.dim() != 3 or latent.size(-1) != self.feat_dim:
            raise ValueError(f"latent must be [B,T,{self.feat_dim}], got {tuple(latent.shape)}")
        if source_ids.shape != (latent.size(0),):
            raise ValueError(f"source_ids must be [B]=({latent.size(0)},), got {tuple(source_ids.shape)}")
        source_ids = source_ids.to(device=latent.device, dtype=torch.long)
        if ((source_ids < self.SOURCE_AUDIO) | (source_ids > self.SOURCE_FUSED)).any():
            raise ValueError("source_ids must contain only 0=audio, 1=video, 2=fused")

        # Compute the three lightweight projections then gather the selected one.
        # Non-selected branches receive no gradient through the gather operation.
        adapted = torch.stack([self.audio_adapter(latent),
                               self.video_adapter(latent),
                               self.fused_adapter(latent),], dim=1,)  # [B,3,T,D]
        gather_index = source_ids.view(-1, 1, 1, 1).expand(-1, 1, latent.size(1), latent.size(2))

        return adapted.gather(1, gather_index).squeeze(1)

    def encode_phase(self, phase_radians: torch.Tensor,
                     phase_reliability: torch.Tensor,
                     reference: torch.Tensor,
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode observable phase without creating false phase in missing frames.

        ``R_phi`` is used only inside the phase representation:

            P_obs = [R_phi*cos(phi), R_phi*sin(phi), R_phi]

        It is intentionally *not* supplied again as a separate Merge input.
        """
        if not self.phase_reconstruction or self.phase_encoder is None:
            raise RuntimeError("encode_phase requires phase_reconstruction=True")
        if phase_radians.dim() != 3 or phase_radians.size(1) != self.phase_bins:
            raise ValueError(f"phase_radians must be [B,{self.phase_bins},T], got {tuple(phase_radians.shape)}")
        if phase_radians.size(0) != reference.size(0) or phase_radians.size(2) != reference.size(1):
            raise ValueError("phase_radians must share batch/time dimensions with latent features")
        if phase_reliability.shape != (reference.size(0), reference.size(1)):
            raise ValueError(f"phase_reliability must be [B,T], got {tuple(phase_reliability.shape)}")

        phase = phase_radians.to(device=reference.device, dtype=reference.dtype)
        reliability = phase_reliability.to(device=reference.device, dtype=reference.dtype).clamp(0.0, 1.0)

        # Compute cos/sin first, then mask.  Masking the angle itself would make
        # missing phase look like the false observation cos(0)=1, sin(0)=0.
        observed_cos = torch.cos(phase) * reliability.unsqueeze(1)
        observed_sin = torch.sin(phase) * reliability.unsqueeze(1)
        phase_input = torch.cat([observed_cos, observed_sin, reliability.unsqueeze(1)], dim=1)
        phase_hidden = self.phase_encoder(phase_input).transpose(1, 2)

        return phase_hidden, observed_cos, observed_sin, reliability

    def encode(self, latent: torch.Tensor, source_ids: torch.Tensor,
               phase_radians: torch.Tensor | None = None,
               phase_reliability: torch.Tensor | None = None,
               ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None,]:

        """Prepare the common Conformer input from latent + optional phase."""
        source_hidden = self.encode_source(latent, source_ids)

        if not self.phase_reconstruction:
            hidden = self.pre_temporal_norm(source_hidden)
            return hidden, None, None, None

        if phase_radians is None or phase_reliability is None:
            raise ValueError("phase_radians and phase_reliability are required when phase_reconstruction=True")

        phase_hidden, observed_cos, observed_sin, reliability = self.encode_phase(phase_radians,
                                                                                  phase_reliability,
                                                                                  source_hidden)
        hidden = self.merge(torch.cat([source_hidden, phase_hidden], dim=-1))
        hidden = self.pre_temporal_norm(hidden)

        return hidden, observed_cos, observed_sin, reliability

    def decode(self, hidden: torch.Tensor, eps: float = 1e-8,) -> tuple[torch.Tensor, torch.Tensor | None,
        torch.Tensor | None, torch.Tensor | None, torch.Tensor | None,]:

        """Decode absolute Mel and, when enabled, unit-circle phase."""
        if hidden.dim() != 3 or hidden.size(-1) != self.feat_dim:
            raise ValueError(f"hidden must be [B,T,{self.feat_dim}], got {tuple(hidden.shape)}")

        predicted_mel = self.mel_head(hidden).transpose(1, 2)

        if not self.phase_reconstruction or self.phase_head is None:
            return predicted_mel, None, None, None, None

        phase_raw = self.phase_head(hidden).transpose(1, 2)
        p_r, p_i = phase_raw.chunk(2, dim=1)
        norm = torch.sqrt(p_r.square() + p_i.square() + eps)
        pred_cos = p_r / norm
        pred_sin = p_i / norm

        return predicted_mel, p_r, p_i, pred_cos, pred_sin

    @staticmethod
    def merge_observed_phase(observed_cos: torch.Tensor, observed_sin: torch.Tensor,
                             pred_cos: torch.Tensor, pred_sin: torch.Tensor,
                             phase_reliability: torch.Tensor,
                             eps: float = 1e-8,) -> tuple[torch.Tensor, torch.Tensor]:

        """Copy observed phase exactly and predict only unavailable frames."""
        r = phase_reliability.unsqueeze(1).to(pred_cos.dtype)
        final_cos = observed_cos + (1.0 - r) * pred_cos
        final_sin = observed_sin + (1.0 - r) * pred_sin
        norm = torch.sqrt(final_cos.square() + final_sin.square() + eps)

        return final_cos / norm, final_sin / norm
