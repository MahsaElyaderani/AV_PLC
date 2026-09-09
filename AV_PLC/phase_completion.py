"""Phase-only time-frequency completion module for AV_PLC.

The AV_PLC Mel reconstruction path is intentionally outside this
module.  This module receives the *completed* normalized log-Mel spectrogram
(observed Mel copied outside PLC gaps, AV_PLC prediction inside gaps) together
with sparse observed STFT phase, converts Mel to a fixed 257-bin linear
magnitude estimate, and predicts only phase.

Shape convention inside this file
---------------------------------
    completed_mel : [B, 80, T]
    phase_radians : [B, 257, T]
    reliability   : [B, T], 1=observed audio/phase, 0=phase to predict
    TF tensors    : [B, C, F, T], F=257

The phase branch is:
    [A_c, R*cos(phi), R*sin(phi)]
      -> shape-preserving Conv2D/dilated-dense TF encoder
      -> N axial TS-Conformer blocks (time then frequency)
      -> small phase-specific 2-D refinement block
      -> parallel 64->1 real/imag heads
      -> unit-circle (cos, sin)
      -> exact observed-phase copy.

The design is MP-SENet-inspired but adapted to AV_PLC: there is no frequency
subsampling/upsampling because exact F x T alignment is useful for PLC phase
inpainting and the Mel branch is already reconstructed separately.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio
from conformer import Conformer


class DilatedDenseTFBlock(nn.Module):
    """Shape-preserving dense 2-D block with temporal dilation.

    Frequency context is mixed locally by the 3x3 kernels; temporal receptive
    field grows as 1, 2, 4, ... without changing F or T.
    """

    def __init__(self, channels: int, depth: int = 3) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if depth <= 0:
            raise ValueError("depth must be positive")

        self.channels = int(channels)
        self.depth = int(depth)
        self.layers = nn.ModuleList()

        for i in range(self.depth):
            dilation_t = 2 ** i
            self.layers.append(nn.Sequential(nn.Conv2d(self.channels * (i + 1), self.channels,
                                                       kernel_size=(3, 3), stride=1,
                                                       padding=(1, dilation_t), dilation=(1, dilation_t),),
                                             nn.InstanceNorm2d(self.channels, affine=True),
                                             nn.PReLU(self.channels),))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.size(1) != self.channels:
            raise ValueError(f"DilatedDenseTFBlock expects [B,{self.channels},F,T], got {tuple(x.shape)}")
        dense = x
        out = x
        for layer in self.layers:
            out = layer(dense)
            dense = torch.cat([dense, out], dim=1)
        return out


class TSConformerBlock(nn.Module):
    """Axial time-then-frequency Conformer for [B,C,F,T] features."""

    def __init__(self, channels: int = 64, heads: int = 4,
                 dropout: float = 0.1, conv_kernel_size: int = 15,) -> None:

        super().__init__()
        if channels <= 0 or heads <= 0:
            raise ValueError("channels and heads must be positive")
        if channels % heads != 0:
            raise ValueError(f"phase channels ({channels}) must be divisible by heads ({heads})")

        dim_head = channels // heads
        conformer_kwargs = dict(dim=channels, depth=1, dim_head=dim_head, heads=heads,
                                ff_mult=4, conv_expansion_factor=2,
                                conv_kernel_size=conv_kernel_size, attn_dropout=dropout,
                                ff_dropout=dropout, conv_dropout=dropout,)
        self.time_conformer = Conformer(**conformer_kwargs)
        self.freq_conformer = Conformer(**conformer_kwargs)
        self.channels = int(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.size(1) != self.channels:
            raise ValueError(f"TSConformerBlock expects [B,{self.channels},F,T], got {tuple(x.shape)}")

        b, c, f, t = x.shape

        # Time modeling: each frequency bin is one sequence.
        h = x.permute(0, 2, 3, 1).contiguous().view(b * f, t, c)
        h = self.time_conformer(h)
        x = h.view(b, f, t, c).permute(0, 3, 1, 2).contiguous()

        # Frequency modeling: each time frame is one frequency sequence.
        h = x.permute(0, 3, 2, 1).contiguous().view(b * t, f, c)
        h = self.freq_conformer(h)
        x = h.view(b, t, f, c).permute(0, 3, 2, 1).contiguous()
        return x


class PhaseCompletion(nn.Module):
    """Parallel magnitude/phase TF reconstruction for AV_PLC.

    The content branch supplies one reconstructed normalized log-Mel estimate
    (audio-only, video-only, or AV-fused).  It is inverted with the fixed Slaney
    Mel pseudo-inverse.  Exact observed linear STFT magnitude is then copied into
    reliable frames, while only PLC-gap frames use the inverse-Mel estimate.

    TFRefine input channels are exactly::

        [A_completed**c, R*cos(phi_obs), R*sin(phi_obs)]

    with c=0.3 by default.  The magnitude head predicts A**c, not raw A.
    It is decompressed back to physical linear STFT magnitude before complex-STFT
    construction, waveform reconstruction, and final observed-bin preservation.

    A shared 2-D Conv -> dense TF encoder -> axial time/frequency Conformer stack
    is followed by two independent branches:

        magnitude branch -> predicted compressed magnitude A**c
        phase branch     -> unit-circle (cos, sin)

    For the final PLC spectrum, observed magnitude and phase are copied exactly
    outside the gap.  Raw branch predictions are retained for masked losses.
    """

    def __init__(self, mel_dim: int = 80, phase_bins: int = 257, sample_rate: int = 16000,
                 n_fft: int = 512, channels: int = 64, num_ts_blocks: int = 2, num_heads: int = 4,
                 encoder_dense_depth: int = 3, decoder_dense_depth: int = 2, dropout: float = 0.1,
                 magnitude_compression: float = 0.3, mel_mean: float = -56.775, mel_std: float = 19.707,
                 eps: float = 1e-8,) -> None:
        super().__init__()
        if phase_bins != n_fft // 2 + 1:
            raise ValueError(f"phase_bins must equal n_fft//2+1; got {phase_bins} for n_fft={n_fft}")
        if num_ts_blocks <= 0:
            raise ValueError("num_ts_blocks must be positive")
        if magnitude_compression <= 0.0:
            raise ValueError("magnitude_compression must be > 0")
        if mel_std <= 0.0:
            raise ValueError("mel_std must be > 0")

        self.mel_dim = int(mel_dim)
        self.phase_bins = int(phase_bins)
        self.channels = int(channels)
        self.magnitude_compression = float(magnitude_compression)
        self.eps = float(eps)

        mel_fb = torchaudio.functional.melscale_fbanks(
            n_freqs=self.phase_bins, f_min=0.0, f_max=float(sample_rate) / 2.0,
            n_mels=self.mel_dim, sample_rate=sample_rate,
            norm="slaney", mel_scale="slaney",
        )  # [F,M]
        mel_pinv = torch.linalg.pinv(mel_fb).float()  # [M,F]
        self.register_buffer("mel_pinv", mel_pinv, persistent=True)
        self.register_buffer("mel_mean", torch.tensor(float(mel_mean)), persistent=True)
        self.register_buffer("mel_std", torch.tensor(float(mel_std)), persistent=True)

        # Shared TFRefine trunk: 3 channels = completed magnitude, masked cos, masked sin.
        self.tf_stem = nn.Sequential(
            nn.Conv2d(3, self.channels, kernel_size=1),
            nn.InstanceNorm2d(self.channels, affine=True),
            nn.PReLU(self.channels),
        )
        self.tf_dense = DilatedDenseTFBlock(self.channels, depth=encoder_dense_depth)
        self.ts_blocks = nn.ModuleList([
            TSConformerBlock(
                channels=self.channels, heads=num_heads,
                dropout=dropout, conv_kernel_size=15,
            )
            for _ in range(num_ts_blocks)
        ])

        # Independent magnitude branch.
        self.mag_refine = DilatedDenseTFBlock(self.channels, depth=decoder_dense_depth)
        self.mag_post = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(self.channels, affine=True),
            nn.PReLU(self.channels),
        )
        self.mag_head = nn.Conv2d(self.channels, 1, kernel_size=1)
        self.mag_positive = nn.Softplus()

        # Independent phase branch.
        self.phase_refine = DilatedDenseTFBlock(self.channels, depth=decoder_dense_depth)
        self.phase_post = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(self.channels, affine=True),
            nn.PReLU(self.channels),
        )
        self.phase_real = nn.Conv2d(self.channels, 1, kernel_size=1)
        self.phase_imag = nn.Conv2d(self.channels, 1, kernel_size=1)

    @torch.no_grad()
    def set_mel_stats(self, mel_mean: float, mel_std: float) -> None:
        if mel_std <= 0.0:
            raise ValueError("mel_std must be > 0")
        self.mel_mean.copy_(self.mel_mean.new_tensor(float(mel_mean)))
        self.mel_std.copy_(self.mel_std.new_tensor(float(mel_std)))

    def mel_to_linear_magnitude(self, reconstructed_mel: torch.Tensor) -> torch.Tensor:
        """Differentiable normalized-logMel -> approximate linear magnitude [B,F,T]."""
        if reconstructed_mel.dim() != 3 or reconstructed_mel.size(1) != self.mel_dim:
            raise ValueError(
                f"reconstructed_mel must be [B,{self.mel_dim},T], got {tuple(reconstructed_mel.shape)}"
            )
        mel_f = reconstructed_mel.float()
        mel_db = mel_f * self.mel_std.float() + self.mel_mean.float()
        mel_mag = torchaudio.functional.DB_to_amplitude(mel_db, ref=1.0, power=0.5)
        linear_mag = torch.matmul(self.mel_pinv.t(), mel_mag).clamp_min(self.eps)
        return linear_mag

    def forward(self, reconstructed_mel: torch.Tensor, observed_magnitude: torch.Tensor,
                phase_radians: torch.Tensor, reliability: torch.Tensor,) -> dict[str, torch.Tensor]:
        if phase_radians.dim() != 3 or phase_radians.size(1) != self.phase_bins:
            raise ValueError(f"phase_radians must be [B,{self.phase_bins},T], got {tuple(phase_radians.shape)}")
        if observed_magnitude.shape != phase_radians.shape:
            raise ValueError(
                f"observed_magnitude must match phase shape {tuple(phase_radians.shape)}, "
                f"got {tuple(observed_magnitude.shape)}"
            )
        if reconstructed_mel.size(0) != phase_radians.size(0) or reconstructed_mel.size(-1) != phase_radians.size(-1):
            raise ValueError("reconstructed_mel and phase_radians must share batch/time axes")
        if reliability.shape != (reconstructed_mel.size(0), reconstructed_mel.size(-1)):
            raise ValueError(f"reliability must be [B,T], got {tuple(reliability.shape)}")

        rel = reliability.to(device=reconstructed_mel.device, dtype=torch.float32).clamp(0.0, 1.0)
        phase = phase_radians.to(device=reconstructed_mel.device, dtype=torch.float32)
        observed_mag = observed_magnitude.to(device=reconstructed_mel.device, dtype=torch.float32).clamp_min(0.0)

        # Inverse-Mel only proposes the missing region.  Reliable magnitude remains exact.
        inverse_mel_mag = self.mel_to_linear_magnitude(reconstructed_mel)
        r = rel[:, None, :]  # [B,1,T], broadcasts over F
        completed_mag = r * observed_mag + (1.0 - r) * inverse_mel_mag

        # MP-SENet-style power compression.  TFRefine works in the compressed
        # magnitude domain because raw STFT magnitude has a very large dynamic range.
        # c=0.3 is the default; c=1.0 remains available as the no-compression ablation.
        network_mag = completed_mag.clamp_min(self.eps).pow(self.magnitude_compression)

        # Mask phase after sin/cos so a missing phase is [0,0], not the false angle 0.
        observed_cos = torch.cos(phase) * r
        observed_sin = torch.sin(phase) * r

        y = torch.stack([network_mag, observed_cos, observed_sin], dim=1)  # [B,3,F,T]
        y = y.to(dtype=reconstructed_mel.dtype)

        hidden = self.tf_stem(y)
        hidden = hidden + self.tf_dense(hidden)
        for block in self.ts_blocks:
            hidden = block(hidden)

        # Magnitude head: predict *compressed* absolute magnitude directly, not a
        # multiplicative mask.  Decompress only when a physical STFT magnitude is
        # needed for the final complex spectrum / waveform.
        mag_hidden = hidden + self.mag_refine(hidden)
        mag_hidden = self.mag_post(mag_hidden)
        predicted_mag_compressed = self.mag_positive(self.mag_head(mag_hidden).squeeze(1)).float()
        predicted_mag = predicted_mag_compressed.clamp_min(self.eps).pow(1.0 / self.magnitude_compression)

        # Phase head: two unconstrained Cartesian auxiliaries -> unit-circle phase.
        phase_hidden = hidden + self.phase_refine(hidden)
        phase_hidden = self.phase_post(phase_hidden)
        p_r = self.phase_real(phase_hidden).squeeze(1).float()
        p_i = self.phase_imag(phase_hidden).squeeze(1).float()
        norm = torch.sqrt(p_r.square() + p_i.square() + self.eps)
        pred_cos = p_r / norm
        pred_sin = p_i / norm
        pred_phase = torch.atan2(pred_sin, pred_cos)

        # Final PLC output preserves every reliable STFT bin exactly.
        final_mag = r * observed_mag + (1.0 - r) * predicted_mag
        final_cos = observed_cos + (1.0 - r) * pred_cos
        final_sin = observed_sin + (1.0 - r) * pred_sin
        final_norm = torch.sqrt(final_cos.square() + final_sin.square() + self.eps)
        final_cos = final_cos / final_norm
        final_sin = final_sin / final_norm
        final_phase = torch.atan2(final_sin, final_cos)

        return {
            "inverse_mel_mag": inverse_mel_mag,
            "completed_input_mag": completed_mag,
            "completed_input_mag_compressed": network_mag,
            "predicted_mag_compressed": predicted_mag_compressed,
            "predicted_mag": predicted_mag,
            "final_mag": final_mag,
            "p_r": p_r,
            "p_i": p_i,
            "pred_cos": pred_cos,
            "pred_sin": pred_sin,
            "pred_phase": pred_phase,
            "final_cos": final_cos,
            "final_sin": final_sin,
            "final_phase": final_phase,
            "reliability": rel,
        }
