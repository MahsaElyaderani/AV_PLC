"""AV_PLC-only waveform STFT/log-Mel frontend.

The legacy projects keep using shared.audio_processing and the stored HDF5 ``spec``.
This module is intentionally local to AV_PLC so the waveform-domain PLC redesign
cannot silently change the reproduced baselines.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
import torchaudio


@dataclass(frozen=True)
class AudioFrontendConfig:
    sample_rate: int = 16000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 80
    lookahead_ms: float = 7.5
    log_floor: float = 1e-5


class AudioFrontend:
    """Configurable-lookahead STFT and fixed-reference log-Mel frontend.

    Lookahead is measured from the end of each 10-ms packet.  For frame ``t``
    the non-zero 400-sample analysis window covers

        [t*hop + hop - win + L,  t*hop + hop + L)

    where ``L`` is the lookahead in samples.  With L=120 (7.5 ms), the
    implementation is exactly the current AV_PLC geometry: center=False with
    176 samples of manual padding on both sides.
    """

    def __init__(self, config: AudioFrontendConfig | None = None,
                 mel_mean: float | None = None, mel_std: float | None = None):
        self.config = config or AudioFrontendConfig()
        self.mel_mean = mel_mean
        self.mel_std = mel_std
        self._window_cache = {}
        self._mel_cache = {}

        c = self.config
        if c.win_length > c.n_fft:
            raise ValueError("win_length must be <= n_fft")
        self.lookahead_samples = int(round(c.lookahead_ms * c.sample_rate / 1000.0))
        if not 0 <= self.lookahead_samples <= c.win_length - c.hop_length:
            raise ValueError(
                f"lookahead_ms={c.lookahead_ms} gives L={self.lookahead_samples}; "
                f"expected 0 <= L <= {c.win_length - c.hop_length} samples"
            )

        # torch.stft(center=False) centers win_length inside n_fft.  These pads
        # place the real 400-sample support at the explicit timing above while
        # keeping the total pad constant, hence the same frame count.
        half_fft_slack = (c.n_fft - c.win_length) // 2  # 56
        self.left_pad = c.win_length - c.hop_length - self.lookahead_samples + half_fft_slack
        self.right_pad = self.lookahead_samples + half_fft_slack

    def set_stats(self, mean: float, std: float) -> None:
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
            raise ValueError(f"Invalid Mel statistics: mean={mean}, std={std}")
        self.mel_mean = float(mean)
        self.mel_std = float(std)

    def _window(self, device, dtype):
        key = (str(device), dtype)
        if key not in self._window_cache:
            self._window_cache[key] = torch.hann_window(
                self.config.win_length, periodic=True, device=device, dtype=dtype
            )
        return self._window_cache[key]

    def _padded_window(self, device, dtype):
        c = self.config
        w = self._window(device, dtype)
        side = (c.n_fft - c.win_length) // 2
        return F.pad(w, (side, c.n_fft - c.win_length - side))

    def _mel_scale(self, device, dtype):
        key = (str(device), dtype)
        if key not in self._mel_cache:
            self._mel_cache[key] = torchaudio.transforms.MelScale(
                n_mels=self.config.n_mels,
                sample_rate=self.config.sample_rate,
                n_stft=self.config.n_fft // 2 + 1,
                norm="slaney",
                mel_scale="slaney",
            ).to(device=device, dtype=dtype)
        return self._mel_cache[key]

    @staticmethod
    def _as_batch(waveform: torch.Tensor):
        if waveform.ndim == 1:
            return waveform.unsqueeze(0), True
        if waveform.ndim != 2:
            raise ValueError(f"waveform must be [N] or [B,N], got {tuple(waveform.shape)}")
        return waveform, False

    def stft(self, waveform: torch.Tensor) -> torch.Tensor:
        x, squeeze = self._as_batch(waveform)
        x = F.pad(x, (self.left_pad, self.right_pad))
        z = torch.stft(
            x,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self._window(x.device, x.dtype),
            center=False,
            return_complex=True,
            onesided=True,
        )
        return z.squeeze(0) if squeeze else z

    def istft(self, spectrum: torch.Tensor, length: int) -> torch.Tensor:
        """Weighted overlap-add inverse matching :meth:`stft` exactly."""
        z = spectrum
        squeeze = False
        if z.ndim == 2:
            z = z.unsqueeze(0)
            squeeze = True
        if z.ndim != 3:
            raise ValueError(f"spectrum must be [F,T] or [B,F,T], got {tuple(z.shape)}")

        c = self.config
        frames = torch.fft.irfft(z.transpose(1, 2), n=c.n_fft, dim=-1)
        win = self._padded_window(frames.device, frames.dtype)
        frames = frames * win

        total_len = c.n_fft + c.hop_length * (frames.shape[1] - 1)
        out = frames.new_zeros((frames.shape[0], total_len))
        norm = frames.new_zeros((frames.shape[0], total_len))
        win2 = win.square()
        for t in range(frames.shape[1]):
            s = t * c.hop_length
            out[:, s:s + c.n_fft] += frames[:, t]
            norm[:, s:s + c.n_fft] += win2
        out = out / norm.clamp_min(1e-8)
        out = out[:, self.left_pad:self.left_pad + int(length)]
        return out.squeeze(0) if squeeze else out

    def magnitude_to_logmel(self, magnitude: torch.Tensor) -> torch.Tensor:
        mel = self._mel_scale(magnitude.device, magnitude.dtype)(magnitude)
        return 20.0 * torch.log10(mel.clamp_min(self.config.log_floor))

    def logmel(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.magnitude_to_logmel(self.stft(waveform).abs())

    def normalize(self, logmel: torch.Tensor) -> torch.Tensor:
        if self.mel_mean is None or self.mel_std is None:
            raise RuntimeError("Mel statistics are not set for the AV_PLC audio frontend")
        return (logmel - self.mel_mean) / self.mel_std

    def denormalize(self, normalized_mel: torch.Tensor) -> torch.Tensor:
        if self.mel_mean is None or self.mel_std is None:
            raise RuntimeError("Mel statistics are not set for the AV_PLC audio frontend")
        return normalized_mel * self.mel_std + self.mel_mean

    def normalized_logmel(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.normalize(self.logmel(waveform))

    def frame_masks(self, sample_keep: torch.Tensor, audio_length: torch.Tensor | int):
        """Return ``frame_valid, hard_keep, soft_keep`` on the STFT frame grid.

        ``hard_keep[t]`` is 1 only when every valid waveform sample touched by the
        Hann window was received.  Padding never counts as either received or lost.
        """
        keep, squeeze = self._as_batch(sample_keep.float())
        B, N = keep.shape
        if torch.is_tensor(audio_length):
            lengths = audio_length.to(device=keep.device, dtype=torch.long).reshape(-1)
        else:
            lengths = torch.full((B,), int(audio_length), device=keep.device, dtype=torch.long)
        if lengths.numel() == 1 and B > 1:
            lengths = lengths.expand(B)
        if lengths.numel() != B:
            raise ValueError("audio_length batch does not match sample_keep batch")

        arange = torch.arange(N, device=keep.device).unsqueeze(0)
        valid = (arange < lengths.unsqueeze(1)).to(keep.dtype)
        keep = torch.where(valid.bool(), keep, torch.ones_like(keep))

        valid_p = F.pad(valid, (self.left_pad, self.right_pad))
        keep_p = F.pad(keep, (self.left_pad, self.right_pad), value=1.0)
        valid_frames = valid_p.unfold(1, self.config.n_fft, self.config.hop_length)
        keep_frames = keep_p.unfold(1, self.config.n_fft, self.config.hop_length)
        w2 = self._padded_window(keep.device, keep.dtype).square().view(1, 1, -1)

        denom = (valid_frames * w2).sum(-1)
        observed = (valid_frames * keep_frames * w2).sum(-1)
        lost = (valid_frames * (1.0 - keep_frames) * w2).sum(-1)
        frame_valid = denom > 1e-8
        hard_keep = frame_valid & (lost <= 1e-8)
        soft_keep = torch.where(frame_valid, observed / denom.clamp_min(1e-8), torch.ones_like(denom))

        if squeeze:
            return frame_valid[0], hard_keep[0], soft_keep[0]
        return frame_valid, hard_keep, soft_keep
