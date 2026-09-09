"""Audio helpers shared by AV_PLC phase diagnostics.

These helpers use the same configurable AV_PLC frontend as training/evaluation;
they never reopen the source media file.
"""
from __future__ import annotations

import torch

from AV_PLC.audio_frontend import AudioFrontend, AudioFrontendConfig
from shared.audio_processing import mel_to_linear_magnitude


def make_frontend(mel_mean: float, mel_std: float, lookahead_ms: float) -> AudioFrontend:
    return AudioFrontend(
        AudioFrontendConfig(lookahead_ms=float(lookahead_ms)),
        mel_mean=float(mel_mean), mel_std=float(mel_std),
    )


def mel_phase_to_audio(mel_norm, phase_cos, phase_sin, frontend: AudioFrontend,
                       output_length: int = 48000):
    squeeze = mel_norm.dim() == 2
    if squeeze:
        mel_norm = mel_norm.unsqueeze(0)
        phase_cos = phase_cos.unsqueeze(0)
        phase_sin = phase_sin.unsqueeze(0)

    mag = mel_to_linear_magnitude(
        mel_norm, mel_mean=frontend.mel_mean, mel_std=frontend.mel_std
    )
    phase_cos = phase_cos.to(device=mag.device, dtype=mag.dtype)
    phase_sin = phase_sin.to(device=mag.device, dtype=mag.dtype)
    norm = torch.sqrt(phase_cos.square() + phase_sin.square()).clamp_min(1e-8)
    z = torch.complex(mag * phase_cos / norm, mag * phase_sin / norm)
    audio = frontend.istft(z, length=int(output_length))
    return audio.squeeze(0) if squeeze and audio.dim() == 2 else audio


def fresh_stft(clean_audio, frontend: AudioFrontend, device=None):
    audio = clean_audio
    if not torch.is_tensor(audio):
        audio = torch.as_tensor(audio, dtype=torch.float32)
    audio = audio.to(device=device or audio.device, dtype=torch.float32)
    return frontend.stft(audio)


def observed_stft(clean_audio, sample_mask, frontend: AudioFrontend, device=None):
    """Complex STFT of the actually packet-corrupted waveform."""
    audio = clean_audio if torch.is_tensor(clean_audio) else torch.as_tensor(clean_audio)
    mask = sample_mask if torch.is_tensor(sample_mask) else torch.as_tensor(sample_mask)
    dev = device or audio.device
    audio = audio.to(device=dev, dtype=torch.float32)
    mask = mask.to(device=dev, dtype=torch.float32)
    return frontend.stft(audio * mask)
