import numpy as np
import librosa
import librosa.filters
import os
from scipy import signal
import soundfile as sf
import torch
import torchaudio
import subprocess, shlex

from evaluations.runtime_config import DATA_ROOT, SEED
# Character vocabulary
# Griffin-Lim implementation by candlewill
# Audio -> Spectrogram / Spectrogram -> Audio conversion
# https://github.com/candlewill/Griffin_lim

mel_mean = -56.775
mel_std = 19.707
hop_len = 160
win_len = 400
n_fft = 512
n_stft = n_fft // 2 + 1
num_mels = 80
sample_rate = 16000
frame_length_ms = 25
frame_shift_ms = 10
min_level_db = -80
griffin_lim_iters = 60

def load_audio_ffmpeg(path, sr=sample_rate, fixlen_sec=None):
    """
    Decode 'path' (mp4/mpg/wav/...) to mono float32 at 'sr' using FFmpeg.
    If fixlen_sec is provided, trim/pad to exactly sr*fixlen_sec samples.
    """
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin",
        "-i", str(path),
        "-vn",           # no video
        "-ac", "1",      # mono
        "-ar", str(sr),  # resample
        # optional: pick first audio stream explicitly
        # "-map", "0:a:0",
    ]
    if fixlen_sec is not None:
        cmd += ["-t", str(float(fixlen_sec))]
    cmd += ["-f", "f32le", "pipe:1"]  # raw float32 PCM to stdout

    out = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    a = np.frombuffer(out, dtype=np.float32).copy()  # 1-D mono float32
    peak = np.max(np.abs(a)) or 1.0
    if peak > 1.0:
        a = a / peak  # peak-normalize

    if fixlen_sec is not None:
        N = int(round(sr * float(fixlen_sec)))
        if a.size < N:
            a = np.pad(a, (0, N - a.size), mode="constant")
        elif a.size > N:
            a = a[:N]
    return a

def read_gt_input(video_path, mask, base_path=None,
                  sample_rate=sample_rate, fixlen_sec=3):
    if base_path is None:
        base_path = DATA_ROOT
    rel_path = video_path.split("datasets", 1)[-1].lstrip(os.sep)
    audio_path = os.path.join(str(base_path), rel_path)
    original_audio_np = load_audio_ffmpeg(audio_path, sr=sample_rate, fixlen_sec=fixlen_sec)
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        mask_t = mask[0].astype(np.float32)  # (T,)
        sample_mask = np.repeat(mask_t, hop_len)  # e.g., hop_length=160 for 16 kHz, 10 ms hop
        sample_mask = sample_mask[:len(original_audio_np)]
        masked_audio_np = original_audio_np * sample_mask

    return original_audio_np, masked_audio_np if mask is not None else None

def torch_mel2spec(mel_norm):

    # 1. Denormalize
    mel_db = (mel_norm * mel_std) + mel_mean

    # 2. Convert dB to power
    mel_mag = torchaudio.functional.DB_to_amplitude(mel_db, ref=1.0, power=0.5)
    mel_mag = mel_mag.clamp(min=1e-5)

    # 3. Mel -> Linear
    inv_mel = torchaudio.transforms.InverseMelScale(
        n_stft=n_stft,
        n_mels=num_mels,
        sample_rate=sample_rate,
        f_min=0.0,
        f_max=8000.0,
        norm='slaney',
        mel_scale='slaney',
    ).to(mel_norm.device)

    linear_spec = inv_mel(mel_mag).clamp(min=1e-5)

    return linear_spec ** 2

# In _compute_loss, replace torch_mel2spec:
def mel_to_power_linear(mel_norm, mel_pinv, mel_mean=-56.775, mel_std=19.707):
    mel_db = mel_norm * mel_std + mel_mean
    mel_mag = torchaudio.functional.DB_to_amplitude(mel_db, ref=1.0, power=0.5)
    # mel_mag: [B, 80, T] → linear: [B, 257, T]
    linear_mag = torch.matmul(mel_pinv.T, mel_mag)  # differentiable
    return linear_mag.clamp(min=1e-5).pow(2)

def torch_mel2audio(mel_norm, mel_mean=mel_mean, mel_std=mel_std):

    # 1. Denormalize
    mel_db = (mel_norm * mel_std) + mel_mean

    # 2. Convert dB to power
    mel_mag = torchaudio.functional.DB_to_amplitude(mel_db, ref=1.0, power=0.5)
    mel_mag = mel_mag.clamp(min=1e-5)

    # 3. Mel -> Linear
    inv_mel = torchaudio.transforms.InverseMelScale(
        n_stft=n_stft,
        n_mels=num_mels,
        sample_rate=sample_rate,
        f_min=0.0,
        f_max=8000.0,
        norm='slaney',
        mel_scale='slaney',
    ).to(mel_norm.device)
    linear_spec = inv_mel(mel_mag).clamp(min=1e-5)

    # 4. Reconstruct waveform (Griffin-Lim)
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=n_fft,
        win_length=win_len,
        hop_length=hop_len,
        power=1.0,
        n_iter=griffin_lim_iters
    )
    audio = griffin_lim(linear_spec)

    # Normalize to [-1, 1] range
    audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
    return audio

def librosa_mel2audio(mel_norm, sr=16000,
                      n_fft=512, win_length=400, hop=160,
                      fmin=0.0, fmax=8000.0,
                      mel_mean=-56.775, mel_std=19.707, n_iter=64):
    # 1) denorm dB -> magnitude  (matches torchaudio AmplitudeToDB with stype='magnitude')
    mel_db  = mel_norm * mel_std + mel_mean
    #mel_db = mel_db.clamp(min=-80.0, max=0.0)  # match top_db=80

    mel_mag = librosa.db_to_amplitude(np.asarray(mel_db.cpu(),
                                                 dtype=np.float32), ref=1.0)  # shape [M, T]

    # 2) mel (magnitude) -> linear (magnitude)
    S_mag = librosa.feature.inverse.mel_to_stft(
        mel_mag,
        sr=sr,
        n_fft=n_fft,
        power=1.0,
        fmin=fmin,
        fmax=fmax,
        norm='slaney',
        htk=False
    )
    # 3) Griffin-Lim with center=False (to mirror your forward path)
    y = librosa.griffinlim(
        S_mag,
        n_iter=n_iter,
        hop_length=hop,
        win_length=win_length,
        window='hann',
        center=False,
        momentum=0.99,
        pad_mode='constant',
        init="random",
        random_state=SEED,
    )
    pad = (n_fft - hop) // 2
    y = y[pad:-pad]  # remove the artificial reflect region
    #peak = np.max(np.abs(y)) or 1.0
    #if peak > 1.0:
    #    y = y / peak  # peak-normalize
    #y = y / (np.max(np.abs(y)) + 1e-8)
    return torch.from_numpy(y.astype(np.float32, copy=False))
# --------------------------------------------------------------------------------------
# Optional learned-phase reconstruction (keeps torch_mel2audio / Griffin-Lim unchanged)
# --------------------------------------------------------------------------------------

def mel_to_linear_magnitude(mel_norm, mel_mean=mel_mean, mel_std=mel_std):
    """Convert normalized AV_PLC Mel to one-sided linear STFT magnitude."""
    squeeze = mel_norm.dim() == 2
    if squeeze:
        mel_norm = mel_norm.unsqueeze(0)
    if mel_norm.dim() != 3 or mel_norm.size(1) != num_mels:
        raise ValueError(f"Expected Mel [B,{num_mels},T] or [{num_mels},T], got {tuple(mel_norm.shape)}")

    mel_db = mel_norm * float(mel_std) + float(mel_mean)
    mel_mag = torchaudio.functional.DB_to_amplitude(mel_db, ref=1.0, power=0.5).clamp_min(1e-5)
    inv_mel = torchaudio.transforms.InverseMelScale(
        n_stft=n_stft,
        n_mels=num_mels,
        sample_rate=sample_rate,
        f_min=0.0,
        f_max=8000.0,
        norm='slaney',
        mel_scale='slaney',
    ).to(device=mel_norm.device, dtype=mel_norm.dtype)
    linear_mag = inv_mel(mel_mag).clamp_min(1e-5)
    return linear_mag.squeeze(0) if squeeze else linear_mag


def _manual_istft_center_false(complex_spec, crop_padding=True, eps=1e-8):
    """Differentiable overlap-add inverse for AV_PLC's center=False STFT.

    torch.istft rejects this exact center=False Hann setup at the zero-valued
    outer window samples (NOLA check).  The explicit weighted overlap-add below
    is the matching inverse and round-trips the padded forward STFT numerically.
    """
    squeeze = complex_spec.dim() == 2
    if squeeze:
        complex_spec = complex_spec.unsqueeze(0)
    if complex_spec.dim() != 3 or complex_spec.size(1) != n_stft:
        raise ValueError(f"Expected complex STFT [B,{n_stft},T], got {tuple(complex_spec.shape)}")

    batch, _, frames_n = complex_spec.shape
    time_frames = torch.fft.irfft(complex_spec, n=n_fft, dim=1)
    window = torch.hann_window(win_len, periodic=True, device=complex_spec.device,
                               dtype=time_frames.dtype)
    left = (n_fft - win_len) // 2
    right = n_fft - win_len - left
    padded_window = torch.nn.functional.pad(window, (left, right))
    time_frames = time_frames * padded_window.view(1, n_fft, 1)

    output_length = n_fft + hop_len * (frames_n - 1)
    audio = time_frames.new_zeros((batch, output_length))
    norm = time_frames.new_zeros(output_length)
    window_sq = padded_window.square()

    for frame_idx in range(frames_n):
        start = frame_idx * hop_len
        audio[:, start:start + n_fft] += time_frames[:, :, frame_idx]
        norm[start:start + n_fft] += window_sq

    audio = audio / norm.clamp_min(eps).unsqueeze(0)
    if crop_padding:
        pad = (n_fft - hop_len) // 2
        if audio.size(-1) <= 2 * pad:
            raise ValueError("STFT is too short to remove AV_PLC boundary padding")
        audio = audio[:, pad:-pad]
    return audio.squeeze(0) if squeeze else audio


def torch_melphase2audio(mel_norm, phase_cos, phase_sin,
                         mel_mean=mel_mean, mel_std=mel_std,
                         normalize_output=False):
    """Reconstruct waveform from Mel magnitude plus predicted unit phase.

    This is the learned-phase alternative to ``torch_mel2audio``.  It performs
    inverse Mel once, combines magnitude with cosine/sine phase, and uses one
    weighted overlap-add iSTFT.  Griffin-Lim remains untouched for the legacy
    path.
    """
    squeeze = mel_norm.dim() == 2
    if squeeze:
        mel_norm = mel_norm.unsqueeze(0)
        phase_cos = phase_cos.unsqueeze(0)
        phase_sin = phase_sin.unsqueeze(0)

    expected = (mel_norm.size(0), n_stft, mel_norm.size(-1))
    if tuple(phase_cos.shape) != expected or tuple(phase_sin.shape) != expected:
        raise ValueError(
            f"phase_cos/phase_sin must be {expected}; got "
            f"{tuple(phase_cos.shape)} and {tuple(phase_sin.shape)}"
        )

    linear_mag = mel_to_linear_magnitude(mel_norm, mel_mean=mel_mean, mel_std=mel_std)
    phase_cos = phase_cos.to(device=linear_mag.device, dtype=linear_mag.dtype)
    phase_sin = phase_sin.to(device=linear_mag.device, dtype=linear_mag.dtype)
    phase_norm = torch.sqrt(phase_cos.square() + phase_sin.square() + 1e-8)
    phase_cos = phase_cos / phase_norm
    phase_sin = phase_sin / phase_norm
    complex_spec = torch.complex(linear_mag * phase_cos, linear_mag * phase_sin)
    audio = _manual_istft_center_false(complex_spec, crop_padding=True)

    if normalize_output:
        if audio.dim() == 1:
            audio = audio / audio.abs().amax().clamp_min(1e-8)
        else:
            audio = audio / audio.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    return audio.squeeze(0) if squeeze and audio.dim() == 2 else audio
