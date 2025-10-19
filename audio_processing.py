import numpy as np
import librosa
import librosa.filters
import os
from scipy import signal
import soundfile as sf
import torch
import torchaudio
import subprocess, shlex

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

def read_gt_input(video_path, mask, base_path="/home/ai/Projects/Mahsa/datasets",
                  sample_rate=sample_rate, fixlen_sec=3):
    rel_path = video_path.split("datasets", 1)[-1].lstrip(os.sep)
    audio_path = os.path.join(base_path, rel_path)
    original_audio_np = load_audio_ffmpeg(audio_path, sr=sample_rate, fixlen_sec=fixlen_sec)
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        mask_t = mask[0].astype(np.float32)  # (T,)
        sample_mask = np.repeat(mask_t, hop_len)  # e.g., hop_length=160 for 16 kHz, 10 ms hop
        sample_mask = sample_mask[:len(original_audio_np)]
        masked_audio_np = original_audio_np * sample_mask

    return original_audio_np, masked_audio_np if mask is not None else None


def torch_audio2mel(audio):

    melspctrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_len,
        hop_length=hop_len,
        center=False,
        power=1.0, # 1 for magnitude, 2 for power, etc
        norm="slaney",
        onesided=True,
        n_mels=num_mels,
        mel_scale="slaney",
    )
    amp2db_transform = torchaudio.transforms.AmplitudeToDB(stype='magnitude', top_db=80)

    padding = (n_fft - hop_len) // 2
    wav = torch.nn.functional.pad(audio, (padding, padding), "reflect")
    mel_mag = melspctrogram(wav)
    mel_mag = mel_mag.clamp(min=1e-5)  # avoid log(0)
    logmel = amp2db_transform(mel_mag)
    logmel_norm = (logmel - mel_mean)/mel_std

    return logmel_norm

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


def torch_mel2audio(mel_norm):

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

def librosa_mel2audio(mel_norm, sr=16000, n_fft=512, win_length=400, hop=160,
                      fmin=0.0, fmax=8000.0, mel_mean=-56.775, mel_std=19.707, n_iter=64):
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
        pad_mode='constant'
    )
    pad = (n_fft - hop) // 2
    y = y[pad:-pad]  # remove the artificial reflect region
    #peak = np.max(np.abs(y)) or 1.0
    #if peak > 1.0:
    #    y = y / peak  # peak-normalize
    y = y / (np.max(np.abs(y)) + 1e-8)
    return torch.from_numpy(y)