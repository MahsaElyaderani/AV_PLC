import numpy as np
import librosa
import librosa.filters
from scipy import signal
import soundfile as sf
import random
from itertools import groupby
from scipy.signal import butter, sosfilt, sosfreqz
# Character vocabulary
# Griffin-Lim implementation by candlewill
# Audio -> Spectrogram / Spectrogram -> Audio conversion
# https://github.com/candlewill/Griffin_lim

num_mels = 80 #128
num_freq = 640 #257 #Number of frequency bins in the linear spectrogram. n_stft = n_fft // 2 + 1 for real-valued signals
sample_rate = 16000
frame_length_ms = 40 #24
frame_shift_ms = 10 #12
preemphasis = 0.97
min_level_db = -80
ref_level_db = 20
griffin_lim_iters = 60

def load_wav(path, sr):
    return librosa.load(path, sr=sr)[0]

def save_wav(wav, path):
    sf.write(path, wav, sample_rate, subtype='PCM_16')

def spectrogram(y):
    D = _stft(_preemphasis(y))
    S = _amp_to_db(np.abs(D)) - ref_level_db
    #phase = np.angle(D)
    #return _normalize(S), phase
    return _normalize(S)

def inv_spectrogram(spectrogram, angles):
    S = _db_to_amp(_denormalize(spectrogram) + ref_level_db)
    if angles is None:
        return _inv_preemphasis(_griffin_lim(S ** 1.5))
    else:
        S_complex = S * np.exp(1j * angles)
        return _inv_preemphasis(_istft(S_complex))

def melspectrogram(y):
    D = _stft(_preemphasis(y))
    S = _amp_to_db(_linear_to_mel(np.abs(D)))
    return _normalize(S)

def inv_melspectrogram(melspectrogram):
    S = _mel_to_linear(_db_to_amp(_denormalize(melspectrogram)))  # Convert back to linear
    return _inv_preemphasis(_griffin_lim(S ** 1.5))  # Reconstruct phase

# Based on https://github.com/librosa/librosa/issues/434
def _griffin_lim(S):

    S_complex = np.abs(S).astype(np.complex_)

    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    for i in range(griffin_lim_iters):
        if i > 0:
            angles = np.exp(1j * np.angle(_stft(y)))
        y = _istft(S_complex * angles)

    return y

def _stft(y):
    n_fft = (num_freq - 1) * 2 #num_freq = n_stft
    hop_length = int(frame_shift_ms / 1000. * sample_rate)
    win_length = int(frame_length_ms / 1000. * sample_rate)
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

def _istft(y):
    hop_length = int(frame_shift_ms / 1000. * sample_rate)
    win_length = int(frame_length_ms / 1000. * sample_rate)
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)

# Conversions:
_mel_basis = None
_inv_mel_basis = None

def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)

def _mel_to_linear(mel_spectrogram):
    global _inv_mel_basis
    if _inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(_build_mel_basis())
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))

def _build_mel_basis():
    n_fft = (num_freq - 1) * 2
    return librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=num_mels)

def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))

def _db_to_amp(x):
    return np.power(10.0, x * 0.05)

def _preemphasis(x):
    return signal.lfilter([1, -preemphasis], [1], x)

def _inv_preemphasis(x):
    return signal.lfilter([1], [1, -preemphasis], x)

def _normalize(S):
    return np.clip((S - min_level_db) / -float(min_level_db), 0, 1)


def _denormalize(S):
    return (np.clip(S, 0, 1) * -min_level_db) + min_level_db


import torch
import torchaudio

def torch_denormalize(S):
    return (torch.clip(S, 0, 1) * -min_level_db) + min_level_db

def torch_db_to_amp(x):
    return torch.pow(10.0, x * 0.05)

def pow_spec(x):
    return torch_db_to_amp(torch_denormalize(x) + ref_level_db) ** 2

def torch_audio2mel(audio):

    melspctrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=640,
        win_length=640,
        hop_length=160,
        center=False,
        power=1.0,
        norm="slaney",
        onesided=True,
        n_mels=80,
        mel_scale="slaney",
    )
    padding = (640 - 160) // 2
    wav = torch.nn.functional.pad(audio, (padding, padding), "reflect")
    mel = melspctrogram(wav)
    logmel = torch.log(torch.clamp(mel, min=1e-5))

    return logmel

def torch_mel2audio(logmel):

    inv_mel = torchaudio.transforms.InverseMelScale(
        n_stft=320+1,
        n_mels=80,
        sample_rate=16000,
        f_min=0.0,
        f_max=8000.0,
        norm='slaney',
        mel_scale='slaney'
    )

    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=640,
        win_length=640,
        hop_length=160,
        power=1.0,
        n_iter=32
    )

    mel = torch.exp(logmel)
    linear_spec = inv_mel(mel)
    audio = griffin_lim(linear_spec)

    return audio