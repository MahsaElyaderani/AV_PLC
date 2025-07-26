import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from scipy import signal

# Parameters
num_mels = 128
num_freq = 1024
sample_rate = 16000
frame_length_ms = 64.0
frame_shift_ms = 32.2
preemphasis = 0.97
min_level_db = -80
ref_level_db = 20
griffin_lim_iters = 60

# Derived parameters
n_fft = (num_freq - 1) * 2
hop_length = int(frame_shift_ms / 1000. * sample_rate)
win_length = int(frame_length_ms / 1000. * sample_rate)
window = torch.hann_window(win_length)

# Cached bases
mel_basis = None
inv_mel_basis = None


def load_wav(path, sr):
    # torchaudio loads audio as (channels, time)
    waveform, sample_rate = torchaudio.load(path)
    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    # Resample if needed
    if sample_rate != sr:
        waveform = torchaudio.transforms.Resample(sample_rate, sr)(waveform)
    return waveform.squeeze().numpy()


def save_wav(wav, path):
    # Convert to tensor if numpy array
    if isinstance(wav, np.ndarray):
        wav = torch.FloatTensor(wav)
    wav = wav.unsqueeze(0)  # Add channel dimension
    torchaudio.save(path, wav, sample_rate)


def _stft(y):
    # Convert to tensor if numpy array
    if isinstance(y, np.ndarray):
        y = torch.FloatTensor(y)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = y.to(device)
    window = window.to(device)

    # Perform STFT
    return torch.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True
    )


def _istft(y):
    return torch.istft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window
    )


def _preemphasis(x):
    if isinstance(x, torch.Tensor):
        # PyTorch implementation
        device = x.device
        return F.lfilter(
            x.unsqueeze(0),
            torch.tensor([1., -preemphasis], device=device),
            torch.tensor([1.], device=device)
        ).squeeze(0)
    else:
        # Fallback to scipy for numpy arrays
        return signal.lfilter([1, -preemphasis], [1], x)


def _inv_preemphasis(x):
    if isinstance(x, torch.Tensor):
        # PyTorch implementation
        device = x.device
        return F.lfilter(
            x.unsqueeze(0),
            torch.tensor([1.], device=device),
            torch.tensor([1., -preemphasis], device=device)
        ).squeeze(0)
    else:
        # Fallback to scipy for numpy arrays
        return signal.lfilter([1], [1, -preemphasis], x)


def _amp_to_db(x):
    if isinstance(x, torch.Tensor):
        return 20 * torch.log10(torch.clamp(x, min=1e-5))
    else:
        import numpy as np
        return 20 * np.log10(np.maximum(1e-5, x))


def _db_to_amp(x):
    if isinstance(x, torch.Tensor):
        return torch.pow(10.0, x * 0.05)
    else:
        import numpy as np
        return np.power(10.0, x * 0.05)


def _normalize(S):
    if isinstance(S, torch.Tensor):
        return torch.clamp((S - min_level_db) / -float(min_level_db), 0, 1)
    else:
        import numpy as np
        return np.clip((S - min_level_db) / -float(min_level_db), 0, 1)


def _denormalize(S):
    if isinstance(S, torch.Tensor):
        return (torch.clamp(S, 0, 1) * -min_level_db) + min_level_db
    else:
        import numpy as np
        return (np.clip(S, 0, 1) * -min_level_db) + min_level_db


def _build_mel_basis():
    global mel_basis
    if mel_basis is None:
        # Create a torchaudio mel filter bank
        mel_basis = T.MelScale(
            n_mels=num_mels,
            sample_rate=sample_rate,
            f_min=0.0,
            f_max=sample_rate / 2.0,
            n_stft=num_freq
        ).fb
    return mel_basis


def _linear_to_mel(spectrogram):
    if isinstance(spectrogram, torch.Tensor):
        mel_basis = _build_mel_basis().to(spectrogram.device)
        return torch.matmul(mel_basis, spectrogram)
    else:
        # Convert numpy spectrogram to tensor for matmul, then back to numpy
        import numpy as np
        spectrogram_tensor = torch.FloatTensor(spectrogram)
        mel_basis = _build_mel_basis()
        return torch.matmul(mel_basis, spectrogram_tensor).numpy()


def _mel_to_linear(mel_spectrogram):
    global inv_mel_basis
    if inv_mel_basis is None:
        mel_basis = _build_mel_basis()
        inv_mel_basis = torch.pinverse(mel_basis)

    if isinstance(mel_spectrogram, torch.Tensor):
        inv_mel_basis_t = inv_mel_basis.to(mel_spectrogram.device)
        linear = torch.matmul(inv_mel_basis_t, mel_spectrogram)
        return torch.clamp(linear, min=1e-10)
    else:
        # Convert numpy mel_spectrogram to tensor for matmul, then back to numpy
        import numpy as np
        mel_spectrogram_tensor = torch.FloatTensor(mel_spectrogram)
        linear = torch.matmul(inv_mel_basis, mel_spectrogram_tensor).numpy()
        return np.maximum(1e-10, linear)


def _griffin_lim(S):
    # Griffin-Lim algorithm to reconstruct phase
    if isinstance(S, np.ndarray):
        S = torch.FloatTensor(S)

    device = S.device if isinstance(S, torch.Tensor) else torch.device("cpu")
    S = S.to(device)

    # Initialize random phase
    angles = torch.exp(2j * torch.pi * torch.rand_like(S, device=device))
    S_complex = S.abs().to(torch.complex64)

    for i in range(griffin_lim_iters):
        y = _istft(S_complex * angles)
        if i < griffin_lim_iters - 1:
            angles = torch.exp(1j * torch.angle(_stft(y)))

    return y.cpu().numpy() if device.type == "cuda" else y.numpy()


def spectrogram(y):
    D = _stft(_preemphasis(y))
    S = _amp_to_db(torch.abs(D)) - ref_level_db
    return _normalize(S)


def inv_spectrogram(spectrogram):
    S = _db_to_amp(_denormalize(spectrogram) + ref_level_db)
    # Convert back to linear
    return _inv_preemphasis(_griffin_lim(S * 1.5))  # Reconstruct phase


def melspectrogram(y):
    D = _stft(_preemphasis(y))
    S = _amp_to_db(_linear_to_mel(torch.abs(D)))
    return _normalize(S)


def inv_melspectrogram(melspectrogram):
    S = _mel_to_linear(_db_to_amp(_denormalize(melspectrogram)))
    # Convert back to linear
    return _inv_preemphasis(_griffin_lim(S * 1.5))  # Reconstruct phase