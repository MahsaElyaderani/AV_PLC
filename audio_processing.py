import torch
import torchaudio


mel_mean = -56.775
mel_std = 19.707

def torch_audio2mel(audio):

    melspctrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=512, #640,
        win_length=400, #640,
        hop_length=160,
        center=False,
        power=1.0,
        norm="slaney",
        onesided=True,
        n_mels=80,
        mel_scale="slaney",
    )
    amp2db_transform = torchaudio.transforms.AmplitudeToDB(stype='magnitude', top_db=80)

    padding = (512 - 160) // 2 #(640 - 160) // 2
    wav = torch.nn.functional.pad(audio, (padding, padding), "reflect")
    mel_mag = melspctrogram(wav)
    mel_mag = mel_mag.clamp(min=1e-5)  # avoid log(0)
    logmel = amp2db_transform(mel_mag)
    #logmel_norm = (logmel + 80) / 80  # maps [-80, 0] -> [0, 1]
    #logmel_norm = torch.clamp(logmel_norm, 0.0, 1.0)
    logmel_norm = (logmel - mel_mean)/mel_std

    #print(mel_db.shape)
    #logmel = torch.log(torch.clamp(mel, min=1e-5))

    return logmel_norm

# Precompute on CPU
fb = torchaudio.functional.melscale_fbanks(
    n_freqs=257,
    f_min=0.0,
    f_max=8000.0,
    n_mels=80,
    sample_rate=16000,
    norm='slaney',
    mel_scale='slaney'
)
fb_inv = torch.linalg.pinv(fb)  # CPU


def torch_mel2spec(mel_norm):

    # 1. Denormalize
    #mel_norm = (mel_norm + 1.0) / 2.0
    #mel_db = mel_norm.clamp(0.0, 1.0) * 80 - 80
    mel_db = (mel_norm * mel_std) + mel_mean

    # 2. Convert dB to power
    mel_mag = torchaudio.functional.DB_to_amplitude(mel_db, ref=1.0, power=0.5)
    mel_mag = mel_mag.clamp(min=1e-5)

    # 3. Mel -> Linear
    if mel_norm.device.type == 'mps':
        # MPS does not support torch.pinverse() yet, so we compute the pinv on CPU and then move to MPS
        fb_inv_device = fb_inv.to(mel_norm.device)
        mel_spec_t = mel_mag.permute(0, 2, 1)
        inv_spec_t = mel_spec_t @ fb_inv_device
        linear_spec = inv_spec_t.permute(0, 2, 1)
        return linear_spec
    else:
        inv_mel = torchaudio.transforms.InverseMelScale(
            n_stft=257,  # 320+1,
            n_mels=80,
            sample_rate=16000,
            f_min=0.0,
            f_max=8000.0,
            norm='slaney',
            mel_scale='slaney',
        ).to(mel_norm.device)

        linear_spec = inv_mel(mel_mag).clamp(min=1e-5)

    #mel = torch.exp(logmel)
    #linear_spec = inv_mel(mel)

    return linear_spec ** 2

def torch_mel2audio(mel_norm):

    mel_db = (mel_norm * mel_std) + mel_mean

    # 2. Convert dB to power
    mel_mag = torchaudio.functional.DB_to_amplitude(mel_db, ref=1.0, power=0.5)
    mel_mag = mel_mag.clamp(min=1e-5)

    # 3. Mel -> Linear
    inv_mel = torchaudio.transforms.InverseMelScale(
        n_stft=257,  # 320+1,
        n_mels=80,
        sample_rate=16000,
        f_min=0.0,
        f_max=8000.0,
        norm='slaney',
        mel_scale='slaney',
    ).to(mel_norm.device)
    linear_spec = inv_mel(mel_mag).clamp(min=1e-5)

    # 4. Reconstruct waveform (Griffin-Lim)
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=512, #640,
        win_length=400, #640,
        hop_length=160,
        power=1.0,
        n_iter=32
    )
    audio = griffin_lim(linear_spec)

    return audio