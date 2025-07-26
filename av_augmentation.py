import numpy as np
import torch
import random
import torch.nn as nn

class RandomFrameDrop(nn.Module):
    def __init__(self, drop_prob=0.01):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, video):

        video = video.copy()
        T = video.shape[0]
        drop_trace = np.random.choice([0, 1], T, p=[1 - self.drop_prob, self.drop_prob])
        for t in range(T):
             if drop_trace[t] == 1:
                video[t, :, :, :] = 0.0
        return video


class RandomHorizontalFlip(nn.Module):
    def __init__(self, p=0.3):
        super().__init__()
        self.p = p

    def forward(self, x):
        # x: (T, H, W, C) or (T, C, H, W)
        if random.random() < self.p:
            if x.ndim == 4 and x.shape[-1] == 3:
                return np.flip(x,axis=2)
            elif x.ndim == 4:
                return np.flip(x, axis=3)
        return x


class AddGaussianNoise(nn.Module):
    def __init__(self, mean=0.0, std=0.01, p=0.3):
        super().__init__()
        self.mean = mean
        self.std = std
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        if isinstance(x, torch.Tensor):
            noise = torch.randn_like(x) * self.std + self.mean
        else:
            noise = np.random.normal(self.mean, self.std, size=x.shape).astype(x.dtype)
        return x + noise


class TemporalJitter(nn.Module):
    def __init__(self, max_jitter=2, p=0.01):
        super().__init__()
        self.max_jitter = max_jitter
        self.p = p

    def forward(self, x):
        # x: (T, H, W, C)
        if random.random() > self.p:
            return x

        T = x.shape[0]
        jitter = random.randint(-self.max_jitter, self.max_jitter)
        indices = np.arange(T)
        indices = np.clip(indices + jitter, 0, T - 1)
        return x[indices]


class SpecAugment(nn.Module):
    def __init__(self, freq_mask_param=10, time_mask_param=10, num_masks=1, p=0.3):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_masks = num_masks
        self.p = p

    def forward(self, spec):
        # spec: (F, T) or (1, F, T)
        if random.random() > self.p:
            return spec

        spec = spec.copy()
        for _ in range(self.num_masks):
            # Frequency mask
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, max(1, spec.shape[-2] - f))
            spec[..., f0:f0+f, :] = 0

            # Time mask
            t = random.randint(0, self.time_mask_param)
            t0 = random.randint(0, max(1, spec.shape[-1] - t))
            spec[..., :, t0:t0+t] = 0

        return spec

class AddGaussianNoiseSpec(nn.Module):
    def __init__(self, mean=0.0, std=0.01, p=0.3):
        super().__init__()
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, spec):
        if random.random() > self.p:
            return spec
        if isinstance(spec, torch.Tensor):
            noise = torch.randn_like(spec) * self.std + self.mean
        else:
            noise = np.random.normal(self.mean, self.std, size=spec.shape).astype(spec.dtype)
        return spec + noise

class MixBackgroundSpec:
    def __init__(self, bg_specs, snr_range=(0, 10)):
        """
        bg_specs: list of background spectrograms (numpy or torch.Tensor)
        snr_range: range of signal-to-noise ratio (dB) to sample from
        """
        self.bg_specs = bg_specs
        self.snr_range = snr_range

    def forward(self, spec):
        # Pick random bg spec
        bg_spec = random.choice(self.bg_specs)

        # Match size (simple version — assume bg spec is big enough)
        T = spec.shape[-1]
        bg_spec = bg_spec[..., :T]

        # Random SNR
        snr_db = np.random.uniform(*self.snr_range)
        snr = 10 ** (snr_db / 20)

        # Normalize
        spec_power = np.mean(spec ** 2)
        bg_power = np.mean(bg_spec ** 2)

        scale = np.sqrt(spec_power / (snr ** 2 * bg_power + 1e-8))
        bg_scaled = bg_spec * scale

        # Mix
        return spec + bg_scaled

class VideoAugmentations(nn.Module):
    def __init__(self, flip_p=0.3, noise_std=0.02, temporal_jitter=2, drop_prob=0.3):
        super().__init__()
        self.transforms = nn.Sequential(
            RandomHorizontalFlip(p=flip_p),
            AddGaussianNoise(std=noise_std),
            TemporalJitter(max_jitter=temporal_jitter),
            RandomFrameDrop(drop_prob=drop_prob)
        )

    def forward(self, x):
        return self.transforms(x)


class SpectrogramAugmentations(nn.Module):
    def __init__(self, p=0.3, freq_mask=10, time_mask=10, num_masks=1, noise_std=0.02):
        super().__init__()
        self.p = p
        self.transform = nn.Sequential(SpecAugment(freq_mask_param=freq_mask,
                                                    time_mask_param=time_mask,
                                                    num_masks=num_masks),
                                        AddGaussianNoiseSpec(std=noise_std))

    def forward(self, x):
        if random.random() > self.p:
            return x
        else:
            return self.transform(x)


class ModalityDropout:
    def __init__(self, mode_probs=None):
        if mode_probs is None:
            mode_probs = [0.6, 0.2, 0.2]

        assert np.isclose(sum(mode_probs), 1.0), "mode_probs must sum to 1.0"

        self.mode_probs = mode_probs
        self.modes = ["audio_video", "audio_only", "video_only"]

    def sample_mode(self):
        mode = np.random.choice(self.modes, p=self.mode_probs)
        return mode

    def forward(self, masked_spec, visual_features):
        mode = self.sample_mode()
        # Apply modality dropout...
        # return masked_spec, visual_features, video_is_missing, audio_is_missing
