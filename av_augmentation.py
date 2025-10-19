# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import torch
import random
import numpy as np
import torch.nn as nn

def load_video(path):
    for i in range(3):
        try:
            cap = cv2.VideoCapture(path)
            frames = []
            while True:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frames.append(frame)
                else:
                    break
            frames = np.stack(frames)
            return frames
        except Exception:
            print(f"failed loading {path} ({i} / 3)")
            if i == 2:
                raise ValueError(f"Unable to load {path}")


class Compose(object):
    """Compose several preprocess together.
    Args:
        preprocess (list of ``Preprocess`` objects): list of preprocess to compose.
    """

    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, sample):
        for t in self.preprocess:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.preprocess:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Normalize(object):
    """Normalize a ndarray image with mean and standard deviation.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, frames):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        frames = (frames - self.mean) / self.std
        return frames

    def __repr__(self):
        return self.__class__.__name__+'(mean={0}, std={1})'.format(self.mean, self.std)

class CenterCrop(object):
    """Crop the given image at the center
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """

        t, h, w = frames.shape
        th, tw = self.size
        delta_w = int(round((w - tw))/2.)
        delta_h = int(round((h - th))/2.)
        frames = frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]
        return frames


class RandomCrop(object):
    """Crop the given image at the center
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = random.randint(0, w-tw)
        delta_h = random.randint(0, h-th)
        frames = frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]
        return frames

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class HorizontalFlip(object):
    """Flip image horizontally.
    """

    def __init__(self, flip_ratio):
        self.flip_ratio = flip_ratio

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be flipped with a probability flip_ratio
        Returns:
            numpy.ndarray: Cropped image.
        """

        t, h, w = frames.shape
        if random.random() < self.flip_ratio:
            for index in range(t):
                frames[index] = cv2.flip(frames[index], 1)
        return frames

import math
class RandomErase(object):
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), replace_with_zero=True):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.replace_with_zero = replace_with_zero

    def get_params(self, frames, scale, ratio):

        t, h, w = frames.shape
        area = h * w

        log_ratio = np.log(np.array(ratio))
        while True:
            erase_area = area * random.uniform(scale[0], scale[1])
            aspect_ratio = np.exp(random.uniform(log_ratio[0], log_ratio[1]))

            erase_h = int(round(math.sqrt(erase_area * aspect_ratio)))
            erase_w = int(round(math.sqrt(erase_area / aspect_ratio)))
            if (erase_h < h and erase_w < w):
                i = random.randint(0, h - erase_h)
                j = random.randint(0, w - erase_w)
                return i, j, h, w

    def __call__(self, frames):
        if random.random() < self.p:
            i, j, h, w = self.get_params(frames, scale=self.scale, ratio=self.ratio)
            if self.replace_with_zero:
                frames[:, i:i+h, j:j+w] = 0.
            else:
                frames[:, i:i+h, j:j+w] = frames.mean()

        return frames

class TimeMask(object):
    """time mask
    """
    def __init__(self, max_mask_T=0.4, hop_T=1., fps=25, replace_with_zero=True, inplace=False):
        self.max_mask_frame = round(max_mask_T * fps)
        self.hop_frame = round(hop_T * fps)

        self.replace_with_zero = replace_with_zero
        self.inplace = inplace

    def __call__(self, x):

        if self.inplace:
            cloned = x
        else:
            cloned = x.copy()

        len_raw = cloned.shape[0]

        for i in range(len_raw//self.hop_frame):
            mask_len = random.randint(0, self.max_mask_frame)
            mask_start = random.randint(0, self.hop_frame - mask_len)
            if self.replace_with_zero:
                cloned[i*self.hop_frame+mask_start : i*self.hop_frame+mask_start+mask_len] = 0.
            else:
                cloned[i*self.hop_frame+mask_start : i*self.hop_frame+mask_start+mask_len] = cloned.mean()

        return cloned


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
