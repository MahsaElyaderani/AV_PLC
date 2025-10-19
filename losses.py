import torch
import librosa
import numpy as np
import torch.nn as nn
import torchaudio.functional as F
import torchaudio.transforms as transforms
from transformers import pipeline
from editdistance import eval as edit_eval
from jiwer import wer
from torch import tensor
import pathlib
import os

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        freqs = librosa.fft_frequencies(sr=16000, n_fft=512)
        # Create a band-pass emphasis mask
        mask_np = np.where((freqs >= 300) & (freqs <= 4000), 1.5, 1.0)  # 2×emphasis
        self.mask = torch.from_numpy(mask_np).float()

    def forward(self, pred, target, mask=False):
        if mask:
            self.mask = self.mask.to(pred.device)
            loss = torch.mean(self.mask[:, None] * (pred - target) ** 2)
        else:
            loss = ((pred - target) ** 2).mean()
        return loss

class MaskedLoss(nn.Module):
    def forward(self, pred, target, mask=None):
        # mask & target: [B, F, T]
        # pred: [B, T, F]
        if mask is not None:
            w_mask = torch.where(mask==1, 1.0, 10.0)
            w_mask = w_mask.to('cuda')
            loss = torch.abs(pred - target)
            loss = (loss * w_mask).sum() / (w_mask.sum() + 1e-8)
        else:
            loss = torch.abs(pred - target).mean()

        return loss

class SVTS_Loss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.l1_loss = L1Loss()
        self.spectral_convergence_loss = SpectralConvergenceLoss()

    def forward(self, preds, gts):
        return self.l1_loss(preds, gts) + self.spectral_convergence_loss(preds, gts)


class L1Loss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.l1_loss = torch.nn.L1Loss()

    def forward(self, preds, gts):
        batch_loss = 0
        for gt, pred in zip(gts, preds):
            batch_loss += (self.l1_loss(gt, pred))

        return batch_loss / len(gts)


class SpectralConvergenceLoss(torch.nn.Module):
    """
    Based on https://github.com/kan-bayashi/ParallelWaveGAN/blob/1f7949f593cc5600478cd4cc23bbf34bbcb0bcff/parallel_wavegan/losses/stft_loss.py#L43
    """

    def __init__(self):
        super().__init__()
        self.norm_f = torch.linalg.norm

    def forward(self, preds, gts):
        batch_loss = 0
        for gt, pred in zip(gts, preds):
            loss = (self.norm_f(gt - pred, ord='fro') / self.norm_f(gt, ord='fro'))
            batch_loss += loss

        return batch_loss / len(gts)

