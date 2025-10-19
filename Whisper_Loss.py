import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper
from typing import Optional, Literal


class WhisperASRLoss(nn.Module):
    """
    L1 loss between Whisper encoder features of GT and Pred log-mel spectrograms.

    - Inputs must be Whisper-style log-mel: [B, 80, T].
    - Pads/truncates to 3000 frames to match Whisper input length.
    - Encodes GT (usually under no_grad) and Pred (with grad).
    - Computes per-time-step L1 in feature space, masks padded tail, then reduces.

    Args:
        model_size: "tiny"|"base"|"small"|"medium"|"large"
        device: torch device string; if None, chooses CUDA if available
        pad_mode: "repeat" (pad with last frame) or "zeros"
        reduction: "mean" or "sum"
        detach_gt: encode GT under no_grad to save memory (recommended)
    """

    def __init__(
        self,
        model_size= "base",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()

        self.n_mels = 80
        self.target_frames = 3000  # 30s @ 100 fps
        self.device = device

        self.whisper = whisper.load_model(model_size, device=self.device)
        self.whisper.eval()

        for p in self.whisper.parameters():
            p.requires_grad = False  # freeze params; grads still flow to inputs

    def _preprocess(self, mel: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        -> (mel_proc [B,80,3000], orig_T_clipped)
        """
        mel = mel.to(self.device, dtype=torch.float32)
        B, C, T = mel.shape
        if C != self.n_mels:
            raise ValueError(f"Expected {self.n_mels} mel bins, got {C}")
        target = self.target_frames
        orig_t= min(T, target)

        if T < target:
            pad = target - T
            mel = F.pad(mel, (0, pad), mode="constant", value=0.0)
        elif T > target:
            mel = mel[:, :, :target]

        return mel, orig_t

    def forward(self, gt_mel: torch.Tensor, pred_mel: torch.Tensor) -> torch.Tensor:
        # 1) pad/trim to 3000 frames
        gt_mel_p, t_gt = self._preprocess(gt_mel)
        pred_mel_p, _ = self._preprocess(pred_mel)

        # 2) encode: GT typically without grad; Pred with grad
        with torch.no_grad():
            gt_feat = self.whisper.encoder(gt_mel_p)   # [B, T', D]

        pred_feat = self.whisper.encoder(pred_mel_p)       # [B, T', D]

        # 3) valid-time mask for padding (Whisper downsamples time by ~2)
        B, Tprime, _ = pred_feat.shape
        valid_Tprime = max(1, int(round(Tprime * (t_gt / self.target_frames))))
        mask = pred_feat.new_zeros((B, Tprime))
        mask[:, :valid_Tprime] = 1.0                       # [B, T']

        # 4) per-time-step L1 in feature space, then masked reduce
        per_t = (gt_feat - pred_feat).abs().mean(dim=-1)   # [B, T']
        per_t = per_t * mask

        denom = mask.sum().clamp_min(1.0)
        loss = per_t.sum() / denom

        return loss
