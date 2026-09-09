import torch
import torch.nn as nn

class MaskedLoss(nn.Module):
    def forward(self, pred, target, mask=None):
        # mask & target: [B, F, T]
        # pred: [B, T, F]
        if mask is not None:
            w_mask = torch.where(mask==1, 1.0, 10.0)
            w_mask = w_mask.to(device=pred.device, dtype=pred.dtype)
            loss = torch.abs(pred - target)
            loss = (loss * w_mask).sum() / (w_mask.sum() + 1e-8)
        else:
            loss = torch.abs(pred - target).mean()

        return loss
