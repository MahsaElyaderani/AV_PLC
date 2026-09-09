import torch
import torch.nn as nn

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask=False):
        return torch.mean((pred - target) ** 2)

class CTCLoss(nn.Module):
    def __init__(self, blank_idx: int = 0, zero_infinity: bool = True):
        super().__init__()

        self.blank_idx = blank_idx

        self.ctc_loss = nn.CTCLoss(
            blank=blank_idx,
            reduction="mean",
            zero_infinity=zero_infinity,
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        target_lengths: torch.Tensor | None = None,
        input_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            pred:
                Raw model logits with shape [B, T, V]

            target:
                Padded target indices with shape [B, S]
                Padding should be blank_idx = 0.

            target_lengths:
                True target lengths before padding, shape [B].
                Strongly recommended.

            input_lengths:
                Valid output lengths from the model, shape [B].
                If None, assumes every sample has length T.

        Returns:
            CTC loss.
        """

        if pred.ndim != 3:
            raise ValueError(f"pred must have shape [B, T, V], got {pred.shape}")

        if target.ndim != 2:
            raise ValueError(f"target must have shape [B, S], got {target.shape}")

        device = pred.device

        B, T, V = pred.shape

        # Convert logits to log-probabilities.
        log_probs = pred.log_softmax(dim=-1)

        # PyTorch CTCLoss expects [T, B, V].
        log_probs = log_probs.permute(1, 0, 2)

        target = target.to(device).long()

        if input_lengths is None:
            input_lengths = torch.full(
                size=(B,),
                fill_value=T,
                dtype=torch.long,
                device=device,
            )
        else:
            input_lengths = input_lengths.to(device).long()

        if target_lengths is None:
            # Infer target lengths by counting non-blank labels.
            # This is okay only if padding is blank_idx and blank never appears
            # inside the true target sequence.
            target_lengths = torch.sum(target != self.blank_idx, dim=1).long()
        else:
            target_lengths = target_lengths.to(device).long()

        # CTCLoss accepts padded targets [B, S] when target_lengths is provided.
        loss = self.ctc_loss(
            log_probs,
            target,
            input_lengths,
            target_lengths,
        )

        return loss