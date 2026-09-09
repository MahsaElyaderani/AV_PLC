import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper

class L1Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, preds, gts):
        return torch.nn.functional.l1_loss(preds, gts)

class SpectralConvergenceLoss(torch.nn.Module):
    def __init__(self, mel_mean=-56.775, mel_std=19.707, eps=1e-8):
        super().__init__()
        self.mel_mean = mel_mean
        self.mel_std = mel_std
        self.eps = eps

    def forward(self, preds_norm, gts_norm):
        # Step 1: undo z-score normalization -> back to dB
        preds_db = preds_norm * self.mel_std + self.mel_mean
        gts_db   = gts_norm.detach() * self.mel_std + self.mel_mean

        # Step 2: dB (magnitude convention) -> linear magnitude
        preds_lin = torch.pow(10.0, preds_db / 20.0)
        gts_lin   = torch.pow(10.0, gts_db / 20.0)

        # Step 3: spectral convergence, on linear magnitude
        diff_norm = torch.linalg.norm(gts_lin - preds_lin, ord='fro', dim=(-2, -1))
        ref_norm  = torch.linalg.norm(gts_lin, ord='fro', dim=(-2, -1))

        return (diff_norm / (ref_norm + self.eps)).mean()

class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, mel_mean=-56.775, mel_std=19.707, eps=1e-8):
        super().__init__()
        self.mel_mean = mel_mean
        self.mel_std = mel_std
        self.eps = eps

    def forward( self, recon_norm,     target_norm) :
        # [B, n_mels, T] -- model output, normalized  # [B, n_mels, T] -- ground truth, normalized
        # ---- Step 1: undo z-score normalization -> back to dB scale ----
        recon_db  = recon_norm  * self.mel_std + self.mel_mean
        target_db = target_norm.detach() * self.mel_std + self.mel_mean   # target has no grad

        # ---- Step 2: dB (magnitude convention) -> linear magnitude ----
        """Inverse of AmplitudeToDB(stype='magnitude'): db = 20*log10(x), ref=1.0"""
        recon_lin  = torch.pow(10.0, recon_db / 20.0).clamp(min=self.eps)
        target_lin = torch.pow(10.0, target_db / 20.0).clamp(min=self.eps)

        # ---- Step 3 & 4 combined for numerical stability ----
        # Instead of computing q = recon_lin/sum then log(q) separately (two divisions),
        # use log(q) = log(recon_lin) - log(sum(recon_lin)) directly.
        p = target_lin / (target_lin.sum(dim=1, keepdim=True) + self.eps)        # [B, n_mels, T]
        log_q = torch.log(recon_lin + self.eps) - torch.log(recon_lin.sum(dim=1, keepdim=True) + self.eps) # [B, n_mels, T]

        # ---- Step 5: cross entropy per frame (sum over the 80 mel bins) ----
        ce_per_frame = -(p * log_q).sum(dim=1)     # [B, T]

        # ---- Step 6: aggregate ----
        return ce_per_frame.mean()

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

    def __init__(self, model_size= "base", device="cuda" if torch.cuda.is_available() else "cpu",):

        super().__init__()

        self.n_mels = 80
        self.target_frames = 3000  # 30s @ 100 fps
        self.device = device

        self.whisper = whisper.load_model(model_size, device="cpu")
        self.whisper.eval()

        for p in self.whisper.parameters():
            p.requires_grad = False  # freeze params; grads still flow to inputs

    def _to_whisper_mel(self, mel_norm: torch.Tensor) -> torch.Tensor:
        """
        Map our stored mel-norm to Whisper's input distribution.

        Dataset pipeline (save_features.py):
            MelSpectrogram(n_fft=512, win=400, hop=160, power=1.0, n_mels=80, slaney)
            -> clamp(1e-5) -> AmplitudeToDB(stype='magnitude', top_db=80)
            -> z-score with mu=-56.775, sigma=19.707
            Note: 20*log10(|M|) == 10*log10(power), so mel_db is dB-of-power.

        Whisper pipeline (whisper/audio.py):
            log10(power).clamp_to_max_minus_8_of_max()
            -> (log_spec + 4) / 4
        """
        # 1) undo our z-score -> 10*log10(power) in dB, i.e. same as Whisper's dB
        mel_db = mel_norm * 19.707 - 56.775  # approx [-80, 0] (per top_db=80)

        # 2) dB -> log10(power): Whisper's domain
        log10_p = mel_db / 10.0  # approx [-8, 0]

        # 3) Whisper's relative 80-dB clamp (idempotent for GT because top_db=80 was
        #    already applied; meaningful for predicted mels which bypass that clamp)
        log10_max = log10_p.amax(dim=(-2, -1), keepdim=True).detach()
        log10_p = torch.maximum(log10_p, log10_max - 8.0)

        # 4) Whisper's canonical final affine -> typical range [-1, 1]
        return (log10_p + 4.0) / 4.0

    def _preprocess(self, mel: torch.Tensor) -> tuple[torch.Tensor, int]:
        mel = self._to_whisper_mel(mel)
        mel = mel.to(self.device, dtype=torch.float32)
        B, C, T = mel.shape
        if C != self.n_mels:
            raise ValueError(f"Expected {self.n_mels} mel bins, got {C}")
        orig_t = min(T, self.target_frames)
        if T > self.target_frames:
            mel = mel[:, :, :self.target_frames]
        elif T < self.target_frames:
            # Whisper encoder requires exactly 3000 frames — pad with zeros
            pad = self.target_frames - T
            mel = F.pad(mel, (0, pad), mode="constant", value=0.0)
        return mel, orig_t  # orig_t = how many frames are real (not padding)

    def forward(self, gt_mel: torch.Tensor, pred_mel: torch.Tensor) -> torch.Tensor:
        gt_mel_p, t_gt = self._preprocess(gt_mel)
        pred_mel_p, _ = self._preprocess(pred_mel)

        with torch.no_grad():
            gt_feat = self.whisper.encoder(gt_mel_p)  # [B, T', D]
        pred_feat = self.whisper.encoder(pred_mel_p)  # [B, T', D]

        # Whisper encoder conv2 has stride=2 → output time = target_frames // 2 = 1500
        # Scale valid input frames to valid output frames
        B, Tprime, _ = pred_feat.shape
        valid_Tprime = max(1, int(round(Tprime * (t_gt / self.target_frames))))

        # Only compute L1 loss over valid (non-padded) encoder frames
        per_t = (gt_feat - pred_feat).abs().mean(dim=-1)  # [B, T']
        loss = per_t[:, :valid_Tprime].mean()

        return loss

# --------------------------------------------------------------------------------------
# phase-reconstruction losses
# --------------------------------------------------------------------------------------

def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None, eps: float = 1e-8):
    if mask is None:
        return values.mean()
    m = mask.to(device=values.device, dtype=values.dtype)
    while m.dim() < values.dim():
        m = m.unsqueeze(1)
    m = m.expand_as(values)
    denom = m.sum().clamp_min(eps)
    return (values * m).sum() / denom

class MaskedMelReconstructionLoss(nn.Module):
    """L1 on the Mel frames the unified decoder is responsible for predicting.

    ``prediction_mask`` is [B,T], with 1 in packet-loss frames for audio-present
    samples and 1 over the complete utterance for video-only samples.  The
    decoder predicts an *absolute* normalized log-Mel spectrogram; there is no
    base-Mel residual in the latent-only architecture.
    """

    def forward(self, predicted_mel, target_mel, prediction_mask=None):
        return _masked_mean((predicted_mel - target_mel).abs(), prediction_mask)
MelRefinementLoss = MaskedMelReconstructionLoss

class UnitPhaseLoss(nn.Module):
    """Circular phase loss: 1 - cos(predicted_angle - target_angle)."""

    def forward(self, pred_cos, pred_sin, target_phase, prediction_mask=None):
        target_cos = torch.cos(target_phase.to(pred_cos.dtype))
        target_sin = torch.sin(target_phase.to(pred_sin.dtype))
        loss = 1.0 - (pred_cos * target_cos + pred_sin * target_sin)
        return _masked_mean(loss, prediction_mask)

def _phase_difference(c0, s0, c1, s1):
    """Unit-vector representation of angle1-angle0 without atan2/wrapping."""
    delta_cos = c1 * c0 + s1 * s0
    delta_sin = s1 * c0 - c1 * s0
    return delta_cos, delta_sin

class TemporalPhaseDifferenceLoss(nn.Module):
    """Circular loss on adjacent-time phase differences (IAF-like structure)."""

    def forward(self, pred_cos, pred_sin, target_phase, prediction_mask=None):
        if pred_cos.size(-1) < 2:
            return pred_cos.new_tensor(0.0)
        target_cos = torch.cos(target_phase.to(pred_cos.dtype))
        target_sin = torch.sin(target_phase.to(pred_sin.dtype))

        pc, ps = _phase_difference(
            pred_cos[..., :-1], pred_sin[..., :-1],
            pred_cos[..., 1:], pred_sin[..., 1:],
        )
        tc, ts = _phase_difference(
            target_cos[..., :-1], target_sin[..., :-1],
            target_cos[..., 1:], target_sin[..., 1:],
        )
        loss = 1.0 - (pc * tc + ps * ts)

        pair_mask = None
        if prediction_mask is not None:
            # Include within-gap transitions and both gap boundaries, while
            # excluding transitions whose two endpoints are both observed.
            pm = prediction_mask.to(dtype=loss.dtype, device=loss.device)
            pair_mask = torch.maximum(pm[..., :-1], pm[..., 1:])
        return _masked_mean(loss, pair_mask)

class FrequencyPhaseDifferenceLoss(nn.Module):
    """Circular loss on adjacent-frequency phase differences (GD-like structure)."""

    def forward(self, pred_cos, pred_sin, target_phase, prediction_mask=None):
        if pred_cos.size(1) < 2:
            return pred_cos.new_tensor(0.0)
        target_cos = torch.cos(target_phase.to(pred_cos.dtype))
        target_sin = torch.sin(target_phase.to(pred_sin.dtype))

        pc, ps = _phase_difference(
            pred_cos[:, :-1, :], pred_sin[:, :-1, :],
            pred_cos[:, 1:, :], pred_sin[:, 1:, :],
        )
        tc, ts = _phase_difference(
            target_cos[:, :-1, :], target_sin[:, :-1, :],
            target_cos[:, 1:, :], target_sin[:, 1:, :],
        )
        loss = 1.0 - (pc * tc + ps * ts)
        return _masked_mean(loss, prediction_mask)

class ComplexSpectrumConsistencyLoss(nn.Module):
    """Relative complex-spectrum L1 loss in Cartesian coordinates."""

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred_mag, pred_cos, pred_sin,
                target_mag, target_phase, prediction_mask=None,):

        target_mag = target_mag.to(device=pred_mag.device, dtype=pred_mag.dtype)
        target_phase = target_phase.to(device=pred_mag.device, dtype=pred_mag.dtype)

        pred_cos = pred_cos.to(pred_mag.dtype)
        pred_sin = pred_sin.to(pred_mag.dtype)

        target_cos = torch.cos(target_phase)
        target_sin = torch.sin(target_phase)

        pred_real = pred_mag * pred_cos
        pred_imag = pred_mag * pred_sin

        target_real = target_mag * target_cos
        target_imag = target_mag * target_sin

        error = ((pred_real - target_real).abs() + (pred_imag - target_imag).abs())

        if prediction_mask is None:
            mask = torch.ones_like(error)
        else:
            mask = prediction_mask.to(device=error.device, dtype=error.dtype).unsqueeze(1).expand_as(error)

        numerator = (error * mask).sum()
        denominator = (target_mag.abs() * mask).sum().clamp_min(self.eps)

        return numerator / denominator
# --------------------------------------------------------------------------------------
# parallel magnitude/phase PLC losses
# --------------------------------------------------------------------------------------

class MagnitudeReconstructionLoss(nn.Module):
    """Masked L1 in the power-compressed STFT-magnitude domain.

    ``pred_mag_compressed`` is the magnitude-head output, representing A**c.
    The clean physical target magnitude is compressed with the same exponent
    before comparison.  c=0.3 is the default; c=1.0 is the no-compression ablation.
    """

    def __init__(self, compression: float = 0.3, eps: float = 1e-8):
        super().__init__()
        if compression <= 0.0:
            raise ValueError("compression must be > 0")
        self.compression = float(compression)
        self.eps = float(eps)

    def forward(self, pred_mag_compressed, target_mag, prediction_mask=None):
        target_mag = target_mag.to(
            device=pred_mag_compressed.device, dtype=pred_mag_compressed.dtype
        ).clamp_min(0.0)
        target_compressed = target_mag.clamp_min(self.eps).pow(self.compression)
        return _masked_mean(
            (pred_mag_compressed - target_compressed).abs(), prediction_mask
        )


def istft_overlap_add(complex_spec: torch.Tensor, n_fft: int = 512,
                  win_length: int = 400, hop_length: int = 160,
                  pad: int = 176, eps: float = 1e-8) -> torch.Tensor:
    """Differentiable overlap-add iSTFT matching AV_PLC's current STFT geometry.

    PyTorch's ``istft(center=False)`` can reject zero-ended Hann coverage at the
    padded boundaries.  The dataset intentionally pads 176 samples at both ends,
    so we perform explicit overlap-add and crop those padding samples afterward.
    """
    if complex_spec.dim() != 3:
        raise ValueError(f"complex_spec must be [B,F,T], got {tuple(complex_spec.shape)}")
    if complex_spec.size(1) != n_fft // 2 + 1:
        raise ValueError(f"Expected {n_fft // 2 + 1} frequency bins")

    b, _, t = complex_spec.shape
    frames = torch.fft.irfft(complex_spec.transpose(1, 2), n=n_fft, dim=-1)  # [B,T,N]

    win = torch.hann_window(win_length, periodic=True, device=frames.device, dtype=frames.dtype)
    left = (n_fft - win_length) // 2
    right = n_fft - win_length - left
    win = torch.nn.functional.pad(win, (left, right))

    out_len = n_fft + hop_length * (t - 1)
    output = frames.new_zeros((b, out_len))
    denom = frames.new_zeros((out_len,))
    win_sq = win.square()

    for i in range(t):
        start = i * hop_length
        output[:, start:start + n_fft] = output[:, start:start + n_fft] + frames[:, i, :] * win
        denom[start:start + n_fft] = denom[start:start + n_fft] + win_sq

    output = output / denom.clamp_min(eps).unsqueeze(0)
    if pad > 0:
        if output.size(-1) <= 2 * pad:
            raise ValueError("iSTFT output is shorter than requested boundary crop")
        output = output[:, pad:-pad]
    return output


class WaveformReconstructionLoss(nn.Module):
    """Waveform L1 after differentiable magnitude+phase synthesis."""

    def __init__(self, n_fft: int = 512, win_length: int = 400,
                 hop_length: int = 160, pad: int = 176, eps: float = 1e-8):
        super().__init__()
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.pad = int(pad)
        self.eps = float(eps)

    def forward(self, pred_mag, pred_cos, pred_sin, target_mag, target_phase):
        target_mag = target_mag.to(device=pred_mag.device, dtype=pred_mag.dtype)
        target_phase = target_phase.to(device=pred_mag.device, dtype=pred_mag.dtype)
        pred_complex = torch.complex(pred_mag * pred_cos, pred_mag * pred_sin)
        target_complex = torch.polar(target_mag, target_phase)

        pred_wav = istft_overlap_add(
            pred_complex, self.n_fft, self.win_length, self.hop_length, self.pad, self.eps
        )
        target_wav = istft_overlap_add(
            target_complex, self.n_fft, self.win_length, self.hop_length, self.pad, self.eps
        )
        return torch.nn.functional.l1_loss(pred_wav, target_wav)

# MP-SENet-style anti-wrapping phase losses used by the new parallel M/P stage.
def anti_wrapping_error(delta: torch.Tensor) -> torch.Tensor:
    two_pi = delta.new_tensor(2.0 * torch.pi)
    return (delta - two_pi * torch.round(delta / two_pi)).abs()


class InstantaneousPhaseLoss(nn.Module):
    """L_IP: anti-wrapped pointwise phase error."""
    def forward(self, pred_phase, target_phase, prediction_mask=None):
        target_phase = target_phase.to(device=pred_phase.device, dtype=pred_phase.dtype)
        return _masked_mean(anti_wrapping_error(pred_phase - target_phase), prediction_mask)


class GroupDelayPhaseLoss(nn.Module):
    """L_GD: anti-wrapped adjacent-frequency phase-difference error."""
    def forward(self, pred_phase, target_phase, prediction_mask=None):
        target_phase = target_phase.to(device=pred_phase.device, dtype=pred_phase.dtype)
        if pred_phase.size(1) < 2:
            return pred_phase.new_tensor(0.0)
        pred_delta = pred_phase[:, 1:, :] - pred_phase[:, :-1, :]
        target_delta = target_phase[:, 1:, :] - target_phase[:, :-1, :]
        return _masked_mean(anti_wrapping_error(pred_delta - target_delta), prediction_mask)


class InstantaneousAngularFrequencyLoss(nn.Module):
    """L_IAF: anti-wrapped adjacent-time phase-difference error."""
    def forward(self, pred_phase, target_phase, prediction_mask=None):
        target_phase = target_phase.to(device=pred_phase.device, dtype=pred_phase.dtype)
        if pred_phase.size(-1) < 2:
            return pred_phase.new_tensor(0.0)
        pred_delta = pred_phase[..., 1:] - pred_phase[..., :-1]
        target_delta = target_phase[..., 1:] - target_phase[..., :-1]
        pair_mask = None
        if prediction_mask is not None:
            pm = prediction_mask.to(device=pred_phase.device, dtype=pred_phase.dtype)
            pair_mask = torch.maximum(pm[..., :-1], pm[..., 1:])
        return _masked_mean(anti_wrapping_error(pred_delta - target_delta), pair_mask)
