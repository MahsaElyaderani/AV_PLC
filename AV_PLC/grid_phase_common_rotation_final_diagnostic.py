#!/usr/bin/env python3
"""Final no-training phase diagnostic for AV_PLC on GRID.

This script is deliberately decision-oriented.  It is intended to be the last
oracle/diagnostic round before the next magnitude+phase architecture change.

What the previous two oracle rounds already established
--------------------------------------------------------
1) A2/B/C oracle:
   - A2 = phase-model completed Mel + Griffin-Lim
   - B  = exact same Mel + learned phase
   - C  = exact same Mel + GT phase in the PLC gap
   C >> B and C > A2 showed that useful phase exists, but the learned phase is
   the bottleneck.  Learned phase tended to improve STOI while hurting PLCMOS.

2) F follow-up oracle:
   - A0 = no-phase-model completed Mel + Griffin-Lim
   - F  = exact same no-phase-model Mel + GT phase
   F > A0 showed that the successful no-phase Mel path can benefit from correct
   phase.  F > C in PESQ/PLCMOS showed that the current joint phase path also
   degrades Mel quality.  Phase-error diagnostics showed approximately random
   absolute/unit and temporal phase, but substantially better frequency-relative
   phase (especially at energetic bins).

Question answered here
----------------------
Does the pattern above come mainly from a frame-dependent common phase rotation?

For every missing frame t, define the unit-vector phase error

    z_err[f,t] = exp(j*(phi_hat[f,t] - phi_gt[f,t])).

Using exact GT STFT magnitude A_gt as a weight, estimate the best single rotation

    alpha_t = angle(sum_f A_gt[f,t] * z_err[f,t])

and its circular concentration

    R_t = |sum_f A_gt[f,t] * z_err[f,t]| / sum_f A_gt[f,t].

R_t close to 1 means the important frequency bins agree on one common rotation;
R_t close to 0 means no single rotation explains the frame.

The script then creates an oracle rotation-corrected condition:

    G = phase-model completed Mel + learned phase corrected by -alpha_t
        only inside the missing gap.

The same Mel is used for A2/B/G/C.  Therefore:

    G - B : perceptual benefit from removing ONLY the best common rotation
    C - G : residual phase headroom after common rotation is removed
    G - A2: whether rotation-corrected learned phase can beat Griffin-Lim

It also recomputes the current three phase errors BEFORE and AFTER correction:
unit, temporal, and frequency.  A true frame-common rotation correction MUST leave
frequency-difference error unchanged (up to floating-point noise).  If it does
not, the diagnostic aborts.

Outputs
-------
  common_rotation_metric_summary.csv
      A2/B/G/C metrics and controlled deltas for every method/gap.

  common_rotation_phase_summary.csv
      Before/after unit, temporal, frequency phase errors plus concentration
      statistics and sanity checks.

  common_rotation_headroom.csv
      Fraction of B->C oracle phase headroom recovered by G for PESQ/PLCMOS/
      STOI/ESTOI/WER/CER.

  common_rotation_temporal_summary.csv
      Whether frame-to-frame changes in alpha_t explain the temporal phase-error
      phasor, including gap-boundary transitions.

  common_rotation_frame_summary.csv
      Per method/gap distribution summaries of alpha_t and R_t.  Raw per-frame
      rows are optional with --save-frame-details.

  common_rotation_frames.csv              (optional)
  common_rotation_temporal_pairs.csv      (optional)
  implementation_decision.txt

No checkpoint is modified and no training occurs.

Default command
---------------
    python AV_PLC/grid_phase_common_rotation_final_diagnostic.py
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import math
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------------------
# Repository discovery
# --------------------------------------------------------------------------------------
HERE = Path(__file__).resolve()
ROOT = None
for p in (HERE.parent, *HERE.parents):
    if (p / "AV_PLC").is_dir() and (p / "evaluations").is_dir():
        ROOT = p
        break
if ROOT is None:
    raise RuntimeError(
        "Copy this script into (or below) the AV_PLC project repository, e.g. "
        "AV_PLC/grid_phase_common_rotation_final_diagnostic.py"
    )
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluations.runtime_config import (  # noqa: E402
    SEED,
    project_checkpoint_dir,
    project_log_dir,
    set_global_seed,
)
from AV_PLC.av_dataloader import AVDataloader  # noqa: E402
from AV_PLC.multimodal_decoder import AV_PLC  # noqa: E402
from shared.audio_processing import (  # noqa: E402
    hop_len,
    n_fft,
    read_gt_input,
    sample_rate,
    torch_melphase2audio,
    win_len,
)
from shared.metrics import calculate_batch_metrics  # noqa: E402

# --------------------------------------------------------------------------------------
# Controlled experiment constants: match the previous two diagnostics.
# --------------------------------------------------------------------------------------
DATASET = "grid"
ARCH = "latent_spectral_v2"
METHODS = ("concat", "temporal_self_cross_attention", "global_local_affinity")
GAPS = (160, 500, 1000)
EPS = 1e-8

MODEL_ARGS = dict(
    mel_dim=80,
    feat_dim=256,
    dropout=0.1,
    video_depth=6,
    video_heads=4,
    video_hidden_size=256,
    audio_depth=4,
    audio_heads=4,
    audio_hidden_size=256,
    audio_ckpt_path=None,
    freeze_audio_enc=False,
)

# A secondary robustness view: bins >= -20 dB from each FRAME's GT STFT peak.
# The primary alpha/R estimate still uses every bin weighted by exact GT magnitude.
HIGH_ENERGY_DB = -20.0


# --------------------------------------------------------------------------------------
# Project/model/data helpers
# --------------------------------------------------------------------------------------
def model_name(method: str, phase_variant: str = "complex_v1") -> str:
    base = (
        f"av_plc_{method}_{ARCH}_frozen_enc_masked_mel_ge_train"
        f"_no_jitter({DATASET})"
    )
    if phase_variant == "plain":
        return base + "_phase_reconstruction"
    if phase_variant == "complex_v1":
        return base + "_phase_reconstruction_complex_v1"
    raise ValueError(f"Unsupported phase variant: {phase_variant}")


def build_model(method: str, phase_variant: str, checkpoint: str, device: torch.device):
    kwargs = dict(
        **MODEL_ARGS,
        fusion_type=method,
        phase_reconstruction=True,
    )
    if method == "global_local_affinity":
        kwargs.update(
            affinity_dim=128,
            max_av_offset=16,
            global_temperature=0.1,
            local_temperature=0.1,
            prior_strength=1.0,
            prior_sigma=2.0,
            min_offset_support=4.0,
        )

    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)

    model = AV_PLC(**kwargs).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get(
        "model_state_dict",
        ckpt.get("model_state", ckpt.get("state_dict", ckpt)),
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def make_loader(args, gap_ms: int):
    factory = AVDataloader(
        dataset_name=DATASET,
        mode="av",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dropout_modality=False,
        video_aug=False,
        test_subset=args.test_subset,
        temporal_jitter=False,
        jitter_p=1.0,
        jitter_max_frames=8,
        phase_reconstruction=True,
    )
    return factory.test_dataloader(
        mask_range="10",  # ignored by deterministic single-gap generation
        seed=SEED,
        mask_type="single_gap",
        gap_ms=gap_ms,
    )


def dataset_stats(loader):
    ds = loader.dataset
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    return (
        float(getattr(ds, "mel_mean", -56.775)),
        float(getattr(ds, "mel_std", 19.707)),
    )


def unpack(batch, device: torch.device):
    if len(batch) != 12:
        raise ValueError(
            "Expected the current phase-enabled AV batch with 12 elements; "
            f"got {len(batch)}."
        )
    (
        visual,
        spk,
        masked,
        spec,
        video_spec,
        phase,
        video_phase,
        length,
        text,
        mask,
        path,
        avail,
    ) = batch
    del video_spec, video_phase, avail

    return (
        visual.float().to(device, non_blocking=True),
        spk.float().to(device, non_blocking=True),
        masked.float().to(device, non_blocking=True),
        spec.float().to(device, non_blocking=True),
        phase.float().to(device, non_blocking=True),
        length.long().to(device, non_blocking=True),
        text,
        mask.float().to(device, non_blocking=True),
        path,
    )


def signature(path, mask):
    m = mask.detach().cpu().numpy().astype(np.float32, copy=False)
    return [
        f"{p}::{hashlib.sha256(m[i].tobytes()).hexdigest()[:16]}"
        for i, p in enumerate(path)
    ]


def force_av(batch_size: int, device: torch.device):
    return torch.tensor([True, True], dtype=torch.bool, device=device).unsqueeze(0).repeat(batch_size, 1)


def forward_av(model, visual, spk, masked, length, mask, phase):
    out = model(
        masked,
        visual,
        spk,
        length,
        avail=force_av(masked.size(0), masked.device),
        audio_mask=mask,
        phase=phase,
    )
    if not isinstance(out, (tuple, list)) or len(out) != 4:
        raise RuntimeError("Expected latent_spectral_v2 AV_PLC to return four outputs.")
    completion = out[3]
    if completion is None:
        raise RuntimeError("Unified completion output is missing.")
    required = (
        "completed_mel", "prediction_mask",
        "pred_cos", "pred_sin", "final_cos", "final_sin",
    )
    for key in required:
        if completion.get(key) is None:
            raise RuntimeError(f"completion_output['{key}'] is missing.")
    return completion


# --------------------------------------------------------------------------------------
# Metrics / CSV helpers
# --------------------------------------------------------------------------------------
class MetricAcc:
    def __init__(self):
        self.s = defaultdict(float)
        self.n = defaultdict(int)

    def add(self, values, batch_size: int):
        for k, v in values.items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                self.s[k] += float(v) * batch_size
                self.n[k] += batch_size

    def result(self):
        return {k: self.s[k] / self.n[k] for k in self.s if self.n[k] > 0}


def metric_batch(spec, recon_mel, text, mask, path, mel_mean, mel_std, recon_audio=None):
    kwargs = dict(
        original_batch=spec.detach().cpu(),
        reconstructed_batch=recon_mel.detach().cpu(),
        texts=text,
        mask=mask.detach().cpu(),
        path=list(path),
        hifigan_vocoder=None,
        tokenizer=None,
        max_samples=spec.size(0),
        sample_rate=sample_rate,
        mel_mean=mel_mean,
        mel_std=mel_std,
        masked_input=False,
    )
    if recon_audio is not None:
        kwargs["reconstructed_audio_batch"] = recon_audio.detach().cpu()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        return calculate_batch_metrics(**kwargs)


def save_csv(rows, path: Path, fixed=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # Still create an empty file so the output contract is explicit.
        path.write_text("", encoding="utf-8")
        return
    fixed = list(fixed or [])
    all_keys = {k for r in rows for k in r}
    rest = sorted(all_keys - set(fixed))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fixed + rest)
        w.writeheader()
        w.writerows(rows)


def delta(new, old):
    out = {}
    for k in set(new) & set(old):
        try:
            a, b = float(new[k]), float(old[k])
        except (TypeError, ValueError):
            continue
        if np.isfinite(a) and np.isfinite(b):
            out[k] = a - b
    return out


def metric_row(method, gap, condition, values, comparison=""):
    r = dict(method=method, gap_ms=int(gap), condition=condition, comparison=comparison)
    r.update(values)
    return r


def quantiles(values):
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {
            "mean": float("nan"), "std": float("nan"),
            "p10": float("nan"), "p25": float("nan"), "median": float("nan"),
            "p75": float("nan"), "p90": float("nan"),
        }
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "p10": float(np.quantile(a, 0.10)),
        "p25": float(np.quantile(a, 0.25)),
        "median": float(np.quantile(a, 0.50)),
        "p75": float(np.quantile(a, 0.75)),
        "p90": float(np.quantile(a, 0.90)),
    }


# --------------------------------------------------------------------------------------
# Exact GT STFT magnitude (same geometry validated by the E oracle)
# --------------------------------------------------------------------------------------
def true_stft_batch(path, mask, stored_phase, device):
    pad = (n_fft - hop_len) // 2  # 176 for current setup
    window = torch.hann_window(
        win_len, periodic=True, dtype=torch.float32, device=device
    )

    wavs = []
    for i, p in enumerate(path):
        original, _ = read_gt_input(str(p), mask[i])
        wavs.append(torch.as_tensor(original, dtype=torch.float32, device=device))
    lengths = {w.numel() for w in wavs}
    if len(lengths) != 1:
        raise RuntimeError(f"Expected fixed 3-s waveforms, got lengths={sorted(lengths)}")
    wav = torch.stack(wavs, dim=0)

    stft = torch.stft(
        F.pad(wav, (pad, pad)),
        n_fft=n_fft,
        hop_length=hop_len,
        win_length=win_len,
        window=window,
        center=False,
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    if tuple(stft.shape) != tuple(stored_phase.shape):
        raise RuntimeError(
            f"GT STFT shape {tuple(stft.shape)} != stored phase shape {tuple(stored_phase.shape)}"
        )

    mag = stft.abs()
    fresh_phase = torch.angle(stft)
    d = stored_phase.to(device) - fresh_phase
    d = torch.atan2(torch.sin(d), torch.cos(d)).abs()
    alignment = (d * mag).sum() / mag.sum().clamp_min(EPS)
    return mag, float(alignment.item())


# --------------------------------------------------------------------------------------
# Circular / phase math
# --------------------------------------------------------------------------------------
def normalize_unit(c, s):
    n = torch.sqrt(c.square() + s.square() + EPS)
    return c / n, s / n


def error_phasor(pred_cos, pred_sin, target_cos, target_sin):
    """Return cos(delta), sin(delta), where delta=pred_phase-target_phase."""
    dot = pred_cos * target_cos + pred_sin * target_sin
    cross = pred_sin * target_cos - pred_cos * target_sin
    return dot.clamp(-1.0, 1.0), cross


def circular_abs_from_phasor(dot, cross):
    return torch.atan2(cross, dot).abs()


def phase_difference(c0, s0, c1, s1):
    """Unit vector for angle1-angle0, matching AV_PLC/losses.py."""
    return (
        c1 * c0 + s1 * s0,
        s1 * c0 - c1 * s0,
    )


def rotate_by_minus_alpha(c, s, alpha_bt):
    """Apply exp(-j*alpha_t) to every frequency bin in frame t."""
    ca = torch.cos(alpha_bt).unsqueeze(1)
    sa = torch.sin(alpha_bt).unsqueeze(1)
    out_c = c * ca + s * sa
    out_s = s * ca - c * sa
    return normalize_unit(out_c, out_s)


def estimate_frame_rotation(pred_cos, pred_sin, target_phase, gt_mag, prediction_mask):
    """Estimate oracle frame-common rotation alpha_t and concentration R_t.

    Primary estimate uses exact GT STFT magnitude over all 257 bins.
    A secondary R_high is calculated only on bins >= -20 dB from that frame's
    GT STFT peak (still magnitude weighted).
    """
    target_cos = torch.cos(target_phase).float()
    target_sin = torch.sin(target_phase).float()
    pred_cos = pred_cos.float()
    pred_sin = pred_sin.float()
    mag = gt_mag.float().clamp_min(0.0)

    dot, cross = error_phasor(pred_cos, pred_sin, target_cos, target_sin)

    # A frame with zero (or numerically negligible) GT STFT magnitude has no
    # physically meaningful target phase orientation.  Do NOT manufacture a
    # rotation estimate by dividing 0 by eps: mark such frames invalid for the
    # rotation statistics and leave their learned phase unchanged in G.
    raw_wsum = mag.sum(dim=1)
    valid_energy = raw_wsum > EPS
    wsum = raw_wsum.clamp_min(EPS)
    mean_c = (mag * dot).sum(dim=1) / wsum
    mean_s = (mag * cross).sum(dim=1) / wsum
    R_raw = torch.sqrt(mean_c.square() + mean_s.square()).clamp(0.0, 1.0)
    alpha_raw = torch.atan2(mean_s, mean_c)
    R = torch.where(valid_energy, R_raw, torch.full_like(R_raw, float("nan")))
    alpha = torch.where(valid_energy, alpha_raw, torch.zeros_like(alpha_raw))

    frame_peak = mag.amax(dim=1, keepdim=True).clamp_min(EPS)
    threshold = frame_peak * (10.0 ** (HIGH_ENERGY_DB / 20.0))
    high_mask = mag >= threshold
    high_w = mag * high_mask
    high_wsum = high_w.sum(dim=1)
    high_c = (high_w * dot).sum(dim=1) / high_wsum.clamp_min(EPS)
    high_s = (high_w * cross).sum(dim=1) / high_wsum.clamp_min(EPS)
    R_high = torch.sqrt(high_c.square() + high_s.square()).clamp(0.0, 1.0)
    R_high = torch.where(high_wsum > EPS, R_high, torch.full_like(R_high, float("nan")))
    high_count = high_mask.sum(dim=1)

    # Weighted unit circular loss before correction is exactly 1-mean_c on
    # frames where phase is defined.  Silent/zero-energy frames are NaN here so
    # they cannot bias concentration/error summaries.
    circ_before_raw = 1.0 - mean_c
    circ_before = torch.where(
        valid_energy, circ_before_raw, torch.full_like(circ_before_raw, float("nan"))
    )
    # The optimal common rotation changes mean_c to R_raw; its minimum circular
    # loss is 1-R_raw.
    circ_after_opt_raw = 1.0 - R_raw

    angle_before = circular_abs_from_phasor(dot, cross)
    angle_before_w_raw = (mag * angle_before).sum(dim=1) / wsum
    angle_before_w = torch.where(
        valid_energy, angle_before_w_raw, torch.full_like(angle_before_w_raw, float("nan"))
    )

    # alpha is meaningful only on missing frames with non-negligible GT energy.
    # Observed frames and silent missing frames get alpha=0, so G leaves them
    # unchanged.  This is scientifically preferable to imposing an arbitrary
    # oracle rotation where GT phase itself is undefined/irrelevant.
    missing = prediction_mask > 0.5
    missing_valid = missing & valid_energy
    alpha_for_correction = torch.where(
        missing_valid, alpha_raw, torch.zeros_like(alpha_raw)
    )

    corr_cos, corr_sin = rotate_by_minus_alpha(pred_cos, pred_sin, alpha_for_correction)
    dot2, cross2 = error_phasor(corr_cos, corr_sin, target_cos, target_sin)
    angle_after = circular_abs_from_phasor(dot2, cross2)
    circ_after_raw = (mag * (1.0 - dot2)).sum(dim=1) / wsum
    angle_after_w_raw = (mag * angle_after).sum(dim=1) / wsum
    circ_after = torch.where(
        valid_energy, circ_after_raw, torch.full_like(circ_after_raw, float("nan"))
    )
    angle_after_w = torch.where(
        valid_energy, angle_after_w_raw, torch.full_like(angle_after_w_raw, float("nan"))
    )

    # This equality is a useful internal check on the circular-mean math, but it
    # is mathematically defined only when the frame has non-zero spectral weight.
    if missing_valid.any():
        max_formula_err = (
            circ_after_raw[missing_valid] - circ_after_opt_raw[missing_valid]
        ).abs().max().item()
    else:
        max_formula_err = 0.0
    if max_formula_err > 2e-5:
        raise RuntimeError(
            "Frame-rotation math self-consistency failed: "
            f"max |recomputed_after-(1-R)|={max_formula_err:.3e}"
        )

    return {
        "alpha": alpha_for_correction,
        "valid_energy": valid_energy,
        "R": R,
        "R_high": R_high,
        "high_count": high_count,
        "circ_before_w": circ_before,
        "circ_after_w": circ_after,
        "angle_before_w": angle_before_w,
        "angle_after_w": angle_after_w,
        "corrected_pred_cos": corr_cos,
        "corrected_pred_sin": corr_sin,
        "error_dot": dot,
        "error_cross": cross,
    }


def build_corrected_final(completion, corrected_pred_cos, corrected_pred_sin, prediction_mask):
    """Keep observed final phase exact; use rotation-corrected raw prediction in gaps."""
    final_cos = completion["final_cos"].float()
    final_sin = completion["final_sin"].float()
    q = prediction_mask.float().unsqueeze(1)
    c = (1.0 - q) * final_cos + q * corrected_pred_cos
    s = (1.0 - q) * final_sin + q * corrected_pred_sin
    return normalize_unit(c, s)


def build_gt_gap_final(completion, target_phase, prediction_mask):
    """Condition C: identical completed Mel, GT phase only in missing frames."""
    final_cos = completion["final_cos"].float()
    final_sin = completion["final_sin"].float()
    gt_cos = torch.cos(target_phase).float()
    gt_sin = torch.sin(target_phase).float()
    q = prediction_mask.float().unsqueeze(1)
    c = (1.0 - q) * final_cos + q * gt_cos
    s = (1.0 - q) * final_sin + q * gt_sin
    return normalize_unit(c, s)


# --------------------------------------------------------------------------------------
# Generic phase-error accumulation before/after rotation
# --------------------------------------------------------------------------------------
class ErrorStat:
    def __init__(self):
        self.count = 0
        self.sum_circ = 0.0
        self.sum_angle = 0.0
        self.sum_mag = 0.0
        self.sum_mag_circ = 0.0
        self.sum_mag_angle = 0.0

    def add(self, circ, angle, mag, valid):
        circ = circ[valid].double()
        angle = angle[valid].double()
        mag = mag[valid].double().clamp_min(0.0)
        if circ.numel() == 0:
            return
        self.count += int(circ.numel())
        self.sum_circ += float(circ.sum().item())
        self.sum_angle += float(angle.sum().item())
        self.sum_mag += float(mag.sum().item())
        self.sum_mag_circ += float((mag * circ).sum().item())
        self.sum_mag_angle += float((mag * angle).sum().item())

    def values(self):
        if self.count == 0:
            return {
                "count": 0,
                "circular_error_mean": float("nan"),
                "angular_mae_rad": float("nan"),
                "angular_mae_deg": float("nan"),
                "magnitude_weighted_circular_error": float("nan"),
                "magnitude_weighted_angular_mae_rad": float("nan"),
                "magnitude_weighted_angular_mae_deg": float("nan"),
            }
        circ = self.sum_circ / self.count
        ang = self.sum_angle / self.count
        if self.sum_mag > 0:
            wc = self.sum_mag_circ / self.sum_mag
            wa = self.sum_mag_angle / self.sum_mag
        else:
            wc = wa = float("nan")
        return {
            "count": self.count,
            "circular_error_mean": circ,
            "angular_mae_rad": ang,
            "angular_mae_deg": ang * 180.0 / math.pi,
            "magnitude_weighted_circular_error": wc,
            "magnitude_weighted_angular_mae_rad": wa,
            "magnitude_weighted_angular_mae_deg": wa * 180.0 / math.pi,
        }


def compute_error_tensors(pred_cos, pred_sin, target_phase, prediction_mask, gt_mag, *, final=False):
    target_cos = torch.cos(target_phase).float()
    target_sin = torch.sin(target_phase).float()
    pred_cos = pred_cos.float()
    pred_sin = pred_sin.float()

    # Unit/absolute phase.
    u_dot, u_cross = error_phasor(pred_cos, pred_sin, target_cos, target_sin)
    unit_circ = 1.0 - u_dot
    unit_ang = circular_abs_from_phasor(u_dot, u_cross)
    unit_mask = prediction_mask.bool().unsqueeze(1).expand_as(unit_circ)

    # Temporal phase difference.
    pdc, pds = phase_difference(
        pred_cos[..., :-1], pred_sin[..., :-1],
        pred_cos[..., 1:], pred_sin[..., 1:],
    )
    tdc, tds = phase_difference(
        target_cos[..., :-1], target_sin[..., :-1],
        target_cos[..., 1:], target_sin[..., 1:],
    )
    t_dot, t_cross = error_phasor(pdc, pds, tdc, tds)
    temporal_circ = 1.0 - t_dot
    temporal_ang = circular_abs_from_phasor(t_dot, t_cross)
    pair = torch.maximum(prediction_mask[..., :-1], prediction_mask[..., 1:]).bool()
    temporal_mask = pair.unsqueeze(1).expand_as(temporal_circ)
    temporal_mag = 0.5 * (gt_mag[..., :-1] + gt_mag[..., 1:])

    # Frequency phase difference.
    pfc, pfs = phase_difference(
        pred_cos[:, :-1, :], pred_sin[:, :-1, :],
        pred_cos[:, 1:, :], pred_sin[:, 1:, :],
    )
    tfc, tfs = phase_difference(
        target_cos[:, :-1, :], target_sin[:, :-1, :],
        target_cos[:, 1:, :], target_sin[:, 1:, :],
    )
    f_dot, f_cross = error_phasor(pfc, pfs, tfc, tfs)
    frequency_circ = 1.0 - f_dot
    frequency_ang = circular_abs_from_phasor(f_dot, f_cross)
    frequency_mask = prediction_mask.bool().unsqueeze(1).expand_as(frequency_circ)
    frequency_mag = 0.5 * (gt_mag[:, :-1, :] + gt_mag[:, 1:, :])

    return {
        "unit": (unit_circ, unit_ang, unit_mask, gt_mag),
        "temporal": (temporal_circ, temporal_ang, temporal_mask, temporal_mag),
        "frequency": (frequency_circ, frequency_ang, frequency_mask, frequency_mag),
    }


# --------------------------------------------------------------------------------------
# Frame/temporal summary accumulators
# --------------------------------------------------------------------------------------
class RotationCollector:
    def __init__(self, save_details=False):
        self.save_details = bool(save_details)
        self.frame_values = defaultdict(lambda: defaultdict(list))
        self.temporal_values = defaultdict(lambda: defaultdict(list))
        self.frame_rows = []
        self.temporal_rows = []

    def add_frames(self, method, gap, path, prediction_mask, gt_mag, rot):
        key = (method, int(gap))
        B, T = prediction_mask.shape
        for b in range(B):
            idx = torch.nonzero(prediction_mask[b] > 0.5, as_tuple=True)[0]
            if idx.numel() == 0:
                raise RuntimeError("Single-gap diagnostic found no missing frames.")
            if idx.numel() > 1 and not torch.all(idx[1:] == idx[:-1] + 1):
                raise RuntimeError("Expected one contiguous single gap per sample.")
            L = int(idx.numel())
            pos = torch.tensor([0.5], device=idx.device) if L == 1 else torch.linspace(0, 1, L, device=idx.device)
            depth = 2.0 * torch.minimum(pos, 1.0 - pos)

            for k, t in enumerate(idx.tolist()):
                valid_energy = bool(rot["valid_energy"][b, t].item())
                mag_sum = float(gt_mag[b, :, t].sum().item())
                vals = self.frame_values[key]
                vals["energy_valid_flag"].append(1.0 if valid_energy else 0.0)
                vals["frame_mag_sum_all"].append(mag_sum)

                # When GT magnitude is zero, phase orientation is undefined.
                # Keep a record of its occurrence, but exclude it from all alpha/R
                # and phase-error statistics so silence cannot masquerade as either
                # strong or weak evidence for common rotation.
                if not valid_energy:
                    if self.save_details:
                        self.frame_rows.append({
                            "method": method,
                            "gap_ms": int(gap),
                            "path": str(path[b]),
                            "frame_index": int(t),
                            "gap_position": float(pos[k].item()),
                            "gap_depth": float(depth[k].item()),
                            "valid_energy_frame": False,
                            "gt_magnitude_sum": mag_sum,
                        })
                    continue

                alpha = float(rot["alpha"][b, t].item())
                Rv = float(rot["R"][b, t].item())
                Rh = float(rot["R_high"][b, t].item())
                before_c = float(rot["circ_before_w"][b, t].item())
                after_c = float(rot["circ_after_w"][b, t].item())
                before_a = float(rot["angle_before_w"][b, t].item())
                after_a = float(rot["angle_after_w"][b, t].item())
                reduction = (before_c - after_c) / max(before_c, EPS)

                vals["alpha_abs_deg"].append(abs(alpha) * 180.0 / math.pi)
                vals["R"].append(Rv)
                vals["R_high"].append(Rh)
                vals["circ_before_w"].append(before_c)
                vals["circ_after_w"].append(after_c)
                vals["angle_before_w_deg"].append(before_a * 180.0 / math.pi)
                vals["angle_after_w_deg"].append(after_a * 180.0 / math.pi)
                vals["circular_reduction_fraction"].append(reduction)
                vals["frame_mag_sum"].append(mag_sum)

                if self.save_details:
                    self.frame_rows.append({
                        "method": method,
                        "gap_ms": int(gap),
                        "path": str(path[b]),
                        "frame_index": int(t),
                        "gap_position": float(pos[k].item()),
                        "gap_depth": float(depth[k].item()),
                        "valid_energy_frame": True,
                        "alpha_rad": alpha,
                        "alpha_deg": alpha * 180.0 / math.pi,
                        "alpha_abs_deg": abs(alpha) * 180.0 / math.pi,
                        "R_concentration": Rv,
                        "R_high_energy": Rh,
                        "high_energy_bin_count": int(rot["high_count"][b, t].item()),
                        "weighted_unit_circular_before": before_c,
                        "weighted_unit_circular_after": after_c,
                        "weighted_unit_angle_before_deg": before_a * 180.0 / math.pi,
                        "weighted_unit_angle_after_deg": after_a * 180.0 / math.pi,
                        "circular_reduction_fraction": reduction,
                        "gt_magnitude_sum": mag_sum,
                    })

    def add_temporal_pairs(self, method, gap, path, prediction_mask, gt_mag,
                           target_phase, final_cos, final_sin, corrected_final_cos,
                           corrected_final_sin, alpha):
        key = (method, int(gap))
        target_cos = torch.cos(target_phase).float()
        target_sin = torch.sin(target_phase).float()

        # Temporal error phasor before correction.
        pdc, pds = phase_difference(
            final_cos[..., :-1], final_sin[..., :-1],
            final_cos[..., 1:], final_sin[..., 1:],
        )
        tdc, tds = phase_difference(
            target_cos[..., :-1], target_sin[..., :-1],
            target_cos[..., 1:], target_sin[..., 1:],
        )
        dot, cross = error_phasor(pdc, pds, tdc, tds)

        # After correction.
        cpdc, cpds = phase_difference(
            corrected_final_cos[..., :-1], corrected_final_sin[..., :-1],
            corrected_final_cos[..., 1:], corrected_final_sin[..., 1:],
        )
        cdot, ccross = error_phasor(cpdc, cpds, tdc, tds)

        pair_mask = torch.maximum(prediction_mask[:, :-1], prediction_mask[:, 1:]) > 0.5
        pair_mag = 0.5 * (gt_mag[..., :-1] + gt_mag[..., 1:])
        raw_wsum = pair_mag.sum(dim=1)
        valid_pair_energy = raw_wsum > EPS
        wsum = raw_wsum.clamp_min(EPS)

        mean_c = (pair_mag * dot).sum(dim=1) / wsum
        mean_s = (pair_mag * cross).sum(dim=1) / wsum
        R_tem_raw = torch.sqrt(mean_c.square() + mean_s.square()).clamp(0, 1)
        beta_raw = torch.atan2(mean_s, mean_c)  # best common temporal error across frequency
        R_tem = torch.where(
            valid_pair_energy, R_tem_raw, torch.full_like(R_tem_raw, float("nan"))
        )
        beta = torch.where(valid_pair_energy, beta_raw, torch.zeros_like(beta_raw))

        mean_c_after = (pair_mag * cdot).sum(dim=1) / wsum
        mean_s_after = (pair_mag * ccross).sum(dim=1) / wsum
        R_tem_after_raw = torch.sqrt(mean_c_after.square() + mean_s_after.square()).clamp(0, 1)
        beta_after_raw = torch.atan2(mean_s_after, mean_c_after)
        R_tem_after = torch.where(
            valid_pair_energy, R_tem_after_raw,
            torch.full_like(R_tem_after_raw, float("nan"))
        )
        beta_after = torch.where(
            valid_pair_energy, beta_after_raw, torch.zeros_like(beta_after_raw)
        )

        # Because alpha=0 on observed frames, this includes left/right gap boundaries.
        delta_alpha = torch.atan2(
            torch.sin(alpha[:, 1:] - alpha[:, :-1]),
            torch.cos(alpha[:, 1:] - alpha[:, :-1]),
        )
        beta_minus_da = torch.atan2(torch.sin(beta - delta_alpha), torch.cos(beta - delta_alpha)).abs()

        B, Tp = pair_mask.shape
        for b in range(B):
            for t in torch.nonzero(pair_mask[b], as_tuple=True)[0].tolist():
                left_missing = bool(prediction_mask[b, t] > 0.5)
                right_missing = bool(prediction_mask[b, t + 1] > 0.5)
                kind = (
                    "internal" if left_missing and right_missing
                    else "right_boundary" if left_missing
                    else "left_boundary"
                )
                vals = self.temporal_values[key]
                pair_valid = bool(valid_pair_energy[b, t].item())
                vals["energy_valid_pair_flag"].append(1.0 if pair_valid else 0.0)
                if not pair_valid:
                    if self.save_details:
                        self.temporal_rows.append({
                            "method": method,
                            "gap_ms": int(gap),
                            "path": str(path[b]),
                            "left_frame": int(t),
                            "right_frame": int(t + 1),
                            "transition_type": kind,
                            "valid_energy_pair": False,
                        })
                    continue

                vals["R_temporal"].append(float(R_tem[b, t].item()))
                vals["R_temporal_after"].append(float(R_tem_after[b, t].item()))
                vals["beta_abs_deg"].append(abs(float(beta[b, t].item())) * 180.0 / math.pi)
                vals["delta_alpha_abs_deg"].append(abs(float(delta_alpha[b, t].item())) * 180.0 / math.pi)
                vals["beta_minus_delta_alpha_abs_deg"].append(float(beta_minus_da[b, t].item()) * 180.0 / math.pi)
                vals["beta_after_abs_deg"].append(abs(float(beta_after[b, t].item())) * 180.0 / math.pi)
                vals[f"kind_{kind}"].append(1.0)

                if self.save_details:
                    self.temporal_rows.append({
                        "method": method,
                        "gap_ms": int(gap),
                        "path": str(path[b]),
                        "left_frame": int(t),
                        "right_frame": int(t + 1),
                        "transition_type": kind,
                        "valid_energy_pair": True,
                        "alpha_left_deg": float(alpha[b, t].item()) * 180.0 / math.pi,
                        "alpha_right_deg": float(alpha[b, t + 1].item()) * 180.0 / math.pi,
                        "delta_alpha_deg": float(delta_alpha[b, t].item()) * 180.0 / math.pi,
                        "temporal_error_common_angle_deg": float(beta[b, t].item()) * 180.0 / math.pi,
                        "temporal_error_concentration_R": float(R_tem[b, t].item()),
                        "beta_minus_delta_alpha_abs_deg": float(beta_minus_da[b, t].item()) * 180.0 / math.pi,
                        "after_rotation_common_angle_deg": float(beta_after[b, t].item()) * 180.0 / math.pi,
                        "after_rotation_concentration_R": float(R_tem_after[b, t].item()),
                    })

    def frame_summary_rows(self):
        rows = []
        for (method, gap), vals in sorted(self.frame_values.items()):
            row = {"method": method, "gap_ms": gap}
            for name, arr in vals.items():
                q = quantiles(arr)
                for stat, v in q.items():
                    row[f"{name}_{stat}"] = v
            R = np.asarray(vals["R"], dtype=float)
            R = R[np.isfinite(R)]
            if R.size:
                row["fraction_R_ge_0_5"] = float(np.mean(R >= 0.5))
                row["fraction_R_ge_0_7"] = float(np.mean(R >= 0.7))
                row["fraction_R_ge_0_9"] = float(np.mean(R >= 0.9))
                row["frame_count"] = int(R.size)
            rows.append(row)
        return rows

    def temporal_summary_rows(self):
        rows = []
        for (method, gap), vals in sorted(self.temporal_values.items()):
            row = {"method": method, "gap_ms": gap}
            for name, arr in vals.items():
                if name.startswith("kind_"):
                    row[name + "_count"] = int(np.sum(arr))
                    continue
                q = quantiles(arr)
                for stat, v in q.items():
                    row[f"{name}_{stat}"] = v
            rows.append(row)
        return rows


# --------------------------------------------------------------------------------------
# Core method/gap pass
# --------------------------------------------------------------------------------------
@torch.inference_mode()
def run_method_gap(args, method, gap, model, device, rot_collector):
    loader = make_loader(args, gap)
    mel_mean, mel_std = dataset_stats(loader)

    acc = {name: MetricAcc() for name in ("A2", "B", "G", "C")}
    err_before = {k: ErrorStat() for k in ("unit", "temporal", "frequency")}
    err_after = {k: ErrorStat() for k in ("unit", "temporal", "frequency")}

    sig = []
    alignment_sum = 0.0
    alignment_n = 0
    frequency_invariance_max = 0.0

    for batch in loader:
        visual, spk, masked, spec, phase, length, text, mask, path = unpack(batch, device)
        sig += signature(path, mask)
        bs = spec.size(0)

        c = forward_av(model, visual, spk, masked, length, mask, phase=phase)
        completed_mel = c["completed_mel"].float()
        prediction_mask = c["prediction_mask"].float()

        expected_prediction_mask = 1.0 - mask.float().mean(dim=1)
        if not torch.allclose(prediction_mask, expected_prediction_mask, atol=1e-6, rtol=0.0):
            maxerr = (prediction_mask - expected_prediction_mask).abs().max().item()
            raise RuntimeError(
                f"Phase-model prediction_mask disagrees with PLC mask (max abs diff={maxerr:.3e})."
            )

        gt_mag, align_mae = true_stft_batch(path, mask, phase, device)
        alignment_sum += align_mae * bs
        alignment_n += bs

        pred_cos = c["pred_cos"].float()
        pred_sin = c["pred_sin"].float()
        final_cos = c["final_cos"].float()
        final_sin = c["final_sin"].float()

        rot = estimate_frame_rotation(
            pred_cos, pred_sin, phase, gt_mag, prediction_mask
        )
        corrected_final_cos, corrected_final_sin = build_corrected_final(
            c, rot["corrected_pred_cos"], rot["corrected_pred_sin"], prediction_mask
        )
        gt_gap_cos, gt_gap_sin = build_gt_gap_final(c, phase, prediction_mask)

        # ----------------------------- phase-error decomposition -----------------------------
        before = compute_error_tensors(
            final_cos, final_sin, phase, prediction_mask, gt_mag, final=True
        )
        # Unit loss in the actual code uses RAW pred, not final. Replace only that entry.
        target_cos = torch.cos(phase).float()
        target_sin = torch.sin(phase).float()
        ud, ux = error_phasor(pred_cos, pred_sin, target_cos, target_sin)
        before["unit"] = (
            1.0 - ud,
            circular_abs_from_phasor(ud, ux),
            prediction_mask.bool().unsqueeze(1).expand_as(ud),
            gt_mag,
        )

        after = compute_error_tensors(
            corrected_final_cos, corrected_final_sin, phase, prediction_mask, gt_mag, final=True
        )
        cud, cux = error_phasor(
            rot["corrected_pred_cos"], rot["corrected_pred_sin"], target_cos, target_sin
        )
        after["unit"] = (
            1.0 - cud,
            circular_abs_from_phasor(cud, cux),
            prediction_mask.bool().unsqueeze(1).expand_as(cud),
            gt_mag,
        )

        for etype in ("unit", "temporal", "frequency"):
            bc, ba, bm, bw = before[etype]
            ac, aa, am, aw = after[etype]
            if not torch.equal(bm, am):
                raise RuntimeError(f"Before/after {etype} masks differ.")
            err_before[etype].add(bc, ba, bw, bm)
            err_after[etype].add(ac, aa, aw, am)

        # A common frame rotation MUST cancel in frequency differences.
        bfc = before["frequency"][0]
        afc = after["frequency"][0]
        fmask = before["frequency"][2]
        if fmask.any():
            frequency_invariance_max = max(
                frequency_invariance_max,
                float((bfc[fmask] - afc[fmask]).abs().max().item()),
            )

        rot_collector.add_frames(method, gap, path, prediction_mask, gt_mag, rot)
        rot_collector.add_temporal_pairs(
            method, gap, path, prediction_mask, gt_mag, phase,
            final_cos, final_sin, corrected_final_cos, corrected_final_sin,
            rot["alpha"],
        )

        # ----------------------------- controlled waveform conditions -----------------------------
        # A2: phase-model completed Mel + Griffin-Lim (no reconstructed audio supplied).
        acc["A2"].add(
            metric_batch(spec, completed_mel, text, mask, path, mel_mean, mel_std, recon_audio=None),
            bs,
        )

        # B: exact current learned phase.
        b_audio = torch_melphase2audio(
            completed_mel, final_cos, final_sin,
            mel_mean=mel_mean, mel_std=mel_std,
        )
        acc["B"].add(
            metric_batch(spec, completed_mel, text, mask, path, mel_mean, mel_std, recon_audio=b_audio),
            bs,
        )

        # G: same Mel + learned phase after removing only oracle common frame rotation.
        g_audio = torch_melphase2audio(
            completed_mel, corrected_final_cos, corrected_final_sin,
            mel_mean=mel_mean, mel_std=mel_std,
        )
        acc["G"].add(
            metric_batch(spec, completed_mel, text, mask, path, mel_mean, mel_std, recon_audio=g_audio),
            bs,
        )

        # C: same Mel + GT phase only in the missing gap.
        c_audio = torch_melphase2audio(
            completed_mel, gt_gap_cos, gt_gap_sin,
            mel_mean=mel_mean, mel_std=mel_std,
        )
        acc["C"].add(
            metric_batch(spec, completed_mel, text, mask, path, mel_mean, mel_std, recon_audio=c_audio),
            bs,
        )

    alignment = alignment_sum / max(alignment_n, 1)
    if alignment > 1e-3:
        raise RuntimeError(
            f"Stored phase is not aligned with fresh GT STFT: weighted MAE={alignment:.3e} rad"
        )
    # Numerical rotation should preserve adjacent-frequency differences essentially exactly.
    if frequency_invariance_max > 2e-5:
        raise RuntimeError(
            "Frequency-difference invariance failed after common rotation: "
            f"max circular-error change={frequency_invariance_max:.3e}"
        )

    metrics = {k: v.result() for k, v in acc.items()}
    phase_rows = []
    for etype in ("unit", "temporal", "frequency"):
        bv = err_before[etype].values()
        av = err_after[etype].values()
        row = {
            "method": method,
            "gap_ms": int(gap),
            "error_type": etype,
            "stored_vs_fresh_phase_mag_weighted_mae_rad": alignment,
            "frequency_invariance_max_abs_circular_change": frequency_invariance_max,
        }
        for k, v in bv.items():
            row[f"before_{k}"] = v
        for k, v in av.items():
            row[f"after_{k}"] = v
        b = bv.get("magnitude_weighted_circular_error", float("nan"))
        a = av.get("magnitude_weighted_circular_error", float("nan"))
        row["mag_weighted_circular_reduction_fraction"] = (
            (b - a) / max(abs(b), EPS) if np.isfinite(b) and np.isfinite(a) else float("nan")
        )
        ba = bv.get("magnitude_weighted_angular_mae_deg", float("nan"))
        aa = av.get("magnitude_weighted_angular_mae_deg", float("nan"))
        row["mag_weighted_angular_reduction_fraction"] = (
            (ba - aa) / max(abs(ba), EPS) if np.isfinite(ba) and np.isfinite(aa) else float("nan")
        )
        phase_rows.append(row)

    return metrics, phase_rows, sig


# --------------------------------------------------------------------------------------
# Headroom / prior-followup support / final decision
# --------------------------------------------------------------------------------------
METRIC_DIRECTION = {
    "pesq": 1.0,
    "plcmos": 1.0,
    "stoi": 1.0,
    "estoi": 1.0,
    "wer": -1.0,
    "cer": -1.0,
}


def controlled_metric_rows(method, gap, metrics):
    rows = [
        metric_row(method, gap, "A2_PHASE_MODEL_MEL_GRIFFIN_LIM", metrics["A2"]),
        metric_row(method, gap, "B_PHASE_MODEL_MEL_LEARNED_PHASE", metrics["B"]),
        metric_row(method, gap, "G_SAME_MEL_ROTATION_CORRECTED_LEARNED_PHASE", metrics["G"]),
        metric_row(method, gap, "C_SAME_MEL_GT_PHASE", metrics["C"]),
    ]
    rows.extend([
        metric_row(method, gap, "G_minus_B", delta(metrics["G"], metrics["B"]),
                   "benefit from removing only oracle frame-common rotation"),
        metric_row(method, gap, "C_minus_B", delta(metrics["C"], metrics["B"]),
                   "full GT-phase headroom with identical Mel"),
        metric_row(method, gap, "C_minus_G", delta(metrics["C"], metrics["G"]),
                   "residual phase headroom after common rotation is removed"),
        metric_row(method, gap, "G_minus_A2", delta(metrics["G"], metrics["A2"]),
                   "rotation-corrected learned phase vs Griffin-Lim with identical Mel"),
    ])
    return rows


def headroom_rows(method, gap, metrics):
    rows = []
    for m, direction in METRIC_DIRECTION.items():
        if not all(m in metrics[x] for x in ("A2", "B", "G", "C")):
            continue
        A2 = float(metrics["A2"][m])
        B = float(metrics["B"][m])
        G = float(metrics["G"][m])
        C = float(metrics["C"][m])
        full = direction * (C - B)
        recovered = direction * (G - B)
        residual = direction * (C - G)
        vs_gl = direction * (G - A2)
        ratio = recovered / full if full > 1e-8 else float("nan")
        rows.append({
            "method": method,
            "gap_ms": int(gap),
            "metric": m,
            "direction": "higher_better" if direction > 0 else "lower_better",
            "A2": A2,
            "B": B,
            "G": G,
            "C": C,
            "full_B_to_C_headroom": full,
            "G_recovered_from_B": recovered,
            "C_minus_G_residual": residual,
            "G_vs_GriffinLim": vs_gl,
            "fraction_B_to_C_headroom_recovered": ratio,
        })
    return rows


def load_followup_deltas(path: Path):
    if not path.is_file():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("condition") not in {"F_minus_previous_C", "D_minus_F", "F_minus_A0"}:
                continue
            parsed = dict(r)
            for k, v in list(parsed.items()):
                if k in {"method", "condition", "comparison"}:
                    continue
                try:
                    parsed[k] = float(v)
                except (TypeError, ValueError):
                    pass
            rows.append(parsed)
    return rows


def mean_metric(rows, condition, metric):
    vals = []
    for r in rows:
        if r.get("condition") == condition:
            try:
                v = float(r.get(metric, "nan"))
            except (TypeError, ValueError):
                continue
            if np.isfinite(v):
                vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def build_decision_report(frame_rows, phase_rows, headroom, temporal_rows, prior_followup_rows):
    def vals(rows, key, where=None):
        out = []
        for r in rows:
            if where and not where(r):
                continue
            try:
                v = float(r[key])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(v):
                out.append(v)
        return np.asarray(out, dtype=float)

    median_R = np.nanmedian(vals(frame_rows, "R_median")) if frame_rows else float("nan")
    p25_R = np.nanmedian(vals(frame_rows, "R_p25")) if frame_rows else float("nan")
    median_R_high = np.nanmedian(vals(frame_rows, "R_high_median")) if frame_rows else float("nan")

    unit_reduction = np.nanmean(vals(
        phase_rows, "mag_weighted_circular_reduction_fraction",
        lambda r: r.get("error_type") == "unit",
    ))
    temporal_reduction = np.nanmean(vals(
        phase_rows, "mag_weighted_circular_reduction_fraction",
        lambda r: r.get("error_type") == "temporal",
    ))
    freq_change = np.nanmax(vals(
        phase_rows, "frequency_invariance_max_abs_circular_change",
        lambda r: r.get("error_type") == "frequency",
    ))

    pesq_rec = vals(headroom, "fraction_B_to_C_headroom_recovered", lambda r: r.get("metric") == "pesq")
    plcmos_rec = vals(headroom, "fraction_B_to_C_headroom_recovered", lambda r: r.get("metric") == "plcmos")
    pesq_gl = vals(headroom, "G_vs_GriffinLim", lambda r: r.get("metric") == "pesq")
    plcmos_gl = vals(headroom, "G_vs_GriffinLim", lambda r: r.get("metric") == "plcmos")

    median_pesq_rec = float(np.nanmedian(pesq_rec)) if pesq_rec.size else float("nan")
    median_plcmos_rec = float(np.nanmedian(plcmos_rec)) if plcmos_rec.size else float("nan")
    fraction_g_beats_gl_pesq = float(np.mean(pesq_gl > 0)) if pesq_gl.size else float("nan")
    fraction_g_beats_gl_plcmos = float(np.mean(plcmos_gl > 0)) if plcmos_gl.size else float("nan")

    beta_match = np.nanmedian(vals(temporal_rows, "beta_minus_delta_alpha_abs_deg_median")) if temporal_rows else float("nan")
    temporal_R = np.nanmedian(vals(temporal_rows, "R_temporal_median")) if temporal_rows else float("nan")

    # These are diagnostic decision rules, not literature-derived universal thresholds.
    rotation_geometrically_strong = (
        np.isfinite(median_R) and np.isfinite(unit_reduction)
        and median_R >= 0.70 and unit_reduction >= 0.50
    )
    rotation_temporally_explanatory = (
        np.isfinite(temporal_reduction) and temporal_reduction >= 0.50
    )
    rotation_perceptually_dominant = (
        np.isfinite(median_pesq_rec) and np.isfinite(median_plcmos_rec)
        and median_pesq_rec >= 0.50 and median_plcmos_rec >= 0.50
    )
    corrected_beats_gl = (
        np.isfinite(fraction_g_beats_gl_pesq) and np.isfinite(fraction_g_beats_gl_plcmos)
        and fraction_g_beats_gl_pesq >= 0.75 and fraction_g_beats_gl_plcmos >= 0.75
    )

    if rotation_geometrically_strong and rotation_temporally_explanatory and rotation_perceptually_dominant and corrected_beats_gl:
        verdict = "COMMON_ROTATION_IS_A_DOMINANT_ACTIONABLE_PHASE_FAILURE"
        phase_impl = (
            "Implement the protected Mel branch, then a separate phase-specialized decoder whose "
            "main new capability is an explicitly boundary-anchored temporal phase-rotation/phase-advance "
            "trajectory. Preserve the already-useful frequency-relative phase modeling."
        )
    elif rotation_geometrically_strong and rotation_perceptually_dominant:
        verdict = "COMMON_ROTATION_IS_IMPORTANT_BUT_NOT_THE_ONLY_PHASE_FAILURE"
        phase_impl = (
            "Implement the protected Mel branch plus a separate phase decoder with TWO explicit roles: "
            "(1) temporal/common-rotation trajectory anchored by observed phase, and (2) residual "
            "frequency-dependent phase refinement. Do not rely on rotation correction alone."
        )
    elif rotation_geometrically_strong and not rotation_perceptually_dominant:
        verdict = "COMMON_ROTATION_EXISTS_GEOMETRICALLY_BUT_IS_NOT_THE_MAIN_PERCEPTUAL_BOTTLENECK"
        phase_impl = (
            "Protect Mel, but redesign the phase decoder more broadly. A rotation/phase-advance component "
            "can be included, yet residual TF phase/consistency must be modeled explicitly because oracle "
            "rotation does not recover enough B->C quality."
        )
    else:
        verdict = "COMMON_ROTATION_HYPOTHESIS_NOT_SUPPORTED_AS_DOMINANT"
        phase_impl = (
            "Protect Mel and build a separate phase-specialized decoder, but do not organize it primarily "
            "around one scalar alpha_t. Directly model temporal phase increments plus frequency-relative "
            "structure (or an explicit complex/phase residual representation) with observed boundary phase."
        )

    prior_lines = []
    if prior_followup_rows:
        f_c_pesq = mean_metric(prior_followup_rows, "F_minus_previous_C", "pesq")
        f_c_plc = mean_metric(prior_followup_rows, "F_minus_previous_C", "plcmos")
        d_f_pesq = mean_metric(prior_followup_rows, "D_minus_F", "pesq")
        d_f_plc = mean_metric(prior_followup_rows, "D_minus_F", "plcmos")
        prior_lines = [
            f"Prior F-C mean PESQ: {f_c_pesq:+.4f}",
            f"Prior F-C mean PLCMOS: {f_c_plc:+.4f}",
            f"Prior D-F mean PESQ remaining Mel headroom: {d_f_pesq:+.4f}",
            f"Prior D-F mean PLCMOS remaining Mel headroom: {d_f_plc:+.4f}",
            "Magnitude-path implication: preserve/protect the no-phase Mel decoder; long-gap Mel quality still has substantial headroom.",
        ]
    else:
        prior_lines = [
            "Prior follow-up CSV not found; the rotation decision remains valid, but the automatic report cannot restate F-C/D-F numerically.",
        ]

    lines = [
        "AV_PLC FINAL DIAGNOSTIC DECISION REPORT",
        "========================================",
        "",
        "This report uses diagnostic thresholds only to make the implementation path explicit; they are not universal statistical significance thresholds.",
        "",
        *prior_lines,
        "",
        "Common-rotation evidence:",
        f"  median frame R concentration (median across method/gap summaries): {median_R:.4f}",
        f"  median frame R p25: {p25_R:.4f}",
        f"  median high-energy R: {median_R_high:.4f}",
        f"  mean magnitude-weighted unit circular-error reduction after oracle rotation: {unit_reduction:.4f}",
        f"  mean magnitude-weighted temporal circular-error reduction after oracle rotation: {temporal_reduction:.4f}",
        f"  max frequency-difference circular-error change (must be ~0): {freq_change:.3e}",
        f"  median temporal-error concentration R: {temporal_R:.4f}",
        f"  median |temporal common-error angle - delta alpha|: {beta_match:.2f} deg",
        "",
        "Perceptual headroom:",
        f"  median fraction of B->C PESQ headroom recovered by G: {median_pesq_rec:.4f}",
        f"  median fraction of B->C PLCMOS headroom recovered by G: {median_plcmos_rec:.4f}",
        f"  fraction of method/gap cells where G beats Griffin-Lim in PESQ: {fraction_g_beats_gl_pesq:.3f}",
        f"  fraction of method/gap cells where G beats Griffin-Lim in PLCMOS: {fraction_g_beats_gl_plcmos:.3f}",
        "",
        f"VERDICT: {verdict}",
        "",
        "NEXT MAGNITUDE+PHASE IMPLEMENTATION:",
        "  Magnitude/Mel: keep the successful no-phase Mel path separate/protected from phase gradients.",
        f"  Phase: {phase_impl}",
        "  Training order: initialize/freeze the successful Mel branch first; train the phase branch; only consider later low-LR Mel unfreezing if it does not reduce Mel/PESQ/PLCMOS.",
        "  Initial phase objective: circular absolute/unit + direct temporal-increment supervision + frequency-relative supervision; do not reintroduce projected complex loss as the first step because the previous complex_v1 result did not improve perceptual quality.",
        "",
        "STOP RULE:",
        "  This diagnostic is designed to choose the phase-decoder structure. Once the verdict is obtained, move to implementation rather than adding another oracle unless a sanity invariant fails.",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------
# Synthetic math self-test: catches sign mistakes before expensive inference.
# --------------------------------------------------------------------------------------
def run_math_self_test(device):
    torch.manual_seed(7)
    B, Fbins, T = 2, 17, 8
    gt = (torch.rand(B, Fbins, T, device=device) * 2.0 - 1.0) * math.pi
    gt_c, gt_s = torch.cos(gt), torch.sin(gt)
    # Variable frame-common rotations, including observed endpoints alpha=0.
    alpha = torch.tensor(
        [[0.0, 0.7, -1.1, 1.6, -0.4, 0.9, -2.0, 0.0],
         [0.0, -0.8, 1.2, -1.7, 0.5, -1.0, 2.1, 0.0]],
        device=device,
    )
    pred_c = torch.cos(gt + alpha.unsqueeze(1))
    pred_s = torch.sin(gt + alpha.unsqueeze(1))
    q = torch.ones(B, T, device=device)
    q[:, 0] = q[:, -1] = 0.0
    mag = torch.rand(B, Fbins, T, device=device) + 0.1

    rot = estimate_frame_rotation(pred_c, pred_s, gt, mag, q)
    missing = q.bool()
    err_alpha = torch.atan2(
        torch.sin(rot["alpha"] - alpha), torch.cos(rot["alpha"] - alpha)
    ).abs()
    if float(err_alpha[missing].max().item()) > 2e-5:
        raise RuntimeError("Synthetic self-test failed to recover known frame rotation.")
    if float((1.0 - rot["R"][missing]).abs().max().item()) > 2e-5:
        raise RuntimeError("Synthetic self-test concentration R should be 1 for exact common rotation.")

    # Build FINAL phase with GT at observed endpoints, then correct.
    final_c = gt_c.clone()
    final_s = gt_s.clone()
    final_c[:, :, 1:-1] = pred_c[:, :, 1:-1]
    final_s[:, :, 1:-1] = pred_s[:, :, 1:-1]
    fake_completion = {"final_cos": final_c, "final_sin": final_s}
    cc, cs = build_corrected_final(
        fake_completion, rot["corrected_pred_cos"], rot["corrected_pred_sin"], q
    )
    phase_err = torch.atan2(
        cs * gt_c - cc * gt_s,
        cc * gt_c + cs * gt_s,
    ).abs()
    if float(phase_err.max().item()) > 3e-5:
        raise RuntimeError("Synthetic self-test rotation correction did not recover GT phase.")

    # Frequency differences must be invariant to common frame rotation.
    before = compute_error_tensors(final_c, final_s, gt, q, mag)
    after = compute_error_tensors(cc, cs, gt, q, mag)
    fm = before["frequency"][2]
    fdiff = float((before["frequency"][0][fm] - after["frequency"][0][fm]).abs().max().item())
    if fdiff > 3e-5:
        raise RuntimeError("Synthetic self-test violated frequency-difference invariance.")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Final AV_PLC common-phase-rotation diagnostic before magnitude+phase redesign."
    )
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    ap.add_argument("--phase-variant", choices=("complex_v1", "plain"), default="complex_v1")
    ap.add_argument("--gaps", nargs="+", type=int, default=list(GAPS))
    ap.add_argument("--test-subset", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--checkpoint-root", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument(
        "--followup-deltas",
        default=None,
        help="Optional F_oracle_followup_deltas.csv. If omitted, the default follow-up output path is tried.",
    )
    ap.add_argument(
        "--save-frame-details",
        action="store_true",
        help="Also save per-missing-frame and per-temporal-pair summaries. Not needed for the main decision.",
    )
    args = ap.parse_args()

    set_global_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_math_self_test(device)

    ckpt_root = args.checkpoint_root or project_checkpoint_dir("AV_PLC")
    log_root = Path(project_log_dir("AV_PLC"))
    outdir = Path(args.output_dir) if args.output_dir else (
        log_root / f"common_rotation_final_{ARCH}_{args.phase_variant}" / DATASET
    )
    outdir.mkdir(parents=True, exist_ok=True)

    followup_path = (
        Path(args.followup_deltas)
        if args.followup_deltas
        else log_root
        / f"oracle_phase_followup_{ARCH}_{args.phase_variant}"
        / DATASET
        / "F_oracle_followup_deltas.csv"
    )
    prior_followup = load_followup_deltas(followup_path)

    print(f"device={device}")
    print(f"checkpoints={ckpt_root}")
    print(f"output={outdir}")
    print(f"followup_deltas={followup_path} ({'found' if prior_followup else 'not found'})")
    print("Synthetic common-rotation math self-test: PASS")

    metric_rows = []
    phase_rows = []
    headroom = []
    rot_collector = RotationCollector(save_details=args.save_frame_details)
    signatures = {}

    for method in args.methods:
        checkpoint = os.path.join(
            ckpt_root, model_name(method, args.phase_variant), "best_model.pt"
        )
        print(f"\n[{method}] checkpoint: {checkpoint}")
        model = build_model(method, args.phase_variant, checkpoint, device)

        for gap in args.gaps:
            print(f"  gap={gap} ms")
            metrics, p_rows, sig = run_method_gap(
                args, method, gap, model, device, rot_collector
            )
            signatures[(method, gap)] = sig
            metric_rows.extend(controlled_metric_rows(method, gap, metrics))
            phase_rows.extend(p_rows)
            headroom.extend(headroom_rows(method, gap, metrics))

            # Compact console result for immediate sanity.
            def g(m, cond):
                return metrics[cond].get(m, float("nan"))
            print(
                "    PLCMOS A2/B/G/C="
                f"{g('plcmos','A2'):.3f}/{g('plcmos','B'):.3f}/{g('plcmos','G'):.3f}/{g('plcmos','C'):.3f} | "
                "PESQ="
                f"{g('pesq','A2'):.3f}/{g('pesq','B'):.3f}/{g('pesq','G'):.3f}/{g('pesq','C'):.3f}"
            )

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    frame_summary = rot_collector.frame_summary_rows()
    temporal_summary = rot_collector.temporal_summary_rows()

    # Save all decision-facing outputs.
    save_csv(
        metric_rows,
        outdir / "common_rotation_metric_summary.csv",
        fixed=["method", "gap_ms", "condition", "comparison"],
    )
    save_csv(
        phase_rows,
        outdir / "common_rotation_phase_summary.csv",
        fixed=["method", "gap_ms", "error_type"],
    )
    save_csv(
        headroom,
        outdir / "common_rotation_headroom.csv",
        fixed=["method", "gap_ms", "metric"],
    )
    save_csv(
        frame_summary,
        outdir / "common_rotation_frame_summary.csv",
        fixed=["method", "gap_ms"],
    )
    save_csv(
        temporal_summary,
        outdir / "common_rotation_temporal_summary.csv",
        fixed=["method", "gap_ms"],
    )

    if args.save_frame_details:
        save_csv(
            rot_collector.frame_rows,
            outdir / "common_rotation_frames.csv",
            fixed=["method", "gap_ms", "path", "frame_index"],
        )
        save_csv(
            rot_collector.temporal_rows,
            outdir / "common_rotation_temporal_pairs.csv",
            fixed=["method", "gap_ms", "path", "left_frame", "right_frame", "transition_type"],
        )

    report = build_decision_report(
        frame_summary, phase_rows, headroom, temporal_summary, prior_followup
    )
    (outdir / "implementation_decision.txt").write_text(report, encoding="utf-8")

    print("\nSaved:")
    for fn in (
        "common_rotation_metric_summary.csv",
        "common_rotation_phase_summary.csv",
        "common_rotation_headroom.csv",
        "common_rotation_frame_summary.csv",
        "common_rotation_temporal_summary.csv",
        "implementation_decision.txt",
    ):
        print(outdir / fn)
    if args.save_frame_details:
        print(outdir / "common_rotation_frames.csv")
        print(outdir / "common_rotation_temporal_pairs.csv")

    print("\n================ IMPLEMENTATION DECISION ================")
    print(report)


if __name__ == "__main__":
    main()
