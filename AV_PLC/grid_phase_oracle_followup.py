#!/usr/bin/env python3
"""GRID follow-up oracle diagnostic for AV_PLC learned phase.

Purpose
-------
This is a no-training diagnostic that follows the previous A/A2/B/C/D/E oracle.
It answers two remaining questions:

1) F oracle: if we keep the *no-phase model's* Mel prediction but replace
   Griffin-Lim phase with ground-truth STFT phase, how good can that model be?

2) Where does the learned phase fail?
   Measure the current phase model's circular errors as a function of:
     - GT STFT magnitude / energy,
     - normalized position inside the PLC gap,
     - normalized depth from the nearest observed gap boundary.

Controlled conditions computed here
-----------------------------------
A0 : no-phase model COMPLETED Mel + Griffin-Lim
F  : exact same no-phase COMPLETED Mel + GT phase

A0 is intentionally added because F-A0 isolates phase while holding Mel fixed.
If the previous oracle CSV is available, this script also reports comparisons to:
A  : previous no-phase raw predicted Mel + Griffin-Lim
C  : previous phase-model completed Mel + GT phase
D  : previous GT Mel + GT phase

Phase-error diagnostics mirror the current training losses:
  unit     : raw pred_cos/pred_sin, missing frames only
  temporal : final_cos/final_sin, adjacent-time pairs where >=1 endpoint missing
  frequency: final_cos/final_sin, adjacent-frequency pairs at missing frames

No model is trained and no checkpoint is modified.

Default command (matches the previous GRID oracle setup):
    python AV_PLC/grid_phase_oracle_followup.py

Outputs are CSV files under the AV_PLC log directory.
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
        "AV_PLC/grid_phase_oracle_followup.py"
    )
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluations.runtime_config import SEED, project_checkpoint_dir, project_log_dir, set_global_seed
from AV_PLC.av_dataloader import AVDataloader
from AV_PLC.multimodal_decoder import AV_PLC
from shared.audio_processing import hop_len, n_fft, read_gt_input, sample_rate, torch_melphase2audio, win_len
from shared.metrics import calculate_batch_metrics  # noqa: E402

# --------------------------------------------------------------------------------------
# Experiment constants: intentionally identical to the previous oracle script.
# --------------------------------------------------------------------------------------
DATASET = "grid"
ARCH = "latent_spectral_v2"
METHODS = ("concat", "temporal_self_cross_attention", "global_local_affinity")
GAPS = (160, 500, 1000)

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

# Relative-to-utterance-peak STFT magnitude bins (dB).
# 20*log10 is used because this is magnitude, not power.
ENERGY_EDGES_DB = (-math.inf, -60.0, -40.0, -20.0, -10.0, math.inf)
ENERGY_LABELS = ("<-60", "-60_to_-40", "-40_to_-20", "-20_to_-10", "-10_to_0")
POSITION_BINS = 10
DEPTH_BINS = 5
EPS = 1e-8


def model_name(method: str, phase_variant: str | None = None) -> str:
    base = (
        f"av_plc_{method}_{ARCH}_frozen_enc_masked_mel_ge_train"
        f"_no_jitter({DATASET})"
    )
    if phase_variant is None:
        return base
    if phase_variant == "plain":
        return base + "_phase_reconstruction"
    if phase_variant == "complex_v1":
        return base + "_phase_reconstruction_complex_v1"
    raise ValueError(f"Unsupported phase variant: {phase_variant}")


def build_model(method: str, phase_reconstruction: bool, checkpoint: str, device: torch.device):
    kwargs = dict(
        **MODEL_ARGS,
        fusion_type=method,
        phase_reconstruction=phase_reconstruction,
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
    # phase_reconstruction=True is required only so the dataloader returns stored
    # GT phase. It does NOT turn phase on inside the no-phase model.
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
    """Stable sample+mask signature used to prove passes are paired."""
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
    for key in ("predicted_mel", "completed_mel", "prediction_mask"):
        if completion.get(key) is None:
            raise RuntimeError(f"completion_output['{key}'] is missing.")
    return completion


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
    """Use the exact shared metric implementation used by the current evaluator."""
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

    # GT/completed controls may make PSNR undefined in the project's metric
    # helper when MSE==0. Suppress only that numpy reduction warning.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Mean of empty slice", category=RuntimeWarning
        )
        return calculate_batch_metrics(**kwargs)


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


def row(method, gap, condition, values, comparison="", **extra):
    r = dict(method=method, gap_ms=int(gap), condition=condition, comparison=comparison)
    r.update(values)
    r.update(extra)
    return r


def save_csv(rows, path: Path, fixed=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fixed = list(fixed or [])
    all_keys = {k for r in rows for k in r}
    rest = sorted(all_keys - set(fixed))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fixed + rest)
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------------------
# Previous oracle CSV support
# --------------------------------------------------------------------------------------
def _parse_num(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return x


def load_previous_oracle(path: Path):
    if not path.is_file():
        return {}
    data = {}
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            method = r.get("method", "")
            condition = r.get("condition", "")
            try:
                gap = int(float(r.get("gap_ms", "")))
            except ValueError:
                continue
            values = {
                k: _parse_num(v)
                for k, v in r.items()
                if k not in {"method", "gap_ms", "condition", "comparison"}
            }
            data[(method, gap, condition)] = values
    return data


# --------------------------------------------------------------------------------------
# F experiment
# --------------------------------------------------------------------------------------
@torch.inference_mode()
def run_F_and_A0(args, gap: int, device: torch.device, model):
    """Compute A0 and F from the same no-phase model and identical batches.

    A0 = no-phase completed Mel + Griffin-Lim
    F  = same no-phase completed Mel + GT phase

    Both conditions therefore have exactly the same Mel and differ only in phase.
    """
    loader = make_loader(args, gap)
    mel_mean, mel_std = dataset_stats(loader)
    A0, F_oracle = MetricAcc(), MetricAcc()
    sig = []

    for batch in loader:
        visual, spk, masked, spec, phase, length, text, mask, path = unpack(batch, device)
        sig += signature(path, mask)
        bs = spec.size(0)

        c = forward_av(
            model, visual, spk, masked, length, mask, phase=None
        )
        completed_mel = c["completed_mel"].float()

        # Verify the model's prediction mask is exactly the PLC gap implied by
        # the dataloader mask for AV/audio-present samples.
        expected_prediction_mask = 1.0 - mask.float().mean(dim=1)
        observed_prediction_mask = c["prediction_mask"].float()
        if not torch.allclose(
            observed_prediction_mask,
            expected_prediction_mask,
            atol=1e-6,
            rtol=0.0,
        ):
            maxerr = (observed_prediction_mask - expected_prediction_mask).abs().max().item()
            raise RuntimeError(
                f"No-phase prediction_mask disagrees with PLC mask (max abs diff={maxerr:.3e})."
            )

        # A0: same completed Mel as F, but Griffin-Lim.  No reconstructed_audio
        # is supplied, so calculate_batch_metrics follows the current GL path.
        A0.add(
            metric_batch(
                spec, completed_mel, text, mask, path, mel_mean, mel_std,
                recon_audio=None,
            ),
            bs,
        )

        # F: exactly the same completed Mel, but with stored GT phase.
        f_audio = torch_melphase2audio(
            completed_mel,
            torch.cos(phase),
            torch.sin(phase),
            mel_mean=mel_mean,
            mel_std=mel_std,
        )
        F_oracle.add(
            metric_batch(
                spec, completed_mel, text, mask, path, mel_mean, mel_std,
                recon_audio=f_audio,
            ),
            bs,
        )

    return A0.result(), F_oracle.result(), sig


# --------------------------------------------------------------------------------------
# Exact STFT magnitude for phase-error analysis
# --------------------------------------------------------------------------------------
def true_stft_batch(path, mask, stored_phase, device):
    """Return exact one-sided GT STFT magnitude aligned with stored phase.

    This reproduces the same forward STFT geometry validated by condition E:
      n_fft=512, win=400, hop=160, center=False, pad=(n_fft-hop)/2=176.
    """
    pad = (n_fft - hop_len) // 2
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

    # Independent alignment guard. At bins with effectively zero magnitude the
    # phase is physically irrelevant, so use magnitude weighting for this check.
    fresh_phase = torch.angle(stft)
    d = stored_phase.to(device) - fresh_phase
    d = torch.atan2(torch.sin(d), torch.cos(d)).abs()
    mag = stft.abs()
    phase_mae_weighted = (d * mag).sum() / mag.sum().clamp_min(EPS)

    return mag, float(phase_mae_weighted.item())


# --------------------------------------------------------------------------------------
# Phase error definitions: exactly match the three current phase losses.
# --------------------------------------------------------------------------------------
def circular_error_from_unit(pred_cos, pred_sin, target_cos, target_sin):
    # cos(pred-target)
    dot = pred_cos * target_cos + pred_sin * target_sin
    # sin(pred-target)
    cross = pred_sin * target_cos - pred_cos * target_sin
    dot = dot.clamp(-1.0, 1.0)
    circular = 1.0 - dot
    angle_abs = torch.atan2(cross, dot).abs()
    return circular, angle_abs


def phase_difference(c0, s0, c1, s1):
    # Unit vector of angle1-angle0; same identities used in AV_PLC/losses.py.
    return (
        c1 * c0 + s1 * s0,
        s1 * c0 - c1 * s0,
    )


def compute_phase_error_tensors(completion, target_phase, prediction_mask, gt_mag):
    """Return unit/temporal/frequency error tensors and matching magnitude weights."""
    target_cos = torch.cos(target_phase)
    target_sin = torch.sin(target_phase)

    pred_cos = completion.get("pred_cos")
    pred_sin = completion.get("pred_sin")
    final_cos = completion.get("final_cos")
    final_sin = completion.get("final_sin")
    if any(x is None for x in (pred_cos, pred_sin, final_cos, final_sin)):
        raise RuntimeError("Phase model must return pred_cos/pred_sin/final_cos/final_sin.")

    pred_cos = pred_cos.float()
    pred_sin = pred_sin.float()
    final_cos = final_cos.float()
    final_sin = final_sin.float()
    target_cos = target_cos.float()
    target_sin = target_sin.float()

    # Unit loss: current code uses RAW predicted phase, then masks to missing frames.
    unit_circ, unit_angle = circular_error_from_unit(
        pred_cos, pred_sin, target_cos, target_sin
    )
    unit_mask = prediction_mask.bool().unsqueeze(1).expand_as(unit_circ)
    unit_mag = gt_mag

    # Temporal loss: current code uses FINAL phase. This is essential because
    # observed endpoints are exact GT anchors at both gap boundaries.
    pdc, pds = phase_difference(
        final_cos[..., :-1], final_sin[..., :-1],
        final_cos[..., 1:], final_sin[..., 1:],
    )
    tdc, tds = phase_difference(
        target_cos[..., :-1], target_sin[..., :-1],
        target_cos[..., 1:], target_sin[..., 1:],
    )
    temporal_circ, temporal_angle = circular_error_from_unit(pdc, pds, tdc, tds)
    temporal_pair_mask = torch.maximum(
        prediction_mask[..., :-1], prediction_mask[..., 1:]
    ).bool()
    temporal_mask = temporal_pair_mask.unsqueeze(1).expand_as(temporal_circ)
    temporal_mag = 0.5 * (gt_mag[..., :-1] + gt_mag[..., 1:])

    # Frequency loss: current code also uses FINAL phase, but only missing time
    # frames are retained. Each value corresponds to one adjacent-frequency pair.
    pfc, pfs = phase_difference(
        final_cos[:, :-1, :], final_sin[:, :-1, :],
        final_cos[:, 1:, :], final_sin[:, 1:, :],
    )
    tfc, tfs = phase_difference(
        target_cos[:, :-1, :], target_sin[:, :-1, :],
        target_cos[:, 1:, :], target_sin[:, 1:, :],
    )
    frequency_circ, frequency_angle = circular_error_from_unit(pfc, pfs, tfc, tfs)
    frequency_mask = prediction_mask.bool().unsqueeze(1).expand_as(frequency_circ)
    frequency_mag = 0.5 * (gt_mag[:, :-1, :] + gt_mag[:, 1:, :])

    return {
        "unit": (unit_circ, unit_angle, unit_mask, unit_mag),
        "temporal": (temporal_circ, temporal_angle, temporal_mask, temporal_mag),
        "frequency": (frequency_circ, frequency_angle, frequency_mask, frequency_mag),
    }


# --------------------------------------------------------------------------------------
# Streaming error aggregators
# --------------------------------------------------------------------------------------
class ErrorStat:
    def __init__(self):
        self.count = 0
        self.sum_circ = 0.0
        self.sum_angle = 0.0
        self.sum_mag = 0.0
        self.sum_mag_circ = 0.0
        self.sum_mag_angle = 0.0

    def add(self, circ, angle, mag, select=None):
        if select is not None:
            circ = circ[select]
            angle = angle[select]
            mag = mag[select]
        else:
            circ = circ.reshape(-1)
            angle = angle.reshape(-1)
            mag = mag.reshape(-1)

        if circ.numel() == 0:
            return
        circ = circ.double()
        angle = angle.double()
        mag = mag.double().clamp_min(0.0)

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
                "magnitude_sum": 0.0,
            }
        circ = self.sum_circ / self.count
        ang = self.sum_angle / self.count
        if self.sum_mag > 0.0:
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
            "magnitude_sum": self.sum_mag,
        }


class ErrorCollectors:
    def __init__(self):
        self.overall = defaultdict(ErrorStat)
        self.energy = defaultdict(ErrorStat)
        self.position = defaultdict(ErrorStat)
        self.depth = defaultdict(ErrorStat)

    @staticmethod
    def _energy_bin_masks(rel_db):
        for i, label in enumerate(ENERGY_LABELS):
            lo, hi = ENERGY_EDGES_DB[i], ENERGY_EDGES_DB[i + 1]
            if i == len(ENERGY_LABELS) - 1:
                sel = (rel_db >= lo) & (rel_db <= hi)
            else:
                sel = (rel_db >= lo) & (rel_db < hi)
            yield i, label, lo, hi, sel

    def add_error_type(
        self,
        method,
        gap,
        error_type,
        circ,
        angle,
        valid_mask,
        mag,
        rel_db,
        position_values,
        depth_values,
    ):
        # Overall.
        key = (method, int(gap), error_type)
        self.overall[key].add(circ, angle, mag, valid_mask)

        # Energy bins.
        for i, label, lo, hi, e_mask in self._energy_bin_masks(rel_db):
            sel = valid_mask & e_mask
            self.energy[(method, int(gap), error_type, i, label, lo, hi)].add(
                circ, angle, mag, sel
            )

        # Gap-position bins, 0=left boundary and 1=right boundary.
        for b in range(POSITION_BINS):
            lo = b / POSITION_BINS
            hi = (b + 1) / POSITION_BINS
            if b == POSITION_BINS - 1:
                psel = (position_values >= lo) & (position_values <= hi)
            else:
                psel = (position_values >= lo) & (position_values < hi)
            sel = valid_mask & psel
            self.position[(method, int(gap), error_type, b, lo, hi)].add(
                circ, angle, mag, sel
            )

        # Symmetric depth from nearest observed boundary:
        # depth=0 at either gap edge, depth=1 at gap center.
        for b in range(DEPTH_BINS):
            lo = b / DEPTH_BINS
            hi = (b + 1) / DEPTH_BINS
            if b == DEPTH_BINS - 1:
                dsel = (depth_values >= lo) & (depth_values <= hi)
            else:
                dsel = (depth_values >= lo) & (depth_values < hi)
            sel = valid_mask & dsel
            self.depth[(method, int(gap), error_type, b, lo, hi)].add(
                circ, angle, mag, sel
            )

    def rows(self, phase_alignment_by_method_gap):
        overall_rows, energy_rows, position_rows, depth_rows = [], [], [], []

        for (method, gap, etype), stat in sorted(self.overall.items()):
            v = stat.values()
            v.update(
                method=method,
                gap_ms=gap,
                error_type=etype,
                stored_vs_fresh_phase_mag_weighted_mae_rad=phase_alignment_by_method_gap[(method, gap)],
            )
            overall_rows.append(v)

        # Energy fraction is normalized within each method/gap/error_type.
        overall_mag = {
            (r["method"], r["gap_ms"], r["error_type"]): r["magnitude_sum"]
            for r in overall_rows
        }
        for (method, gap, etype, idx, label, lo, hi), stat in sorted(self.energy.items()):
            v = stat.values()
            denom = overall_mag.get((method, gap, etype), 0.0)
            v.update(
                method=method,
                gap_ms=gap,
                error_type=etype,
                energy_bin=label,
                energy_low_db=("-inf" if not np.isfinite(lo) else lo),
                energy_high_db=("inf" if not np.isfinite(hi) else hi),
                magnitude_fraction=(v["magnitude_sum"] / denom if denom > 0 else float("nan")),
            )
            energy_rows.append(v)

        for (method, gap, etype, idx, lo, hi), stat in sorted(self.position.items()):
            v = stat.values()
            v.update(
                method=method,
                gap_ms=gap,
                error_type=etype,
                position_bin=idx,
                position_low=lo,
                position_high=hi,
                position_definition="0=first missing frame/left boundary; 1=last missing frame/right boundary",
            )
            position_rows.append(v)

        for (method, gap, etype, idx, lo, hi), stat in sorted(self.depth.items()):
            v = stat.values()
            v.update(
                method=method,
                gap_ms=gap,
                error_type=etype,
                depth_bin=idx,
                depth_low=lo,
                depth_high=hi,
                depth_definition="0=nearest observed boundary; 1=center of gap",
            )
            depth_rows.append(v)

        return overall_rows, energy_rows, position_rows, depth_rows


def frame_position_and_depth(prediction_mask):
    """Create [B,T] normalized position/depth only on missing frames.

    The single-gap test is required to be one contiguous missing run per sample.
    """
    B, T = prediction_mask.shape
    pos = torch.full_like(prediction_mask, float("nan"), dtype=torch.float32)
    depth = torch.full_like(prediction_mask, float("nan"), dtype=torch.float32)

    for b in range(B):
        idx = torch.nonzero(prediction_mask[b] > 0.5, as_tuple=True)[0]
        if idx.numel() == 0:
            raise RuntimeError("Single-gap diagnostic found a sample with no missing frames.")
        if idx.numel() > 1 and not torch.all(idx[1:] == idx[:-1] + 1):
            raise RuntimeError("Expected one contiguous single gap, but missing frames are non-contiguous.")

        L = idx.numel()
        if L == 1:
            p = torch.tensor([0.5], device=prediction_mask.device)
        else:
            p = torch.linspace(0.0, 1.0, L, device=prediction_mask.device)
        d = 2.0 * torch.minimum(p, 1.0 - p)  # 0 edges, 1 center
        pos[b, idx] = p
        depth[b, idx] = d
    return pos, depth


def temporal_position_and_depth(prediction_mask, frame_pos, frame_depth):
    """Assign each supervised temporal transition to its missing-side position.

    Left boundary observed->missing gets position 0.
    Right boundary missing->observed gets position 1.
    Internal missing->missing pairs get the midpoint of the two frame positions.
    """
    B, T = prediction_mask.shape
    pair_mask = torch.maximum(prediction_mask[:, :-1], prediction_mask[:, 1:]) > 0.5
    pos = torch.full(
        (B, T - 1), float("nan"), device=prediction_mask.device, dtype=torch.float32
    )
    depth = torch.full_like(pos, float("nan"))

    for b in range(B):
        for t in torch.nonzero(pair_mask[b], as_tuple=True)[0].tolist():
            left_missing = bool(prediction_mask[b, t] > 0.5)
            right_missing = bool(prediction_mask[b, t + 1] > 0.5)
            if left_missing and right_missing:
                p = 0.5 * (frame_pos[b, t] + frame_pos[b, t + 1])
            elif left_missing:
                p = frame_pos[b, t]
            elif right_missing:
                p = frame_pos[b, t + 1]
            else:
                raise RuntimeError("pair_mask included an observed->observed transition.")
            pos[b, t] = p
            depth[b, t] = 2.0 * torch.minimum(p, 1.0 - p)
    return pos, depth


@torch.inference_mode()
def run_phase_error_diagnostic(
    args,
    gap: int,
    device: torch.device,
    method: str,
    model,
    reference_signature,
    collectors: ErrorCollectors,
):
    loader = make_loader(args, gap)
    sig = []
    alignment_sum = 0.0
    alignment_n = 0

    for batch in loader:
        visual, spk, masked, spec, phase, length, text, mask, path = unpack(batch, device)
        sig += signature(path, mask)
        bs = spec.size(0)

        c = forward_av(model, visual, spk, masked, length, mask, phase=phase)
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

        errors = compute_phase_error_tensors(c, phase, prediction_mask, gt_mag)

        frame_pos, frame_depth = frame_position_and_depth(prediction_mask)
        temporal_pos, temporal_depth = temporal_position_and_depth(
            prediction_mask, frame_pos, frame_depth
        )

        # Relative magnitude in dB uses each utterance's own STFT peak as 0 dB.
        # This avoids treating speaker recording level as phase importance.
        peak = gt_mag.amax(dim=(1, 2), keepdim=True).clamp_min(EPS)

        for etype, (circ, angle, valid_mask, mag) in errors.items():
            rel_db = 20.0 * torch.log10((mag / peak).clamp_min(1e-12))

            if etype == "temporal":
                pos = temporal_pos.unsqueeze(1).expand_as(circ)
                dep = temporal_depth.unsqueeze(1).expand_as(circ)
            else:
                pos = frame_pos.unsqueeze(1).expand_as(circ)
                dep = frame_depth.unsqueeze(1).expand_as(circ)

            collectors.add_error_type(
                method=method,
                gap=gap,
                error_type=etype,
                circ=circ,
                angle=angle,
                valid_mask=valid_mask,
                mag=mag,
                rel_db=rel_db,
                position_values=pos,
                depth_values=dep,
            )

    if sig != reference_signature:
        raise RuntimeError(
            f"Phase diagnostic for {method}, gap={gap} did not use exactly the same samples/masks as F."
        )

    alignment = alignment_sum / max(alignment_n, 1)
    # Previous E oracle was ~1e-5 rad. This guard is intentionally loose enough
    # for numerical noise but strict enough to catch a preprocessing mismatch.
    if alignment > 1e-3:
        raise RuntimeError(
            f"Stored phase is not aligned with freshly computed GT STFT: weighted MAE={alignment:.3e} rad"
        )
    return alignment


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="F oracle + phase-error-by-energy/gap-position diagnostic on GRID."
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
        "--previous-oracle-summary",
        default=None,
        help="Optional path to oracle_phase_diagnostic_summary.csv from the previous A-E test.",
    )
    args = ap.parse_args()

    set_global_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_root = args.checkpoint_root or project_checkpoint_dir("AV_PLC")
    log_root = Path(project_log_dir("AV_PLC"))
    outdir = Path(args.output_dir) if args.output_dir else (
        log_root / f"oracle_phase_followup_{ARCH}_{args.phase_variant}" / DATASET
    )

    previous_path = (
        Path(args.previous_oracle_summary)
        if args.previous_oracle_summary
        else log_root
        / f"oracle_phase_diagnostic_{ARCH}_{args.phase_variant}"
        / DATASET
        / "oracle_phase_diagnostic_summary.csv"
    )
    previous = load_previous_oracle(previous_path)

    print(f"device={device}")
    print(f"checkpoints={ckpt_root}")
    print(f"output={outdir}")
    print(f"previous_oracle={previous_path} ({'found' if previous else 'not found'})")

    f_summary = []
    f_deltas = []
    reference_signatures = {}
    collectors = ErrorCollectors()
    phase_alignment = {}

    # ------------------------------------------------------------------
    # FIRST: F oracle, using no-phase checkpoints.
    # ------------------------------------------------------------------
    print("\n================ F ORACLE ================")
    for method in args.methods:
        name = model_name(method, None)
        checkpoint = os.path.join(ckpt_root, name, "best_model.pt")
        print(f"\n[{method}] no-phase checkpoint: {checkpoint}")
        model = build_model(method, False, checkpoint, device)

        for gap in args.gaps:
            print(f"  F/A0 gap={gap} ms")
            A0, F_result, sig = run_F_and_A0(args, gap, device, model)
            reference_signatures[(method, gap)] = sig

            f_summary.append(row(method, gap, "A0_NO_PHASE_COMPLETED_MEL_GRIFFIN_LIM", A0))
            f_summary.append(row(method, gap, "F_NO_PHASE_COMPLETED_MEL_GT_PHASE", F_result))
            f_deltas.append(
                row(
                    method,
                    gap,
                    "F_minus_A0",
                    delta(F_result, A0),
                    "GT phase vs Griffin-Lim with identical no-phase completed Mel",
                )
            )

            # Direct comparisons against the previous oracle, when available.
            prev_A = previous.get((method, gap, "A_NO_PHASE_MEL_GRIFFIN_LIM"))
            prev_C = previous.get((method, gap, "C_PHASE_MODEL_MEL_GT_PHASE"))
            prev_D = previous.get(("oracle", gap, "D_GT_MEL_GT_PHASE"))
            if prev_A:
                f_deltas.append(
                    row(
                        method,
                        gap,
                        "F_minus_previous_A",
                        delta(F_result, prev_A),
                        "end-to-end gain vs previous no-phase raw-Mel Griffin-Lim baseline; not a strictly fixed-Mel comparison",
                    )
                )
            if prev_C:
                f_deltas.append(
                    row(
                        method,
                        gap,
                        "F_minus_previous_C",
                        delta(F_result, prev_C),
                        "no-phase Mel vs joint-phase-model Mel with GT phase in both",
                    )
                )
            if prev_D:
                f_deltas.append(
                    row(
                        method,
                        gap,
                        "D_minus_F",
                        delta(prev_D, F_result),
                        "remaining Mel-prediction headroom after giving no-phase model GT phase",
                    )
                )

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # SECOND: phase-error diagnostic, using learned-phase checkpoints.
    # ------------------------------------------------------------------
    print("\n============= PHASE ERROR DIAGNOSTIC =============")
    for method in args.methods:
        name = model_name(method, args.phase_variant)
        checkpoint = os.path.join(ckpt_root, name, "best_model.pt")
        print(f"\n[{method}] phase checkpoint: {checkpoint}")
        model = build_model(method, True, checkpoint, device)

        for gap in args.gaps:
            print(f"  phase errors gap={gap} ms")
            align = run_phase_error_diagnostic(
                args=args,
                gap=gap,
                device=device,
                method=method,
                model=model,
                reference_signature=reference_signatures[(method, gap)],
                collectors=collectors,
            )
            phase_alignment[(method, gap)] = align
            print(f"    stored-vs-fresh phase weighted MAE={align:.3e} rad")

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    overall, energy, position, depth = collectors.rows(phase_alignment)

    outdir.mkdir(parents=True, exist_ok=True)
    save_csv(
        f_summary,
        outdir / "F_oracle_followup_summary.csv",
        fixed=["method", "gap_ms", "condition", "comparison"],
    )
    save_csv(
        f_deltas,
        outdir / "F_oracle_followup_deltas.csv",
        fixed=["method", "gap_ms", "condition", "comparison"],
    )
    save_csv(
        overall,
        outdir / "phase_error_overall.csv",
        fixed=["method", "gap_ms", "error_type"],
    )
    save_csv(
        energy,
        outdir / "phase_error_by_energy.csv",
        fixed=["method", "gap_ms", "error_type", "energy_bin", "energy_low_db", "energy_high_db"],
    )
    save_csv(
        position,
        outdir / "phase_error_by_gap_position.csv",
        fixed=["method", "gap_ms", "error_type", "position_bin", "position_low", "position_high", "position_definition"],
    )
    save_csv(
        depth,
        outdir / "phase_error_by_gap_depth.csv",
        fixed=["method", "gap_ms", "error_type", "depth_bin", "depth_low", "depth_high", "depth_definition"],
    )

    print("\nSaved:")
    for fn in (
        "F_oracle_followup_summary.csv",
        "F_oracle_followup_deltas.csv",
        "phase_error_overall.csv",
        "phase_error_by_energy.csv",
        "phase_error_by_gap_position.csv",
        "phase_error_by_gap_depth.csv",
    ):
        print(outdir / fn)

    print("\nInterpretation order:")
    print("  1) F-A0: GT-phase benefit with the no-phase model's Mel held exactly fixed.")
    print("  2) F-C : if positive, joint phase training/path degraded Mel relative to the no-phase path under identical GT phase.")
    print("  3) D-F : remaining Mel headroom once phase is perfect.")
    print("  4) phase_error_by_gap_depth.csv: does error grow from observed boundaries toward the gap center?")
    print("  5) phase_error_by_energy.csv: are high-magnitude TF bins disproportionately inaccurate?")
    print("  6) Compare unweighted vs magnitude-weighted errors before proposing a magnitude-weighted phase loss.")


if __name__ == "__main__":
    main()
