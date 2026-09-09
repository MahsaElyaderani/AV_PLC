#!/usr/bin/env python3
"""No-training GRID diagnostic for AV_PLC phase reconstruction.

Conditions (same deterministic test samples/masks):
  A  no-phase model raw predicted Mel + Griffin-Lim (matches current evaluator)
  A2 phase model completed Mel + Griffin-Lim   (extra control; same Mel as B/C)
  B  phase model completed Mel + learned phase
  C  same completed Mel as B + GT phase only in the missing gap
  D  GT Mel + GT stored phase
  E  true STFT magnitude + true STFT phase + current manual iSTFT

A2 is important: B-vs-A2 and C-vs-A2 isolate phase while holding Mel fixed.

"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Find repository root.
HERE = Path(__file__).resolve()
ROOT = None
for p in (HERE.parent, *HERE.parents):
    if (p / "AV_PLC").is_dir() and (p / "evaluations").is_dir():
        ROOT = p
        break
if ROOT is None:
    raise RuntimeError("Copy this script into (or below) the AV_PLC project repository.")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluations.runtime_config import SEED, project_checkpoint_dir, project_log_dir, set_global_seed
from AV_PLC.av_dataloader import AVDataloader
from AV_PLC.multimodal_decoder import AV_PLC
from shared.audio_processing import (
    _manual_istft_center_false,
    hop_len,
    n_fft,
    sample_rate,
    torch_melphase2audio,
    win_len,
)
from shared.metrics import calculate_batch_metrics
from AV_PLC.batch_utils import split_waveform_aux
from AV_PLC.diagnostic_audio import make_frontend, mel_phase_to_audio, fresh_stft, observed_stft

DATASET = "grid"
ARCH = "latent_spectral_v2"
METHODS = ("concat", "temporal_self_cross_attention", "global_local_affinity")
GAPS = (160, 500, 1000)

MODEL_ARGS = dict(
    mel_dim=80, feat_dim=256, dropout=0.1,
    video_depth=6, video_heads=4, video_hidden_size=256,
    audio_depth=4, audio_heads=4, audio_hidden_size=256,
    audio_ckpt_path=None, freeze_audio_enc=False,
)


def model_name(method, phase_variant=None):
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
    raise ValueError(phase_variant)


def build_model(method, phase_reconstruction, checkpoint, device):
    kwargs = dict(**MODEL_ARGS, fusion_type=method, phase_reconstruction=phase_reconstruction)
    if method == "global_local_affinity":
        kwargs.update(
            affinity_dim=128, max_av_offset=16,
            global_temperature=0.1, local_temperature=0.1,
            prior_strength=1.0, prior_sigma=2.0, min_offset_support=4.0,
        )
    model = AV_PLC(**kwargs).to(device)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt.get("state_dict", ckpt)))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def make_loader(args, gap_ms):
    factory = AVDataloader(
        dataset_name="grid", mode="av",
        batch_size=args.batch_size, num_workers=args.num_workers,
        dropout_modality=False, video_aug=False,
        test_subset=args.test_subset,
        temporal_jitter=False, jitter_p=1.0, jitter_max_frames=8,
        phase_reconstruction=True,
        frontend_lookahead_ms=args.frontend_lookahead_ms,
    )
    return factory.test_dataloader(
        mask_range="10", seed=SEED,
        mask_type="single_gap", gap_ms=gap_ms,
    )


def stats(loader):
    ds = loader.dataset
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    return float(ds.mel_mean), float(ds.mel_std)


def unpack(batch, device):
    core, clean_audio, sample_mask, frame_valid, _soft_keep = split_waveform_aux(batch)
    if len(core) != 14:
        raise ValueError(f"Expected phase-enabled AV core batch with 14 elements; got {len(core)}")
    (visual, spk, masked, spec, _video_spec, stft_mag, _video_mag, phase, _video_phase,
     length, text, mask, path, _avail) = core
    return (
        visual.float().to(device), spk.float().to(device), masked.float().to(device),
        spec.float().to(device), stft_mag.float().to(device), phase.float().to(device),
        length.long().to(device), text, mask.float().to(device), path,
        clean_audio.float(), sample_mask.float(), frame_valid.bool(),
    )


class Acc:
    def __init__(self):
        self.s = defaultdict(float)
        self.n = defaultdict(int)

    def add(self, metrics, batch_size):
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                self.s[k] += float(v) * batch_size
                self.n[k] += batch_size

    def result(self):
        return {k: self.s[k] / self.n[k] for k in self.s if self.n[k]}


def signature(path, mask):
    m = mask.detach().cpu().numpy().astype(np.float32, copy=False)
    return [f"{p}::{hashlib.sha256(m[i].tobytes()).hexdigest()[:16]}" for i, p in enumerate(path)]


def metrics(spec, recon_mel, text, mask, path, mel_mean, mel_std,
            clean_audio, sample_mask, audio_length, frame_valid, recon_audio=None):
    kwargs = dict(
        original_batch=spec.detach().cpu(), reconstructed_batch=recon_mel.detach().cpu(),
        texts=text, mask=mask.detach().cpu(), path=list(path),
        hifigan_vocoder=None, tokenizer=None, max_samples=spec.size(0),
        sample_rate=sample_rate, mel_mean=mel_mean, mel_std=mel_std,
        masked_input=False,
        original_audio_batch=clean_audio.detach().cpu(),
        sample_mask_batch=sample_mask.detach().cpu(),
        audio_lengths=audio_length.detach().cpu(),
        frame_valid_batch=frame_valid.detach().cpu(),
        normalize_wer_text=True,
    )
    if recon_audio is not None:
        kwargs["reconstructed_audio_batch"] = recon_audio.detach().cpu()
    return calculate_batch_metrics(**kwargs)


def forward_av(model, visual, masked, length, mask, phase=None, stft_magnitude=None):
    avail = torch.tensor([True, True], dtype=torch.bool, device=masked.device)
    avail = avail.unsqueeze(0).repeat(masked.size(0), 1)
    out = model(masked, visual, None, length, avail=avail, audio_mask=mask, phase=phase, stft_magnitude=stft_magnitude)
    if not isinstance(out, (tuple, list)) or len(out) != 4:
        raise RuntimeError("Expected latent_spectral_v2 AV_PLC to return 4 outputs.")
    completion = out[3]
    if completion is None or completion.get("completed_mel") is None:
        raise RuntimeError("completion_output['completed_mel'] is missing.")
    return completion


def gt_phase_only_in_gap(final_cos, final_sin, gt_phase, mel_mask):
    # Match AV_PLC._audio_reliability exactly for a [B,F,T] mask: mean over Mel bins.
    keep = mel_mask.float().mean(dim=1, keepdim=True).to(final_cos.dtype)
    gt_cos, gt_sin = torch.cos(gt_phase), torch.sin(gt_phase)
    return (
        keep * final_cos + (1.0 - keep) * gt_cos,
        keep * final_sin + (1.0 - keep) * gt_sin,
    )


def true_stft_roundtrip(clean_audio, audio_length, stored_mag, stored_phase, frontend, device):
    """Validate HDF5 waveform <-> active configurable STFT geometry."""
    wav = clean_audio.to(device=device, dtype=torch.float32)
    fresh = fresh_stft(wav, frontend, device=device)
    if tuple(fresh.shape) != tuple(stored_phase.shape):
        raise RuntimeError(f"Fresh STFT {tuple(fresh.shape)} != stored phase {tuple(stored_phase.shape)}")
    dphi = stored_phase.to(device) - torch.angle(fresh)
    dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi))
    phase_mae = dphi.abs().mean(dim=(1, 2))
    mag_mae = (stored_mag.to(device) - fresh.abs()).abs().mean(dim=(1, 2))
    exact = torch.polar(stored_mag.to(device), stored_phase.to(device))
    rt = frontend.istft(exact, length=wav.size(-1))
    rmses=[]; snrs=[]; maxerrs=[]
    for i in range(wav.size(0)):
        n=max(1, min(int(audio_length[i]), wav.size(-1), rt.size(-1)))
        err=rt[i,:n]-wav[i,:n]
        rmses.append(torch.sqrt(err.square().mean()).item())
        snrs.append((10*torch.log10(wav[i,:n].square().sum().clamp_min(1e-20)/err.square().sum().clamp_min(1e-20))).item())
        maxerrs.append(err.abs().max().item())
    return rt, {
        "roundtrip_rmse": float(np.mean(rmses)),
        "roundtrip_snr_db": float(np.mean(snrs)),
        "roundtrip_max_abs_error": float(np.max(maxerrs)),
        "stored_phase_circular_mae_rad": float(phase_mae.mean().item()),
        "stored_magnitude_mae": float(mag_mae.mean().item()),
    }


@torch.inference_mode()
def run_oracles(args, gap, device):
    loader = make_loader(args, gap)
    mel_mean, mel_std = stats(loader)
    frontend = make_frontend(mel_mean, mel_std, args.frontend_lookahead_ms)
    D, E = Acc(), Acc()
    sig = []
    diag_sum = defaultdict(float)
    diag_count = 0
    diag_max = 0.0

    for batch in loader:
        visual, spk, masked, spec, stft_mag, phase, length, text, mask, path, clean_audio, sample_mask, frame_valid = unpack(batch, device)
        sig += signature(path, mask)
        bs = spec.size(0)

        # D: GT Mel + GT stored phase through the learned-phase reconstruction path.
        d_audio = mel_phase_to_audio(
            spec, torch.cos(phase), torch.sin(phase), frontend, output_length=clean_audio.size(-1)
        )
        D.add(metrics(spec, spec, text, mask, path, mel_mean, mel_std, clean_audio, sample_mask, length, frame_valid, d_audio), bs)

        # E: exact STFT complex spectrum -> same manual inverse.
        e_audio, diag = true_stft_roundtrip(clean_audio, length, stft_mag, phase, frontend, device)
        E.add(metrics(spec, spec, text, mask, path, mel_mean, mel_std, clean_audio, sample_mask, length, frame_valid, e_audio), bs)
        for k in ("roundtrip_rmse", "roundtrip_snr_db", "stored_phase_circular_mae_rad"):
            diag_sum[k] += diag[k] * bs
        diag_max = max(diag_max, diag["roundtrip_max_abs_error"])
        diag_count += bs

    diag = {k: v / diag_count for k, v in diag_sum.items()}
    diag["roundtrip_max_abs_error"] = diag_max
    return D.result(), E.result(), diag, sig


@torch.inference_mode()
def run_A(args, gap, device, model, ref_sig):
    loader = make_loader(args, gap)
    mel_mean, mel_std = stats(loader)
    A, sig = Acc(), []
    for batch in loader:
        visual, spk, masked, spec, stft_mag, phase, length, text, mask, path, clean_audio, sample_mask, frame_valid = unpack(batch, device)
        sig += signature(path, mask)
        c = forward_av(model, visual, masked, length, mask, phase=None)
        # A matches the CURRENT no-phase evaluator: raw predicted_mel is sent to
        # Griffin-Lim, and calculate_batch_metrics uses the gap mask for PLC scoring.
        # Do not use A's MSE/PSNR for the completed-Mel comparison; the user plans
        # to fix those separately with M_completed.
        A.add(metrics(spec, c["predicted_mel"], text, mask, path, mel_mean, mel_std, clean_audio, sample_mask, length, frame_valid), spec.size(0))
    if sig != ref_sig:
        raise RuntimeError("A pass did not use exactly the same samples/masks as oracle pass.")
    return A.result()


@torch.inference_mode()
def run_A2_B_C(args, gap, device, model, ref_sig):
    loader = make_loader(args, gap)
    mel_mean, mel_std = stats(loader)
    frontend = make_frontend(mel_mean, mel_std, args.frontend_lookahead_ms)
    A2, B, C, sig = Acc(), Acc(), Acc(), []

    for batch in loader:
        visual, spk, masked, spec, stft_mag, phase, length, text, mask, path, clean_audio, sample_mask, frame_valid = unpack(batch, device)
        sig += signature(path, mask)
        bs = spec.size(0)
        obs = observed_stft(clean_audio, sample_mask, frontend, device)
        c = forward_av(
            model, visual, masked, length, mask,
            phase=torch.angle(obs), stft_magnitude=obs.abs(),
        )
        mel = c["completed_mel"]
        final_cos, final_sin = c.get("final_cos"), c.get("final_sin")
        if final_cos is None or final_sin is None:
            raise RuntimeError("Phase model did not return final_cos/final_sin.")

        # A2: exact same phase-model Mel as B/C, but Griffin-Lim.
        A2.add(metrics(spec, mel, text, mask, path, mel_mean, mel_std, clean_audio, sample_mask, length, frame_valid), bs)

        # B: learned phase.
        b_audio = mel_phase_to_audio(mel, final_cos, final_sin, frontend, output_length=clean_audio.size(-1))
        B.add(metrics(spec, mel, text, mask, path, mel_mean, mel_std, clean_audio, sample_mask, length, frame_valid, b_audio), bs)

        # C: same Mel; replace only missing phase frames with stored GT phase.
        c_cos, c_sin = gt_phase_only_in_gap(final_cos, final_sin, phase, mask)
        c_audio = mel_phase_to_audio(mel, c_cos, c_sin, frontend, output_length=clean_audio.size(-1))
        C.add(metrics(spec, mel, text, mask, path, mel_mean, mel_std, clean_audio, sample_mask, length, frame_valid, c_audio), bs)

    if sig != ref_sig:
        raise RuntimeError("Phase pass did not use exactly the same samples/masks as oracle pass.")
    return A2.result(), B.result(), C.result()


def delta(new, old):
    return {
        k: float(new[k]) - float(old[k])
        for k in set(new) & set(old)
        if isinstance(new[k], (int, float)) and isinstance(old[k], (int, float))
        and np.isfinite(new[k]) and np.isfinite(old[k])
    }


def save_csv(rows, path):
    keys = ["method", "gap_ms", "condition", "comparison"]
    extra = sorted({k for r in rows for k in r if k not in keys})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys + extra)
        w.writeheader()
        w.writerows(rows)


def row(method, gap, condition, values, comparison="", **extra):
    r = dict(method=method, gap_ms=gap, condition=condition, comparison=comparison)
    r.update(values)
    r.update(extra)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    ap.add_argument("--phase-variant", choices=("complex_v1", "plain"), default="complex_v1")
    ap.add_argument("--gaps", nargs="+", type=int, default=list(GAPS))
    ap.add_argument("--test-subset", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--checkpoint-root", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--frontend-lookahead-ms", type=float, default=7.5)
    args = ap.parse_args()

    set_global_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_root = args.checkpoint_root or project_checkpoint_dir("AV_PLC")
    outdir = Path(args.output_dir) if args.output_dir else (
        Path(project_log_dir("AV_PLC")) /
        f"oracle_phase_diagnostic_{ARCH}_{args.phase_variant}" / "grid"
    )

    print(f"device={device}\ncheckpoints={ckpt_root}\noutput={outdir}")
    summary, comparisons = [], []

    # D/E are independent of fusion method; compute once per gap.
    oracle = {}
    ref_sigs = {}
    for gap in args.gaps:
        print(f"\n[oracle] gap={gap} ms")
        D, E, diag, sig = run_oracles(args, gap, device)
        oracle[gap] = (D, E)
        ref_sigs[gap] = sig
        summary += [
            row("oracle", gap, "D_GT_MEL_GT_PHASE", D),
            row("oracle", gap, "E_TRUE_STFT_GT_PHASE", E, **diag),
        ]
        comparisons.append(row(
            "oracle", gap, "E_minus_D", delta(E, D),
            "true-STFT magnitude gain over inverse-Mel GT magnitude"
        ))
        print(
            f"E sanity: RMSE={diag['roundtrip_rmse']:.3e}, "
            f"SNR={diag['roundtrip_snr_db']:.2f} dB, "
            f"maxerr={diag['roundtrip_max_abs_error']:.3e}, "
            f"stored-phase MAE={diag['stored_phase_circular_mae_rad']:.3e} rad"
        )

    for method in args.methods:
        print(f"\n=== {method} ===")

        # A: current no-phase latent model.
        nameA = model_name(method)
        pathA = os.path.join(ckpt_root, nameA, "best_model.pt")
        print("A checkpoint:", pathA)
        modelA = build_model(method, False, pathA, device)
        A_by_gap = {}
        for gap in args.gaps:
            A = run_A(args, gap, device, modelA, ref_sigs[gap])
            A_by_gap[gap] = A
            summary.append(row(method, gap, "A_NO_PHASE_MEL_GRIFFIN_LIM", A))
        del modelA
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # A2/B/C: latest phase model (complex_v1 by default).
        nameP = model_name(method, args.phase_variant)
        pathP = os.path.join(ckpt_root, nameP, "best_model.pt")
        print("Phase checkpoint:", pathP)
        modelP = build_model(method, True, pathP, device)

        for gap in args.gaps:
            A2, B, C = run_A2_B_C(args, gap, device, modelP, ref_sigs[gap])
            D, E = oracle[gap]
            summary += [
                row(method, gap, "A2_PHASE_MODEL_MEL_GRIFFIN_LIM", A2),
                row(method, gap, "B_PHASE_MODEL_MEL_LEARNED_PHASE", B),
                row(method, gap, "C_PHASE_MODEL_MEL_GT_PHASE", C),
            ]
            for label, new, old, desc in (
                ("B_minus_A2", B, A2, "learned phase vs Griffin-Lim with identical Mel"),
                ("C_minus_A2", C, A2, "GT phase vs Griffin-Lim with identical Mel"),
                ("C_minus_B", C, B, "remaining phase-prediction headroom"),
                ("D_minus_C", D, C, "GT-Mel gain with GT phase"),
            ):
                comparisons.append(row(method, gap, label, delta(new, old), desc))

        del modelP
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_csv(summary, outdir / "oracle_phase_diagnostic_summary.csv")
    save_csv(comparisons, outdir / "oracle_phase_diagnostic_deltas.csv")

    print("\nSaved:")
    print(outdir / "oracle_phase_diagnostic_summary.csv")
    print(outdir / "oracle_phase_diagnostic_deltas.csv")
    print("\nInterpret these first:")
    print("  B-A2 : learned phase effect with Mel fixed")
    print("  C-A2 : whether GT phase can beat Griffin-Lim with Mel fixed")
    print("  C-B  : phase-prediction headroom")
    print("  D-C  : predicted-Mel limitation")
    print("  E-D  : inverse-Mel magnitude limitation")
    print("  A is retained only to reproduce the current no-phase baseline; A2 is the controlled GL reference for B/C.")
    print("  E sanity errors should be near zero; otherwise inspect STFT/iSTFT geometry first.")


if __name__ == "__main__":
    main()

# python AV_PLC/grid_phase_oracle_diagnostic.py \
#   --methods concat temporal_self_cross_attention global_local_affinity \
#   --phase-variant complex_v1 \
#   --gaps 160 500 1000 \
#   --test-subset 500