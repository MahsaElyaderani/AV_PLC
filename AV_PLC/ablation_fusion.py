"""
Current-architecture AV-PLC fusion ablation, with optional phase reconstruction.

Important checkpoint policy:
  * Historical fusion-ablation checkpoints are never loaded as full models.
  * The historical concat checkpoint is used only as an audio/video encoder donor.
  * No-phase concat/TSCA/GLA baselines are trained with the current shared-decoder
    architecture and saved under ARCH_VERSION-specific names.
  * Phase runs load the corresponding current no-phase baseline, freeze it, and
    train only the optional shared spectral-completion modules.

Evaluation uses deterministic single gaps of 160, 500, and 1000 ms by default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import torch

_HERE = Path(__file__).resolve()
_ROOT = None
for candidate in (_HERE.parent, *_HERE.parents):
    if (candidate / "AV_PLC").is_dir() and (candidate / "evaluations").is_dir():
        _ROOT = candidate
        break
if _ROOT is None:
    _ROOT = _HERE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluations.runtime_config import SEED, project_checkpoint_dir, project_log_dir, set_global_seed
from AV_PLC.av_dataloader import AVDataloader
from AV_PLC.multimodal_decoder import AV_PLC
from AV_PLC.trainer import Trainer


REPRESENTATIVE_GAPS_MS = [160, 500, 1000]
JITTER_MAX_FRAMES = 8
ARCH_VERSION = "shared_decoder_v1"  # Bump this whenever the model architecture changes.
ABLATION_METHODS = ("concat", "temporal_self_cross_attention", "global_local_affinity",)
PLOT_METRICS = {
    "mse": "MSE (↓)",
    "psnr": "PSNR (dB, ↑)",
    "pesq": "PESQ (↑)",
    "stoi": "STOI (↑)",
    "estoi": "ESTOI (↑)",
    "wer": "WER (↓)",
    "cer": "CER (↓)",
    "plcmos": "PLCMOS (↑)",
}

ENCODER_ARGS = dict(mel_dim=80, feat_dim=256, dropout=0.1,
                    video_depth=6, video_heads=4, video_hidden_size=256,
                    audio_depth=4, audio_heads=4, audio_hidden_size=256,
                    audio_ckpt_path=None, freeze_audio_enc=False,)

def concat_model_name(dataset: str) -> str:
    """Historical concat checkpoint used only to initialize audio/video encoders."""
    return (f"av_wide_masking_mlp_av_only_fusion_5loss_bursty_l1_only_enc_loss({dataset})")

def experiment_suffix(temporal_jitter: bool) -> str:
    """Condition suffix shared by checkpoints, logs, CSVs, and plots."""
    return f"_jitter{JITTER_MAX_FRAMES}" if temporal_jitter else "_no_jitter"

def experiment_name(temporal_jitter: bool, phase_reconstruction: bool = False) -> str:
    name = f"ablation_fusion_{ARCH_VERSION}{experiment_suffix(temporal_jitter)}"
    return name + ("_phase_reconstruction" if phase_reconstruction else "")

def phase_model_name(base_model_name: str) -> str:
    """Keep learned-phase checkpoints separate from all existing fusion checkpoints."""
    return f"{base_model_name}_phase_reconstruction"

def frozen_concat_model_name(dataset: str, temporal_jitter: bool = False) -> str:
    return (f"av_plc_concat_{ARCH_VERSION}_frozen_enc_fused_l1_ge_train"
            f"{experiment_suffix(temporal_jitter)}({dataset})")

def frozen_temporal_self_cross_attention_model_name(dataset: str, temporal_jitter: bool = False,) -> str:
    return (f"av_plc_temporal_self_cross_attention_{ARCH_VERSION}_frozen_enc_fused_l1_ge_train"
            f"{experiment_suffix(temporal_jitter)}({dataset})")

def frozen_global_local_affinity_model_name(dataset: str, temporal_jitter: bool = False,) -> str:
    return (f"av_plc_global_local_affinity_{ARCH_VERSION}_frozen_enc_fused_l1_ge_train"
            f"{experiment_suffix(temporal_jitter)}({dataset})")

def configs(dataset: str, radius: int, args=None):
    temporal_jitter = bool(args and args.temporal_jitter)
    phase_reconstruction = bool(args and args.phase_reconstruction)

    bases = [
        dict(
            tag="concat",
            fusion_type="concat",
            freeze_encoders=True,
            base_model_name=frozen_concat_model_name(dataset, temporal_jitter),
        ),
        dict(
            tag="temporal_self_cross_attention",
            fusion_type="temporal_self_cross_attention",
            freeze_encoders=True,
            base_model_name=frozen_temporal_self_cross_attention_model_name(dataset, temporal_jitter),
        ),
        dict(
            tag="global_local_affinity",
            fusion_type="global_local_affinity",
            freeze_encoders=True,
            base_model_name=frozen_global_local_affinity_model_name(dataset, temporal_jitter),
        ),
    ]
    for config in bases:
        config["model_name"] = (
            phase_model_name(config["base_model_name"])
            if phase_reconstruction
            else config["base_model_name"]
        )
    return bases

def make_shared_encoder_state(args, checkpoint_dir):
    """Extract only audio_enc.* and video_enc.* from the historical concat checkpoint."""
    concat_name = args.concat_model_name or concat_model_name(args.dataset)
    checkpoint_path = os.path.join(checkpoint_dir, concat_name, "best_model.pt",)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError("The historical concat checkpoint is required only as an encoder donor: "
                                f"{checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False,)
    full_state = checkpoint.get("model_state_dict", checkpoint.get("model_state", checkpoint),)
    state = {
        key: value.detach().cpu().clone()
        for key, value in full_state.items()
        if key.startswith("audio_enc.") or key.startswith("video_enc.")
    }
    if not state:
        raise RuntimeError(f"No audio_enc.* or video_enc.* weights found in {checkpoint_path}.")
    return state

def make_shared_phase_state():
    """Create one identical initialization for the new completion modules."""
    set_global_seed(SEED)
    reference = AV_PLC(**ENCODER_ARGS, fusion_type="concat", phase_reconstruction=True,)
    state = {
        key: value.detach().cpu().clone()
        for key, value in reference.state_dict().items()
        if key.startswith("spectral_completion.") or key.startswith("spectral_temporal.")
    }
    if not state:
        raise RuntimeError("Failed to construct shared phase-completion initialization")
    del reference
    return state

def state_hash(state) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()

def _load_current_baseline_checkpoint(model, config, checkpoint_dir):
    """Load this script's current-architecture no-phase baseline into a phase model.

    The phase-enabled model may be missing only the newly introduced phase modules.
    Any other missing/unexpected key means the baseline is not compatible with the
    current ARCH_VERSION and must be retrained with this script first.
    """
    base_path = os.path.join(checkpoint_dir, config["base_model_name"], "best_model.pt")
    if not os.path.isfile(base_path):
        raise FileNotFoundError("Phase reconstruction requires the corresponding current no-phase "
            f"baseline first. Run ablation_fusion.py without --phase-reconstruction. Missing: {base_path}")
    checkpoint = torch.load(base_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint.get("model_state", checkpoint))
    incompatible = model.load_state_dict(state, strict=False)
    allowed_missing_prefixes = ("spectral_completion.", "spectral_temporal.")
    bad_missing = [
        key for key in incompatible.missing_keys
        if not key.startswith(allowed_missing_prefixes)
    ]
    bad_unexpected = list(incompatible.unexpected_keys)
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "Current no-phase baseline is incompatible with the phase model. "
            f"ARCH_VERSION={ARCH_VERSION}; missing={bad_missing}, unexpected={bad_unexpected}. "
            "Retrain the no-phase baseline with the current ablation_fusion.py."
        )
    return base_path

def build_model(config, shared_encoder_state, shared_phase_state, args, checkpoint_dir):
    set_global_seed(SEED)

    model_kwargs = dict(**ENCODER_ARGS, fusion_type=config["fusion_type"],
                        phase_reconstruction=args.phase_reconstruction,)

    # Keep the global-local affinity hyperparameters fixed in code.
    if config["fusion_type"] == "global_local_affinity":
        model_kwargs.update(
            affinity_dim=128,
            max_av_offset=16,
            global_temperature=0.1,
            local_temperature=0.1,
            prior_strength=1.0,
            prior_sigma=2.0,
            min_offset_support=4.0,
        )

    model = AV_PLC(**model_kwargs)

    if args.phase_reconstruction:
        # Load the corresponding current shared-decoder baseline, then train only
        # the new spectral-completion modules. This isolates phase reconstruction.
        _load_current_baseline_checkpoint(model, config, checkpoint_dir)
        incompatible = model.load_state_dict(shared_phase_state, strict=False)
        bad_phase_missing = [key for key in shared_phase_state if key in incompatible.missing_keys]
        bad_phase_unexpected = [
            key for key in incompatible.unexpected_keys
            if key.startswith("spectral_completion.") or key.startswith("spectral_temporal.")
        ]
        if bad_phase_missing or bad_phase_unexpected:
            raise RuntimeError("Shared phase initialization failed: "
                f"missing={bad_phase_missing}, unexpected={bad_phase_unexpected}")
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.spectral_completion.parameters():
            parameter.requires_grad = True
        for parameter in model.spectral_temporal.parameters():
            parameter.requires_grad = True

        original_train = model.train

        def train_phase_only(self, mode=True):
            original_train(mode)
            if mode:
                # Keep the entire current no-phase AV-PLC deterministic/frozen.
                self.audio_enc.eval()
                self.video_enc.eval()
                self.fusion.eval()
                self.temporal.eval()
                self.out.eval()
                self.spectral_completion.train()
                self.spectral_temporal.train()
            return self

        model.train = types.MethodType(train_phase_only, model)
        return model

    # Current no-phase baseline: historical checkpoint contributes encoders only;
    # fusion and the shared temporal decoder are freshly initialized here.
    incompatible = model.load_state_dict(shared_encoder_state, strict=False)
    bad_missing = [
        key for key in incompatible.missing_keys
        if key.startswith("audio_enc.") or key.startswith("video_enc.")
    ]
    bad_unexpected = [
        key for key in incompatible.unexpected_keys
        if key.startswith("audio_enc.") or key.startswith("video_enc.")
    ]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "Pretrained encoder loading failed: "
            f"missing={bad_missing}, unexpected={bad_unexpected}"
        )

    if config["freeze_encoders"]:
        for parameter in model.audio_enc.parameters():
            parameter.requires_grad = False
        for parameter in model.video_enc.parameters():
            parameter.requires_grad = False

        model.audio_enc.eval()
        model.video_enc.eval()
        original_train = model.train

        def train_with_frozen_encoders(self, mode=True):
            original_train(mode)
            if mode:
                self.audio_enc.eval()
                self.video_enc.eval()
            return self

        model.train = types.MethodType(train_with_frozen_encoders, model)

    return model

def make_train_val_loaders(args):
    factory = AVDataloader(
        dataset_name=args.dataset,
        mode="av",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        video_aug=True,
        dropout_modality=True,
        train_subset=args.train_subset,
        val_subset=args.val_subset,
        temporal_jitter=args.temporal_jitter,
        jitter_p=0.5,
        jitter_max_frames=JITTER_MAX_FRAMES,
        phase_reconstruction=args.phase_reconstruction,
    )
    return factory.train_dataloader(), factory.val_dataloader()


def make_eval_factory(args):
    return AVDataloader(
        dataset_name=args.dataset,
        mode="av",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dropout_modality=False,
        test_subset=args.test_subset,
        video_aug=False,
        temporal_jitter=args.temporal_jitter,
        jitter_p=1.0,
        jitter_max_frames=JITTER_MAX_FRAMES,
        phase_reconstruction=args.phase_reconstruction,
    )

def build_trainer(config, args, shared_encoder_state, shared_phase_state, checkpoint_dir, log_dir):
    # Rebuild the same data pipeline for every method.
    set_global_seed(SEED)
    train_loader, val_loader = make_train_val_loaders(args)

    trainer = Trainer(
        model=build_model(config, shared_encoder_state, shared_phase_state, args, checkpoint_dir),
        model_name=config["model_name"],
        mode="av",
        train_loader=train_loader,
        val_loader=val_loader,
        drop_av=True,
        enc_loss=True,
        sc_loss=False,
        ce_loss=False,
        pesq_loss=False,
        stoi_loss=False,
        asr_loss=False,
        learning_rate=args.learning_rate,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        vocoder_path=None,
        mixed_precision=True,
        use_bf16=True,
        grad_clip=1.0,
        cosine_Tmax=args.num_epochs,
        weight_decay=1e-2,
        betas=(0.9, 0.98),
        early_stop_patience=None,
        phase_reconstruction=args.phase_reconstruction,
        phase_refine_loss=not args.no_phase_refine_loss,
        phase_unit_loss=not args.no_phase_unit_loss,
        phase_temporal_loss=not args.no_phase_temporal_loss,
        phase_frequency_loss=not args.no_phase_frequency_loss,
        w_phase_refine=args.w_phase_refine,
        w_phase_unit=args.w_phase_unit,
        w_phase_temporal=args.w_phase_temporal,
        w_phase_frequency=args.w_phase_frequency,
        phase_losses_only=args.phase_reconstruction,
    )

    assert not trainer.sc_loss
    assert not trainer.ce_loss
    assert not trainer.pesq_loss
    assert not trainer.stoi_loss
    assert not trainer.asr_loss
    return trainer

def save_manifest(trainer, config, args, encoder_sha, phase_sha=None):
    manifest = {
        "experiment": experiment_name(args.temporal_jitter, args.phase_reconstruction),
        "dataset": args.dataset,
        "architecture_version": ARCH_VERSION,
        "seed": SEED,
        "method": config,
        "pretrained_encoder_sha256": encoder_sha,
        "shared_phase_initialization_sha256": phase_sha,
        "controlled_settings": {
            "encoders": ENCODER_ARGS,
            "encoder_source": (
                f"corresponding {ARCH_VERSION} no-phase best_model.pt"
                if args.phase_reconstruction else "historical concat checkpoint: audio_enc.* and video_enc.* only"
            ),
            "encoders_frozen": True,
            "phase_reconstruction": args.phase_reconstruction,
            "phase_training": (
                "new spectral-completion modules only; corresponding current no-phase model frozen; phase losses only drive optimization/checkpoint selection"
                if args.phase_reconstruction else "disabled"
            ),
            "phase_loss_weights": {
                "refine": args.w_phase_refine,
                "unit": args.w_phase_unit,
                "temporal": args.w_phase_temporal,
                "frequency": args.w_phase_frequency,
            },
            "training_mask": "bursty GE; per-access loss rate uniform in [0.3, 0.9]",
            "validation_mask": "stable bursty GE; seed=SEED; range [0.3, 0.9]",
            "modality_dropout": True,
            "augmentations": [
                "HorizontalFlip(0.5)",
                "RandomErase(0.4)",
                "TimeMask(0.4 s)",
            ],
            "temporal_jitter_train_val": args.temporal_jitter,
            "jitter_probability_train_val": 0.5 if args.temporal_jitter else 0.0,
            "jitter_max_frames": JITTER_MAX_FRAMES if args.temporal_jitter else 0,
            "loss": (
                "current baseline reconstruction plus optional Mel-refinement/unit-phase/"
                "temporal-phase/frequency-phase losses; only completion modules train"
                if args.phase_reconstruction
                else "L1 only: fused reconstruction plus audio/video encoder auxiliary losses; pretrained encoders frozen"
            ),
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": 1e-2,
            "betas": [0.9, 0.98],
            "epochs": args.num_epochs,
            "scheduler": f"CosineAnnealingLR(T_max={args.num_epochs})",
            "batch_size": args.batch_size,
            "train_subset": args.train_subset,
            "val_subset": args.val_subset,
            "test_subset": args.test_subset,
        },
        "evaluation": {
            "mask_type": "single_gap",
            "single_gap_ms": args.gap_ms,
            "temporal_jitter": args.temporal_jitter,
            "jitter_probability": 1.0 if args.temporal_jitter else 0.0,
            "jitter_max_frames": JITTER_MAX_FRAMES if args.temporal_jitter else 0,
            "test_seed": SEED,
            "output": "phase-refined fused" if args.phase_reconstruction else "fused",
            "waveform_reconstruction": (
                "predicted unit phase + inverse Mel + overlap-add iSTFT"
                if args.phase_reconstruction else "Griffin-Lim"
            ),
        },
    }
    path = os.path.join(trainer.run_dir, "fusion_l1_ablation_config.json")
    os.makedirs(trainer.run_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, default=str)
    trainer.logger.info("Saved experiment manifest: %s", path)


def train_if_needed(trainer, config, args, checkpoint_path):
    force = config["tag"] in set(args.force_train)

    # Reuse is safe only because checkpoint names/results contain ARCH_VERSION.
    # Historical fusion checkpoints use different names and can never enter here.
    if os.path.isfile(checkpoint_path) and not force:
        trainer.logger.info("Reusing current-architecture checkpoint: %s", checkpoint_path)
        return False

    if args.eval_only:
        raise FileNotFoundError(
            f"Evaluation-only mode requested but current checkpoint is missing: {checkpoint_path}"
        )

    # New architecture runs always start from epoch 0.  Do not call
    # Trainer.load_checkpoint() here, because that could revive an incompatible
    # optimizer/model state after architecture refactors.
    start_epoch = 0
    trainer.best_val_loss = float("inf")
    trainer.epoch_no_improve = 0
    trainer.logger.info(
        "%s %s from epoch 0 (%s)",
        "Force-training" if force else "Training",
        config["tag"],
        ARCH_VERSION,
    )

    # Align the runtime random streams before training each method.
    set_global_seed(SEED)
    trainer.train(
        num_epochs=args.num_epochs,
        save_interval=args.save_interval,
        samples_to_log=args.samples_to_log,
        start_epoch=start_epoch,
    )

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Training finished but best checkpoint was not found: {checkpoint_path}"
        )
    return True


def strict_load_model_checkpoint(trainer, checkpoint_path):
    """Avoid Trainer.load_checkpoint silently continuing after an incompatibility."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=trainer.device,
        weights_only=False,
    )
    state = checkpoint.get(
        "model_state_dict",
        checkpoint.get("model_state", checkpoint),
    )
    trainer.model.load_state_dict(state, strict=True)
    trainer.model.eval()
    trainer.logger.info("Strictly loaded checkpoint: %s", checkpoint_path)


def add_result_rows(rows, config, condition, loss_rate, gap_ms, results):
    # Save masked input too, so each condition remains self-contained.
    for output in ("masked_input", "fused"):
        metrics = results.get(output)
        if not isinstance(metrics, dict):
            continue
        row = {
            "method": config["tag"],
            "model_name": config["model_name"],
            "fusion_type": config["fusion_type"],
            # Retained for backward-compatible CSV schema; none of the focused
            # methods uses the local-cross radius.
            "local_radius": "",
            "condition": condition,
            "loss_rate": "" if loss_rate is None else loss_rate,
            "gap_ms": "" if gap_ms is None else gap_ms,
            "output": output,
        }
        row.update(metrics)
        rows.append(row)


def evaluate_method(trainer, config, args):
    """Evaluate only deterministic single gaps."""
    rows = []
    factory = make_eval_factory(args)

    for gap_ms in args.gap_ms:
        loader = factory.test_dataloader(
            mask_range="10",  # ignored by single-gap generation
            seed=SEED,
            mask_type="single_gap",
            gap_ms=gap_ms,
        )
        results = trainer.evaluate(
            loader,
            loss_rate=None,
            mask_type="single_gap",
            gap_ms=gap_ms,
            save_output=False,
            save_metrics=True,
            skip_load=True,
        )
        add_result_rows(
            rows,
            config,
            "single_gap",
            None,
            gap_ms,
            results,
        )
        trainer.logger.info(
            "[%s] single gap %d ms complete",
            config["tag"],
            gap_ms,
        )

    return rows

def save_summary(rows, path):
    if not rows:
        return
    fixed = [
        "method", "model_name", "fusion_type", "local_radius",
        "condition", "loss_rate", "gap_ms", "output",
    ]
    metrics = sorted({key for row in rows for key in row if key not in fixed})
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fixed + metrics)
        writer.writeheader()
        writer.writerows(rows)


def fusion_result_dir(log_dir, dataset, temporal_jitter=False, phase_reconstruction=False):
    """One persistent result folder per dataset, condition, and phase variant."""
    path = Path(log_dir) / experiment_name(temporal_jitter, phase_reconstruction) / dataset
    path.mkdir(parents=True, exist_ok=True)
    return path


def method_summary_path(log_dir, dataset, method, temporal_jitter=False, phase_reconstruction=False):
    return (
        fusion_result_dir(log_dir, dataset, temporal_jitter, phase_reconstruction)
        / f"{method}_summary.csv"
    )


def load_summary(path):
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def summary_is_current(path, checkpoint_path, config, expected_gaps):
    """Return True only when the cached CSV has a fused row for every requested gap."""
    if not path.is_file():
        return False
    if os.path.isfile(checkpoint_path) and path.stat().st_mtime < os.path.getmtime(checkpoint_path):
        return False

    rows = load_summary(path)
    found_gaps = {
        int(float(row.get("gap_ms") or 0))
        for row in rows
        if row.get("output") == "fused"
        and row.get("gap_ms") not in (None, "")
    }
    return set(expected_gaps).issubset(found_gaps)


def _to_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _normalize_row(row: dict, method: str) -> dict:
    """Normalize legacy summary rows to the current common CSV schema."""
    normalized = dict(row)
    # Inject method tag when the CSV was written without one (old concat format).
    if not normalized.get("method"):
        normalized["method"] = method
    # Old concat CSVs use 'mask_type' for what is now called 'condition'.
    if "condition" not in normalized and "mask_type" in normalized:
        normalized["condition"] = normalized["mask_type"]
    return normalized


def plot_fusion_summaries(log_dir, dataset, temporal_jitter=False, phase_reconstruction=False):
    """Create one gap-duration plot per metric from cached method summaries."""
    result_dir = fusion_result_dir(log_dir, dataset, temporal_jitter, phase_reconstruction)
    plot_dir = result_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for method in ABLATION_METHODS:
        path = method_summary_path(log_dir, dataset, method, temporal_jitter, phase_reconstruction)
        if path.is_file():
            raw = load_summary(path)
            # Normalize schema and drop aggregate/summary rows that have no gap_ms.
            summaries[method] = [
                _normalize_row(row, method)
                for row in raw
                if row.get("gap_ms") not in (None, "")
            ]

    if not summaries:
        print(f"No fusion summaries available for {dataset}; no plots created.")
        return

    # Keep the original per-method summaries and also write one comparison-ready
    # CSV in the experiment directory.
    combined_rows = [
        row
        for method in ABLATION_METHODS
        for row in summaries.get(method, [])
    ]
    save_summary(combined_rows, result_dir / "combined_summary.csv")

    for metric, ylabel in PLOT_METRICS.items():
        curves = {}
        all_gaps = set()

        for method, rows in summaries.items():
            points = {}
            for row in rows:
                if row.get("condition") != "single_gap":
                    continue
                if row.get("output") != "fused":
                    continue

                gap_ms = _to_float(row.get("gap_ms"))
                value = _to_float(row.get(metric))
                if gap_ms is not None and value is not None:
                    points[int(gap_ms)] = value

            if points:
                curves[method] = points
                all_gaps.update(points)

        if not curves:
            continue

        ordered_gaps = sorted(all_gaps)
        gap_to_x = {gap: index for index, gap in enumerate(ordered_gaps)}

        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        for method, points in curves.items():
            method_gaps = sorted(points)
            x_values = [gap_to_x[gap] for gap in method_gaps]
            y_values = [points[gap] for gap in method_gaps]
            label = {
                "concat": "MLP concatenation",
                "temporal_self_cross_attention": (
                    "Temporal self + AV cross-attention"
                ),
                "global_local_affinity": "Global-local affinity",
            }.get(method, method)
            ax.plot(x_values, y_values, marker="o", label=label)

        ax.set_title(
            "Frozen encoders — "
            + (
                f"temporal video jitter (max {JITTER_MAX_FRAMES} frames)"
                if temporal_jitter
                else "no temporal video jitter"
            )
        )
        ax.set_xlabel("Audio gap duration (ms)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(
            range(len(ordered_gaps)),
            [str(gap) for gap in ordered_gaps],
        )
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(
            plot_dir / f"{metric}_vs_gap_duration.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)



def plot_phase_vs_baseline_summaries(log_dir, dataset, temporal_jitter=False):
    """Compare learned phase reconstruction against current no-phase Griffin-Lim baselines."""
    phase_dir = fusion_result_dir(
        log_dir, dataset, temporal_jitter, phase_reconstruction=True
    )
    comparison_dir = phase_dir / "phase_vs_griffin_lim"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    rows_out = []
    variants = {}
    for method in ABLATION_METHODS:
        for variant, phase_flag in (("griffin_lim", False), ("learned_phase", True)):
            path = method_summary_path(
                log_dir, dataset, method, temporal_jitter, phase_flag
            )
            if not path.is_file():
                continue
            rows = [
                _normalize_row(row, method)
                for row in load_summary(path)
                if row.get("gap_ms") not in (None, "")
            ]
            variants[(method, variant)] = rows
            for row in rows:
                item = dict(row)
                item["variant"] = variant
                rows_out.append(item)

    if not rows_out:
        return

    fixed = [
        "method", "variant", "model_name", "fusion_type", "local_radius",
        "condition", "loss_rate", "gap_ms", "output",
    ]
    metrics = sorted({key for row in rows_out for key in row if key not in fixed})
    csv_path = comparison_dir / "phase_vs_griffin_lim_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fixed + metrics)
        writer.writeheader()
        writer.writerows(rows_out)

    for metric, ylabel in PLOT_METRICS.items():
        curves = {}
        all_gaps = set()
        for key, rows in variants.items():
            points = {}
            for row in rows:
                if row.get("condition") != "single_gap" or row.get("output") != "fused":
                    continue
                gap_ms = _to_float(row.get("gap_ms"))
                value = _to_float(row.get(metric))
                if gap_ms is not None and value is not None:
                    points[int(gap_ms)] = value
            if points:
                curves[key] = points
                all_gaps.update(points)
        if not curves:
            continue

        ordered_gaps = sorted(all_gaps)
        gap_to_x = {gap: i for i, gap in enumerate(ordered_gaps)}
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        for (method, variant), points in curves.items():
            method_gaps = sorted(points)
            label_method = {
                "concat": "Concat",
                "temporal_self_cross_attention": "TSCA",
                "global_local_affinity": "GLA",
            }.get(method, method)
            label_variant = "learned phase" if variant == "learned_phase" else "Griffin-Lim"
            ax.plot(
                [gap_to_x[g] for g in method_gaps],
                [points[g] for g in method_gaps],
                marker="o",
                linestyle="-" if variant == "learned_phase" else "--",
                label=f"{label_method} — {label_variant}",
            )
        ax.set_title("Learned phase reconstruction vs Griffin-Lim")
        ax.set_xlabel("Audio gap duration (ms)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(ordered_gaps)), [str(g) for g in ordered_gaps])
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(comparison_dir / f"{metric}_phase_vs_griffin_lim.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser(
        description=f"AV-PLC fusion ablation for current architecture {ARCH_VERSION}, with optional learned phase reconstruction."
    )
    parser.add_argument(
        "--dataset",
        choices=["grid", "lrs2", "voxceleb2"],
        required=True,
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=ABLATION_METHODS,
        default=list(ABLATION_METHODS),
    )
    parser.add_argument(
        "--concat-model-name",
        type=str,
        default=None,
        help="Override the historical concat encoder-donor checkpoint folder name.",
    )
    parser.add_argument(
        "--temporal-jitter",
        action="store_true",
        help=(
            "Run the temporal-video-jitter experiment with max_frames=8. "
            "Without this flag, the no-temporal-jitter experiment is run. "
            "Checkpoints, CSVs, and plots are saved in separate condition-specific "
            "folders."
        ),
    )
    parser.add_argument("--local-radius", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--samples-to-log", type=int, default=2)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--val-subset", type=int, default=None)
    parser.add_argument("--test-subset", type=int, default=None)
    parser.add_argument(
        "--gap-ms", nargs="+", type=int, default=REPRESENTATIVE_GAPS_MS
    )
    parser.add_argument(
        "--force-train",
        nargs="*",
        choices=ABLATION_METHODS,
        default=[],
        help=("Retrain selected methods from scratch even when a checkpoint exists. "
            "Use all selected method names after fusion-code changes."),
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--force-eval",
        action="store_true",
        help="Ignore cached method summaries and run evaluation again.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Generate plots from existing CSV summaries without training or inference.",
    )
    parser.add_argument(
        "--phase-reconstruction", action="store_true",
        help=("Load each corresponding current-architecture no-phase baseline, freeze it, "
              "and train/evaluate only the optional shared spectral-completion head. "
              "Run this script without the flag first."),
    )
    parser.add_argument("--no-phase-refine-loss", action="store_true")
    parser.add_argument("--no-phase-unit-loss", action="store_true")
    parser.add_argument("--no-phase-temporal-loss", action="store_true")
    parser.add_argument("--no-phase-frequency-loss", action="store_true")
    # Starting weights for the new auxiliary objectives; validate/tune on validation data.
    parser.add_argument("--w-phase-refine", type=float, default=0.10)
    parser.add_argument("--w-phase-unit", type=float, default=0.10)
    parser.add_argument("--w-phase-temporal", type=float, default=0.05)
    parser.add_argument("--w-phase-frequency", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.local_radius < 0:
        raise ValueError("--local-radius must be non-negative")
    if not args.gap_ms or any(gap <= 0 for gap in args.gap_ms):
        raise ValueError("--gap-ms values must be positive")
    if args.eval_only and args.force_train:
        raise ValueError("--eval-only and --force-train cannot be combined")

    log_dir = project_log_dir("AV_PLC")
    os.makedirs(log_dir, exist_ok=True)

    if args.plot_only:
        missing = []

        for method in args.methods:
            summary_path = method_summary_path(
                log_dir,
                args.dataset,
                method,
                args.temporal_jitter,
                args.phase_reconstruction,
            )

            if not summary_path.is_file():
                missing.append(str(summary_path))

        if missing:
            raise FileNotFoundError(
                "Cannot use --plot-only because these summaries are missing:\n"
                + "\n".join(missing)
            )

        plot_fusion_summaries(
            log_dir=log_dir,
            dataset=args.dataset,
            temporal_jitter=args.temporal_jitter,
            phase_reconstruction=args.phase_reconstruction,
        )
        if args.phase_reconstruction:
            plot_phase_vs_baseline_summaries(log_dir, args.dataset, args.temporal_jitter)

        print(
            f"Plots saved to: "
            f"{fusion_result_dir(log_dir, args.dataset, args.temporal_jitter, args.phase_reconstruction) / 'plots'}"
        )
        return

    checkpoint_dir = project_checkpoint_dir("AV_PLC")
    os.makedirs(checkpoint_dir, exist_ok=True)

    if args.phase_reconstruction:
        # Each phase run is initialized from the corresponding current no-phase
        # baseline produced by this same ARCH_VERSION.
        shared_encoder_state = {}
        encoder_sha = f"inherited-from-{ARCH_VERSION}-method-baseline"
        shared_phase_state = make_shared_phase_state()
        phase_sha = state_hash(shared_phase_state)
    else:
        shared_encoder_state = make_shared_encoder_state(args, checkpoint_dir)
        encoder_sha = state_hash(shared_encoder_state)
        shared_phase_state = {}
        phase_sha = None
    selected = set(args.methods)
    selected_configs = [
        config
        for config in configs(args.dataset, args.local_radius, args)
        if config["tag"] in selected
    ]

    for config in selected_configs:
        set_global_seed(SEED)
        trainer = build_trainer(config, args, shared_encoder_state,
                                shared_phase_state, checkpoint_dir, log_dir,)
        try:
            trainer.logger.info("Fusion ablation config: %s", config)
            trainer.logger.info("Architecture version: %s", ARCH_VERSION)
            trainer.logger.info("Shared encoder SHA256: %s", encoder_sha)
            if phase_sha is not None:
                trainer.logger.info("Shared phase initialization SHA256: %s", phase_sha)
            trainer.logger.info("Trainable parameters: %d",
                sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),)
            save_manifest(trainer, config, args, encoder_sha, phase_sha)

            checkpoint_path = os.path.join(checkpoint_dir, config["model_name"], "best_model.pt",)
            trained_now = train_if_needed(trainer, config, args, checkpoint_path,)

            if args.skip_eval:
                continue

            summary_path = method_summary_path(
                log_dir,
                args.dataset,
                config["tag"],
                args.temporal_jitter,
                args.phase_reconstruction,
            )

            # A newly trained checkpoint must always receive a fresh evaluation.
            use_cached_summary = (
                not args.force_eval
                and not trained_now
                and summary_is_current(
                    summary_path,
                    checkpoint_path,
                    config,
                    args.gap_ms,
                )
            )
            if use_cached_summary:
                trainer.logger.info("Reusing cached evaluation summary: %s",summary_path,)
                continue

            strict_load_model_checkpoint(trainer, checkpoint_path)
            rows = evaluate_method(trainer, config, args)
            save_summary(rows, summary_path)
            trainer.logger.info("Saved method summary: %s", summary_path)
        finally:
            trainer.close()

    # Summaries are the source of truth for plotting. This also regenerates
    # plots when both evaluations were skipped because their CSVs already exist.
    if not args.skip_eval:
        plot_fusion_summaries(log_dir, args.dataset, args.temporal_jitter, args.phase_reconstruction)
        if args.phase_reconstruction:
            plot_phase_vs_baseline_summaries(log_dir, args.dataset, args.temporal_jitter)
        print("Fusion results: "
            f"{fusion_result_dir(log_dir, args.dataset, args.temporal_jitter, args.phase_reconstruction)}")


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
# QUICK GRID WORKFLOW
# -----------------------------------------------------------------------------
# 1) Current shared-decoder baselines (no learned phase; Griffin-Lim evaluation):
#
# python AV_PLC/ablation_fusion.py \
#     --dataset grid \
#     --methods concat temporal_self_cross_attention global_local_affinity \
#     --train-subset 5000 --val-subset 500 --test-subset 500 \
#     --num-epochs 50
#
# 2) learned phase completion.  Requires step 1 checkpoints and the precomputed HDF5 phase datasets:
#
# python AV_PLC/ablation_fusion.py \
#    --dataset grid \
#    --methods concat temporal_self_cross_attention global_local_affinity \
#    --phase-reconstruction \
#    --train-subset 5000 --val-subset 500 --test-subset 500 \
#    --num-epochs 30
#
# Add --temporal-jitter to either command for the separate jitter-8 experiment.
# Add --force-train <method names...> only when intentionally overwriting the
# corresponding ARCH_VERSION checkpoint.
