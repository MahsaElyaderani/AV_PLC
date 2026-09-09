import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import argparse
import json
import os

import torch

from evaluations.runtime_config import (
    SEED,
    project_checkpoint_dir,
    project_log_dir,
    set_global_seed,
)
from evaluations.result_utils import save_combined_results

from AV_PLC.av_dataloader import AVDataloader
from AV_PLC.multimodal_decoder import AV_PLC
from AV_PLC.trainer import Trainer, setup_logging


# Representative single-gap conditions.
SINGLE_GAPS_MS = [160, 500, 1000]


def build_test_loader(
    av_dataloader,
    *,
    mask_type: str,
    loss_rate=None,
    gap_ms=None,
    seed: int = SEED,
):
    if mask_type == "gilbert":
        return av_dataloader.test_dataloader(
            mask_range=str(loss_rate),
            mask_type="gilbert",
            gap_ms=None,
            seed=seed,
        )
    if mask_type == "single_gap":
        return av_dataloader.test_dataloader(
            mask_range="10",
            mask_type="single_gap",
            gap_ms=int(gap_ms),
            seed=seed,
        )
    raise ValueError(f"Unsupported mask_type: {mask_type}")


def build_model(device):
    return AV_PLC(
        video_depth=6,
        video_heads=4,
        audio_depth=4,
        audio_heads=4,
        video_hidden_size=256,
        audio_hidden_size=256,
        feat_dim=256,
        audio_ckpt_path=None,
        freeze_audio_enc=False,
    ).to(device)


def l1_baseline_model_name(dataset_name: str) -> str:
    return (
        "av_wide_masking_mlp_av_only_fusion_5loss_bursty"
        "_l1_only"
        "_enc_loss"
        f"({dataset_name})"
    )


def experiment_configs(dataset_name: str):
    return [
        {
            "tag": "l1_only",
            "model_name": l1_baseline_model_name(dataset_name),
            "sc_loss": False,
            "ce_loss": False,
            "synth_sc_weight": 0.0,
            "synth_ce_weight": 0.0,
        },
        {
            "tag": "l1_synth_sc_0p05",
            "model_name": (
                "av_wide_masking_mlp_av_only_fusion_5loss_bursty"
                "_l1_synth_sc_0p05"
                "_enc_loss"
                f"({dataset_name})"
            ),
            "sc_loss": True,
            "ce_loss": False,
            "synth_sc_weight": 0.05,
            "synth_ce_weight": 0.0,
        },
        {
            "tag": "l1_synth_ce_0p005",
            "model_name": (
                "av_wide_masking_mlp_av_only_fusion_5loss_bursty"
                "_l1_synth_ce_0p005"
                "_enc_loss"
                f"({dataset_name})"
            ),
            "sc_loss": False,
            "ce_loss": True,
            "synth_sc_weight": 0.0,
            "synth_ce_weight": 0.005,
        },
    ]


def make_train_val_dataloaders(dataset_name: str, batch_size: int, num_workers: int):
    av_dataloader = AVDataloader(
        mode="av",
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_workers=num_workers,
        dropout_modality=True,
        video_aug=True,
    )
    return av_dataloader.train_dataloader(), av_dataloader.val_dataloader()


def make_eval_dataloader(dataset_name: str, batch_size: int, num_workers: int, test_subset):
    return AVDataloader(
        mode="av",
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_workers=num_workers,
        dropout_modality=False,
        video_aug=False,
        test_subset=test_subset,
    )


def build_trainer(
    *,
    config: dict,
    dataset_name: str,
    batch_size: int,
    num_workers: int,
    learning_rate: float,
    num_epochs: int,
    checkpoint_dir: str,
    log_dir: str,
    vocoder_path,
    device,
):
    train_loader, val_loader = make_train_val_dataloaders(
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    trainer = Trainer(
        model=build_model(device),
        mode="av",
        drop_av=True,
        sc_loss=config["sc_loss"],
        ce_loss=config["ce_loss"],
        pesq_loss=False,
        stoi_loss=False,
        asr_loss=False,
        enc_loss=True,
        model_name=config["model_name"],
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        vocoder_path=vocoder_path,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        mixed_precision=True,
        use_bf16=True,
        cosine_Tmax=num_epochs,
        early_stop_patience=None,
    )

    # These attributes are used in _compute_loss(), they already have been created
    # inside _initialize_components() when the corresponding flag is True.
    trainer.synth_sc_weight = float(config["synth_sc_weight"])
    trainer.synth_ce_weight = float(config["synth_ce_weight"])

    return trainer


def save_experiment_config(trainer: Trainer, config: dict, args: argparse.Namespace):
    payload = {
        "dataset": args.dataset,
        "seed": SEED,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "test_subset": args.test_subset,
        "experiment": config,
        "loss_design": (
            "L1 on fused/audio/synth heads through the existing trainer logic; "
            "optional Mel-SC or Mel-CE is added only to synth_spec."
        ),
    }
    os.makedirs(trainer.run_dir, exist_ok=True)
    path = os.path.join(trainer.run_dir, "synth_loss_ablation_config.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    trainer.logger.info("Saved experiment config to %s", path)


def train_if_needed(
    trainer: Trainer,
    *,
    checkpoint_path: str,
    num_epochs: int,
    save_interval: int,
    samples_to_log: int,
    force_train: bool,
    eval_only: bool,
):
    if os.path.isfile(checkpoint_path) and not force_train:
        trainer.logger.info(
            "Found existing best checkpoint; skipping training: %s",
            checkpoint_path,
        )
        return

    if eval_only:
        raise FileNotFoundError(
            f"Evaluation-only mode requested, but checkpoint is missing: {checkpoint_path}"
        )

    start_epoch = trainer.load_checkpoint(load_best=True)
    trainer.train(
        num_epochs=num_epochs,
        save_interval=save_interval,
        samples_to_log=samples_to_log,
        start_epoch=start_epoch,
    )

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Training finished but best checkpoint was not found: {checkpoint_path}"
        )


def evaluate_fused_single_gaps(trainer, eval_dataloader, config, logger, records):
    for gap_ms in SINGLE_GAPS_MS:
        loader = build_test_loader(
            eval_dataloader,
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
        records["single_gap"].append(
            {
                "source": "evaluate_fused",
                "gap_ms": gap_ms,
                "results": results,
            }
        )
        logger.info("[%s] fused single gap=%d ms complete", config["tag"], gap_ms)


def evaluate_synth_full_clip(trainer, eval_dataloader, config, logger, records):
    # The 500 ms single-gap loader is only used as a deterministic source of
    # test samples. In mask_type='video_only', evaluate_ablation evaluates the
    # synth/video output as full generated speech, without inserting into a gap.
    loader = build_test_loader(
        eval_dataloader,
        mask_type="single_gap",
        gap_ms=500,
    )
    results = trainer.evaluate_ablation(
        loader,
        modes=("video",),
        mask_type="video_only",
        loss_rate=None,
        gap_ms=None,
        skip_load=True,
    )
    records["video_only"].append(
        {
            "source": "evaluate_synth_full_clip",
            "results": results,
        }
    )
    logger.info("[%s] synth full-clip video-only evaluation complete", config["tag"])


def save_records(trainer: Trainer, logger, records: dict):
    for mask_type, mask_records in records.items():
        if not mask_records:
            continue
        combined_path = save_combined_results(
            model_log_dir=trainer.run_dir,
            mask_type=mask_type,
            records=mask_records,
        )
        logger.info("Saved combined %s results to %s", mask_type, combined_path)


def main():
    parser = argparse.ArgumentParser(description=("Train/evaluate the AV-PLC synth-head loss ablation: "
            "L1, L1+synth Mel-SC, and L1+synth Mel-CE."))
    parser.add_argument("--dataset", choices=["grid", "lrs2", "voxceleb2"], required=True,)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--samples-to-log", type=int, default=2)
    parser.add_argument("--test_subset", type=int, default=None,
        help="Number of test samples to evaluate. Default: full test set.",)
    parser.add_argument("--experiments", nargs="+", default=None,
        choices=["l1_only", "l1_synth_sc_0p05", "l1_synth_ce_0p005"],
        help="Optional subset of experiments to run/evaluate.",)
    parser.add_argument("--force-train", action="store_true",
        help="Train even if best_model.pt already exists.",)
    parser.add_argument("--eval-only", action="store_true",
        help="Do not train; require each selected checkpoint to already exist.",)
    parser.add_argument("--skip-eval", action="store_true",
        help="Train/verify checkpoints only; skip test evaluations.",)
    args = parser.parse_args()

    set_global_seed(SEED)

    dataset_name = args.dataset
    log_dir = project_log_dir("AV_PLC")
    checkpoint_dir = project_checkpoint_dir("AV_PLC")
    vocoder_path = None
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configs = experiment_configs(dataset_name)
    if args.experiments is not None:
        selected = set(args.experiments)
        configs = [config for config in configs if config["tag"] in selected]

    for config in configs:
        set_global_seed(SEED)
        model_name = config["model_name"]
        logger = setup_logging(model_name, log_dir)
        logger.info("Starting synth-loss ablation config: %s", config)

        trainer = build_trainer(
            config=config,
            dataset_name=dataset_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            vocoder_path=vocoder_path,
            device=device,
        )
        save_experiment_config(trainer, config, args)

        checkpoint_path = os.path.join(checkpoint_dir, model_name, "best_model.pt")
        train_if_needed(
            trainer,
            checkpoint_path=checkpoint_path,
            num_epochs=args.num_epochs,
            save_interval=args.save_interval,
            samples_to_log=args.samples_to_log,
            force_train=args.force_train,
            eval_only=args.eval_only,
        )

        if args.skip_eval:
            trainer.close()
            continue

        trainer.load_checkpoint(checkpoint_path)

        eval_dataloader = make_eval_dataloader(
            dataset_name=dataset_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            test_subset=args.test_subset,
        )

        records = {"single_gap": [], "video_only": []}
        evaluate_fused_single_gaps(
            trainer=trainer,
            eval_dataloader=eval_dataloader,
            config=config,
            logger=logger,
            records=records,
        )

        evaluate_synth_full_clip(
            trainer=trainer,
            eval_dataloader=eval_dataloader,
            config=config,
            logger=logger,
            records=records,
        )
        save_records(trainer, logger, records)
        trainer.close()


if __name__ == "__main__":
    main()

# Examples:
# python ablation_synth_losses.py --dataset grid
# python ablation_synth_losses.py --dataset lrs2 --test_subset 512
# python ablation_synth_losses.py --dataset voxceleb2 --experiments l1_synth_sc_0p05
# python ablation_synth_losses.py --dataset grid --eval-only

# python AV_PLC/ablation_synth_losses.py --dataset grid --experiments l1_synth_sc_0p05 l1_synth_ce_0p005 --force-train
# python AV_PLC/ablation_synth_losses.py --dataset lrs2 --experiments l1_synth_sc_0p05 l1_synth_ce_0p005
# python AV_PLC/ablation_synth_losses.py --dataset voxceleb2 --experiments l1_synth_sc_0p05 l1_synth_ce_0p005
