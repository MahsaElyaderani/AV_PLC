import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from evaluations.runtime_config import project_log_dir, project_checkpoint_dir, set_global_seed, SEED
import argparse
import os
import torch

from AV_PLC.multimodal_decoder import AV_PLC
from AV_PLC.av_dataloader import AVDataloader
from AV_PLC.trainer import Trainer, setup_logging

PLC_LOSS_RATES = ["10", "20", "30", "40", "50"] #, "60", "70", "80", "90"
SINGLE_GAPS_MS = [160, 500, 1000, 1500]

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


def evaluate_loss_ablation(trainer, av_dataloader, tag, logger, mask_types ):
    if "gilbert" in mask_types:
        for loss_rate in PLC_LOSS_RATES:
            loader = build_test_loader(
                av_dataloader, mask_type="gilbert", loss_rate=loss_rate
            )
            trainer.evaluate(
                loader,
                loss_rate=loss_rate,
                mask_type="gilbert",
                gap_ms=None,
                save_output=False,
                save_metrics=True,
                skip_load=True,
            )
            logger.info("[%s] GE=%s%% complete", tag, loss_rate)

    if "single_gap" in mask_types:
        for gap_ms in SINGLE_GAPS_MS:
            loader = build_test_loader(
                av_dataloader, mask_type="single_gap", gap_ms=gap_ms
            )
            trainer.evaluate(
                loader,
                loss_rate=None,
                mask_type="single_gap",
                gap_ms=gap_ms,
                save_output=False,
                save_metrics=True,
                skip_load=True,
            )
            logger.info("[%s] single gap=%d ms complete", tag, gap_ms)


def evaluate_modality_ablation(trainer, av_dataloader, logger, mask_types):
    # Video synthesis does not depend on masked audio, so evaluate it once.
    reference_loader = build_test_loader(av_dataloader, mask_type="gilbert", loss_rate=PLC_LOSS_RATES[0])
    trainer.evaluate_ablation(
        reference_loader,
        modes=("video",),
        mask_type="video_only",
        loss_rate=None,
        gap_ms=None,
        skip_load=True,
    )
    logger.info("[modality] video-only evaluation complete")

    if "gilbert" in mask_types:
        for loss_rate in PLC_LOSS_RATES:
            loader = build_test_loader(av_dataloader, mask_type="gilbert", loss_rate=loss_rate)
            trainer.evaluate_ablation(
                loader,
                modes=("av", "audio"),
                mask_type="gilbert",
                loss_rate=loss_rate,
                gap_ms=None,
                skip_load=True,
            )
            logger.info("[modality] GE=%s%% complete", loss_rate)

    if "single_gap" in mask_types:
        for gap_ms in SINGLE_GAPS_MS:
            loader = build_test_loader(av_dataloader, mask_type="single_gap", gap_ms=gap_ms)
            trainer.evaluate_ablation(
                loader,
                modes=("av", "audio"),
                mask_type="single_gap",
                loss_rate=None,
                gap_ms=gap_ms,
                skip_load=True,
            )
            logger.info("[modality] single gap=%d ms complete", gap_ms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["grid", "lrs2", "voxceleb2"], required=True,)
    parser.add_argument("--mask_types", nargs="+", choices=["gilbert", "single_gap"],
                        default=["gilbert", "single_gap"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--train-missing", action="store_true")
    args = parser.parse_args()

    set_global_seed(SEED)
    dataset_name = args.dataset
    num_epochs = 100
    learning_rate = 1e-4
    log_dir = project_log_dir('AV_PLC')
    checkpoint_dir = project_checkpoint_dir('AV_PLC')
    vocoder_path = None
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ablation_configs = [
        {
            "asr_flag": True,
            "pesq_flag": True,
            "enc_loss": True,
            "tag": "full",
            "model_name_override": (
                "av_wide_masking_mlp_av_only_fusion_5loss_bursty2"
                "_plc_a0.05_v0.1_pesq_0.01_asr_0.1"
                f"({dataset_name})"
            ),
            "skip_train": True,
        },
        {
            "asr_flag": False,
            "pesq_flag": True,
            "enc_loss": True,
            "tag": "no_asr",
            "skip_train": True,
        },
        {
            "asr_flag": True,
            "pesq_flag": False,
            "enc_loss": True,
            "tag": "no_pmsqe",
            "skip_train": True,
        },
        {
            "asr_flag": False,
            "pesq_flag": False,
            "enc_loss": True,
            "tag": "l1_only",
            "skip_train": True,
        },
    ]

    for config in ablation_configs:
        set_global_seed(SEED)
        model_name = config.get("model_name_override") or (
            "av_wide_masking_mlp_av_only_fusion_5loss_bursty"
            f"_{config['tag']}"
            f"{'_pesq_0.01' if config['pesq_flag'] else ''}"
            f"{'_asr_0.1' if config['asr_flag'] else ''}"
            "_enc_loss"
            f"({dataset_name})"
        )
        logger = setup_logging(model_name, log_dir)

        model = AV_PLC(
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

        av_dataloader = AVDataloader(
            mode="av",
            dataset_name=dataset_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            dropout_modality=True,
            video_aug=True,
        )
        train_loader = av_dataloader.train_dataloader()
        val_loader = av_dataloader.val_dataloader()

        trainer = Trainer(
            model=model,
            mode="av",
            drop_av=True,
            sc_loss=False,
            pesq_loss=config["pesq_flag"],
            stoi_loss=False,
            asr_loss=config["asr_flag"],
            enc_loss=config["enc_loss"],
            model_name=model_name,
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

        checkpoint_path = os.path.join(checkpoint_dir, model_name, "best_model.pt")
        if not config["skip_train"]:
            start_epoch = trainer.load_checkpoint(load_best=True)
            trainer.train(num_epochs=num_epochs, start_epoch=start_epoch)

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Missing checkpoint for {config['tag']}: {checkpoint_path}"
            )
        trainer.load_checkpoint(checkpoint_path)

        evaluate_loss_ablation(
            trainer=trainer,
            av_dataloader=av_dataloader,
            tag=config["tag"],
            logger=logger,
            mask_types=args.mask_types,
        )

        # Modality ablation is needed only for the full proposed model.
        if config["tag"] == "full":
            evaluate_modality_ablation(
                trainer=trainer,
                av_dataloader=av_dataloader,
                logger=logger,
                mask_types=args.mask_types,
            )


if __name__ == "__main__":
    main()

# python ablation.py --dataset grid
# python ablation.py --dataset lrs2 --mask_types single_gap
# python ablation.py --dataset voxceleb2