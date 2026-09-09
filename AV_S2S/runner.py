# Portable root configuration
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from evaluations.runtime_config import project_log_dir, project_checkpoint_dir, VOCODER_PATH, SEED, set_global_seed
from evaluations.result_utils import save_combined_results

import os
import torch
import argparse

from AV_S2S.av_l_dataloader import AVDataloader
from AV_S2S.trainer import Trainer, setup_logging
from AV_S2S.revoice_v1 import ReVoice_v1

def av_s2s_runner(mode, phase, dataset_name, asr_flag, pesq_flag, plc_loss_rates, mask_types=("gilbert",), gap_durations=(), save_output=False, sample_paths=None):
    set_global_seed(SEED)

    batch_size = 32
    num_epochs = 100
    learning_rate = 0.001

    dropout = 0.1
    num_layers = 3
    hidden_size = 256
    bi_dir_flag = True

    log_dir = project_log_dir('AV_S2S')
    checkpoint_dir = project_checkpoint_dir('AV_S2S')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = f"revoice_bursty{'_asr' if asr_flag else ''}{'_pesq' if pesq_flag else ''}({dataset_name})"
    logger = setup_logging(model_name, log_dir)
    logger.info(f"Using device: {device}")

    model = ReVoice_v1(bi_dir=bi_dir_flag, asr_loss=asr_flag,
                       hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info("Initializing dataloaders...")

    vocoder_path = None #'/home/amin/Projects/Mahsa/models/AV_PLC/hifigan/checkpoints/model-best.pt'

    if phase == 'train':

        av_dataloader = AVDataloader(mode=mode, dataset_name=dataset_name,
                                     batch_size=batch_size, num_workers=2)
        train_loader = av_dataloader.train_dataloader()
        val_loader = av_dataloader.val_dataloader()

        trainer = Trainer(
            model=model,
            mode=mode,
            asr_loss=asr_flag,
            pesq_loss=False,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=learning_rate,
            device=device,
            vocoder_path=vocoder_path,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            mixed_precision=False,
            early_stopping_patience=101
        )

        logger.info(f"Starting training for {num_epochs} epochs...")
        start_epoch = trainer.load_checkpoint(load_best=False)
        trainer.train(num_epochs=num_epochs, start_epoch=start_epoch)

    if phase == 'test':
        save_metrics = not save_output
        if save_output:
            logger.info("Output-saving mode enabled; metric computation is disabled.")

        combined_records = {"gilbert": [], "single_gap": []}
        av_dataloader = AVDataloader(mode=mode, dataset_name=dataset_name,
                                     batch_size=batch_size, num_workers=4)

        trainer = Trainer(
            model=model,
            mode=mode,
            asr_loss=asr_flag,
            pesq_loss=False,
            model_name=model_name,
            train_loader=None,
            val_loader=None,
            learning_rate=learning_rate,
            device=device,
            vocoder_path=vocoder_path,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            mixed_precision=False,
            early_stopping_patience=101
        )

        if "gilbert" in mask_types:
            for plc_loss_rate in plc_loss_rates:
                test_loader = av_dataloader.test_dataloader(mask_range=plc_loss_rate,
                                                            mask_type="gilbert", seed=SEED)
                logger.info(f"Evaluating Gilbert-Elliott loss rate {plc_loss_rate}%...")
                results = trainer.evaluate(
                    test_loader,
                    loss_rate=plc_loss_rate,
                    mask_type="gilbert",
                    gap_ms=None,
                    save_output=save_output,
                    save_metrics=save_metrics,
                    sample_paths=sample_paths,
                )
                if save_metrics:
                    combined_records["gilbert"].append({
                        "source": "evaluate",
                        "loss_rate": plc_loss_rate,
                        "results": results,
                    })

            if save_metrics:
                csv_path = save_combined_results(
                    model_log_dir=trainer.run_dir,
                    mask_type="gilbert",
                    records=combined_records["gilbert"],
                )
                logger.info(f"Saved combined Gilbert results to {csv_path}")

        if "single_gap" in mask_types:
            for gap_ms in gap_durations:
                test_loader = av_dataloader.test_dataloader(mask_type="single_gap",
                                                            gap_ms=gap_ms, seed=SEED)
                logger.info(f"Evaluating deterministic single gap {gap_ms} ms...")
                results = trainer.evaluate(test_loader, mask_type="single_gap",
                                           loss_rate=None, gap_ms=gap_ms,
                                           save_output=save_output, save_metrics=save_metrics, sample_paths=sample_paths)
                if save_metrics:
                    combined_records["single_gap"].append({
                        "source": "evaluate",
                        "gap_ms": gap_ms,
                        "results": results,
                    })

            if save_metrics:
                csv_path = save_combined_results(
                    model_log_dir=trainer.run_dir,
                    mask_type="single_gap",
                    records=combined_records["single_gap"],
                )
                logger.info(f"Saved combined single-gap results to {csv_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["a", "av"], required=True,
                        help="Modality: 'a:audio' or 'av:audio_video'")
    parser.add_argument("--phase", choices=["train", "test"], required=True,
                        help="Run phase: 'train' or 'test'")
    parser.add_argument("--asr", action="store_true",
                        help="ASR: add asr loss function")
    parser.add_argument("--dataset", choices=['lrs2', 'grid', 'voxceleb2'], required=True, )
    parser.add_argument("--plc_loss_rates", nargs="+", default=['10', '20', '30', '40', '50', '60', '70', '80', '90'],
                        help="PLC loss rates for test mode")
    parser.add_argument("--mask_types", nargs="+", choices=["gilbert", "single_gap"],
                        default=["gilbert", "single_gap"])
    parser.add_argument("--gap_durations", nargs="+", type=int,
                        default=[10, 20, 40, 80, 160, 320, 500, 750, 1000, 1250, 1500])
    parser.add_argument("--save_output", action="store_true",
                        help="Save Mel-spectrogram and waveform outputs.")
    parser.add_argument("--sample_paths", nargs="+", default=None,
                        help="Optional dataset-relative or absolute video paths to save. "
                             "If omitted, save all test utterances.")
    args = parser.parse_args()

    av_s2s_runner(mode=args.mode, phase=args.phase,
                  dataset_name=args.dataset,
                  asr_flag=args.asr, pesq_flag=False,
                  mask_types=args.mask_types,
                  plc_loss_rates=args.plc_loss_rates, gap_durations=args.gap_durations,
                  save_output=args.save_output, sample_paths=args.sample_paths)
