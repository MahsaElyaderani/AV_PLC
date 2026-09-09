
# Portable root configuration
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from evaluations.runtime_config import project_log_dir, project_checkpoint_dir, VOCODER_PATH, SEED, set_global_seed

import os
import torch
import argparse

from AV_PLC.multimodal_decoder import AV_PLC, Audio_Encoder
from AV_PLC.av_dataloader import AVDataloader
from AV_PLC.trainer import Trainer, setup_logging
from evaluations.result_utils import save_combined_results

def av_conformer_runner(mode, phase, dataset_name, asr_flag, pesq_flag, plc_loss_rates,
                        mask_types=("gilbert",), gap_durations=(), save_output=False,
                        sample_paths=None, phase_reconstruction=False,
                        phase_refine_loss=True, phase_unit_loss=True,
                        phase_temporal_loss=True, phase_frequency_loss=True,
                        phase_init_checkpoint=None,
                        w_phase_refine=0.10, w_phase_unit=0.10,
                        w_phase_temporal=0.05, w_phase_frequency=0.05):
    set_global_seed(SEED)
    if phase_reconstruction and mode != "av":
        raise ValueError("The shared phase-completion head is implemented in AV_PLC; use --mode av.")

    batch_size = 16
    num_epochs = 100
    learning_rate = 1e-4

    sc_flag = False
    l2s_flag = True
    stoi_flag = False
    pretrained_enc = False
    fusion_name = "concat_mlp"#,"gated_sum", "concat_time", "film", "cross_attn"]
    vocoder_path = None #'/home/ai/Projects/Mahsa/sources/AV_PLC/hifigan/checkpoints/model-best.pt'

    log_dir = project_log_dir('AV_PLC')
    checkpoint_dir = project_checkpoint_dir('AV_PLC')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = (f"{'av_wide_masking_mlp_av_only_fusion_5loss_bursty2'}"
                  f"_plc_a0.05_v0.1"
                  f"{'_pretraind' if pretrained_enc else ''}"
                  f"{'_sc' if sc_flag else ''}"
                  f"{'_pesq_0.01' if pesq_flag else ''}"
                  f"{'_stoi_0.01' if stoi_flag else ''}"
                  f"{'_asr_0.1' if asr_flag else ''}"
                  f"{'_phase_recon' if phase_reconstruction else ''}"
                  f"({dataset_name})")
    logger = setup_logging(model_name, log_dir)
    logger.info(f"Using device: {device}; Fusion: {fusion_name}")

    video_depth = 6 if dataset_name == 'grid' else 6
    video_heads = 4 if dataset_name == 'grid' else 4

    audio_depth = 4 if dataset_name == 'grid' else 4
    audio_heads = 4 if dataset_name == 'grid' else 4

    video_hidden_size = 256 if dataset_name == 'grid' else 256
    audio_hidden_size = 256
    feat_dim = 256 #if dataset_name == 'grid' else 512

    if mode == 'a':
        model = Audio_Encoder(conformer_block=audio_depth, num_heads=audio_heads).to(device)
    elif mode == 'av':
        model = AV_PLC(
            video_depth=video_depth, video_heads=video_heads,
            audio_depth=audio_depth, audio_heads=audio_heads,
            video_hidden_size=video_hidden_size,
            audio_hidden_size=audio_hidden_size, feat_dim=feat_dim,
            audio_ckpt_path=None, #"/home/ai/Projects/Mahsa/sources/AV_PLC/checkpoints/audio_plc_pesq_0.01(grid)/best_model.pt",
            freeze_audio_enc=False,
            phase_reconstruction=phase_reconstruction,).to(device)

    if phase_reconstruction and phase_init_checkpoint:
        ckpt = torch.load(phase_init_checkpoint, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
        incompatible = model.load_state_dict(state, strict=False)
        bad_missing = [k for k in incompatible.missing_keys
                       if not (k.startswith("spectral_completion.") or k.startswith("spectral_temporal."))]
        if bad_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Phase initialization mismatch: missing={bad_missing}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        logger.info("Initialized legacy AV_PLC weights from %s", phase_init_checkpoint)

    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    if phase == 'train':
        av_dataloader = AVDataloader(mode=mode,
                                     dataset_name=dataset_name,
                                     batch_size=batch_size, num_workers=8,
                                     dropout_modality=l2s_flag, video_aug=True,
                                     phase_reconstruction=phase_reconstruction,)
        train_loader = av_dataloader.train_dataloader()
        val_loader = av_dataloader.val_dataloader()

        trainer = Trainer(
            model=model,
            mode=mode,
            drop_av=l2s_flag,
            sc_loss=sc_flag,
            ce_loss=False,
            pesq_loss=pesq_flag,
            stoi_loss=stoi_flag,
            asr_loss=asr_flag,
            enc_loss=l2s_flag,
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
            phase_reconstruction=phase_reconstruction,
            phase_refine_loss=phase_refine_loss,
            phase_unit_loss=phase_unit_loss,
            phase_temporal_loss=phase_temporal_loss,
            phase_frequency_loss=phase_frequency_loss,
            w_phase_refine=w_phase_refine,
            w_phase_unit=w_phase_unit,
            w_phase_temporal=w_phase_temporal,
            w_phase_frequency=w_phase_frequency,
        )

        start_epoch = trainer.load_checkpoint(load_best=True)
        trainer.train(num_epochs=num_epochs, start_epoch=start_epoch)

    elif phase == 'test':
        save_metrics = not save_output
        if save_output:
            logger.info("Output-saving mode enabled; metric computation is disabled.")


        combined_records = {"gilbert": [], "single_gap": []}
        av_dataloader = AVDataloader(mode=mode,
                                     dataset_name=dataset_name,
                                     batch_size=batch_size, num_workers=4,
                                     dropout_modality=False, video_aug=False,
                                     phase_reconstruction=phase_reconstruction)

        trainer = Trainer(
            model=model,
            mode=mode,
            drop_av=l2s_flag,
            sc_loss=sc_flag,
            ce_loss=False,
            pesq_loss=pesq_flag,
            stoi_loss=stoi_flag,
            asr_loss=asr_flag,
            enc_loss=True,
            model_name=model_name,
            train_loader=None,
            val_loader=None,
            learning_rate=learning_rate,
            vocoder_path=vocoder_path,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            mixed_precision=False,
            use_bf16=True,
            cosine_Tmax=num_epochs,
            early_stop_patience=None,
            phase_reconstruction=phase_reconstruction,
            phase_refine_loss=phase_refine_loss,
            phase_unit_loss=phase_unit_loss,
            phase_temporal_loss=phase_temporal_loss,
            phase_frequency_loss=phase_frequency_loss,
            w_phase_refine=w_phase_refine,
            w_phase_unit=w_phase_unit,
            w_phase_temporal=w_phase_temporal,
            w_phase_frequency=w_phase_frequency,
        )
        trainer.load_checkpoint(os.path.join(checkpoint_dir, model_name, "best_model.pt"))

        if "gilbert" in mask_types:
            for plc_loss_rate in plc_loss_rates:
                test_loader = av_dataloader.test_dataloader(
                    mask_range=plc_loss_rate, seed=SEED, mask_type="gilbert"
                )
                results = trainer.evaluate(test_loader, mask_type="gilbert",
                                 loss_rate=plc_loss_rate, gap_ms=None,
                                 save_output=save_output, save_metrics=save_metrics, sample_paths=sample_paths, skip_load=True)
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
                test_loader = av_dataloader.test_dataloader(
                    seed=SEED, mask_type="single_gap", gap_ms=gap_ms
                )
                results = trainer.evaluate(test_loader, mask_type="single_gap",
                                           loss_rate=None, gap_ms=gap_ms,
                                           save_output=save_output, save_metrics=save_metrics, sample_paths=sample_paths,
                                           skip_load=True)
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
    parser.add_argument("--pmsqe", action="store_true",
                        help="PMSQE: add pmsqe loss function")
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
    parser.add_argument("--phase-reconstruction", action="store_true",
                        help="Enable the optional shared post-Mel phase-completion head.")
    parser.add_argument("--phase-init-checkpoint", type=str, default=None,
                        help="Optional legacy AV_PLC checkpoint used to initialize non-phase weights.")
    parser.add_argument("--no-phase-refine-loss", action="store_true")
    parser.add_argument("--no-phase-unit-loss", action="store_true")
    parser.add_argument("--no-phase-temporal-loss", action="store_true")
    parser.add_argument("--no-phase-frequency-loss", action="store_true")
    parser.add_argument("--w-phase-refine", type=float, default=0.10)
    parser.add_argument("--w-phase-unit", type=float, default=0.10)
    parser.add_argument("--w-phase-temporal", type=float, default=0.05)
    parser.add_argument("--w-phase-frequency", type=float, default=0.05)
    args = parser.parse_args()

    av_conformer_runner(mode=args.mode, phase=args.phase,
                   dataset_name=args.dataset, plc_loss_rates=args.plc_loss_rates,
                   asr_flag=args.asr, pesq_flag=args.pmsqe, mask_types=args.mask_types,
                   gap_durations=args.gap_durations, save_output=args.save_output,
                   sample_paths=args.sample_paths,
                   phase_reconstruction=args.phase_reconstruction,
                   phase_refine_loss=not args.no_phase_refine_loss,
                   phase_unit_loss=not args.no_phase_unit_loss,
                   phase_temporal_loss=not args.no_phase_temporal_loss,
                   phase_frequency_loss=not args.no_phase_frequency_loss,
                   phase_init_checkpoint=args.phase_init_checkpoint,
                   w_phase_refine=args.w_phase_refine,
                   w_phase_unit=args.w_phase_unit,
                   w_phase_temporal=args.w_phase_temporal,
                   w_phase_frequency=args.w_phase_frequency)