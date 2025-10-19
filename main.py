import os
import torch
import glob
import argparse
from pathlib import Path

from save_features import extract_features_parallel
from multimodal_decoder import AV_PLC
from trainer import Trainer, setup_logging
from av_dataloader import AVDataloader


# bool parser that accepts: true/false, 1/0, yes/no, y/n
def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true", "t", "1", "yes", "y", "on"}:
        return True
    if s in {"false", "f", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got: {v}")


def main(args):
    if args.save_features:
        splits = {"test", "val", "train"}

        for split in splits:
            path = os.path.join('datasets', args.dataset, split)
            video_list = glob.glob(os.path.join(path, 's*/*.mpg'))
            feats_filename = os.path.join('datasets', args.dataset, f"{args.dataset}_{split}_features.h5")
            extract_features_parallel(video_list, feats_filename)
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = (f"{'av' if args.l2s else 'audio'}"
                  f"_plc_a0.05_v0.1_"
                  f"{'_sc' if args.sc_flags else ''}"
                  f"{'_pesq_0.01' if args.pesq else ''}"
                  f"{'_stoi_0.01' if args.stoi else ''}"
                  f"{'_asr_0.1' if args.asr else ''}"
                  f"({args.datasets})")
    logger = setup_logging(model_name, args.log_dir)

    video_depth = 6 if args.datasets == 'grid' else 6
    video_heads = 4 if args.datasets == 'grid' else 4

    audio_depth = 4 if args.datasets == 'grid' else 4
    audio_heads = 4 if args.datasets == 'grid' else 4

    video_hidden_size = 256 if args.datasets == 'grid' else 256
    audio_hidden_size = 256
    feat_dim = 256  # if dataset_name == 'grid' else 512

    model = AV_PLC(video_depth=video_depth, video_heads=video_heads,
                   audio_depth=audio_depth, audio_heads=audio_heads,
                   video_hidden_size=video_hidden_size,
                   audio_hidden_size=audio_hidden_size, feat_dim=feat_dim).to(device)
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    av_dataloader = AVDataloader(mode='av' if args.l2s else 'a',
                                 dataset_name=args.datasets,
                                 batch_size=args.batch_size, num_workers=4,
                                 dropout_modality=args.l2s, video_aug=True, )
    train_loader = av_dataloader.train_dataloader()
    val_loader = av_dataloader.val_dataloader()

    trainer = Trainer(
        model=model,
        mode='av' if args.l2s else 'a',
        drop_av=args.l2s,
        sc_loss=args.sc_flags,
        pesq_loss=args.pesq,
        stoi_loss=args.stoi,
        asr_loss=args.asr,
        model_name=model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        vocoder_path= str(args.vocoder_path),
        checkpoint_dir=str(args.checkpoint_dir),
        log_dir=str(args.log_dir),
        mixed_precision=args.mixed_precision,
        use_bf16=args.use_bf16,
        cosine_Tmax=args.epochs
    )
    start_epoch = trainer.load_checkpoint(load_best=False)
    trainer.train(num_epochs=args.epochs, start_epoch=start_epoch)

    for plc_loss_rate in args.plc_loss_rates:
        test_loader = av_dataloader.test_dataloader(plc_loss_rate)
        trainer.evaluate(test_loader, plc_loss_rate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AV-PLC training/eval runner (exposes all main() defaults)."
    )

    # Training basics
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--lr", "--learning-rate", dest="learning_rate",
                        type=float, default=1e-4, help="Optimizer learning rate.")
    # Loss toggles
    g = parser.add_argument_group("Loss toggles")
    g.add_argument("--pesq", type=str2bool, default=True, metavar="{true|false}",
                   help="Enable PESQ loss.")
    g.add_argument("--stoi", type=str2bool, default=False, metavar="{true|false}",
                   help="Enable STOI loss.")
    g.add_argument("--asr", type=str2bool, default=True, metavar="{true|false}",
                   help="Enable ASR perceptual loss.")
    g.add_argument("--pretrained-enc", type=str2bool, default=False, metavar="{true|false}",
                   help="Use a pretrained encoder.")
    g.add_argument("--l2s", type=str2bool, default=True, metavar="{true|false}",
                   help="Enable audio-video (lip-to-speech) fusion path (if false, audio-only).")

    # Grid-style options (can pass multiple)
    h = parser.add_argument_group("Dataset options")
    h.add_argument("--datasets", nargs="+",
                   choices=["grid", "lrs2", "voxceleb2"],
                   default=["grid"],
                   help="Datasets to run.")
    h.add_argument("--fusion", nargs="+",
                   choices=["concat_mlp", "gated_sum", "concat_time", "film", "cross_attn"],
                   default=["concat_mlp"],
                   help="Fusion modules to try (used when --l2s true).")
    h.add_argument("--plc-loss-rates", nargs="+", default=["60"],
                   metavar="RATE",
                   help="PLC loss rates to evaluate at test time (e.g., 20 30 40 50 60).")

    h.add_argument("--sc-flags", nargs="+", type=str2bool, default=[False],
                   metavar="{true|false}",
                   help="List of booleans for spectral-consistency (SC) loss grid.")

    # Paths & IO
    p = parser.add_argument_group("Paths")
    p.add_argument("--vocoder-path", type=Path,
                   default=Path("/home/ai/Projects/Mahsa/sources/AV_PLC/hifigan/checkpoints/model-best.pt"),
                   help="HiFi-GAN checkpoint path.")
    p.add_argument("--log-dir", type=Path, default=Path("logs"), help="Logs directory.")
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"),
                   help="Checkpoints directory.")

    # Dataloader / runtime
    r = parser.add_argument_group("Runtime")
    r.add_argument("--num-workers", type=int, default=4, help="Dataloader workers.")
    r.add_argument("--mixed-precision", type=str2bool, default=True, metavar="{true|false}",
                   help="Use automatic mixed precision (AMP).")
    r.add_argument("--use-bf16", type=str2bool, default=True, metavar="{true|false}",
                   help="Prefer bfloat16 where supported.")
    r.add_argument("--cosine-tmax", type=int, default=100,
                   help="T_max for cosine LR schedule (defaults to --epochs if not set).")
    args = parser.parse_args()

    # default cosine_Tmax to epochs if not provided
    if args.cosine_tmax is None:
        args.cosine_tmax = args.epochs

    main(args)
