import os
import torch
import glob
import argparse

from save_features import extract_features_parallel
from model import AV_ReVoice
from trainer import Trainer, setup_logging
from av_l_dataloader import AVDataloader

def main(args):
    if args.save_features:
        splits = {"test", "val", "train"}

        for split in splits:
            path = os.path.join('datasets', args.dataset, split)
            video_list = glob.glob(os.path.join(path, 's*/*.mpg'))
            feats_filename = os.path.join('datasets', args.dataset, f"{args.dataset}_{split}_features.h5")
            extract_features_parallel(video_list, feats_filename)
        return

    # Setup
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    av_dataloader = AVDataloader(mode='v', dataset_name=args.dataset,
                                 batch_size=args.batch_size, num_workers=args.num_workers)
    train_loader = av_dataloader.train_dataloader()
    val_loader = av_dataloader.val_dataloader()

    for l2s_flag in [True]:
        model_name = f"synth_plc_{'_asr' if l2s_flag else ''}{'_pesq' if args.pesq else ''}({args.dataset})"
        logger = setup_logging(model_name, args.log_dir)
        logger.info(f"Using device: {device}")

        model = AV_ReVoice(l2s_loss=l2s_flag)
        logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
        logger.info("Initializing dataloaders...")

        trainer = Trainer(
            model=model,
            mode='v',
            l2s_loss=l2s_flag,
            pesq_loss=args.pesq,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=args.learning_rate,
            device=device,
            vocoder_path=None,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir
        )

        logger.info(f"Starting training for {args.epochs} epochs...")
        start_epoch = trainer.load_checkpoint(load_best=True)
        trainer.train(num_epochs=args.epochs, start_epoch=start_epoch)

        for plc_loss_rate in args.plc_rates:
            test_loader = av_dataloader.test_dataloader(plc_loss_rate)
            logger.info("Evaluating model on test set...")
            test_loss = trainer.evaluate(test_loader, plc_loss_rate)
            logger.info(f"Final test loss: {test_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AV_ReVoice model or extract features")
    parser.add_argument('--save_features', action='store_true', help='Whether to extract features')
    parser.add_argument('--dataset', type=str, default='grid', help='Name of the dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for dataloaders')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for dataloaders')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--pesq', action='store_true', help='Use PESQ loss')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--plc_rates', nargs='+', default=['20', '30', '40', '50', '60', 'rand'],
                        help='PLC loss rates to evaluate on')

    args = parser.parse_args()
    main(args)
