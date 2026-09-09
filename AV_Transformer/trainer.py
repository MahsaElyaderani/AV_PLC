import os
import hashlib
import re
from pathlib import Path
import time
import glob
import json
import csv
import logging
import numpy as np
from collections import defaultdict
from tqdm import tqdm, trange
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from asteroid.losses.pmsqe import SingleSrcPMSQE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from AV_Transformer.losses import MaskedLoss
from shared.metrics import calculate_batch_metrics, calculate_metrics
from shared.metrics import Vocoder,torch_mel_to_audio, mel_to_audio_hifigan
from shared.audio_processing import read_gt_input
from evaluations.runtime_config import DATA_ROOT
from shared.text_processing import TextTokenizer

def setup_logging(model_name, log_dir='logs'):
    run_dir = os.path.join(log_dir, f"{model_name}")
    os.makedirs(run_dir, exist_ok=True)

    # Remove existing handlers from root logger
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(run_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(f'{model_name}')


class Trainer:

    def __init__(self, model, model_name, mode, asr_loss=False, pesq_loss=False,
                 train_loader=None, val_loader=None,
                 weight_decay=0.0, mixed_precision=False,
                 early_stopping_patience=15, sample_rate=16000,
                 learning_rate=0.001, device='cuda',
                 checkpoint_dir='checkpoints', log_dir='logs',
                 vocoder_path=None):

        self.model = model
        self.model_name = model_name
        self.mode = mode
        self.asr_loss = asr_loss
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.mixed_precision = mixed_precision
        self.early_stopping_patience = early_stopping_patience
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.sample_rate = sample_rate
        self.tokenizer = None
        self.vocoder = None
        self.vocoder_path = vocoder_path
        self.pesq_loss = pesq_loss
        self.w_pmsqe = 0.0
        self.w = 0.0

        # Dataset-specific normalization used when converting normalized mels
        # back to audio for PESQ/STOI/ASR and saved waveform samples.
        self.mel_mean, self.mel_std = self._infer_mel_stats(train_loader or val_loader)

        if self.vocoder_path is not None:
            self.vocoder = Vocoder(self.vocoder_path)


        self.run_dir = os.path.join(log_dir, f"{model_name}")
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, f"{model_name}")

        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.logger = setup_logging(f"{model_name}", self.run_dir)
        self.logger.info(f"Initializing {model_name} trainer")
        self.writer = SummaryWriter(log_dir=self.run_dir)

        self.config = {
            'model_name': self.model_name,
            'mode': self.mode,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'mixed_precision': self.mixed_precision,
            'early_stopping_patience': self.early_stopping_patience,
            'sample_rate': self.sample_rate
        }

        self._save_config()

        if self.mode in ('a', 'av'):
            # Both reconstruction models must optimize the missing region with
            # the same mask convention (mask: 1=kept, 0=lost).
            self.criterion = MaskedLoss()
        else:
            raise ValueError(f"Unsupported trainer mode: {self.mode}")

        if self.pesq_loss:
            self.pmsqe= SingleSrcPMSQE().to(device)
            self.w_pmsqe = 0.001

        if self.asr_loss:
            self.w = 0.001
            self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
            self.tokenizer = TextTokenizer()


        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                     weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',
                                                              patience=5, factor=0.1,
                                                              min_lr=1e-7)

        self.scaler = GradScaler() if mixed_precision else None

        self.model.to(device)

        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.global_step = 0
        self.train_step = 0

        self.metrics = {
            'train': {'loss': [], 'lr': []},
            'val': {'loss': [], 'pesq': [], 'stoi': [],'estoi': [], 'mse': [], 'psnr': [], 'cer': [], 'wer': [], 'plcmos': []},
            'test': {'loss': [], 'pesq': [], 'stoi': [], 'estoi': [], 'mse': [], 'psnr': [], 'cer': [], 'wer': [], 'plcmos': []}
        }

        self.logger.info(
            f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")

    @staticmethod
    def _infer_mel_stats(loader):
        """Read mel normalization from a DataLoader dataset, unwrapping Subset."""
        dataset = getattr(loader, "dataset", None) if loader is not None else None
        while dataset is not None and hasattr(dataset, "dataset"):
            dataset = dataset.dataset
        return (
            float(getattr(dataset, "mel_mean", -56.775)),
            float(getattr(dataset, "mel_std", 19.707)),
        )

    def _set_mel_stats_from_loader(self, loader):
        self.mel_mean, self.mel_std = self._infer_mel_stats(loader)

    def _save_config(self):

        config_path = os.path.join(self.run_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        self.logger.info(f"Saved configuration to {config_path}")

    def _save_checkpoint(self, epoch, is_best=False):

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step,
            'train_step': self.train_step,
            'metrics': self.metrics
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pt')
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model at epoch {epoch + 1} with validation loss {self.best_val_loss:.4f}")

    def load_checkpoint(self, checkpoint_path=None, load_best=False):

        if checkpoint_path is None:
            if load_best and os.path.exists(os.path.join(self.checkpoint_dir, 'best_model.pt')):
                checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
                self.logger.info(f"Loading best model checkpoint from {checkpoint_path}")
            else:
                checkpoints = glob.glob(os.path.join(self.checkpoint_dir, 'checkpoint_epoch_*.pt'))
                if not checkpoints:
                    self.logger.info("No checkpoints found. Starting from scratch.")
                    return 0

                latest_epoch = max([int(os.path.basename(cp).split('_')[-1].split('.')[0])
                                    for cp in checkpoints])
                checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{latest_epoch}.pt')
                self.logger.info(f"Loading latest checkpoint from {checkpoint_path} (epoch {latest_epoch})")

        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Checkpoint {checkpoint_path} not found. Starting from scratch.")
            return 0

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            if 'scaler_state_dict' in checkpoint and self.scaler is not None:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

            if 'best_val_loss' in checkpoint:
                self.best_val_loss = checkpoint['best_val_loss']

            if 'global_step' in checkpoint:
                self.global_step = checkpoint['global_step']

            if 'train_step' in checkpoint:
                self.train_step = checkpoint['train_step']

            if 'metrics' in checkpoint:
                self.metrics = checkpoint['metrics']

            epoch = checkpoint['epoch']
            self.logger.info(f"Successfully loaded checkpoint from epoch {epoch}")
            self.logger.info(f"Best validation loss so far: {self.best_val_loss:.4f}")

            return epoch

        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {str(e)}")
            self.logger.warning("Starting from scratch.")
            return 0

    def train(self, num_epochs=100, save_interval=10, samples_to_log=2, start_epoch=0):

        self.logger.info(f"Starting training from epoch {start_epoch} for {num_epochs} epochs")
        self.logger.info(f"Training on device: {self.device}")

        total_start_time = time.time()

        epoch_iterator = trange(start_epoch, num_epochs, desc="Training", unit="epoch")
        for epoch in epoch_iterator:
            epoch_start_time = time.time()

            current_lr = self.optimizer.param_groups[0]['lr']
            if current_lr < 1e-7:
                self.logger.info(
                    f"Stopping training. Learning rate ({current_lr:.9f}) is set below threshold (1e-7).")
                break

            train_loss = self._train_epoch(epoch, num_epochs)

            val_metrics = None
            if self.val_loader is not None:
                val_metrics = self._validate_epoch(epoch, num_epochs)
                val_loss = val_metrics['loss']
                self.scheduler.step(val_loss)

                for metric_name, value in val_metrics.items():
                    self.writer.add_scalar(f'Validation/{metric_name}', value, self.global_step)
                    self.metrics['val'][metric_name].append(value)

                if (epoch + 1) % 5 == 0 or epoch == start_epoch:
                    self._log_results(epoch + 1, samples_to_log)

                if val_loss <= self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch, is_best=True)
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
                    if self.epochs_without_improvement >= self.early_stopping_patience:
                        self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                        break

            if (epoch + 1) % save_interval == 0:
                self._save_checkpoint(epoch)

            epoch_time = time.time() - epoch_start_time

            val_metrics_str = ""
            if val_metrics:
                val_metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])

            self.logger.info(
                f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_time:.2f}s - "
                f"Train Loss: {train_loss:.4f}, LR: {current_lr:.9f}"
                + (f", Val: {val_metrics_str}" if val_metrics else "")
            )

            val_loss_str = f", Val Loss: {val_loss:.4f}" if val_metrics else ""
            epoch_iterator.set_postfix_str(f"Train Loss: {train_loss:.4f}{val_loss_str},"
                                           f" LR: {current_lr:.9f}")

            self.writer.add_scalar('Loss/train', train_loss, self.global_step)
            self.writer.add_scalar('Learning_rate', current_lr, self.global_step)

            self.metrics['train']['loss'].append(train_loss)
            self.metrics['train']['lr'].append(current_lr)

        total_time = time.time() - total_start_time
        self.logger.info(f"Training completed in {total_time:.2f}s!")

        metrics_path = os.path.join(self.run_dir, 'metrics.json')

        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4,
                      default=lambda x: float(x)
                      if isinstance(x, (np.float32, np.float64)) else x)
        self.logger.info(f"Saved metrics to {metrics_path}")

        self.writer.close()

    def _train_epoch(self, epoch, num_epochs):

        self.model.train()
        train_loss = 0.0

        train_iterator = tqdm(self.train_loader,
                              desc=f"Epoch {epoch + 1}/{num_epochs} [Train]",
                              leave=False, unit="batch")

        for batch_idx, batch in enumerate(train_iterator):

            if self.mode == 'a':
                masked_spec, spec, text, mask, path = batch

                spec = spec.float().to(self.device)
                masked_spec = masked_spec.float().to(self.device)
                mask = mask.float().to(self.device)

                rec_spec = self.model(masked_spec)
                rec_loss = self.criterion(rec_spec, spec, mask=mask)

                loss = rec_loss

            else:
                masked_spec, visual_feats, spec, text, mask, path = batch

                spec = spec.float().to(self.device)
                masked_spec = masked_spec.float().to(self.device)
                visual_feats = visual_feats.float().to(self.device)
                mask = mask.float().to(self.device)

                rec_spec = self.model(masked_spec, visual_feats)
                rec_loss = self.criterion(rec_spec, spec, mask)

                loss = rec_loss

            self.optimizer.zero_grad()

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            train_loss += loss.item()
            current_lr = self.optimizer.param_groups[0]['lr']

            train_iterator.set_postfix({"loss": f"{loss.item():.4f}",
                                        "lr": f"{current_lr:.9f}"})

            if self.train_step % 1000 == 0:
                self.writer.add_scalar('Loss/train_step', loss.item(), self.train_step)

            self.train_step += 1
            self.global_step += 1

        avg_train_loss = train_loss / len(self.train_loader)
        return avg_train_loss

    def _validate_epoch(self, epoch, num_epochs):

        val_loss = 0.0
        count = 0
        total_metrics = defaultdict(float)

        self.model.eval()

        val_iterator = tqdm(self.val_loader,
                            desc=f"Epoch {epoch + 1}/{num_epochs} [Valid]",
                            leave=False, unit="batch")

        with (torch.no_grad()):
            for batch_idx, batch in enumerate(val_iterator):
                if self.mode == 'a':
                    masked_spec, spec, text, mask, path = batch

                    spec = spec.float().to(self.device)
                    masked_spec = masked_spec.float().to(self.device)
                    mask = mask.float().to(self.device)

                    rec_spec = self.model(masked_spec)
                    rec_loss = self.criterion(rec_spec, spec, mask)

                    loss = rec_loss

                else:
                    masked_spec, visual_feats, spec, text, mask, path = batch

                    spec = spec.float().to(self.device)
                    masked_spec = masked_spec.float().to(self.device)
                    visual_feats = visual_feats.float().to(self.device)
                    mask = mask.float().to(self.device)

                    rec_spec = self.model(masked_spec, visual_feats)
                    rec_loss = self.criterion(rec_spec, spec, mask)

                    loss = rec_loss

                val_loss += loss.item()
                val_iterator.set_postfix({"loss": f"{loss.item():.4f}"})

                if batch_idx == 0:
                    metrics = calculate_batch_metrics(
                        original_batch=spec.detach().cpu(),
                        reconstructed_batch=rec_spec.detach().cpu(),
                        texts=text,
                        path=path,
                        mask=mask.detach().cpu(),
                        hifigan_vocoder=self.vocoder,
                        tokenizer=None,
                        max_samples=len(spec),
                        mel_mean=self.mel_mean,
                        mel_std=self.mel_std
                    )
                    del spec, masked_spec, rec_spec
                    torch.cuda.empty_cache()

                    for k, v in metrics.items():
                        total_metrics[k] += v
                    count += 1

        for k in total_metrics:
            total_metrics[k] /= count

        total_metrics['loss'] = val_loss / len(self.val_loader)
        return total_metrics

    def _log_results(self, epoch, num_samples=2):

        val_dir = os.path.join(self.run_dir, 'validation')
        audio_dir = os.path.join(val_dir, 'audio_samples')
        plot_dir = os.path.join(val_dir, 'plots')
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)

        self.model.eval()
        batch = next(iter(self.val_loader))
        if self.mode == 'a':
            masked_spec, spec, text, mask, path = batch
            spec = spec.float().to(self.device)
            masked_spec = masked_spec.float().to(self.device)

            spec = spec[:num_samples]
            masked_spec = masked_spec[:num_samples]

            with torch.no_grad():
                rec_spec = self.model(masked_spec)

        else:
            masked_spec, visual_feats, spec, text, mask, path = batch
            visual_feats = visual_feats.float().to(self.device)
            spec = spec.float().to(self.device)
            masked_spec = masked_spec.float().to(self.device)

            spec = spec[:num_samples]
            masked_spec = masked_spec[:num_samples]
            visual_feats = visual_feats[:num_samples]

            with torch.no_grad():
                rec_spec = self.model(masked_spec, visual_feats)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig, axes = plt.subplots(num_samples * 2, 3, figsize=(20, 5 * num_samples))

        for i in range(num_samples):
            original_audio, masked_audio = read_gt_input(video_path=path[i],
                                                         mask=mask[i])
            if self.vocoder is None:
                reconstructed_audio = torch_mel_to_audio(rec_spec[i].cpu(), self.mel_mean, self.mel_std)
            else:
                reconstructed_audio = mel_to_audio_hifigan(rec_spec[i], self.vocoder, self.mel_mean, self.mel_std)
            reconstructed_audio = reconstructed_audio.cpu().numpy()

            spec_np = spec[i].cpu().numpy()
            masked_spec_np = masked_spec[i].cpu().numpy()
            rec_spec_np = rec_spec[i].cpu().numpy()

            row = i * 2

            im1 = axes[row, 0].imshow(masked_spec_np, aspect='auto', origin='lower', interpolation='none')
            axes[row, 0].set_title(f'Input Spec - Sample {i + 1}')
            fig.colorbar(im1, ax=axes[row, 0], format='%+2.0f')

            im2 = axes[row, 1].imshow(rec_spec_np, aspect='auto', origin='lower', interpolation='none')
            axes[row, 1].set_title(f'Reconstructed Spec - Sample {i + 1}')
            fig.colorbar(im2, ax=axes[row, 1], format='%+2.0f')

            im4 = axes[row, 2].imshow(spec_np, aspect='auto', origin='lower', interpolation='none')
            axes[row, 2].set_title(f'Original Spec - Sample {i + 1}')
            fig.colorbar(im4, ax=axes[row, 2], format='%+2.0f')

            mse = np.mean((spec_np - rec_spec_np) ** 2)
            axes[row, 0].set_ylabel(f'MSE: {mse:.4f}')

            time_orig = np.arange(len(original_audio)) / self.sample_rate
            time_masked = np.arange(len(masked_audio)) / self.sample_rate
            time_recon = np.arange(len(reconstructed_audio)) / self.sample_rate

            axes[row + 1, 0].plot(time_masked, masked_audio)
            axes[row + 1, 0].set_title(f'Input Waveform - Sample {i + 1}')
            axes[row + 1, 0].set_xlabel('Time (s)')
            axes[row + 1, 0].set_ylabel('Amplitude')

            axes[row + 1, 1].plot(time_recon, reconstructed_audio)
            axes[row + 1, 1].set_title(f'Reconstructed Waveform - Sample {i + 1}')
            axes[row + 1, 1].set_xlabel('Time (s)')

            axes[row + 1, 2].plot(time_orig, original_audio)
            axes[row + 1, 2].set_title(f'Original Waveform - Sample {i + 1}')
            axes[row + 1, 2].set_xlabel('Time (s)')

        plt.suptitle(f'Spectrogram and Waveform Comparison - Epoch {epoch}')
        plt.tight_layout()

        fig_path = os.path.join(plot_dir, f'Spec_waveform_comparison_epoch{epoch}_{timestamp}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.writer.add_figure(f'Spectrogram_Waveform/epoch_{epoch}', fig, epoch)
        avg_mse = np.mean([np.mean((spec[j].cpu().numpy() - rec_spec[j].cpu().numpy()) ** 2)
                           for j in range(num_samples)])
        self.writer.add_scalar('Validation/MSE', avg_mse, epoch)

    @staticmethod
    def _get_dataset_stats(test_loader):
        dataset = test_loader.dataset
        while hasattr(dataset, "dataset"):
            dataset = dataset.dataset
        return float(getattr(dataset, "mel_mean", -56.775)), float(getattr(dataset, "mel_std", 19.707))


    @staticmethod
    def _relative_dataset_path(path):
        """Return a cross-server identifier relative to the dataset root."""
        normalized = str(path).replace("\\", "/")
        marker = "/datasets/"
        if marker in normalized:
            return normalized.split(marker, 1)[1].lstrip("/")
        if normalized.startswith("datasets/"):
            return normalized[len("datasets/"):].lstrip("/")
        return normalized.lstrip("/")

    @classmethod
    def _resolve_dataset_path(cls, path):
        return Path(DATA_ROOT) / Path(cls._relative_dataset_path(path))

    @classmethod
    def _make_sample_key(cls, path):
        rel_path = cls._relative_dataset_path(path)
        stem = Path(rel_path).with_suffix("").as_posix()
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_")
        digest = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:10]
        return f"{safe}_{digest}"

    @staticmethod
    def _merge_reconstructed_spec(clean_spec, predicted_spec, mask):
        """Use clean Mel values outside gaps and predictions inside gaps."""
        clean = clean_spec.detach().cpu().numpy().astype(np.float32)
        predicted = predicted_spec.detach().cpu().numpy().astype(np.float32)
        mask_np = mask.detach().cpu().numpy().astype(np.float32)
        if clean.shape != predicted.shape:
            raise ValueError(f"Mel shape mismatch: {clean.shape} vs {predicted.shape}")
        if mask_np.shape != clean.shape:
            raise ValueError(f"Mask shape mismatch: {mask_np.shape} vs {clean.shape}")
        return clean * mask_np + predicted * (1.0 - mask_np)

    @staticmethod
    def _insert_reconstructed_gap(original_audio, reconstructed_audio, mask, hop_length=160):
        original_audio = np.asarray(original_audio, dtype=np.float32).squeeze()
        reconstructed_audio = np.asarray(reconstructed_audio, dtype=np.float32).squeeze()
        time_keep = mask[0].detach().cpu().numpy().astype(np.float32)
        waveform_mask = np.repeat(time_keep, hop_length)
        n = min(len(original_audio), len(reconstructed_audio), len(waveform_mask))
        return (original_audio[:n] * waveform_mask[:n] +
                reconstructed_audio[:n] * (1.0 - waveform_mask[:n]))

    @staticmethod
    def _aggregate_metric_chunks(chunks):
        sums, counts = defaultdict(float), defaultdict(int)
        for metrics, num_samples in chunks:
            if num_samples <= 0:
                continue
            for key, value in metrics.items():
                if value is None:
                    continue
                value = float(value)
                if not np.isfinite(value):
                    continue
                sums[key] += value * num_samples
                counts[key] += num_samples
        return {key: sums[key] / counts[key] for key in sums if counts[key] > 0}

    def _write_outputs(self, spec, masked_spec, rec_spec, mask, path, sample_id,
                       timestamp, mel_mean, mel_std, spec_dir=None, audio_dir=None):
        def _to_float(x):
            if x is None:
                return None
            x = np.asarray(x)
            if x.dtype.kind in "iu":
                x = x.astype(np.float32) / np.iinfo(x.dtype).max
            return np.clip(x.astype(np.float32), -1.0, 1.0)

        for i in range(len(spec)):
            sid = sample_id + i
            video_path = str(path[i])
            relative_video_path = self._relative_dataset_path(video_path)
            resolved_video_path = str(self._resolve_dataset_path(video_path))
            sample_key = self._make_sample_key(video_path)

            original_audio, masked_audio = read_gt_input(video_path, mask[i])
            original_audio = _to_float(original_audio)
            masked_audio = _to_float(masked_audio)
            if self.vocoder is None:
                raw_recon = torch_mel_to_audio(rec_spec[i].detach().cpu(), mel_mean, mel_std)
            else:
                raw_recon = mel_to_audio_hifigan(
                    rec_spec[i].detach().cpu(), self.vocoder, mel_mean, mel_std)
            if torch.is_tensor(raw_recon):
                raw_recon = raw_recon.detach().cpu().numpy()
            reconstructed_audio = self._insert_reconstructed_gap(
                original_audio, raw_recon, mask[i])
            merged_spec = self._merge_reconstructed_spec(spec[i], rec_spec[i], mask[i])

            if spec_dir is not None:
                sample_spec_dir = os.path.join(spec_dir, sample_key)
                os.makedirs(sample_spec_dir, exist_ok=True)
                np.savez_compressed(
                    os.path.join(sample_spec_dir, f"{sample_key}_{timestamp}.npz"),
                    sample_id=np.asarray(sid),
                    sample_key=np.asarray(sample_key),
                    relative_video_path=np.asarray(relative_video_path),
                    resolved_video_path=np.asarray(resolved_video_path),
                    original_spec=spec[i].detach().cpu().numpy().astype("float32"),
                    masked_spec=masked_spec[i].detach().cpu().numpy().astype("float32"),
                    predicted_spec=rec_spec[i].detach().cpu().numpy().astype("float32"),
                    merged_spec=merged_spec,
                    mask=mask[i].detach().cpu().numpy().astype("float32"),
                    original_audio=original_audio,
                    masked_audio=masked_audio,
                    reconstructed_audio=reconstructed_audio,
                )
            if audio_dir is not None:
                sample_audio_dir = os.path.join(audio_dir, sample_key)
                os.makedirs(sample_audio_dir, exist_ok=True)
                write(os.path.join(sample_audio_dir, f"{sample_key}_original.wav"),
                      self.sample_rate, original_audio)
                write(os.path.join(sample_audio_dir, f"{sample_key}_masked.wav"),
                      self.sample_rate, masked_audio)
                write(os.path.join(sample_audio_dir, f"{sample_key}_reconstructed.wav"),
                      self.sample_rate, reconstructed_audio)
        return sample_id + len(spec)

    @torch.no_grad()
    def evaluate(self, test_loader, loss_rate=None, mask_type="gilbert", gap_ms=None,
                 save_output=False, save_metrics=True, checkpoint_path=None,
                 skip_load=False, sample_paths=None):

        if mask_type not in {"gilbert", "single_gap"}:
            raise ValueError(f"Unsupported mask_type: {mask_type}")
        if mask_type == "gilbert" and loss_rate is None:
            raise ValueError("loss_rate is required for Gilbert-Elliott evaluation")
        if mask_type == "single_gap" and gap_ms is None:
            raise ValueError("gap_ms is required for single-gap evaluation")

        # Output-only mode: saving reconstructions should not run costly metrics.
        if save_output and save_metrics:
            self.logger.info("Output saving enabled; metric computation is disabled.")
            save_metrics = False


        selected_sample_paths = None
        if sample_paths:
            selected_sample_paths = {self._relative_dataset_path(p) for p in sample_paths}
        saved_sample_paths = set()

        condition_name = f"ge_{loss_rate}" if mask_type == "gilbert" else f"gap_{int(gap_ms)}ms"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = os.path.join(self.run_dir, f"test_{condition_name}")
        os.makedirs(eval_dir, exist_ok=True)
        audio_dir = os.path.join(eval_dir, "audio_samples") if save_output else None
        spec_dir = os.path.join(eval_dir, "spectrograms") if save_output else None
        if save_output:
            os.makedirs(audio_dir, exist_ok=True)
            os.makedirs(spec_dir, exist_ok=True)

        if not skip_load:
            load_path = checkpoint_path or os.path.join(self.checkpoint_dir, "best_model.pt")
            if not os.path.isfile(load_path):
                raise FileNotFoundError(load_path)
            self.load_checkpoint(load_path)

        mel_mean, mel_std = self._get_dataset_stats(test_loader)
        self.mel_mean, self.mel_std = mel_mean, mel_std
        self.model.eval()
        self.logger.info(f"Evaluating {condition_name} with mel_mean={mel_mean}, mel_std={mel_std}")

        model_chunks, masked_chunks = [], []
        specs_accum, recons_accum, masked_accum = [], [], []
        masks_accum, texts_accum, paths_accum = [], [], []
        
        sample_id = 0
        chunk_batches = 20

        for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Testing {condition_name}", unit="batch")):
            if self.mode == "a":
                masked_spec, spec, text, mask, path = batch
                visual_feats = None
            else:
                masked_spec, visual_feats, spec, text, mask, path = batch
                visual_feats = visual_feats.float().to(self.device)
            spec = spec.float().to(self.device)
            masked_spec = masked_spec.float().to(self.device)
            rec_spec = self.model(masked_spec) if self.mode == "a" else self.model(masked_spec, visual_feats)

            if save_output:
                if selected_sample_paths is None:
                    selected_indices = list(range(len(path)))
                else:
                    selected_indices = [
                        i for i, p in enumerate(path)
                        if self._relative_dataset_path(p) in selected_sample_paths
                    ]
                if selected_indices:
                    idx_spec = torch.as_tensor(selected_indices, dtype=torch.long, device=spec.device)
                    idx_mask = idx_spec.to(mask.device)
                    selected_paths = [path[i] for i in selected_indices]
                    sample_id = self._write_outputs(
                        spec.index_select(0, idx_spec),
                        masked_spec.index_select(0, idx_spec.to(masked_spec.device)),
                        rec_spec.index_select(0, idx_spec.to(rec_spec.device)),
                        mask.index_select(0, idx_mask),
                        selected_paths, sample_id, timestamp, mel_mean, mel_std,
                        spec_dir=spec_dir, audio_dir=audio_dir)
                    saved_sample_paths.update(
                        self._relative_dataset_path(p) for p in selected_paths)

            if not save_metrics:
                continue
            specs_accum.append(spec.detach().cpu())
            recons_accum.append(rec_spec.detach().cpu())
            masked_accum.append(masked_spec.detach().cpu())
            masks_accum.append(mask)
            texts_accum.append(text)
            paths_accum.extend(list(path))

            flush = ((batch_idx + 1) % chunk_batches == 0 or batch_idx + 1 == len(test_loader))
            if flush:
                originals = torch.cat(specs_accum, dim=0)
                reconstructions = torch.cat(recons_accum, dim=0)
                masked_inputs = torch.cat(masked_accum, dim=0)
                masks = torch.cat(masks_accum, dim=0)
                texts = torch.cat(texts_accum, dim=0)

                n_samples = originals.size(0)
                common = dict(
                    original_batch=originals, texts=texts, path=list(paths_accum), mask=masks,
                    hifigan_vocoder=self.vocoder, max_samples=n_samples,
                    sample_rate=self.sample_rate, mel_mean=mel_mean, mel_std=mel_std)
                model_metrics = calculate_batch_metrics(
                    reconstructed_batch=reconstructions, masked_input=False,
                    tokenizer=None,
                    **common)
                masked_metrics = calculate_batch_metrics(
                    reconstructed_batch=masked_inputs, masked_input=True,
                    tokenizer=None, **common)
                model_chunks.append((model_metrics, n_samples))
                masked_chunks.append((masked_metrics, n_samples))
                specs_accum.clear(); recons_accum.clear(); masked_accum.clear()
                masks_accum.clear(); texts_accum.clear(); paths_accum.clear()
                
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        if save_output and selected_sample_paths is not None:
            missing_paths = selected_sample_paths - saved_sample_paths
            if missing_paths:
                self.logger.warning(
                    "Requested sample paths not found:\n%s",
                    "\n".join(sorted(missing_paths)))

        if not save_metrics:
            self.logger.info("Outputs saved without computing evaluation metrics.")
            return None

        masked_metrics = self._aggregate_metric_chunks(masked_chunks)
        model_metrics = self._aggregate_metric_chunks(model_chunks)
        all_keys = sorted(set(masked_metrics) | set(model_metrics))
        csv_path = os.path.join(eval_dir, f"test_metrics_{timestamp}.csv")
        fieldnames = ["mask_type", "loss_rate", "gap_ms", "output", "loss"] + all_keys
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            baseline_row = {
                "mask_type": mask_type,
                "loss_rate": loss_rate if mask_type == "gilbert" else "",
                "gap_ms": gap_ms if mask_type == "single_gap" else "",
                "output": "masked_input", "loss": ""}
            baseline_row.update(masked_metrics)
            writer.writerow(baseline_row)
            model_row = {
                "mask_type": mask_type,
                "loss_rate": loss_rate if mask_type == "gilbert" else "",
                "gap_ms": gap_ms if mask_type == "single_gap" else "",
                "output": "reconstructed"}
            model_row.update(model_metrics)
            writer.writerow(model_row)
        self.logger.info(f"Saved metrics CSV to {csv_path}")
        return {"masked_input": masked_metrics, "reconstructed": model_metrics,"csv_path": csv_path}

    def evaluate_plots(self, test_loader, loss_rate):
        self._set_mel_stats_from_loader(test_loader)
        self.model.eval()

        if test_loader is not None:
            self.logger.info("Evaluating on test set...")
            best_ckpt = os.path.join(self.checkpoint_dir, 'best_model.pt')
            if os.path.exists(best_ckpt):
                self.load_checkpoint(best_ckpt)

        eval_dir = os.path.join(self.run_dir, f'test_{loss_rate}')
        spec_dir = os.path.join(eval_dir, 'spectrograms')
        audio_dir = os.path.join(eval_dir, 'audio_samples')
        plot_dir = os.path.join(eval_dir, 'plots')
        for d in (eval_dir, spec_dir, audio_dir, plot_dir):
            os.makedirs(d, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_csv_path = os.path.join(eval_dir, f'sample_metrics_{timestamp}.csv')

        with open(metrics_csv_path, 'w', newline='') as csvfile:
            fieldnames = ['sample_id', 'mse', 'psnr', 'pesq', 'stoi', 'estoi', 'cer', 'wer']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            sample_id = 0
            test_iterator = tqdm(test_loader, desc="Testing", unit="batch")

            with torch.no_grad():
                for batch_idx, batch in enumerate(test_iterator):
                    if self.mode == 'a':
                        masked_spec, spec, text, mask, path = batch
                        spec = spec.float().to(self.device)
                        masked_spec = masked_spec.float().to(self.device)

                        rec_spec = self.model(masked_spec)
                    else:
                        masked_spec, visual_feats, spec, text, mask, path = batch
                        spec = spec.float().to(self.device)
                        masked_spec = masked_spec.float().to(self.device)
                        visual_feats = visual_feats.float().to(self.device)

                        rec_spec = self.model(masked_spec, visual_feats)

                    if batch_idx % 25 == 0:
                        for i in range(len(spec)):

                            original_audio, masked_audio = read_gt_input(video_path=path[i],
                                                                         mask=mask[i])

                            if self.vocoder is None:
                                reconstructed_audio = torch_mel_to_audio(rec_spec[i].detach().cpu(), self.mel_mean, self.mel_std)
                            else:
                                reconstructed_audio = mel_to_audio_hifigan(rec_spec[i], self.vocoder, self.mel_mean, self.mel_std)
                            reconstructed_audio = reconstructed_audio.detach().cpu().numpy()

                            spec_np = spec[i].detach().cpu().numpy()
                            masked_spec_np = masked_spec[i].detach().cpu().numpy()
                            rec_spec_np = rec_spec[i].detach().cpu().numpy()

                            # write(os.path.join(audio_dir, f'original_{sample_id}_{timestamp}.wav'),
                            #       self.sample_rate, original_audio)
                            # write(os.path.join(audio_dir, f'masked_{sample_id}_{timestamp}.wav'),
                            #       self.sample_rate, masked_audio)
                            # write(os.path.join(audio_dir, f'reconstructed_{sample_id}_{timestamp}.wav'),
                            #       self.sample_rate, reconstructed_audio)
                            #
                            # fig, axs = plt.subplots(3, 2, figsize=(15, 12))
                            # im1 = axs[0, 0].imshow(masked_spec_np, aspect='auto',
                            #                        origin='lower', interpolation='none')
                            # axs[0, 0].set_title('Masked Spectrogram')
                            # plt.colorbar(im1, ax=axs[0, 0])
                            # im2 = axs[1, 0].imshow(rec_spec_np, aspect='auto',
                            #                        origin='lower', interpolation='none')
                            # axs[1, 0].set_title('Reconstructed Spectrogram')
                            # plt.colorbar(im2, ax=axs[1, 0])
                            # im3 = axs[2, 0].imshow(spec_np, aspect='auto',
                            #                        origin='lower', interpolation='none')
                            # axs[2, 0].set_title('Original Spectrogram')
                            # plt.colorbar(im3, ax=axs[2, 0])
                            #
                            # t_masked = np.arange(len(masked_audio)) / self.sample_rate
                            # t_recon = np.arange(len(reconstructed_audio)) / self.sample_rate
                            # t_orig = np.arange(len(original_audio)) / self.sample_rate
                            # axs[0, 1].plot(t_masked, masked_audio)
                            # axs[0, 1].set_title('Masked Audio')
                            # axs[1, 1].plot(t_recon, reconstructed_audio)
                            # axs[1, 1].set_title('Reconstructed Audio')
                            # axs[2, 1].plot(t_orig, original_audio)
                            # axs[2, 1].set_title('Original Audio')
                            # for ax in (axs[0, 1], axs[1, 1], axs[2, 1]):
                            #     ax.set_xlabel('Time (s)')
                            # plt.tight_layout()
                            # plt.savefig(os.path.join(plot_dir, f'comparison_{sample_id}_{timestamp}.png'), dpi=300)
                            # plt.close(fig)

                            # --- Save spectrograms and waveforms ---

                            def _to_float_m1p1(x):
                                x = np.asarray(x)
                                if x.dtype.kind in "iu":
                                    x = x.astype(np.float32) / np.iinfo(x.dtype).max
                                return np.clip(x.astype(np.float32), -1.0, 1.0)

                            # Normalize audio before saving
                            original_audio_f32 = _to_float_m1p1(original_audio)
                            masked_audio_f32 = _to_float_m1p1(masked_audio)
                            reconstructed_audio_f32 = _to_float_m1p1(reconstructed_audio)

                            np.savez_compressed(
                                os.path.join(spec_dir, f'transformer_{sample_id}_{timestamp}.npz'),
                                masked_spec=masked_spec_np.astype("float32"),
                                recon_spec=rec_spec_np.astype("float32"),
                                original_spec=spec_np.astype("float32"),
                                masked_audio=masked_audio_f32,
                                recon_audio=reconstructed_audio_f32,
                                original_audio=original_audio_f32,
                            )
                            sample_id += 1

    def evaluate_samples(self, test_loader, loss_rate):
        self._set_mel_stats_from_loader(test_loader)
        eval_dir = os.path.join(self.run_dir, f'test_{loss_rate}')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if test_loader is not None:
            self.logger.info("Evaluating on test set...")
            if os.path.exists(os.path.join(self.checkpoint_dir, 'best_model.pt')):
                self.load_checkpoint(os.path.join(self.checkpoint_dir, 'best_model.pt'))

        self.model.eval()

        metrics_csv_path = os.path.join(eval_dir, f'sample_metrics_{loss_rate}_{timestamp}.csv')
        with open(metrics_csv_path, 'w', newline='') as csvfile:
            fieldnames = ['sample_id', 'path', 'mse', 'psnr', 'pesq', 'stoi', 'estoi', 'cer', 'wer']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            test_iterator = tqdm(test_loader, desc="Testing", unit="batch")
            sample_id = 0

            with torch.no_grad():
                for batch_idx, batch in enumerate(test_iterator):

                    masked_spec, visual_feats, spec, text, mask, path = batch
                    spec = spec.float().to(self.device)
                    masked_spec = masked_spec.float().to(self.device)
                    visual_feats = visual_feats.float().to(self.device)

                    rec_spec = self.model(masked_spec, visual_feats)

                    for i in range(len(spec)):
                        sample_id += 1
                        sample_metrics = calculate_metrics(original_spec=spec[i].detach().cpu(),
                                                           reconstructed_spec=rec_spec[i].detach().cpu(),
                                                           path=path[i],
                                                           text=text[i],
                                                           mask=mask[i],
                                                           hifigan_vocoder=self.vocoder,
                                                           mel_mean=self.mel_mean,
                                                           mel_std=self.mel_std)
                        sample_metrics = {
                            'sample_id': sample_id,
                            'path': path[i],
                            'mse': sample_metrics['mse'],
                            'psnr': sample_metrics['psnr'],
                            'pesq': sample_metrics['pesq'],
                            'stoi': sample_metrics['stoi'],
                            'estoi': sample_metrics['estoi'],
                            'wer': sample_metrics['wer'],
                            'cer': sample_metrics['cer'],
                        }
                        writer.writerow(sample_metrics)

