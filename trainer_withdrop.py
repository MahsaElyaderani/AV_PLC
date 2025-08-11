import os
import csv
import time
import glob
import json
import logging
import librosa
import numpy as np

from tqdm import tqdm, trange
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from collections import defaultdict

import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from asteroid.losses.pmsqe import SingleSrcPMSQE

from audio_processing import torch_mel2spec
from metrics import Vocoder, torch_mel_to_audio
from metrics import calculate_batch_metrics, calculate_metrics
from losses import MSELoss, L1Loss, SpectralConvergenceLoss#, SVTS_Loss



def setup_logging(model_name, log_dir='logs'):
    run_dir = os.path.join(log_dir, f"{model_name}")
    os.makedirs(run_dir, exist_ok=True)

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

    def __init__(self, model, model_name, mode, l2s_loss, pesq_loss,
                 train_loader, val_loader=None, sample_rate=16000,
                 learning_rate=0.001, device='cuda',
                 checkpoint_dir='checkpoints', log_dir='logs',
                 vocoder_path=None):

        self.model = model
        self.model_name = model_name
        self.sample_rate = sample_rate

        self.mode = mode
        self.l2s_loss = l2s_loss
        self.pesq_loss = pesq_loss

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        self.tokenizer = None
        self.vocoder = None
        self.vocoder_path = vocoder_path

        self.w_pmsqe = 0.0
        self.w = 0.0

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
            'sample_rate': self.sample_rate
        }

        self._save_config()

        if self.mode == 'motion' or self.mode == 'a':
            self.rec_criterion = L1Loss().to(device) #MSELoss()
        elif self.mode == 'av':
            self.rec_criterion = L1Loss().to(device) #MSELoss() #MaskedLoss()
        elif self.mode == 'v':
            self.rec_criterion = L1Loss().to(device) #MSELoss()

        if self.pesq_loss:
            self.pesq_criterion = SingleSrcPMSQE().to(device)
            self.w_pmsqe = 0.01 #0.01

        if self.l2s_loss:
            self.w = 1 #0.001
            self.sc_criterion = SpectralConvergenceLoss().to(device)


        # self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
        #                             weight_decay=0.01, betas=(0.9, 0.98))
        #self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=150)  # 150 epochs
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',
                                                              patience=5, factor=0.1,
                                                              verbose=True, min_lr=1e-7)

        self.model.to(device)

        self.best_val_loss = float('inf')
        self.global_step = 0
        self.train_step = 0

        self.metrics = {
            'train': {'loss': [], 'lr': []},
            'val': {'loss': [], 'pesq': [], 'stoi': [], 'mse': [], 'psnr': [], 'cer': [], 'wer': []},
            'test': {'loss': [], 'pesq': [], 'stoi': [],'mse': [], 'psnr': [], 'cer': [], 'wer': []}
        }

        self.logger.info(
            f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")


    def modality_dropout_probs(self, epoch):

        if epoch <= 5:# Start → mostly audio+video
            return [0.9, 0.05, 0.05]
        elif 5 < epoch <= 10:
            return [0.8, 0.1, 0.1]
        elif 10 < epoch <= 15:
            return [0.7, 0.15, 0.15]
        else:
            return [0.6, 0.2, 0.2]

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
            #self.scheduler.step()

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

        if self.mode == 'av':
            new_mode_probs = self.modality_dropout_probs(epoch)
            self.train_loader.dataset.modality_dropout.mode_probs = new_mode_probs
            self.logger.info(f"New modality dropout probability is set to {new_mode_probs}")

        train_iterator = tqdm(self.train_loader,
                              desc=f"Epoch {epoch + 1}/{num_epochs} [Train]",
                              leave=False, unit="batch")

        for batch_idx, batch in enumerate(train_iterator):

            visual_feats, spk_emb, masked_spec, spec, mask = batch

            if self.mode == 'a':
                masked_spec = masked_spec.float().to(self.device)
                spec = spec.float().to(self.device)

                rec_spec = self.model(masked_spec)
                rec_loss = self.rec_criterion(rec_spec, spec)
                synth_loss = 0.0

                if self.pesq_loss:
                    pmsqe_loss = torch.mean(
                        self.pesq_criterion(torch_mel2spec(rec_spec).permute(0, 2, 1),
                                            torch_mel2spec(spec).permute(0, 2, 1)))
                else:
                    pmsqe_loss = 0

                loss = (rec_loss + self.w_pmsqe * pmsqe_loss + self.w * synth_loss)

            elif self.mode == 'v':

                visual_feats = visual_feats.float().to(self.device)
                spk_emb = spk_emb.float().to(self.device)
                spec = spec.float().to(self.device)

                synth_spec = self.model(visual_feats, spk_emb)
                synth_loss = self.rec_criterion(synth_spec, spec)

                if self.l2s_loss:
                    sc_loss = self.sc_criterion(synth_spec, spec)
                else:
                    sc_loss = 0.0

                if self.pesq_loss:
                    pmsqe_loss = torch.mean(
                        self.pesq_criterion(torch_mel2spec(synth_spec).permute(0, 2, 1),
                                            torch_mel2spec(spec).permute(0, 2, 1)))
                else:
                    pmsqe_loss = 0

                loss = (sc_loss + synth_loss + self.w_pmsqe * pmsqe_loss)

            elif self.mode == 'av':

                visual_feats = visual_feats.float().to(self.device)
                spk_emb = spk_emb.float().to(self.device)
                masked_spec = masked_spec.float().to(self.device)
                spec = spec.float().to(self.device)

                rec_spec, synth_spec = self.model(masked_spec, visual_feats, spk_emb)
                rec_loss = self.rec_criterion(rec_spec, spec)
                synth_loss = self.rec_criterion(synth_spec, spec)

                if self.l2s_loss:
                    sc_loss = self.sc_criterion(synth_spec, spec)
                else:
                    sc_loss = 0.0

                if self.pesq_loss:
                    pmsqe_loss = torch.mean(
                        self.pesq_criterion(torch_mel2spec(rec_spec).permute(0, 2, 1),
                                            torch_mel2spec(spec).permute(0, 2, 1)))
                else:
                    pmsqe_loss = 0

                loss = (rec_loss + self.w_pmsqe * pmsqe_loss + self.w * (synth_loss + sc_loss))

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            train_loss += loss.item()
            current_lr = self.optimizer.param_groups[0]['lr']

            train_iterator.set_postfix({"loss": f"{loss.item():.4f}",
                                        "rec loss": f"{rec_loss.item():.5f}" if self.mode == 'a' or self.mode == 'av' else None,
                                        "l2s loss": f"{synth_loss.item():.4f}" if self.mode == 'v' or self.mode == 'av' else None,
                                        "sc loss": f"{sc_loss.item():.4f}" if self.l2s_loss else None,
                                        "pesq loss": f"{pmsqe_loss.item():.4f}" if self.pesq_loss else None,
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

                visual_feats, spk_emb, masked_spec, spec, mask = batch

                if self.mode == 'a':
                    masked_spec = masked_spec.float().to(self.device)
                    spec = spec.float().to(self.device)

                    rec_spec = self.model(masked_spec)
                    rec_loss = self.rec_criterion(rec_spec, spec)
                    synth_loss = 0.0

                    if self.pesq_loss:
                        pmsqe_loss = torch.mean(
                            self.pesq_criterion(torch_mel2spec(rec_spec).permute(0, 2, 1),
                                                torch_mel2spec(spec).permute(0, 2, 1)))
                    else:
                        pmsqe_loss = 0

                    loss = (rec_loss + self.w_pmsqe * pmsqe_loss + self.w * synth_loss)

                elif self.mode == 'v':

                    visual_feats = visual_feats.float().to(self.device)
                    spk_emb = spk_emb.float().to(self.device)
                    spec = spec.float().to(self.device)

                    rec_spec = self.model(visual_feats, spk_emb)
                    synth_loss = self.rec_criterion(rec_spec, spec)

                    if self.l2s_loss:
                        sc_loss = self.sc_criterion(rec_spec, spec)
                    else:
                        sc_loss = 0.0

                    if self.pesq_loss:
                        pmsqe_loss = torch.mean(
                            self.pesq_criterion(torch_mel2spec(rec_spec).permute(0, 2, 1),
                                                torch_mel2spec(spec).permute(0, 2, 1)))
                    else:
                        pmsqe_loss = 0

                    loss = (sc_loss + synth_loss + self.w_pmsqe * pmsqe_loss)

                elif self.mode == 'av':

                    visual_feats = visual_feats.float().to(self.device)
                    spk_emb = spk_emb.float().to(self.device)
                    masked_spec = masked_spec.float().to(self.device)
                    spec = spec.float().to(self.device)

                    rec_spec, synth_spec = self.model(masked_spec, visual_feats, spk_emb)
                    rec_loss = self.rec_criterion(rec_spec, spec)
                    synth_loss = self.rec_criterion(synth_spec, spec)

                    if self.l2s_loss:
                        sc_loss = self.sc_criterion(synth_spec, spec)
                    else:
                        sc_loss = 0.0

                    if self.pesq_loss:
                        pmsqe_loss = torch.mean(
                            self.pesq_criterion(torch_mel2spec(rec_spec).permute(0, 2, 1),
                                                torch_mel2spec(spec).permute(0, 2, 1)))
                    else:
                        pmsqe_loss = 0

                    loss = (rec_loss + self.w_pmsqe * pmsqe_loss + self.w * (synth_loss + sc_loss))

                val_loss += loss.item()
                val_iterator.set_postfix({"loss": f"{loss.item():.4f}"})

                if batch_idx == 0:
                    metrics = calculate_batch_metrics(
                        spec,
                        rec_spec,
                        None,
                        None,#text.cpu().numpy() if self.asr_loss else None,
                        None,#pred_text.cpu().numpy() if self.asr_loss else None,
                        None, #self.vocoder,
                        None,#self.tokenizer if self.asr_loss else None,
                        max_samples=len(spec)
                    )
                    #del spec, masked_spec, rec_spec
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
            visual_feats, spk_emb, masked_spec, spec, mask = batch
            spec = spec.float().to(self.device)
            masked_spec = masked_spec.float().to(self.device)

            spec = spec[:num_samples]
            masked_spec = masked_spec[:num_samples]

            with torch.no_grad():
                rec_spec = self.model(masked_spec)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fig, axes = plt.subplots(num_samples * 2, 3, figsize=(20, 5 * num_samples))

            for i in range(num_samples):
                if self.vocoder is None:
                    original_audio = torch_mel_to_audio(spec[i].cpu(), None)
                    masked_audio = torch_mel_to_audio(masked_spec[i].cpu(), None)
                    reconstructed_audio = torch_mel_to_audio(rec_spec[i].cpu(), None)
                else:
                    original_audio = self.vocoder.convert(spec[i])
                    masked_audio = self.vocoder.convert(masked_spec[i])
                    reconstructed_audio = self.vocoder.convert(rec_spec[i])

                original_audio = original_audio.cpu().numpy()
                masked_audio = masked_audio.cpu().numpy()
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

                # orig_path = os.path.join(audio_dir, f'original_sample{i + 1}_epoch{epoch}_{timestamp}.wav')
                # masked_path = os.path.join(audio_dir, f'masked_sample{i + 1}_epoch{epoch}_{timestamp}.wav')
                # recon_path = os.path.join(audio_dir,
                #                           f'reconstructed_sample{i + 1}_epoch{epoch}_{timestamp}.wav')
                #
                # write(orig_path, self.sample_rate, original_audio)
                # write(masked_path, self.sample_rate, masked_audio)
                # write(recon_path, self.sample_rate, reconstructed_audio)

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


        elif self.mode == 'v':

            visual_feats, spk_emb, masked_spec, spec, mask = batch
            visual_feats = visual_feats.float().to(self.device)
            spk_emb = spk_emb.float().to(self.device)
            spec = spec.float().to(self.device)
            masked_spec = masked_spec.float().to(self.device)

            visual_feats = visual_feats[:num_samples]
            spk_emb = spk_emb[:num_samples]
            spec = spec[:num_samples]

            with torch.no_grad():
                rec_spec = self.model(visual_feats, spk_emb)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fig, axes = plt.subplots(num_samples * 2, 3, figsize=(20, 5 * num_samples))

            for i in range(num_samples):
                if self.vocoder is None:
                    original_audio = torch_mel_to_audio(spec[i].cpu(), None)
                    masked_audio = torch_mel_to_audio(masked_spec[i].cpu(), None)
                    reconstructed_audio = torch_mel_to_audio(rec_spec[i].cpu(), None)
                else:
                    original_audio = self.vocoder.convert(spec[i])
                    masked_audio = self.vocoder.convert(masked_spec[i])
                    reconstructed_audio = self.vocoder.convert(rec_spec[i])

                original_audio = original_audio.cpu().numpy()
                masked_audio = masked_audio.cpu().numpy()
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

                # orig_path = os.path.join(audio_dir, f'original_sample{i + 1}_epoch{epoch}_{timestamp}.wav')
                # masked_path = os.path.join(audio_dir, f'masked_sample{i + 1}_epoch{epoch}_{timestamp}.wav')
                # recon_path = os.path.join(audio_dir,
                #                           f'reconstructed_sample{i + 1}_epoch{epoch}_{timestamp}.wav')
                #
                # write(orig_path, self.sample_rate, original_audio)
                # write(masked_path, self.sample_rate, masked_audio)
                # write(recon_path, self.sample_rate, reconstructed_audio)

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

        elif self.mode == 'av':

            visual_feats, spk_emb, masked_spec, spec, mask = batch
            visual_feats = visual_feats.float().to(self.device)
            spk_emb = spk_emb.float().to(self.device)
            spec = spec.float().to(self.device)
            masked_spec = masked_spec.float().to(self.device)

            spec = spec[:num_samples]
            masked_spec = masked_spec[:num_samples]
            visual_feats = visual_feats[:num_samples]
            spk_emb = spk_emb[:num_samples]

            with torch.no_grad():
                rec_spec, synth_spec = self.model(masked_spec, visual_feats, spk_emb)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                fig, axes = plt.subplots(num_samples * 2, 4, figsize=(20, 5 * num_samples))

                for i in range(num_samples):
                    if self.vocoder is None:
                        original_audio = torch_mel_to_audio(spec[i].cpu(), None)
                        masked_audio = torch_mel_to_audio(masked_spec[i].cpu(), None)
                        reconstructed_audio = torch_mel_to_audio(rec_spec[i].cpu(), None)
                        Synthesized_audio = torch_mel_to_audio(synth_spec[i].cpu(), None)
                    else:
                        original_audio = self.vocoder.convert(spec[i])
                        masked_audio = self.vocoder.convert(masked_spec[i])
                        reconstructed_audio = self.vocoder.convert(rec_spec[i])
                        Synthesized_audio = self.vocoder.convert(synth_spec[i])

                    original_audio = original_audio.cpu().numpy()
                    masked_audio = masked_audio.cpu().numpy()
                    reconstructed_audio = reconstructed_audio.cpu().numpy()
                    Synthesized_audio = Synthesized_audio.cpu().numpy()


                    spec_np = spec[i].cpu().numpy()
                    masked_spec_np = masked_spec[i].cpu().numpy()
                    rec_spec_np = rec_spec[i].cpu().numpy()
                    synth_spec_np = synth_spec[i].cpu().numpy()

                    row = i * 2

                    im1 = axes[row, 0].imshow(masked_spec_np, aspect='auto', origin='lower', interpolation='none')
                    axes[row, 0].set_title(f'Input Spec - Sample {i + 1}')
                    fig.colorbar(im1, ax=axes[row, 0], format='%+2.0f')

                    im2 = axes[row, 1].imshow(rec_spec_np, aspect='auto', origin='lower', interpolation='none')
                    axes[row, 1].set_title(f'Reconstructed Spec - Sample {i + 1}')
                    fig.colorbar(im2, ax=axes[row, 1], format='%+2.0f')

                    im3 = axes[row, 2].imshow(synth_spec_np, aspect='auto', origin='lower', interpolation='none')
                    axes[row, 2].set_title(f'Synthesized Spec - Sample {i + 1}')
                    fig.colorbar(im3, ax=axes[row, 2], format='%+2.0f')

                    im4 = axes[row, 3].imshow(spec_np, aspect='auto', origin='lower', interpolation='none')
                    axes[row, 3].set_title(f'Original Spec - Sample {i + 1}')
                    fig.colorbar(im4, ax=axes[row, 3], format='%+2.0f')

                    mse = np.mean((spec_np - rec_spec_np) ** 2)
                    axes[row, 0].set_ylabel(f'MSE: {mse:.4f}')

                    # orig_path = os.path.join(audio_dir, f'original_sample{i + 1}_epoch{epoch}_{timestamp}.wav')
                    # masked_path = os.path.join(audio_dir, f'masked_sample{i + 1}_epoch{epoch}_{timestamp}.wav')
                    # recon_path = os.path.join(audio_dir,
                    #                           f'reconstructed_sample{i + 1}_epoch{epoch}_{timestamp}.wav')
                    #
                    # write(orig_path, self.sample_rate, original_audio)
                    # write(masked_path, self.sample_rate, masked_audio)
                    # write(recon_path, self.sample_rate, reconstructed_audio)

                    time_orig = np.arange(len(original_audio)) / self.sample_rate
                    time_masked = np.arange(len(masked_audio)) / self.sample_rate
                    time_recon = np.arange(len(reconstructed_audio)) / self.sample_rate
                    time_synth = np.arange(len(Synthesized_audio)) / self.sample_rate

                    axes[row + 1, 0].plot(time_masked, masked_audio)
                    axes[row + 1, 0].set_title(f'Input Waveform - Sample {i + 1}')
                    axes[row + 1, 0].set_xlabel('Time (s)')
                    axes[row + 1, 0].set_ylabel('Amplitude')

                    axes[row + 1, 1].plot(time_recon, reconstructed_audio)
                    axes[row + 1, 1].set_title(f'Reconstructed Waveform - Sample {i + 1}')
                    axes[row + 1, 1].set_xlabel('Time (s)')

                    axes[row + 1, 2].plot(time_synth, Synthesized_audio)
                    axes[row + 1, 2].set_title(f'Synthesized Waveform - Sample {i + 1}')
                    axes[row + 1, 2].set_xlabel('Time (s)')

                    axes[row + 1, 3].plot(time_orig, original_audio)
                    axes[row + 1, 3].set_title(f'Original Waveform - Sample {i + 1}')
                    axes[row + 1, 3].set_xlabel('Time (s)')

                plt.suptitle(f'Spectrogram and Waveform Comparison - Epoch {epoch}')
                plt.tight_layout()

                fig_path = os.path.join(plot_dir, f'Spec_waveform_comparison_epoch{epoch}_{timestamp}.png')
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close()

                self.writer.add_figure(f'Spectrogram_Waveform/epoch_{epoch}', fig, epoch)
                avg_mse = np.mean([np.mean((spec[j].cpu().numpy() - rec_spec[j].cpu().numpy()) ** 2)
                                   for j in range(num_samples)])
                self.writer.add_scalar('Validation/MSE', avg_mse, epoch)

    def evaluate(self, test_loader, loss_rate):

        total_loss = 0.0
        count = 0
        test_metrics = defaultdict(float)
        input_metrics = defaultdict(float)
        all_original = []
        all_reconstructed = []
        all_masked = []
        all_synth = []

        eval_dir = os.path.join(self.run_dir, f'test_{loss_rate}')
        spec_dir = os.path.join(eval_dir, 'spectrograms')
        audio_dir = os.path.join(eval_dir, 'audio_samples')
        plot_dir = os.path.join(eval_dir, 'plots')

        for directory in [eval_dir, spec_dir, audio_dir, plot_dir]:
            os.makedirs(directory, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if test_loader is not None:
            self.logger.info("Evaluating on test set...")
            if os.path.exists(os.path.join(self.checkpoint_dir, 'best_model.pt')):
                self.load_checkpoint(os.path.join(self.checkpoint_dir, 'best_model.pt'))

        self.model.eval()

        metrics_csv_path = os.path.join(eval_dir, f'sample_metrics_{timestamp}.csv')
        with open(metrics_csv_path, 'w', newline='') as csvfile:
            fieldnames = ['sample_id', 'mse', 'psnr', 'pesq', 'stoi', 'cer', 'wer']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            test_iterator = tqdm(test_loader, desc="Testing", unit="batch")
            sample_id = 0

            with torch.no_grad():
                for batch_idx, batch in enumerate(test_iterator):

                    visual_feats, spk_emb, masked_spec, spec, mask = batch

                    if self.mode == 'a':
                        masked_spec = masked_spec.float().to(self.device)
                        spec = spec.float().to(self.device)

                        rec_spec = self.model(masked_spec)
                        rec_loss = self.rec_criterion(rec_spec, spec)
                        synth_loss = 0.0

                        if self.pesq_loss:
                            pmsqe_loss = torch.mean(
                                self.pesq_criterion(torch_mel2spec(rec_spec).permute(0, 2, 1),
                                                    torch_mel2spec(spec).permute(0, 2, 1)))
                        else:
                            pmsqe_loss = 0

                        loss = (rec_loss + self.w_pmsqe * pmsqe_loss + self.w * synth_loss)

                    elif self.mode == 'v':

                        masked_spec = masked_spec.float().to(self.device)
                        visual_feats = visual_feats.float().to(self.device)
                        spk_emb = spk_emb.float().to(self.device)
                        spec = spec.float().to(self.device)

                        rec_spec = self.model(visual_feats, spk_emb)
                        synth_loss = self.rec_criterion(rec_spec, spec)

                        if self.l2s_loss:
                            sc_loss = self.sc_criterion(rec_spec, spec)
                        else:
                            sc_loss = 0.0

                        if self.pesq_loss:
                            pmsqe_loss = torch.mean(
                                self.pesq_criterion(torch_mel2spec(rec_spec).permute(0, 2, 1),
                                                    torch_mel2spec(spec).permute(0, 2, 1)))
                        else:
                            pmsqe_loss = 0

                        loss = (sc_loss + synth_loss + self.w_pmsqe * pmsqe_loss)

                    elif self.mode == 'av':

                        visual_feats = visual_feats.float().to(self.device)
                        spk_emb = spk_emb.float().to(self.device)
                        masked_spec = masked_spec.float().to(self.device)
                        spec = spec.float().to(self.device)

                        rec_spec, synth_spec = self.model(masked_spec, visual_feats, spk_emb)
                        rec_loss = self.rec_criterion(rec_spec, spec)
                        synth_loss = self.rec_criterion(synth_spec, spec)

                        if self.l2s_loss:
                            sc_loss = self.sc_criterion(synth_spec, spec)
                        else:
                            sc_loss = 0.0

                        if self.pesq_loss:
                            pmsqe_loss = torch.mean(
                                self.pesq_criterion(torch_mel2spec(rec_spec).permute(0, 2, 1),
                                                    torch_mel2spec(spec).permute(0, 2, 1)))
                        else:
                            pmsqe_loss = 0

                        loss = (rec_loss + self.w_pmsqe * pmsqe_loss + self.w * (synth_loss + sc_loss))

                    total_loss += loss.item()

                    #if batch_idx == 0 or batch_idx == 100 or batch_idx == 200:
                    if batch_idx == 100:
                        for i in range(len(spec)):
                            sample_id += 1
                            if self.vocoder is None:
                                original_audio = torch_mel_to_audio(spec[i].cpu(), None)
                                masked_audio = torch_mel_to_audio(masked_spec[i].cpu(), None)
                                reconstructed_audio = torch_mel_to_audio(rec_spec[i].cpu(), None)
                            else:
                                original_audio = self.vocoder.convert(spec[i])
                                masked_audio = self.vocoder.convert(masked_spec[i])
                                reconstructed_audio = self.vocoder.convert(rec_spec[i])

                            original_audio = original_audio.cpu().numpy()
                            masked_audio = masked_audio.cpu().numpy()
                            reconstructed_audio = reconstructed_audio.cpu().numpy()

                            spec_np = spec[i].cpu().numpy()
                            masked_spec_np = masked_spec[i].cpu().numpy()
                            rec_spec_np = rec_spec[i].cpu().numpy()

                            sample_metrics = calculate_metrics(spec[i],
                                                               rec_spec[i],
                                                               None,
                                                               self.vocoder)

                            write(os.path.join(audio_dir, f'original_{sample_id}_{timestamp}.wav'),
                                     self.sample_rate, original_audio)
                            write(os.path.join(audio_dir, f'masked_{sample_id}_{timestamp}.wav'),
                                     self.sample_rate, masked_audio)
                            write(os.path.join(audio_dir, f'reconstructed_{sample_id}_{timestamp}.wav'),
                                     self.sample_rate, reconstructed_audio)

                            fig, axs = plt.subplots(3, 2, figsize=(15, 12))

                            im1 = axs[0, 0].imshow(masked_spec_np, aspect='auto', origin='lower')
                            axs[0, 0].set_title('Masked Spectrogram')
                            plt.colorbar(im1, ax=axs[0, 0])

                            im2 = axs[1, 0].imshow(rec_spec_np, aspect='auto', origin='lower')
                            axs[1, 0].set_title('Reconstructed Spectrogram')
                            plt.colorbar(im2, ax=axs[1, 0])

                            im3 = axs[2, 0].imshow(spec_np, aspect='auto', origin='lower')
                            axs[2, 0].set_title('Original Spectrogram')
                            plt.colorbar(im3, ax=axs[2, 0])

                            time_masked = np.arange(len(masked_audio)) / self.sample_rate
                            time_recon = np.arange(len(reconstructed_audio)) / self.sample_rate
                            time_orig = np.arange(len(original_audio)) / self.sample_rate

                            axs[0, 1].plot(time_masked, masked_audio)
                            axs[0, 1].set_title('Masked Audio Waveform')
                            axs[0, 1].set_xlabel('Time (s)')

                            axs[1, 1].plot(time_recon, reconstructed_audio)
                            axs[1, 1].set_title('Reconstructed Audio Waveform')
                            axs[1, 1].set_xlabel('Time (s)')

                            axs[2, 1].plot(time_orig, original_audio)
                            axs[2, 1].set_title('Original Audio Waveform')
                            axs[2, 1].set_xlabel('Time (s)')

                            plt.tight_layout()
                            plt.savefig(os.path.join(plot_dir, f'comparison_{sample_id}_{timestamp}.png'),
                                        dpi=300)
                            plt.close(fig)

                            sample_metrics = {
                                'sample_id': sample_id,
                                'mse': sample_metrics['mse'],
                                'psnr': sample_metrics['psnr'],
                                'pesq': sample_metrics['pesq'],
                                'stoi': sample_metrics['stoi'],
                            }
                            writer.writerow(sample_metrics)

                    # batch_original = spec.cpu().numpy()
                    # batch_masked = masked_spec.cpu().numpy()
                    # batch_reconstructed = rec_spec.cpu().numpy()
                    all_original.append(spec)
                    all_masked.append(masked_spec)
                    all_reconstructed.append(rec_spec)

                    if self.mode == 'av':
                        #batch_synth = synth_spec.cpu().numpy()
                        all_synth.append(synth_spec)

                    if (batch_idx + 1) % 40 == 0 or (batch_idx + 1) == len(test_loader):

                        if self.mode == 'av':
                            all_synth = torch.concatenate(all_synth, dim=0)

                        all_original = torch.concatenate(all_original, dim=0)
                        all_reconstructed = torch.concatenate(all_reconstructed, dim=0)
                        all_masked = torch.concatenate(all_masked, dim=0)

                        rec_metrics = calculate_batch_metrics(
                            all_original,
                            all_reconstructed,
                            None,
                            None,#all_text if self.l2s_loss else None,
                            None, #all_pred_text if self.l2s_loss else None,
                            self.vocoder,
                            None, #self.tokenizer if self.asr_loss else None,
                            max_samples=len(spec)
                        )
                        for k, v in rec_metrics.items():
                            test_metrics[k] += v

                        in_metrics = calculate_batch_metrics(
                            all_original, all_masked, None,
                            None,None,
                            self.vocoder, None,
                            max_samples=len(spec)
                        )
                        for k, v in in_metrics.items():
                            input_metrics[k] += v
                        count += 1

                        # release CPU memory
                        del all_original, all_reconstructed, all_masked
                        if self.l2s_loss:
                            del all_synth

                        all_original = []
                        all_reconstructed = []
                        all_masked = []
                        all_synth = []

                        torch.cuda.empty_cache()

                    test_iterator.set_postfix({"loss": f"{loss.item():.4f}"})


        for k in test_metrics:
            test_metrics[k] /= count

        for k in input_metrics:
            input_metrics[k] /= count

        avg_test_loss = total_loss / len(test_loader)

        input_test_metrics_str = ""
        if input_metrics:
            input_test_metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in input_metrics.items()])
        self.logger.info(f"Evaluation completed-"
                         f" Input: {input_test_metrics_str}" if input_metrics else "")

        for metric_name, value in test_metrics.items():
            self.metrics['test'][metric_name] = value
        self.metrics['test']['loss'] = avg_test_loss

        test_metrics_str = ""
        if test_metrics:
            test_metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in test_metrics.items()])
        self.logger.info(f"Evaluation completed-"
                         f" Test: {test_metrics_str}" if test_metrics else "")

        metrics_json_path = os.path.join(eval_dir, f'test_metrics_{timestamp}.json')
        with open(metrics_json_path, 'w') as f:
            json.dump(self.metrics['test'], f, indent=4,
                      default=lambda x: float(x)
                      if isinstance(x, (np.float32, np.float64)) else x)

        self.logger.info(f"Test Loss: {avg_test_loss:.4f}")
        self.logger.info(f"Saved evaluation results to {eval_dir}")

        return avg_test_loss