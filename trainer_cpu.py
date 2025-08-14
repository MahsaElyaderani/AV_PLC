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
import matplotlib

matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor

import gc
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
from metrics import calculate_batch_metrics
from losses import MSELoss, L1Loss, SpectralConvergenceLoss


def setup_logging(model_name, log_dir='logs'):
    run_dir = os.path.join(log_dir, f"{model_name}")
    os.makedirs(run_dir, exist_ok=True)
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh = logging.FileHandler(os.path.join(run_dir, 'training.log'))
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


class Trainer:
    """
      'a'  : audio-only PLC (masked_spec -> rec_spec)
      'v'  : video-only synthesis (visual_feats, spk_emb -> rec_spec)
      'av' : audio-visual (masked_spec, visual_feats, spk_emb -> rec_spec[, synth_spec])
      'motion': treated like 'a' for criterion choice
    """

    def __init__(
            self,
            model,
            model_name,
            mode,
            l2s_loss: bool,
            pesq_loss: bool,
            train_loader,
            val_loader=None,
            sample_rate=16000,
            learning_rate=1e-3,
            device='cuda',
            checkpoint_dir='checkpoints',
            log_dir='logs',
            vocoder_path=None,
            use_bf16: bool = True,
            grad_clip: float = 1.0,
            cosine_Tmax: int = 100,
            weight_decay: float = 1e-2,
            betas=(0.9, 0.98),
            # New CPU optimization parameters
            cpu_workers: int = 4,
            memory_cleanup_interval: int = 50,
            enable_plot_generation: bool = True,
            cache_size: int = 100
    ):
        self.model = model
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.mode = mode
        self.l2s_loss = bool(l2s_loss)
        self.pesq_loss = bool(pesq_loss)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        self.device = torch.device(device if device == 'cpu' or torch.cuda.is_available() else 'cpu')
        self.checkpoint_root = checkpoint_dir
        self.log_root = log_dir
        self.vocoder_path = vocoder_path

        # CPU optimization settings
        self.cpu_workers = cpu_workers
        self.memory_cleanup_interval = memory_cleanup_interval
        self.enable_plot_generation = enable_plot_generation
        self.cache_size = cache_size

        # Thread pool for CPU operations
        self.thread_pool = ThreadPoolExecutor(max_workers=self.cpu_workers)

        # Memory management
        self.tensor_cache = {}
        self.loss_history = deque(maxlen=1000)  # More efficient than list

        # runtime knobs
        self.grad_clip = grad_clip
        self.mixed_precision = (self.device.type == 'cuda')
        self.amp_dtype = torch.bfloat16 if (
                    use_bf16 and torch.cuda.is_available() and torch.cuda.get_device_capability()[
                0] >= 8) else torch.float16

        # dirs + logging
        self.run_dir = os.path.join(self.log_root, f"{model_name}")
        self.ckpt_dir = os.path.join(self.checkpoint_root, f"{model_name}")
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.logger = setup_logging(model_name, self.run_dir)
        self.writer = SummaryWriter(log_dir=self.run_dir)

        # loss weights - use constants to avoid repeated calculations
        self.w_pmsqe = 0.01 if self.pesq_loss else 0.0
        self.w_synth = 1.0

        # init components/opt/sched
        self._initialize_components(weight_decay, betas, cosine_Tmax)

        # trackers
        self.best_val_loss = float('inf')
        self.global_step = 0
        self.train_step = 0

        # housekeeping cadence
        self.log_interval = 100

        # CPU performance hints
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            try:
                torch.set_float32_matmul_precision('high')  # PyTorch 2.0+
            except Exception:
                pass
        else:
            # CPU-specific optimizations
            torch.set_num_threads(self.cpu_workers)
            torch.set_num_interop_threads(1)

    def _initialize_components(self, weight_decay, betas, cosine_Tmax):
        # criteria - initialize once and reuse
        self.rec_criterion = L1Loss().to(self.device)

        if self.l2s_loss:
            self.sc_criterion = SpectralConvergenceLoss().to(self.device)
        else:
            self.sc_criterion = None

        if self.pesq_loss:
            self.pmsqe = SingleSrcPMSQE().to(self.device)
        else:
            self.pmsqe = None

        # optional vocoder
        self.vocoder = Vocoder(self.vocoder_path) if self.vocoder_path is not None else None

        # optimizer + scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=weight_decay,
            betas=betas
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cosine_Tmax)

        self.model.to(self.device)
        self.scaler = GradScaler(enabled=self.mixed_precision)

    # ----------------------- core steps -----------------------

    def _move_batch_to_device(self, batch):
        """Optimized batch movement with minimal tensor operations."""
        visual_feats, spk_emb, masked_spec, spec, mask = batch

        # Use in-place operations where possible and batch the transfers
        device_kwargs = {'device': self.device, 'non_blocking': self.device.type == 'cuda'}

        if visual_feats is not None:
            visual_feats = visual_feats.float().to(**device_kwargs)
        if spk_emb is not None:
            spk_emb = spk_emb.float().to(**device_kwargs)
        if masked_spec is not None:
            masked_spec = masked_spec.float().to(**device_kwargs)
        if spec is not None:
            spec = spec.float().to(**device_kwargs)

        return visual_feats, spk_emb, masked_spec, spec, mask

    def _forward(self, visual_feats, spk_emb, masked_spec):
        """Unified forward by mode with tensor reuse."""
        if self.mode == 'a':
            rec_spec = self.model(masked_spec)
            return rec_spec, None
        elif self.mode == 'v':
            rec_spec = self.model(visual_feats, spk_emb)
            return rec_spec, None
        elif self.mode == 'av':
            out = self.model(masked_spec, visual_feats, spk_emb)
            if isinstance(out, (tuple, list)) and len(out) == 2:
                return out[0], out[1]
            return out, None
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _compute_loss(self, rec_spec, synth_spec, spec):
        """
        Optimized loss computation with reduced tensor operations.
        """
        loss_parts = {}
        total_loss = None

        # Use in-place operations where safe
        if rec_spec is not None:
            rec_loss = self.rec_criterion(rec_spec, spec)
            loss_parts['rec_loss'] = rec_loss.item()  # Convert to scalar immediately
            total_loss = rec_loss

        if synth_spec is not None:
            synth_loss = self.rec_criterion(synth_spec, spec)
            loss_parts['synth_loss'] = synth_loss.item()

            if self.l2s_loss and self.sc_criterion is not None:
                sc_loss = self.sc_criterion(synth_spec, spec)
                loss_parts['sc_loss'] = sc_loss.item()
                weighted_loss = self.w_synth * (synth_loss + sc_loss)
            else:
                weighted_loss = self.w_synth * synth_loss
                loss_parts['sc_loss'] = None

            total_loss = total_loss + weighted_loss if total_loss is not None else weighted_loss
        else:
            loss_parts['synth_loss'] = None
            loss_parts['sc_loss'] = None

        # PMSQE computation with memory optimization
        if self.pesq_loss and self.pmsqe is not None and rec_spec is not None:
            # Reuse tensor operations
            with torch.no_grad():
                ref = torch_mel2spec(spec).permute(0, 2, 1).contiguous()
                est = torch_mel2spec(rec_spec).permute(0, 2, 1).contiguous()

            pmsqe_loss = torch.mean(self.pmsqe(est, ref))
            loss_parts['pmsqe_loss'] = pmsqe_loss.item()
            total_loss = total_loss + self.w_pmsqe * pmsqe_loss if total_loss is not None else self.w_pmsqe * pmsqe_loss
        else:
            loss_parts['pmsqe_loss'] = None

        loss_parts['loss'] = total_loss.item() if total_loss is not None else 0.0
        return total_loss, loss_parts

    def _cleanup_memory(self):
        """Aggressive memory cleanup."""
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        # Clear tensor cache periodically
        if len(self.tensor_cache) > self.cache_size:
            self.tensor_cache.clear()

    # ----------------------- epochs -----------------------

    def _train_epoch(self, epoch, num_epochs):
        self.model.train()

        # Use efficient accumulation
        loss_accumulator = 0.0
        batch_count = 0

        iterator = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]",
                        leave=False, unit="batch", mininterval=1.0)

        for bidx, batch in enumerate(iterator):
            # Memory cleanup at intervals
            if bidx % self.memory_cleanup_interval == 0:
                self._cleanup_memory()

            visual_feats, spk_emb, masked_spec, spec, mask = self._move_batch_to_device(batch)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.mixed_precision):
                rec_spec, synth_spec = self._forward(visual_feats, spk_emb, masked_spec)
                loss, parts = self._compute_loss(rec_spec, synth_spec, spec)

            self.scaler.scale(loss).backward()

            if self.grad_clip is not None and self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Efficient loss tracking
            loss_val = loss.item()
            loss_accumulator += loss_val
            batch_count += 1
            self.train_step += 1
            self.global_step += 1

            # Store in deque for efficient history
            self.loss_history.append(loss_val)

            # Less frequent logging updates
            if bidx % self.log_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                postfix = {
                    "loss": f"{loss_val:.4f}",
                    "lr": f"{lr:.9f}",
                    "avg": f"{loss_accumulator / batch_count:.4f}"
                }

                # Add component losses efficiently
                for key in ['rec_loss', 'synth_loss', 'sc_loss', 'pmsqe_loss']:
                    val = parts.get(key)
                    if val is not None:
                        postfix[key[:4]] = f"{val:.5f}"

                iterator.set_postfix(postfix)

            # Sparse tensorboard logging
            if self.train_step % 5000 == 0:
                self.writer.add_scalar('Train/loss', loss_val, self.global_step)
                for k, v in parts.items():
                    if k != 'loss' and v is not None:
                        self.writer.add_scalar(f'Train/{k}', v, self.global_step)
                self.writer.add_scalar('Train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)

        return loss_accumulator / max(1, batch_count)

    @torch.inference_mode()
    def _validate_epoch(self, epoch, num_epochs):
        if self.val_loader is None:
            return None

        self.model.eval()
        iterator = tqdm(self.val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Valid]",
                        leave=False, unit="batch", mininterval=1.0)

        # Efficient accumulators
        loss_accumulator = 0.0
        batch_count = 0
        acc_parts = defaultdict(float)
        acc_counts = defaultdict(int)

        # Limited metrics collection
        target_metric_samples = 8  # Reduced for efficiency
        specs_list, recs_list = [], []

        for bidx, batch in enumerate(iterator):
            if bidx % (self.memory_cleanup_interval // 2) == 0:
                self._cleanup_memory()

            visual_feats, spk_emb, masked_spec, spec, _ = self._move_batch_to_device(batch)
            rec_spec, synth_spec = self._forward(visual_feats, spk_emb, masked_spec)
            loss, parts = self._compute_loss(rec_spec, synth_spec, spec)

            loss_val = loss.item()
            loss_accumulator += loss_val
            batch_count += 1

            # Efficient component tracking
            for k, v in parts.items():
                if v is not None and k != 'loss':
                    acc_parts[k] += v
                    acc_counts[k] += 1

            # Efficient sample collection
            if len(specs_list) < target_metric_samples:
                remaining = target_metric_samples - len(specs_list)
                take = min(remaining, spec.shape[0])
                specs_list.append(spec[:take].detach().cpu())
                recs_list.append(rec_spec[:take].detach().cpu())

            # Less frequent progress updates
            if bidx % 50 == 0:
                iterator.set_postfix({"loss": f"{loss_val:.4f}", "avg": f"{loss_accumulator / batch_count:.4f}"})

        # Compute metrics efficiently
        metrics = {'mse': 0.0, 'psnr': 0.0, 'pesq': 0.0, 'stoi': 0.0}

        if specs_list and recs_list:
            specs_cat = torch.cat(specs_list, dim=0)
            recs_cat = torch.cat(recs_list, dim=0)
            metrics = calculate_batch_metrics(
                specs_cat, recs_cat, None, None, None,
                hifigan_vocoder=self.vocoder, tokenizer=None,
                max_samples=specs_cat.shape[0], sample_rate=self.sample_rate
            )

        metrics['loss'] = loss_accumulator / max(1, batch_count)

        # Compute averages efficiently
        parts_avg = {'loss': metrics['loss']}
        for k, total in acc_parts.items():
            count = acc_counts[k]
            parts_avg[k] = total / count if count > 0 else None

        self._log_loss_components(parts_avg, metrics, phase="validation", epoch=epoch + 1)
        return metrics

    # ----------------------- logging + plots -----------------------

    def _log_loss_components(self, loss_components, metrics_dict, phase="train", epoch=None):
        # Efficient string building
        parts = []
        if phase == "train":
            header = f"Epoch {epoch} TRAINING" if epoch is not None else "TRAINING"
        elif phase == "validation":
            header = f"Epoch {epoch} VALIDATION"
        else:
            header = "TEST EVALUATION"

        if 'loss' in loss_components:
            parts.append(f"Total Loss: {loss_components['loss']:.6f}")

        # Build component string efficiently
        comp_parts = []
        for k in ['rec_loss', 'synth_loss', 'sc_loss', 'pmsqe_loss']:
            v = loss_components.get(k)
            if v is not None:
                comp_parts.append(f"{k.replace('_', ' ').title()}: {float(v):.6f}")
        if comp_parts:
            parts.append("Components [" + ", ".join(comp_parts) + "]")

        if metrics_dict and phase in ["validation", "test"]:
            metric_parts = []
            for k, v in metrics_dict.items():
                if k != 'loss' and isinstance(v, (float, int)):
                    metric_parts.append(f"{k.upper()}: {v:.4f}")
            if metric_parts:
                parts.append("Metrics [" + ", ".join(metric_parts) + "]")

        if phase == "train":
            lr = self.optimizer.param_groups[0]['lr']
            parts.append(f"LR: {lr:.9f}")

        self.logger.info(" | ".join([header] + parts))

        # Efficient tensorboard logging
        step = self.global_step if phase == "train" else (epoch if epoch is not None else 0)
        for k, v in loss_components.items():
            if v is not None:
                self.writer.add_scalar(f'{phase.capitalize()}/{k}', float(v), step)

        if metrics_dict and phase in ["validation", "test"]:
            for k, v in metrics_dict.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f'{phase.capitalize()}/{k}', float(v), step)

        if phase == "train":
            self.writer.add_scalar(f'{phase.capitalize()}/lr', self.optimizer.param_groups[0]['lr'], step)

    @torch.no_grad()
    def _log_results(self, epoch, num_samples=2):
        if not self.enable_plot_generation or self.val_loader is None or num_samples <= 0:
            return

        self.model.eval()

        plot_dir = os.path.join(self.run_dir, 'validation', 'plots')
        os.makedirs(plot_dir, exist_ok=True)

        try:
            batch = next(iter(self.val_loader))
        except StopIteration:
            return

        visual_feats, spk_emb, masked_spec, spec, mask = self._move_batch_to_device(batch)

        # Limit samples for efficiency
        num_samples = min(num_samples, spec.shape[0])
        spec = spec[:num_samples]
        masked_spec = masked_spec[:num_samples]
        if visual_feats is not None:
            visual_feats = visual_feats[:num_samples]
        if spk_emb is not None:
            spk_emb = spk_emb[:num_samples]

        rec_spec, synth_spec = self._forward(visual_feats, spk_emb, masked_spec)
        if rec_spec is not None:
            rec_spec = rec_spec[:num_samples]
        if synth_spec is not None:
            synth_spec = synth_spec[:num_samples]

        # Use thread pool for plot generation
        self.thread_pool.submit(
            self._create_plots_async, spec, masked_spec, rec_spec, synth_spec, epoch, plot_dir
        )

    def _create_plots_async(self, spec, masked_spec, rec_spec, synth_spec, epoch, plot_dir):
        """Create plots in background thread to avoid blocking main training."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            n = spec.size(0)
            cols = 4 if synth_spec is not None else 3

            # Use tight layout and smaller figure size for efficiency
            fig, axes = plt.subplots(n * 2, cols, figsize=(4 * cols, 4 * n), dpi=100)

            for i in range(n):
                # Move to CPU once and reuse
                spec_np = spec[i].cpu().numpy()
                masked_np = masked_spec[i].cpu().numpy()
                rec_np = rec_spec[i].cpu().numpy() if rec_spec is not None else None

                # Compute audio efficiently
                if self.vocoder is None:
                    orig_audio = torch_mel_to_audio(spec[i].cpu(), None).numpy()
                    masked_audio = torch_mel_to_audio(masked_spec[i].cpu(), None).numpy()
                    rec_audio = torch_mel_to_audio(rec_spec[i].cpu(), None).numpy() if rec_spec is not None else None
                    synth_audio = torch_mel_to_audio(synth_spec[i].cpu(),
                                                     None).numpy() if synth_spec is not None else None
                else:
                    orig_audio = self.vocoder.convert(spec[i]).cpu().numpy()
                    masked_audio = self.vocoder.convert(masked_spec[i]).cpu().numpy()
                    rec_audio = self.vocoder.convert(rec_spec[i]).cpu().numpy() if rec_spec is not None else None
                    synth_audio = self.vocoder.convert(synth_spec[i]).cpu().numpy() if synth_spec is not None else None

                row = i * 2
                ax_spec_row = axes[row] if n > 1 else [axes[0], axes[1], axes[2]]
                ax_wave_row = axes[row + 1] if n > 1 else [axes[3], axes[4], axes[5]] if cols > 3 else [axes[3],
                                                                                                        axes[4]]

                self._plot_spectrogram_row(ax_spec_row, masked_np, rec_np, spec_np, synth_spec, i, fig)
                self._plot_waveform_row(ax_wave_row, masked_audio, rec_audio, orig_audio, synth_audio, i)

            plt.suptitle(f'Comparison - Epoch {epoch}')
            plt.tight_layout()

            fig_path = os.path.join(plot_dir, f'comparison_epoch{epoch}_{timestamp}.png')
            plt.savefig(fig_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            print(f"Error creating plots: {e}")
        finally:
            gc.collect()

    def _plot_spectrogram_row(self, axes, masked_spec, rec_spec, orig_spec, synth_spec, sample_idx, fig):
        col = 0
        im1 = axes[col].imshow(masked_spec, aspect='auto', origin='lower', interpolation='nearest')
        axes[col].set_title(f'Input - #{sample_idx + 1}')
        fig.colorbar(im1, ax=axes[col], format='%+2.0f');
        col += 1

        if rec_spec is not None:
            im2 = axes[col].imshow(rec_spec, aspect='auto', origin='lower', interpolation='nearest')
            axes[col].set_title(f'Reconstructed - #{sample_idx + 1}')
            fig.colorbar(im2, ax=axes[col], format='%+2.0f');
            col += 1

        if synth_spec is not None:
            synth_np = synth_spec[sample_idx].detach().cpu().numpy()
            im3 = axes[col].imshow(synth_np, aspect='auto', origin='lower', interpolation='nearest')
            axes[col].set_title(f'Synthesized - #{sample_idx + 1}')
            fig.colorbar(im3, ax=axes[col], format='%+2.0f');
            col += 1

        im4 = axes[col].imshow(orig_spec, aspect='auto', origin='lower', interpolation='nearest')
        axes[col].set_title(f'Original - #{sample_idx + 1}')
        fig.colorbar(im4, ax=axes[col], format='%+2.0f')

        if rec_spec is not None:
            mse = float(np.mean((orig_spec - rec_spec) ** 2))
            axes[0].set_ylabel(f'MSE: {mse:.4f}')

    def _plot_waveform_row(self, axes, masked_audio, rec_audio, orig_audio, synth_audio, sample_idx):
        col = 0
        t_masked = np.linspace(0, len(masked_audio) / self.sample_rate, len(masked_audio))
        axes[col].plot(t_masked, masked_audio)
        axes[col].set_title(f'Input Waveform - #{sample_idx + 1}')
        axes[col].set_xlabel('Time (s)');
        col += 1

        if rec_audio is not None:
            t_rec = np.linspace(0, len(rec_audio) / self.sample_rate, len(rec_audio))
            axes[col].plot(t_rec, rec_audio)
            axes[col].set_title(f'Reconstructed Waveform - #{sample_idx + 1}')
            axes[col].set_xlabel('Time (s)');
            col += 1

        if synth_audio is not None:
            t_s = np.linspace(0, len(synth_audio) / self.sample_rate, len(synth_audio))
            axes[col].plot(t_s, synth_audio)
            axes[col].set_title(f'Synthesized Waveform - #{sample_idx + 1}')
            axes[col].set_xlabel('Time (s)');
            col += 1

        t_o = np.linspace(0, len(orig_audio) / self.sample_rate, len(orig_audio))
        axes[col].plot(t_o, orig_audio)
        axes[col].set_title(f'Original Waveform - #{sample_idx + 1}')
        axes[col].set_xlabel('Time (s)')

    # ----------------------- checkpoints -----------------------

    def _save_checkpoint(self, epoch, is_best=False):
        """Optimized checkpoint saving."""
        payload = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step,
            'train_step': self.train_step,
        }

        ckpt_dir = self.ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)

        if is_best:
            path = os.path.join(ckpt_dir, 'best_model.pt')
        else:
            path = os.path.join(ckpt_dir, f'checkpoint_epoch_{epoch}.pt')

        # Use thread pool for I/O
        self.thread_pool.submit(torch.save, payload, path)
        self.logger.info(f"Queued checkpoint save to {path}")

    def load_checkpoint(self, checkpoint_path=None, load_best=False):
        """Optimized checkpoint loading."""
        ckpt_dir = self.ckpt_dir

        if checkpoint_path is None:
            if load_best and os.path.exists(os.path.join(ckpt_dir, 'best_model.pt')):
                checkpoint_path = os.path.join(ckpt_dir, 'best_model.pt')
                self.logger.info(f"Loading best model checkpoint from {checkpoint_path}")
            else:
                checkpoints = glob.glob(os.path.join(ckpt_dir, 'checkpoint_epoch_*.pt'))
                if not checkpoints:
                    self.logger.info("No checkpoints found. Starting from scratch.")
                    return 0

                # More efficient sorting
                latest_epoch = max(int(os.path.basename(cp).split('_')[-1].split('.')[0])
                                   for cp in checkpoints)
                checkpoint_path = os.path.join(ckpt_dir, f'checkpoint_epoch_{latest_epoch}.pt')
                self.logger.info(f"Loading latest checkpoint from {checkpoint_path} (epoch {latest_epoch})")

        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Checkpoint {checkpoint_path} not found. Starting from scratch.")
            return 0

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Efficient state loading
            self.model.load_state_dict(checkpoint['model_state_dict'])

            if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception:
                    self.logger.warning("Scheduler state couldn't be restored; continuing with current scheduler.")

            # Restore state efficiently
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.global_step = checkpoint.get('global_step', 0)
            self.train_step = checkpoint.get('train_step', 0)

            epoch = int(checkpoint.get('epoch', 0))
            self.logger.info(f"Successfully loaded checkpoint from epoch {epoch}")
            self.logger.info(f"Best validation loss so far: {self.best_val_loss:.4f}")
            return epoch

        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {str(e)}")
            self.logger.warning("Starting from scratch.")
            return 0

    # ----------------------- evaluation -----------------------

    @torch.no_grad()
    def _save_sample_audio_async(self, specs, recs, audio_dir, start_id, timestamp):
        """Save audio samples asynchronously."""

        def save_audio():
            for i, (spec, rec) in enumerate(zip(specs, recs)):
                sid = start_id + i
                if self.vocoder is None:
                    orig_audio = torch_mel_to_audio(spec, None).numpy()
                    rec_audio = torch_mel_to_audio(rec, None).numpy()
                else:
                    orig_audio = self.vocoder.convert(spec).cpu().numpy()
                    rec_audio = self.vocoder.convert(rec).cpu().numpy()

                write(os.path.join(audio_dir, f'original_{sid}_{timestamp}.wav'),
                      self.sample_rate, orig_audio)
                write(os.path.join(audio_dir, f'reconstructed_{sid}_{timestamp}.wav'),
                      self.sample_rate, rec_audio)

        self.thread_pool.submit(save_audio)

    @torch.no_grad()
    def evaluate(self, test_loader, loss_rate=""):
        """Optimized evaluation with streaming metrics computation."""
        loss_accumulator = 0.0
        batch_count = 0
        acc_parts = defaultdict(float)
        acc_counts = defaultdict(int)

        eval_dir = os.path.join(self.run_dir, f'test_{loss_rate}')
        audio_dir = os.path.join(eval_dir, 'audio_samples')
        os.makedirs(eval_dir, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)

        self.logger.info("Evaluating on test set...")
        best_path = os.path.join(self.ckpt_dir, 'best_model.pt')
        if os.path.exists(best_path):
            self.load_checkpoint(best_path)

        self.model.eval()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Streaming metrics computation
        metrics_computer = StreamingMetricsComputer(
            self.vocoder, self.sample_rate, max_samples=200
        )

        saved_audio_count = 0
        max_audio_saves = 5

        iterator = tqdm(test_loader, desc="Testing", unit="batch", mininterval=1.0)

        for bidx, batch in enumerate(iterator):
            if bidx % self.memory_cleanup_interval == 0:
                self._cleanup_memory()

            visual_feats, spk_emb, masked_spec, spec, mask = self._move_batch_to_device(batch)
            rec_spec, synth_spec = self._forward(visual_feats, spk_emb, masked_spec)
            loss, parts = self._compute_loss(rec_spec, synth_spec, spec)

            loss_val = loss.item()
            loss_accumulator += loss_val
            batch_count += 1

            # Efficient component tracking
            for k, v in parts.items():
                if v is not None and k != 'loss':
                    acc_parts[k] += v
                    acc_counts[k] += 1

            # Add to streaming metrics
            metrics_computer.add_batch(spec.cpu(), rec_spec.cpu())

            # Save limited audio samples
            if saved_audio_count < max_audio_saves:
                save_count = min(max_audio_saves - saved_audio_count, spec.shape[0])
                self._save_sample_audio_async(
                    spec[:save_count].cpu(), rec_spec[:save_count].cpu(),
                    audio_dir, saved_audio_count, timestamp
                )
                saved_audio_count += save_count

            if bidx % 20 == 0:
                iterator.set_postfix({"loss": f"{loss_val:.4f}", "avg": f"{loss_accumulator / batch_count:.4f}"})

        # Compute final metrics
        final_metrics = metrics_computer.compute_final_metrics()
        final_metrics['loss'] = loss_accumulator / max(1, batch_count)

        # Component averages
        parts_avg = {'loss': final_metrics['loss']}
        for k, total in acc_parts.items():
            count = acc_counts[k]
            parts_avg[k] = total / count if count > 0 else None

        self._log_loss_components(parts_avg, final_metrics, phase="test")

        # Save results asynchronously
        def save_results():
            out_json = os.path.join(eval_dir, f'test_metrics_{timestamp}.json')
            with open(out_json, 'w') as f:
                json.dump({**parts_avg, **final_metrics}, f, indent=4)

        self.thread_pool.submit(save_results)
        self.logger.info(f"Queued evaluation results save")

        return final_metrics['loss']

    # ----------------------- training loop -----------------------

    def train(self, num_epochs=100, save_interval=10, samples_to_log=2, start_epoch=0):
        self.logger.info(f"Starting training from epoch {start_epoch} for {num_epochs} epochs")
        self.logger.info(f"Training on device: {self.device}")
        self.logger.info(f"CPU workers: {self.cpu_workers}")

        total_start = time.time()
        epoch_iter = trange(start_epoch, num_epochs, desc="Training", unit="epoch")

        # Pre-allocate for efficiency
        best_metrics = None
        last_cleanup = 0

        for epoch in epoch_iter:
            epoch_start = time.time()

            # Early stopping check
            current_lr = self.optimizer.param_groups[0]['lr']
            if current_lr < 1e-7:
                self.logger.info(f"Stopping: LR ({current_lr:.9f}) below threshold.")
                break

            # Periodic deep cleanup
            if epoch - last_cleanup >= 10:
                self._cleanup_memory()
                last_cleanup = epoch

            train_loss = self._train_epoch(epoch, num_epochs)
            self.scheduler.step()

            val_metrics = None
            if self.val_loader is not None:
                val_metrics = self._validate_epoch(epoch, num_epochs)

                if val_metrics is not None:
                    val_loss = val_metrics['loss']

                    # Sparse tensorboard logging
                    if (epoch + 1) % 5 == 0:  # Less frequent than original
                        for k, v in val_metrics.items():
                            self.writer.add_scalar(f'Validation/{k}', float(v), self.global_step)

                    # Even sparser plot generation
                    if (epoch + 1) % 20 == 0 or epoch == start_epoch:
                        self._log_results(epoch + 1, samples_to_log)

                    # Best model tracking
                    if val_loss <= self.best_val_loss:
                        self.best_val_loss = val_loss
                        best_metrics = val_metrics.copy()
                        self._save_checkpoint(epoch, is_best=True)

            # Less frequent checkpoint saving
            if (epoch + 1) % save_interval == 0:
                self._save_checkpoint(epoch, is_best=False)

            # Efficient epoch logging
            elapsed = time.time() - epoch_start
            log_parts = [
                f"Epoch {epoch + 1}/{num_epochs} in {elapsed:.2f}s",
                f"Train Loss: {train_loss:.4f}",
                f"LR: {current_lr:.9f}"
            ]

            if val_metrics:
                val_str = f"Val Loss: {val_metrics['loss']:.4f}"
                if best_metrics and val_metrics['loss'] <= self.best_val_loss:
                    val_str += " (BEST)"
                log_parts.append(val_str)

            self.logger.info(" - ".join(log_parts))

            # Efficient postfix
            postfix_parts = [f"Train: {train_loss:.4f}"]
            if val_metrics:
                postfix_parts.append(f"Val: {val_metrics['loss']:.4f}")
            postfix_parts.append(f"LR: {current_lr:.9f}")
            epoch_iter.set_postfix_str(", ".join(postfix_parts))

            # Sparse tensorboard logging
            if (epoch + 1) % 5 == 0:
                self.writer.add_scalar('Train/epoch_loss', train_loss, epoch + 1)
                self.writer.add_scalar('Train/epoch_lr', current_lr, epoch + 1)

        total_time = time.time() - total_start
        self.logger.info(f"Training completed in {total_time / 3600:.2f} h")

        # Final cleanup
        self._cleanup_memory()

    def close(self):
        """Clean shutdown with resource cleanup."""
        self.writer.flush()
        self.writer.close()

        # Wait for background tasks to complete
        self.thread_pool.shutdown(wait=True)

        # Final memory cleanup
        self._cleanup_memory()


class StreamingMetricsComputer:
    """Efficient streaming computation of metrics to avoid memory buildup."""

    def __init__(self, vocoder, sample_rate, max_samples=200):
        self.vocoder = vocoder
        self.sample_rate = sample_rate
        self.max_samples = max_samples

        # Streaming accumulators
        self.mse_accumulator = 0.0
        self.sample_count = 0
        self.specs_buffer = []
        self.recs_buffer = []

    def add_batch(self, spec_batch, rec_batch):
        """Add a batch for metrics computation."""
        if self.sample_count >= self.max_samples:
            return

        # Take only what we need
        remaining = self.max_samples - self.sample_count
        take = min(remaining, spec_batch.shape[0])

        spec_slice = spec_batch[:take]
        rec_slice = rec_batch[:take]

        # Compute MSE incrementally
        mse = torch.mean((spec_slice - rec_slice) ** 2).item()
        self.mse_accumulator += mse * take
        self.sample_count += take

        # Store limited samples for other metrics
        if len(self.specs_buffer) < 50:  # Keep buffer small
            self.specs_buffer.append(spec_slice)
            self.recs_buffer.append(rec_slice)

    def compute_final_metrics(self):
        """Compute final metrics efficiently."""
        metrics = {
            'mse': self.mse_accumulator / max(1, self.sample_count),
            'psnr': 0.0,  # Placeholder - expensive to compute
            'pesq': 0.0,  # Placeholder - expensive to compute
            'stoi': 0.0  # Placeholder - expensive to compute
        }

        # Compute PSNR from MSE efficiently
        if metrics['mse'] > 0:
            metrics['psnr'] = 10 * np.log10(1.0 / metrics['mse'])

        return metrics