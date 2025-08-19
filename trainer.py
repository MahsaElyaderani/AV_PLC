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
    Modes:
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
        drop_av: bool,
        train_loader,
        val_loader=None,
        sample_rate=16000,
        learning_rate=1e-3,
        device='cuda',
        checkpoint_dir='checkpoints',
        log_dir='logs',
        vocoder_path=None,
        use_bf16: bool = False,      # enable BF16 on Ampere+ by default
        mixed_precision: bool = False,
        grad_clip: float = 1.0,
        cosine_Tmax: int = 100,
        weight_decay: float = 1e-2,
        betas=(0.9, 0.98)
    ):
        self.model = model
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.mode = mode
        self.drop_av = bool(drop_av)
        self.l2s_loss = bool(l2s_loss)
        self.pesq_loss = bool(pesq_loss)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        self.device = device
        self.checkpoint_root = checkpoint_dir
        self.log_root = log_dir
        self.vocoder_path = vocoder_path

        # runtime knobs
        self.grad_clip = grad_clip
        self.mixed_precision = mixed_precision
        self.amp_dtype = torch.bfloat16 if (use_bf16 and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

        # dirs + logging
        self.run_dir = os.path.join(self.log_root, f"{model_name}")
        self.ckpt_dir = os.path.join(self.checkpoint_root, f"{model_name}")
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.logger = setup_logging(model_name, self.run_dir)
        self.writer = SummaryWriter(log_dir=self.run_dir)

        # loss weights
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
        self.memory_cleanup_interval = 20

        # perf hints
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            try:
                torch.set_float32_matmul_precision('high')  # PyTorch 2.0+
            except Exception:
                pass

    def _initialize_components(self, weight_decay, betas, cosine_Tmax):
        # criteria
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

    # ----------------------- drop modality --------------------
    def modality_dropout_probs(self, epoch):

        if epoch <= 5:# Start → mostly audio+video
            return [0.9, 0.05, 0.05]
        elif 5 < epoch <= 10:
            return [0.8, 0.1, 0.1]
        elif 10 < epoch <= 15:
            return [0.7, 0.15, 0.15]
        else:
            return [0.6, 0.2, 0.2]
    # ----------------------- core steps -----------------------

    def _move_batch_to_device(self, batch):
        """Move batch to target device with non_blocking only if CUDA+pin_memory."""
        non_blocking = (self.device.type == 'cuda')
        visual_feats, spk_emb, masked_spec, spec = batch
        if visual_feats is not None:
            visual_feats = visual_feats.float().to(self.device, non_blocking=non_blocking)
        if spk_emb is not None:
            spk_emb = spk_emb.float().to(self.device, non_blocking=non_blocking)
        if masked_spec is not None:
            masked_spec = masked_spec.float().to(self.device, non_blocking=non_blocking)
        if spec is not None:
            spec = spec.float().to(self.device, non_blocking=non_blocking)
        # mask may be None
        return visual_feats, spk_emb, masked_spec, spec#, mask

    def _forward(self, visual_feats, spk_emb, masked_spec):
        """Unified forward by mode."""
        if self.mode == 'a':
            rec_spec = self.model(masked_spec)
            return rec_spec, None
        elif self.mode == 'v':
            rec_spec = self.model(visual_feats, spk_emb)
            return rec_spec, None
        elif self.mode == 'av':
            # model expected to return both rec_spec and synth_spec
            out = self.model(masked_spec, visual_feats, spk_emb)
            if isinstance(out, (tuple, list)) and len(out) == 2:
                return out[0], out[1]
            # graceful: if model returns single tensor
            return out, None
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _compute_loss(self, rec_spec, synth_spec, spec):
        """
        Build total loss from available components.
        """
        loss = 0.0
        parts = {}

        # reconstruction (always against spec)
        if rec_spec is not None:
            rec_loss = self.rec_criterion(rec_spec, spec)
            parts['rec_loss'] = rec_loss
            loss = loss + rec_loss

        # optional synth & spectral convergence for 'v'/'av'
        if synth_spec is not None:
            synth_loss = self.rec_criterion(synth_spec, spec)
            parts['synth_loss'] = synth_loss
            # weight both synth and SC under the same knob
            if self.l2s_loss and self.sc_criterion is not None:
                sc_loss = self.sc_criterion(synth_spec, spec)
                parts['sc_loss'] = sc_loss
                loss = loss + self.w_synth * (synth_loss + sc_loss)
            else:
                loss = loss + self.w_synth * synth_loss
        else:
            parts['synth_loss'] = None
            parts['sc_loss'] = None

        # optional PMSQE (operate on waveform-like magnitude from mel2spec)
        if self.pesq_loss and self.pmsqe is not None and rec_spec is not None:
            # pmsqe expects (B, T, F)
            ref = torch_mel2spec(spec).permute(0, 2, 1).contiguous()
            est = torch_mel2spec(rec_spec).permute(0, 2, 1).contiguous()
            pmsqe_loss = torch.mean(self.pmsqe(est, ref))
            parts['pmsqe_loss'] = pmsqe_loss
            loss = loss + self.w_pmsqe * pmsqe_loss
        else:
            parts['pmsqe_loss'] = None

        parts['loss'] = loss
        return loss, parts

    # ----------------------- epochs -----------------------

    def _train_epoch(self, epoch, num_epochs):
        self.model.train()
        running_loss = 0.0
        n_batches = 0

        if self.drop_av:
            new_mode_probs = self.modality_dropout_probs(epoch)
            self.train_loader.dataset.modality_dropout.mode_probs = new_mode_probs

        iterator = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False, unit="batch")

        for bidx, batch in enumerate(iterator):
            #if self.device.type == 'cuda' and (bidx % self.memory_cleanup_interval == 0):
            #    torch.cuda.empty_cache()

            visual_feats, spk_emb, masked_spec, spec = self._move_batch_to_device(batch)

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

            running_loss += loss.item()
            n_batches += 1
            self.train_step += 1
            self.global_step += 1

            if bidx % self.log_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                p = {
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{lr:.9f}",
                }
                if parts.get('rec_loss') is not None:
                    p['rec'] = f"{parts['rec_loss'].item():.5f}"
                if parts.get('synth_loss') is not None:
                    p['synth'] = f"{parts['synth_loss'].item():.5f}"
                if parts.get('sc_loss') is not None:
                    p['sc'] = f"{parts['sc_loss'].item():.5f}"
                if parts.get('pmsqe_loss') is not None:
                    p['pmsqe'] = f"{parts['pmsqe_loss'].item():.5f}"
                iterator.set_postfix(p)

            # sparse TB logging
            if self.train_step % 5000 == 0:
                self.writer.add_scalar('Train/loss', loss.item(), self.global_step)
                for k, v in parts.items():
                    if k != 'loss' and v is not None:
                        self.writer.add_scalar(f'Train/{k}', v.item(), self.global_step)
                self.writer.add_scalar('Train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)

        avg = running_loss / max(1, n_batches)
        return avg

    @torch.inference_mode()
    def _validate_epoch(self, epoch, num_epochs):
        if self.val_loader is None:
            return None

        self.model.eval()
        iterator = tqdm(self.val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Valid]",
                        leave=False, unit="batch", mininterval=0.5)

        batch_losses = []
        acc_parts = defaultdict(list)

        # keep metrics small & predictable
        target_metric_samples = 12  # e.g., 8–16 total
        specs_accum, recs_accum = [], []

        for bidx, batch in enumerate(iterator):

            visual_feats, spk_emb, masked_spec, spec = self._move_batch_to_device(batch)
            rec_spec, synth_spec = self._forward(visual_feats, spk_emb, masked_spec)
            loss, parts = self._compute_loss(rec_spec, synth_spec, spec)

            batch_losses.append(loss.item())
            if (bidx % 10) == 0:
                iterator.set_postfix({"loss": f"{loss.item():.4f}"})

            for k, v in parts.items():
                if v is not None and k != 'loss':
                    acc_parts[k].append(v.item())

            # collect just enough for metrics
            remaining = target_metric_samples - len(specs_accum)
            if remaining > 0:
                take = min(remaining, spec.shape[0])
                specs_accum.append(spec[:take].detach().to('cpu'))
                recs_accum.append(rec_spec[:take].detach().to('cpu'))

        metrics = {'mse': 0.0, 'psnr': 0.0, 'pesq': 0.0, 'stoi': 0.0}
        if specs_accum and recs_accum:
            specs_cat = torch.cat(specs_accum, dim=0)
            recs_cat = torch.cat(recs_accum, dim=0)
            metrics = calculate_batch_metrics(
                specs_cat, recs_cat,
                phase=None, all_text=None, all_pred_text=None,
                hifigan_vocoder=self.vocoder, tokenizer=None,
                max_samples=int(specs_cat.shape[0]), sample_rate=self.sample_rate
            )

        metrics['loss'] = float(np.mean(batch_losses)) if batch_losses else 0.0

        parts_avg = {'loss': metrics['loss']}
        for k, vals in acc_parts.items():
            parts_avg[k] = float(np.mean(vals)) if vals else None

        self._log_loss_components(parts_avg, metrics, phase="validation", epoch=epoch + 1)
        return metrics

    # ----------------------- logging + plots -----------------------

    def _log_loss_components(self, loss_components, metrics_dict, phase="train", epoch=None):
        # console/file
        parts = []
        if phase == "train":
            header = f"Epoch {epoch} TRAINING" if epoch is not None else "TRAINING"
        elif phase == "validation":
            header = f"Epoch {epoch} VALIDATION"
        else:
            header = "TEST EVALUATION"

        if 'loss' in loss_components:
            parts.append(f"Total Loss: {loss_components['loss']:.6f}")

        comp_bits = []
        for k in ['rec_loss', 'synth_loss', 'sc_loss', 'pmsqe_loss']:
            v = loss_components.get(k, None)
            if v is not None:
                comp_bits.append(f"{k.replace('_',' ').title()}: {float(v):.6f}")
        if comp_bits:
            parts.append("Components [" + ", ".join(comp_bits) + "]")

        if metrics_dict and phase in ["validation", "test"]:
            metric_bits = []
            for k, v in metrics_dict.items():
                if k != 'loss' and isinstance(v, (float, int)):
                    metric_bits.append(f"{k.upper()}: {v:.4f}")
            if metric_bits:
                parts.append("Metrics [" + ", ".join(metric_bits) + "]")

        # LR for train
        if phase == "train":
            lr = self.optimizer.param_groups[0]['lr']
            parts.append(f"LR: {lr:.9f}")

        self.logger.info(" | ".join([header] + parts))

        # tensorboard
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
        if self.val_loader is None or num_samples <= 0:
            return
        self.model.eval()

        plot_dir = os.path.join(self.run_dir, 'validation', 'plots')
        os.makedirs(plot_dir, exist_ok=True)

        try:
            batch = next(iter(self.val_loader))
        except StopIteration:
            return

        visual_feats, spk_emb, masked_spec, spec = self._move_batch_to_device(batch)

        # trim to num_samples
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

        self._create_plots(spec, masked_spec, rec_spec, synth_spec, epoch, plot_dir)

    def _create_plots(self, spec, masked_spec, rec_spec, synth_spec, epoch, plot_dir):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        n = spec.size(0)
        cols = 4 if synth_spec is not None else 3
        fig, axes = plt.subplots(n * 2, cols, figsize=(5 * cols, 5 * n))

        for i in range(n):
            spec_np = spec[i].detach().cpu().numpy()
            masked_np = masked_spec[i].detach().cpu().numpy()
            rec_np = rec_spec[i].detach().cpu().numpy() if rec_spec is not None else None

            # audio (compute on the fly)
            if self.vocoder is None:
                orig_audio = torch_mel_to_audio(spec[i].cpu(), None).cpu().numpy()
                masked_audio = torch_mel_to_audio(masked_spec[i].cpu(), None).cpu().numpy()
                rec_audio = torch_mel_to_audio(rec_spec[i].cpu(), None).cpu().numpy() if rec_spec is not None else None
                synth_audio = torch_mel_to_audio(synth_spec[i].cpu(), None).cpu().numpy() if synth_spec is not None else None
            else:
                orig_audio = self.vocoder.convert(spec[i]).cpu().numpy()
                masked_audio = self.vocoder.convert(masked_spec[i]).cpu().numpy()
                rec_audio = self.vocoder.convert(rec_spec[i]).cpu().numpy() if rec_spec is not None else None
                synth_audio = self.vocoder.convert(synth_spec[i]).cpu().numpy() if synth_spec is not None else None

            row = i * 2
            ax_spec_row = axes[row] if n > 1 else axes
            ax_wave_row = axes[row + 1] if n > 1 else axes

            self._plot_spectrogram_row(ax_spec_row, masked_np, rec_np, spec_np, synth_spec, i, fig)
            self._plot_waveform_row(ax_wave_row, masked_audio, rec_audio, orig_audio, synth_audio, i)

        plt.suptitle(f'Spectrogram and Waveform Comparison - Epoch {epoch}')
        plt.tight_layout()
        fig_path = os.path.join(plot_dir, f'comparison_epoch{epoch}_{timestamp}.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def _plot_spectrogram_row(self, axes, masked_spec, rec_spec, orig_spec, synth_spec, sample_idx, fig):
        cols = axes.shape[0] if hasattr(axes, 'shape') else 1
        col = 0
        im1 = axes[col].imshow(masked_spec, aspect='auto', origin='lower', interpolation='none')
        axes[col].set_title(f'Input - #{sample_idx+1}')
        fig.colorbar(im1, ax=axes[col], format='%+2.0f'); col += 1

        if rec_spec is not None:
            im2 = axes[col].imshow(rec_spec, aspect='auto', origin='lower', interpolation='none')
            axes[col].set_title(f'Reconstructed - #{sample_idx+1}')
            fig.colorbar(im2, ax=axes[col], format='%+2.0f'); col += 1

        if synth_spec is not None:
            synth_np = synth_spec[sample_idx].detach().cpu().numpy()
            im3 = axes[col].imshow(synth_np, aspect='auto', origin='lower', interpolation='none')
            axes[col].set_title(f'Synthesized - #{sample_idx+1}')
            fig.colorbar(im3, ax=axes[col], format='%+2.0f'); col += 1

        im4 = axes[col].imshow(orig_spec, aspect='auto', origin='lower', interpolation='none')
        axes[col].set_title(f'Original - #{sample_idx+1}')
        fig.colorbar(im4, ax=axes[col], format='%+2.0f')

        if rec_spec is not None:
            mse = float(np.mean((orig_spec - rec_spec) ** 2))
            axes[0].set_ylabel(f'MSE: {mse:.4f}')

    def _plot_waveform_row(self, axes, masked_audio, rec_audio, orig_audio, synth_audio, sample_idx):
        col = 0
        t_masked = np.arange(len(masked_audio)) / self.sample_rate
        axes[col].plot(t_masked, masked_audio); axes[col].set_title(f'Input Waveform - #{sample_idx+1}'); axes[col].set_xlabel('Time (s)'); col += 1

        if rec_audio is not None:
            t_rec = np.arange(len(rec_audio)) / self.sample_rate
            axes[col].plot(t_rec, rec_audio); axes[col].set_title(f'Reconstructed Waveform - #{sample_idx+1}'); axes[col].set_xlabel('Time (s)'); col += 1

        if synth_audio is not None:
            t_s = np.arange(len(synth_audio)) / self.sample_rate
            axes[col].plot(t_s, synth_audio); axes[col].set_title(f'Synthesized Waveform - #{sample_idx+1}'); axes[col].set_xlabel('Time (s)'); col += 1

        t_o = np.arange(len(orig_audio)) / self.sample_rate
        axes[col].plot(t_o, orig_audio); axes[col].set_title(f'Original Waveform - #{sample_idx+1}'); axes[col].set_xlabel('Time (s)')

    # ----------------------- checkpoints -----------------------

    def _save_checkpoint(self, epoch, is_best=False):
        """
        Save training state. Regular saves go to checkpoint_epoch_{N}.pt
        Best model goes to best_model.pt
        """
        payload = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
            'best_val_loss': getattr(self, 'best_val_loss', float('inf')),
            'global_step': getattr(self, 'global_step', 0),
            'train_step': getattr(self, 'train_step', 0),
            'metrics': getattr(self, 'metrics', None),
        }

        # NOTE: use self.checkpoint_dir (rename from self.ckpt_dir if needed)
        ckpt_dir = getattr(self, "checkpoint_dir", None) or getattr(self, "ckpt_dir", None)
        if ckpt_dir is None:
            ckpt_dir = self.ckpt_dir = self.checkpoint_dir = os.path.join(self.checkpoint_root, f"{self.model_name}")
        os.makedirs(ckpt_dir, exist_ok=True)

        if is_best:
            path = os.path.join(ckpt_dir, 'best_model.pt')
        else:
            path = os.path.join(ckpt_dir, f'checkpoint_epoch_{epoch}.pt')

        torch.save(payload, path)
        self.logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, checkpoint_path=None, load_best=False):
        """
        Load a checkpoint. If `checkpoint_path` is None:
          - if `load_best` and best exists -> load best_model.pt
          - else -> load latest checkpoint_epoch_*.pt
        Returns: epoch (int) to resume from, or 0 if starting fresh.
        """

        ckpt_dir = getattr(self, "checkpoint_dir", None) or getattr(self, "ckpt_dir", None)
        if ckpt_dir is None:
            ckpt_dir = self.ckpt_dir = os.path.join(self.checkpoint_root, f"{self.model_name}")

        if checkpoint_path is None:
            if load_best and os.path.exists(os.path.join(ckpt_dir, 'best_model.pt')):
                checkpoint_path = os.path.join(ckpt_dir, 'best_model.pt')
                self.logger.info(f"Loading best model checkpoint from {checkpoint_path}")
            else:
                checkpoints = glob.glob(os.path.join(ckpt_dir, 'checkpoint_epoch_*.pt'))
                if not checkpoints:
                    self.logger.info("No checkpoints found. Starting from scratch.")
                    return 0

                latest_epoch = max([int(os.path.basename(cp).split('_')[-1].split('.')[0])
                                    for cp in checkpoints])
                checkpoint_path = os.path.join(ckpt_dir, f'checkpoint_epoch_{latest_epoch}.pt')
                self.logger.info(f"Loading latest checkpoint from {checkpoint_path} (epoch {latest_epoch})")

        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Checkpoint {checkpoint_path} not found. Starting from scratch.")
            return 0

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # primary (requested) key names
            model_key = 'model_state_dict'
            opt_key = 'optimizer_state_dict'
            sch_key = 'scheduler_state_dict'

            # backward-compat fallbacks
            if model_key not in checkpoint and 'model_state' in checkpoint:
                model_key = 'model_state'
            if opt_key not in checkpoint and 'optimizer_state' in checkpoint:
                opt_key = 'optimizer_state'
            if sch_key not in checkpoint and 'scheduler_state' in checkpoint:
                sch_key = 'scheduler_state'

            self.model.load_state_dict(checkpoint[model_key])
            if opt_key in checkpoint and checkpoint[opt_key] is not None:
                self.optimizer.load_state_dict(checkpoint[opt_key])
            if sch_key in checkpoint and checkpoint[sch_key] is not None:
                try:
                    self.scheduler.load_state_dict(checkpoint[sch_key])
                except Exception as _:
                    # scheduler shape might differ across runs; warn but continue
                    self.logger.warning("Scheduler state couldn't be restored; continuing with current scheduler.")

            # optional extras
            self.best_val_loss = checkpoint.get('best_val_loss', getattr(self, 'best_val_loss', float('inf')))
            self.global_step = checkpoint.get('global_step', getattr(self, 'global_step', 0))
            self.train_step = checkpoint.get('train_step', getattr(self, 'train_step', 0))
            if 'metrics' in checkpoint:
                self.metrics = checkpoint['metrics']

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
    def _save_sample_audio(self, specs, recs, audio_dir, start_id, timestamp):
        for i, (spec, rec) in enumerate(zip(specs, recs)):
            sid = start_id + i
            if self.vocoder is None:
                orig_audio = torch_mel_to_audio(spec.cpu(), None).cpu().numpy()
                rec_audio = torch_mel_to_audio(rec.cpu(), None).cpu().numpy()
            else:
                orig_audio = self.vocoder.convert(spec).cpu().numpy()
                rec_audio = self.vocoder.convert(rec).cpu().numpy()

            write(os.path.join(audio_dir, f'original_{sid}_{timestamp}.wav'), self.sample_rate, orig_audio)
            write(os.path.join(audio_dir, f'reconstructed_{sid}_{timestamp}.wav'), self.sample_rate, rec_audio)

    @torch.no_grad()
    def evaluate(self, test_loader, loss_rate=""):
        total_losses = []
        all_metrics = []
        acc_parts = defaultdict(list)

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

        specs_accum, recs_accum = [], []
        chunk_size = 50
        chunk_idx = 0
        sample_id = 0

        iterator = tqdm(test_loader, desc="Testing", unit="batch")

        for bidx, batch in enumerate(iterator):
            visual_feats, spk_emb, masked_spec, spec = self._move_batch_to_device(batch)
            rec_spec, synth_spec = self._forward(visual_feats, spk_emb, masked_spec)
            loss, parts = self._compute_loss(rec_spec, synth_spec, spec)

            total_losses.append(loss.item())
            iterator.set_postfix({"loss": f"{loss.item():.4f}"})

            for k, v in parts.items():
                if v is not None and k != 'loss':
                    acc_parts[k].append(v.item())

            specs_accum.append(spec.detach().cpu())
            recs_accum.append(rec_spec.detach().cpu())

            # process chunk
            do_flush = ((bidx + 1) % chunk_size == 0) or (bidx + 1 == len(test_loader))
            if do_flush:
                specs_cat = torch.cat(specs_accum, dim=0)
                recs_cat = torch.cat(recs_accum, dim=0)
                metrics = calculate_batch_metrics(
                    specs_cat, recs_cat,
                    phase=None, all_text=None, all_pred_text=None,
                    hifigan_vocoder=self.vocoder, tokenizer=None,
                    max_samples=len(specs_cat), sample_rate=self.sample_rate
                )
                all_metrics.append(metrics)

                if chunk_idx == 0:
                    keep = min(5, specs_cat.size(0))
                    self._save_sample_audio(specs_cat[:keep], recs_cat[:keep], audio_dir, sample_id, timestamp)
                    sample_id += keep

                specs_accum, recs_accum = [], []
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                chunk_idx += 1

        final_metrics = {}
        if all_metrics:
            keys = all_metrics[0].keys()
            for k in keys:
                final_metrics[k] = float(np.mean([m[k] for m in all_metrics]))
        final_metrics['loss'] = float(np.mean(total_losses)) if total_losses else 0.0

        parts_avg = {'loss': final_metrics['loss']}
        for k, vals in acc_parts.items():
            parts_avg[k] = float(np.mean(vals)) if vals else None

        self._log_loss_components(parts_avg, final_metrics, phase="test")

        out_json = os.path.join(eval_dir, f'test_metrics_{timestamp}.json')
        with open(out_json, 'w') as f:
            json.dump({**parts_avg, **final_metrics}, f, indent=4)
        self.logger.info(f"Saved comprehensive evaluation results to {out_json}")

        return final_metrics['loss']

    # ----------------------- training loop -----------------------

    def train(self, num_epochs=100, save_interval=10, samples_to_log=2, start_epoch=0):
        self.logger.info(f"Starting training from epoch {start_epoch} for {num_epochs} epochs")
        self.logger.info(f"Training on device: {self.device}")

        total_start = time.time()
        epoch_iter = trange(start_epoch, num_epochs, desc="Training", unit="epoch")

        for epoch in epoch_iter:
            epoch_start = time.time()

            # stop if LR extremely small (optional early stop)
            current_lr = self.optimizer.param_groups[0]['lr']
            if current_lr < 1e-7:
                self.logger.info(f"Stopping: LR ({current_lr:.9f}) below threshold.")
                break

            train_loss = self._train_epoch(epoch, num_epochs)
            self.scheduler.step()

            val_metrics = self._validate_epoch(epoch, num_epochs) if self.val_loader is not None else None
            if val_metrics is not None:
                val_loss = val_metrics['loss']

                # sparse TB
                if (epoch + 1) % 2 == 0:
                    for k, v in val_metrics.items():
                        self.writer.add_scalar(f'Validation/{k}', float(v), self.global_step)

                # plots sparsely
                if (epoch + 1) % 10 == 0 or epoch == start_epoch:
                    self._log_results(epoch + 1, samples_to_log)

                # best model
                if val_loss <= self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch, is_best=True)

            # periodic checkpoint
            if (epoch + 1) % save_interval == 0:
                self._save_checkpoint(epoch, is_best=False)

            # log epoch
            elapsed = time.time() - epoch_start
            if val_metrics:
                val_str = ", ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
            else:
                val_str = ""
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} in {elapsed:.2f}s - "
                f"Train Loss: {train_loss:.4f}, LR: {current_lr:.9f}"
                + (f", Val: {val_str}" if val_str else "")
            )

            postfix = f"Train Loss: {train_loss:.4f}"
            if val_metrics:
                postfix += f", Val Loss: {val_metrics['loss']:.4f}"
            postfix += f", LR: {current_lr:.9f}"
            epoch_iter.set_postfix_str(postfix)

            # sparse TB
            if (epoch + 1) % 2 == 0:
                self.writer.add_scalar('Train/epoch_loss', train_loss, epoch+1)
                self.writer.add_scalar('Train/epoch_lr', current_lr, epoch+1)

        total_time = time.time() - total_start
        self.logger.info(f"Training completed in {total_time/3600:.2f} h")

    def close(self):
        self.writer.flush()
        self.writer.close()
