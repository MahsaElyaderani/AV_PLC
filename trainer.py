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
from matplotlib.ticker import FormatStrFormatter

from torch.utils.tensorboard import SummaryWriter
from asteroid.losses.stoi import NegSTOILoss
from asteroid.losses.pmsqe import SingleSrcPMSQE

from audio_processing import torch_mel2spec, read_gt_input, torch_mel2audio
from metrics import Vocoder, mel_to_audio_hifigan, torch_mel_to_audio
from metrics import calculate_batch_metrics, calculate_metrics
from losses import MSELoss, L1Loss, SpectralConvergenceLoss
from Whisper_Loss import WhisperASRLoss

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["CMU Serif", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
})


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
      'av' : audio-visual (masked_spec, visual_feats, spk_emb -> fused_spc, rec_spec, synth_spec)
    """
    def __init__(
        self,
        model,
        model_name,
        mode,
        sc_loss: bool,
        pesq_loss: bool,
        stoi_loss: bool,
        asr_loss: bool,
        drop_av: bool,
        train_loader,
        val_loader=None,
        sample_rate=16000,
        learning_rate=1e-3,
        checkpoint_dir='checkpoints',
        log_dir='logs',
        vocoder_path=None,
        mixed_precision: bool = True,  # enable mixed precision by default
        use_bf16: bool = True,      # enable BF16 on Ampere+ by default
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
        self.sc_loss = bool(sc_loss)
        self.pesq_loss = bool(pesq_loss)
        self.stoi_loss = bool(stoi_loss)
        self.asr_loss = bool(asr_loss)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_root = checkpoint_dir
        self.log_root = log_dir
        self.vocoder_path = vocoder_path

        # runtime knobs
        self.grad_clip = grad_clip
        self.mixed_precision = mixed_precision
        self.amp_dtype = torch.bfloat16 if (use_bf16 and torch.cuda.is_available() and
                                            torch.cuda.get_device_capability()[0] >= 8)\
            else torch.float16

        # dirs + logging
        self.run_dir = os.path.join(self.log_root, f"{model_name}")
        self.ckpt_dir = os.path.join(self.checkpoint_root, f"{model_name}")
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.logger = setup_logging(model_name, self.run_dir)
        self.writer = SummaryWriter(log_dir=self.run_dir)

        # loss weights
        self.w_pmsqe = 0.01 if self.pesq_loss else 0.0
        self.w_stoi = 0.01 if self.stoi_loss else 0.0 #from 0.01 -> 0.05
        self.w_asr = 0.1 if self.asr_loss else 0.0
        self.w_synth = 0.1
        self.w_rec = 0.05

        # init components
        self._initialize_components(weight_decay, betas, cosine_Tmax)

        # trackers
        self.best_val_loss = float('inf')
        self.global_step = 0
        self.train_step = 0

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

        if self.sc_loss:
            self.sc_criterion = SpectralConvergenceLoss().to(self.device)
        else:
            self.sc_criterion = None

        if self.pesq_loss:
            self.pmsqe = SingleSrcPMSQE().to(self.device)
        else:
            self.pmsqe = None

        if self.stoi_loss:
            self.stoi_criterion = NegSTOILoss(sample_rate=self.sample_rate).to(self.device)
        else:
            self.stoi_criterion = None

        if self.asr_loss:
            self.asr_criterion = WhisperASRLoss(model_size="tiny.en").to(self.device)

        else:
            self.asr_criterion = None

        # vocoder
        self.vocoder = Vocoder(self.vocoder_path) if self.vocoder_path is not None else None

        # optimizer + scheduler
        #trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = optim.AdamW(
            self.model.parameters(), #trainable_params, #
            lr=self.learning_rate,
            weight_decay=weight_decay,
            betas=betas
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cosine_Tmax)

        self.model.to(self.device)
        self.scaler = torch.amp.GradScaler(enabled=self.mixed_precision)

    # ----------------------- drop modality --------------------
    def modality_dropout_probs(self, epoch):

            if epoch <= 5:
                return [1, 0.0, 0.0]
            if 5 < epoch <= 10:
                return [0.9, 0.05, 0.05]
            elif 10 < epoch <= 15:
                return [0.8, 0.1, 0.1]
            elif 15 < epoch <= 20:
                return [0.7, 0.15, 0.15]
            elif 20 < epoch <= 25:
                return [0.6, 0.2, 0.2]
            else:
                return [0.5, 0.25, 0.25]
    # --------------------------------------------------------------------------------------------

    def _move_batch_to_device(self, batch):

        visual_feats, spk_emb, masked_spec, spec, audio_length, mask, path = None, None, None, None, None, None, None
        non_blocking = (self.device.type == 'cuda')

        if self.mode == 'a':
            masked_spec, spec, audio_length, mask, path = batch
        elif self.mode == 'v':
            visual_feats, spk_emb, spec, audio_length, mask, path = batch
        elif self.mode == 'av':
            visual_feats, spk_emb, masked_spec, spec, audio_length, mask, path = batch

        if visual_feats is not None:
            visual_feats = visual_feats.float().to(self.device, non_blocking=non_blocking)
        if spk_emb is not None:
            spk_emb = spk_emb.float().to(self.device, non_blocking=non_blocking)
        if masked_spec is not None:
            masked_spec = masked_spec.float().to(self.device, non_blocking=non_blocking)
        if spec is not None:
            spec = spec.float().to(self.device, non_blocking=non_blocking)
        if audio_length is not None:
            audio_length = audio_length.long().to(self.device, non_blocking=non_blocking)

        return visual_feats, spk_emb, masked_spec, spec, audio_length, mask, path

    def _forward(self, visual_feats, spk_emb, masked_spec, audio_length):
        if self.mode == 'a':
            rec = self.model(masked_spec, None, None, audio_length)
            return None, rec, None
        elif self.mode == 'v':
            synth = self.model(None, visual_feats, spk_emb, audio_length)
            return None, None, synth
        elif self.mode == 'av':
            out = self.model(masked_spec, visual_feats, spk_emb, audio_length)
            if isinstance(out, (tuple, list)):
                if len(out) == 3:
                    return out[0], out[1], out[2]
            return out, None, None
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _compute_loss(self, fused_spec, rec_spec, synth_spec, spec):

        loss = 0.0
        parts = {}

        # reconstruction (always against spec)
        if fused_spec is not None:
            fused_loss = self.rec_criterion(fused_spec, spec)
            parts['fused_loss'] = fused_loss
            loss = loss + fused_loss

        if rec_spec is not None:
            rec_loss = self.rec_criterion(rec_spec, spec)
            parts['rec_loss'] = rec_loss
            loss = loss + self.w_rec * rec_loss

        # synth & spectral convergence for 'v'/'av'
        if synth_spec is not None:
            synth_loss = self.rec_criterion(synth_spec, spec)
            parts['synth_loss'] = synth_loss
            # weight both synth and SC under the same knob
            if self.sc_loss and self.sc_criterion is not None:
                _sc_loss = self.sc_criterion(synth_spec, spec)
                parts['sc_loss'] = _sc_loss
                loss = loss + self.w_synth * (synth_loss + _sc_loss)
            else:
                loss = loss + self.w_synth * synth_loss
        else:
            parts['synth_loss'] = None
            parts['sc_loss'] = None

        #  PMSQE (operate over power spectrogram domain)
        if self.pesq_loss and self.pmsqe is not None and fused_spec is not None:
            # pmsqe expects (B, T, F)
            with torch.amp.autocast('cuda',enabled=False):
                pow_ref = torch_mel2spec(spec.float()).permute(0, 2, 1).contiguous()
                pow_est = torch_mel2spec(fused_spec.float()).permute(0, 2, 1).contiguous()
                pmsqe_loss = torch.mean(self.pmsqe(pow_est, pow_ref))

            parts['pmsqe_loss'] = pmsqe_loss
            loss = loss + self.w_pmsqe * pmsqe_loss
        else:
            parts['pmsqe_loss'] = None

        if self.stoi_loss and self.stoi_criterion is not None and fused_spec is not None:
            ref = torch_mel2audio(spec)
            est = torch_mel2audio(fused_spec)
            nstoi_loss = torch.mean(self.stoi_criterion(est, ref))
            parts['stoi_loss'] = nstoi_loss
            loss = loss + self.w_stoi * nstoi_loss
        else:
            parts['stoi_loss'] = None

        if self.asr_loss and self.asr_criterion is not None and fused_spec is not None:

            l_asr = self.asr_criterion(spec, fused_spec)
            parts['asr_loss'] = l_asr
            loss = loss + self.w_asr * l_asr
        else:
            parts['asr_loss'] = None

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
            self.logger.info(f"New dropout modality probabilities: {new_mode_probs}")

        iterator = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False, unit="batch")

        for bidx, batch in enumerate(iterator):

            if torch.cuda.memory_allocated() > 8e9:
                torch.cuda.empty_cache()

            visual_feats, spk_emb, masked_spec, spec, audio_length, mask, path = self._move_batch_to_device(batch)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=self.mixed_precision):
                fused_spec, rec_spec, synth_spec = self._forward(visual_feats, spk_emb, masked_spec, audio_length)
                loss, parts = self._compute_loss(fused_spec, rec_spec, synth_spec, spec)


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
                if parts.get('stoi_loss') is not None:
                    p['stoi'] = f"{parts['stoi_loss'].item():.5f}"
                if parts.get('asr_loss') is not None:
                    p['asr'] = f"{parts['asr_loss'].item():.5f}"
                if parts.get('fused_loss') is not None:
                    p['fused'] = f"{parts['fused_loss'].item():.5f}"
                iterator.set_postfix(p)

            # logging
            if self.train_step % 5000 == 0:
                self.writer.add_scalar('Train/loss', loss.item(), self.global_step)
                for k, v in parts.items():
                    if k != 'loss' and v is not None:
                        self.writer.add_scalar(f'Train/{k}', v.item(), self.global_step)
                self.writer.add_scalar('Train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)

        avg = running_loss / max(1, n_batches)
        self._log_loss_components(parts, False, phase="train", epoch=epoch + 1)
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

        target_metric_samples = 16
        specs_accum, fused_accum = [], []
        recs_accum, synth_accum = [], []
        mask_accum, path_accum = [], []

        for bidx, batch in enumerate(iterator):

            visual_feats, spk_emb, masked_spec, spec, audio_length, mask, path = self._move_batch_to_device(batch)
            fused_spec, rec_spec, synth_spec = self._forward(visual_feats, spk_emb, masked_spec, audio_length)
            loss, parts = self._compute_loss(fused_spec, rec_spec, synth_spec, spec)

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
                #recs_accum.append(rec_spec[:take].detach().to('cpu') if rec_spec is not None else None)
                #synth_accum.append(synth_spec[:take].detach().to('cpu') if synth_spec is not None else None)
                fused_accum.append(fused_spec[:take].detach().to('cpu') if fused_spec is not None else None)

                mask_accum.append(mask[:take])
                path_accum.extend(list(path[:take]))

        metrics = {'mse': 0.0, 'psnr': 0.0, 'pesq': 0.0, 'stoi': 0.0, 'estoi':0.0,
                   'plcmos': 0.0, 'cer': 0.0, 'wer': 0.0}
        if specs_accum and fused_accum:
            specs_cat = torch.cat(specs_accum, dim=0)
            fused_accum = torch.cat(fused_accum, dim=0)
            mask_cat = torch.cat(mask_accum, dim=0)

            metrics = calculate_batch_metrics(
                original_batch=specs_cat, reconstructed_batch=fused_accum,
                all_text=None, all_pred_text=None,
                mask=None, path=path_accum,
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
            header = f"Epoch {epoch} Train" if epoch is not None else "Train"
        elif phase == "validation":
            header = f"Epoch {epoch} Val"
        else:
            if phase == "input_test":
                header = "Input"
            else:
                header = "Test"

        if 'loss' in loss_components:
            parts.append(f"Total Loss: {loss_components['loss']:.6f}")

        comp_bits = []
        for k in ['fused_loss' ,'rec_loss', 'synth_loss', 'sc_loss', 'pmsqe_loss', 'stoi_loss', 'asr_loss']:
            v = loss_components.get(k, None)
            if v is not None:
                comp_bits.append(f"{k.replace('_',' ').title()}: {float(v):.6f}")
        if comp_bits:
            parts.append("Components [" + ", ".join(comp_bits) + "]")

        if metrics_dict and phase in ["validation", "test", "input_test"]:
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
        if metrics_dict and phase in ["validation", "test", "input_test"]:
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

        visual_feats, spk_emb, masked_spec, spec, audio_length, mask, path = self._move_batch_to_device(batch)

        if spec is not None:
            spec = spec[:num_samples]
        if masked_spec is not None:
            masked_spec = masked_spec[:num_samples]
        if visual_feats is not None:
            visual_feats = visual_feats[:num_samples]
        if spk_emb is not None:
            spk_emb = spk_emb[:num_samples]

        fused_spec, rec_spec, synth_spec = self._forward(visual_feats, spk_emb, masked_spec, audio_length)

        if fused_spec is not None:
            fused_spec = fused_spec[:num_samples]
        if rec_spec is not None:
            rec_spec = rec_spec[:num_samples]
        if synth_spec is not None:
            synth_spec = synth_spec[:num_samples]
        mask = mask[:num_samples]
        path = path[:num_samples]

        self._create_plots(spec, masked_spec, fused_spec, rec_spec, synth_spec, mask, path, epoch, plot_dir)

    def _create_plots(self, spec, masked_spec, fused_spec, rec_spec, synth_spec, mask, path, epoch, plot_dir):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        n = spec.size(0)

        cols = 1
        cols += int(masked_spec is not None)
        cols += int(rec_spec is not None)
        cols += int(synth_spec is not None)
        cols += int(fused_spec is not None)

        w, h = 3.5,  4.5
        fig, axes = plt.subplots(n * 2, cols, figsize=(w * cols, h * n))

        for i in range(n):
            spec_np = spec[i].detach().cpu().numpy()
            masked_np = masked_spec[i].detach().cpu().numpy() if masked_spec is not None else None
            fused_np = fused_spec[i].detach().cpu().numpy() if fused_spec is not None else None
            rec_np = rec_spec[i].detach().cpu().numpy() if rec_spec is not None else None
            synth_np = synth_spec[i].detach().cpu().numpy() if synth_spec is not None else None

            orig_audio, masked_audio = read_gt_input(path[i], mask[i])
            masked_audio = masked_audio if masked_spec is not None else None

            if self.vocoder is None:
                fused_audio = torch_mel_to_audio(fused_spec[i].cpu()).cpu().numpy() if fused_spec is not None else None
                rec_audio = torch_mel_to_audio(rec_spec[i].cpu()).cpu().numpy() if rec_spec is not None else None
                synth_audio = torch_mel_to_audio(synth_spec[i].cpu()).cpu().numpy() if synth_spec is not None else None
            else:
                fused_audio = mel_to_audio_hifigan(fused_spec[i].cpu(), self.vocoder).cpu().numpy() if fused_spec is not None else None
                rec_audio = mel_to_audio_hifigan(rec_spec[i], self.vocoder).cpu().numpy() if rec_spec is not None else None
                synth_audio = mel_to_audio_hifigan(synth_spec[i], self.vocoder).cpu().numpy() if synth_spec is not None else None

            row = i * 2
            ax_spec_row = axes[row]
            ax_wave_row = axes[row + 1]

            self._plot_spectrogram_row(ax_spec_row, masked_np, fused_np, rec_np, spec_np, synth_np, i, fig)
            self._plot_waveform_row(ax_wave_row, masked_audio, fused_audio, rec_audio, orig_audio, synth_audio, i)

        #plt.suptitle(f'Spectrogram and Waveform Comparison - Epoch {epoch}')
        plt.tight_layout()
        fig_path = os.path.join(plot_dir, f'comparison_epoch{epoch}_{timestamp}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def _plot_spectrogram_row(self, axes, masked_spec ,fused_spec, rec_spec, orig_spec, synth_spec, sample_idx, fig):
        cols = axes.shape[0] if hasattr(axes, 'shape') else 1
        col = 0
        plt.rcParams.update({'font.size': 15})

        if masked_spec is not None:
            im1 = axes[col].imshow(masked_spec, aspect='auto',
                                   origin='lower', interpolation='none')
            #set the subfig size to be (4.5, 3.5)
            #axes[col].set_title(f'Input')# - #{sample_idx+1}')

            #mse = float(np.mean((orig_spec - masked_spec) ** 2))
            #axes[col].set_ylabel(f'MSE Masked: {mse:.4f}')
            col += 1

        if fused_spec is not None:
            im11 = axes[col].imshow(fused_spec, aspect='auto', origin='lower', interpolation='none')
            #axes[col].set_title(f'Fused')# - #{sample_idx+1}')

            #mse = float(np.mean((orig_spec - fused_spec) ** 2))
            #axes[col].set_ylabel(f'MSE FUSE: {mse:.2f}')
            col += 1

        if rec_spec is not None:
            im2 = axes[col].imshow(rec_spec, aspect='auto', origin='lower', interpolation='none')
            #axes[col].set_title(f'Reconstructed')# - #{sample_idx+1}')

            #mse = float(np.mean((orig_spec - rec_spec) ** 2))
            #axes[col].set_ylabel(f'MSE REC: {mse:.2f}')
            col += 1

        if synth_spec is not None:
            im3 = axes[col].imshow(synth_spec, aspect='auto', origin='lower', interpolation='none')
            #axes[col].set_title(f'Synthesized')# - #{sample_idx+1}')

            #mse = float(np.mean((orig_spec - synth_spec) ** 2))
            #axes[col].set_ylabel(f'MSE SYNTH: {mse:.2f}')
            col += 1

        im4 = axes[col].imshow(orig_spec, aspect='auto', origin='lower', interpolation='none')
        #axes[col].set_title(f'Original')# - #{sample_idx+1}')

        height, width = orig_spec.shape

        for ax in axes:
            ax.set_xlim(0, width)
            ax.set_ylim(0, height)

            # Force tick labels to appear on both axes
            ax.set_xticks(np.linspace(0, width, 5))
            ax.set_yticks(np.linspace(0, height, 5))

            # Display numeric labels (not scientific notation)
            ax.ticklabel_format(axis='both', style='plain')
            ax.tick_params(axis='both', which='both', labelsize=12)

            # Keep proper aspect ratio
            ax.set_aspect('auto')

        # --- Colorbar ---
        # fig.canvas.draw()
        # last_ax = axes[col]
        # bbox = last_ax.get_position()
        # pad = 0.1  # gap between last axis and colorbar
        # cbar_w = 0.003  # width of the colorbar axis
        #
        # cax = fig.add_axes([bbox.x1 + pad, bbox.y0 + 0.05, cbar_w, bbox.height])
        # fig.colorbar(im4, cax=cax, format='%+2.0f')

    def _plot_waveform_row(self, axes, masked_audio, fused_audio, rec_audio, orig_audio, synth_audio, sample_idx):
        col = 0
        plt.rcParams.update({'font.size': 15})

        def _to_float_m1p1(x):
            x = np.asarray(x)
            if x.dtype.kind in "iu":
                x = x.astype(np.float32) / np.iinfo(x.dtype).max
            return np.clip(x.astype(np.float32), -1.0, 1.0)

        if masked_audio is not None:
            t_masked = np.arange(len(masked_audio)) / self.sample_rate
            axes[col].plot(t_masked, _to_float_m1p1(masked_audio))
            #axes[col].set_title(f'Input Waveform - #{sample_idx+1}')
            #axes[col].set_xlabel('Time (s)')
            col += 1

        if fused_audio is not None:
            t_fused = np.arange(len(fused_audio)) / self.sample_rate
            axes[col].plot(t_fused, _to_float_m1p1(fused_audio))
            #axes[col].set_title(f'Fused Waveform - #{sample_idx+1}')
            #axes[col].set_xlabel('Time (s)')
            col += 1

        if rec_audio is not None:
            t_rec = np.arange(len(rec_audio)) / self.sample_rate
            axes[col].plot(t_rec, _to_float_m1p1(rec_audio))
            #axes[col].set_title(f'Reconstructed Waveform - #{sample_idx+1}')
            #axes[col].set_xlabel('Time (s)')
            col += 1

        if synth_audio is not None:
            t_s = np.arange(len(synth_audio)) / self.sample_rate
            axes[col].plot(t_s, _to_float_m1p1(synth_audio))
            #axes[col].set_title(f'Synthesized Waveform - #{sample_idx+1}')
            #axes[col].set_xlabel('Time (s)')
            col += 1

        t_o = np.arange(len(orig_audio)) / self.sample_rate
        axes[col].plot(t_o, _to_float_m1p1(orig_audio))
        #axes[col].set_title(f'Original Waveform - #{sample_idx+1}')
        #axes[col].set_xlabel('Time (s)')
        # --- Set consistent time and amplitude ranges ---
        max_len = max(len(x) for x in [
            a for a in [masked_audio, fused_audio, rec_audio, synth_audio, orig_audio] if a is not None
        ])
        time_max = max_len / self.sample_rate
        min_amp, max_amp = orig_audio.min(), orig_audio.max()

        for ax in axes:
            ax.set_xlim(0, time_max)
            ax.set_ylim(min_amp, max_amp)
            ax.set_xticks(np.round(np.linspace(0, time_max, int(time_max) + 1)).astype(int))

            ax.set_yticks(np.linspace(min_amp, max_amp, 5))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            ax.tick_params(axis='both', which='both', labelsize=12)

    # ----------------------- checkpoints -----------------------

    def _save_checkpoint(self, epoch, is_best=False):

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
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

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
    def _save_sample_audio(self, specs, recs, audio_dir, start_id, timestamp, output="fused"):
        for i, (spec, rec) in enumerate(zip(specs, recs)):
            sid = start_id + i
            if self.vocoder is None:
                orig_audio = torch_mel_to_audio(spec.cpu()).cpu().numpy()
                rec_audio = torch_mel_to_audio(rec.cpu()).cpu().numpy()
            else:
                orig_audio = mel_to_audio_hifigan(spec.cpu(), self.vocoder).cpu().numpy()
                rec_audio = mel_to_audio_hifigan(rec.cpu(), self.vocoder).cpu().numpy()

            write(os.path.join(audio_dir, f'original_{sid}_{timestamp}.wav'),
                  self.sample_rate, orig_audio)
            write(os.path.join(audio_dir, f'{output}_{sid}_{timestamp}.wav'),
                  self.sample_rate, rec_audio)

    @torch.no_grad()
    def evaluate(self, test_loader, loss_rate="", checkpoint_path=None):
        total_losses = []
        fused_metrics = []
        rec_metrics = []
        synth_metrics = []
        input_metrics = []
        acc_parts = defaultdict(list)

        eval_dir = os.path.join(self.run_dir, f'test_{loss_rate}')
        audio_dir = os.path.join(eval_dir, 'audio_samples')
        plot_dir = os.path.join(eval_dir, 'plots')
        os.makedirs(eval_dir, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)

        self.logger.info(f"Evaluating on test set with loss rates {loss_rate}...")
        best_path = os.path.join(self.ckpt_dir, 'best_model.pt')
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        elif os.path.exists(best_path):
            self.load_checkpoint(best_path)

        self.model.eval()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        specs_accum, fused_accum, recs_accum, synth_accum, maskd_accum = [], [], [], [], []
        all_masks, all_paths = [], []
        chunk_size = 200
        chunk_idx = 0
        sample_id = 0

        iterator = tqdm(test_loader, desc="Testing", unit="batch")

        for bidx, batch in enumerate(iterator):
            visual_feats, spk_emb, masked_spec, spec, audio_length, mask, path = self._move_batch_to_device(batch)
            fused_spec, rec_spec, synth_spec = self._forward(visual_feats, spk_emb, masked_spec, audio_length)
            loss, parts = self._compute_loss(fused_spec, rec_spec, synth_spec, spec)

            total_losses.append(loss.item())
            iterator.set_postfix({"loss": f"{loss.item():.4f}"})

            for k, v in parts.items():
                if v is not None and k != 'loss':
                    acc_parts[k].append(v.item())

            #save the first batch results
            if bidx == 0:
                for i in range(len(spec)):
                    self._create_plots(
                        spec[i:i + 1], masked_spec[i:i + 1],
                        fused_spec[i:i + 1], rec_spec[i:i + 1], synth_spec[i:i + 1],
                        mask[i:i + 1], path[i:i + 1],  # keeps it as a list of length 1
                        "TEST", plot_dir)

            specs_accum.append(spec.detach().cpu())

            if fused_spec is not None:
                fused_accum.append(fused_spec.detach().cpu())
            if rec_spec is not None:
                recs_accum.append(rec_spec.detach().cpu())
            if synth_spec is not None:
                synth_accum.append(synth_spec.detach().cpu())
            if masked_spec is not None:
                maskd_accum.append(masked_spec.detach().cpu())
            all_masks.append(mask.detach().cpu())
            all_paths.extend(list(path))

            # process chunk
            do_flush = ((bidx + 1) % chunk_size == 0) or (bidx + 1 == len(test_loader))
            if do_flush:

                specs_cat = torch.cat(specs_accum, dim=0)
                keep = min(10, specs_cat.size(0))

                if len(fused_accum) > 0:
                    fused_cat = torch.cat(fused_accum, dim=0)
                    metrics = calculate_batch_metrics(
                        original_batch=specs_cat, reconstructed_batch=fused_cat,
                        all_text=None, all_pred_text=None,
                        mask=None, path=all_paths,
                        hifigan_vocoder=self.vocoder, tokenizer=None,
                        max_samples=len(specs_cat), sample_rate=self.sample_rate)
                    fused_metrics.append(metrics)
                    self._save_sample_audio(specs_cat[:keep],
                                            fused_cat[:keep],
                                            audio_dir, sample_id,
                                            timestamp, output="fused")


                if len(recs_accum) > 0:
                    recs_cat = torch.cat(recs_accum, dim=0)
                    metrics = calculate_batch_metrics(
                        original_batch=specs_cat, reconstructed_batch=recs_cat,
                        all_text=None, all_pred_text=None,
                        mask=None, path=all_paths,
                        hifigan_vocoder=self.vocoder, tokenizer=None,
                        max_samples=len(specs_cat), sample_rate=self.sample_rate)
                    rec_metrics.append(metrics)
                    self._save_sample_audio(specs_cat[:keep],
                                             recs_cat[:keep],
                                             audio_dir, sample_id,
                                             timestamp, output="recs")

                if len(synth_accum) > 0:
                    synth_cat = torch.cat(synth_accum, dim=0)
                    metrics = calculate_batch_metrics(original_batch=specs_cat, reconstructed_batch=synth_cat,
                        all_text=None, all_pred_text=None,
                        mask=None, path=all_paths,
                        hifigan_vocoder=self.vocoder, tokenizer=None,
                        max_samples=len(specs_cat), sample_rate=self.sample_rate)
                    synth_metrics.append(metrics)
                    self._save_sample_audio(specs_cat[:keep],
                                             synth_cat[:keep],
                                             audio_dir, sample_id,
                                             timestamp, output="synth")

                if len(maskd_accum) > 0:
                    maskd_cat = torch.cat(maskd_accum, dim=0)
                    all_masks_cat = torch.cat(all_masks, dim=0)
                    in_metrics = calculate_batch_metrics(original_batch=specs_cat, reconstructed_batch=maskd_cat,
                    all_text = None, all_pred_text = None,
                    mask=all_masks_cat, path=all_paths,
                    hifigan_vocoder = self.vocoder, tokenizer = None,
                    max_samples = len(specs_cat), sample_rate = self.sample_rate)
                    input_metrics.append(in_metrics)
                    self._save_sample_audio(specs_cat[:keep],
                                            maskd_cat[:keep],
                                            audio_dir, sample_id,
                                            timestamp, output="masked")
                sample_id += keep

                specs_accum, maskd_accum = [], []
                fused_accum, recs_accum, synth_accum = [], [], []
                all_masks, all_paths = [], []

                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                chunk_idx += 1

        final_metrics = {}
        if fused_metrics:
            keys = fused_metrics[0].keys()
            for k in keys:
                final_metrics[k] = float(np.mean([m[k] for m in fused_metrics]))
        final_metrics['loss'] = float(np.mean(total_losses)) if total_losses else 0.0

        final_in_metrics = {}
        if input_metrics:
            keys = input_metrics[0].keys()
            for k in keys:
                final_in_metrics[k] = float(np.mean([m[k] for m in input_metrics]))

        parts_avg = {'loss': final_metrics['loss']}
        for k, vals in acc_parts.items():
            parts_avg[k] = float(np.mean(vals)) if vals else None


        self._log_loss_components(parts_avg, final_in_metrics, phase="input_test")
        self._log_loss_components(parts_avg, final_metrics, phase="test")

        out_json = os.path.join(eval_dir, f'test_metrics_{timestamp}.json')
        with open(out_json, 'w') as f:
            json.dump({**parts_avg, **final_metrics}, f, indent=4)
        self.logger.info(f"Saved comprehensive evaluation results to {out_json}")

        return final_metrics['loss']

    @torch.no_grad()
    def evaluate_plots(self, test_loader, loss_rate,checkpoint_path=None):

            eval_dir = os.path.join(self.run_dir, f'test_{loss_rate}')
            audio_dir = os.path.join(eval_dir, 'audio_samples')
            plot_dir = os.path.join(eval_dir, 'plots')
            os.makedirs(eval_dir, exist_ok=True)
            os.makedirs(audio_dir, exist_ok=True)
            os.makedirs(plot_dir, exist_ok=True)

            self.logger.info(f"Evaluating on test set with loss rates {loss_rate}...")
            best_path = os.path.join(self.ckpt_dir, 'best_model.pt')
            if checkpoint_path is not None:
                self.load_checkpoint(checkpoint_path)
            elif os.path.exists(best_path):
                self.load_checkpoint(best_path)

            self.model.eval()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sample_id = 0

            iterator = tqdm(test_loader, desc="Testing", unit="batch")

            for bidx, batch in enumerate(iterator):
                if bidx <= 5:
                    visual_feats, spk_emb, masked_spec, spec, audio_length, mask, path = self._move_batch_to_device(batch)
                    fused_spec, _, _ = self._forward(visual_feats, spk_emb,
                                                     masked_spec, audio_length)
                    rec_spec, _, _ = self._forward(torch.zeros_like(visual_feats),
                                                   torch.zeros_like(spk_emb),
                                                   masked_spec, audio_length)
                    synth_spec, _, _ = self._forward(visual_feats, spk_emb,
                                                     torch.zeros_like(masked_spec), audio_length)
                    loss, parts = self._compute_loss(fused_spec, rec_spec, synth_spec, spec)

                    iterator.set_postfix({"loss": f"{loss.item():.4f}"})

                    for i in range(len(spec)):
                        self._create_plots(
                            spec[i:i + 1], masked_spec[i:i + 1],
                            fused_spec[i:i + 1],
                            rec_spec[i:i + 1] if rec_spec is not None else None,
                            synth_spec[i:i + 1],
                            mask[i:i + 1], path[i:i + 1],  # keeps it as a list of length 1
                            f"Test{sample_id}", plot_dir)
                        sample_id += 1

    @torch.no_grad()
    def save_plots(self, test_loader, loss_rate, checkpoint_path=None):

        eval_dir = os.path.join(self.run_dir, f'test_{loss_rate}')
        audio_dir = os.path.join(eval_dir, 'audio_samples')
        spec_dir = os.path.join(eval_dir, 'spectrograms')
        os.makedirs(eval_dir, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(spec_dir, exist_ok=True)

        self.logger.info(f"Evaluating on test set with loss rates {loss_rate}...")
        best_path = os.path.join(self.ckpt_dir, 'best_model.pt')
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        elif os.path.exists(best_path):
            self.load_checkpoint(best_path)

        self.model.eval()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_id = 0

        iterator = tqdm(test_loader, desc="Testing", unit="batch")

        for bidx, batch in enumerate(iterator):
            if bidx <= 5:
                visual_feats, spk_emb, masked_spec, spec, audio_length, mask, path = self._move_batch_to_device(batch)
                fused_spec, _, _ = self._forward(visual_feats, spk_emb,
                                                 masked_spec, audio_length)
                rec_spec, _, _ = self._forward(torch.zeros_like(visual_feats), spk_emb,
                                               masked_spec, audio_length)
                synth_spec, _, _ = self._forward(visual_feats, spk_emb,
                                                 torch.zeros_like(masked_spec), audio_length)
                loss, parts = self._compute_loss(fused_spec, rec_spec, synth_spec, spec)

                iterator.set_postfix({"loss": f"{loss.item():.4f}"})

                for i in range(len(spec)):
                    spec_np = spec[i].detach().cpu().numpy()
                    masked_np = masked_spec[i].detach().cpu().numpy() if masked_spec is not None else None
                    fused_np = fused_spec[i].detach().cpu().numpy() if fused_spec is not None else None
                    rec_np = rec_spec[i].detach().cpu().numpy() if rec_spec is not None else None
                    synth_np = synth_spec[i].detach().cpu().numpy() if synth_spec is not None else None

                    orig_audio, masked_audio = read_gt_input(path[i], mask[i])
                    masked_audio = masked_audio if masked_spec is not None else None

                    if self.vocoder is None:
                        fused_audio = torch_mel_to_audio(
                            fused_spec[i].cpu()).cpu().numpy() if fused_spec is not None else None
                        rec_audio = torch_mel_to_audio(
                            rec_spec[i].cpu()).cpu().numpy() if rec_spec is not None else None
                        synth_audio = torch_mel_to_audio(
                            synth_spec[i].cpu()).cpu().numpy() if synth_spec is not None else None
                    else:
                        fused_audio = mel_to_audio_hifigan(fused_spec[i].cpu(),
                                                           self.vocoder).cpu().numpy() if fused_spec is not None else None
                        rec_audio = mel_to_audio_hifigan(rec_spec[i],
                                                         self.vocoder).cpu().numpy() if rec_spec is not None else None
                        synth_audio = mel_to_audio_hifigan(synth_spec[i],
                                                           self.vocoder).cpu().numpy() if synth_spec is not None else None

                    def _to_float_m1p1(x):
                        x = np.asarray(x)
                        if x.dtype.kind in "iu":
                            x = x.astype(np.float32) / np.iinfo(x.dtype).max
                        return np.clip(x.astype(np.float32), -1.0, 1.0)

                    # Normalize audio before saving
                    original_audio_f32 = _to_float_m1p1(orig_audio)
                    masked_audio_f32 = _to_float_m1p1(masked_audio)
                    reconstructed_audio_f32 = _to_float_m1p1(fused_audio)

                    spec_path = os.path.join(spec_dir, f'conformer_{sample_id}_{timestamp}.npz')
                    np.savez_compressed(
                        spec_path,
                        masked_spec=masked_np.astype("float32"),
                        recon_spec=fused_np.astype("float32"),
                        original_spec=spec_np.astype("float32"),
                        masked_audio=masked_audio_f32,
                        recon_audio=reconstructed_audio_f32,
                        original_audio=original_audio_f32,
                    )
                    sample_id += 1

    def evaluate_samples(self, test_loader, loss_rate):

        eval_dir = os.path.join(self.run_dir, f'test_{loss_rate}')
        os.makedirs(eval_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.logger.info(f"Evaluating on test set with loss rates {loss_rate}...")
        best_path = os.path.join(self.ckpt_dir, 'best_model.pt')
        if os.path.exists(best_path):
            self.load_checkpoint(best_path)

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
                    visual_feats, spk_emb, masked_spec, spec, audio_length, mask, path = self._move_batch_to_device(
                        batch)
                    fused_spec, rec_spec, synth_spec = self._forward(visual_feats, spk_emb, masked_spec, audio_length)

                    for i in range(len(spec)):
                        sample_id += 1
                        sample_metrics = calculate_metrics(original_spec=spec[i],
                                                           reconstructed_spec=fused_spec[i],
                                                           path=path[i],
                                                           mask=None,
                                                           hifigan_vocoder=self.vocoder)
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

    @torch.no_grad()
    def evaluate_synth(self, test_loader, loss_rate="", checkpoint_path=None):

        total_losses = []
        synth_metrics = []
        acc_parts = defaultdict(list)

        eval_dir = os.path.join(self.run_dir, f'test_{loss_rate}')
        audio_dir = os.path.join(eval_dir, 'audio_samples')
        plot_dir = os.path.join(eval_dir, 'plots')
        os.makedirs(eval_dir, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)

        self.logger.info(f"Evaluating on test set with loss rates {loss_rate}...")
        best_path = os.path.join(self.ckpt_dir, 'best_model.pt')
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        elif os.path.exists(best_path):
            self.load_checkpoint(best_path)

        self.model.eval()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        specs_accum, synth_accum, all_paths = [], [], []

        chunk_size = 200
        chunk_idx = 0
        sample_id = 0

        iterator = tqdm(test_loader, desc="Testing", unit="batch")

        for bidx, batch in enumerate(iterator):
            visual_feats, spk_emb, masked_spec, spec, audio_length, mask, path = self._move_batch_to_device(batch)

            fused_spec, _, _ = self._forward(visual_feats, spk_emb,
                                             masked_spec, audio_length)
            rec_spec, _, _ = self._forward(torch.zeros_like(visual_feats), spk_emb,
                                           masked_spec, audio_length)
            synth_spec, _, _ = self._forward(visual_feats, spk_emb,
                                             torch.zeros_like(masked_spec), audio_length)
            loss, parts = self._compute_loss(fused_spec, rec_spec, synth_spec, spec)

            total_losses.append(loss.item())
            iterator.set_postfix({"loss": f"{loss.item():.4f}"})

            for k, v in parts.items():
                if v is not None and k != 'loss':
                    acc_parts[k].append(v.item())

            specs_accum.append(spec.detach().cpu())

            if synth_spec is not None:
                synth_accum.append(synth_spec.detach().cpu())
            all_paths.extend(list(path))

            do_flush = ((bidx + 1) % chunk_size == 0) or (bidx + 1 == len(test_loader))
            if do_flush:

                specs_cat = torch.cat(specs_accum, dim=0)
                keep = min(10, specs_cat.size(0))

                if len(synth_accum) > 0:
                    synth_cat = torch.cat(synth_accum, dim=0)
                    metrics = calculate_batch_metrics(original_batch=specs_cat, reconstructed_batch=synth_cat,
                            all_text=None, all_pred_text=None,
                            mask=None, path=all_paths,
                            hifigan_vocoder=self.vocoder, tokenizer=None,
                            max_samples=len(specs_cat), sample_rate=self.sample_rate)
                    synth_metrics.append(metrics)

                sample_id += keep

                specs_accum, synth_accum, all_paths = [], [], []

                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                chunk_idx += 1

        final_metrics = {}
        if synth_metrics:
            keys = synth_metrics[0].keys()
            for k in keys:
                final_metrics[k] = float(np.mean([m[k] for m in synth_metrics]))
        final_metrics['loss'] = float(np.mean(total_losses)) if total_losses else 0.0

        parts_avg = {'loss': final_metrics['loss']}
        for k, vals in acc_parts.items():
            parts_avg[k] = float(np.mean(vals)) if vals else None

        self._log_loss_components(parts_avg, final_metrics, phase="test")

        out_json = os.path.join(eval_dir, f'test_metrics_synth_{timestamp}.json')
        with open(out_json, 'w') as f:
            json.dump({**parts_avg, **final_metrics}, f, indent=4)
        self.logger.info(f"Saved comprehensive evaluation results to {out_json}")

        return final_metrics['loss']


    @torch.no_grad()
    def evaluate_rec(self, test_loader, loss_rate="", checkpoint_path=None):
        total_losses = []
        rec_metrics = []
        input_metrics = []
        acc_parts = defaultdict(list)

        eval_dir = os.path.join(self.run_dir, f'test_{loss_rate}')
        audio_dir = os.path.join(eval_dir, 'audio_samples')
        plot_dir = os.path.join(eval_dir, 'plots')
        os.makedirs(eval_dir, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)

        self.logger.info(f"Evaluating on test set with loss rates {loss_rate}...")
        best_path = os.path.join(self.ckpt_dir, 'best_model.pt')
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        elif os.path.exists(best_path):
            self.load_checkpoint(best_path)

        self.model.eval()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        specs_accum, fused_accum, recs_accum, synth_accum, maskd_accum = [], [], [], [], []
        all_masks, all_paths = [], []
        chunk_size = 200
        chunk_idx = 0
        sample_id = 0

        iterator = tqdm(test_loader, desc="Testing", unit="batch")

        for bidx, batch in enumerate(iterator):
            visual_feats, spk_emb, masked_spec, spec, audio_length, mask, path = self._move_batch_to_device(batch)

            fused_spec, _, _ = self._forward(visual_feats, spk_emb,
                                             masked_spec, audio_length)
            rec_spec, _, _ = self._forward(torch.zeros_like(visual_feats),
                                           torch.zeros_like(spk_emb),
                                           masked_spec, audio_length)
            synth_spec, _, _ = self._forward(visual_feats, spk_emb,
                                             torch.zeros_like(masked_spec), audio_length)
            loss, parts = self._compute_loss(fused_spec, rec_spec, synth_spec, spec)

            total_losses.append(loss.item())
            iterator.set_postfix({"loss": f"{loss.item():.4f}"})

            for k, v in parts.items():
                if v is not None and k != 'loss':
                    acc_parts[k].append(v.item())

            specs_accum.append(spec.detach().cpu())

            if rec_spec is not None:
                recs_accum.append(rec_spec.detach().cpu())
            if masked_spec is not None:
                maskd_accum.append(masked_spec.detach().cpu())
            all_masks.append(mask.detach().cpu())
            all_paths.extend(list(path))

            do_flush = ((bidx + 1) % chunk_size == 0) or (bidx + 1 == len(test_loader))
            if do_flush:

                specs_cat = torch.cat(specs_accum, dim=0)
                keep = min(10, specs_cat.size(0))

                if len(recs_accum) > 0:
                    recs_cat = torch.cat(recs_accum, dim=0)
                    metrics = calculate_batch_metrics(
                        original_batch=specs_cat, reconstructed_batch=recs_cat,
                        all_text=None, all_pred_text=None,
                        mask=None, path=all_paths,
                        hifigan_vocoder=self.vocoder, tokenizer=None,
                        max_samples=len(specs_cat), sample_rate=self.sample_rate)
                    rec_metrics.append(metrics)

                sample_id += keep

                specs_accum, maskd_accum = [], []
                fused_accum, recs_accum, synth_accum = [], [], []
                all_masks, all_paths = [], []

                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                chunk_idx += 1

        final_metrics = {}
        if rec_metrics:
            keys = rec_metrics[0].keys()
            for k in keys:
                final_metrics[k] = float(np.mean([m[k] for m in rec_metrics]))
        final_metrics['loss'] = float(np.mean(total_losses)) if total_losses else 0.0

        final_in_metrics = {}
        if input_metrics:
            keys = input_metrics[0].keys()
            for k in keys:
                final_in_metrics[k] = float(np.mean([m[k] for m in input_metrics]))

        parts_avg = {'loss': final_metrics['loss']}
        for k, vals in acc_parts.items():
            parts_avg[k] = float(np.mean(vals)) if vals else None


        self._log_loss_components(parts_avg, final_in_metrics, phase="input_test")
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

            # stop if LR extremely small (early stop)
            current_lr = self.optimizer.param_groups[0]['lr']
            if current_lr < 1e-7:
                self.logger.info(f"Stopping: LR ({current_lr:.9f}) below threshold.")
                break

            train_loss = self._train_epoch(epoch, num_epochs)
            self.scheduler.step()

            val_metrics = self._validate_epoch(epoch, num_epochs) if self.val_loader is not None else None
            if val_metrics is not None:
                val_loss = val_metrics['loss']

                if (epoch + 1) % 10 == 0 or epoch == start_epoch:
                    self._log_results(epoch + 1, samples_to_log)

                # best model
                if val_loss <= self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch, is_best=True)

            if (epoch + 1) % save_interval == 0:
                self._save_checkpoint(epoch, is_best=False)

            elapsed = time.time() - epoch_start

            postfix = f"Train Loss: {train_loss:.4f}"
            if val_metrics:
                postfix += f", Val Loss: {val_metrics['loss']:.4f}"
            postfix += f", LR: {current_lr:.9f}"
            postfix += f", elapsed time: {elapsed:.4f}"
            epoch_iter.set_postfix_str(postfix)

        total_time = time.time() - total_start
        self.logger.info(f"Training completed in {total_time/3600:.2f} h")

    def close(self):
        self.writer.flush()
        self.writer.close()
