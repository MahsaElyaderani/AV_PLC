import os
import hashlib
import re
from pathlib import Path
import csv
import time
import glob
import gc
import logging
import numpy as np

from tqdm import tqdm, trange
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.io.wavfile import write
from collections import defaultdict

import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from asteroid.losses.stoi import NegSTOILoss
from asteroid.losses.pmsqe import SingleSrcPMSQE

from shared.audio_processing import (mel_to_power_linear, read_gt_input, torch_mel2audio,
                                     load_audio_ffmpeg)
from evaluations.runtime_config import DATA_ROOT
from shared.metrics import Vocoder, mel_to_audio_hifigan, torch_mel_to_audio, asr_transcribe_np
from shared.metrics import calculate_batch_metrics, calculate_metrics
from AV_PLC.losses import (L1Loss, SpectralConvergenceLoss, CrossEntropyLoss, WhisperASRLoss,
                           MaskedMelReconstructionLoss, UnitPhaseLoss,
                           TemporalPhaseDifferenceLoss, FrequencyPhaseDifferenceLoss,
                           ComplexSpectrumConsistencyLoss, MagnitudeReconstructionLoss,
                           WaveformReconstructionLoss, istft_overlap_add,
                           InstantaneousPhaseLoss, GroupDelayPhaseLoss,
                           InstantaneousAngularFrequencyLoss)

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
      'av' : audio-visual (masked_spec, visual_feats, spk_emb -> rec_spec[, synth_spec])
      'motion': treated like 'a' for criterion choice
    """
    def __init__(
        self,
        model,
        model_name,
        mode,
        sc_loss: bool,
        ce_loss:bool,
        pesq_loss: bool,
        stoi_loss: bool,
        asr_loss: bool,
        enc_loss: bool,
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
        betas=(0.9, 0.98),
        early_stop_patience: int | None = 10,
        phase_reconstruction: bool = False,
        completion_mel_loss: bool = True,
        w_completion_mel: float = 1.0,
        phase_refine_loss: bool = True,
        phase_unit_loss: bool = True,
        phase_temporal_loss: bool = True,
        phase_frequency_loss: bool = True,
        phase_complex_loss: bool = True,
        w_phase_refine: float = 0.10,
        w_phase_unit: float = 0.10,
        w_phase_temporal: float = 0.05,
        w_phase_frequency: float = 0.05,
        w_phase_complex: float = 0.10,
        magnitude_loss: bool = True,
        waveform_loss: bool = True,
        w_magnitude: float = 1.0,
        w_waveform: float = 1.0,
        content_mel_losses: bool = True,
        w_audio_mel: float = 1.0,
        w_video_mel: float = 1.0,
        w_fused_mel: float = 1.0,
        phase_losses_only: bool = False,
        phase_only_training: bool = False,
    ):
        self.model = model
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.mode = mode
        self.drop_av = bool(drop_av)
        self.sc_loss = bool(sc_loss)
        self.ce_loss = bool(ce_loss)
        self.pesq_loss = bool(pesq_loss)
        self.stoi_loss = bool(stoi_loss)
        self.asr_loss = bool(asr_loss)
        self.enc_loss = bool(enc_loss)
        self.phase_reconstruction = bool(phase_reconstruction)
        self.completion_mel_loss = bool(completion_mel_loss)
        self.w_completion_mel = float(w_completion_mel)
        # phase_refine_loss/w_phase_refine are retained only as API compatibility
        # aliases from the old post-Mel residual design.  The new decoder always
        # uses the explicit completion_mel_loss below.
        self.phase_refine_loss = bool(phase_refine_loss)
        self.phase_unit_loss = bool(phase_unit_loss)
        self.phase_temporal_loss = bool(phase_temporal_loss)
        self.phase_frequency_loss = bool(phase_frequency_loss)
        self.phase_complex_loss = bool(phase_complex_loss)
        self.w_phase_refine = float(w_phase_refine)
        self.w_phase_unit = float(w_phase_unit)
        self.w_phase_temporal = float(w_phase_temporal)
        self.w_phase_frequency = float(w_phase_frequency)
        self.w_phase_complex = float(w_phase_complex)
        self.magnitude_loss = bool(magnitude_loss)
        self.waveform_loss = bool(waveform_loss)
        self.w_magnitude = float(w_magnitude)
        self.w_waveform = float(w_waveform)
        self.content_mel_losses = bool(content_mel_losses)
        self.w_audio_mel = float(w_audio_mel)
        self.w_video_mel = float(w_video_mel)
        self.w_fused_mel = float(w_fused_mel)
        self.phase_losses_only = bool(phase_losses_only)
        self.phase_only_training = bool(phase_only_training)
        if self.phase_losses_only and not self.phase_reconstruction:
            raise ValueError("phase_losses_only=True requires phase_reconstruction=True")
        if self.phase_losses_only and not (
            self.completion_mel_loss or self.phase_unit_loss or
            self.phase_temporal_loss or self.phase_frequency_loss or
            self.phase_complex_loss
        ):
            raise ValueError("phase_losses_only=True requires at least one completion/phase loss")
        if self.phase_reconstruction and not getattr(model, 'phase_reconstruction', False):
            raise ValueError('Trainer phase_reconstruction=True requires an AV_PLC model with phase_reconstruction=True')
        if self.phase_only_training:
            if not self.phase_reconstruction:
                raise ValueError("phase_only_training=True requires phase_reconstruction=True")
            if getattr(model, "phase_completion", None) is None:
                raise ValueError("phase_only_training=True requires model.phase_completion")
            if not self.phase_losses_only:
                raise ValueError("parallel magnitude/phase frozen-backbone training requires phase_losses_only=True")
            if self.enc_loss or self.sc_loss or self.ce_loss or self.pesq_loss or self.stoi_loss or self.asr_loss:
                raise ValueError(
                    "phase-only training requires encoder/spectral/perceptual auxiliary losses to be disabled"
                )

        self.train_loader = train_loader
        self.val_loader = val_loader
        if self.phase_reconstruction and getattr(model, "phase_completion", None) is not None:
            mel_mean, mel_std = self._get_dataset_stats(train_loader)
            model.phase_completion.set_mel_stats(mel_mean, mel_std)
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_root = checkpoint_dir
        self.log_root = log_dir
        self.vocoder_path = vocoder_path

        # runtime knobs
        self.grad_clip = grad_clip
        # The parallel M/P branch uses axial Conformer attention on 257-bin STFT
        # grids plus phase/waveform math. In CUDA autocast this can produce NaNs
        # on the first optimizer step, after which phase_completion is poisoned.
        self.mixed_precision = bool(mixed_precision and not self.phase_reconstruction)
        self.amp_dtype = torch.bfloat16 if (use_bf16 and torch.cuda.is_available() and
                                            torch.cuda.get_device_capability()[0] >= 8)\
            else torch.float16

        # dirs + logging
        self.run_dir = os.path.join(self.log_root, f"{model_name}")
        self.ckpt_dir = os.path.join(self.checkpoint_root, f"{model_name}")
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.logger = setup_logging(model_name, self.run_dir)
        if self.phase_reconstruction and mixed_precision:
            self.logger.info(
                "Disabled mixed precision for parallel_mp_v1 to avoid non-finite "
                "TF Conformer/phase losses."
            )
        self.writer = SummaryWriter(log_dir=self.run_dir)

        # loss weights
        self.w_pmsqe = 0.01 if self.pesq_loss else 0.0
        self.w_stoi = 0.01 if self.stoi_loss else 0.0 #from 0.01 -> 0.05
        self.w_asr = 0.1 if self.asr_loss else 0.0
        self.w_synth = 0.1
        self.w_rec = 0.05

        # init components/opt/sched
        self._initialize_components(weight_decay, betas, cosine_Tmax)

        # trackers
        self.best_val_loss = float('inf')
        self.epoch_no_improve = 0
        self.global_step = 0
        self.train_step = 0

        self.log_interval = 100
        self.memory_cleanup_interval = 20
        self.early_stop_patience = early_stop_patience

        # perf hints
        if self.device.type == 'cuda':
            if not torch.backends.cudnn.deterministic:
                torch.backends.cudnn.benchmark = True
            try:
                torch.set_float32_matmul_precision('high')  # PyTorch 2.0+
            except Exception:
                pass
    # --------------------------------------------------------------------------------------------
    def _initialize_components(self, weight_decay, betas, cosine_Tmax):

        self.rec_criterion = L1Loss().to(self.device)

        if self.sc_loss:
            self.synth_sc_weight = 0.05
            self.sc_criterion = SpectralConvergenceLoss().to(self.device)
        else:
            self.synth_sc_weight = 0.0
            self.sc_criterion = None

        if self.ce_loss:
            self.synth_ce_weight = 0.005
            self.ce_criterion = CrossEntropyLoss().to(self.device)
        else:
            self.synth_ce_weight = 0.0
            self.ce_criterion = None

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

        self.completion_mel_criterion = (
            MaskedMelReconstructionLoss().to(self.device)
            if self.completion_mel_loss else None
        )
        if self.phase_reconstruction:
            self.phase_unit_criterion = InstantaneousPhaseLoss().to(self.device)
            self.phase_temporal_criterion = InstantaneousAngularFrequencyLoss().to(self.device)
            self.phase_frequency_criterion = GroupDelayPhaseLoss().to(self.device)
            self.phase_complex_criterion = (
                ComplexSpectrumConsistencyLoss().to(self.device)
                if self.phase_complex_loss else None
            )
            magnitude_compression = float(
                getattr(getattr(self.model, "phase_completion", None), "magnitude_compression", 0.3)
            )
            self.magnitude_criterion = (
                MagnitudeReconstructionLoss(compression=magnitude_compression).to(self.device)
                if self.magnitude_loss else None
            )
            self.waveform_criterion = (
                WaveformReconstructionLoss().to(self.device) if self.waveform_loss else None
            )
        else:
            self.phase_unit_criterion = None
            self.phase_temporal_criterion = None
            self.phase_frequency_criterion = None
            self.phase_complex_criterion = None
            self.magnitude_criterion = None
            self.waveform_criterion = None
        # Deprecated attribute kept so external inspection does not fail.
        self.phase_refine_criterion = self.completion_mel_criterion

        # vocoder
        self.vocoder = Vocoder(self.vocoder_path) if self.vocoder_path is not None else None

        if self.phase_complex_loss or self.pesq_loss:
            mel_fb = torchaudio.functional.melscale_fbanks(
                n_freqs=257, f_min=0.0, f_max=8000.0,
                n_mels=80, sample_rate=16000, norm='slaney', mel_scale='slaney'
            )
            self.mel_pinv = torch.linalg.pinv(mel_fb).float().to(self.device)  # [80, 257]
        else:
            self.mel_pinv = None

        # optimizer + scheduler
        trainable_named = [(name, p) for name, p in self.model.named_parameters() if p.requires_grad]
        if not trainable_named:
            raise ValueError("Model has no trainable parameters")
        if self.phase_only_training:
            bad = [name for name, _ in trainable_named if not name.startswith("phase_completion.")]
            if bad:
                raise ValueError(
                    "phase-only training found trainable non-phase parameters: "
                    + ", ".join(bad[:20])
                )
            if not any(name.startswith("phase_completion.") for name, _ in trainable_named):
                raise ValueError("phase-only training has no trainable phase_completion parameters")
        trainable_params = [p for _, p in trainable_named]
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=weight_decay,
            betas=betas
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cosine_Tmax)

        self.model.to(self.device)

        #self.scaler = torch.amp.GradScaler(enabled=self.mixed_precision)
        self.scaler = torch.amp.GradScaler(
            enabled=(self.mixed_precision and self.amp_dtype == torch.float16)
        )

    # ----------------------- drop modality ------------------------------------------------------
    def modality_dropout_probs(self, epoch):
        if epoch <= 10:
            return [0.85, 0.05, 0.1]
        elif epoch <= 30:
            return [0.70, 0.10, 0.20]
        else:
            return [0.6, 0.15, 0.25]
    # --------------------------------------------------------------------------------------------
    def _move_batch_to_device(self, batch):
        visual_feats = spk_emb = masked_spec = spec = video_aligned_spec = None
        stft_magnitude = video_aligned_magnitude = None
        phase = video_aligned_phase = None
        audio_length = text = mask = path = avail = None
        non_blocking = (self.device.type == 'cuda')

        if self.mode == 'a':
            if self.phase_reconstruction:
                if len(batch) != 8:
                    raise ValueError(f"Expected 8 elements for parallel M/P audio mode, got {len(batch)}")
                masked_spec, spec, stft_magnitude, phase, audio_length, text, mask, path = batch
            else:
                masked_spec, spec, audio_length, text, mask, path = batch

        elif self.mode == 'v':
            if self.phase_reconstruction:
                if len(batch) != 12:
                    raise ValueError(f"Expected 12 elements for parallel M/P video mode, got {len(batch)}")
                (visual_feats, spk_emb, spec, video_aligned_spec,
                 stft_magnitude, video_aligned_magnitude, phase, video_aligned_phase,
                 audio_length, text, mask, path) = batch
            elif len(batch) == 8:
                (visual_feats, spk_emb, spec, video_aligned_spec, audio_length, text, mask, path,) = batch
            elif len(batch) == 7:
                visual_feats, spk_emb, spec, audio_length, text, mask, path = batch
            else:
                raise ValueError(f"Expected 7 or 8 elements for video mode, got {len(batch)}.")

        elif self.mode == 'av':
            if self.phase_reconstruction:
                if len(batch) != 14:
                    raise ValueError(f"Expected 14 elements for parallel M/P AV mode, got {len(batch)}")
                (visual_feats, spk_emb, masked_spec, spec, video_aligned_spec,
                 stft_magnitude, video_aligned_magnitude, phase, video_aligned_phase,
                 audio_length, text, mask, path, avail) = batch
            elif len(batch) == 10:
                (visual_feats, spk_emb, masked_spec, spec, video_aligned_spec,
                 audio_length, text, mask, path, avail) = batch
            elif len(batch) == 9:
                visual_feats, spk_emb, masked_spec, spec, audio_length, text, mask, path, avail = batch
            else:
                visual_feats, spk_emb, masked_spec, spec, audio_length, text, mask, path = batch

        def _to_float(x):
            return x.float().to(self.device, non_blocking=non_blocking) if x is not None else None

        visual_feats = _to_float(visual_feats)
        spk_emb = _to_float(spk_emb)
        masked_spec = _to_float(masked_spec)
        spec = _to_float(spec)
        video_aligned_spec = _to_float(video_aligned_spec)
        stft_magnitude = _to_float(stft_magnitude)
        video_aligned_magnitude = _to_float(video_aligned_magnitude)
        phase = _to_float(phase)
        video_aligned_phase = _to_float(video_aligned_phase)
        if audio_length is not None:
            audio_length = audio_length.long().to(self.device, non_blocking=non_blocking)
        if avail is not None:
            avail = avail.to(self.device, non_blocking=non_blocking).bool()

        return (visual_feats, spk_emb, masked_spec, spec, video_aligned_spec,
                stft_magnitude, video_aligned_magnitude, phase, video_aligned_phase,
                audio_length, text, mask, path, avail)

    # --------------------------------------------------------------------------------------------
    def _forward(self, visual_feats, spk_emb, masked_spec, audio_length,
                 avail=None, audio_mask=None, phase=None, stft_magnitude=None):
        if self.mode == 'a':
            rec, _ = self.model(masked_spec, audio_length)
            return None, rec, None, None
        elif self.mode == 'v':
            synth, _ = self.model(visual_feats, spk_emb, audio_length)
            return None, None, synth, None
        elif self.mode == 'av':
            if audio_mask is not None:
                audio_mask = audio_mask.to(device=masked_spec.device,
                    dtype=masked_spec.dtype, non_blocking=True,)
            out = self.model(masked_spec, visual_feats, spk_emb, audio_length,
                             avail=avail, audio_mask=audio_mask, phase=phase,
                             stft_magnitude=stft_magnitude)
            if isinstance(out, (tuple, list)):
                if len(out) == 4:
                    return out[0], out[1], out[2], out[3]
                if len(out) == 3:
                    return out[0], out[1], out[2], None
            return out, None, None, None
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")


    def _parallel_mp_audio(self, phase_output):
        if phase_output is None or phase_output.get("final_mag") is None:
            return None
        z = torch.complex(
            phase_output["final_mag"].float() * phase_output["final_cos"].float(),
            phase_output["final_mag"].float() * phase_output["final_sin"].float(),
        )
        return istft_overlap_add(z)

    # --------------------------------------------------------------------------------------------
    def _compute_parallel_mp_loss(self, fused_spec, rec_spec, synth_spec, spec, avail,
                                  phase_output, stft_magnitude, phase,
                                  video_aligned_spec=None, video_aligned_magnitude=None,
                                  video_aligned_phase=None):
        """Objective for the parallel magnitude/phase AV-PLC stage.

        Mel heads remain explicit auxiliary/content losses.  The TF reconstructor is
        supervised with magnitude, circular phase (IP/GD/IAF), complex-spectrum,
        and waveform losses.  Observed magnitude/phase are already hard-copied by
        the model, so magnitude/IP/complex supervision is masked to PLC frames;
        GD/IAF use final phase so gap-boundary transitions are included.
        """
        if phase_output is None or stft_magnitude is None or phase is None:
            raise ValueError("Parallel magnitude/phase training requires model output, STFT magnitude and phase")

        loss = spec.new_tensor(0.0)
        parts = {}
        if avail is None:
            avail = torch.ones(spec.size(0), 2, dtype=torch.bool, device=spec.device)
        a_on = avail[:, 0]
        v_on = avail[:, 1]
        both = a_on & v_on
        only_v = (~a_on) & v_on

        # Ground-truth alignment follows the current temporal-jitter target convention.
        target_mel = spec
        target_mag = stft_magnitude
        target_phase = phase
        if only_v.any():
            if video_aligned_spec is not None:
                target_mel = spec.clone()
                target_mel[only_v] = video_aligned_spec[only_v]
            if video_aligned_magnitude is not None:
                target_mag = stft_magnitude.clone()
                target_mag[only_v] = video_aligned_magnitude[only_v]
            if video_aligned_phase is not None:
                target_phase = phase.clone()
                target_phase[only_v] = video_aligned_phase[only_v]

        # Three explicit L1 Mel heads: M_A, M_V, M_AV.
        if self.content_mel_losses and rec_spec is not None and a_on.any():
            value = self.rec_criterion(rec_spec[a_on], spec[a_on])
            parts["audio_mel_l1"] = value
            loss = loss + self.w_audio_mel * value
        else:
            parts["audio_mel_l1"] = spec.new_tensor(0.0)

        if self.content_mel_losses and synth_spec is not None and v_on.any():
            v_target = video_aligned_spec if video_aligned_spec is not None else spec
            value = self.rec_criterion(synth_spec[v_on], v_target[v_on])
            parts["video_mel_l1"] = value
            loss = loss + self.w_video_mel * value
        else:
            parts["video_mel_l1"] = spec.new_tensor(0.0)

        fused_mel = phase_output.get("fused_mel")
        if self.content_mel_losses and fused_mel is not None and both.any():
            value = self.rec_criterion(fused_mel[both], spec[both])
            parts["fused_mel_l1"] = value
            loss = loss + self.w_fused_mel * value
        else:
            parts["fused_mel_l1"] = spec.new_tensor(0.0)

        prediction_mask = phase_output["prediction_mask"].to(spec.dtype)

        if self.magnitude_loss and self.magnitude_criterion is not None:
            value = self.magnitude_criterion(
                phase_output["predicted_mag_compressed"], target_mag, prediction_mask
            )
            parts["magnitude_loss"] = value
            loss = loss + self.w_magnitude * value
        else:
            parts["magnitude_loss"] = spec.new_tensor(0.0)

        # L_phi = L_IP + L_GD + L_IAF.
        if self.phase_unit_loss and self.phase_unit_criterion is not None:
            value = self.phase_unit_criterion(
                phase_output["pred_phase"], target_phase, prediction_mask
            )
            parts["phase_ip_loss"] = value
            loss = loss + self.w_phase_unit * value
        else:
            parts["phase_ip_loss"] = spec.new_tensor(0.0)

        if self.phase_frequency_loss and self.phase_frequency_criterion is not None:
            value = self.phase_frequency_criterion(
                phase_output["final_phase"], target_phase, prediction_mask
            )
            parts["phase_gd_loss"] = value
            loss = loss + self.w_phase_frequency * value
        else:
            parts["phase_gd_loss"] = spec.new_tensor(0.0)

        if self.phase_temporal_loss and self.phase_temporal_criterion is not None:
            value = self.phase_temporal_criterion(
                phase_output["final_phase"], target_phase, prediction_mask
            )
            parts["phase_iaf_loss"] = value
            loss = loss + self.w_phase_temporal * value
        else:
            parts["phase_iaf_loss"] = spec.new_tensor(0.0)

        if self.phase_complex_loss and self.phase_complex_criterion is not None:
            value = self.phase_complex_criterion(
                pred_mag=phase_output["final_mag"],
                pred_cos=phase_output["final_cos"],
                pred_sin=phase_output["final_sin"],
                target_mag=target_mag,
                target_phase=target_phase,
                prediction_mask=prediction_mask,
            )
            parts["complex_loss"] = value
            loss = loss + self.w_phase_complex * value
        else:
            parts["complex_loss"] = spec.new_tensor(0.0)

        if self.waveform_loss and self.waveform_criterion is not None:
            # FFT/OLA is more numerically stable outside mixed precision.
            with torch.amp.autocast('cuda', enabled=False):
                value = self.waveform_criterion(
                    phase_output["final_mag"].float(),
                    phase_output["final_cos"].float(),
                    phase_output["final_sin"].float(),
                    target_mag.float(), target_phase.float(),
                )
            parts["waveform_loss"] = value
            loss = loss + self.w_waveform * value
        else:
            parts["waveform_loss"] = spec.new_tensor(0.0)

        parts["loss"] = loss
        return loss, parts

    # --------------------------------------------------------------------------------------------
    def _compute_loss(self, fused_spec, rec_spec, synth_spec, spec, avail,
                      video_aligned_spec=None, phase_output=None,
                      stft_magnitude=None, video_aligned_magnitude=None,
                      phase=None, video_aligned_phase=None):

        if self.phase_reconstruction:
            return self._compute_parallel_mp_loss(
                fused_spec, rec_spec, synth_spec, spec, avail, phase_output,
                stft_magnitude, phase, video_aligned_spec,
                video_aligned_magnitude, video_aligned_phase,
            )

        loss = spec.new_tensor(0.0)
        parts = {}

        if avail is None:
            B = spec.size(0)
            avail = torch.ones(B, 2, dtype=torch.bool, device=spec.device)

        both = avail[:, 0] & avail[:, 1]
        a_on = avail[:, 0]
        v_on = avail[:, 1]
        only_v = (~a_on) & v_on

        # The unified decoder output is carried in phase_output for API
        # compatibility even when learned phase is disabled.  In the new
        # architecture it is the primary reconstruction path for AV/audio/video.
        completion_output = phase_output

        # Legacy fused L1 is used only for models that do not expose the unified
        # completion output.  Counting it as well would duplicate Mel supervision.
        if completion_output is None and fused_spec is not None and both.any():
            fused_loss = self.rec_criterion(fused_spec[both], spec[both])
            parts['fused_loss'] = fused_loss
            loss = loss + fused_loss
        else:
            parts['fused_loss'] = spec.new_tensor(0.0)

        # Encoder Mel heads are auxiliary only.  They are not inputs to the
        # latent-only spectral decoder.
        _supervise_enc = self.enc_loss or (completion_output is None and fused_spec is None)
        if self.drop_av and not self.enc_loss and completion_output is None:
            raise ValueError(
                "Modality dropout requires either unified completion supervision "
                "or enc_loss=True."
            )

        if _supervise_enc and rec_spec is not None and a_on.any():
            rec_loss = self.rec_criterion(rec_spec[a_on], spec[a_on])
            parts['rec_loss'] = rec_loss
            rec_w = 0.3 if fused_spec is not None else 1.0
            loss = loss + rec_w * rec_loss
        else:
            parts['rec_loss'] = spec.new_tensor(0.0)

        if _supervise_enc and synth_spec is not None and v_on.any():
            synth_w = 0.1 if fused_spec is not None else 1.0
            synth_pred = synth_spec[v_on]
            target_all = video_aligned_spec if video_aligned_spec is not None else spec
            synth_target = target_all[v_on]

            synth_loss = self.rec_criterion(synth_pred, synth_target)
            parts["synth_loss"] = synth_loss
            loss = loss + synth_w * synth_loss

            if self.synth_sc_weight > 0.0 and self.sc_criterion is not None:
                synth_sc_loss = self.sc_criterion(synth_pred.float(), synth_target.float())
                parts["synth_sc_loss"] = synth_sc_loss
                loss = loss + synth_w * self.synth_sc_weight * synth_sc_loss
            else:
                parts["synth_sc_loss"] = spec.new_tensor(0.0)

            if self.synth_ce_weight > 0.0 and self.ce_criterion is not None:
                synth_ce_loss = self.ce_criterion(synth_pred.float(), synth_target.float())
                parts["synth_ce_loss"] = synth_ce_loss
                loss = loss + synth_w * self.synth_ce_weight * synth_ce_loss
            else:
                parts["synth_ce_loss"] = spec.new_tensor(0.0)
        else:
            parts["synth_loss"] = spec.new_tensor(0.0)
            parts["synth_sc_loss"] = spec.new_tensor(0.0)
            parts["synth_ce_loss"] = spec.new_tensor(0.0)

        completion_weighted_loss = spec.new_tensor(0.0)
        phase_weighted_loss = spec.new_tensor(0.0)

        target_mel = spec
        if only_v.any() and video_aligned_spec is not None:
            target_mel = spec.clone()
            target_mel[only_v] = video_aligned_spec[only_v]

        if completion_output is not None:
            prediction_mask = completion_output["prediction_mask"].to(spec.dtype)
            if self.completion_mel_loss and self.completion_mel_criterion is not None:
                value = self.completion_mel_criterion(
                    completion_output["predicted_mel"], target_mel, prediction_mask
                )
                parts["completion_mel_loss"] = value
                weighted = self.w_completion_mel * value
                loss = loss + weighted
                completion_weighted_loss = completion_weighted_loss + weighted
            else:
                parts["completion_mel_loss"] = spec.new_tensor(0.0)
        else:
            prediction_mask = None
            parts["completion_mel_loss"] = spec.new_tensor(0.0)

        phase_keys = (
            "phase_unit_loss",
            "phase_temporal_loss",
            "phase_frequency_loss",
            "phase_complex_loss",
        )
        if self.phase_reconstruction:
            if completion_output is None or phase is None:
                raise ValueError(
                    "Phase reconstruction is enabled but completion outputs/phase targets are missing"
                )
            if completion_output.get("pred_cos") is None:
                raise ValueError("Phase reconstruction is enabled but predicted phase is missing")

            target_phase = phase
            if only_v.any() and video_aligned_phase is not None:
                target_phase = phase.clone()
                target_phase[only_v] = video_aligned_phase[only_v]

            if self.phase_unit_loss and self.phase_unit_criterion is not None:
                value = self.phase_unit_criterion(
                    completion_output["pred_cos"], completion_output["pred_sin"],
                    target_phase, prediction_mask
                )
                parts["phase_unit_loss"] = value
                weighted = self.w_phase_unit * value
                loss = loss + weighted
                phase_weighted_loss = phase_weighted_loss + weighted
            else:
                parts["phase_unit_loss"] = spec.new_tensor(0.0)

            if self.phase_temporal_loss and self.phase_temporal_criterion is not None:
                value = self.phase_temporal_criterion(
                    completion_output["final_cos"], completion_output["final_sin"],
                    target_phase, prediction_mask
                )
                parts["phase_temporal_loss"] = value
                weighted = self.w_phase_temporal * value
                loss = loss + weighted
                phase_weighted_loss = phase_weighted_loss + weighted
            else:
                parts["phase_temporal_loss"] = spec.new_tensor(0.0)

            if self.phase_frequency_loss and self.phase_frequency_criterion is not None:
                value = self.phase_frequency_criterion(
                    completion_output["final_cos"], completion_output["final_sin"],
                    target_phase, prediction_mask
                )
                parts["phase_frequency_loss"] = value
                weighted = self.w_phase_frequency * value
                loss = loss + weighted
                phase_weighted_loss = phase_weighted_loss + weighted
            else:
                parts["phase_frequency_loss"] = spec.new_tensor(0.0)

            # Couple the predicted Mel magnitude and predicted phase in the same
            # 257-bin linear-frequency domain.  This is a projected complex-STFT
            # consistency objective because the magnitude is obtained through the
            # fixed pseudo-inverse Mel mapping already used elsewhere in AV_PLC.
            if self.phase_complex_loss and self.phase_complex_criterion is not None:
                with torch.amp.autocast('cuda', enabled=False):
                    pred_power = mel_to_power_linear(
                        completion_output["predicted_mel"].float(), self.mel_pinv
                    )
                    target_power = mel_to_power_linear(
                        target_mel.float(), self.mel_pinv
                    )
                    pred_mag = torch.sqrt(pred_power.clamp_min(1e-8))
                    target_mag = torch.sqrt(target_power.clamp_min(1e-8))

                    value = self.phase_complex_criterion(
                        pred_mag=pred_mag,
                        pred_cos=completion_output["pred_cos"].float(),
                        pred_sin=completion_output["pred_sin"].float(),
                        target_mag=target_mag,
                        target_phase=target_phase.float(),
                        prediction_mask=prediction_mask.float(),
                    )

                parts["phase_complex_loss"] = value
                weighted = self.w_phase_complex * value
                loss = loss + weighted
                phase_weighted_loss = phase_weighted_loss + weighted
            else:
                parts["phase_complex_loss"] = spec.new_tensor(0.0)
        else:
            for key in phase_keys:
                parts[key] = spec.new_tensor(0.0)

        # Preserve the old key in logs as a zero-valued compatibility field.
        parts["phase_refine_loss"] = spec.new_tensor(0.0)

        # Perceptual losses remain AV-only.  fused_spec now contains the unified
        # decoder prediction for AV samples, so existing behavior remains valid.
        groups = [('fused', fused_spec, both, 1.0)]
        pmsqe_accum = spec.new_tensor(0.0)
        asr_accum = spec.new_tensor(0.0)
        w_total = 0.0

        with torch.amp.autocast('cuda', enabled=False):
            for _, head_output, idx, w in groups:
                if head_output is None or not idx.any():
                    continue
                p = head_output[idx].float()
                r = spec[idx].float()
                if self.pesq_loss and self.pmsqe is not None:
                    pow_ref = mel_to_power_linear(r, self.mel_pinv).permute(0, 2, 1).contiguous()
                    pow_est = mel_to_power_linear(p, self.mel_pinv).permute(0, 2, 1).contiguous()
                    pmsqe_accum = pmsqe_accum + w * torch.mean(self.pmsqe(pow_est, pow_ref))
                if self.asr_loss and self.asr_criterion is not None:
                    asr_accum = asr_accum + w * self.asr_criterion(r, p)
                w_total += w

        if w_total > 0:
            if self.pesq_loss and self.pmsqe is not None:
                pmsqe_loss = pmsqe_accum / w_total
                parts['pmsqe_loss'] = pmsqe_loss
                loss = loss + self.w_pmsqe * pmsqe_loss
            else:
                parts['pmsqe_loss'] = None
            if self.asr_loss and self.asr_criterion is not None:
                asr_loss = asr_accum / w_total
                parts['asr_loss'] = asr_loss
                loss = loss + self.w_asr * asr_loss
            else:
                parts['asr_loss'] = None

        if self.phase_losses_only:
            # Restrict checkpoint/training objective to explicitly enabled completion
            # and phase terms.  In the older phase-only experiment completion Mel is
            # disabled, so this becomes exactly Lu + Lt + Lf (weighted).
            loss = completion_weighted_loss + phase_weighted_loss

        parts['loss'] = loss
        return loss, parts

    # ---------------------------------------- train epoch ---------------------------------------
    def _train_epoch(self, epoch, num_epochs):

        n_batches = 0
        running_loss = 0.0
        acc_parts = defaultdict(list)
        self.model.train()

        if self.drop_av:
            new_mode_probs = self.modality_dropout_probs(epoch)
            train_dataset = self.train_loader.dataset

            while isinstance(train_dataset, torch.utils.data.Subset):
                train_dataset = train_dataset.dataset

            train_dataset.modality_dropout.mode_probs = new_mode_probs
            self.logger.info(f"New dropout modality probabilities: {new_mode_probs}")

        iterator = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]",
                        leave=False, unit="batch")

        for bidx, batch in enumerate(iterator):
            (visual_feats, spk_emb, masked_spec, spec, video_aligned_spec,
             stft_magnitude, video_aligned_magnitude, phase, video_aligned_phase,
             audio_length, text, mask, path, avail) = self._move_batch_to_device(batch)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=self.amp_dtype, enabled=self.mixed_precision):
                fused_spec, rec_spec, synth_spec, phase_output = self._forward(
                    visual_feats, spk_emb, masked_spec, audio_length, avail, audio_mask=mask,
                    phase=phase, stft_magnitude=stft_magnitude)
                loss, parts = self._compute_loss(
                    fused_spec, rec_spec, synth_spec, spec, avail,
                    video_aligned_spec=video_aligned_spec, phase_output=phase_output,
                    stft_magnitude=stft_magnitude, video_aligned_magnitude=video_aligned_magnitude,
                    phase=phase, video_aligned_phase=video_aligned_phase)

            if not torch.isfinite(loss):
                bad_parts = [
                    name for name, value in parts.items()
                    if torch.is_tensor(value) and not torch.isfinite(value).all()
                ]
                bad_outputs = []
                if isinstance(phase_output, dict):
                    for name, value in phase_output.items():
                        if (
                            torch.is_tensor(value)
                            and value.is_floating_point()
                            and not torch.isfinite(value).all()
                        ):
                            bad_outputs.append(name)
                raise FloatingPointError(
                    "Non-finite training loss before backward at "
                    f"epoch={epoch + 1}, batch={bidx}, paths={path[:2] if path is not None else None}. "
                    f"Bad loss components={bad_parts}; bad phase outputs={bad_outputs}."
                )

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

            for k, v in parts.items():
                if v is not None:
                    acc_parts[k].append(v.item() if torch.is_tensor(v) else float(v))

        # ---- epoch-averaged loss dict ----
        avg_loss = running_loss / max(1, n_batches)
        parts_avg = {'loss': avg_loss}
        for k, vals in acc_parts.items():
            if k != 'loss':
                parts_avg[k] = float(np.mean(vals)) if vals else None

        self._log_loss_components(parts_avg, False, phase="train", epoch=epoch + 1)
        self.writer.flush()

        return avg_loss
    # --------------------------------------- validate epoch -------------------------------------
    @torch.inference_mode()
    def _validate_epoch(self, epoch, num_epochs):
        if self.val_loader is None:
            return None

        self.model.eval()
        iterator = tqdm(self.val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Valid]",
                        leave=False, unit="batch", mininterval=0.5)
        n_collected = 0
        target_metric_samples = 6
        acc_parts = defaultdict(list)

        batch_losses = []
        specs_accum, res_accum = [], []
        phase_audio_accum = []
        mask_accum, path_accum = [], []
        gt_audio_accum, gt_ref_text = [], []
        mel_mean, mel_std = self._get_dataset_stats(self.val_loader)

        for bidx, batch in enumerate(iterator):
            (visual_feats, spk_emb, masked_spec, spec, video_aligned_spec,
             stft_magnitude, video_aligned_magnitude, phase, video_aligned_phase,
             audio_length, text, mask, path, avail) = self._move_batch_to_device(batch)
            fused_spec, rec_spec, synth_spec, phase_output = self._forward(
                visual_feats, spk_emb, masked_spec, audio_length, avail, audio_mask=mask,
                phase=phase, stft_magnitude=stft_magnitude)
            loss, parts = self._compute_loss(
                fused_spec, rec_spec, synth_spec, spec, avail,
                video_aligned_spec=video_aligned_spec, phase_output=phase_output,
                stft_magnitude=stft_magnitude, video_aligned_magnitude=video_aligned_magnitude,
                phase=phase, video_aligned_phase=video_aligned_phase)

            batch_losses.append(loss.item())
            if (bidx % 10) == 0:
                iterator.set_postfix({"loss": f"{loss.item():.4f}"})

            for k, v in parts.items():
                if v is not None and k != 'loss':
                    acc_parts[k].append(v.item())

            res = fused_spec if fused_spec is not None else rec_spec if rec_spec is not None else synth_spec
            metric_res = phase_output["predicted_mel"] if phase_output is not None else res
            remaining = target_metric_samples - n_collected
            if remaining > 0 and metric_res is not None:
                take = min(remaining, spec.shape[0])

                specs_accum.append(spec[:take].detach().cpu())
                res_accum.append(metric_res[:take].detach().cpu())
                if phase_output is not None and phase_output.get("final_cos") is not None:
                    phase_audio = self._parallel_mp_audio({k: (v[:take] if torch.is_tensor(v) else v) for k, v in phase_output.items()})
                    phase_audio_accum.append(phase_audio.detach().cpu())
                path_accum.extend(path[:take])
                gt_ref_text.extend(text[:take])
                mask_accum.extend(mask[:take].detach().cpu())

                n_collected += take

        metrics = {'mse': 0.0, 'psnr': 0.0, 'pesq': 0.0, 'stoi': 0.0, 'estoi':0.0,
                   'plcmos': 0.0, 'cer': 0.0, 'wer': 0.0}

        if specs_accum and len(res_accum) > 0:
            specs_cat = torch.cat(specs_accum, dim=0)
            fused_cat = torch.cat(res_accum, dim=0)
            mask_cat = torch.cat(mask_accum, dim=0)

            metrics = calculate_batch_metrics(original_batch=specs_cat,
                                              reconstructed_batch=fused_cat,
                                              texts=gt_ref_text,
                                              mask=mask_cat,
                                              path=path_accum,
                                              hifigan_vocoder=self.vocoder,
                                              tokenizer=None,
                                              max_samples=len(specs_cat),
                                              sample_rate=self.sample_rate,
                                              mel_mean=mel_mean, mel_std=mel_std,
                                              reconstructed_audio_batch=(
                                                  torch.cat(phase_audio_accum, dim=0)
                                                  if phase_audio_accum else None))

        metrics['loss'] = float(np.mean(batch_losses)) if batch_losses else 0.0

        parts_avg = {'loss': metrics['loss']}
        for k, vals in acc_parts.items():
            parts_avg[k] = float(np.mean(vals)) if vals else None

        self._log_loss_components(parts_avg, metrics, phase="validation", epoch=epoch + 1)
        self.writer.flush()
        return metrics
    # ---------------------------------------- logging loss --------------------------------------
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
        for k in [
            'fused_loss',
            'rec_loss',
            'synth_loss',
            'synth_sc_loss',
            'synth_ce_loss',
            'pmsqe_loss',
            'stoi_loss',
            'asr_loss',
            'completion_mel_loss',
            'phase_refine_loss',
            'phase_unit_loss',
            'phase_temporal_loss',
            'phase_frequency_loss',
            'phase_complex_loss',
            'audio_mel_l1',
            'video_mel_l1',
            'fused_mel_l1',
            'magnitude_loss',
            'phase_ip_loss',
            'phase_gd_loss',
            'phase_iaf_loss',
            'complex_loss',
            'waveform_loss',
        ]:
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
    # ---------------------------------------- plotting results -----------------------------------
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

        (visual_feats, spk_emb, masked_spec, spec, video_aligned_spec,
         stft_magnitude, video_aligned_magnitude, phase, video_aligned_phase,
         audio_length, text, mask, path, avail) = self._move_batch_to_device(batch)

        if spec is not None:
            spec = spec[:num_samples]
        if masked_spec is not None:
            masked_spec = masked_spec[:num_samples]
        if visual_feats is not None:
            visual_feats = visual_feats[:num_samples]
        if spk_emb is not None:
            spk_emb = spk_emb[:num_samples]
        if avail is not None:
            avail = avail[:num_samples]
        if audio_length is not None:
            audio_length = audio_length[:num_samples]
        if mask is not None:
            mask = mask[:num_samples]
        if stft_magnitude is not None:
            stft_magnitude = stft_magnitude[:num_samples]
        if video_aligned_magnitude is not None:
            video_aligned_magnitude = video_aligned_magnitude[:num_samples]
        if phase is not None:
            phase = phase[:num_samples]
        if video_aligned_phase is not None:
            video_aligned_phase = video_aligned_phase[:num_samples]

        fused_spec, rec_spec, synth_spec, phase_output = self._forward(visual_feats, spk_emb, masked_spec,
                                                         audio_length, avail, audio_mask=mask, phase=phase,
                                                         stft_magnitude=stft_magnitude)

        if phase_output is not None:
            fused_spec = phase_output["predicted_mel"][:num_samples]
        elif fused_spec is not None:
            fused_spec = fused_spec[:num_samples]

        if not self.enc_loss:
             rec_spec = None
             synth_spec = None
        else:
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
        fig, axes = plt.subplots(n * 2, cols, figsize=(w * cols, h * n), squeeze=False)

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

        plt.tight_layout()
        fig_path = os.path.join(plot_dir, f'comparison_epoch{epoch}_{timestamp}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def _plot_spectrogram_row(self, axes, masked_spec ,fused_spec, rec_spec, orig_spec, synth_spec, sample_idx, fig):

        col = 0
        plt.rcParams.update({'font.size': 15})

        if masked_spec is not None:
            axes[col].imshow(masked_spec, aspect='auto',
                                   origin='lower', interpolation='none')
            axes[col].set_title(f'Input')
            col += 1

        if fused_spec is not None:
            axes[col].imshow(fused_spec, aspect='auto', origin='lower', interpolation='none')
            axes[col].set_title(f'Fused')
            col += 1

        if rec_spec is not None:
            axes[col].imshow(rec_spec, aspect='auto', origin='lower', interpolation='none')
            axes[col].set_title(f'Reconstructed')
            col += 1

        if synth_spec is not None:
            axes[col].imshow(synth_spec, aspect='auto', origin='lower', interpolation='none')
            axes[col].set_title(f'Synthesized')
            col += 1

        axes[col].imshow(orig_spec, aspect='auto', origin='lower', interpolation='none')
        axes[col].set_title(f'Original')

        height, width = orig_spec.shape

        for ax in axes:
            ax.set_xlim(0, width)
            ax.set_ylim(0, height)

            ax.set_xticks(np.linspace(0, width, 5))
            ax.set_yticks(np.linspace(0, height, 5))

            ax.ticklabel_format(axis='both', style='plain')
            ax.tick_params(axis='both', which='both', labelsize=12)

            ax.set_aspect('auto')

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
            col += 1

        if fused_audio is not None:
            t_fused = np.arange(len(fused_audio)) / self.sample_rate
            axes[col].plot(t_fused, _to_float_m1p1(fused_audio))
            col += 1

        if rec_audio is not None:
            t_rec = np.arange(len(rec_audio)) / self.sample_rate
            axes[col].plot(t_rec, _to_float_m1p1(rec_audio))
            col += 1

        if synth_audio is not None:
            t_s = np.arange(len(synth_audio)) / self.sample_rate
            axes[col].plot(t_s, _to_float_m1p1(synth_audio))
            col += 1

        # normalize original audio
        orig_audio_f = _to_float_m1p1(orig_audio)
        t_o = np.arange(len(orig_audio_f)) / self.sample_rate
        axes[col].plot(t_o, orig_audio_f)

        max_len = max(len(x) for x in [
            a for a in [masked_audio, fused_audio, rec_audio, synth_audio, orig_audio_f] if a is not None
        ])
        time_max = max_len / self.sample_rate
        min_amp = min(
            _to_float_m1p1(a).min() for a in [masked_audio, fused_audio, rec_audio, synth_audio, orig_audio_f] if
            a is not None)
        max_amp = max(
            _to_float_m1p1(a).max() for a in [masked_audio, fused_audio, rec_audio, synth_audio, orig_audio_f] if
            a is not None)

        for ax in axes:
            ax.set_xlim(0, time_max)
            ax.set_ylim(min_amp, max_amp)
            ax.set_xticks(np.round(np.linspace(0, time_max, int(time_max) + 1)).astype(int))

            ax.set_yticks(np.linspace(min_amp, max_amp, 5))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            ax.tick_params(axis='both', which='both', labelsize=12)
    # ---------------------------------------- checkpoints ---------------------------------------
    def _save_checkpoint(self, epoch, is_best=False):

        payload = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
            'scaler_state_dict': self.scaler.state_dict() if hasattr(self, 'scaler') else None,
            'best_val_loss': getattr(self, 'best_val_loss', float('inf')),
            'global_step': getattr(self, 'global_step', 0),
            'train_step': getattr(self, 'train_step', 0),
            'metrics': getattr(self, 'metrics', None),
            'epoch_no_improve': getattr(self, 'epoch_no_improve', 0),
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
                except Exception as e:
                    # scheduler shape might differ across runs; warn but continue
                    self.logger.warning(f"Scheduler state couldn't be restored: {e}; continuing with current scheduler.")

            if 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            # optional extras
            self.best_val_loss = checkpoint.get('best_val_loss', getattr(self, 'best_val_loss', float('inf')))
            self.global_step = checkpoint.get('global_step', getattr(self, 'global_step', 0))
            self.train_step = checkpoint.get('train_step', getattr(self, 'train_step', 0))
            self.epoch_no_improve = checkpoint.get('epoch_no_improve', 0)
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
    # --------------------------------------- training loop ---------------------------------------
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
                    self.epoch_no_improve = 0
                else:
                    self.epoch_no_improve += 1

                if self.early_stop_patience and self.epoch_no_improve >= self.early_stop_patience:
                    self.logger.info(
                        f"Early stopping at epoch {epoch + 1} due to no improvement "
                        f"in validation loss for {self.early_stop_patience} epochs.")
                    break

            if (epoch + 1) % save_interval == 0:
                self._save_checkpoint(epoch, is_best=False)

            elapsed = time.time() - epoch_start

            postfix = f"Train Loss: {train_loss:.4f}"
            if val_metrics:
                postfix += f", Val Loss: {val_metrics['loss']:.4f}"
            postfix += f", LR: {current_lr:.9f}"
            postfix += f", elapsed time: {elapsed:.4f}"
            epoch_iter.set_postfix_str(postfix)

            self.writer.add_scalar('Train/epoch_loss', train_loss, epoch+1)
            self.writer.add_scalar('Train/epoch_lr', current_lr, epoch+1)

        total_time = time.time() - total_start
        self.logger.info(f"Training completed in {total_time/3600:.2f} h")

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

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        selected_sample_paths = None
        if sample_paths:
            selected_sample_paths = {self._relative_dataset_path(p) for p in sample_paths}
        saved_sample_paths = set()

        condition_name = f"ge_{loss_rate}" if mask_type == "gilbert" else f"gap_{int(gap_ms)}ms"
        eval_dir = os.path.join(self.run_dir, f"test_{condition_name}")
        os.makedirs(eval_dir, exist_ok=True)
        audio_dir = os.path.join(eval_dir, "audio_samples") if save_output else None
        plot_dir = os.path.join(eval_dir, "plots") if save_output else None
        spec_dir = os.path.join(eval_dir, "spectrograms") if save_output else None
        if save_output:
            for directory in (audio_dir, plot_dir, spec_dir):
                os.makedirs(directory, exist_ok=True)

        if not skip_load:
            load_path = checkpoint_path or os.path.join(self.ckpt_dir, "best_model.pt")
            if not os.path.isfile(load_path):
                raise FileNotFoundError(load_path)
            self.load_checkpoint(load_path)

        mel_mean, mel_std = self._get_dataset_stats(test_loader)
        self.model.eval()
        self.logger.info(f"Evaluating {condition_name} with mel_mean={mel_mean}, mel_std={mel_std}")

        model_chunks, masked_chunks = [], []
        specs_accum, head_accum, masked_accum, masks_accum = [], [], [], []
        phase_audio_accum = []
        texts_accum, paths_accum = [], []
        sample_id, chunk_batches = 0, 20
        head_name = "rec" if self.mode == "a" else "synth" if self.mode == "v" else "fused"

        for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Testing {condition_name}", unit="batch")):
            (visual_feats, spk_emb, masked_spec, spec, video_aligned_spec,
             stft_magnitude, video_aligned_magnitude, phase, video_aligned_phase,
             audio_length, text, mask, path, avail) = self._move_batch_to_device(batch)

            fused_spec, rec_spec, synth_spec, phase_output = self._forward(
                visual_feats, spk_emb, masked_spec, audio_length, avail, audio_mask=mask,
                phase=phase, stft_magnitude=stft_magnitude)
            head_spec = rec_spec if self.mode == "a" else synth_spec if self.mode == "v" else fused_spec
            if head_spec is None:
                raise RuntimeError(f"No evaluation output for mode={self.mode}")
            metric_head_spec = phase_output["predicted_mel"] if phase_output is not None else head_spec
            phase_audio = None
            if phase_output is not None and phase_output.get("final_cos") is not None:
                phase_audio = self._parallel_mp_audio({k: (v if torch.is_tensor(v) else v) for k, v in phase_output.items()}).detach().cpu()

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

                    def _select_optional(tensor):
                        if tensor is None:
                            return None
                        return tensor.index_select(0, idx_spec.to(tensor.device))

                    selected_phase_output = None
                    if phase_output is not None:
                        selected_phase_output = {
                            key: _select_optional(value) if torch.is_tensor(value) else value
                            for key, value in phase_output.items()
                        }
                    sample_id = self._write_outputs(
                        spec.index_select(0, idx_spec),
                        masked_spec.index_select(0, idx_spec.to(masked_spec.device)),
                        _select_optional(fused_spec),
                        _select_optional(rec_spec),
                        _select_optional(synth_spec),
                        mask.index_select(0, idx_mask),
                        selected_paths, sample_id, timestamp, mel_mean, mel_std,
                        phase_output=selected_phase_output,
                        audio_dir=audio_dir, spec_dir=spec_dir, plot_dir=plot_dir)
                    saved_sample_paths.update(
                        self._relative_dataset_path(p) for p in selected_paths)
            if not save_metrics:
                continue

            specs_accum.append(spec.detach().cpu())
            head_accum.append(metric_head_spec.detach().cpu())
            if phase_audio is not None:
                phase_audio_accum.append(phase_audio)
            masked_accum.append(masked_spec.detach().cpu())
            masks_accum.append(mask.detach().cpu())
            texts_accum.append(text)
            paths_accum.extend(list(path))

            flush = ((batch_idx + 1) % chunk_batches == 0 or batch_idx + 1 == len(test_loader))
            if flush:
                originals = torch.cat(specs_accum, dim=0)
                heads = torch.cat(head_accum, dim=0)
                masked_inputs = torch.cat(masked_accum, dim=0)
                masks = torch.cat(masks_accum, dim=0)
                texts = torch.cat(texts_accum, dim=0)
                n_samples = originals.size(0)
                common = dict(
                    original_batch=originals, texts=texts, path=list(paths_accum), mask=masks,
                    hifigan_vocoder=self.vocoder, tokenizer=None, max_samples=n_samples,
                    sample_rate=self.sample_rate, mel_mean=mel_mean, mel_std=mel_std)
                model_chunks.append((calculate_batch_metrics(
                    reconstructed_batch=heads, masked_input=False,
                    reconstructed_audio_batch=(torch.cat(phase_audio_accum, dim=0)
                                               if phase_audio_accum else None),
                    **common), n_samples))
                masked_chunks.append((calculate_batch_metrics(
                    reconstructed_batch=masked_inputs, masked_input=True, **common), n_samples))
                specs_accum.clear(); head_accum.clear(); masked_accum.clear()
                phase_audio_accum.clear()
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
        fieldnames = ["mask_type", "loss_rate", "gap_ms", "output"] + all_keys
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            baseline_row = {
                "mask_type": mask_type,
                "loss_rate": loss_rate if mask_type == "gilbert" else "",
                "gap_ms": gap_ms if mask_type == "single_gap" else "",
                "output": "masked_input"}
            baseline_row.update(masked_metrics)
            writer.writerow(baseline_row)
            model_row = {
                "mask_type": mask_type,
                "loss_rate": loss_rate if mask_type == "gilbert" else "",
                "gap_ms": gap_ms if mask_type == "single_gap" else "",
                "output": head_name}
            model_row.update(model_metrics)
            writer.writerow(model_row)
        self.logger.info(f"Saved metrics CSV to {csv_path}")
        return {"masked_input": masked_metrics, head_name: model_metrics, "csv_path": csv_path}

    @torch.no_grad()
    def evaluate_samples(self, test_loader, loss_rate):

        total_loss = 0.0

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
                    (visual_feats, spk_emb, masked_spec, spec, video_aligned_spec,
                     stft_magnitude, video_aligned_magnitude, phase, video_aligned_phase,
                     audio_length, text, mask, path, avail) = self._move_batch_to_device(batch)
                    fused_spec, rec_spec, synth_spec, phase_output = self._forward(visual_feats, spk_emb,
                                                                     masked_spec, audio_length,
                                                                     avail, audio_mask=mask, phase=phase,
                                                                     stft_magnitude=stft_magnitude)
                    metric_spec = phase_output["predicted_mel"] if phase_output is not None else fused_spec
                    phase_audio_batch = None
                    if phase_output is not None and phase_output.get("final_cos") is not None:
                        mel_mean, mel_std = self._get_dataset_stats(test_loader)
                        phase_audio_batch = self._parallel_mp_audio({k: (v if torch.is_tensor(v) else v) for k, v in phase_output.items()}).detach().cpu()
                    for i in range(len(spec)):
                        sample_id += 1
                        sample_metrics = calculate_metrics(original_spec=spec[i].detach().cpu(),
                                                           reconstructed_spec=metric_spec[i].detach().cpu() if metric_spec is not None else None,
                                                           path=path[i],
                                                           mask=mask[i].detach().cpu(),
                                                           text=text[i],
                                                           hifigan_vocoder=self.vocoder,
                                                           reconstructed_audio=(phase_audio_batch[i] if phase_audio_batch is not None else None))
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
    def evaluate_ablation(
        self,
        test_loader,
        modes=("av", "audio", "video"),
        loss_rate=None,
        mask_type="gilbert",
        gap_ms=None,
        checkpoint_path=None,
        skip_load=False,
    ):
        """Evaluate modality availability under one masking condition.

        For ``video_only`` the synthesized video output is evaluated once over
        the full clip because it does not depend on the masked audio input.
        For ``gilbert`` and ``single_gap`` conditions, AV/audio outputs are
        evaluated by inserting only the reconstructed region into the original
        waveform through ``calculate_batch_metrics(..., mask=mask)``.
        """
        valid_modes = {"av", "audio", "video"}
        modes = tuple(modes)
        unknown = set(modes) - valid_modes
        if unknown:
            raise ValueError(f"Unsupported ablation modes: {sorted(unknown)}")

        if mask_type not in {"gilbert", "single_gap", "video_only"}:
            raise ValueError(f"Unsupported mask_type: {mask_type}")
        if mask_type == "gilbert" and loss_rate is None:
            raise ValueError("loss_rate is required for Gilbert-Elliott ablation")
        if mask_type == "single_gap" and gap_ms is None:
            raise ValueError("gap_ms is required for single-gap ablation")
        if mask_type == "video_only" and modes != ("video",):
            raise ValueError("video_only evaluation must use modes=('video',)")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if mask_type == "gilbert":
            condition_name = f"ge_{loss_rate}"
        elif mask_type == "single_gap":
            condition_name = f"gap_{int(gap_ms)}ms"
        else:
            condition_name = "video_only"

        eval_dir = os.path.join(self.run_dir, f"modality_ablation_{condition_name}")
        os.makedirs(eval_dir, exist_ok=True)
        csv_path = os.path.join(eval_dir, f"modality_metrics_{timestamp}.csv")

        if not skip_load:
            load_path = checkpoint_path or os.path.join(self.ckpt_dir, "best_model.pt")
            if not os.path.isfile(load_path):
                raise FileNotFoundError(load_path)
            self.load_checkpoint(load_path)

        mel_mean, mel_std = self._get_dataset_stats(test_loader)
        self.model.eval()

        availability = {
            "audio": (True, False),
            "av": (True, True),
            "video": (False, True),
        }
        include_masked_input = mask_type != "video_only"
        output_modes = (["masked_input"] if include_masked_input else []) + list(modes)
        summary = {name: defaultdict(list) for name in output_modes}

        for batch in tqdm(
            test_loader,
            desc=f"Testing modality ablation ({condition_name})",
            unit="batch",
        ):
            (visual_feats, spk_emb, masked_spec, spec, video_aligned_spec,
             stft_magnitude, video_aligned_magnitude, phase, video_aligned_phase,
             audio_length, text, mask, path, avail) = self._move_batch_to_device(batch)
            batch_size = spec.size(0)
            spec_cpu = spec.detach().cpu()
            mask_cpu = mask.detach().cpu()

            common = dict(
                original_batch=spec_cpu,
                texts=text,
                path=list(path),
                hifigan_vocoder=self.vocoder,
                tokenizer=None,
                max_samples=batch_size,
                sample_rate=self.sample_rate,
                mel_mean=mel_mean,
                mel_std=mel_std,
            )

            if include_masked_input:
                metrics = calculate_batch_metrics(
                    reconstructed_batch=masked_spec.detach().cpu(),
                    mask=mask_cpu,
                    masked_input=True,
                    **common,
                )
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        summary["masked_input"][key].append((float(value), batch_size))

            for mode_name in modes:
                audio_on, video_on = availability[mode_name]
                forced_avail = torch.tensor(
                    [audio_on, video_on], device=self.device, dtype=torch.bool
                ).unsqueeze(0).repeat(batch_size, 1)

                fused_spec, rec_spec, synth_spec, phase_output = self._forward(
                    visual_feats,
                    spk_emb,
                    masked_spec,
                    audio_length,
                    avail=forced_avail,
                    audio_mask=mask,
                    phase=phase,
                    stft_magnitude=stft_magnitude,
                )
                primary_spec = (
                    fused_spec if mode_name == "av"
                    else rec_spec if mode_name == "audio"
                    else synth_spec
                )
                if primary_spec is None:
                    continue
                metric_spec = phase_output["predicted_mel"] if phase_output is not None else primary_spec
                phase_audio = None
                if phase_output is not None and phase_output.get("final_cos") is not None:
                    phase_audio = self._parallel_mp_audio({k: (v if torch.is_tensor(v) else v) for k, v in phase_output.items()}).detach().cpu()

                # AV/audio: insert only the missing region. Video-only is a
                # full speech-synthesis result and is therefore evaluated once
                # over the complete waveform with no gap mask.
                metric_mask = None if mask_type == "video_only" else mask_cpu
                metrics = calculate_batch_metrics(
                    reconstructed_batch=metric_spec.detach().cpu(),
                    reconstructed_audio_batch=phase_audio,
                    mask=metric_mask,
                    masked_input=False,
                    **common,
                )
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        summary[mode_name][key].append((float(value), batch_size))

        final = {}
        for mode_name, metrics_dict in summary.items():
            final[mode_name] = {}
            for key, values in metrics_dict.items():
                weighted_sum = sum(value * weight for value, weight in values)
                total_weight = sum(weight for _, weight in values)
                if total_weight > 0:
                    final[mode_name][key] = weighted_sum / total_weight

        all_keys = sorted(set().union(*(set(v.keys()) for v in final.values())))
        fieldnames = ["mask_type", "loss_rate", "gap_ms", "mode"] + all_keys
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for mode_name in output_modes:
                row = {
                    "mask_type": mask_type,
                    "loss_rate": loss_rate if mask_type == "gilbert" else "",
                    "gap_ms": gap_ms if mask_type == "single_gap" else "",
                    "mode": mode_name,
                }
                row.update(final.get(mode_name, {}))
                writer.writerow(row)

            if "av" in final and "audio" in final:
                delta_row = {
                    "mask_type": mask_type,
                    "loss_rate": loss_rate if mask_type == "gilbert" else "",
                    "gap_ms": gap_ms if mask_type == "single_gap" else "",
                    "mode": "delta_av_minus_audio",
                }
                for key in all_keys:
                    if key in final["av"] and key in final["audio"]:
                        delta_row[key] = final["av"][key] - final["audio"][key]
                writer.writerow(delta_row)

        self.logger.info(
            "Saved modality ablation CSV for %s to %s", condition_name, csv_path
        )
        return final, csv_path

    def _write_outputs(self, spec, masked_spec, fused_spec, rec_spec, synth_spec,
                       mask, path, sample_id, timestamp, mel_mean, mel_std,
                       phase_output=None, plot_dir=None, spec_dir=None, audio_dir=None):
        def _to_float(x):
            if x is None:
                return None
            x = np.asarray(x)
            if x.dtype.kind in "iu":
                x = x.astype(np.float32) / np.iinfo(x.dtype).max
            return np.clip(x.astype(np.float32), -1.0, 1.0)

        def _mel_to_audio(t):
            if t is None:
                return None
            audio = (torch_mel_to_audio(t.cpu(), mel_mean, mel_std) if self.vocoder is None
                     else mel_to_audio_hifigan(t.cpu(), self.vocoder, mel_mean, mel_std))
            return audio.detach().cpu().numpy() if torch.is_tensor(audio) else np.asarray(audio)

        def _head_audio(head, i):
            return _to_float(_mel_to_audio(head[i] if head is not None else None))

        for i in range(len(spec)):
            sid = sample_id + i
            video_path = str(path[i])
            relative_video_path = self._relative_dataset_path(video_path)
            resolved_video_path = str(self._resolve_dataset_path(video_path))
            sample_key = self._make_sample_key(video_path)

            if plot_dir is not None:
                sample_plot_dir = os.path.join(plot_dir, sample_key)
                os.makedirs(sample_plot_dir, exist_ok=True)
                plot_fused = (
                    phase_output["predicted_mel"][i:i + 1]
                    if phase_output is not None else
                    fused_spec[i:i + 1] if fused_spec is not None else None
                )
                self._create_plots(
                    spec[i:i + 1],
                    masked_spec[i:i + 1] if masked_spec is not None else None,
                    plot_fused,
                    rec_spec[i:i + 1] if rec_spec is not None else None,
                    synth_spec[i:i + 1] if synth_spec is not None else None,
                    mask[i:i + 1], path[i:i + 1], f"sample_{sid}", sample_plot_dir)

            orig_raw, masked_raw = read_gt_input(video_path, mask[i])
            orig_f32 = _to_float(orig_raw)
            masked_f32 = _to_float(masked_raw if masked_spec is not None else None)
            if phase_output is not None and phase_output.get("final_cos") is not None:
                sample_out = {
                    k: (v[i:i + 1] if torch.is_tensor(v) and v.size(0) == spec.size(0) else v)
                    for k, v in phase_output.items()
                }
                fused_phase_audio = self._parallel_mp_audio(sample_out)[0]
                fused_raw = _to_float(fused_phase_audio.detach().cpu().numpy())
            elif phase_output is not None:
                fused_raw = _head_audio(phase_output["selected_mel"], i)
            else:
                fused_raw = _head_audio(fused_spec, i)
            rec_raw = _head_audio(rec_spec, i)
            synth_raw = _head_audio(synth_spec, i)
            fused_f32 = (self._insert_reconstructed_gap(orig_f32, fused_raw, mask[i])
                         if fused_raw is not None else None)
            rec_f32 = (self._insert_reconstructed_gap(orig_f32, rec_raw, mask[i])
                       if rec_raw is not None else None)
            synth_f32 = (self._insert_reconstructed_gap(orig_f32, synth_raw, mask[i])
                         if synth_raw is not None else None)

            fused_source = (phase_output["predicted_mel"] if phase_output is not None else fused_spec)
            fused_pred = (fused_source[i].detach().cpu().numpy().astype("float32")
                          if fused_source is not None else np.array([], dtype=np.float32))
            rec_pred = (rec_spec[i].detach().cpu().numpy().astype("float32")
                        if rec_spec is not None else np.array([], dtype=np.float32))
            synth_pred = (synth_spec[i].detach().cpu().numpy().astype("float32")
                          if synth_spec is not None else np.array([], dtype=np.float32))
            fused_merged = (self._merge_reconstructed_spec(spec[i], fused_source[i], mask[i])
                            if fused_source is not None else np.array([], dtype=np.float32))
            rec_merged = (self._merge_reconstructed_spec(spec[i], rec_spec[i], mask[i])
                          if rec_spec is not None else np.array([], dtype=np.float32))
            synth_merged = (self._merge_reconstructed_spec(spec[i], synth_spec[i], mask[i])
                            if synth_spec is not None else np.array([], dtype=np.float32))

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
                    masked_spec=(masked_spec[i].detach().cpu().numpy().astype("float32")
                                 if masked_spec is not None else np.array([], dtype=np.float32)),
                    mask=mask[i].detach().cpu().numpy().astype("float32"),
                    fused_predicted_spec=fused_pred,
                    audio_predicted_spec=rec_pred,
                    video_predicted_spec=synth_pred,
                    phase_completed_spec=(phase_output["selected_mel"][i].detach().cpu().numpy().astype("float32")
                                          if phase_output is not None else np.array([], dtype=np.float32)),
                    phase_final_cos=(phase_output["final_cos"][i].detach().cpu().numpy().astype("float32")
                                     if phase_output is not None and phase_output.get("final_cos") is not None
                                     else np.array([], dtype=np.float32)),
                    phase_final_sin=(phase_output["final_sin"][i].detach().cpu().numpy().astype("float32")
                                     if phase_output is not None and phase_output.get("final_sin") is not None
                                     else np.array([], dtype=np.float32)),
                    fused_merged_spec=fused_merged,
                    audio_merged_spec=rec_merged,
                    video_merged_spec=synth_merged,
                    original_audio=orig_f32,
                    masked_audio=(masked_f32 if masked_f32 is not None else np.array([], dtype=np.float32)),
                    fused_audio=(fused_f32 if fused_f32 is not None else np.array([], dtype=np.float32)),
                    audio_only_audio=(rec_f32 if rec_f32 is not None else np.array([], dtype=np.float32)),
                    video_only_audio=(synth_f32 if synth_f32 is not None else np.array([], dtype=np.float32)),
                )

            if audio_dir is not None:
                sample_audio_dir = os.path.join(audio_dir, sample_key)
                os.makedirs(sample_audio_dir, exist_ok=True)
                outputs = {
                    "original": orig_f32,
                    "masked": masked_f32,
                    "fused": fused_f32,
                    "audio_only": rec_f32,
                    "video_only": synth_f32,
                }
                for tag, audio in outputs.items():
                    if audio is not None:
                        write(os.path.join(sample_audio_dir, f"{sample_key}_{tag}.wav"),
                              self.sample_rate, audio)

        return sample_id + len(spec)

    def close(self):
        self.writer.flush()
        self.writer.close()

