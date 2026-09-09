import torch
import torch.nn as nn
import torch.nn.functional as F
from conformer import Conformer

from AV_PLC.audio_encoder import Audio_Encoder
from AV_PLC.video_encoder import Video_Encoder
from AV_PLC.fusion import MLP_Fusion, TemporalSelfCrossAttentionFusion, GlobalLocalAffinityFusion
from AV_PLC.spectral_completion import SpectralCompletion


class AV_PLC(nn.Module):
    """AV-PLC with one common latent-to-spectrum temporal decoder.

    Reconstruction path
    -------------------
      audio-only : A     -> E_A --\
      video-only : V     -> E_V ----> spectral_temporal -> Mel (+ phase)
      AV         : F_av  -> E_F --/

    When learned phase is enabled, one phase encoder consumes observable phase
    ``[R_phi*cos(phi), R_phi*sin(phi), R_phi]`` and is merged with the selected
    latent representation *before* ``spectral_temporal``.

    The audio/video Mel heads are retained only as optional auxiliary encoder
    outputs; they are not inputs to the reconstruction decoder.
    """

    def __init__(self, mel_dim=80, feat_dim=256, dropout=0.1,
                 video_depth=6, video_heads=4, audio_depth=4, audio_heads=4,
                 video_hidden_size=256, audio_hidden_size=256,
                 audio_ckpt_path=None, freeze_audio_enc=False,
                 fusion_type="concat",
                 decoder_depth=2, decoder_heads=4,
                 affinity_dim=128, max_av_offset=16, global_temperature=0.1, local_temperature=0.1,
                 prior_strength=1.0, prior_sigma=2.0, min_offset_support=4.0,
                 phase_reconstruction=False, phase_bins=257, phase_decoder_depth=None,):

        super().__init__()

        self.phase_reconstruction = bool(phase_reconstruction)
        self.phase_bins = int(phase_bins)
        self.feat_dim = int(feat_dim)

        self.video_enc = Video_Encoder(conformer_block=video_depth, num_heads=video_heads,
                                       hidden_size=video_hidden_size, feat_dim=feat_dim,)
        self.audio_enc = Audio_Encoder(conformer_block=audio_depth, num_heads=audio_heads,
                                       mel_emb=mel_dim, hidden_size=audio_hidden_size,
                                       feat_dim=feat_dim,)

        if audio_ckpt_path is not None:
            ckpt = torch.load(audio_ckpt_path, map_location="cpu", weights_only=False)
            state = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
            state = {k: v for k, v in state.items() if not k.startswith("feat_proj.")}
            self.audio_enc.load_state_dict(state, strict=False)

        if freeze_audio_enc:
            for p in self.audio_enc.audio_emb.parameters():
                p.requires_grad = False
            for p in self.audio_enc.conformer.parameters():
                p.requires_grad = False
            for p in self.audio_enc.norm_layer.parameters():
                p.requires_grad = False
            # Preserve the existing auxiliary heads for compatibility.
            for p in self.audio_enc.mel_proj.parameters():
                p.requires_grad = True
            for p in self.audio_enc.feat_proj.parameters():
                p.requires_grad = True

        # One shared post-encoder/fusion temporal reconstruction stack for both
        # Griffin-Lim and learned-phase variants.  phase_decoder_depth is kept as
        # a compatibility alias; when supplied it controls this same decoder.
        spectral_depth = decoder_depth if phase_decoder_depth is None else phase_decoder_depth
        # Build the shared Conformer before optional phase-specific modules so
        # its initialization is identical between phase/no-phase variants when
        # the same seed is used.
        self.spectral_temporal = Conformer(dim=feat_dim, depth=spectral_depth, dim_head=64,
                                           heads=decoder_heads, ff_mult=4, conv_expansion_factor=2,
                                           conv_kernel_size=31, attn_dropout=dropout,
                                           ff_dropout=dropout,conv_dropout=dropout,)

        self.spectral_completion = SpectralCompletion(mel_dim=mel_dim, phase_bins=self.phase_bins,
                                                      feat_dim=feat_dim, dropout=dropout,
                                                      phase_reconstruction=self.phase_reconstruction,)

        self.fusion_type = fusion_type

        if fusion_type == "concat":
            self.fusion = MLP_Fusion(feat_dim=feat_dim, mel_dim=mel_dim, depth=2, dropout=dropout)
        elif fusion_type == "temporal_self_cross_attention":
            self.fusion = TemporalSelfCrossAttentionFusion(feat_dim=feat_dim, mel_dim=mel_dim,
                                                           dropout=dropout)
        elif fusion_type == "global_local_affinity":
            self.fusion = GlobalLocalAffinityFusion(feat_dim=feat_dim, mel_dim=mel_dim,
                                                    affinity_dim=affinity_dim, dropout=dropout,
                                                    max_offset=max_av_offset,
                                                    global_temperature=global_temperature,
                                                    local_temperature=local_temperature,
                                                    prior_strength=prior_strength,
                                                    prior_sigma=prior_sigma,
                                                    min_offset_support=min_offset_support,)
        else:
            raise ValueError("fusion_type must be 'concat', 'temporal_self_cross_attention', "
                "or 'global_local_affinity'.")

    @staticmethod
    def _audio_reliability(dec_input, audio_mask, target_steps):
        """Return [B,T] reliability, with 1=observed and 0=PLC gap."""
        if audio_mask is not None:
            if audio_mask.dim() == 3:
                reliability = audio_mask.float().mean(dim=1)
            elif audio_mask.dim() == 2:
                reliability = audio_mask.float()
            else:
                raise ValueError("audio_mask must be [B,F,T] or [B,T].")
        else:
            if dec_input is None:
                raise ValueError("dec_input or audio_mask is required to infer audio reliability")
            # Current dataloader zeros every Mel bin in missing frames.
            reliability = (dec_input.abs().amax(dim=1) > 1e-8).float()

        if reliability.size(1) != target_steps:
            reliability = F.interpolate(reliability.unsqueeze(1), size=target_steps, mode="nearest").squeeze(1)
        return reliability

    @staticmethod
    def _video_reliability(enc_input, target_steps):
        """Estimate [B,T] reliability from zero-valued video augmentation."""
        spatial_dims = tuple(range(2, enc_input.dim()))
        reliability = (enc_input.abs() > 1e-8).float().mean(dim=spatial_dims)
        if reliability.size(1) != target_steps:
            reliability = F.interpolate(reliability.unsqueeze(1), size=target_steps, mode="nearest").squeeze(1)
        return reliability

    @staticmethod
    def _select_latent(fused_feature, afeature, vfeature, avail):
        """Select F_av/A/V per sample and return [B,T,D] + source ids.

        Source ids are 0=audio, 1=video, 2=fused.  A mixed batch requires the
        selected representations to share T and D, which is the normal AV_PLC
        Mel-rate setup.  A clear error is raised instead of silently resampling
        latent features if this invariant is violated.
        """
        a_on = avail[:, 0]
        v_on = avail[:, 1]
        both = a_on & v_on
        only_a = a_on & (~v_on)
        only_v = (~a_on) & v_on

        if not torch.all(a_on | v_on):
            raise ValueError("Each sample must have at least one available modality.")

        candidates = []
        if both.any():
            if fused_feature is None:
                raise RuntimeError("AV samples require fused features.")
            candidates.append(fused_feature[both])
        if only_a.any():
            if afeature is None:
                raise RuntimeError("Audio-only samples require audio features.")
            candidates.append(afeature[only_a])
        if only_v.any():
            if vfeature is None:
                raise RuntimeError("Video-only samples require video features.")
            candidates.append(vfeature[only_v])
        if not candidates:
            raise RuntimeError("No latent representation is available.")

        reference_shape = candidates[0].shape[1:]
        for tensor in candidates[1:]:
            if tensor.shape[1:] != reference_shape:
                raise ValueError("Selected A/V/F_av representations must share [T,D] in a mixed batch; "
                                 f"got {reference_shape} and {tensor.shape[1:]}")

        template = candidates[0]
        batch = avail.size(0)

        latent = template.new_zeros((batch,) + reference_shape)
        source_ids = torch.empty(batch, dtype=torch.long, device=avail.device)

        if both.any():
            latent[both] = fused_feature[both].to(device=latent.device, dtype=latent.dtype)
            source_ids[both] = SpectralCompletion.SOURCE_FUSED

        if only_a.any():
            latent[only_a] = afeature[only_a].to(device=latent.device, dtype=latent.dtype)
            source_ids[only_a] = SpectralCompletion.SOURCE_AUDIO

        if only_v.any():
            latent[only_v] = vfeature[only_v].to(device=latent.device, dtype=latent.dtype)
            source_ids[only_v] = SpectralCompletion.SOURCE_VIDEO

        return latent, source_ids

    def forward(self, dec_input=None, enc_input=None, spk_emb=None,
                audio_length=None, avail=None, audio_mask=None, phase=None):

        amel = afeature = vmel = vfeature = None

        if avail is None:
            batch = dec_input.size(0) if dec_input is not None else enc_input.size(0)
            device = dec_input.device if dec_input is not None else enc_input.device
            avail = torch.ones(batch, 2, dtype=torch.bool, device=device)

        if avail.dim() != 2 or avail.size(1) != 2:
            raise ValueError("avail must be [B,2] with columns [audio, video]")

        a_on = avail[:, 0]
        v_on = avail[:, 1]
        both = a_on & v_on

        if a_on.any():
            if dec_input is None:
                raise ValueError("Audio-present samples require dec_input")
            idx = a_on.nonzero(as_tuple=True)[0]
            if audio_length is not None:
                idx = idx.to(audio_length.device)
            a_in = dec_input[idx]
            a_len = audio_length[idx] if audio_length is not None else None
            amel_sub, afeat_sub = self.audio_enc(a_in, a_len)

            batch = avail.size(0)
            amel = amel_sub.new_zeros((batch,) + amel_sub.shape[1:])
            afeature = afeat_sub.new_zeros((batch,) + afeat_sub.shape[1:])
            amel[idx] = amel_sub
            afeature[idx] = afeat_sub

        if v_on.any():
            if enc_input is None:
                raise ValueError("Video-present samples require enc_input")
            idx = v_on.nonzero(as_tuple=True)[0]
            v_in = enc_input[idx]
            spk_sub = spk_emb[idx] if spk_emb is not None else None
            vmel_sub, vfeat_sub = self.video_enc(v_in, spk_sub)

            batch = avail.size(0)
            vmel = vmel_sub.new_zeros((batch,) + vmel_sub.shape[1:])
            vfeature = vfeat_sub.new_zeros((batch,) + vfeat_sub.shape[1:])
            vmel[idx] = vmel_sub
            vfeature[idx] = vfeat_sub

        fused_feature = None
        if both.any():
            batch = avail.size(0)
            afeat_both = afeature[both]
            vfeat_both = vfeature[both]

            if self.fusion_type != "temporal_self_cross_attention" and afeat_both.size(1) != vfeat_both.size(1):
                raise ValueError(f"{self.fusion_type} requires equal audio/video feature lengths. "
                    f"Got T_audio={afeat_both.size(1)} and T_video={vfeat_both.size(1)}.")

            if self.fusion_type == "concat":
                fused_sub = self.fusion(afeat=afeat_both, vfeat=vfeat_both, avail=None)
            else:
                audio_rel_both = self._audio_reliability(dec_input=dec_input, audio_mask=audio_mask,
                                                         target_steps=afeature.size(1),)[both]
                video_rel_both = self._video_reliability(enc_input=enc_input,
                                                         target_steps=vfeature.size(1),)[both]
                fused_sub = self.fusion(afeat=afeat_both, vfeat=vfeat_both,
                                        audio_reliability=audio_rel_both,
                                        video_reliability=video_rel_both,)

            if fused_sub.dim() != 3 or fused_sub.size(-1) != self.feat_dim:
                raise ValueError("Fusion modules must return [B,T,feat_dim] latent features; "
                                 f"got {tuple(fused_sub.shape)}")

            fused_feature = fused_sub.new_zeros((batch,) + fused_sub.shape[1:])
            fused_feature[both] = fused_sub

        latent, source_ids = self._select_latent(fused_feature, afeature, vfeature, avail)
        target_steps = latent.size(1)

        # R_m and R_phi are identical for audio-present samples in the current
        # setup: observed audio frames are copied, gaps are predicted.  For
        # video-only samples a_on=0 makes reliability zero over the full utterance.
        if dec_input is not None or audio_mask is not None:
            audio_rel = self._audio_reliability(dec_input=dec_input, audio_mask=audio_mask,
                                                target_steps=target_steps,).to(latent.device)
        else:
            audio_rel = latent.new_zeros((latent.size(0), target_steps))

        mel_reliability = audio_rel * a_on.to(audio_rel.dtype).unsqueeze(1)
        prediction_mask = 1.0 - mel_reliability
        observed_cos = observed_sin = phase_reliability = None

        if self.phase_reconstruction:
            if phase is None:
                raise ValueError("phase is required when phase_reconstruction=True")
            if phase.size(-1) != target_steps:
                raise ValueError(f"Phase/latent time mismatch: T_phase={phase.size(-1)}, T_latent={target_steps}")
            if phase.size(1) != self.phase_bins:
                raise ValueError(f"Expected {self.phase_bins} phase bins, got {phase.size(1)}")

            phase_reliability = mel_reliability
            hidden, observed_cos, observed_sin, phase_reliability = self.spectral_completion.encode(latent,
                                                                                                    source_ids,
                                                                                                    phase_radians=phase,
                                                                                                    phase_reliability=phase_reliability,)
        else:
            hidden, _, _, _ = self.spectral_completion.encode(latent, source_ids)

        hidden = self.spectral_temporal(hidden)
        predicted_mel, p_r, p_i, pred_cos, pred_sin = self.spectral_completion.decode(hidden)

        if dec_input is None:
            observed_mel = torch.zeros_like(predicted_mel)
        else:
            observed_mel = dec_input.to(device=predicted_mel.device, dtype=predicted_mel.dtype)
            if observed_mel.size(-1) != predicted_mel.size(-1):
                raise ValueError("Observed Mel and predicted Mel must share the same time axis; "
                                 f"got {observed_mel.size(-1)} and {predicted_mel.size(-1)}")

        r_m = mel_reliability.unsqueeze(1).to(predicted_mel.dtype)
        completed_mel = r_m * observed_mel + (1.0 - r_m) * predicted_mel

        final_cos = final_sin = None
        if self.phase_reconstruction:
            final_cos, final_sin = self.spectral_completion.merge_observed_phase(observed_cos, observed_sin,
                                                                                 pred_cos, pred_sin,
                                                                                 phase_reliability,)

        completion_output = {
            "predicted_mel": predicted_mel,
            "completed_mel": completed_mel,
            "mel_reliability": mel_reliability,
            "prediction_mask": prediction_mask,
            "source_ids": source_ids,
            "phase_reliability": phase_reliability,
            "p_r": p_r,
            "p_i": p_i,
            "pred_cos": pred_cos,
            "pred_sin": pred_sin,
            "final_cos": final_cos,
            "final_sin": final_sin,
        }

        # Keep auxiliary encoder Mel outputs available for optional regularization.
        # The primary AV output is the unified decoder prediction for AV samples.
        fused_mel = None
        if both.any():
            fused_mel = predicted_mel.new_zeros(predicted_mel.shape)
            fused_mel[both] = predicted_mel[both]

        return fused_mel, amel, vmel, completion_output
