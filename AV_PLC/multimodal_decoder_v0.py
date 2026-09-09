import torch
import torch.nn as nn
import torch.nn.functional as F
from conformer import Conformer

from AV_PLC.audio_encoder import Audio_Encoder
from AV_PLC.video_encoder import Video_Encoder
from AV_PLC.fusion import MLP_Fusion, TemporalSelfCrossAttentionFusion, GlobalLocalAffinityFusion
from AV_PLC.spectral_completion import SpectralCompletion


class AV_PLC(nn.Module):
    def __init__(self, mel_dim=80, feat_dim=256, dropout=0.1,
                 video_depth=6, video_heads=4, audio_depth=4, audio_heads=4,
                 video_hidden_size=256, audio_hidden_size=256,
                 audio_ckpt_path=None, freeze_audio_enc=False,
                 fusion_type="concat",
                 decoder_depth=2, decoder_heads=4,
                 affinity_dim=128, max_av_offset=16, global_temperature=0.1, local_temperature=0.1,
                 prior_strength=1.0, prior_sigma=2.0, min_offset_support=4.0,
                 phase_reconstruction=False, phase_bins=257, phase_decoder_depth=2,):

        super().__init__()
        self.phase_reconstruction = bool(phase_reconstruction)
        self.phase_bins = int(phase_bins)

        self.video_enc = Video_Encoder(conformer_block=video_depth, num_heads=video_heads,
                                       hidden_size=video_hidden_size, feat_dim=feat_dim,)
        self.audio_enc = Audio_Encoder(conformer_block=audio_depth, num_heads=audio_heads,
                                       mel_emb=mel_dim, hidden_size=audio_hidden_size, feat_dim=feat_dim,)

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
            for p in self.audio_enc.mel_proj.parameters():
                p.requires_grad = True
            for p in self.audio_enc.feat_proj.parameters():
                p.requires_grad = True

        self.fusion_type = fusion_type

        if fusion_type == "concat":
            self.fusion = MLP_Fusion(feat_dim=feat_dim, mel_dim=mel_dim, depth=2, dropout=dropout,)

        elif fusion_type == "temporal_self_cross_attention":
            self.fusion = TemporalSelfCrossAttentionFusion(feat_dim=feat_dim, mel_dim=mel_dim, dropout=dropout,)

        elif fusion_type == "global_local_affinity":
            self.fusion = GlobalLocalAffinityFusion(feat_dim=feat_dim, mel_dim=mel_dim,
                                                    affinity_dim=affinity_dim, dropout=dropout,
                                                    max_offset=max_av_offset, global_temperature=global_temperature,
                                                    local_temperature=local_temperature, prior_strength=prior_strength,
                                                    prior_sigma=prior_sigma, min_offset_support=min_offset_support,)
        else:
            raise ValueError("fusion_type must be 'concat', 'temporal_self_cross_attention', or 'global_local_affinity'.")

        self.temporal = Conformer(dim=feat_dim, depth=decoder_depth, dim_head=64,
                                  heads=decoder_heads, ff_mult=4, conv_expansion_factor=2,
                                  conv_kernel_size=31, attn_dropout=dropout, ff_dropout=dropout,
                                  conv_dropout=dropout,)

        self.out = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, mel_dim),)

        if self.phase_reconstruction:
            self.spectral_completion = SpectralCompletion(mel_dim=mel_dim, phase_bins=self.phase_bins,
                                                          feat_dim=feat_dim, dropout=dropout,)

            self.spectral_temporal = Conformer(dim=feat_dim, depth=phase_decoder_depth,
                                               dim_head=64, heads=decoder_heads, ff_mult=4,
                                               conv_expansion_factor=2, conv_kernel_size=31,
                                               attn_dropout=dropout, ff_dropout=dropout,
                                               conv_dropout=dropout,)

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
            # Backward-compatible fallback: masked Mel frames are exactly zero
            # across every Mel bin in the current dataloader.
            reliability = (dec_input.abs().amax(dim=1) > 1e-8).float()

        if reliability.size(1) != target_steps:
            reliability = F.interpolate(reliability.unsqueeze(1), size=target_steps, mode="nearest",).squeeze(1)
        return reliability

    @staticmethod
    def _video_reliability(enc_input, target_steps):
        """Estimate [B,T] reliability from zero-valued video augmentation.
        TimeMask gives reliability 0 for a fully hidden frame. RandomErase gives
        a fractional reliability equal to the visible pixel fraction.
        """
        spatial_dims = tuple(range(2, enc_input.dim()))
        reliability = (enc_input.abs() > 1e-8).float().mean(dim=spatial_dims)
        if reliability.size(1) != target_steps:
            reliability = F.interpolate(reliability.unsqueeze(1), size=target_steps, mode="nearest",).squeeze(1)
        return reliability

    @staticmethod
    def _select_base_mel(fused_mel, amel, vmel, avail):
        """Select M0 per sample without changing any legacy Mel-producing path."""
        a_on = avail[:, 0]
        v_on = avail[:, 1]
        both = a_on & v_on
        only_a = a_on & (~v_on)
        only_v = (~a_on) & v_on

        if not torch.all(a_on | v_on):
            raise ValueError("Each sample must have at least one available modality.")

        template = fused_mel if fused_mel is not None else amel if amel is not None else vmel
        if template is None:
            raise RuntimeError("No Mel output is available for spectral completion.")
        base = template.new_zeros(template.shape)
        if both.any():
            if fused_mel is None:
                raise RuntimeError("AV samples require fused_mel before spectral completion.")
            base[both] = fused_mel[both]
        if only_a.any():
            if amel is None:
                raise RuntimeError("Audio-only samples require amel before spectral completion.")
            base[only_a] = amel[only_a]
        if only_v.any():
            if vmel is None:
                raise RuntimeError("Video-only samples require vmel before spectral completion.")
            base[only_v] = vmel[only_v]
        return base

    def forward(self, dec_input=None, enc_input=None, spk_emb=None,
        audio_length=None, avail=None, audio_mask=None, phase=None,):

        amel = afeature = vmel = vfeature = None

        if avail is None:
            batch = dec_input.size(0) if dec_input is not None else enc_input.size(0)
            device = dec_input.device if dec_input is not None else enc_input.device
            avail = torch.ones(batch, 2, dtype=torch.bool, device=device)

        a_on = avail[:, 0]
        v_on = avail[:, 1]
        both = a_on & v_on

        if dec_input is not None and a_on.any():
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

        if enc_input is not None and v_on.any():
            idx = v_on.nonzero(as_tuple=True)[0]
            v_in = enc_input[idx]
            spk_sub = spk_emb[idx] if spk_emb is not None else None
            vmel_sub, vfeat_sub = self.video_enc(v_in, spk_sub)

            batch = avail.size(0)
            vmel = vmel_sub.new_zeros((batch,) + vmel_sub.shape[1:])
            vfeature = vfeat_sub.new_zeros((batch,) + vfeat_sub.shape[1:])
            vmel[idx] = vmel_sub
            vfeature[idx] = vfeat_sub

        fused_mel = None

        if both.any():
            batch = avail.size(0)

            afeat_both = afeature[both]
            vfeat_both = vfeature[both]

            if (self.fusion_type != "temporal_self_cross_attention"
                and afeat_both.size(1) != vfeat_both.size(1)):
                raise ValueError(f"{self.fusion_type} requires equal audio/video feature lengths. "
                    f"Got T_audio={afeat_both.size(1)} and T_video={vfeat_both.size(1)}.")

            if self.fusion_type == "concat":
                fused_sub = self.fusion(afeat=afeat_both, vfeat=vfeat_both, avail=None,)
            else:
                audio_rel = self._audio_reliability(dec_input=dec_input, audio_mask=audio_mask,
                    target_steps=afeature.size(1),)[both]

                video_rel = self._video_reliability(enc_input=enc_input,
                    target_steps=vfeature.size(1),)[both]

                fused_sub = self.fusion(afeat=afeat_both, vfeat=vfeat_both,
                    audio_reliability=audio_rel, video_reliability=video_rel,)

            fused_sub = self.temporal(fused_sub)
            fused_sub = self.out(fused_sub).permute(0, 2, 1)
            fused_mel = fused_sub.new_zeros((batch,) + fused_sub.shape[1:])
            fused_mel[both] = fused_sub

        if not self.phase_reconstruction:
            return fused_mel, amel, vmel

        if phase is None:
            raise ValueError("phase is required when phase_reconstruction=True")

        base_mel = self._select_base_mel(fused_mel, amel, vmel, avail)
        target_steps = base_mel.size(-1)
        if phase.size(-1) != target_steps:
            raise ValueError(f"Phase/Mel time mismatch: T_phase={phase.size(-1)}, T_mel={target_steps}")
        if phase.size(1) != self.phase_bins:
            raise ValueError(f"Expected {self.phase_bins} phase bins, got {phase.size(1)}")

        # The packet mask is defined on Mel time.  Audio modality dropout must
        # additionally remove *all* phase input for video-only samples.
        audio_rel = self._audio_reliability(dec_input=dec_input, audio_mask=audio_mask,
                                            target_steps=target_steps,).to(base_mel.device)

        phase_reliability = audio_rel * a_on.to(audio_rel.dtype).unsqueeze(1)
        hidden, observed_cos, observed_sin, phase_reliability = self.spectral_completion.encode(base_mel,
                                                                                                phase,
                                                                                                phase_reliability)
        hidden = self.spectral_temporal(hidden)
        delta_mel, p_r, p_i, pred_cos, pred_sin = self.spectral_completion.decode(hidden)

        # Refine only spectral content that is unavailable from audio.  This keeps
        # observed-region legacy predictions unchanged for an apples-to-apples
        # comparison, while video-only (R_phi=0) can refine its complete Mel.
        prediction_mask = 1.0 - phase_reliability
        effective_delta_mel = delta_mel * prediction_mask.unsqueeze(1).to(delta_mel.dtype)
        refined_mel = base_mel + effective_delta_mel

        # Preserve exact received Mel where audio is available; for video-only,
        # phase_reliability is zero everywhere so the full Mel is synthesized.
        if dec_input is None:
            observed_mel = torch.zeros_like(refined_mel)
        else:
            observed_mel = dec_input.to(refined_mel.dtype)

        r = phase_reliability.unsqueeze(1).to(refined_mel.dtype)
        completed_mel = r * observed_mel + (1.0 - r) * refined_mel
        final_cos, final_sin = self.spectral_completion.merge_observed_phase(observed_cos, observed_sin,
                                                                             pred_cos, pred_sin,
                                                                             phase_reliability)

        phase_output = {
            "base_mel": base_mel,
            "delta_mel": delta_mel,
            "effective_delta_mel": effective_delta_mel,
            "refined_mel": refined_mel,
            "completed_mel": completed_mel,
            "p_r": p_r,
            "p_i": p_i,
            "pred_cos": pred_cos,
            "pred_sin": pred_sin,
            "final_cos": final_cos,
            "final_sin": final_sin,
            "phase_reliability": phase_reliability,
            "prediction_mask": prediction_mask,
        }
        return fused_mel, amel, vmel, phase_output
