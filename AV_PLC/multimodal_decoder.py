import torch
import torch.nn as nn
import torch.nn.functional as F
from conformer import Conformer

from AV_PLC.audio_encoder import Audio_Encoder
from AV_PLC.video_encoder import Video_Encoder
from AV_PLC.fusion import MLP_Fusion, TemporalSelfCrossAttentionFusion, GlobalLocalAffinityFusion
from AV_PLC.spectral_completion import SpectralCompletion
from AV_PLC.phase_completion import PhaseCompletion


class AV_PLC(nn.Module):
    """AV-PLC with three Mel content heads and parallel magnitude/phase TF reconstruction.

    Content interface::

      audio-only : audio encoder Mel head -> M_A
      video-only : video encoder Mel head -> M_V
      AV         : fused latent -> existing spectral temporal decoder -> M_AV

    The selected M_A/M_V/M_AV is inverse-Mel projected to linear magnitude.
    Exact observed STFT magnitude replaces that projection outside PLC gaps.
    The completed magnitude is concatenated with R*cos(phi), R*sin(phi), passed
    through TFRefine, and decoded by independent magnitude and phase branches.
    The inherited content backbone can be frozen while phase_completion.* trains.
    """

    def __init__(self, mel_dim=80, feat_dim=256, dropout=0.1,
                 video_depth=6, video_heads=4, audio_depth=4, audio_heads=4,
                 video_hidden_size=256, audio_hidden_size=256,
                 audio_ckpt_path=None, freeze_audio_enc=False,
                 fusion_type="concat",
                 decoder_depth=2, decoder_heads=4,
                 affinity_dim=128, max_av_offset=16, global_temperature=0.1, local_temperature=0.1,
                 prior_strength=1.0, prior_sigma=2.0, min_offset_support=4.0,
                 phase_reconstruction=False, phase_bins=257, phase_decoder_depth=None,
                 phase_channels=64, phase_ts_blocks=2, phase_ts_heads=4,
                 phase_encoder_dense_depth=3, phase_refine_dense_depth=2,
                 phase_magnitude_compression=0.3,
    ):
        super().__init__()
        self.phase_reconstruction = bool(phase_reconstruction)
        self.phase_bins = int(phase_bins)
        self.feat_dim = int(feat_dim)

        self.video_enc = Video_Encoder(
            conformer_block=video_depth,
            num_heads=video_heads,
            hidden_size=video_hidden_size,
            feat_dim=feat_dim,
        )
        self.audio_enc = Audio_Encoder(
            conformer_block=audio_depth,
            num_heads=audio_heads,
            mel_emb=mel_dim,
            hidden_size=audio_hidden_size,
            feat_dim=feat_dim,
        )

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

        # Protected no-phase Mel decoder.  phase_decoder_depth is retained only
        # as a deprecated constructor compatibility argument; it no longer alters
        # the Mel decoder because phase has its own branch.
        self.spectral_temporal = Conformer(
            dim=feat_dim,
            depth=decoder_depth,
            dim_head=64,
            heads=decoder_heads,
            ff_mult=4,
            conv_expansion_factor=2,
            conv_kernel_size=31,
            attn_dropout=dropout,
            ff_dropout=dropout,
            conv_dropout=dropout,
        )
        # Always construct the phase-independent SpectralCompletion shape.  This
        # is essential for strict reuse of the successful no-phase checkpoint.
        self.spectral_completion = SpectralCompletion(
            mel_dim=mel_dim,
            phase_bins=self.phase_bins,
            feat_dim=feat_dim,
            dropout=dropout,
            phase_reconstruction=False,
        )

        # Construct the new phase branch before fusion so its initialization is
        # identical across concat/TSCAF/GLA under the same global seed.
        self.phase_completion = (
            PhaseCompletion(
                mel_dim=mel_dim,
                phase_bins=self.phase_bins,
                channels=phase_channels,
                num_ts_blocks=phase_ts_blocks,
                num_heads=phase_ts_heads,
                encoder_dense_depth=phase_encoder_dense_depth,
                decoder_dense_depth=phase_refine_dense_depth,
                dropout=dropout,
                magnitude_compression=phase_magnitude_compression,
            )
            if self.phase_reconstruction
            else None
        )

        self.fusion_type = fusion_type
        if fusion_type == "concat":
            self.fusion = MLP_Fusion(
                feat_dim=feat_dim, mel_dim=mel_dim, depth=2, dropout=dropout
            )
        elif fusion_type == "temporal_self_cross_attention":
            self.fusion = TemporalSelfCrossAttentionFusion(
                feat_dim=feat_dim, mel_dim=mel_dim, dropout=dropout
            )
        elif fusion_type == "global_local_affinity":
            self.fusion = GlobalLocalAffinityFusion(
                feat_dim=feat_dim,
                mel_dim=mel_dim,
                affinity_dim=affinity_dim,
                dropout=dropout,
                max_offset=max_av_offset,
                global_temperature=global_temperature,
                local_temperature=local_temperature,
                prior_strength=prior_strength,
                prior_sigma=prior_sigma,
                min_offset_support=min_offset_support,
            )
        else:
            raise ValueError(
                "fusion_type must be 'concat', 'temporal_self_cross_attention', "
                "or 'global_local_affinity'."
            )

    @staticmethod
    def _audio_reliability(dec_input, audio_mask, target_steps):
        """Return explicit [B,T] waveform-derived reliability (1=fully observed)."""
        if audio_mask is None:
            raise ValueError("audio_mask is required for waveform-domain AV_PLC")
        if audio_mask.dim() == 3:
            reliability = audio_mask.float().mean(dim=1)
        elif audio_mask.dim() == 2:
            reliability = audio_mask.float()
        else:
            raise ValueError("audio_mask must be [B,F,T] or [B,T].")
        if reliability.size(1) != target_steps:
            reliability = F.interpolate(
                reliability.unsqueeze(1), size=target_steps, mode="nearest"
            ).squeeze(1)
        return reliability

    @staticmethod
    def _video_reliability(enc_input, target_steps):
        """Estimate [B,T] reliability from zero-valued video augmentation."""
        spatial_dims = tuple(range(2, enc_input.dim()))
        reliability = (enc_input.abs() > 1e-8).float().mean(dim=spatial_dims)
        if reliability.size(1) != target_steps:
            reliability = F.interpolate(
                reliability.unsqueeze(1), size=target_steps, mode="nearest"
            ).squeeze(1)
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
                raise ValueError(
                    "Selected A/V/F_av representations must share [T,D] in a mixed batch; "
                    f"got {reference_shape} and {tensor.shape[1:]}"
                )

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
                audio_length=None, avail=None, audio_mask=None, phase=None,
                stft_magnitude=None):

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
            # Speaker embeddings are retained in the dataset for metrics only.
            vmel_sub, vfeat_sub = self.video_enc(v_in)

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

            if (
                self.fusion_type != "temporal_self_cross_attention"
                and afeat_both.size(1) != vfeat_both.size(1)
            ):
                raise ValueError(
                    f"{self.fusion_type} requires equal audio/video feature lengths. "
                    f"Got T_audio={afeat_both.size(1)} and T_video={vfeat_both.size(1)}."
                )

            if self.fusion_type == "concat":
                fused_sub = self.fusion(
                    afeat=afeat_both, vfeat=vfeat_both, avail=None
                )
            else:
                audio_rel_both = self._audio_reliability(
                    dec_input=dec_input,
                    audio_mask=audio_mask,
                    target_steps=afeature.size(1),
                )[both]
                video_rel_both = self._video_reliability(
                    enc_input=enc_input,
                    target_steps=vfeature.size(1),
                )[both]
                fused_sub = self.fusion(
                    afeat=afeat_both,
                    vfeat=vfeat_both,
                    audio_reliability=audio_rel_both,
                    video_reliability=video_rel_both,
                )

            if fused_sub.dim() != 3 or fused_sub.size(-1) != self.feat_dim:
                raise ValueError(
                    "Fusion modules must return [B,T,feat_dim] latent features; "
                    f"got {tuple(fused_sub.shape)}"
                )
            fused_feature = fused_sub.new_zeros((batch,) + fused_sub.shape[1:])
            fused_feature[both] = fused_sub

        latent, source_ids = self._select_latent(
            fused_feature, afeature, vfeature, avail
        )
        target_steps = latent.size(1)

        # Acoustic reliability follows the selected modality semantics.  In strict
        # video-only mode there is no usable audio anywhere, so R=0 for the whole
        # utterance.  For audio-only / AV samples, R is the PLC packet mask.
        if dec_input is not None or audio_mask is not None:
            audio_rel = self._audio_reliability(
                dec_input=dec_input,
                audio_mask=audio_mask,
                target_steps=target_steps,
            ).to(latent.device)
        else:
            audio_rel = latent.new_zeros((latent.size(0), target_steps))
        packet_reliability = audio_rel * a_on.to(audio_rel.dtype).unsqueeze(1)
        prediction_mask = 1.0 - packet_reliability

        # -------------------------- three Mel content heads --------------------------
        # The existing post-fusion temporal decoder provides M_AV.  Audio/video
        # content uses the encoder Mel heads directly.  This makes M_A/M_V/M_AV a
        # common normalized-logMel interface for the downstream TF reconstructor.
        hidden, _, _, _ = self.spectral_completion.encode(latent, source_ids)
        hidden = self.spectral_temporal(hidden)
        decoded_mel, _, _, _, _ = self.spectral_completion.decode(hidden)

        fused_mel = None
        if both.any():
            fused_mel = decoded_mel.new_zeros(decoded_mel.shape)
            fused_mel[both] = decoded_mel[both]

        selected_mel = decoded_mel.new_zeros(decoded_mel.shape)
        only_a = a_on & (~v_on)
        only_v = (~a_on) & v_on
        if both.any():
            selected_mel[both] = decoded_mel[both]
        if only_a.any():
            if amel is None:
                raise RuntimeError("Audio-only sample is missing the audio Mel head")
            selected_mel[only_a] = amel[only_a]
        if only_v.any():
            if vmel is None:
                raise RuntimeError("Video-only sample is missing the video Mel head")
            selected_mel[only_v] = vmel[only_v]

        # Completed Mel is the physically meaningful PLC Mel: copy only frames
        # whose waveform support is fully observed; predict every affected frame.
        if dec_input is not None:
            observed_mel = dec_input
            if observed_mel.size(-1) != target_steps:
                observed_mel = F.interpolate(observed_mel, size=target_steps, mode="nearest")
            r_mel = packet_reliability.to(selected_mel.dtype).unsqueeze(1)
            completed_mel = r_mel * observed_mel + (1.0 - r_mel) * selected_mel
        else:
            completed_mel = selected_mel

        # --------------------- parallel magnitude/phase TF branch ---------------------
        phase_reliability = None
        mp_out = None
        if self.phase_reconstruction:
            if self.phase_completion is None:
                raise RuntimeError("phase_reconstruction=True but phase_completion is missing")
            if phase is None or stft_magnitude is None:
                raise ValueError(
                    "phase and stft_magnitude are required for parallel magnitude/phase reconstruction"
                )
            if phase.size(-1) != target_steps or stft_magnitude.size(-1) != target_steps:
                raise ValueError(
                    "STFT magnitude/phase and Mel content must share the same time axis"
                )
            if phase.size(1) != self.phase_bins or stft_magnitude.size(1) != self.phase_bins:
                raise ValueError(f"Expected {self.phase_bins} STFT bins")

            phase_reliability = packet_reliability
            mp_out = self.phase_completion(
                reconstructed_mel=completed_mel,
                observed_magnitude=stft_magnitude,
                phase_radians=phase,
                reliability=phase_reliability,
            )
        completion_output = {
            # Mel content heads / selected content representation.
            "predicted_mel": selected_mel,  # compatibility alias
            "selected_mel": selected_mel,
            "completed_mel": completed_mel,
            "fused_mel": fused_mel,
            "audio_mel": amel,
            "video_mel": vmel,
            "packet_reliability": packet_reliability,
            "mel_reliability": packet_reliability,  # compatibility alias
            "prediction_mask": prediction_mask,
            "source_ids": source_ids,
            "phase_reliability": phase_reliability,
        }
        if mp_out is not None:
            completion_output.update(mp_out)
            # Compatibility with the previous phase-only trainer/evaluator.
            completion_output["phase_reliability"] = mp_out["reliability"]

        return fused_mel, amel, vmel, completion_output
