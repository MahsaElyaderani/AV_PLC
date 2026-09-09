import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from conformer import Conformer


class MLP_Fusion(nn.Module):
    """ MLP concatenation fusion, the baseline."""

    def __init__(self, feat_dim=256, mel_dim=80, depth=2, dropout=0.1):
        super().__init__()
        self.norm_a = nn.LayerNorm(feat_dim)
        self.norm_v = nn.LayerNorm(feat_dim)

        self.mix = nn.Sequential(
            nn.Linear(2 * feat_dim, 2 * feat_dim),
            nn.GELU(),
            nn.Linear(2 * feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
        )

        # self.temporal = Conformer(
        #     dim=feat_dim,
        #     depth=depth,
        #     dim_head=64,
        #     heads=4,
        #     ff_mult=4,
        #     conv_expansion_factor=2,
        #     conv_kernel_size=31,
        #     attn_dropout=dropout,
        #     ff_dropout=dropout,
        #     conv_dropout=dropout,
        # )
        #
        # self.out = nn.Sequential(
        #     nn.LayerNorm(feat_dim),
        #     nn.Linear(feat_dim, mel_dim),
        # )

    def forward(self, afeat=None, vfeat=None, avail=None):
        if avail is None:
            if afeat is None and vfeat is None:
                raise ValueError("At least one modality must be provided.")
            if afeat is not None and vfeat is not None:
                a = self.norm_a(afeat)
                v = self.norm_v(vfeat)
                x = self.mix(torch.cat([a, v], dim=-1))
            else:
                x = self.norm_a(afeat) if afeat is not None else self.norm_v(vfeat)
        else:
            if avail.dim() != 2 or avail.size(1) != 2:
                raise ValueError("avail must be [B,2] bool tensor: [audio_present, video_present].")
            a_on = avail[:, 0]
            v_on = avail[:, 1]
            if not torch.all(a_on | v_on):
                raise ValueError("Each sample must have at least one modality present.")

            template = afeat if afeat is not None else vfeat
            if template is None:
                raise ValueError("No feature tensor available for fusion.")
            x = template.new_zeros(template.shape)

            both = a_on & v_on
            only_a = a_on & (~v_on)
            only_v = (~a_on) & v_on

            if both.any():
                a = self.norm_a(afeat[both]).to(dtype=x.dtype)
                v = self.norm_v(vfeat[both]).to(dtype=x.dtype)
                x[both] = self.mix(torch.cat([a, v], dim=-1)).to(dtype=x.dtype)

            if only_a.any():
                x[only_a] = self.norm_a(afeat[only_a]).to(dtype=x.dtype)

            if only_v.any():
                x[only_v] = self.norm_v(vfeat[only_v]).to(dtype=x.dtype)

        #x = self.temporal(x)
        #return self.out(x).permute(0, 2, 1)
        return x

class SinusoidalPositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embedding similar to av_transformer.

    Input:
        positions: [T] integer or floating-point positions

    Output:
        embeddings: [T, D]
    """

    def __init__(self, dim: int):
        super().__init__()

        if dim < 2:
            raise ValueError("dim must be at least 2.")

        self.dim = dim

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        if positions.dim() != 1:
            raise ValueError("positions must be a one-dimensional tensor [T].")

        device = positions.device
        dtype = torch.float32

        half_dim = self.dim // 2

        if half_dim == 1:
            frequencies = torch.ones(1, device=device, dtype=dtype)
        else:
            scale = math.log(10000.0) / (half_dim - 1)
            frequencies = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -scale)

        angles = positions.to(dtype=dtype).unsqueeze(-1) * frequencies.unsqueeze(0)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1,)

        # Support odd feature dimensions.
        if embedding.size(-1) < self.dim:
            embedding = torch.cat(
                [embedding, torch.zeros(embedding.size(0), self.dim - embedding.size(-1),
                                        device=device, dtype=embedding.dtype,),], dim=-1,)

        return embedding[:, : self.dim]


class TemporalSelfCrossAttentionFusion(nn.Module):
    """
    Reliability-aware global AV self-attention followed by audio-query
    cross-attention.

    Processing:
        1. Normalize audio and video encoder features.
        2. Add modality, sinusoidal position, and reliability embeddings.
        3. Concatenate audio and video tokens.
        4. Apply global AV self-attention to construct AV memory.
        5. Let T_audio audio tokens query the complete AV memory.

    Inputs:
        afeat: [B, T_audio, D]
        vfeat: [B, T_video, D]
        audio_reliability: [B, T_audio], 1=observed, 0=missing
        video_reliability: [B, T_video], 1=reliable, 0=unavailable

    Output:
        predicted Mel spectrogram: [B, mel_dim, T_audio]
    """

    def __init__(self, feat_dim: int = 256, mel_dim: int = 80, depth: int = 2,
                 dropout: float = 0.1, heads: int = 4, conformer_depth: int = 2,):
        super().__init__()

        if feat_dim % heads != 0:
            raise ValueError(f"feat_dim={feat_dim} must be divisible by heads={heads}.")

        self.feat_dim = feat_dim

        # Modality-specific normalization preserves the different statistics
        # of the separately trained audio and video encoders.
        self.norm_a = nn.LayerNorm(feat_dim)
        self.norm_v = nn.LayerNorm(feat_dim)

        self.audio_modality = nn.Parameter(torch.zeros(1, 1, feat_dim))
        self.video_modality = nn.Parameter(torch.zeros(1, 1, feat_dim))

        self.position_embedding = SinusoidalPositionalEmbedding(feat_dim)

        # Reliability is represented continuously rather than only through
        # binary attention masking.
        self.audio_reliability_proj = nn.Linear(1, feat_dim, bias=False)
        self.video_reliability_proj = nn.Linear(1, feat_dim, bias=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=heads,
            dim_feedforward=4 * feat_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.av_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=depth,
            norm=nn.LayerNorm(feat_dim),
        )

        # One global decoder-like AV cross-attention layer.
        self.audio_from_av = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )

        self.cross_dropout = nn.Dropout(dropout)
        self.cross_norm = nn.LayerNorm(feat_dim)

        # Lightweight decoder feed-forward sublayer.
        self.cross_ffn = nn.Sequential(
            nn.Linear(feat_dim, 4 * feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * feat_dim, feat_dim),
            nn.Dropout(dropout),
        )
        self.cross_ffn_norm = nn.LayerNorm(feat_dim)

        # Existing AV_PLC temporal decoder.
        # self.temporal = Conformer(dim=feat_dim, depth=conformer_depth, dim_head=64, heads=heads,
        #                           ff_mult=4, conv_expansion_factor=2, conv_kernel_size=31,
        #                           attn_dropout=dropout, ff_dropout=dropout, conv_dropout=dropout,)
        #
        # self.out = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, mel_dim),)

        nn.init.normal_(self.audio_modality, std=0.02)
        nn.init.normal_(self.video_modality, std=0.02)

    @staticmethod
    def _validate_reliability(
        reliability: torch.Tensor,
        batch: int,
        steps: int,
        name: str,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if reliability.shape != (batch, steps):
            raise ValueError(
                f"{name} must have shape [B,T]=({batch},{steps}), "
                f"got {tuple(reliability.shape)}."
            )

        return reliability.to(
            device=reference.device,
            dtype=reference.dtype,
        ).clamp(0.0, 1.0)

    def forward(
        self,
        afeat: torch.Tensor,
        vfeat: torch.Tensor,
        audio_reliability: torch.Tensor,
        video_reliability: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if afeat is None or vfeat is None:
            raise ValueError("TemporalSelfCrossAttentionFusion requires both modalities.")

        if afeat.dim() != 3 or vfeat.dim() != 3:
            raise ValueError("afeat and vfeat must have shape [B,T,D].")

        batch_a, audio_steps, audio_dim = afeat.shape
        batch_v, video_steps, video_dim = vfeat.shape

        if batch_a != batch_v:
            raise ValueError("Audio and video batch sizes must match.")

        if audio_dim != self.feat_dim or video_dim != self.feat_dim:
            raise ValueError(f"Expected feature dimension {self.feat_dim}; "
                             f"got audio={audio_dim}, video={video_dim}.")

        audio_reliability = self._validate_reliability(
            reliability=audio_reliability,
            batch=batch_a,
            steps=audio_steps,
            name="audio_reliability",
            reference=afeat,
        )

        if video_reliability is None:
            video_reliability = vfeat.new_ones(batch_v, video_steps)
        else:
            video_reliability = self._validate_reliability(
                reliability=video_reliability,
                batch=batch_v,
                steps=video_steps,
                name="video_reliability",
                reference=vfeat,
            )

        # Each modality receives its own timeline beginning at position zero.
        # This avoids implying that video begins after the final audio token.
        audio_positions = torch.arange(audio_steps, device=afeat.device,)
        video_positions = torch.arange(video_steps, device=vfeat.device,)

        audio_pos = self.position_embedding(audio_positions).to(dtype=afeat.dtype).unsqueeze(0)
        video_pos = self.position_embedding(video_positions).to(dtype=vfeat.dtype).unsqueeze(0)

        audio_tokens = (
            self.norm_a(afeat)
            + self.audio_modality
            + audio_pos
            + self.audio_reliability_proj(
                audio_reliability.unsqueeze(-1)
            )
        )

        video_tokens = (
            self.norm_v(vfeat)
            + self.video_modality
            + video_pos
            + self.video_reliability_proj(
                video_reliability.unsqueeze(-1)
            )
        )

        av_tokens = torch.cat([audio_tokens, video_tokens], dim=1,)

        # Raw unavailable tokens cannot serve as self-attention keys/values.
        # Missing audio tokens remain present as queries and can receive
        # surrounding audio and visual context.
        encoder_key_mask = torch.cat([audio_reliability <= 0.0, video_reliability <= 0.0,], dim=1,)

        # Prevent undefined attention if a pathological sample has no
        # reliable audio or video tokens.
        all_invalid = encoder_key_mask.all(dim=1)
        if all_invalid.any():
            encoder_key_mask = encoder_key_mask.clone()
            encoder_key_mask[all_invalid, 0] = False

        # [B, T_audio + T_video, D]
        av_memory = self.av_encoder(av_tokens, src_key_padding_mask=encoder_key_mask,)

        # Audio-length queries retrieve information from the complete,
        # globally contextualized AV memory.
        #
        # No cross-attention key mask is applied here deliberately:
        # positions that were missing in the raw input now contain contextual
        # encoder outputs and may be useful memory states, as in av_transformer.
        cross_update, _ = self.audio_from_av(
            query=audio_tokens,
            key=av_memory,
            value=av_memory,
            need_weights=False,
        )

        fused_audio = self.cross_norm(audio_tokens + self.cross_dropout(cross_update))
        fused_audio = self.cross_ffn_norm(fused_audio + self.cross_ffn(fused_audio))

        # The Conformer receives only T_audio tokens.
        #fused_audio = self.temporal(fused_audio)
        #return self.out(fused_audio).permute(0, 2, 1)
        return fused_audio

class GlobalLocalAffinityFusion(nn.Module):
    """
    Reliability-Guided Global-Local Affinity Bridge.

    Inputs
    ------
    afeat:
        Audio features [B, T, D].
    vfeat:
        Speaker-conditioned video features [B, T, D].
    audio_reliability:
        Audio reliability [B, T], where 1=observed and 0=missing.
    video_reliability:
        Video reliability [B, T], where 1=reliable and 0=unavailable.

    Output
    ------
    fused_mel:
        Reconstructed Mel spectrogram [B, mel_dim, T].

    Optional diagnostics contain:
        affinity:          S       [B, T, T]
        offset_posterior:  p_g     [B, K]
        global_prior:      G       [B, T, T]
        alignment:         W       [B, T, T]
        offsets:                   [K]
        offset_support:    d_k     [B, K]
    """

    def __init__(
        self,
        feat_dim=256,
        mel_dim=80,
        affinity_dim=128,
        depth=2,
        heads=4,
        dropout=0.1,
        max_offset=16,
        global_temperature=0.1,
        local_temperature=0.1,
        prior_strength=1.0,
        prior_sigma=2.0,
        min_offset_support=4.0,
        eps=1e-6,
    ):
        super().__init__()

        if max_offset < 0:
            raise ValueError("max_offset must be non-negative.")
        if global_temperature <= 0:
            raise ValueError("global_temperature must be positive.")
        if local_temperature <= 0:
            raise ValueError("local_temperature must be positive.")
        if prior_sigma <= 0:
            raise ValueError("prior_sigma must be positive.")
        if feat_dim % heads != 0:
            raise ValueError("feat_dim must be divisible by heads.")

        self.feat_dim = feat_dim
        self.max_offset = int(max_offset)
        self.global_temperature = float(global_temperature)
        self.local_temperature = float(local_temperature)
        self.prior_strength = float(prior_strength)
        self.prior_sigma = float(prior_sigma)
        self.min_offset_support = float(min_offset_support)
        self.eps = float(eps)

        # Normalize encoder feature distributions before affinity computation.
        self.audio_norm = nn.LayerNorm(feat_dim)
        self.video_norm = nn.LayerNorm(feat_dim)

        # P_a and P_v: projections into the common affinity space.
        self.audio_query = nn.Linear(feat_dim, affinity_dim, bias=False)
        self.video_key = nn.Linear(feat_dim, affinity_dim, bias=False)

        # P_U: visual value projection.
        self.video_value = nn.Linear(feat_dim, feat_dim, bias=False)

        # P_A and P_V: projections into the common fusion space.
        self.audio_fusion = nn.Linear(feat_dim, feat_dim, bias=False)
        self.video_fusion = nn.Linear(feat_dim, feat_dim, bias=False)

        self.fusion_norm = nn.LayerNorm(feat_dim)

        # Existing AV_PLC-style temporal refinement.
        # self.temporal = Conformer(
        #     dim=feat_dim,
        #     depth=depth,
        #     dim_head=64,
        #     heads=heads,
        #     ff_mult=4,
        #     conv_expansion_factor=2,
        #     conv_kernel_size=31,
        #     attn_dropout=dropout,
        #     ff_dropout=dropout,
        #     conv_dropout=dropout,
        # )
        #
        # self.out = nn.Sequential(
        #     nn.LayerNorm(feat_dim),
        #     nn.Linear(feat_dim, mel_dim),
        # )

        offsets = torch.arange(-self.max_offset, self.max_offset + 1, dtype=torch.long,)
        self.register_buffer("offsets", offsets, persistent=False)

    def _validate_inputs(
        self,
        afeat,
        vfeat,
        audio_reliability,
        video_reliability,
    ):
        if afeat is None or vfeat is None:
            raise ValueError(
                "GlobalLocalAffinityFusion requires audio and video features."
            )

        if afeat.dim() != 3 or vfeat.dim() != 3:
            raise ValueError("afeat and vfeat must be [B, T, D].")

        if afeat.shape != vfeat.shape:
            raise ValueError(
                "Audio and video features must have equal [B,T,D] shapes. "
                f"Got {tuple(afeat.shape)} and {tuple(vfeat.shape)}."
            )

        batch, steps, dim = afeat.shape

        if dim != self.feat_dim:
            raise ValueError(
                f"Expected feature dimension {self.feat_dim}, got {dim}."
            )

        if audio_reliability.shape != (batch, steps):
            raise ValueError(
                "audio_reliability must be [B,T], got "
                f"{tuple(audio_reliability.shape)}."
            )

        if video_reliability is not None:
            if video_reliability.shape != (batch, steps):
                raise ValueError(
                    "video_reliability must be [B,T], got "
                    f"{tuple(video_reliability.shape)}."
                )

    def _compute_affinity(self, afeat, vfeat):
        """
        S[t,j] = normalized(P_a(A_t))^T normalized(P_v(V_j))
        """
        audio = self.audio_norm(afeat)
        video = self.video_norm(vfeat)

        queries = F.normalize(
            self.audio_query(audio).float(),
            p=2,
            dim=-1,
            eps=self.eps,
        )
        keys = F.normalize(
            self.video_key(video).float(),
            p=2,
            dim=-1,
            eps=self.eps,
        )

        # [B, T_audio, T_video]
        affinity = torch.matmul(queries, keys.transpose(1, 2))
        return affinity

    def _compute_global_offset(
        self,
        affinity,
        audio_reliability,
        video_reliability,
    ):
        """
        Compute reliability-weighted affinity along each candidate diagonal.

        z_k =
            sum_t r_a(t) r_v(t+k) S[t,t+k]
            --------------------------------
            sum_t r_a(t) r_v(t+k) + eps
        """
        batch, steps, _ = affinity.shape
        device = affinity.device

        positions = torch.arange(steps, device=device)

        offset_scores = []
        offset_supports = []

        for offset in self.offsets.tolist():
            video_positions = positions + offset
            valid = (
                (video_positions >= 0)
                & (video_positions < steps)
            )

            audio_idx = positions[valid]
            video_idx = video_positions[valid]

            # [B, number_of_valid_diagonal_positions]
            pair_reliability = (
                audio_reliability[:, audio_idx]
                * video_reliability[:, video_idx]
            )

            diagonal_affinity = affinity[
                :, audio_idx, video_idx
            ]

            support = pair_reliability.sum(dim=-1)

            numerator = (
                pair_reliability * diagonal_affinity
            ).sum(dim=-1)

            score = numerator / support.clamp_min(self.eps)

            offset_scores.append(score)
            offset_supports.append(support)

        # [B, K]
        scores = torch.stack(offset_scores, dim=-1)
        supports = torch.stack(offset_supports, dim=-1)

        supported = supports >= self.min_offset_support

        # Unsupported offsets must not enter the posterior.
        logits = scores / self.global_temperature
        logits = logits.masked_fill(~supported, -1.0e4)

        # If a sample has no supported offset, fall back to k=0.
        no_supported_offset = ~supported.any(dim=-1)

        if no_supported_offset.any():
            zero_index = self.max_offset

            logits = logits.clone()
            logits[no_supported_offset] = -1.0e4
            logits[no_supported_offset, zero_index] = 0.0

        posterior = torch.softmax(logits, dim=-1)

        return posterior, supports

    def _build_global_prior(
        self,
        offset_posterior,
        video_reliability,
    ):
        """
        G[t,j] proportional to

            r_v(j) * sum_k p_g(k)
            exp(-(j - (t+k))^2 / (2 sigma^2))
        """
        batch, steps = video_reliability.shape
        device = video_reliability.device

        audio_positions = torch.arange(
            steps, device=device, dtype=torch.float32
        ).view(1, steps, 1)

        video_positions = torch.arange(
            steps, device=device, dtype=torch.float32
        ).view(1, 1, steps)

        prior = torch.zeros(
            batch,
            steps,
            steps,
            device=device,
            dtype=torch.float32,
        )

        variance_scale = 2.0 * self.prior_sigma**2

        # Looping over offsets avoids allocating [B,K,T,T].
        for offset_index, offset in enumerate(self.offsets.tolist()):
            centre = audio_positions + float(offset)

            gaussian = torch.exp(
                -((video_positions - centre) ** 2)
                / variance_scale
            )  # [1,T,T]

            offset_probability = offset_posterior[
                :, offset_index
            ].view(batch, 1, 1)

            prior = prior + offset_probability * gaussian

        # Remove or down-weight unreliable video positions.
        prior = prior * video_reliability[:, None, :]

        row_sum = prior.sum(dim=-1, keepdim=True)

        # Rows with no reliable video remain exactly zero.
        prior = torch.where(
            row_sum > self.eps,
            prior / row_sum.clamp_min(self.eps),
            torch.zeros_like(prior),
        )

        return prior

    def _safe_masked_softmax(self, logits, allowed):
        """
        Softmax that returns all zeros when a row has no valid video key.
        """
        logits = logits.masked_fill(~allowed, -1.0e4)

        weights = torch.softmax(logits, dim=-1)
        weights = weights * allowed.to(weights.dtype)

        denominator = weights.sum(dim=-1, keepdim=True)

        weights = torch.where(
            denominator > self.eps,
            weights / denominator.clamp_min(self.eps),
            torch.zeros_like(weights),
        )

        return weights

    def _compute_alignment(self, affinity, global_prior, audio_reliability, video_reliability,):
        """
        Reliable audio:
            content affinity + global timing prior.

        Missing audio:
            global timing prior only.
        """
        observed_logits = (
            affinity / self.local_temperature
            + self.prior_strength
            * torch.log(global_prior.clamp_min(self.eps))
        )

        # Completely unavailable video positions cannot be selected.
        allowed_video = video_reliability > 0.0
        allowed = allowed_video[:, None, :].expand_as(observed_logits)

        observed_alignment = self._safe_masked_softmax(observed_logits, allowed,)

        # The prior is already normalized and reliability masked.
        gap_alignment = global_prior
        audio_gate = audio_reliability.unsqueeze(-1)
        alignment = (audio_gate * observed_alignment + (1.0 - audio_gate) * gap_alignment)

        return alignment

    def forward(self, afeat, vfeat, audio_reliability, video_reliability=None,): #return_diagnostics=False,):

        self._validate_inputs(afeat, vfeat, audio_reliability, video_reliability,)

        batch, steps, _ = afeat.shape
        original_dtype = afeat.dtype

        audio_reliability = audio_reliability.to(device=afeat.device, dtype=torch.float32,).clamp(0.0, 1.0)

        if video_reliability is None:
            video_reliability = torch.ones(batch, steps, device=vfeat.device, dtype=torch.float32,)
        else:
            video_reliability = video_reliability.to(device=vfeat.device, dtype=torch.float32,).clamp(0.0, 1.0)

        # Perform affinity and probability calculations in FP32.
        affinity = self._compute_affinity(afeat, vfeat)

        offset_posterior, offset_support = (self._compute_global_offset(affinity=affinity,
                                                                        audio_reliability=audio_reliability,
                                                                        video_reliability=video_reliability,))

        global_prior = self._build_global_prior(offset_posterior=offset_posterior,
                                                video_reliability=video_reliability,)

        alignment = self._compute_alignment(affinity=affinity, global_prior=global_prior,
                                            audio_reliability=audio_reliability,
                                            video_reliability=video_reliability,)

        # Video values [B,T,D].
        video_values = self.video_value(self.video_norm(vfeat)).float()

        # Warp video features onto the audio timeline.
        aligned_video = torch.matmul(alignment, video_values,)  # [B,T,D]
        audio_features = self.audio_fusion(self.audio_norm(afeat)).float()
        visual_features = self.video_fusion(aligned_video.to(self.video_fusion.weight.dtype)).float()

        # Hard reliability-guided replacement.
        audio_gate = audio_reliability.unsqueeze(-1)
        fused_features = (audio_gate * audio_features + (1.0 - audio_gate) * visual_features)
        fused_features = self.fusion_norm(fused_features.to(original_dtype))

        #refined = self.temporal(fused_features)
        #fused_mel = self.out(refined).permute(0, 2, 1)

        #if not return_diagnostics:
        #    return fused_mel

        #diagnostics = {
        #    "affinity": affinity.detach(),
        #    "offset_posterior": offset_posterior.detach(),
        #    "offset_support": offset_support.detach(),
        #    "global_prior": global_prior.detach(),
        #    "alignment": alignment.detach(),
        #    "offsets": self.offsets.detach().clone(),
        #}

        #return fused_mel, diagnostics
        return fused_features