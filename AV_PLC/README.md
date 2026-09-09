# AV_PLC parallel magnitude–phase patch (`parallel_mp_v1`)

This patch implements the **first-stage** redesign discussed on 2026-09-01. It intentionally does **not** implement the later waveform-direct frontend, latency-flexible inference, adversarial training, or temporal jitter redesign.

## Target architecture

1. Keep the existing content pathway and its pretrained/frozen components:
   - audio encoder -> `M_A` + audio latent
   - video encoder -> `M_V` + video latent
   - A/V fusion -> fused latent -> supervised `M_AV`
2. Choose the content Mel for each sample:
   - audio-only content: `M_C = M_A`
   - video-only content: `M_C = M_V`
   - audio-video content: `M_C = M_AV`
3. Convert `M_C` to an approximate linear STFT magnitude with the existing fixed Slaney Mel pseudo-inverse.
4. **Protect observed physical magnitude** before TFRefine:

   `A_complete = R * A_observed + (1-R) * A_invMel`

   This is differentiable: gradients reach the Mel-derived magnitude only where `R=0`.
5. Power-compress the completed magnitude with **c=0.3**:

   `A_complete_c = (A_complete + eps) ** 0.3`

   `c=1.0` is retained only as the no-compression ablation.
6. Mask phase only after circular conversion:

   `C_obs = R * cos(phi_gt)`

   `S_obs = R * sin(phi_gt)`

7. Feed `[A_complete_c, C_obs, S_obs]` to a shared TFRefine backbone:

   `2-D Conv -> dilated Dense TF -> axial time/frequency Conformer blocks`
8. Split into two independent heads:
   - magnitude head -> compressed magnitude `A_pred_c ≈ A**0.3`
   - phase head -> pseudo real/imag -> unit `(cos(phi_pred), sin(phi_pred))`

   Decompress the predicted magnitude before physical reconstruction:

   `A_pred = (A_pred_c + eps) ** (1/0.3)`
9. Protect observed outputs again:

   `A_final = R * A_observed + (1-R) * A_pred`

   observed phase is copied exactly; predicted phase is used only for missing frames.
10. Reconstruct the complex STFT and waveform for losses/evaluation:

   `S_hat = A_final * (cos(phi_final) + j sin(phi_final))`

   followed by differentiable overlap-add iSTFT matching the current AV_PLC STFT geometry.

## Losses implemented

The new parallel M/P training path uses:

- three content-Mel L1 terms: `L_A_mel`, `L_V_mel`, `L_AV_mel`
- `L_mag`: masked L1 in the compressed magnitude domain: `|A_pred_c - A_gt**0.3|`
- `L_phi = L_IP + L_GD + L_IAF` using anti-wrapping phase errors
- `L_complex`: relative Cartesian complex-STFT L1
- `L_waveform`: waveform L1 after differentiable reconstruction

### Important frozen-backbone note

The ablation script still loads the complete successful no-phase checkpoint and freezes all inherited AV_PLC parameters, then trains only `phase_completion.*` (which now contains **both** magnitude and phase heads). The three Mel losses can still be logged, but if the corresponding heads are frozen they cannot update those heads. If you want the new `M_AV` supervision to *train* the fusion/content decoder, first train/update the no-phase content checkpoint with fused-Mel supervision, then freeze it for the controlled parallel M/P experiment.

## Modality/reliability semantics

`avail = [audio_available, video_available]` selects `M_A`, `M_V`, or `M_AV`.
For the current strict modality ablation, **video-only means no usable audio anywhere**.
Therefore:

- audio-only / AV: `R` is the PLC packet-reliability mask;
- video-only: `R=0` for the whole utterance.

Consequently the video-only TFRefine input is
`[InvMel(M_V)**0.3, 0, 0]`, while audio-only/AV preserve observed physical
magnitude and phase outside packet-loss regions.

## Changed files

Copy these files into the corresponding project locations:

- `AV_PLC/phase_completion.py`
- `AV_PLC/multimodal_decoder.py`
- `AV_PLC/losses.py`
- `AV_PLC/precompute_phase.py`
- `AV_PLC/av_dataloader.py`
- `AV_PLC/trainer.py`
- `AV_PLC/ablation_fusion.py`

No structural changes are required at this stage in:

- `AV_PLC/audio_encoder.py`
- `AV_PLC/video_encoder.py`
- `AV_PLC/fusion.py`
- `AV_PLC/spectral_completion.py`
- `AV_PLC/resnet_.py`

The existing audio/video encoder outputs and fusion latent interface are reused.

## Installation / migration

### 1. Back up the current project

Do not overwrite your working experiment branch without a checkpoint/code backup.

### 2. Copy the patch files

From the unpacked patch root, copy `AV_PLC/*.py` over the project counterparts.

### 3. Recompute STFT metadata

The old preprocessing stored phase only. `parallel_mp_v1` also needs clean linear STFT magnitude on the **same** grid.

Run:

```bash
python AV_PLC/precompute_phase.py --datasets grid lrs2 voxceleb2 --splits train val test --overwrite
```

This writes, per HDF5 sample:

- `<sample>/phase`
- `<sample>/stft_magnitude`

using the current geometry: 16 kHz, `n_fft=512`, `win_length=400`, `hop_length=160`, `center=False`, Hann, and 176-sample boundary padding.

For a quick first test, restrict `--datasets` and `--splits` instead of rewriting everything.

### 4. Train/verify the content checkpoint first

The parallel M/P ablation assumes the method-specific no-phase content model already exists. In particular, verify that its AV path produces a supervised `M_AV` comparable to the audio/video Mel heads.

If your existing checkpoint predates the fused-Mel supervision change, retrain/update that no-phase content checkpoint before treating it as the frozen baseline for the new experiment.

### 5. Run the new parallel magnitude/phase ablation

Use the same command style as the previous `phase_tf_v1` run, but the `--phase-reconstruction` flag now selects `parallel_mp_v1`.

Example:

```bash
python AV_PLC/ablation_fusion.py \
  --dataset lrs2 \
  --methods concat temporal_self_cross_attention global_local_affinity \
  --phase-reconstruction \
  --num-epochs 100 \
  --phase-batch-size 1
```

Magnitude compression exposed by the script:

- `--phase-mag-compression` (default **0.3**; use `1.0` only for the raw-magnitude ablation)

The same exponent is used for TFRefine magnitude input, magnitude-head prediction/target,
and `L_mag`. Magnitude is decompressed before `L_complex`, `L_waveform`, and iSTFT.

Starting loss weights exposed by the script:

- `--w-magnitude` (default 1.0)
- `--w-phase-unit` = `L_IP` (default 0.10)
- `--w-phase-frequency` = `L_GD` (default 0.05)
- `--w-phase-temporal` = `L_IAF` (default 0.05)
- `--w-complex` (default 0.10)
- `--w-waveform` (default 1.0)

Treat these as starting values, not literature-proven optimal AV_PLC weights.

### 6. Start without temporal jitter

For the first architecture validation, omit `--temporal-jitter`. Once the parallel M/P path is stable, enable your planned jitter experiment separately so architectural gains are not confounded with synchronization augmentation.

## Sanity checks before a long run

1. Verify HDF5 `phase` and `stft_magnitude` both have `[257,T]` and the same `T` as Mel.
2. For one batch, assert `A_complete == A_observed` wherever `R=1`.
3. Assert final magnitude and final `(cos,sin)` exactly match observed values wherever `R=1` (within floating-point tolerance).
4. Check `predicted_mag_compressed` and decompressed `predicted_mag` are finite and non-negative.
5. Check `pred_cos**2 + pred_sin**2 ~= 1`.
6. Backprop one batch and confirm:
   - `phase_completion.*` gets nonzero gradients;
   - frozen encoder/fusion/content parameters get no gradients in the controlled ablation.
7. Compare reconstructed clean audio using **ground-truth magnitude + ground-truth phase** through the new overlap-add function. It should closely reproduce the source under the same STFT geometry. This isolates inverse-STFT implementation errors before training.

## What is intentionally deferred

After this version is validated, the planned sequence is:

1. waveform-direct model input with in-model STFT/iSTFT;
2. latency-flexible context/causal masking;
3. temporal-jitter experiments under the latency-aware setup.

Adversarial training is intentionally excluded.
