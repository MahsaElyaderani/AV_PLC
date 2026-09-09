# AV_PLC waveform-domain + aligned-video update

## Scope and compatibility

This update changes AV_PLC to apply packet loss to the stored waveform before its own STFT/log-Mel frontend. AV_LSTM, AV_S2S and AV_Transformer remain spectrogram-domain reproductions. All four projects now generate the same deterministic 10-ms packet trace over **valid audio only** for validation/test; the three baselines apply that trace to their stored `spec`, while AV_PLC expands it to waveform samples before STFT.

Existing shared HDF5 names are preserved. The important data layout is:

- `audio`: new clean padded waveform, float32 or int16 PCM
- `audio_len`: true number of valid waveform samples
- `spec`: unchanged legacy log-Mel used by AV_LSTM/AV_S2S/AV_Transformer
- `frames`: new AV_PLC reference-face-aligned 96x96 grayscale mouth ROI
- `landmarks`: unchanged legacy lip landmarks
- `full_landmarks`: new full MediaPipe landmarks
- `visual_features`: unchanged AV-HuBERT feature-extraction path
- `text`, `phone_indices`, `spkr_embd`: retained
- `video_len`, `video_path`: retained/provided as metadata

`visual_features` are deliberately generated from the same legacy 112x112 dynamic mouth ROI that the uploaded project used. The new aligned `frames` path is separate and does not alter the AV-HuBERT comparison representation.

## Important latency limitation

`--frontend-lookahead-ms` controls only the STFT analysis-window placement. The current temporal Conformers/fusion are not causalized by this patch. Therefore a zero-lookahead frontend is **not yet equivalent to a complete sub-40-ms real-time AV_PLC system**. Network context/causalization must be addressed separately before making an end-to-end real-time latency claim.

## 1. Configure dataset root

The default layout is `../datasets` next to this `models` directory. Otherwise set:

```bash
export AV_DATA_ROOT=/absolute/path/to/datasets
```

Run commands from the `models` directory so `python -m ...` resolves the project packages consistently.

## 2. Build one fixed reference face from TRAINING data only

Use representative training videos. If one common template is desired across datasets, provide training globs from all desired datasets in the same invocation. Never include validation/test clips.

```bash
python -m AV_PLC.build_reference_face \
  --videos "/path/to/grid/train/**/*.mpg" "/path/to/lrs2/train/**/*.mp4" \
  --output AV_PLC/reference_face.npy \
  --max-faces 2000 \
  --frame-stride 10
```

This requires MediaPipe. Inspect a sample of resulting aligned crops before bulk regeneration.

## 3. Rebuild HDF5 features

Before rebuilding, make sure the existing AV-HuBERT feature extractor is configured and `load_avhubert()` succeeds (the current code mentions `AVHUBERT_CKPT`). If AV-HuBERT fails to load, `visual_features` will not be produced and AV_Transformer cannot use that regenerated HDF5.

Example GRID split:

```bash
python dataset/features/save_features.py \
  --videos "/path/to/grid/train/**/*.mpg" \
  --output "$AV_DATA_ROOT/grid/grid_train_features.h5" \
  --reference-face AV_PLC/reference_face.npy \
  --audio-dtype float32 \
  --num-workers 2 \
  --chunk-size 1000
```

This writes files such as `grid_train_features_chunk0.h5` and a metadata text file. Repeat for validation/test and for LRS2/VoxCeleb2 with names matching `evaluations/runtime_config.py` patterns.

For maximum preservation use `--audio-dtype float32`. To halve waveform storage use `--audio-dtype int16`; AV_PLC decodes it exactly back to the stored PCM grid as `float32 / 32768.0`.

## 4. Compute and save AV_PLC Mel statistics ONCE per dataset

The real GRID/LRS2/VoxCeleb2 HDF5 files were not included in the uploaded archive, so this package intentionally contains no fabricated numerical statistics. After rebuilding the real training HDF5 files run:

```bash
python -m AV_PLC.compute_audio_stats --dataset grid
python -m AV_PLC.compute_audio_stats --dataset lrs2
python -m AV_PLC.compute_audio_stats --dataset voxceleb2
```

These create:

```text
AV_PLC/mel_stats/grid.json
AV_PLC/mel_stats/lrs2.json
AV_PLC/mel_stats/voxceleb2.json
```

Each file is computed once from clean valid training frames at the 7.5-ms reference geometry and is reused for every frontend-lookahead experiment. AV_PLC refuses to silently substitute the old legacy statistics if the required file is absent.

You can override a path with `--mel-stats /path/to/file.json` when running AV_PLC.

## 5. Run the repository smoke test

```bash
python -m AV_PLC.integration_smoke_test
```

It uses a synthetic non-3-second HDF5 sample and checks valid-length masking, cross-project deterministic packet traces, STFT geometry and inversion, plus AV_PLC tensor plumbing. If the external `conformer` package is unavailable, the script clearly reports that it used an identity-Conformer stub **only for shape plumbing**.

## 6. Train AV_PLC

### Audio-only

```bash
python -m AV_PLC.runner \
  --mode a \
  --phase train \
  --dataset grid \
  --frontend-lookahead-ms 7.5
```

### Audio-video

```bash
python -m AV_PLC.runner \
  --mode av \
  --phase train \
  --dataset grid \
  --frontend-lookahead-ms 7.5
```

### Audio-video with learned phase reconstruction

```bash
python -m AV_PLC.runner \
  --mode av \
  --phase train \
  --dataset grid \
  --frontend-lookahead-ms 7.5 \
  --phase-reconstruction
```

Optional perceptual losses use the same active dataset Mel statistics:

```bash
--pmsqe
--asr
```

`--asr` requires OpenAI Whisper. PMSQE paths require the repository's existing Asteroid/PMSQE dependency setup.

Valid frontend lookahead with the current 25-ms window/10-ms hop is 0 to 15 ms. The default 7.5 ms is exactly time-aligned with the old `center=False + 176/176` STFT geometry.

## 7. Test AV_PLC

Example full test sweep:

```bash
python -m AV_PLC.runner \
  --mode av \
  --phase test \
  --dataset grid \
  --frontend-lookahead-ms 7.5 \
  --mask_types gilbert single_gap
```

To test a phase-enabled checkpoint, add the same `--phase-reconstruction` flag used when training it.

The deterministic Gilbert/single-gap packet traces are generated only over `audio_len`. Padded audio is never considered a lost packet.

## 8. Run the three comparison projects

Their model representations are unchanged. Their dataset masking was only corrected to use valid audio length and the same deterministic packet trace as AV_PLC.

```bash
python -m AV_LSTM.runner --mode av --phase train --dataset grid
python -m AV_S2S.runner --mode av --phase train --dataset grid
python -m AV_Transformer.runner --mode av --phase train --dataset grid
```

Testing examples:

```bash
python -m AV_LSTM.runner --mode av --phase test --dataset grid
python -m AV_S2S.runner --mode av --phase test --dataset grid
python -m AV_Transformer.runner --mode av --phase test --dataset grid
```

For a given test sample/seed/condition, these baselines and AV_PLC now start from the same missing 10-ms packet trace. AV_PLC's later **STFT hard-reliability mask is expected to cover more frames** because a 25-ms analysis window overlaps neighboring packets; that is correct and should not be confused with a different underlying packet-loss trace.

## 9. Phase diagnostics

The active phase diagnostic scripts were updated so phase-enabled models receive observed complex STFT values from `clean_audio * sample_mask`, not clean target phase/magnitude. Clean phase is retained only for target/oracle comparisons.

The active scripts are:

```text
AV_PLC/grid_phase_oracle_diagnostic.py
AV_PLC/grid_phase_oracle_followup.py
AV_PLC/grid_phase_common_rotation_final_diagnostic.py
```

Use them only with checkpoints matching the new waveform/aligned-frame architecture and the corresponding saved dataset stats.
