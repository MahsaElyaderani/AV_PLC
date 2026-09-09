# AV_PLC update test report

Tested working tree: `models` from the uploaded `models(7).zip`, patched in this session.

## Passed

1. **Repository-wide compile**: `python -m compileall -q .` — PASS.
2. **Synthetic HDF5, non-3-second valid audio**: 48,000 stored samples with `audio_len=34,123` — PASS.
3. **AV_PLC valid-only masking**: packet corruption restricted to `[0,audio_len)` — PASS.
4. **Cross-project deterministic packet trace equality** — PASS for:
   - fixed-rate Gilbert-Elliott (60% test condition),
   - fixed-seed validation random-rate mask,
   - deterministic 500-ms single gap,
   across AV_PLC, AV_LSTM, AV_S2S and AV_Transformer.
5. **Padded tail never lost** — PASS for AV_PLC sample mask and all three baseline spectrogram masks.
6. **7.5-ms STFT vs legacy geometry** — PASS, complex STFT maximum absolute difference = `0.0` for the test waveform.
7. **Matching WOLA/iSTFT round trip** — PASS:
   - mean absolute error = `4.343164050624182e-09`
   - maximum absolute error = `4.470348358154297e-08`
8. **88x88 Video_Encoder tensor plumbing** — PASS with explicit identity-Conformer test stub:
   - input `[1,75,88,88]`
   - feature `[1,300,256]`
   - Mel `[1,80,300]`
9. **Synthetic AV_PLC forward tensor plumbing** — PASS with the same explicit identity-Conformer test stub.
10. **Active ablation/diagnostic compilation** — PASS for:
    - `ablation.py`
    - `ablation_av_sync_shift.py`
    - `ablation_fusion.py`
    - `ablation_gap_sweep.py`
    - `ablation_synth_losses.py`
    - `ablation_video_reduction.py`
    - `check_weights.py`
    - `grid_phase_oracle_diagnostic.py`
    - `grid_phase_oracle_followup.py`
    - `grid_phase_common_rotation_final_diagnostic.py`
11. **Dataset-stat JSON writer** — PASS on synthetic training HDF5. Synthetic values are test-only and are NOT shipped as real dataset stats.
12. **Whisper Mel conversion uses active stats** — PASS with import-only Whisper stub; conversion changes when active `mel_mean` changes.
13. **int16 HDF5 decode** — PASS; loaded float32 equals `stored_int16.astype(float32)/32768.0` exactly.
14. **Phase-observation logic audit** — active trainer and active phase diagnostics now derive observed phase/magnitude from the masked waveform, preventing clean-phase leakage into PLC gaps.

## Environment limitations

The execution image used for this verification has:

- PyTorch `2.10.0+cpu`
- torchaudio `2.10.0+cpu`
- h5py `3.15.1`
- OpenCV `4.13.0`

but does NOT have these repository runtime dependencies installed:

- `conformer`
- `mediapipe`
- `whisper`
- `asteroid`
- `resemblyzer`

Therefore:

- the real Conformer numerical forward was not executed here; only tensor plumbing was verified with an explicitly temporary identity stub inside `integration_smoke_test.py`;
- MediaPipe reference-face extraction/alignment could be compiled and statically reviewed, but not executed in this container;
- the real Whisper encoder and PMSQE/Asteroid losses could not be executed; their active-stat wiring was verified in code, and the pure Whisper Mel conversion was unit-tested with an import stub;
- real GRID/LRS2/VoxCeleb2 Mel statistics could not be computed because those HDF5 datasets were not supplied. The code requires you to compute and save them once from the real training data before AV_PLC training.

## Scientific boundary

This update parameterizes STFT frontend lookahead but does not make all Conformers/fusion blocks causal. End-to-end real-time latency claims require a separate network-context/causalization pass.
