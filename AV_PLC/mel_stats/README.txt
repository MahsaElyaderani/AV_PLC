This directory intentionally contains no fabricated statistics.
Generate each dataset's fixed AV_PLC statistics once from its real clean training HDF5 files:

  python -m AV_PLC.compute_audio_stats --dataset grid
  python -m AV_PLC.compute_audio_stats --dataset lrs2
  python -m AV_PLC.compute_audio_stats --dataset voxceleb2

Each command writes <dataset>.json here. The AV_PLC dataloader refuses to run if the required file is absent.
