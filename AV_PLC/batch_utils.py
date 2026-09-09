"""Small helpers for the additive AV_PLC waveform batch fields."""
from __future__ import annotations


def split_waveform_aux(batch):
    """Return ``(core, audio, sample_mask, frame_valid, soft_keep)``.

    New AV_PLC datasets append exactly four fields to the historical core tuple.
    Keeping the core order unchanged minimizes breakage in older AV_PLC scripts.
    """
    if len(batch) < 4:
        raise ValueError(f"AV_PLC batch is too short: {len(batch)}")
    core = batch[:-4]
    audio, sample_mask, frame_valid, soft_keep = batch[-4:]
    return core, audio, sample_mask, frame_valid, soft_keep
