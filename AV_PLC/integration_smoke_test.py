"""Repository smoke tests for the waveform-domain AV_PLC update.

This script uses a synthetic HDF5 sample and does not require real datasets.
It validates data/masking/STFT logic directly.  If the external ``conformer``
package is unavailable, a temporary identity stub is used ONLY for tensor-shape
plumbing tests; that limitation is printed explicitly.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import types
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluations.runtime_config import SEED
from AV_PLC.audio_frontend import AudioFrontend, AudioFrontendConfig
from AV_PLC.compute_audio_stats import compute
from AV_PLC.av_dataloader import AVDataset as PLCData
from AV_LSTM.av_l_dataset import AVDataset as LSTMData
from AV_S2S.av_l_dataset import AVDataset as S2SData
from AV_Transformer.av_l_dataset import AVDataset as TransformerData


def _make_synthetic_h5(path: Path, audio_len: int = 34123) -> None:
    rng = np.random.default_rng(123)
    n = 48000
    t = np.arange(audio_len, dtype=np.float32) / 16000.0
    valid = (0.15 * np.sin(2 * np.pi * 220.0 * t) +
             0.04 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    audio = np.zeros(n, dtype=np.float32)
    audio[:audio_len] = valid

    spec = (-60.0 + 8.0 * rng.standard_normal((80, 300))).astype(np.float32)
    frames = rng.integers(1, 255, size=(75, 96, 96, 1), dtype=np.uint8)
    landmarks = rng.normal(size=(75, 40, 2)).astype(np.float32)
    visual_features = rng.normal(size=(75, 768)).astype(np.float32)
    text = np.zeros(128, dtype=np.int32)
    phone_indices = np.zeros(128, dtype=np.int32)
    spk = rng.normal(size=256).astype(np.float32)

    with h5py.File(path, "w") as h5f:
        g = h5f.create_group("video_0")
        g.create_dataset("audio", data=audio)
        g.create_dataset("spec", data=spec)
        g.create_dataset("frames", data=frames)
        g.create_dataset("landmarks", data=landmarks)
        g.create_dataset("visual_features", data=visual_features)
        g.create_dataset("text", data=text)
        g.create_dataset("phone_indices", data=phone_indices)
        g.create_dataset("spkr_embd", data=spk)
        g.create_dataset("mask", data=np.ones((80, 300), dtype=np.float32))
        g.create_dataset("mask_60", data=np.ones((80, 300), dtype=np.float32))
        h5f.attrs["video_0/audio_len"] = int(audio_len)
        h5f.attrs["video_0/video_len"] = 75
        h5f.attrs["video_0/video_path"] = "/synthetic/grid/video_0.mpg"


def _mask_from_baseline(item, which: str):
    if which == "lstm":
        return np.asarray(item[4])
    if which in {"s2s", "transformer"}:
        return np.asarray(item[3])
    raise ValueError(which)


def test_synthetic_hdf5_and_masks(tmp: Path):
    h5 = tmp / "grid_test_features_chunk0.h5"
    _make_synthetic_h5(h5)
    pattern = str(tmp / "grid_test_features_chunk*.h5")
    stats = compute(pattern, lookahead_ms=7.5)

    plc = PLCData(pattern, mode="a", mask_range="60", augment=False,
                  set_seed=SEED, mel_mean=stats["mean"], mel_std=stats["std"])
    lstm = LSTMData(pattern, mode="a", mask_range="60", set_seed=SEED)
    s2s = S2SData(pattern, mode="a", mask_range="60", set_seed=SEED)
    tr = TransformerData(pattern, mode="a", mask_range="60", set_seed=SEED)

    audio_len = 34123
    valid_t = math.ceil(audio_len / 160)
    sample_id = f"{h5.name}:video_0"
    trace = plc._packet_trace(audio_len, sample_id)

    plc_item = plc[0]
    sample_mask = np.asarray(plc_item[-3])
    frame_valid = np.asarray(plc_item[-2])
    assert int(plc_item[2]) == audio_len
    assert plc_item[0].shape == (80, 300)
    assert plc_item[1].shape == (80, 300)
    assert sample_mask.shape == (48000,)
    assert frame_valid.shape == (300,)
    assert np.all(sample_mask[audio_len:] == 1.0), "AV_PLC padded samples were marked lost"
    np.testing.assert_array_equal(sample_mask[:audio_len:160], trace)

    masks = {
        "lstm": _mask_from_baseline(lstm[0], "lstm"),
        "s2s": _mask_from_baseline(s2s[0], "s2s"),
        "transformer": _mask_from_baseline(tr[0], "transformer"),
    }
    for name, mask in masks.items():
        np.testing.assert_array_equal(mask[0, :valid_t], trace)
        assert np.all(mask[:, valid_t:] == 1.0), f"{name} padded Mel tail was marked lost"
    np.testing.assert_array_equal(masks["lstm"], masks["s2s"])
    np.testing.assert_array_equal(masks["lstm"], masks["transformer"])

    # Deterministic validation-random masks must also match exactly.
    plc_v = PLCData(pattern, mode="a", mask_range="rand", augment=False,
                    set_seed=SEED, online_loss_bounds=(0.3, 0.9),
                    mel_mean=stats["mean"], mel_std=stats["std"])
    lstm_v = LSTMData(pattern, mode="a", mask_range="rand", set_seed=SEED,
                      online_loss_bounds=(0.3, 0.9))
    s2s_v = S2SData(pattern, mode="a", mask_range="rand", set_seed=SEED,
                    online_loss_bounds=(0.3, 0.9))
    tr_v = TransformerData(pattern, mode="a", mask_range="rand", set_seed=SEED,
                           online_loss_bounds=(0.3, 0.9))
    trace_v = plc_v._packet_trace(audio_len, sample_id)
    np.testing.assert_array_equal(_mask_from_baseline(lstm_v[0], "lstm")[0, :valid_t], trace_v)
    np.testing.assert_array_equal(_mask_from_baseline(s2s_v[0], "s2s")[0, :valid_t], trace_v)
    np.testing.assert_array_equal(_mask_from_baseline(tr_v[0], "transformer")[0, :valid_t], trace_v)

    # Single-gap location must match across all projects as well.
    gap_ms = 500
    plc_g = PLCData(pattern, mode="a", mask_range="rand", augment=False,
                    mask_type="single_gap", gap_ms=gap_ms, mask_seed=SEED,
                    mel_mean=stats["mean"], mel_std=stats["std"])
    lstm_g = LSTMData(pattern, mode="a", mask_range="rand", mask_type="single_gap",
                      gap_ms=gap_ms, mask_seed=SEED)
    s2s_g = S2SData(pattern, mode="a", mask_range="rand", mask_type="single_gap",
                    gap_ms=gap_ms, mask_seed=SEED)
    tr_g = TransformerData(pattern, mode="a", mask_range="rand", mask_type="single_gap",
                           gap_ms=gap_ms, mask_seed=SEED)
    trace_g = plc_g._packet_trace(audio_len, sample_id)
    np.testing.assert_array_equal(_mask_from_baseline(lstm_g[0], "lstm")[0, :valid_t], trace_g)
    np.testing.assert_array_equal(_mask_from_baseline(s2s_g[0], "s2s")[0, :valid_t], trace_g)
    np.testing.assert_array_equal(_mask_from_baseline(tr_g[0], "transformer")[0, :valid_t], trace_g)

    return stats


def test_stft_geometry_and_inverse():
    torch.manual_seed(7)
    x = torch.randn(48000, dtype=torch.float32) * 0.05
    frontend = AudioFrontend(AudioFrontendConfig(lookahead_ms=7.5))
    new_z = frontend.stft(x)
    window = torch.hann_window(400, periodic=True, dtype=x.dtype)
    legacy_z = torch.stft(
        F.pad(x, (176, 176)), n_fft=512, hop_length=160, win_length=400,
        window=window, center=False, return_complex=True, onesided=True,
    )
    assert new_z.shape == legacy_z.shape == (257, 300)
    max_diff = float((new_z - legacy_z).abs().max())
    if max_diff != 0.0:
        raise AssertionError(f"7.5-ms STFT differs from legacy geometry: max={max_diff}")

    x_hat = frontend.istft(new_z, length=x.numel())
    mae = float((x_hat - x).abs().mean())
    max_err = float((x_hat - x).abs().max())
    if mae > 1e-6 or max_err > 1e-4:
        raise AssertionError(f"STFT/iSTFT round trip too large: mae={mae}, max={max_err}")
    return max_diff, mae, max_err


def _install_identity_conformer_stub():
    try:
        import conformer  # noqa: F401
        return False
    except ModuleNotFoundError:
        module = types.ModuleType("conformer")
        class Conformer(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
            def forward(self, x, *args, **kwargs):
                return x
        module.Conformer = Conformer
        sys.modules["conformer"] = module
        return True


def test_video_and_av_shapes():
    stubbed = _install_identity_conformer_stub()
    from AV_PLC.video_encoder import Video_Encoder
    from AV_PLC.multimodal_decoder import AV_PLC

    torch.manual_seed(9)
    with torch.no_grad():
        video = torch.randn(1, 75, 88, 88)
        ve = Video_Encoder(conformer_block=1, num_heads=4).eval()
        mel_v, feat_v = ve(video)
        assert feat_v.shape == (1, 300, 256), feat_v.shape
        assert mel_v.shape == (1, 80, 300), mel_v.shape

        model = AV_PLC(video_depth=1, video_heads=4, audio_depth=1, audio_heads=4,
                       phase_reconstruction=False).eval()
        audio_mel = torch.randn(1, 80, 300)
        audio_len = torch.tensor([34123], dtype=torch.long)
        avail = torch.tensor([[True, True]])
        hard_keep = torch.ones(1, 80, 300)
        hard_keep[:, :, 80:130] = 0.0
        fused, amel, vmel, completion = model(
            audio_mel, video, None, audio_len, avail=avail, audio_mask=hard_keep
        )
        assert fused.shape == (1, 80, 300), fused.shape
        assert amel.shape == (1, 80, 300), amel.shape
        assert vmel.shape == (1, 80, 300), vmel.shape
        assert completion["selected_mel"].shape == (1, 80, 300)
        assert completion["completed_mel"].shape == (1, 80, 300)
        # Fully observed frames must be copied from the observed Mel; missing
        # frames use the selected network prediction.
        np.testing.assert_allclose(
            completion["completed_mel"][:, :, :80].cpu().numpy(),
            audio_mel[:, :, :80].cpu().numpy(), atol=1e-6, rtol=0.0,
        )
    return stubbed


def test_whisper_mapping_uses_active_stats():
    # The execution image may not have openai-whisper installed.  Stub only the
    # import so we can unit-test the pure Mel conversion without loading a model.
    had_whisper = "whisper" in sys.modules
    old_whisper = sys.modules.get("whisper")
    if not had_whisper:
        module = types.ModuleType("whisper")
        module.load_model = lambda *args, **kwargs: None
        sys.modules["whisper"] = module
    try:
        from AV_PLC.losses import WhisperASRLoss
        obj = WhisperASRLoss.__new__(WhisperASRLoss)
        nn.Module.__init__(obj)
        obj.mel_mean = -80.0
        obj.mel_std = 20.0
        x = torch.zeros(1, 80, 4)
        y = obj._to_whisper_mel(x)
        # z=0 -> -80 dB -> -8 log10 power; relative clamp is unchanged,
        # then Whisper affine scaling gives (-8 + 4)/4 = -1.
        assert torch.allclose(y, torch.full_like(y, -1.0))
        obj.mel_mean = -60.0
        y2 = obj._to_whisper_mel(x)
        assert not torch.allclose(y, y2), "Whisper conversion ignored active mel_mean"
    finally:
        if not had_whisper:
            sys.modules.pop("whisper", None)
        elif old_whisper is not None:
            sys.modules["whisper"] = old_whisper


def main():
    torch.set_num_threads(1)
    results = {}
    with tempfile.TemporaryDirectory(prefix="avplc_smoke_") as d:
        results["synthetic_stats"] = test_synthetic_hdf5_and_masks(Path(d))
    results["stft"] = dict(zip(
        ["legacy_max_abs_diff", "roundtrip_mae", "roundtrip_max_abs"],
        test_stft_geometry_and_inverse(),
    ))
    results["conformer_identity_stub_used"] = test_video_and_av_shapes()
    test_whisper_mapping_uses_active_stats()
    results["whisper_active_stats_mapping"] = "pass"
    print(json.dumps(results, indent=2))
    print("PASS")


if __name__ == "__main__":
    main()
