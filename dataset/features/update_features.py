from pathlib import Path
import sys

MODELS_ROOT = Path(__file__).resolve().parents[2]
if str(MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(MODELS_ROOT))

import os
import json
import glob
import concurrent.futures
import multiprocessing
import argparse
import subprocess
import shlex

import h5py
import torch
import numpy as np
from tqdm import tqdm

from shared.masking import GilbertElliottModel
from evaluations.runtime_config import DATA_ROOT
# NOTE: VoiceEncoder is intentionally NOT loaded at module level.
# It is instantiated inside process_single_chunk_spkr_embd() so that
# worker processes spawned by multiprocessing don't all load it on import.

SR = 16000
TARGET_DURATION_S = 3.0
DATASET_BASE = DATA_ROOT

def read_audio_ffmpeg(path: str, target_sr: int = SR, max_sec: float = TARGET_DURATION_S):
    """Decode mono float32 audio, trim/pad to exactly ``max_sec``.

    Kept local to this migration module so running the migration does not import
    save_features.py (and therefore does not load AV-HuBERT or speaker models).
    """
    target_len = int(round(target_sr * max_sec))
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(target_sr),
        "-t", str(float(max_sec)), "-f", "f32le", "pipe:1",
    ]
    out = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    a = np.frombuffer(out, dtype=np.float32).copy()
    valid_len = min(int(a.size), target_len)
    if valid_len == 0:
        return None, None
    audio = np.zeros(target_len, dtype=np.float32)
    audio[:valid_len] = a[:valid_len]
    return torch.from_numpy(audio), valid_len


def melspectrogram(audio):

    import torchaudio
    melspctrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=512,
        win_length=400,
        hop_length=160,
        center=False,
        power=1.0,
        norm="slaney",
        onesided=True,
        n_mels=80,
        mel_scale="slaney",
    )
    amp2db_transform = torchaudio.transforms.AmplitudeToDB(stype='magnitude', top_db=80)

    padding = (512 - 160) // 2
    wav = torch.nn.functional.pad(audio, (padding, padding), "constant")
    mel_mag = melspctrogram(wav)
    mel_mag = mel_mag.clamp(min=1e-5)  # avoid log(0)
    logmel = amp2db_transform(mel_mag)

    return logmel

def extract_spectral(audio):

    spec = melspectrogram(audio)
    return spec.cpu().numpy()

def process_single_chunk_spkr_embd(chunk_file, dataset_base=DATASET_BASE):
    # Legacy utility. Not used by the AV_PLC migration path.
    from resemblyzer import VoiceEncoder
    from dataset.features.save_features import embed_from_window_05s
    voice_encoder = VoiceEncoder()  # loaded once per call, not at import

    print(f"Processing: {chunk_file}", flush=True)
    try:
        h5f = h5py.File(chunk_file, 'r+')
    except Exception as e:
        print(f"  FATAL: cannot open {chunk_file}: {e}", flush=True)
        return

    try:
        video_keys = list(h5f.keys())
        for video_key in tqdm(video_keys, desc=os.path.basename(chunk_file),
                              leave=False, mininterval=5.0):
            video_path = h5f.attrs.get(f"{video_key}/video_path", None)
            if video_path is None:
                print(f"  WARNING: no video_path for {video_key}", flush=True)
                continue
            try:
                rel_path = video_path.split("datasets", 1)[-1].lstrip(os.sep)
                path = os.path.join(dataset_base, rel_path)
                audio, valid_len = read_audio_ffmpeg(path, target_sr=SR,
                                                     max_sec=TARGET_DURATION_S)
                spkr_embd = embed_from_window_05s(audio, valid_len, sr=SR,
                                                   voice_encoder=voice_encoder,
                                                   seconds=0.5, policy="max_energy")
                if f"{video_key}/spkr_embd" in h5f:
                    del h5f[f"{video_key}/spkr_embd"]
                h5f.create_dataset(f"{video_key}/spkr_embd",
                                   data=spkr_embd, compression="gzip")
            except ZeroDivisionError as e:
                print(f"  ZeroDivisionError for {video_key}: {e}", flush=True)
                if video_key in h5f:
                    del h5f[video_key]
            except Exception as e:
                print(f"  ERROR for {video_key}: {e}", flush=True)
    finally:
        h5f.flush()
        h5f.close()

    print(f"Completed: {chunk_file}", flush=True)

def process_single_chunk_mel_spec(chunk_file):

    print(f"Processing: {chunk_file}")
    with h5py.File(chunk_file, 'r+') as h5f:
        video_keys = list(h5f.keys())
        for video_key in tqdm(video_keys, desc=f"{os.path.basename(chunk_file)}"):
            video_path = h5f.attrs.get(f"{video_key}/video_path", None)
            if video_path is not None:
                audio, _ = read_audio_ffmpeg(video_path, target_sr=SR, max_sec=TARGET_DURATION_S)

                mel_spec = extract_spectral(audio)
                #print(torch.max(mel_spec))
                #print(torch.min(mel_spec))
                print(mel_spec.shape)

                if f"{video_key}/mel_spec" in h5f:
                    del h5f[f"{video_key}/mel_spec"]
                h5f.create_dataset(f"{video_key}/spec",
                                   data=mel_spec, compression="gzip")
            else:
                print(f"Warning: No video path found for {video_key}")
    print(f"Completed: {chunk_file}")


def rgb_to_gray_uint8(frames):  # frames: (T,H,W,3), uint8 or float
    if frames.dtype != np.uint8:
        f = frames.astype(np.float32)
        if f.max() <= 1.0:
            f = np.clip(f * 255.0, 0, 255).astype(np.uint8)
        else:
            f = np.clip(f, 0, 255).astype(np.uint8)
    else:
        f = frames

    r = f[..., 0].astype(np.uint16)
    g = f[..., 1].astype(np.uint16)
    b = f[..., 2].astype(np.uint16)
    # 0.299, 0.587, 0.114 ≈ 77/256, 150/256, 29/256
    gray = ((77 * r + 150 * g + 29 * b) >> 8).astype(np.uint8)  # (T,H,W)
    return gray

def process_single_chunk_frames(chunk_file, keep_channel_dim=False):
    print(f"Processing: {chunk_file}")
    with h5py.File(chunk_file, "r+") as h5f:
        video_keys = list(h5f.keys())

        for video_key in tqdm(video_keys, desc=os.path.basename(chunk_file), leave=False):
            grp = h5f[video_key]

            if "frames" not in grp:
                continue

            src = grp["frames"]  # expect (T,H,W,3)
            # Skip if already grayscale
            if src.ndim == 3 or (src.ndim == 4 and src.shape[-1] == 1):
                continue
            if src.ndim != 4 or src.shape[-1] != 3:
                print(f"Skipping {video_key}: unexpected frames shape {src.shape}")
                continue

            T, H, W, _ = src.shape

            # Load ALL frames at once (T is small)
            frames_rgb = src[...]                  # (T,H,W,3)
            gray = rgb_to_gray_uint8(frames_rgb)  # (T,H,W)

            # Create a CONTIGUOUS dataset (no chunks, no compression)
            out_shape = (T, H, W, 1) if keep_channel_dim else (T, H, W)
            tmp_name = f"{video_key}/_frames_gray_tmp"
            if tmp_name in h5f:
                del h5f[tmp_name]

            dst = h5f.create_dataset(tmp_name, shape=out_shape, dtype=np.uint8)

            if keep_channel_dim:
                dst[..., 0] = gray
            else:
                dst[...] = gray

            # Swap in atomically; remove old RGB
            grp.move("frames", "_frames_rgb_old")
            h5f.move(tmp_name, f"{video_key}/frames")
            del grp["_frames_rgb_old"]

    print(f"Completed: {chunk_file}")

def process_single_chunk_frames_sizes(chunk_file, keep_channel_dim=True, min_valid=10):
    """
    Upgrade frames to [75,112,112,1] (or [75,112,112]) by:
      1) dropping zero-padded frames,
      2) temporal interpolation to 75 frames,
      3) spatial resize to 112x112.

    NOTE: This performs *temporal warping* of the remaining valid frames.
    If you intended to fill only the zero-gaps in-place (no warping), this is NOT that.
    """
    target_T, target_H, target_W = 75, 112, 112

    print(f"Processing: {chunk_file}")
    with h5py.File(chunk_file, "r+") as h5f:
        video_keys = list(h5f.keys())

        for video_key in tqdm(video_keys, desc=os.path.basename(chunk_file), leave=False):
            grp = h5f[video_key]
            if "frames" not in grp:
                continue

            src = grp["frames"]  # HDF5 dataset

            # Load into memory as numpy (safer than fancy indexing on h5 objects)
            arr = src[()]  # expect (T,H,W,1) or (T,H,W)
            if arr.ndim == 3:
                # (T,H,W) -> add channel dim
                arr = arr[..., None]
            elif arr.ndim != 4 or arr.shape[-1] != 1:
                # Unexpected shape; skip safely
                continue

            T0, H0, W0, C0 = arr.shape
            if (H0, W0) == (96, 96):
                # New AV_PLC aligned frames are already in their canonical
                # stored geometry.  Never turn them back into legacy 112x112.
                continue
            # Identify non-zero frames across all pixels/channels
            valid_mask = np.any(arr != 0, axis=(1, 2, 3))
            arr_valid = arr[valid_mask]  # (T_valid, H0, W0, 1)

            # If too few valid frames, drop the whole video safely
            if arr_valid.shape[0] < min_valid:
                # delete the *group* under the file (not `del grp`)
                del h5f[video_key]
                continue

            # Convert to torch for interpolation: (N,C,D,H,W)
            # Squeeze channel to (T_valid,H0,W0), then permute to (D,H,W)
            x = torch.from_numpy(arr_valid.squeeze(-1))  # uint8 -> (T_valid,H0,W0)
            # Cast to float for interpolation
            x = x.to(torch.float32)

            # Add N,C dims and permute to N,C,D,H,W
            x = x.unsqueeze(0).unsqueeze(0)              # (1,1,T_valid,H0,W0)

            # First ensure spatial size to 112x112 if needed (can do in one go too)
            # We’ll do both temporal and spatial in ONE interpolate:
            # size=(D,H,W) -> (75,112,112)
            x = torch.nn.functional.interpolate(
                x,
                size=(target_T, target_H, target_W),
                mode="trilinear",
                align_corners=False,
            )  # (1,1,75,112,112)

            # Back to numpy uint8 in the desired shape
            x = x.squeeze(0).squeeze(0)                 # (75,112,112)
            x = x.clamp(0, 255).round().to(torch.uint8).cpu().numpy()

            if keep_channel_dim:
                out_np = x[..., None]                   # (75,112,112,1)
                out_shape = (target_T, target_H, target_W, 1)
            else:
                out_np = x                              # (75,112,112)
                out_shape = (target_T, target_H, target_W)

            # Write to a temporary dataset under the same group
            tmp_name = f"{video_key}/_frames_tmp"
            if tmp_name in h5f:
                del h5f[tmp_name]
            dst = h5f.create_dataset(tmp_name, shape=out_shape, dtype=np.uint8)

            # Copy attributes from original dataset (e.g., video_path)
            for k, v in src.attrs.items():
                dst.attrs[k] = v

            # Write data (contiguous, no compression)
            dst[...] = out_np
            print(out_np.shape)

            # Atomic swap
            grp.move("frames", "_frames_old")                       # old -> backup
            h5f.move(tmp_name, f"{video_key}/frames")               # tmp -> frames
            del grp["_frames_old"]                                  # remove backup

def generate_random_mask(spec_len, spec_dim, loss_rate):

    model = GilbertElliottModel(loss_rate=loss_rate)
    return model.simulate(spec_len, spec_dim)

def generate_random_uniform_mask(spec_len, spec_dim, loss_bounds=(0.3, 0.7)):

    model = GilbertElliottModel(loss_rate=np.random.uniform(*loss_bounds))
    return model.simulate(spec_len, spec_dim)

def fill_mask(spec_shape, valid_mask):
    mask = np.ones(spec_shape, dtype=np.float32)
    mask[:, :valid_mask.shape[1]] = valid_mask
    return mask

def process_single_chunk_masks(chunk_file, dataset_base=DATASET_BASE):
    print(f"Processing: {chunk_file}", flush=True)
    try:
        h5f = h5py.File(chunk_file, 'r+')
    except Exception as e:
        print(f"  FATAL: cannot open {chunk_file}: {e}", flush=True)
        return

    try:
        video_keys = list(h5f.keys())
        for video_key in tqdm(video_keys, desc=os.path.basename(chunk_file),
                              leave=False, mininterval=5.0):
            video_path = h5f.attrs.get(f"{video_key}/video_path", None)
            if video_path is None:
                print(f"  WARNING: no video_path for {video_key}", flush=True)
                continue
            try:
                rel_path = video_path.split("datasets", 1)[-1].lstrip(os.sep)
                path = os.path.join(dataset_base, rel_path)
                audio, valid_len = read_audio_ffmpeg(path, target_sr=SR,
                                                     max_sec=TARGET_DURATION_S)
                spec = extract_spectral(audio)
                spec_f, spec_t = spec.shape          # (80, T)
                valid_t = min(spec_t, int(round(valid_len / 160)))  # Bug #7 fix: / not //

                for pct in ['', '_skw', '_10', '_20', '_30', '_40', '_50', '_60', '_70']:
                    key = f"{video_key}/mask{pct}"
                    if key in h5f:
                        del h5f[key]
                if f"{video_key}/audio_len" in h5f.attrs:
                    del h5f.attrs[f"{video_key}/audio_len"]

                # Bug #7 fix: correct arg order — (spec_f, valid_t) not (valid_t, spec_f)
                mask_valid = generate_random_uniform_mask(spec_f, valid_t,
                                                          loss_bounds=(0.3, 0.7))
                mask = fill_mask(spec.shape, mask_valid)
                h5f.create_dataset(f"{video_key}/mask", data=mask, compression="gzip")

                if 'test' in chunk_file:
                    for pct in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
                        # Bug #7 fix: correct arg order
                        mask_valid = generate_random_mask(spec_f, valid_t,
                                                          loss_rate=pct / 100)
                        mask = fill_mask(spec.shape, mask_valid)
                        h5f.create_dataset(f"{video_key}/mask_{pct}",
                                           data=mask, compression="gzip")

                h5f.attrs[f"{video_key}/audio_len"] = valid_len

            except Exception as e:
                print(f"  ERROR for {video_key}: {e}", flush=True)
    finally:
        h5f.flush()
        h5f.close()

    print(f"Completed: {chunk_file}", flush=True)


def update_h5_mel_spec(base_path, chunk_pattern="_chunk*.h5"):
    chunk_files = sorted(glob.glob(f"{base_path}{chunk_pattern}"))

    with multiprocessing.Pool(processes=min(16, len(chunk_files))) as pool:
        pool.map(process_single_chunk_mel_spec, chunk_files)

def update_h5_mask(base_path, chunk_pattern="_chunk*.h5"):
    chunk_files = sorted(glob.glob(f"{base_path}{chunk_pattern}"))
    for chunk_file in chunk_files:
        process_single_chunk_masks(chunk_file)

def update_spkr_embd_h5(base_path, chunk_pattern="_chunk*.h5"):
    chunk_files = sorted(glob.glob(f"{base_path}{chunk_pattern}"))
    for chunk_file in chunk_files:
        process_single_chunk_spkr_embd(chunk_file)


def update_h5_frames_sizes(base_path, chunk_pattern="_chunk*.h5", processes=16):
    chunk_files = sorted(glob.glob(f"{base_path}{chunk_pattern}"))
    if not chunk_files:
        print("No chunk files found.")
        return
    with multiprocessing.Pool(processes=min(processes, len(chunk_files))) as pool:
        pool.map(process_single_chunk_frames_sizes, chunk_files)

def update_h5_frames(base_path, chunk_pattern="_chunk*.h5", processes=16):
    chunk_files = sorted(glob.glob(f"{base_path}{chunk_pattern}"))
    if not chunk_files:
        print("No chunk files found.")
        return
    with multiprocessing.Pool(processes=min(processes, len(chunk_files))) as pool:
        pool.map(process_single_chunk_frames, chunk_files)

def process_single_chunk_text(chunk_file: str, dataset_base: str = DATASET_BASE,
                                    max_sec: float = 3.0) -> None:
    """
    Re-encode 'text' for every LRS2 sample using time-bounded transcripts.
    Always overwrites. Safe to re-run.
    """
    from shared.text_processing import TextTokenizer
    tokenizer = TextTokenizer()

    print(f"Processing: {chunk_file}", flush=True)
    skipped, updated, errors = 0, 0, 0

    try:
        h5f = h5py.File(chunk_file, "r+")
    except Exception as e:
        print(f"  FATAL: cannot open {chunk_file}: {e}", flush=True)
        return

    try:
        video_keys = list(h5f.keys())
        for video_key in tqdm(video_keys, desc=os.path.basename(chunk_file),
                              leave=False, mininterval=5.0):

            video_path = h5f.attrs.get(f"{video_key}/video_path", None)
            if video_path is None:
                print(f"  WARNING: no video_path for {video_key}", flush=True)
                skipped += 1
                continue

            rel_path = video_path.split("datasets", 1)[-1].lstrip(os.sep)
            txt_path = os.path.splitext(os.path.join(dataset_base, rel_path))[0] + ".txt"
            if not os.path.isfile(txt_path):
                print(f"  WARNING: .txt not found: {txt_path}", flush=True)
                skipped += 1
                continue

            audio_len = h5f.attrs.get(f"{video_key}/audio_len", None)
            valid_sec = float(audio_len) / SR if audio_len is not None else None

            try:
                chunk_duration = min(max_sec, valid_sec) if valid_sec is not None else max_sec
                transcript, _ = tokenizer.read_transcripts_for_chunk(
                    txt_path, start_sec=0.0, max_sec=chunk_duration
                )
                encoded, _ = tokenizer.encode(transcript, max_length=128)

                ds_key = f"{video_key}/text"
                if ds_key in h5f:
                    del h5f[ds_key]
                h5f.create_dataset(ds_key, data=encoded, dtype=np.int32,
                                   compression="gzip")
                updated += 1

            except Exception as e:
                print(f"  ERROR at {video_key}: {e}", flush=True)
                errors += 1
    finally:
        h5f.flush()
        h5f.close()

    print(f"  Done — updated={updated}, skipped={skipped}, errors={errors}", flush=True)

def _worker_text(args):
    """Top-level picklable wrapper for ProcessPoolExecutor."""
    chunk_file, dataset_base, max_sec = args
    process_single_chunk_text(chunk_file, dataset_base=dataset_base, max_sec=max_sec)
    return chunk_file

def update_h5_text(base_path: str, dataset_base: str = DATASET_BASE,
                         chunk_pattern: str = "_chunk*.h5",
                         max_sec: float = 3.0,
                         processes: int = 1,
                         chunk_timeout: int = 300,
                         checkpoint_file: str = None) -> None:
    """
    Update 'text' datasets across all LRS2 HDF5 chunks.
    - checkpoint_file: completed chunks are recorded here; re-runs skip them.
    - chunk_timeout:   seconds before a stuck chunk is abandoned (parallel only).
    """
    chunk_files = sorted(glob.glob(f"{base_path}{chunk_pattern}"))
    if not chunk_files:
        print(f"No HDF5 chunk files found at: {base_path}{chunk_pattern}")
        return

    # ── checkpoint ────────────────────────────────────────────────────────
    if checkpoint_file is None:
        checkpoint_file = base_path + "_text_update_ckpt.json"

    completed = set()
    if os.path.isfile(checkpoint_file):
        try:
            with open(checkpoint_file) as f:
                completed = set(json.load(f))
            print(f"Checkpoint: {len(completed)} chunk(s) already done.")
        except Exception as e:
            print(f"WARNING: could not read checkpoint ({e}), starting fresh.")

    pending = [cf for cf in chunk_files if cf not in completed]
    print(f"{len(chunk_files)} chunk(s) total — {len(completed)} done, "
          f"{len(pending)} pending.")
    if not pending:
        print("Nothing to do.")
        return

    def _save_ckpt():
        try:
            with open(checkpoint_file, "w") as f:
                json.dump(sorted(completed), f, indent=2)
        except Exception as e:
            print(f"WARNING: checkpoint save failed: {e}", flush=True)

    args_list = [(cf, dataset_base, max_sec) for cf in pending]

    if processes <= 1:
        for args in args_list:
            try:
                process_single_chunk_text(args[0], dataset_base=args[1],
                                               max_sec=args[2])
                completed.add(args[0])
                _save_ckpt()
            except Exception as e:
                print(f"ERROR on {args[0]}: {e}", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=processes) as ex:
            future_to_chunk = {ex.submit(_worker_text, a): a[0] for a in args_list}
            for future in concurrent.futures.as_completed(future_to_chunk):
                cf = future_to_chunk[future]
                try:
                    future.result(timeout=chunk_timeout)
                    completed.add(cf)
                    _save_ckpt()
                    print(f"  ✓ {os.path.basename(cf)}", flush=True)
                except concurrent.futures.TimeoutError:
                    print(f"  TIMEOUT ({chunk_timeout}s): {cf} — skipping.", flush=True)
                    future.cancel()
                except Exception as e:
                    print(f"  ERROR: {cf}: {e}", flush=True)

    print(f"\nFinished. {len(completed)}/{len(chunk_files)} chunk(s) completed.")
    print(f"Checkpoint: {checkpoint_file}")

def _get_h5_attr_any(h5f, keys, default=None):
    for k in keys:
        if k in h5f.attrs:
            return h5f.attrs[k]
    return default


def _get_video_attr_any(h5f, video_key, names, default=None):
    """
    Your code stores attrs like:
        h5f.attrs[f"{video_key}/video_path"]

    This helper checks several possible names.
    """
    for name in names:
        k = f"{video_key}/{name}"
        if k in h5f.attrs:
            return h5f.attrs[k]
    return default


def _to_str(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def process_single_chunk_phone_indices(
    chunk_file: str,
    dataset_base: str = DATASET_BASE,
    max_sec: float = 3.0,
    max_phone_length: int = 128,
    require_full_word_inside: bool = True,
) -> None:

    from shared.text_processing import PhonemeTokenizer, TextTokenizer
    phoneme_encoder = PhonemeTokenizer()
    tokenizer = TextTokenizer()

    print(f"Processing: {chunk_file}", flush=True)
    skipped, updated, errors = 0, 0, 0

    try:
        h5f = h5py.File(chunk_file, "r+")
    except Exception as e:
        print(f"  FATAL: cannot open {chunk_file}: {e}", flush=True)
        return

    try:
        video_keys = list(h5f.keys())

        for video_key in tqdm(
            video_keys,
            desc=os.path.basename(chunk_file),
            leave=False,
            mininterval=5.0,
        ):
            try:
                video_path = _get_video_attr_any(
                    h5f,
                    video_key,
                    names=["video_path"],
                    default=None,
                )

                if video_path is None:
                    print(f"  WARNING: no video_path for {video_key}", flush=True)
                    skipped += 1
                    continue

                video_path = _to_str(video_path)

                rel_path = video_path.split("datasets", 1)[-1].lstrip(os.sep)
                txt_path = os.path.splitext(os.path.join(dataset_base, rel_path))[0] + ".txt"

                if not os.path.isfile(txt_path):
                    print(f"  WARNING: .txt not found: {txt_path}", flush=True)
                    skipped += 1
                    continue

                chunk_start = _get_video_attr_any(
                    h5f,
                    video_key,
                    names=[
                        "chunk_start",
                        "start_sec",
                        "start_time",
                        "t_start",
                        "offset_sec",
                    ],
                    default=0.0,
                )
                chunk_start = float(chunk_start)

                audio_len = _get_video_attr_any(
                    h5f,
                    video_key,
                    names=["audio_len"],
                    default=None,
                )

                if audio_len is not None:
                    valid_sec = float(audio_len) / SR
                    chunk_duration = min(max_sec, valid_sec)
                else:
                    chunk_duration = max_sec

                transcript, timed_words = tokenizer.read_transcripts_for_chunk(
                    txt_path=txt_path,
                    start_sec=chunk_start,
                    max_sec=chunk_duration,
                    require_full_word_inside=require_full_word_inside,
                )

                if not transcript:
                    print(
                        f"  WARNING: no timed words inside "
                        f"[{chunk_start:.2f}, {chunk_start + chunk_duration:.2f}] "
                        f"for {video_key}",
                        flush=True,
                    )
                    skipped += 1
                    continue

                phone_indices, _, _ = phoneme_encoder.encode(
                    transcript,
                    max_length=max_phone_length,
                )

                phone_indices = np.asarray(phone_indices, dtype=np.int32)

                # --------------------------------------------------
                # Keep only phone_indices.
                # Delete old phone-related datasets if present.
                # --------------------------------------------------
                old_datasets = [
                    f"{video_key}/phone_indices",
                    f"{video_key}/phone_length",
                    f"{video_key}/phones",
                    f"{video_key}/phonemes",
                    f"{video_key}/phone_mask",
                ]

                for ds in old_datasets:
                    if ds in h5f:
                        del h5f[ds]

                # --------------------------------------------------
                # Delete old phone-related attributes if present.
                # --------------------------------------------------
                old_attrs = [
                    f"{video_key}/phone_transcript",
                    f"{video_key}/phones_arpabet",
                    f"{video_key}/phone_chunk_start",
                    f"{video_key}/phone_chunk_end",
                    f"{video_key}/num_timed_words",
                    f"{video_key}/phone_length",
                ]

                for attr in old_attrs:
                    if attr in h5f.attrs:
                        del h5f.attrs[attr]

                # --------------------------------------------------
                # Save only phone_indices.
                # --------------------------------------------------
                h5f.create_dataset(
                    f"{video_key}/phone_indices",
                    data=phone_indices,
                    dtype=np.int32,
                    compression="gzip",
                )

                updated += 1

            except Exception as e:
                print(f"  ERROR at {video_key}: {e}", flush=True)
                errors += 1

    finally:
        h5f.flush()
        h5f.close()

    print(f"  Done — updated={updated}, skipped={skipped}, errors={errors}", flush=True)


def update_h5_phone_indices(
    base_path: str,
    dataset_base: str = DATASET_BASE,
    max_sec: float = 3.0,
    chunk_pattern: str = "_chunk*.h5",
    processes: int = 1,
    max_phone_length: int = 128,
) -> None:
    """
    Regenerate CTC phone_indices from WhisperX timed transcripts.

    This version uses:
        WhisperX timed words -> 3-sec chunk transcript -> g2p-en -> ARPAbet phones
    """
    chunk_files = sorted(glob.glob(f"{base_path}{chunk_pattern}"))

    if not chunk_files:
        print(f"No HDF5 chunk files found at: {base_path}{chunk_pattern}")
        return

    if processes > 1:
        print(
            "WARNING: processes > 1 is not used here. "
            "Keeping sequential processing for safer HDF5 writes.",
            flush=True,
        )
        processes = 1

    print(f"Found {len(chunk_files)} chunk file(s).")

    for chunk_file in chunk_files:
        process_single_chunk_phone_indices(
            chunk_file=chunk_file,
            dataset_base=dataset_base,
            max_sec=max_sec,
            max_phone_length=max_phone_length,
        )

def audit_oov_phones(base_path: str, dataset_base: str = DATASET_BASE,
                     chunk_pattern: str = "_chunk*.h5",
                     max_sec: float = 3.0,
                     sample_limit: int = 2000) -> None:
    """
    Scan chunk files, re-phonemize transcripts, and report every phone token
    that is NOT in PhonemeTokenizer._EN_PHONES. Prints a frequency-sorted table.
    Run this BEFORE re-encoding to know exactly what to add to _EN_PHONES.
    """
    from collections import Counter
    from text_processing import TextTokenizer, PhonemeTokenizer
    from phonemizer.backend import EspeakBackend
    from phonemizer.separator import Separator

    tokenizer = TextTokenizer()
    known = set(PhonemeTokenizer._ARPABET_PHONES)
    sep = Separator(phone=' ', word=' | ', syllable='')

    # Persistent espeak backend — same settings as PhonemeTokenizer
    backend = EspeakBackend(language='en-us', with_stress=False,
                            language_switch='remove-flags')

    oov_counter = Counter()   # oov_phone -> count
    oov_contexts = {}         # oov_phone -> example word context

    chunk_files = sorted(glob.glob(f"{base_path}{chunk_pattern}"))
    total_samples = 0

    for chunk_file in chunk_files:
        if total_samples >= sample_limit:
            break
        try:
            h5f = h5py.File(chunk_file, 'r')
        except Exception as e:
            print(f"Cannot open {chunk_file}: {e}")
            continue

        try:
            for video_key in list(h5f.keys()):
                if total_samples >= sample_limit:
                    break

                video_path = h5f.attrs.get(f"{video_key}/video_path", None)
                if video_path is None:
                    continue

                rel_path = video_path.split("datasets", 1)[-1].lstrip(os.sep)
                txt_path = os.path.splitext(os.path.join(dataset_base, rel_path))[0] + ".txt"
                if not os.path.isfile(txt_path):
                    continue

                audio_len = h5f.attrs.get(f"{video_key}/audio_len", None)
                valid_sec = float(audio_len) / SR if audio_len is not None else None

                transcript, _ = tokenizer.read_transcript_lrs2(
                    txt_path, valid_sec=valid_sec, max_sec=max_sec
                )
                if not transcript:
                    continue

                # Normalize same way PhonemeTokenizer does
                import re, unicodedata
                text = transcript.lower()
                text = re.sub(r"[^\w\s']", "", text)
                text = re.sub(r"\s+", " ", text).strip()

                raw = backend.phonemize([text], separator=sep, strip=True)
                raw = raw[0] if raw else ''
                tokens = [t for t in raw.split() if t]

                for tok in tokens:
                    if tok != '|' and tok not in known:
                        oov_counter[tok] += 1
                        if tok not in oov_contexts:
                            oov_contexts[tok] = text[:60]  # store first example

                total_samples += 1
        finally:
            h5f.close()

    print(f"\n{'='*60}")
    print(f"OOV phone audit — {total_samples} samples scanned")
    print(f"{'='*60}")
    if not oov_counter:
        print("No OOV phones found. _EN_PHONES is complete.")
        return

    print(f"{'Phone':<15} {'Count':>8}   Example transcript")
    print(f"{'-'*60}")
    for phone, count in oov_counter.most_common():
        ctx = oov_contexts.get(phone, '')
        print(f"{phone!r:<15} {count:>8}   {ctx}")

    print(f"\nAdd these to PhonemeTokenizer._EN_PHONES:")
    print(sorted(oov_counter.keys()))

# 
# -----------------------------------------------------------------------------
# AV_PLC additive migration
# -----------------------------------------------------------------------------
# Protected existing datasets are NEVER rewritten by this migration:
#   spec, spkr_embd, visual_features, landmarks, text, phone_indices, mask*
# Only these AV_PLC fields are added/updated:
#   audio, full_landmarks, frames (legacy -> aligned 96x96), video_len
# ``audio_len`` is added only when absent; if present it is verified and retained.


def _resolve_source_path(video_path, dataset_base: str) -> str:
    if isinstance(video_path, bytes):
        video_path = video_path.decode("utf-8")
    video_path = str(video_path)
    if os.path.isfile(video_path):
        return video_path
    if "datasets" in video_path:
        rel = video_path.split("datasets", 1)[-1].lstrip(os.sep)
        candidate = os.path.join(dataset_base, rel)
        if os.path.isfile(candidate):
            return candidate
    candidate = os.path.join(dataset_base, video_path.lstrip(os.sep))
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(f"Cannot resolve source media: {video_path}")


def _encode_audio_for_h5(audio: torch.Tensor, storage_dtype: str) -> np.ndarray:
    x = audio.detach().cpu().numpy().astype(np.float32, copy=False)
    if storage_dtype == "float32":
        return x
    if storage_dtype == "int16":
        return np.clip(np.rint(x * 32768.0), -32768, 32767).astype(np.int16)
    raise ValueError("audio_dtype must be 'float32' or 'int16'")


def _extract_avplc_visual_updates(video_path: str, reference_face: str,
                                  target_fps: float = 25.0,
                                  max_sec: float = TARGET_DURATION_S):
    """Return aligned AV_PLC frames, full MediaPipe landmarks and video_len.

    This function intentionally does not compute legacy lip landmarks or
    AV-HuBERT visual features. Those existing HDF5 datasets are preserved.
    """
    import cv2
    import mediapipe
    from AV_PLC.video_preprocessing import ReferenceFaceAligner

    aligner = ReferenceFaceAligner.from_npy(reference_face)
    face_mesh = mediapipe.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    target_T = int(round(target_fps * max_sec))
    landmark_dim = 478  # refine_landmarks=True MediaPipe FaceMesh output capacity
    raw_frames = []
    full = np.zeros((target_T, landmark_dim, 2), dtype=np.float32)
    detected = np.zeros(target_T, dtype=bool)

    cap = cv2.VideoCapture(video_path)
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if not src_fps or src_fps <= 1e-3:
            src_fps = target_fps
        ratio = max(1, int(round(src_fps / target_fps)))
        i = 0
        t = 0
        while t < target_T:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if (i % ratio) != 0:
                i += 1
                continue
            i += 1

            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            raw_frames.append(frame)
            h, w = frame.shape[:2]
            result = face_mesh.process(frame)
            if result.multi_face_landmarks:
                pts = np.asarray(
                    [[p.x * w, p.y * h] for p in result.multi_face_landmarks[0].landmark],
                    dtype=np.float32,
                )
                n = min(landmark_dim, pts.shape[0])
                full[t, :n] = pts[:n]
                detected[t] = True
            t += 1
    finally:
        cap.release()
        face_mesh.close()

    video_len = len(raw_frames)
    if video_len == 0:
        raise RuntimeError("No video frames decoded")
    if int(detected[:video_len].sum()) < 2:
        raise RuntimeError("Fewer than two valid MediaPipe face detections")

    rgb = np.stack(raw_frames, axis=0)
    aligned_valid = aligner.align_sequence(
        rgb, full[:video_len], detected[:video_len]
    )
    frames = np.zeros((target_T, 96, 96, 1), dtype=np.uint8)
    frames[:video_len] = aligned_valid
    return frames, full, int(video_len)


def _replace_dataset(h5f: h5py.File, video_key: str, name: str,
                     data: np.ndarray, compression="gzip") -> None:
    """Replace one explicitly allowed AV_PLC dataset with crash-recoverable swap."""
    grp = h5f[video_key]
    tmp = f"__avplc_tmp_{name}"
    backup = f"__avplc_old_{name}"
    if tmp in grp:
        del grp[tmp]
    if backup in grp:
        # A previous interrupted migration left a backup. Restore only if the
        # canonical dataset is missing; otherwise the backup is stale.
        if name not in grp:
            grp.move(backup, name)
        else:
            del grp[backup]

    kwargs = {"data": data}
    if compression is not None:
        kwargs["compression"] = compression
    grp.create_dataset(tmp, **kwargs)

    if name in grp:
        grp.move(name, backup)
    grp.move(tmp, name)
    if backup in grp:
        del grp[backup]


def migrate_single_chunk_avplc(
    chunk_file: str,
    reference_face: str,
    dataset_base: str = DATASET_BASE,
    audio_dtype: str = "float32",
    overwrite_new_fields: bool = False,
) -> dict:
    """Add only the data needed by the new AV_PLC pipeline.

    Existing shared/reference features are intentionally untouched:
    ``spec``, ``spkr_embd``, ``visual_features``, ``landmarks``, ``text``,
    ``phone_indices`` and any stored ``mask*`` datasets.

    ``frames`` is the sole pre-existing dataset that may be replaced, because
    AV_PLC now requires reference-face-aligned 96x96 pixel input. If it already
    has 96x96 spatial geometry it is left untouched unless
    ``overwrite_new_fields=True``.
    """
    reference_face = os.path.abspath(reference_face)
    if not os.path.isfile(reference_face):
        raise FileNotFoundError(f"Reference face not found: {reference_face}")

    summary = {"updated": 0, "skipped": 0, "errors": 0}
    print(f"Migrating: {chunk_file}", flush=True)

    with h5py.File(chunk_file, "r+") as h5f:
        for video_key in tqdm(list(h5f.keys()), desc=os.path.basename(chunk_file), leave=False):
            try:
                video_path = h5f.attrs.get(f"{video_key}/video_path", None)
                if video_path is None:
                    raise KeyError("missing video_path attribute")
                source_path = _resolve_source_path(video_path, dataset_base)

                grp = h5f[video_key]
                need_audio = overwrite_new_fields or "audio" not in grp
                need_full = overwrite_new_fields or "full_landmarks" not in grp
                frames_already_aligned = (
                    "frames" in grp
                    and grp["frames"].ndim in (3, 4)
                    and tuple(grp["frames"].shape[1:3]) == (96, 96)
                )
                need_frames = overwrite_new_fields or not frames_already_aligned
                need_video_len = overwrite_new_fields or f"{video_key}/video_len" not in h5f.attrs

                # Decode audio once. Existing audio_len is verification-only.
                audio, decoded_audio_len = read_audio_ffmpeg(
                    source_path, target_sr=SR, max_sec=TARGET_DURATION_S
                )
                if audio is None:
                    raise RuntimeError("no usable audio")

                audio_len_key = f"{video_key}/audio_len"
                existing_audio_len = h5f.attrs.get(audio_len_key, None)
                if existing_audio_len is None:
                    h5f.attrs[audio_len_key] = int(decoded_audio_len)
                elif int(existing_audio_len) != int(decoded_audio_len):
                    raise ValueError(
                        f"audio_len mismatch for {video_key}: HDF5={int(existing_audio_len)} "
                        f"decoded={int(decoded_audio_len)}. Existing value was NOT changed."
                    )

                if need_audio:
                    _replace_dataset(
                        h5f, video_key, "audio",
                        _encode_audio_for_h5(audio, audio_dtype),
                    )

                if need_frames or need_full or need_video_len:
                    aligned_frames, full_landmarks, video_len = _extract_avplc_visual_updates(
                        source_path, reference_face
                    )
                    if need_frames:
                        _replace_dataset(h5f, video_key, "frames", aligned_frames)
                    if need_full:
                        _replace_dataset(h5f, video_key, "full_landmarks", full_landmarks)
                    if need_video_len:
                        h5f.attrs[f"{video_key}/video_len"] = int(video_len)

                summary["updated"] += 1
            except Exception as exc:
                summary["errors"] += 1
                print(f"  ERROR {video_key}: {exc}", flush=True)

        h5f.flush()

    print(
        f"Completed {os.path.basename(chunk_file)}: "
        f"updated={summary['updated']} errors={summary['errors']}",
        flush=True,
    )
    return summary


def migrate_h5_avplc(
    base_path: str,
    reference_face: str,
    dataset_base: str = DATASET_BASE,
    chunk_pattern: str = "_chunk*.h5",
    audio_dtype: str = "float32",
    overwrite_new_fields: bool = False,
) -> None:
    """Migrate every matching HDF5 chunk sequentially for safe in-place writes."""
    chunk_files = sorted(glob.glob(f"{base_path}{chunk_pattern}"))
    if not chunk_files and os.path.isfile(base_path):
        chunk_files = [base_path]
    if not chunk_files:
        raise FileNotFoundError(f"No HDF5 files matched {base_path}{chunk_pattern}")

    total_errors = 0
    for chunk_file in chunk_files:
        result = migrate_single_chunk_avplc(
            chunk_file=chunk_file,
            reference_face=reference_face,
            dataset_base=dataset_base,
            audio_dtype=audio_dtype,
            overwrite_new_fields=overwrite_new_fields,
        )
        total_errors += result["errors"]
    if total_errors:
        raise RuntimeError(f"Migration completed with {total_errors} sample error(s)")


def _build_cli():
    parser = argparse.ArgumentParser(
        description=(
            "Add only the new AV_PLC waveform/aligned-video fields to existing "
            "HDF5 files. Existing spec, speaker embedding, AV-HuBERT features, "
            "legacy landmarks, text and phone targets are preserved."
        )
    )
    parser.add_argument(
        "--base-path", required=True,
        help="HDF5 file or chunk prefix, e.g. /data/grid/grid_train_features",
    )
    parser.add_argument("--reference-face", required=True, help="AV_PLC reference_face.npy")
    parser.add_argument("--dataset-base", default=DATASET_BASE)
    parser.add_argument("--chunk-pattern", default="_chunk*.h5")
    parser.add_argument("--audio-dtype", choices=("float32", "int16"), default="float32")
    parser.add_argument(
        "--overwrite-new-fields", action="store_true",
        help="Recompute only AV_PLC-owned fields (audio/full_landmarks/frames/video_len).",
    )
    return parser


if __name__ == "__main__":
    args = _build_cli().parse_args()
    migrate_h5_avplc(
        base_path=args.base_path,
        reference_face=args.reference_face,
        dataset_base=args.dataset_base,
        chunk_pattern=args.chunk_pattern,
        audio_dtype=args.audio_dtype,
        overwrite_new_fields=args.overwrite_new_fields,
    )
# python dataset/features/update_features.py \
#     --base-path "../datasets/grid/grid_train_features" \
#     --reference-face AV_PLC/reference_face_grid.npy \
#     --audio-dtype float32