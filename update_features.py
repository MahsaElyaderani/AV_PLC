import os
import h5py
import glob
import torch
import librosa
import logging
import numpy as np

from tqdm import tqdm
import multiprocessing
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav

from torchvision.io import read_video
import torchaudio

from masking import GilbertElliottModel

SR = 16000
TARGET_DURATION_S = 3.0


def melspectrogram(audio):

    melspctrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=512, #640,
        win_length=400, #640,
        hop_length=160,
        center=False,
        power=1.0,
        norm="slaney",
        onesided=True,
        n_mels=80,
        mel_scale="slaney",
    )
    amp2db_transform = torchaudio.transforms.AmplitudeToDB(stype='magnitude', top_db=80)

    padding = (512 - 160) // 2 #(640 - 160) // 2
    wav = torch.nn.functional.pad(audio, (padding, padding), "constant")
    mel_mag = melspctrogram(wav)
    mel_mag = mel_mag.clamp(min=1e-5)  # avoid log(0)
    logmel = amp2db_transform(mel_mag)
    #logmel_norm = (logmel + 80) / 80  # maps [-80, 0] -> [0, 1]
    #logmel_norm = torch.clamp(logmel_norm, 0.0, 1.0)

    return logmel #logmel_norm

def curr_read_video(filename):
    video, audio, info = read_video(filename)
    video_fps, audio_sr = info['video_fps'], info['audio_fps']

    if 'grid' in filename:
        audio = audio.numpy()
        audio = audio.astype(np.float32) / (np.max(audio))
        if audio.shape[0] > 1:
            audio = np.mean(audio, axis=0, keepdims=True)
        if info['audio_fps'] != SR:
            audio = librosa.resample(audio, orig_sr=info['audio_fps'], target_sr=SR)
    return video, torch.FloatTensor(audio), info

def extract_spectral(audio):

    spec = melspectrogram(audio)
    return spec.squeeze()

def process_single_chunk_speech_units(chunk_file):
    # Load checkpoint (either hubert_soft or hubert_discrete)
    hubert_ = torch.hub.load("bshall/hubert:main",
                                     "hubert_soft", trust_repo=True).cuda()
    logging.info(f"hubert_ model loaded successfully in process {os.getpid()}")

    print(f"Processing: {chunk_file}")
    with h5py.File(chunk_file, 'r+') as h5f:
        video_keys = list(h5f.keys())
        for video_key in tqdm(video_keys, desc=f"{os.path.basename(chunk_file)}"):
            video_path = h5f.attrs.get(f"{video_key}/video_path", None)
            if video_path is not None:
                video, audio, info = curr_read_video(video_path)

                audio_duration = int(TARGET_DURATION_S * SR)
                audio_len = audio.size(1)
                if audio_len > audio_duration:
                    audio = audio[:, :audio_duration]
                elif audio_len < audio_duration:
                    audio = torch.nn.functional.pad(audio, (0, audio_duration - audio.size(1)), 'constant')

                soft_units = hubert_.units(audio.unsqueeze(0).cuda())
                print(soft_units.squeeze(0).shape)
                if f"{video_key}/soft_units" in h5f:
                    del h5f[f"{video_key}/soft_units"]
                h5f.create_dataset(f"{video_key}/soft_units",
                                   data=soft_units.squeeze(0).cpu().numpy(), compression="gzip")
            else:
                print(f"Warning: No video path found for {video_key}")
    print(f"Completed: {chunk_file}")


def process_single_chunk_spkr_embd(chunk_file):
    voice_encoder = VoiceEncoder()
    print(f"Processing: {chunk_file}")
    with h5py.File(chunk_file, 'r+') as h5f:
        video_keys = list(h5f.keys())
        for video_key in tqdm(video_keys, desc=f"{os.path.basename(chunk_file)}"):
            video_path = h5f.attrs.get(f"{video_key}/video_path", None)
            if video_path is not None:
                try:
                    fpath = Path(video_path)
                    wav = preprocess_wav(fpath)
                    spkr_embd = voice_encoder.embed_utterance(wav)

                    np.set_printoptions(precision=3, suppress=True)
                    print(spkr_embd.shape)

                    if f"{video_key}/spkr_embd" in h5f:
                        del h5f[f"{video_key}/spkr_embd"]

                    h5f.create_dataset(f"{video_key}/spkr_embd",
                                       data=spkr_embd, compression="gzip")

                except ZeroDivisionError as e:
                    print(f"ZeroDivisionError for {video_key}: {e}")
                    if video_key in h5f:
                        del h5f[video_key]  # delete entire group

                except Exception as e:
                    print(f"General error for {video_key}: {e}")
            else:
                print(f"Warning: No video path found for {video_key}")
    print(f"Completed: {chunk_file}")


def process_single_chunk_mel_spec(chunk_file):

    print(f"Processing: {chunk_file}")
    with h5py.File(chunk_file, 'r+') as h5f:
        video_keys = list(h5f.keys())
        for video_key in tqdm(video_keys, desc=f"{os.path.basename(chunk_file)}"):
            video_path = h5f.attrs.get(f"{video_key}/video_path", None)
            if video_path is not None:
                video, audio, info = curr_read_video(video_path)
                audio_len = audio.size(1)
                audio_duration = int(TARGET_DURATION_S * SR)
                if audio_len > audio_duration:
                    audio = audio[:, :audio_duration]

                elif audio_len < audio_duration:
                    audio = torch.nn.functional.pad(audio,
                                                    (0, audio_duration - audio.size(1)),
                                                    'constant')

                mel_spec = extract_spectral(audio)
                #print(torch.max(mel_spec))
                #print(torch.min(mel_spec))
                print(mel_spec.shape)

                if f"{video_key}/mel_spec" in h5f:
                    del h5f[f"{video_key}/mel_spec"]
                h5f.create_dataset(f"{video_key}/spec",
                                   data=mel_spec.cpu().numpy(), compression="gzip")
            else:
                print(f"Warning: No video path found for {video_key}")
    print(f"Completed: {chunk_file}")

def generate_random_mask(spec_len, spec_dim, loss_rate):

    model = GilbertElliottModel(loss_rate=loss_rate)
    return model.simulate(spec_len, spec_dim)

def generate_random_uniform_mask(spec_len, spec_dim, loss_bounds=(0.3, 0.7)):

    model = GilbertElliottModel(loss_rate=np.random.uniform(*loss_bounds))
    return model.simulate(spec_len, spec_dim)


def process_single_chunk_masks(chunk_file):

    print(f"Processing: {chunk_file}")
    with h5py.File(chunk_file, 'r+') as h5f:
        video_keys = list(h5f.keys())
        for video_key in tqdm(video_keys, desc=f"{os.path.basename(chunk_file)}"):
            video_path = h5f.attrs.get(f"{video_key}/video_path", None)
            spec_len, spec_dim = 300, 80
            if video_path is not None:
                print(video_path)
                if 'train' in video_path or 'dev' in video_path or 'val' in video_path:
                    if f"{video_key}/mask" in h5f:
                         del h5f[f"{video_key}/mask"]
                    if f"{video_key}/mask_skw" in h5f:
                         del h5f[f"{video_key}/mask_skw"]
                    if f"{video_key}/mask_10" in h5f:
                        del h5f[f"{video_key}/mask_10"]
                    if f"{video_key}/mask_20" in h5f:
                        del h5f[f"{video_key}/mask_20"]
                    if f"{video_key}/mask_30" in h5f:
                        del h5f[f"{video_key}/mask_30"]
                    if f"{video_key}/mask_40" in h5f:
                        del h5f[f"{video_key}/mask_40"]
                    if f"{video_key}/mask_50" in h5f:
                        del h5f[f"{video_key}/mask_50"]
                    if f"{video_key}/mask_60" in h5f:
                        del h5f[f"{video_key}/mask_60"]
                    if f"{video_key}/mask_70" in h5f:
                        del h5f[f"{video_key}/mask_70"]

                    mask = generate_random_uniform_mask(spec_len=spec_len,
                                                        spec_dim=spec_dim,
                                                        loss_bounds=[0.3, 0.7])
                    #mask_skw = generate_random_skewed_mask(spec_len=spec_len, spec_dim=spec_dim)
                    h5f.create_dataset(f"{video_key}/mask", data=mask,
                                       compression="gzip")
                else:
                    if f"{video_key}/mask" in h5f:
                        del h5f[f"{video_key}/mask"]
                    if f"{video_key}/mask_10" in h5f:
                        del h5f[f"{video_key}/mask_10"]
                    if f"{video_key}/mask_20" in h5f:
                        del h5f[f"{video_key}/mask_20"]
                    if f"{video_key}/mask_30" in h5f:
                            del h5f[f"{video_key}/mask_30"]
                    if f"{video_key}/mask_40" in h5f:
                             del h5f[f"{video_key}/mask_40"]
                    if f"{video_key}/mask_50" in h5f:
                        del h5f[f"{video_key}/mask_50"]
                    if f"{video_key}/mask_60" in h5f:
                        del h5f[f"{video_key}/mask_60"]
                    if f"{video_key}/mask_70" in h5f:
                        del h5f[f"{video_key}/mask_70"]

                    mask = generate_random_uniform_mask(spec_len=spec_len,
                                                        spec_dim=spec_dim,
                                                        loss_bounds=[0.2, 0.6])
                    mask_20 = generate_random_mask(spec_len=spec_len,
                                                   spec_dim=spec_dim,
                                                   loss_rate=0.2)
                    mask_30 = generate_random_mask(spec_len=spec_len,
                                                   spec_dim=spec_dim,
                                                   loss_rate=0.3)
                    mask_40 = generate_random_mask(spec_len=spec_len,
                                                   spec_dim=spec_dim,
                                                   loss_rate=0.4)
                    mask_50 = generate_random_mask(spec_len=spec_len,
                                                   spec_dim=spec_dim,
                                                   loss_rate=0.5)
                    mask_60 = generate_random_mask(spec_len=spec_len,
                                                   spec_dim=spec_dim,
                                                   loss_rate=0.6)


                    h5f.create_dataset(f"{video_key}/mask", data=mask,
                                       compression="gzip")
                    h5f.create_dataset(f"{video_key}/mask_20", data=mask_20,
                                       compression="gzip")
                    h5f.create_dataset(f"{video_key}/mask_30", data=mask_30,
                                       compression="gzip")
                    h5f.create_dataset(f"{video_key}/mask_40", data=mask_40,
                                       compression="gzip")
                    h5f.create_dataset(f"{video_key}/mask_50", data=mask_50,
                                       compression="gzip")
                    h5f.create_dataset(f"{video_key}/mask_60", data=mask_60,
                                       compression="gzip")
            else:
                print(f"Warning: No video path found for {video_key}")
    print(f"Completed: {chunk_file}")

def process_single_chunk_grid_mel_spec(chunk_file):

    print(f"Processing: {chunk_file}")
    with h5py.File(chunk_file, 'r+') as h5f:
        video_keys = list(h5f.keys())
        for video_key in tqdm(video_keys, desc=f"{os.path.basename(chunk_file)}"):
            video_path = h5f.attrs.get(f"{video_key}/video_path", None)
            if video_path is not None:
                if f"{video_key}/mel_spec" in h5f:
                    del h5f[f"{video_key}/mel_spec"]
            else:
                print(f"Warning: No video path found for {video_key}")
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


# Fast integer RGB -> Gray (BT.601 luma). Works on whole array at once.
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

def update_h5_frames(base_path, chunk_pattern="_chunk*.h5", processes=16):
    chunk_files = sorted(glob.glob(f"{base_path}{chunk_pattern}"))
    if not chunk_files:
        print("No chunk files found.")
        return
    with multiprocessing.Pool(processes=min(processes, len(chunk_files))) as pool:
        pool.map(process_single_chunk_frames, chunk_files)

def update_h5_frames_sizes(base_path, chunk_pattern="_chunk*.h5", processes=16):
    chunk_files = sorted(glob.glob(f"{base_path}{chunk_pattern}"))
    if not chunk_files:
        print("No chunk files found.")
        return
    with multiprocessing.Pool(processes=min(processes, len(chunk_files))) as pool:
        pool.map(process_single_chunk_frames_sizes, chunk_files)

def update_h5_grid_mel_spec(base_path, chunk_pattern="_chunk*.h5"):
    chunk_files = sorted(glob.glob(f"{base_path}{chunk_pattern}"))

    with multiprocessing.Pool(processes=min(16, len(chunk_files))) as pool:
        pool.map(process_single_chunk_grid_mel_spec, chunk_files)

def update_h5_mel_spec(base_path, chunk_pattern="_chunk*.h5"):
    chunk_files = sorted(glob.glob(f"{base_path}{chunk_pattern}"))

    with multiprocessing.Pool(processes=min(16, len(chunk_files))) as pool:
        pool.map(process_single_chunk_mel_spec, chunk_files)

def update_h5_mask(base_path, chunk_pattern="_chunk*.h5"):
    chunk_files = sorted(glob.glob(f"{base_path}{chunk_pattern}"))

    with multiprocessing.Pool(processes=min(16, len(chunk_files))) as pool:
        pool.map(process_single_chunk_masks, chunk_files)

def update_spkr_embd_h5(base_path, chunk_pattern="_chunk*.h5"):
    chunk_files = sorted(glob.glob(f"{base_path}{chunk_pattern}"))
    for chunk_file in chunk_files:
        process_single_chunk_spkr_embd(chunk_file)


if __name__ == "__main__":

    splits = {"train"} # "train"/"dev", "val", "test"

    for split in splits:
        #path = f'/home/ai/Projects/Mahsa/datasets/vox2_short/vox2_{split}_mp4/'
        #video_list = glob.glob(os.path.join(path, '*/*/*.mp4'))

        #feats_filename = f'/home/ai/Projects/Mahsa/datasets/vox2_short/vox2_short_{split}_features.h5'
        #extract_av_features_parallel(video_list, feats_filename)

        #feats_path = f'/home/ai/Projects/Mahsa/datasets/vox2_short/vox2_short_{split}_features'
        #update_h5_mel_spec(feats_path)
        #update_h5_mask(feats_path)
        #update_spkr_embd_h5(feats_path)

        feats_path = f'datasets/grid_{split}_features'
        update_h5_frames_sizes(feats_path)