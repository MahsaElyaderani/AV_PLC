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

from stable_diffusion.dataset.masking import GilbertElliottModel

SR = 16000
TARGET_DURATION_S = 3.0

voice_encoder = VoiceEncoder()

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

        feats_path = f'/home/ai/Projects/Mahsa/datasets/grid/grid_{split}_features'
        update_h5_grid_mel_spec(feats_path)