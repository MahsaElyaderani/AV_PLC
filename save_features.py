import os
import gc
import time
import cv2
from PIL import Image
import h5py
import glob
import random
import librosa
import logging
import numpy as np
from tqdm import tqdm
import multiprocessing
import tempfile
import dlib
from pathlib import Path
import traceback
import skvideo.io

import torch
import torchaudio
from torchvision.io import read_video
from transformers import pipeline
from matplotlib import pyplot as plt
from resemblyzer import VoiceEncoder, preprocess_wav

#from preparation.align_mouth import landmarks_interpolate, crop_patch
import mediapipe
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LIPS

from text_processing import  CTCTokenizer
from masking import GilbertElliottModel


LANDMARK_DIM = 478
SR = 16000
FPS = 25.0
DURATION_SEC = 3.0
lip_indices = sorted(set(i for connection in FACEMESH_LIPS for i in connection))

def init_worker():

    global mel_transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR,
        n_fft=640,
        win_length=640,
        hop_length=160,
        center=False,
        power=1.0,
        norm="slaney",
        onesided=True,
        n_mels=80,
        mel_scale="slaney",
    )
    global voice_encoder
    voice_encoder = VoiceEncoder()

    global face_mesh
    face_mesh = mediapipe.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    global ctc_tokenizer
    ctc_tokenizer = CTCTokenizer()

    global asr
    try:
        asr = pipeline(
            task="automatic-speech-recognition",
            model="distil-whisper/distil-small.en",
            device=0  # Use GPU if available (0), or -1 for CPU
        )
        logging.info("ASR model loaded successfully")
        return True
    except Exception as e:
        logging.error(f"Error loading ASR model: {e}")
        return False


def curr_read_video(filename):
    try:
        video, audio, info = read_video(filename)
    except Exception as e:
        print(f"[WARN] Failed to read video '{filename}': {e}")
        return None, None, None

    video_fps = info.get('video_fps', 0)
    audio_fps = info.get('audio_fps', 0)

    if video_fps != FPS or video is None or video.shape[0] == 0:
        print(f"[SKIP] Unsupported FPS or empty video: {filename}")
        return None, None, None

    # Normalize audio
    audio = audio.numpy().astype(np.float32)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    # Convert stereo audio to mono
    if audio.ndim > 1 and audio.shape[0] > 1:
        audio = np.mean(audio, axis=0, keepdims=True)

    # Resample audio if needed
    if audio_fps != SR:
        try:
            audio = librosa.resample(audio[0], orig_sr=audio_fps, target_sr=SR)
        except Exception as e:
            print(f"[WARN] Resampling failed: {e}")
            return None
    else:
        audio = audio[0]
    audio = torch.from_numpy(audio)
    # Truncate audio and video
    audio_len = int(DURATION_SEC * SR)
    #video_len = int(DURATION_SEC * FPS)

    if audio_len > audio.size(0):
        audio = torch.nn.functional.pad(audio, (0, audio_len - audio.size(0)), 'constant')

    elif audio_len < audio.size(0):
        audio = audio[:audio_len]


    audio = audio[: audio_len]
    #video = video[: video_len]

    return video, audio, info



def extract_spectral(audio):
    padding = (640 - 160) // 2
    audio = torch.nn.functional.pad(audio, (padding, padding), "constant")
    mel_spec = mel_transform(audio)
    logmel = torch.log(torch.clamp(mel_spec, min=1e-5))

    return logmel

def generate_random_mask(size, loss_rate):

    model = GilbertElliottModel(loss_rate=loss_rate)
    return model.simulate(*size)

def generate_uniform_mask(size, loss_bounds=(0.3, 0.7)):

    model = GilbertElliottModel(loss_rate=np.random.uniform(*loss_bounds))
    return model.simulate(*size)

def extract_roi_landmarks(video, face_mesh, frame_size = (96, 96)):

    video_np = video.numpy()
    h, w, _ = video_np[0].shape
    landmark_seq = []
    cropped_seq = []

    for frame in video_np:
        result = face_mesh.process(frame)
        landmarks = np.zeros((LANDMARK_DIM, 2), dtype=np.float32)

        if result.multi_face_landmarks:
            face = result.multi_face_landmarks[0]
            landmarks = np.array([[lm.x * w, lm.y * h] for lm in face.landmark], dtype=np.float32)
            lip_pts = landmarks[lip_indices]

            x1, y1 = np.min(lip_pts, axis=0)
            x2, y2 = np.max(lip_pts, axis=0)

            pad_x = (x2 - x1) * 0.5
            pad_y = (y2 - y1) * 0.5

            x1 = int(max(0, x1 - pad_x))
            y1 = int(max(0, y1 - pad_y))
            x2 = int(min(w, x2 + pad_x))
            y2 = int(min(h, y2 + pad_y))

            cropped = frame[y1:y2, x1:x2]
            cropped = cv2.resize(cropped, frame_size)
        else:
            cropped = np.zeros((*frame_size, 3), dtype=np.float32)

        landmark_seq.append(landmarks[lip_indices])
        cropped_seq.append(cropped)

    landmark_seq = np.stack(landmark_seq).astype(np.float32)
    cropped_seq = np.stack(cropped_seq).astype(np.float32)

    return cropped_seq, landmark_seq

def transcribe_speech(audio, filename):

    output_filename = os.path.splitext(filename)[0] + '.txt'
    if audio is None:
        logging.info("Audio not found, please retry.")
        return ""
    try:
        output = asr(audio.squeeze().numpy())
        with open(output_filename, 'w') as file:
            file.write(output["text"])
        return output["text"]
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        return ""

def extract_features(video_path):
    results = []
    try:
        global face_mesh

        video, audio, info = curr_read_video(video_path)

        if video is not None:
            roi_frames, landmarks = extract_roi_landmarks(video, face_mesh)
            mel_spec = extract_spectral(audio)
            transcription = transcribe_speech(audio, video_path)
            encoded_text = ctc_tokenizer.encode(transcription)
            spkr_embed = voice_encoder.embed_utterance(preprocess_wav(Path(video_path)))

            print(video_path)
            if 'train' in video_path or 'val' in video_path:
                mask = generate_uniform_mask(mel_spec.shape, loss_bounds=[0.3, 0.7])
                if landmarks is not None and roi_frames is not None:
                    results = {
                        'video_path': video_path,
                        'text': encoded_text,
                        'frames': roi_frames,
                        'landmarks': landmarks,
                        'mel_spec': mel_spec,
                        'spkr_embd': spkr_embed,
                        'mask': mask
                    }
            else:
                mask = generate_uniform_mask(mel_spec.shape, loss_bounds=[0.2, 0.6])
                mask_20 = generate_random_mask(mel_spec.shape, loss_rate=0.2)
                mask_30 = generate_random_mask(mel_spec.shape, loss_rate=0.3)
                mask_40 = generate_random_mask(mel_spec.shape, loss_rate=0.4)
                mask_50 = generate_random_mask(mel_spec.shape, loss_rate=0.5)
                mask_60 = generate_random_mask(mel_spec.shape, loss_rate=0.6)
                if landmarks is not None and roi_frames is not None:
                    results = {
                        'video_path': video_path,
                        'text': encoded_text,
                        'frames': roi_frames,
                        'landmarks': landmarks,
                        'mel_spec': mel_spec,
                        'spkr_embd': spkr_embed,
                        'mask': mask,
                        'mask_20': mask_20,
                        'mask_30': mask_30,
                        'mask_40': mask_40,
                        'mask_50': mask_50,
                        'mask_60': mask_60,
                    }
    except Exception as e:
        logging.error(f"Error processing {video_path}: {str(e)}")
        logging.error(traceback.format_exc())
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return results


def extract_features_parallel(video_list, output_file, num_workers=None,
                                 chunk_size=1000, mode='w'):
    # mode = 'w' : for separate saving of files
    # mode = 'a' for appending files and single file saving (could become very large)
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    base_name, ext = os.path.splitext(output_file)
    if not ext:
        ext = '.h5'

    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), 4)

    chunk_info = []
    total_processed = 0
    with multiprocessing.Pool(processes=num_workers, initializer=init_worker) as pool:

        results = []
        chunk_count = 0

        for result in tqdm(
                pool.imap_unordered(extract_features, video_list, chunksize=4),
                total=len(video_list),
                desc="Processing videos"):
            if result:
                results.append(result)
            if len(results) >= chunk_size:
                if mode == 'w':
                    current_output_file = f"{base_name}_chunk{chunk_count}{ext}"
                elif mode == 'a':
                    current_output_file = output_file

                write_h5(current_output_file, results, chunk_count * chunk_size, mode=mode)
                chunk_info.append({
                    'chunk_id': chunk_count,
                    'filename': current_output_file,
                    'num_videos': len(results),
                    'start_idx': total_processed,
                    'end_idx': total_processed + chunk_size - 1
                })
                total_processed += chunk_size
                results.clear()
                chunk_count += 1
                gc.collect()
                time.sleep(1)

    if results:
        if mode == 'w':
            current_output_file = f"{base_name}_chunk{chunk_count}{ext}"
        elif mode == 'a':
            current_output_file = output_file

        write_h5(current_output_file, results, chunk_count * chunk_size, mode=mode)
        chunk_info.append({
            'chunk_id': chunk_count,
            'filename': current_output_file,
            'num_videos': len(results),
            'start_idx': total_processed,
            'end_idx': total_processed + chunk_size - 1
        })
        total_processed += chunk_size
        results.clear()
        chunk_count += 1

    metadata_file = f"{base_name}_metadata.txt"
    with open(metadata_file, 'w') as f:
        f.write(f"Total videos processed: {total_processed}\n")
        f.write(f"Total chunks: {chunk_count + 1}\n")
        f.write("\nChunk details:\n")

        for chunk in chunk_info:
            f.write(f"Chunk {chunk['chunk_id']}: {chunk['filename']} - "
                    f"{chunk['num_videos']} videos (indices {chunk['start_idx']}-{chunk['end_idx']})\n")

    if mode == 'w':
        logging.info(f"Saved {total_processed} results across {chunk_count + 1} files")
        logging.info(f"Metadata saved to {metadata_file}")
    else:
        logging.info(f"Appended {total_processed} results to {output_file}")


def write_h5(output_file, results, start_index, mode):
    with h5py.File(output_file, mode, libver='latest') as h5f:
        for idx, result in enumerate(results):
            video_idx = start_index + idx
            video_key = f"video_{video_idx}"
            h5f.create_dataset(f"{video_key}/frames", data=result["frames"], compression="gzip")
            h5f.create_dataset(f"{video_key}/landmarks", data=result["landmarks"], compression="gzip")
            h5f.create_dataset(f"{video_key}/mel_spec", data=result["mel_spec"], compression="gzip")
            h5f.create_dataset(f"{video_key}/text", data=result["text"], compression="gzip")
            h5f.create_dataset(f"{video_key}/spkr_embd", data=result["spkr_embd"], compression="gzip")
            h5f.create_dataset(f"{video_key}/mask", data=result["mask"], compression="gzip")
            h5f.attrs[f"{video_key}/video_path"] = result["video_path"]

            if 'test' in result["video_path"]:
                h5f.create_dataset(f"{video_key}/mask_20", data=result["mask_20"], compression="gzip")
                h5f.create_dataset(f"{video_key}/mask_30", data=result["mask_30"], compression="gzip")
                h5f.create_dataset(f"{video_key}/mask_40", data=result["mask_40"], compression="gzip")
                h5f.create_dataset(f"{video_key}/mask_50", data=result["mask_50"], compression="gzip")
                h5f.create_dataset(f"{video_key}/mask_60", data=result["mask_60"], compression="gzip")

    logging.info(f"Wrote {len(results)} results to {output_file}")


def update_h5(chunk_file):
    # Load checkpoint to extract speech units (either hubert_soft or hubert_discrete)
    hubert_discrete = torch.hub.load("bshall/hubert:main",
                                     "hubert_discrete", trust_repo=True).cuda()
    logging.info(f"hubert_discrete model loaded successfully in process {os.getpid()}")

    print(f"Processing: {chunk_file}")
    with h5py.File(chunk_file, 'r+') as h5f:
        video_keys = list(h5f.keys())
        for video_key in tqdm(video_keys, desc=f"{os.path.basename(chunk_file)}"):
            video_path = h5f.attrs.get(f"{video_key}/video_path", None)
            if video_path is not None:

                video, audio, info = curr_read_video(video_path)
                units = hubert_discrete.units(audio.unsqueeze(0).cuda())
                print(units.shape)

                if f"{video_key}/units" in h5f:
                    del h5f[f"{video_key}/units"]
                h5f.create_dataset(f"{video_key}/units", data=units, compression="gzip")
            else:
                print(f"Warning: No video path found for {video_key}")
    print(f"Completed: {chunk_file}")


def update_h5_parallel(base_path, chunk_pattern="_chunk*.h5"):
    chunk_files = sorted(glob.glob(f"{base_path}{chunk_pattern}"))

    with multiprocessing.Pool(processes=min(16, len(chunk_files))) as pool:
        pool.map(update_h5, chunk_files)

if __name__ == "__main__":

    splits = {"test", "val", "train"}

    for split in splits:
        path = f'datasets/{split}/'
        video_list = glob.glob(os.path.join(path, 's*/*.mpg'))

        feats_filename = f'datasets/grid_{split}_features.h5'
        extract_features_parallel(video_list, feats_filename)


        #feats_path = f'/home/ai/Projects/Mahsa/datasets/vox2_short/vox2_short_{split}_features'
        #update_h5_parallel(feats_path)