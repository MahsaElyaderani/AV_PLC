import os
import gc
import time
import cv2
import h5py
import glob
import logging
import numpy as np
from tqdm import tqdm
import multiprocessing
from pathlib import Path
import traceback
import subprocess, shlex
import re
import random
import sys

import torch
import torchaudio
from resemblyzer import VoiceEncoder

import mediapipe
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LIPS

# Optional fairseq / AV-HuBERT
try:
    import fairseq

    HAS_FAIRSEQ = True
except Exception:
    HAS_FAIRSEQ = False

from text_processing import CTCTokenizer
from masking import GilbertElliottModel
# from avhubert_featurizer import load_avhubert, extract_visual_feature

# -------------------- constants --------------------
LANDMARK_DIM = 478
SR = 16000
FPS = 25.0
DURATION_SEC = 3.0
T_TARGET = int(round(FPS * DURATION_SEC))
AUDIO_LEN = int(round(SR * DURATION_SEC))
lip_indices = sorted(set(i for connection in FACEMESH_LIPS for i in connection))


# -------------------- init per-worker --------------------
def init_worker():
    class Mel_Spectrogram(torch.nn.Module):
        def __init__(self):
            super(Mel_Spectrogram, self).__init__()
            self.melspctrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=SR,
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
            self.amp2db_transform = torchaudio.transforms.AmplitudeToDB(stype='magnitude', top_db=80)

        def forward(self, audio_1d: torch.Tensor):
            padding = (512 - 160) // 2
            wav = torch.nn.functional.pad(audio_1d, (padding, padding), "constant")
            mel_mag = self.melspctrogram(wav)  # [n_mels, frames]
            mel_mag = mel_mag.clamp(min=1e-5)
            logmel = self.amp2db_transform(mel_mag)  # [n_mels, frames]
            return logmel

    global mel_transform
    mel_transform = Mel_Spectrogram()

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

    # ---- AV-HuBERT load (optional) ----
    # global avhubert_model, task
    # avhubert_model, task = None, None
    #
    # try:
    #     avhubert_model, task = load_avhubert()
    #     logging.info(f"Loaded AV-HuBERT")
    # except Exception as e:
    #     logging.warning(f"AV-HuBERT not available: {e}")
    # else:
    #     logging.info("AV-HuBERT not configured; set AVHUBERT_CKPT.")


# -------------------- streaming helpers (OpenCV + FFmpeg) --------------------
def read_audio_ffmpeg(path: str, target_sr: int = SR, max_sec: float = DURATION_SEC) -> torch.Tensor:
    """
    Stream audio as float32 mono at target_sr directly from ffmpeg.
    Returns 1-D torch.float32 of exact length target_sr * max_sec (trimmed/padded).
    """
    audio_len = int(round(target_sr * max_sec))
    cmd = f'ffmpeg -v error -i {shlex.quote(path)} -vn -ac 1 -ar {target_sr} -t {max_sec} -f f32le -'
    p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE)
    raw = p.stdout.read()
    p.stdout.close()
    p.wait()

    # a = np.frombuffer(raw, dtype=np.float32)
    a = np.frombuffer(raw, dtype=np.float32).copy()  # writeable
    a = torch.from_numpy(a)
    if a.numel() < audio_len:
        a = torch.nn.functional.pad(a, (0, audio_len - a.numel()))
    else:
        a = a[:audio_len]
    return a


def stream_decode_opencv(
        path: str,
        face_mesh,
        frame_size=(96, 96),
        target_fps: float = FPS,
        max_sec: float = DURATION_SEC,
        lip_indices=lip_indices
):
    """
    Stream video frames via OpenCV and crop mouth ROI per frame with MediaPipe.
    Returns:
        frames_roi: np.uint8  [T, H, W, 1]  (grayscale with channel dim)
        landmarks:  np.float32[T, L, 2]
    """
    T = int(round(target_fps * max_sec))
    frames_roi = np.zeros((T, frame_size[0], frame_size[1], 1), dtype=np.uint8)
    landmarks = np.zeros((T, len(lip_indices), 2), dtype=np.float32)

    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 1e-3:
        src_fps = target_fps

    ratio = max(1, int(round(src_fps / target_fps)))
    t_written, i = 0, 0

    while t_written < T:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if (i % ratio) != 0:
            i += 1
            continue
        i += 1

        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # RGB uint8
        res = face_mesh.process(frame)
        lm_full = np.zeros((LANDMARK_DIM, 2), dtype=np.float32)
        h, w = frame.shape[:2]

        if res.multi_face_landmarks:
            face = res.multi_face_landmarks[0]
            lm_full = np.array([[lm.x * w, lm.y * h] for lm in face.landmark], dtype=np.float32)
            lips = lm_full[lip_indices]
            x1, y1 = np.min(lips, axis=0)
            x2, y2 = np.max(lips, axis=0)
            pad_x = (x2 - x1) * 0.5
            pad_y = (y2 - y1) * 0.5
            x1 = int(max(0, x1 - pad_x))
            y1 = int(max(0, y1 - pad_y))
            x2 = int(min(w, x2 + pad_x))
            y2 = int(min(h, y2 + pad_y))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                gray = np.zeros(frame_size, dtype=np.uint8)
            else:
                gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                gray = cv2.resize(gray, frame_size)
        else:
            gray = np.zeros(frame_size, dtype=np.uint8)

        frames_roi[t_written, :, :, 0] = gray
        landmarks[t_written] = lm_full[lip_indices]
        t_written += 1

    cap.release()

    if t_written < T:
        frames_roi[t_written:] = 0

    return frames_roi, landmarks


# -------------------- masks & spec --------------------
def extract_spectral(audio_1d: torch.Tensor) -> np.ndarray:
    """Returns np.float16 spectrogram (n_mels, Tspec)"""
    spec = mel_transform(audio_1d)  # torch.float32
    spec = spec.clamp_min(1e-5).to(torch.float32)
    return spec.cpu().numpy()


def generate_random_mask(size, loss_rate):
    model = GilbertElliottModel(loss_rate=loss_rate)
    return model.simulate(*size)


def generate_uniform_mask(size, loss_bounds=(0.3, 0.7)):
    model = GilbertElliottModel(loss_rate=np.random.uniform(*loss_bounds))
    return model.simulate(*size)


# -------------------- main feature extraction --------------------
def extract_features(video_path):
    results = []
    try:
        # Stream decode (never load full clip)
        frames_roi, landmarks = stream_decode_opencv(
            video_path, face_mesh,
            frame_size=(96, 96),
            target_fps=FPS,
            max_sec=DURATION_SEC,
            lip_indices=lip_indices
        )
        audio = read_audio_ffmpeg(video_path, target_sr=SR, max_sec=DURATION_SEC)

        if audio is None or frames_roi is None:
            return results

        # Spectrogram (np.float32)
        spec = extract_spectral(audio)

        # Alignment text for GRID-style dataset
        align_path = Path(video_path).parent / "align" / (Path(video_path).stem + ".align")
        encoded_text = ctc_tokenizer.load_alignment(str(align_path))

        # Speaker embedding (from this clip's audio; keep it simple here)
        spkr_embed = voice_encoder.embed_utterance(audio.detach().cpu().numpy().astype(np.float32))

        # --- AV-HuBERT visual features (optional) ---
        # avhubert_vis = None
        # if avhubert_model is not None:
        #     try:
        #         with torch.no_grad():
        #             avhubert_vis, _ = extract_visual_feature(avhubert_model,
        #                                                      task,
        #                                                      frames_roi[..., 0])
        #             print(f"Video feature shape: {avhubert_vis}")
        #             avhubert_vis = avhubert_vis.squeeze(dim=0)
        #
        #     except Exception as e:
        #         logging.warning(f"AV-HuBERT features failed for {video_path}: {e}")
        #         avhubert_vis = None

        is_trainval = ('/train/' in video_path.lower()) or ('/val/' in video_path.lower())

        base = {
            'video_path': video_path,
            'text': encoded_text,
            'frames': frames_roi.astype(np.uint8),  # [T,H,W,1]
            'landmarks': landmarks.astype(np.float32),  # [T,L,2]
            'spec': spec,  # [80, Tspec] float32
            'spkr_embd': spkr_embed,  # [256] float32
        }
        # if avhubert_vis is not None:
        #     base['avhubert_vis'] = avhubert_vis  # [T', C] float32

        if is_trainval:
            base['mask'] = generate_uniform_mask(spec.shape, loss_bounds=(0.3, 0.7))
        else:
            base['mask'] = generate_uniform_mask(spec.shape, loss_bounds=(0.2, 0.6))
            for pct in (20, 30, 40, 50, 60, 70):
                base[f'mask_{pct}'] = generate_random_mask(spec.shape, loss_rate=pct / 100.0)

        results = base

    except Exception as e:
        logging.error(f"Error processing {video_path}: {str(e)}")
        logging.error(traceback.format_exc())
    finally:
        gc.collect()
    return results


# -------------------- parallel driver --------------------
def extract_features_parallel(video_list, output_file, num_workers=None,
                              chunk_size=1000, mode='w'):  # small chunk_size to lower peak RAM
    # mode = 'w' : separate chunk files
    # mode = 'a' : append to a single file
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    base_name, ext = os.path.splitext(output_file)
    if not ext:
        ext = '.h5'

    if num_workers is None:
        num_workers = min(2, multiprocessing.cpu_count())  # cap for laptops

    chunk_info = []
    total_processed = 0

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=num_workers, initializer=init_worker, maxtasksperchild=25) as pool:

        results = []
        chunk_count = 0

        for result in tqdm(
                pool.imap_unordered(extract_features, video_list, chunksize=4),
                total=len(video_list),
                desc="Processing videos"):
            if result:
                results.append(result)
            if len(results) >= chunk_size:
                current_output_file = f"{base_name}_chunk{chunk_count}{ext}" if mode == 'w' else output_file

                n = len(results)
                write_h5(current_output_file, results, total_processed, mode=mode)
                chunk_info.append({
                    'chunk_id': chunk_count,
                    'filename': current_output_file,
                    'num_videos': n,
                    'start_idx': total_processed,
                    'end_idx': total_processed + n - 1
                })
                total_processed += n
                results.clear()
                chunk_count += 1
                gc.collect()
                time.sleep(0.1)

    if results:
        current_output_file = f"{base_name}_chunk{chunk_count}{ext}" if mode == 'w' else output_file
        n = len(results)
        write_h5(current_output_file, results, total_processed, mode=mode)
        chunk_info.append({
            'chunk_id': chunk_count,
            'filename': current_output_file,
            'num_videos': n,
            'start_idx': total_processed,
            'end_idx': total_processed + n - 1
        })
        total_processed += n
        results.clear()
        chunk_count += 1

    metadata_file = f"{base_name}_metadata.txt"
    with open(metadata_file, 'w') as f:
        f.write(f"Total videos processed: {total_processed}\n")
        f.write(f"Total chunks: {chunk_count}\n")
        f.write("\nChunk details:\n")
        for chunk in chunk_info:
            f.write(
                f"Chunk {chunk['chunk_id']}: {chunk['filename']} - "
                f"{chunk['num_videos']} videos (indices {chunk['start_idx']}-{chunk['end_idx']})\n"
            )

    if mode == 'w':
        logging.info(f"Saved {total_processed} results across {chunk_count} files")
        logging.info(f"Metadata saved to {metadata_file}")
    else:
        logging.info(f"Appended {total_processed} results to {output_file}")


# -------------------- HDF5 I/O --------------------
def write_h5(output_file, results, start_index, mode):
    with h5py.File(output_file, mode, libver='latest') as h5f:
        for idx, result in enumerate(results):
            video_idx = start_index + idx
            video_key = f"video_{video_idx}"
            h5f.create_dataset(f"{video_key}/frames", data=result["frames"], compression="gzip")
            h5f.create_dataset(f"{video_key}/landmarks", data=result["landmarks"], compression="gzip")
            h5f.create_dataset(f"{video_key}/spec", data=result["spec"], compression="gzip")
            h5f.create_dataset(f"{video_key}/text", data=result["text"], compression="gzip")
            h5f.create_dataset(f"{video_key}/spkr_embd", data=result["spkr_embd"], compression="gzip")
            h5f.create_dataset(f"{video_key}/mask", data=result["mask"], compression="gzip")
            h5f.attrs[f"{video_key}/video_path"] = result["video_path"]

            # Optional masks
            for pct in (20, 30, 40, 50, 60, 70, 80):
                key = f"mask_{pct}"
                if key in result:
                    h5f.create_dataset(f"{video_key}/{key}", data=result[key], compression="gzip")

            # AV-HuBERT features
            # if 'avhubert_vis' in result and result['avhubert_vis'] is not None:
            #     h5f.create_dataset(f"{video_key}/avhubert_vis", data=result['avhubert_vis'], compression="gzip")

    logging.info(f"Wrote {len(results)} results to {output_file}")


# -------------------- update_h5 (units) --------------------
def update_h5(chunk_file):
    hubert_discrete = torch.hub.load("bshall/hubert:main", "hubert_discrete", trust_repo=True).cuda()
    logging.info(f"hubert_discrete model loaded successfully in process {os.getpid()}")

    print(f"Processing: {chunk_file}")
    with h5py.File(chunk_file, 'r+') as h5f:
        video_keys = list(h5f.keys())
        for video_key in tqdm(video_keys, desc=f"{os.path.basename(chunk_file)}"):
            video_path = h5f.attrs.get(f"{video_key}/video_path", None)
            if video_path is not None:
                audio = read_audio_ffmpeg(video_path, target_sr=SR, max_sec=DURATION_SEC)
                if audio is None or audio.numel() == 0:
                    print(f"Warning: No usable audio for {video_key}")
                    continue
                units = hubert_discrete.units(audio.unsqueeze(0).cuda())
                if f"{video_key}/units" in h5f:
                    del h5f[f"{video_key}/units"]
                h5f.create_dataset(f"{video_key}/units", data=units.cpu().numpy(), compression="gzip")
            else:
                print(f"Warning: No video path found for {video_key}")
    print(f"Completed: {chunk_file}")


def update_h5_parallel(base_path, chunk_pattern="_chunk*.h5"):
    chunk_files = sorted(glob.glob(f"{base_path}{chunk_pattern}"))
    with multiprocessing.Pool(processes=min(16, len(chunk_files))) as pool:
        pool.map(update_h5, chunk_files)


# -------------------- entry --------------------
if __name__ == "__main__":
    # Keep threads/processes tame (helps avoid SIGKILL on laptops)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    torch.set_num_threads(1)
    multiprocessing.set_start_method("spawn", force=True)

    splits = {"train"}  # add "val", "test" if desired

    for split in splits:
        path = f'/Users/kadkhodm/PycharmProjects/speech_inpainting/datasets/grid/{split}/'
        video_list = glob.glob(os.path.join(path, 's*/*.mpg'))

        feats_filename = f'datasets/grid_{split}_features.h5'
        extract_features_parallel(video_list, feats_filename)

        # feats_path = f'/home/ai/Projects/Mahsa/datasets/vox2_short/vox2_short_{split}_features'
        # update_h5_parallel(feats_path)
