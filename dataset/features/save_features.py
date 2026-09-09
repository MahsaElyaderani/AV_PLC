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
import hashlib

import torch
import torchaudio
from resemblyzer import VoiceEncoder

import mediapipe
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LIPS

# Install mediapipe; that is compatible OpenCV build
#pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless
#pip install mediapipe

# fairseq / AV-HuBERT
try:
    from avhubert_featurizer import load_avhubert, extract_visual_feature
    HAS_FAIRSEQ = True
except Exception:
    HAS_FAIRSEQ = False

from shared.text_processing import TextTokenizer, PhonemeTokenizer
from shared.masking import (
    GilbertElliottModel, generate_ge_trace_bursty, trace_to_spec_mask,
    packet_count_from_audio_len,
)
from AV_PLC.video_preprocessing import ReferenceFaceAligner

# -------------------- constants --------------------
LANDMARK_DIM = 478
SR = 16000
FPS = 25.0
DURATION_SEC = 3.0
T_TARGET = int(round(FPS * DURATION_SEC))
AUDIO_LEN = int(round(SR * DURATION_SEC))
lip_indices = sorted(set(i for connection in FACEMESH_LIPS for i in connection))
AUDIO_STORAGE_DTYPE = os.environ.get("AVPLC_AUDIO_STORAGE_DTYPE", "float32").strip().lower()

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

    global ctc_tokenizer, phoneme_tokenizer
    ctc_tokenizer = TextTokenizer()
    phoneme_tokenizer = PhonemeTokenizer()

    global avplc_face_aligner
    reference_path = os.environ.get(
        "AVPLC_REFERENCE_FACE",
        str(Path(__file__).resolve().parents[2] / "AV_PLC" / "reference_face.npy"),
    )
    if not os.path.isfile(reference_path):
        raise FileNotFoundError(
            f"AV_PLC reference face not found: {reference_path}. "
            "Build it first with python -m AV_PLC.build_reference_face ..."
        )
    avplc_face_aligner = ReferenceFaceAligner.from_npy(reference_path)

    ######### ---- AV-HuBERT load ---- ########
    global avhubert_model, task
    avhubert_model, task = None, None

    try:
        avhubert_model, task = load_avhubert()
    except Exception as e:
        logging.warning(f"AV-HuBERT not available: {e}. Not configured? Set AVHUBERT_CKPT.")
    else:
        logging.info("Loaded AV-HuBERT")

# -------------------- streaming helpers (OpenCV + FFmpeg) --------------------

def read_audio_ffmpeg(path: str, target_sr: int, max_sec: float):
    """
    Read mono float32 audio from a media file using ffmpeg, resampled to target_sr,
    trimmed/padded to exactly target_sr * max_sec samples.

    Returns:
        audio      : torch.float32 tensor [target_sr * max_sec]
        valid_len  : int, number of real samples before padding
    """
    audio_len = int(round(target_sr * max_sec))
    cmd = f'ffmpeg -v error -i {shlex.quote(path)} -vn -ac 1 -ar {target_sr} -t {max_sec} -f f32le -'
    out = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, check=True).stdout
    a = np.frombuffer(out, dtype=np.float32).copy()  # writeable
    a = torch.from_numpy(a)
    valid_len = min(a.numel(), audio_len)
    if valid_len == 0:
        return None, None  # no audio at all

    diff = audio_len - valid_len
    audio = a[:audio_len] if diff <= 0 else torch.cat([a, torch.zeros(diff, dtype=a.dtype, device=a.device)], dim=0)

    return audio, valid_len

def stream_decode_opencv(
        path: str,
        face_mesh,
        aligner,
        frame_size=(96, 96),
        legacy_avhubert_size=(112, 112),
        target_fps: float = FPS,
        max_sec: float = DURATION_SEC,
        lip_indices=lip_indices
):
    """Decode one clip and build both visual paths.

    Returns:
        aligned_frames:   [T,96,96,1] AV_PLC reference-face-aligned mouth ROI
        lip_landmarks:    [T,L,2] legacy key used by AV_LSTM/AV_S2S
        full_landmarks:   [T,478,2] MediaPipe full-face coordinates
        video_len:        number of decoded 25-Hz frames before zero padding
        avhubert_frames:  [T,112,112,1] legacy dynamic ROI used ONLY to keep
                           stored AV-HuBERT visual_features unchanged
    """
    T = int(round(target_fps * max_sec))
    raw_frames = []
    legacy_frames = np.zeros((T, legacy_avhubert_size[0], legacy_avhubert_size[1], 1), dtype=np.uint8)
    full_landmarks = np.zeros((T, LANDMARK_DIM, 2), dtype=np.float32)
    lip_landmarks = np.zeros((T, len(lip_indices), 2), dtype=np.float32)
    detected = np.zeros(T, dtype=bool)

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

        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        raw_frames.append(frame)
        h, w = frame.shape[:2]
        result = face_mesh.process(frame)

        if result.multi_face_landmarks:
            face = result.multi_face_landmarks[0]
            lm = np.array([[p.x * w, p.y * h] for p in face.landmark], dtype=np.float32)
            n = min(LANDMARK_DIM, lm.shape[0])
            full_landmarks[t_written, :n] = lm[:n]
            lip_landmarks[t_written] = full_landmarks[t_written, lip_indices]
            detected[t_written] = True

            # Preserve the exact legacy dynamic mouth crop for AV-HuBERT features.
            lips = lip_landmarks[t_written]
            x1, y1 = np.min(lips, axis=0)
            x2, y2 = np.max(lips, axis=0)
            pad_x = (x2 - x1) * 0.5
            pad_y = (y2 - y1) * 0.5
            x1 = int(max(0, x1 - pad_x)); y1 = int(max(0, y1 - pad_y))
            x2 = int(min(w, x2 + pad_x)); y2 = int(min(h, y2 + pad_y))
            crop = frame[y1:y2, x1:x2]
            if crop.size:
                gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                legacy_frames[t_written, :, :, 0] = cv2.resize(gray, legacy_avhubert_size)
        t_written += 1

    cap.release()
    if detected[:t_written].sum() < 2:
        return None, None, None, 0, None

    actual_rgb = np.stack(raw_frames[:t_written], axis=0)
    aligned_actual = aligner.align_sequence(
        actual_rgb,
        full_landmarks[:t_written],
        detected[:t_written],
    )
    aligned_frames = np.zeros((T, frame_size[0], frame_size[1], 1), dtype=np.uint8)
    aligned_frames[:t_written] = aligned_actual
    return aligned_frames, lip_landmarks, full_landmarks, t_written, legacy_frames


def embed_from_window_05s(
    audio: torch.Tensor,
    valid_len: int,
    sr: int,
    voice_encoder,
    seconds: float = 0.5,
    policy: str = "center",  # "center" | "first" | "last" | "random" | "max_energy"
):
    """
    Return a speaker embedding from a ~0.5s window inside the real (non-padded) audio.
    - audio: 1D float32 torch.Tensor
    - valid_len: number of real samples before padding
    - sr: sample rate (use 16000 for resemblyzer)
    - voice_encoder: resemblyzer VoiceEncoder
    """
    win = int(round(seconds * sr))
    real = audio[:valid_len]  # avoid trailing zero padding

    if real.numel() == 0:
        return None  # no audio at all
    if real.numel() < win:
        # Not enough audio: either return None or pad to length 'win'
        seg = torch.nn.functional.pad(real, (0, win - real.numel()))
    else:
        if policy == "first":
            start = 0
        elif policy == "last":
            start = real.numel() - win
        elif policy == "random":
            start = int(np.random.randint(0, real.numel() - win + 1))
        elif policy == "max_energy":
            # pick the highest-average-abs window (quick voiced heuristic)
            x = real.abs().unsqueeze(0).unsqueeze(0)             # [1,1,N]
            en = torch.nn.functional.avg_pool1d(x, kernel_size=win, stride=1).squeeze()  # [N-win+1]
            start = int(torch.argmax(en).item())
            #print(start, en)
        else:  # "center"
            start = (real.numel() - win) // 2

        seg = real[start:start + win]

    wav = seg.detach().cpu().numpy().astype(np.float32)
    # Optional light normalization (resemblyzer is robust, but this can help):
    #peak = np.max(np.abs(wav)) + 1e-7
    #wav = wav / peak

    return voice_encoder.embed_utterance(wav)

# -------------------- masks & spec --------------------
def extract_spectral(audio_1d: torch.Tensor) -> np.ndarray:
    spec = mel_transform(audio_1d)  # torch.float32
    return spec.cpu().numpy()

def generate_random_mask(size, loss_rate):
    model = GilbertElliottModel(loss_rate=loss_rate)
    return model.simulate(*size)

def generate_uniform_mask(size, loss_bounds=(0.3, 0.7)):
    model = GilbertElliottModel(loss_rate=np.random.uniform(*loss_bounds))
    return model.simulate(*size)

def fill_mask(spec_shape, valid_mask):
    mask = np.ones(spec_shape, dtype=np.float32)
    mask[:, :valid_mask.shape[1]] = valid_mask
    return mask

# -------------------- main feature extraction --------------------
def extract_features(video_path):
    results = []
    try:
        # Stream decode (never load full clip)
        frames_roi, landmarks, full_landmarks, video_len, avhubert_frames = stream_decode_opencv(
            video_path, face_mesh, avplc_face_aligner,
            frame_size=(96, 96), legacy_avhubert_size=(112, 112),
            target_fps=FPS, max_sec=DURATION_SEC, lip_indices=lip_indices
        )
        audio, valid_len = read_audio_ffmpeg(video_path, target_sr=SR, max_sec=DURATION_SEC)

        if audio is None or frames_roi is None:
            return results

        # Spectrogram (np.float32)
        spec = extract_spectral(audio)
        spec_f, spec_t = spec.shape
        valid_t = min(spec_t, (int(valid_len) + 159) // 160)  # valid 10-ms packets

        transcript = ""
        if 'grid' in video_path.lower():
            # Parse the GRID alignment once so the character and phoneme
            # targets correspond to the same valid utterance.
            align_path = Path(video_path).parent / "align" / (Path(video_path).stem + ".align")
            words = []
            with open(align_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3 and parts[2].lower() != "sil":
                        words.append(parts[2])
            transcript = " ".join(words)
        elif 'lrs2' in video_path.lower():
            txt_path = os.path.splitext(video_path)[0] + ".txt"
            valid_sec = min(DURATION_SEC, valid_len / SR)
            transcript, _ = ctc_tokenizer.read_transcripts_for_chunk(
                txt_path, start_sec=0.0, max_sec=valid_sec
            )
        else:
            # VoxCeleb2 has no authoritative transcript in this extractor.
            # Store empty targets rather than inventing WER references.
            transcript = ""

        encoded_text, _ = ctc_tokenizer.encode(transcript, max_length=128)
        phone_indices, _, _ = phoneme_tokenizer.encode(transcript, max_length=128)

        # Speaker embedding (from this clip's audio; keep it simple here)
        #spkr_embed = voice_encoder.embed_utterance(audio.detach().cpu().numpy().astype(np.float32))
        spkr_embed = embed_from_window_05s(audio, valid_len, sr=SR, voice_encoder=voice_encoder,
                                           seconds=0.5, policy="max_energy")

        # --- AV-HuBERT visual features ---
        avhubert_vis = None
        if avhubert_model is not None:
            try:
                avhubert_vis = extract_visual_feature(avhubert_model,
                                                             task,
                                                             avhubert_frames[..., 0])
            except Exception as e:
                logging.warning(f"AV-HuBERT features failed for {video_path}: {e}")
                avhubert_vis = None

        is_trainval = ('/train/' in video_path.lower()) or ('/val/' in video_path.lower())

        audio_np = audio.detach().cpu().numpy().astype(np.float32)
        if AUDIO_STORAGE_DTYPE == "int16":
            audio_store = np.clip(np.rint(audio_np * 32768.0), -32768, 32767).astype(np.int16)
        elif AUDIO_STORAGE_DTYPE == "float32":
            audio_store = audio_np
        else:
            raise ValueError("AVPLC_AUDIO_STORAGE_DTYPE must be 'float32' or 'int16'")

        base = {
            'video_path': video_path,
            'audio_len': valid_len,
            'video_len': video_len,
            'audio': audio_store,
            'text': encoded_text,
            'phone_indices': np.asarray(phone_indices, dtype=np.int32),
            'frames': frames_roi.astype(np.uint8),  # new aligned 96x96 AV_PLC frames
            'landmarks': landmarks.astype(np.float32),  # keep legacy lip-only key
            'full_landmarks': full_landmarks.astype(np.float32),
            'spec': spec,  # keep legacy target for comparison projects
            'spkr_embd': spkr_embed,
        }
        if avhubert_vis is not None:
            base['visual_features'] = avhubert_vis  # [T', C] float32

        if is_trainval:
            mask_valid = generate_uniform_mask((spec_f, valid_t), loss_bounds=(0.3, 0.7))
            print("mask valid shape", mask_valid.shape)
            base['mask'] = fill_mask(spec.shape, mask_valid)
        else:
            mask_valid = generate_uniform_mask((spec_f, valid_t), loss_bounds=(0.2, 0.6))
            base['mask'] = fill_mask(spec.shape, mask_valid)
            for pct in (20, 30, 40, 50, 60):
                mask_valid = generate_random_mask((spec_f, valid_t), loss_rate=pct / 100.0)
                base[f'mask_{pct}'] = fill_mask(spec.shape, mask_valid)

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
            h5f.create_dataset(f"{video_key}/audio", data=result["audio"], compression="gzip")
            h5f.create_dataset(f"{video_key}/frames", data=result["frames"], compression="gzip")
            h5f.create_dataset(f"{video_key}/landmarks", data=result["landmarks"], compression="gzip")
            h5f.create_dataset(f"{video_key}/full_landmarks", data=result["full_landmarks"], compression="gzip")
            h5f.create_dataset(f"{video_key}/spec", data=result["spec"], compression="gzip")
            h5f.create_dataset(f"{video_key}/text", data=result["text"], compression="gzip")
            h5f.create_dataset(f"{video_key}/phone_indices", data=result["phone_indices"], compression="gzip")
            h5f.create_dataset(f"{video_key}/spkr_embd", data=result["spkr_embd"], compression="gzip")
            h5f.create_dataset(f"{video_key}/mask", data=result["mask"], compression="gzip")
            h5f.attrs[f"{video_key}/video_path"] = result["video_path"]
            h5f.attrs[f"{video_key}/audio_len"] = result["audio_len"]
            h5f.attrs[f"{video_key}/video_len"] = result["video_len"]

            #### masks
            for pct in (20, 30, 40, 50, 60, 70, 80):
                key = f"mask_{pct}"
                if key in result:
                    h5f.create_dataset(f"{video_key}/{key}", data=result[key], compression="gzip")

            ### AV-HuBERT features
            if 'visual_features' in result and result['visual_features'] is not None:
                h5f.create_dataset(f"{video_key}/visual_features", data=result['visual_features'], compression="gzip")

    logging.info(f"Wrote {len(results)} results to {output_file}")


# -------------------- update_h5 (units) --------------------
def update_h5(chunk_file):
    hubert_discrete = torch.hub.load("bshall/hubert:main", "hubert_discrete", trust_repo=True).cuda()
    logging.info(f"hubert_discrete model loaded successfully in process")

    print(f"Processing: {chunk_file}")
    with h5py.File(chunk_file, 'r+') as h5f:
        video_keys = list(h5f.keys())
        for video_key in tqdm(video_keys, desc=f"{os.path.basename(chunk_file)}"):
            video_path = h5f.attrs.get(f"{video_key}/video_path", None)
            if video_path is not None:
                audio, _ = read_audio_ffmpeg(video_path, target_sr=SR, max_sec=DURATION_SEC)
                if audio is None or audio.numel() == 0:
                    print(f"Warning: No usable audio for {video_key}")
                    continue
                units = hubert_discrete.units(audio.unsqueeze(0).unsqueeze(0).cuda()) #expects a [batch, channel, time]
                print(units.shape)
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
        #pool.map(update_mel_h5, chunk_files)

def load_video_list_from_txt(split_file, root_path, extension=".mp4"):
    rel_paths = []
    with open(split_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # take only the first token before any space/tab
            first_token = line.split()[0]
            rel_paths.append(first_token)

    return [os.path.join(root_path, rel + extension) for rel in rel_paths]

def update_h5_test_masks(base_path, chunk_pattern="_chunk*.h5", seed=42,
                         loss_rates=(1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99),
                         overwrite=False):
    """Optionally cache deterministic test masks using the CURRENT dataset logic.

    Active runners generate masks online, so this cache is not required.  When
    used for legacy/debug workflows, it must still obey ``audio_len`` and use
    the same stable sample-specific seed as all four dataset implementations.
    Padding frames are always keep=1.
    """
    import re

    def _chunk_idx(path):
        m = re.search(r'_chunk(\d+)\.h5$', os.path.basename(path))
        return int(m.group(1)) if m else -1

    chunk_files = sorted(glob.glob(f"{base_path}{chunk_pattern}"), key=_chunk_idx)
    if not chunk_files:
        raise FileNotFoundError(f"No HDF5 files found matching: {base_path}{chunk_pattern}")

    for chunk_file in chunk_files:
        with h5py.File(chunk_file, 'r+') as h5f:
            for video_key in tqdm(list(h5f.keys()), desc=os.path.basename(chunk_file)):
                spec_shape = h5f[f"{video_key}/spec"].shape
                audio_len = int(h5f.attrs.get(f"{video_key}/audio_len", spec_shape[1] * 160))
                valid_t = min(spec_shape[1], packet_count_from_audio_len(audio_len, 160))
                sample_id = f"{os.path.basename(chunk_file)}:{video_key}"

                for rate in loss_rates:
                    ds_name = f"{video_key}/mask_{rate}"
                    if not overwrite and ds_name in h5f:
                        continue
                    token = f"{seed}:{rate}:{sample_id}".encode("utf-8")
                    item_seed = int.from_bytes(hashlib.sha256(token).digest()[:4], "little")
                    state = np.random.get_state()
                    try:
                        np.random.seed(item_seed)
                        trace = generate_ge_trace_bursty(valid_t, float(rate) / 100.0)
                    finally:
                        np.random.set_state(state)
                    mask = trace_to_spec_mask(trace, spec_shape)
                    if ds_name in h5f:
                        del h5f[ds_name]
                    h5f.create_dataset(ds_name, data=mask, compression="gzip")

    logging.info("Finished deterministic valid-length test-mask cache update.")

# -------------------- entry --------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build shared HDF5 features while adding the waveform/aligned-frame data required by AV_PLC."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--videos", nargs="+", help="Video glob pattern(s); recursive ** is supported.")
    source.add_argument("--list-file", type=str, help="Split file with one relative video id per line.")
    parser.add_argument("--root", type=str, default=None, help="Root used with --list-file.")
    parser.add_argument("--extension", type=str, default=".mp4", help="Extension appended to ids from --list-file.")
    parser.add_argument("--output", required=True, help="Output base HDF5 path; write mode creates _chunkN files.")
    parser.add_argument("--reference-face", required=True, help="Reference face .npy built by AV_PLC.build_reference_face.")
    parser.add_argument("--audio-dtype", choices=["float32", "int16"], default="float32")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=1000)
    args = parser.parse_args()

    os.environ["AVPLC_REFERENCE_FACE"] = os.path.abspath(args.reference_face)
    os.environ["AVPLC_AUDIO_STORAGE_DTYPE"] = args.audio_dtype
    global AUDIO_STORAGE_DTYPE
    AUDIO_STORAGE_DTYPE = args.audio_dtype

    if args.videos:
        video_list = []
        for pattern in args.videos:
            video_list.extend(glob.glob(pattern, recursive=True))
        video_list = sorted(set(video_list))
    else:
        if not args.root:
            parser.error("--root is required with --list-file")
        video_list = load_video_list_from_txt(args.list_file, args.root, extension=args.extension)

    if not video_list:
        raise FileNotFoundError("No input videos were found")

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    torch.set_num_threads(1)
    multiprocessing.set_start_method("spawn", force=True)
    extract_features_parallel(
        video_list, args.output, num_workers=args.num_workers,
        chunk_size=args.chunk_size, mode="w",
    )


if __name__ == "__main__":
    main()
