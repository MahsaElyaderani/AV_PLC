"""
transcribe_lrs2_main.py

Uses WhisperX to transcribe every video/audio in the LRS2 *main* split
and writes a .txt sidecar file in the same pretrain format:

    Text:  AND WE NEED TO LOOK AT FISH
    Conf:  <avg_confidence>

    WORD START END ASDSCORE
    AND  0.04 0.21 3.2
    WE   0.22 0.35 4.1
    ...

If a .txt already exists (original LRS2 main annotation), it is BACKED UP
as .txt.orig before being overwritten — so you can always restore.

Requirements:
    pip install whisperx

Usage:
    python transcribe_lrs2_main.py
    (edit the CONFIG section at the bottom before running)
"""

import os
import glob
import json
import subprocess
import shlex
import concurrent.futures
import traceback
import numpy as np

import torch
from tqdm import tqdm
import whisperx


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_audio_np(path: str, sr: int = 16000) -> np.ndarray:
    """Load audio as float32 numpy array via ffmpeg. Same approach as save_features.py."""
    cmd = f'ffmpeg -v error -i {shlex.quote(path)} -vn -ac 1 -ar {sr} -f f32le -'
    out = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, check=True).stdout
    audio = np.frombuffer(out, dtype=np.float32).copy()
    peak = np.max(np.abs(audio)) or 1.0
    if peak > 1.0:
        audio /= peak
    return audio


def format_txt(words: list[dict], avg_conf: float) -> str:
    """
    Format transcription result into LRS2 pretrain-style .txt content.

    words: list of dicts with keys 'word', 'start', 'end', 'score'
    """
    sentence = ' '.join(w['word'].upper() for w in words)

    lines = [
        f"Text:  {sentence}",
        f"Conf:  {avg_conf:.1f}",
        "",
        "WORD START END ASDSCORE",
    ]
    for w in words:
        word  = w['word'].upper()
        start = round(w.get('start', 0.0), 2)
        end   = round(w.get('end',   0.0), 2)
        score = round(w.get('score', 0.0), 1)
        lines.append(f"{word} {start} {end} {score}")

    return '\n'.join(lines) + '\n'


def write_txt(txt_path: str, content: str, backup: bool = True) -> None:
    """Write content to txt_path, backing up any existing file first."""
    if backup and os.path.isfile(txt_path):
        backup_path = txt_path + '.orig'
        if not os.path.isfile(backup_path):           # only backup once
            os.rename(txt_path, backup_path)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(content)


# ──────────────────────────────────────────────────────────────────────────────
# Per-file transcription
# ──────────────────────────────────────────────────────────────────────────────

def transcribe_file(video_path: str, model, align_model, align_metadata,
                    device: str, sr: int = 16000,
                    backup: bool = True) -> str | None:
    """
    Transcribe one video/audio file and write the .txt sidecar.
    Returns the txt_path on success, None on failure.
    """
    txt_path = os.path.splitext(video_path)[0] + '.txt'

    # Skip if already transcribed (has word-level timestamps)
    if os.path.isfile(txt_path) and not os.path.isfile(txt_path + '.orig'):
        with open(txt_path) as f:
            content = f.read()
        # If it already has a WORD block it's already done (or is pretrain format)
        if 'WORD START END' in content:
            return txt_path   # already has timestamps — skip

    try:
        audio_np = load_audio_np(video_path, sr=sr)

        # WhisperX transcribe — returns dict with 'segments'
        result = model.transcribe(audio_np, batch_size=1, language='en')

        # Align to get word-level timestamps
        result_aligned = whisperx.align(
            result['segments'],
            align_model,
            align_metadata,
            audio_np,
            device,
            return_char_alignments=False,
        )

        # Collect all words across all segments
        all_words = []
        for seg in result_aligned.get('word_segments', []):
            # whisperx word_segments: {'word': str, 'start': float, 'end': float, 'score': float}
            if 'start' not in seg or 'end' not in seg:
                continue   # word with no alignment (silence etc.)
            all_words.append({
                'word':  seg['word'].strip(),
                'start': seg['start'],
                'end':   seg['end'],
                'score': seg.get('score', 0.0),
            })

        if not all_words:
            print(f"  WARNING: no aligned words for {video_path}", flush=True)
            return None

        scores = [w['score'] for w in all_words if w['score'] > 0]
        avg_conf = float(np.mean(scores)) if scores else 0.0

        content = format_txt(all_words, avg_conf)
        write_txt(txt_path, content, backup=backup)
        return txt_path

    except Exception as e:
        print(f"  ERROR: {video_path}: {e}", flush=True)
        traceback.print_exc()
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Batch driver
# ──────────────────────────────────────────────────────────────────────────────

def transcribe_lrs2_main(
    root_path: str,
    split_list_txt: str,
    device: str = 'cuda',
    whisper_model_size: str = 'medium',
    compute_type: str = 'float16',       # 'float16' on GPU, 'int8' on CPU
    batch_size: int = 1,
    checkpoint_file: str = None,
    backup: bool = True,
    num_workers: int = 1,                # >1 only if you have multiple GPUs
) -> None:
    """
    Transcribe all videos listed in split_list_txt using WhisperX and write
    word-timestamped .txt files in LRS2 pretrain format.

    Parameters
    ----------
    root_path       : root of the LRS2 main split, e.g.
                      '/home/amin/.../lrs2_v1/mvlrs_v1/main'
    split_list_txt  : path to LRS2 split list file, e.g.
                      '/home/amin/.../lrs2/train.txt'
                      Each line: 'speaker/clip_id [optional_duration]'
    device          : 'cuda' or 'cpu'
    whisper_model_size : WhisperX model size ('tiny','base','small','medium','large-v2')
    compute_type    : 'float16' for GPU, 'int8' for CPU
    checkpoint_file : JSON file tracking completed files; auto-named if None
    backup          : if True, rename existing .txt to .txt.orig before overwriting
    num_workers     : parallel workers (keep 1 unless multi-GPU)
    """

    # ── build video list ─────────────────────────────────────────────────
    with open(split_list_txt) as f:
        rel_paths = [line.strip().split()[0] for line in f if line.strip()]
    video_paths = [os.path.join(root_path, r + '.mp4') for r in rel_paths]
    video_paths = [p for p in video_paths if os.path.isfile(p)]
    print(f"Found {len(video_paths)} video files.", flush=True)

    # ── checkpoint ───────────────────────────────────────────────────────
    if checkpoint_file is None:
        checkpoint_file = os.path.join(
            os.path.dirname(split_list_txt),
            os.path.basename(split_list_txt).replace('.txt', '_transcribe_ckpt.json')
        )
    completed = set()
    if os.path.isfile(checkpoint_file):
        try:
            with open(checkpoint_file) as f:
                completed = set(json.load(f))
            print(f"Checkpoint: {len(completed)} already done.")
        except Exception:
            pass

    pending = [p for p in video_paths if p not in completed]
    print(f"{len(pending)} files to process.", flush=True)
    if not pending:
        print("Nothing to do.")
        return

    def _save_ckpt():
        with open(checkpoint_file, 'w') as f:
            json.dump(sorted(completed), f, indent=2)

    # ── load WhisperX model (once, on main process) ──────────────────────
    print(f"Loading WhisperX '{whisper_model_size}' on {device}...", flush=True)
    model = whisperx.load_model(
        whisper_model_size,
        device=device,
        compute_type=compute_type,
        language='en',
    )

    # Load alignment model (wav2vec2 for en)
    align_model, align_metadata = whisperx.load_align_model(
        language_code='en',
        device=device,
    )
    print("Models loaded.", flush=True)

    # ── process files ────────────────────────────────────────────────────
    ok, failed = 0, 0
    for video_path in tqdm(pending, desc="Transcribing", mininterval=5.0):
        result = transcribe_file(
            video_path, model, align_model, align_metadata,
            device=device, backup=backup,
        )
        if result:
            ok += 1
            completed.add(video_path)
            if ok % 100 == 0:          # save checkpoint every 100 files
                _save_ckpt()
        else:
            failed += 1

    _save_ckpt()
    print(f"\nDone. ok={ok}, failed={failed}")
    print(f"Checkpoint saved to: {checkpoint_file}")


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG — edit here before running
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    DATASET_BASE = '/home/amin/Projects/Mahsa/datasets/lrs2'

    for split in ['train', 'val', 'test']:
        print(f'\n{"="*60}\nProcessing split: {split}\n{"="*60}')
        transcribe_lrs2_main(
            root_path=os.path.join(DATASET_BASE, 'lrs2_v1/mvlrs_v1/main'),
            split_list_txt=os.path.join(DATASET_BASE, f'{split}.txt'),
            device='cuda' if torch.cuda.is_available() else 'cpu',
            whisper_model_size='medium',   # 'large-v2' for best accuracy
            compute_type='float16',        # change to 'int8' if running on CPU
            backup=True,                   # backs up original .txt as .txt.orig
        )