import os
import glob
import csv
from tqdm import tqdm
import numpy as np
import h5py
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# ---------- core checks ----------

def _expected_spk_dim():
    # adjust if your speaker embedding size differs
    return 256

def _mask_key(mask_range: str) -> str:
    return 'mask' if mask_range == 'rand' else f'mask_{mask_range}'

def _quick_check(h5f, key, mode: str, mask_range: str):
    """Check required keys + shapes without full reads. Return reason str or None."""
    required = [f"{key}/spec", f"{key}/{_mask_key(mask_range)}", f"{key}/text"]
    if mode in ('v', 'av'):
        required += [f"{key}/frames", f"{key}/spkr_embd"]
    if mode == 'motion':
        required += [f"{key}/landmarks"]

    for g in required:
        if g not in h5f:
            return f"missing dataset '{g}'"

    spec_ds = h5f[f"{key}/spec"]
    mask_ds = h5f[f"{key}/{_mask_key(mask_range)}"]

    if spec_ds.ndim != 2:
        return f"spec.ndim={spec_ds.ndim} (expected 2)"
    if mask_ds.shape != spec_ds.shape:
        return f"mask shape {mask_ds.shape} != spec shape {spec_ds.shape}"
    if spec_ds.shape[0] <= 0 or spec_ds.shape[1] <= 0:
        return f"non-positive spec shape {spec_ds.shape}"

    if mode in ('v','av'):
        frames_ds = h5f[f"{key}/frames"]
        if frames_ds.ndim != 4:
            return f"frames.ndim={frames_ds.ndim} (expected 4: [T,H,W,C])"
        T_v, H, W, C = frames_ds.shape
        if C not in (1,3,4):
            return f"frames C={C} (expected 1,3,4)"
        if T_v <= 0 or H <= 0 or W <= 0:
            return f"non-positive frames shape {frames_ds.shape}"

        # rough A:V alignment (audio hop 10ms, video ~40ms → 4:1)
        T_a = spec_ds.shape[1]
        if abs((T_a / 4.0) - T_v) > 10:
            return f"time misalignment: T_audio={T_a} vs T_video={T_v}"

        spk = h5f[f"{key}/spkr_embd"]
        if spk.ndim == 1 and spk.shape[0] != _expected_spk_dim():
            return f"spk_embd dim {spk.shape} (expected {_expected_spk_dim()})"

    if mode == 'motion':
        lm = h5f[f"{key}/landmarks"]
        if lm.ndim != 3 or lm.shape[-1] != 2:
            return f"landmarks shape {lm.shape} (expected [T,points,2])"

    return None

def _light_data_check(h5f, key, mode: str, mask_range: str, frames_mb_cap: int = 300):
    """Read tiny slices to catch NaN/Inf & all-zero frames; avoid heavy I/O."""
    spec = h5f[f"{key}/spec"]
    mask = h5f[f"{key}/{_mask_key(mask_range)}"]
    F, T = spec.shape
    f0, f1 = max(0, F//2 - 8), min(F, F//2 + 8)
    t0, t1 = max(0, T//2 - 32), min(T, T//2 + 32)

    spec_probe = spec[f0:f1, t0:t1][()]
    mask_probe = mask[f0:f1, t0:t1][()]
    if not np.isfinite(spec_probe).all():
        return "spec has NaN/Inf (probe)"
    if not np.isfinite(mask_probe).all():
        return "mask has NaN/Inf (probe)"

    if mode in ('v','av'):
        frames = h5f[f"{key}/frames"]
        T_v, H, W, C = frames.shape
        # estimate memory to guard OOM
        est_bytes = T_v * H * W * C * frames.dtype.itemsize
        if est_bytes > frames_mb_cap * 1024 * 1024:
            return f"frames too large (≈{est_bytes/1e6:.1f} MB)"

        # sample up to 8 frames
        idxs = np.linspace(0, T_v-1, num=min(8, T_v), dtype=int)
        some = frames[idxs][()]
        if not np.isfinite(some).all():
            return "frames have NaN/Inf (probe)"
        if not np.any(some != 0):
            return "frames all-zero (probe)"

    if mode == 'motion':
        lm = h5f[f"{key}/landmarks"]
        idxs = np.linspace(0, lm.shape[0]-1, num=min(8, lm.shape[0]), dtype=int)
        some = lm[idxs][()]
        if not np.isfinite(some).all():
            return "landmarks have NaN/Inf (probe)"
        if not np.any(some != 0):
            return "landmarks all-zero (probe)"

    return None

def is_bad_sample(h5f, video_key, mode='v', mask_range='rand'):
    """Return (True, reason) if bad; else (False, None)."""
    reason = _quick_check(h5f, video_key, mode, mask_range)
    if reason: return True, f"quick_check: {reason}"
    reason = _light_data_check(h5f, video_key, mode, mask_range)
    if reason: return True, f"light_check: {reason}"
    # Optional: do a cheap “all-zero after mask” check on audio
    spec = h5f[f"{video_key}/spec"][()]
    mask = h5f[f"{video_key}/{_mask_key(mask_range)}"][()]
    mel = (spec.astype(np.float32))  # normalization not needed for this check
    if not np.any((mel * mask) != 0):
        return True, "masked_spec all zero"
    return False, None

# ---------- your revised functions ----------

def process_single_chunk_filter(
    chunk_file,
    mode='v',
    mask_range='rand',
    delete=True,
    log_dir=None,
    frames_mb_cap=300
):
    """
    Scan a single H5 chunk, print/log reasons, and optionally delete bad groups.
    """
    removed = 0
    bad_rows = []

    # prepare per-chunk log
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{os.path.basename(chunk_file)}.bad.csv")
        write_header = not os.path.exists(log_path)
    else:
        log_path = None

    with h5py.File(chunk_file, 'r+') as h5f:
        video_keys = list(h5f.keys())  # list copy so we can delete safely
        for video_key in tqdm(video_keys, desc=f"{os.path.basename(chunk_file)}"):
            try:
                video_path = h5f.attrs.get(f"{video_key}/video_path", "")
                bad, reason = is_bad_sample(h5f, video_key, mode=mode, mask_range=mask_range)
                if bad:
                    print(f"[BAD] {video_key} | {reason} | path={video_path}")
                    if delete and video_key in h5f:
                        del h5f[video_key]
                        removed += 1
                    bad_rows.append({
                        "chunk": os.path.basename(chunk_file),
                        "key": video_key,
                        "path": video_path,
                        "reason": reason
                    })
            except Exception as e:
                # if evaluating the sample itself crashed, record and delete it
                print(f"[ERROR] {video_key}: {e}")
                if delete and video_key in h5f:
                    del h5f[video_key]
                    removed += 1
                bad_rows.append({
                    "chunk": os.path.basename(chunk_file),
                    "key": video_key,
                    "path": h5f.attrs.get(f"{video_key}/video_path", ""),
                    "reason": f"exception: {e}"
                })

    # write CSV log (append)
    if log_path and bad_rows:
        with open(log_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["chunk", "key", "path", "reason"])
            if write_header:
                w.writeheader()
            w.writerows(bad_rows)

    print(f"{os.path.basename(chunk_file)}: removed {removed} bad samples.")
    return removed, bad_rows

def update_h5(base_path, mode='v', mask_range='rand', chunk_pattern="_chunk*.h5", log_dir=None, delete=True):
    chunk_files = sorted(glob.glob(f"{base_path}{chunk_pattern}"))
    total_removed = 0
    for chunk_file in chunk_files:
        removed, _ = process_single_chunk_filter(
            chunk_file,
            mode=mode,
            mask_range=mask_range,
            delete=delete,
            log_dir=log_dir
        )
        total_removed += removed
    print(f"\nDone. Total removed across chunks: {total_removed}")
    return total_removed

# assumes process_single_chunk_filter(...) is defined at module top-level

def _worker_process_one(chunk_file, mode, mask_range, delete, log_dir):
    # per-process imports/env to avoid inherited handles & ensure locking
    os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'TRUE')  # keep locking on
    # Open/modify *only this* file in this process:
    return process_single_chunk_filter(
        chunk_file,
        mode=mode,
        mask_range=mask_range,
        delete=delete,
        log_dir=log_dir
    )

def update_h5_parallel(
    base_path,
    mode='v',
    mask_range='rand',
    chunk_pattern="_chunk*.h5",
    log_dir=None,
    delete=True,
    max_workers=None
):
    chunk_files = sorted(glob.glob(f"{base_path}{chunk_pattern}"))
    if not chunk_files:
        print(f"No HDF5 files found at pattern: {base_path}{chunk_pattern}")
        return 0

    # sensible default: half your cores, capped by num files
    if max_workers is None:
        cores = os.cpu_count() or 4
        max_workers = min(len(chunk_files), max(1, cores // 2))

    total_removed = 0
    # Use "spawn" to avoid inherited HDF5 handles; guard with if __name__ == '__main__' on Windows/Mac.
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        futures = {
            ex.submit(_worker_process_one, cf, mode, mask_range, delete, log_dir): cf
            for cf in chunk_files
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Filtering chunks"):
            cf = futures[fut]
            try:
                removed, _rows = fut.result()
                total_removed += removed
            except Exception as e:
                print(f"[WORKER ERROR] {cf}: {e}")

    print(f"\nDone. Total removed across chunks: {total_removed}")
    return total_removed


if __name__ == "__main__":
    splits = {"val"}  # or {'train','dev','val','test'}
    for split in splits:
        #feats_path = f'/home/ai/Projects/Mahsa/datasets/vox2_short/vox2_short_{split}_features'
        feats_path = f'/home/ai/Projects/Mahsa/datasets/grid/grid_{split}_features'
        update_h5_parallel(
            feats_path,
            mode='v',                 # 'v' or 'av' for video checks; use 'a' or 'motion' if needed
            mask_range='rand',
            chunk_pattern="_chunk*.h5",
            log_dir=f"./bad_logs_{split}",  # per-chunk CSVs with reasons
            delete=True
        )
