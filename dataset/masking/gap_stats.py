import numpy as np
import h5py


def get_burst_lengths_frames(loss_trace: np.ndarray) -> list[int]:
    """
    loss_trace: 1D array of 0/1 where 1 = lost, 0 = received.
    Returns a list of contiguous loss-burst lengths in *frames*.
    """
    lengths = []
    current = 0
    for v in loss_trace:
        if v == 0:
            current += 1
        else:
            if current > 0:
                lengths.append(current)
                current = 0
    # If we ended in a loss burst, close it
    if current > 0:
        lengths.append(current)
    return lengths

def categorize_bursts_ms(burst_lengths_frames: list[int],
                         frame_ms: int = 10,
                         small_ms: int = 200,
                         medium_ms: int = 400,
                         large_ms: int = 1000):
    """
    Returns counts per category:
      - 'short'  : < small_ms
      - 'medium' : [small_ms, medium_ms)
      - 'large'  : [medium_ms, large_ms]
      - 'extra'  : > large_ms (optional, but useful to track)
    """
    small_f  = small_ms  // frame_ms  # 200ms -> 20 frames
    medium_f = medium_ms // frame_ms  # 400ms -> 40 frames
    large_f  = large_ms  // frame_ms  # 1000ms -> 100 frames

    counts = {
        "short": 0,
        "medium": 0,
        "large": 0,
        "extra": 0,  # > large_ms
    }

    for L in burst_lengths_frames:
        if L < small_f:
            counts["short"] += 1
        elif L < medium_f:
            counts["medium"] += 1
        elif L <= large_f:
            counts["large"] += 1
        else:
            counts["extra"] += 1

    return counts

def compute_gap_stats(h5_file_paths,
                      pct_list=(20, 30, 40, 50, 60),
                      frame_ms: int = 10):
    """
    h5_file_paths: list of paths to HDF5 files.
    Returns a dict: {pct: {"short": ..., "medium": ..., "large": ..., "extra": ...}}
    aggregated over all files and all samples.
    """
    # Global accumulators per loss rate
    total_counts = {
        pct: {"short": 0, "medium": 0, "large": 0, "extra": 0}
        for pct in pct_list
    }

    for h5_file_path in h5_file_paths:
        print(f"Processing {h5_file_path}")
        with h5py.File(h5_file_path, "r") as f:
            for key in f.keys():
                grp = f[key]  # group for one sample or chunk
                # print(grp.keys())  # uncomment if you want to inspect structure

                for pct in pct_list:
                    ds_name = f"mask_{pct}"
                    if ds_name not in grp:
                        # This group doesn't have this mask; skip
                        continue

                    mask_ds = grp[ds_name]

                    # Convert to numpy
                    mask_arr = np.asarray(mask_ds)

                    # Handle shapes: (freq, time) or (time,)
                    if mask_arr.ndim == 2:
                        # Use the first frequency row; masks are same over freqs
                        loss_trace = mask_arr[0, :]
                    elif mask_arr.ndim == 1:
                        loss_trace = mask_arr
                    else:
                        raise ValueError(
                            f"Unexpected mask shape {mask_arr.shape} for {ds_name} in group {key}"
                        )

                    # Ensure it's 0 (lost) / 1 (kept) integers
                    loss_trace = (loss_trace > 0).astype(int)
                    # BUT: get_burst_lengths_frames expects 0 = lost, non-zero = received,
                    # and we treat 0 as lost already. So this is okay.

                    burst_lengths = get_burst_lengths_frames(loss_trace)
                    if not burst_lengths:
                        continue  # no loss bursts in this sample

                    counts = categorize_bursts_ms(
                        burst_lengths,
                        frame_ms=frame_ms,
                    )

                    # Accumulate
                    for k in total_counts[pct]:
                        total_counts[pct][k] += counts[k]

    return total_counts


if __name__ == "__main__":
    import glob

    h5_files = glob.glob("/home/ai/Projects/Mahsa/datasets/grid/grid_test_features_chunk*.h5")
    stats = compute_gap_stats(h5_files)

    # Pretty-print counts and fractions
    for pct in sorted(stats.keys()):
        counts = stats[pct]
        total_bursts = sum(counts.values()) or 1  # avoid div-by-zero
        fractions = {k: v / total_bursts for k, v in counts.items()}

        print(f"\nLoss rate ≈ {pct}%")
        print(f"  Short  (<200 ms):      {counts['short']} ({fractions['short']:.3f})")
        print(f"  Medium (200–400 ms):   {counts['medium']} ({fractions['medium']:.3f})")
        print(f"  Large  (400–1000 ms):  {counts['large']} ({fractions['large']:.3f})")
        print(f"  Extra  (>1000 ms):     {counts['extra']} ({fractions['extra']:.3f})")

