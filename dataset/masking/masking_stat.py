"""
Burst-duration statistics for generate_ge_mask_bursty.
Mel frame rate = 100 fps (10 ms / frame), so frames = ms / 10.
"""
import numpy as np
from collections import defaultdict
from shared.masking import generate_ge_mask_bursty

# ----- config -----
LOSS_RATES   = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_CLIPS      = 5000                          # samples per loss rate
SPEC_SHAPE   = (80, 300)                     # (F, T) — 3 s clip, 100 fps
BIN_EDGES_MS = list(range(0, 200, 20)) + [np.inf]   # 0-20, 20-40, ..., 160-180, 180+
BIN_LABELS   = [f"{a}-{b}" for a, b in zip(BIN_EDGES_MS[:-1], BIN_EDGES_MS[1:])]

def burst_lengths_frames(mask):
    """Return list of consecutive-loss run lengths (in frames) along time axis."""
    trace = 1 - mask[0].astype(np.int8)        # 1 where lost; rows are identical
    runs, cur = [], 0
    for v in trace:
        if v:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    return runs

def bin_index(duration_ms, edges):
    for i in range(len(edges) - 1):
        if edges[i] <= duration_ms < edges[i + 1]:
            return i
    return len(edges) - 2

# ----- simulate -----
rng = np.random.default_rng(0)
np.random.seed(0)   # GE model uses global numpy state

results = {}     # loss_rate -> dict
for r in LOSS_RATES:
    counts        = np.zeros(len(BIN_LABELS), dtype=np.int64)
    all_durations = []
    for _ in range(N_CLIPS):
        m = generate_ge_mask_bursty(SPEC_SHAPE, r)
        for L in burst_lengths_frames(m):
            dur_ms = L * 10
            counts[bin_index(dur_ms, BIN_EDGES_MS)] += 1
            all_durations.append(dur_ms)

    all_durations = np.array(all_durations)
    n_total       = len(all_durations)
    results[r] = {
        "counts"      : counts,
        "n_total"     : n_total,
        "n_under_50"  : int((all_durations <  50).sum()),
        "n_50_200"    : int(((all_durations >= 50) & (all_durations < 200)).sum()),
        "n_200_500": int(((all_durations >= 200) & (all_durations < 500)).sum()),
        "n_500_800": int(((all_durations >= 500) & (all_durations < 800)).sum()),
        "n_800_1000": int(((all_durations >= 800) & (all_durations < 1000)).sum()),
        "n_1000_1600": int(((all_durations >= 1000) & (all_durations < 1600)).sum()),
        "n_over_1600": int((all_durations >= 1600).sum()),
        "mean_ms"     : float(all_durations.mean())   if n_total else 0.0,
        "median_ms"   : float(np.median(all_durations)) if n_total else 0.0,
        "p95_ms"      : float(np.percentile(all_durations, 95)) if n_total else 0.0,
        "max_ms"      : int(all_durations.max())      if n_total else 0,
    }

# ----- print -----
print(f"\n{N_CLIPS} clips per loss rate, clip length = {SPEC_SHAPE[1]} frames "
      f"({SPEC_SHAPE[1]*10} ms)\n")

# Per-bin counts table
header = f"{'rate':>5} | " + " | ".join(f"{lbl:>7}" for lbl in BIN_LABELS) + " | total"
print(header)
print("-" * len(header))
for r in LOSS_RATES:
    row = f"{r:>5.2f} | " + " | ".join(f"{c:>7d}" for c in results[r]["counts"]) \
          + f" | {results[r]['n_total']:>5d}"
    print(row)

# Aggregate stats
print("\nAggregate (counts + percentages of all bursts at that loss rate):")
print(f"{'rate':>5} | {'<50ms':>14} | {'50-200ms':>14} | {'200_500ms':>14} | "
      f"| {'500_800ms':>14} | {'800_1000ms':>14} | {'1000_1600ms':>14} |{'>1600ms':>14} | "
      f"{'mean':>7} | {'med':>5} | {'p95':>5} | {'max':>5}")
print("-" * 95)
for r in LOSS_RATES:
    s   = results[r]
    tot = s["n_total"] or 1
    print(f"{r:>5.2f} | "
          f"{s['n_under_50']:>6d} ({100*s['n_under_50']/tot:>4.1f}%) | "
          f"{s['n_50_200']  :>6d} ({100*s['n_50_200']  /tot:>4.1f}%) | "
          f"{s['n_200_500']:>6d} ({100*s['n_200_500']/tot:>4.1f}%) | "
          f"{s['n_500_800']:>6d} ({100*s['n_500_800']/tot:>4.1f}%) | "
          f"{s['n_800_1000']:>6d} ({100*s['n_800_1000']/tot:>4.1f}%) | "
          f"{s['n_1000_1600']:>6d} ({100*s['n_1000_1600']/tot:>4.1f}%) | "
          f"{s['n_over_1600']:>6d} ({100*s['n_over_1600']/tot:>4.1f}%) | "
          f"{s['mean_ms']:>5.0f}ms | {s['median_ms']:>3.0f}ms | "
          f"{s['p95_ms']:>3.0f}ms | {s['max_ms']:>3d}ms")
