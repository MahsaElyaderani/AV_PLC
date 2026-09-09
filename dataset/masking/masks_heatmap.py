"""
Heatmap of burst-duration distribution vs. loss rate.
- Rows  : loss rates (10%..90%)
- Cols  : burst-duration bins (100 ms..1000 ms, plus >1000 ms)
- Cell  : % of bursts in that duration bin for that loss rate.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap

# Choose which mask function to inspect.
# - generate_ge_mask_bursty : your training-time bursty masks
# - generate_ge_mask        : your test-time per-rate masks
from shared.masking import generate_ge_mask_bursty as MASK_FN

# ----- config -----
LOSS_RATES      = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
N_SAMPLES       = 5000                        # masks per loss rate
SPEC_SHAPE      = (80, 300)                   # (F, T) — 3 s clip @ 100 fps
FPS             = 100                         # frames per second
BIN_EDGES_MS    = list(range(0, 1100, 100)) + [np.inf]
                                              # 0-100, 100-200, ..., 900-1000, >1000
SEED            = 0
OUT_PNG         = "mask_burst_heatmap.png"
ANNOTATE        = True                        # draw % values on each cell

# ----- helpers -----
def burst_lengths_ms(mask, fps=FPS):
    """Lengths (ms) of consecutive lost-frame runs along time axis."""
    trace = 1 - mask[0].astype(np.int8)        # 1 where lost
    runs, cur = [], 0
    for v in trace:
        if v:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return [r * (1000 / fps) for r in runs]

def bin_label(lo, hi):
    if hi == np.inf:
        return f">{int(lo)}"
    return f"{int(lo)}-{int(hi)}"

# ----- simulate -----
np.random.seed(SEED)

bin_labels = [bin_label(BIN_EDGES_MS[i], BIN_EDGES_MS[i+1])
              for i in range(len(BIN_EDGES_MS) - 1)]
n_rates    = len(LOSS_RATES)
n_bins     = len(bin_labels)

# matrix[i, j] = % of bursts at loss_rate i that fall in bin j
matrix     = np.zeros((n_rates, n_bins), dtype=np.float64)
total_bursts_per_rate = np.zeros(n_rates, dtype=np.int64)
mean_ms_per_rate      = np.zeros(n_rates, dtype=np.float64)
median_ms_per_rate    = np.zeros(n_rates, dtype=np.float64)

print(f"Generating {N_SAMPLES} masks for each of {n_rates} loss rates...")
for i, r in enumerate(LOSS_RATES):
    durations = []
    for _ in range(N_SAMPLES):
        m = MASK_FN(SPEC_SHAPE, loss_rate=r)
        durations.extend(burst_lengths_ms(m))

    durations = np.array(durations)
    if len(durations) == 0:
        print(f"  rate={r:.2f} : NO BURSTS GENERATED")
        continue

    counts, _ = np.histogram(durations, bins=BIN_EDGES_MS)
    #matrix[i] = 100.0 * counts / counts.sum()              # row sums to 100%
    matrix[i] = counts

    total_bursts_per_rate[i] = len(durations)
    mean_ms_per_rate[i]      = durations.mean()
    median_ms_per_rate[i]    = np.median(durations)

    print(f"  rate={r:.2f} : {len(durations):>6d} bursts, "
          f"mean={mean_ms_per_rate[i]:>4.0f}ms, "
          f"median={median_ms_per_rate[i]:>4.0f}ms")

# ----- plot -----
fig, ax = plt.subplots(figsize=(13, 7))

# Use a perceptually uniform colormap; cap vmax at the 95th-percentile cell value
# so that a few very-frequent bins don't compress everything else.
vmax = max(np.percentile(matrix, 95), matrix.max() * 0.6)
cmap = "YlOrRd"

#im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
# Log-scale color so small bins stay visible
im = ax.imshow(
    matrix,
    aspect="auto",
    cmap="YlOrRd",
    norm=LogNorm(vmin=max(matrix[matrix > 0].min(), 1), vmax=matrix.max()),
)

cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cbar.set_label("# of gaps (log scale)", fontsize=10)

# Tick labels
ax.set_xticks(np.arange(n_bins))
ax.set_xticklabels(bin_labels, rotation=30, ha="right")
ax.set_yticks(np.arange(n_rates))
ax.set_yticklabels([f"{int(r*100)}%" for r in LOSS_RATES])

ax.set_xlabel("Burst duration (ms)", fontsize=11)
ax.set_ylabel("Loss rate", fontsize=11)
# ax.set_title(
#     f"Burst-duration distribution per loss rate\n"
#     f"({N_SAMPLES} masks/rate, mask fn = {MASK_FN.__name__}, "
#     f"clip = {SPEC_SHAPE[1]} frames @ {FPS} fps)",
#     fontsize=12,
# )

# Color bar
# cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
# cbar.set_label("% of bursts (row sums to 100%)", fontsize=10)

# Annotate cells with their percentage value
# if ANNOTATE:
#     for i in range(n_rates):
#         for j in range(n_bins):
#             v = matrix[i, j]
#             if v < 0.5:
#                 txt = ""
#             else:
#                 txt = f"{v:.1f}"
#             color = "white" if v > vmax * 0.55 else "black"
#             ax.text(j, i, txt, ha="center", va="center",
#                     fontsize=8, color=color)

# Right-side annotations: total bursts + mean burst length per row
ax2 = ax.twinx()
ax2.set_yticks(np.arange(n_rates))
ax2.set_yticklabels(
    [f"n={total_bursts_per_rate[i]:>5d}\nmean={mean_ms_per_rate[i]:>3.0f}ms"
     for i in range(n_rates)],
    fontsize=8,
)
ax2.set_ylim(ax.get_ylim())
ax2.tick_params(axis="y", length=0)
ax2.set_ylabel("Bursts collected / mean", fontsize=10, labelpad=10)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
print(f"\nSaved heatmap: {OUT_PNG}")
plt.show()
