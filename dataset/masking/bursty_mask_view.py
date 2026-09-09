"""
Visualize 100 masks from generate_ge_mask_bursty across loss rates 0.2-0.9.
Confirms (a) all loss rates appear, (b) burst structure looks reasonable,
(c) realized loss rate matches requested loss rate.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from shared.masking import generate_ge_mask_bursty

# ----- config -----
SPEC_SHAPE   = (80, 300)            # (F, T) — 3 s clip @ 100 fps
N_MASKS      = 100
LOSS_RATES   = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SEED         = 0                    # reproducibility

np.random.seed(SEED)

# ----- generate masks -----
# Sample loss rate uniformly from {0.2, ..., 0.9} for each of the 100 masks
# (matches what the training loader does via online_loss_bounds=(0.2, 0.9)).
requested_rates = np.random.uniform(0.3, 0.9, size=N_MASKS)
masks           = []
realized_rates  = []

for r in requested_rates:
    m = generate_ge_mask_bursty(SPEC_SHAPE, loss_rate=r)
    masks.append(m)
    realized_rates.append(float(1.0 - m.mean()))   # 0 = lost, so loss = 1 - mean(mask)

masks          = np.stack(masks)                   # [N, F, T]
realized_rates = np.array(realized_rates)

# ----- coverage check: do all 0.2-0.9 buckets appear? -----
bins      = np.arange(0.20, 1.01, 0.10)            # 0.2, 0.3, ..., 1.0
hist, _   = np.histogram(realized_rates, bins=bins)
bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 1)]

print("=" * 60)
print(f"Coverage check: realized loss rates across {N_MASKS} masks")
print("=" * 60)
print(f"{'Bin':>10} | {'Count':>6} | {'%':>5}")
print("-" * 32)
for lbl, cnt in zip(bin_labels, hist):
    pct = 100 * cnt / N_MASKS
    bar = "█" * int(pct / 2)
    print(f"{lbl:>10} | {cnt:>6d} | {pct:>4.1f}% {bar}")

print(f"\nRequested rates : min={requested_rates.min():.2f}, "
      f"max={requested_rates.max():.2f}, mean={requested_rates.mean():.2f}")
print(f"Realized rates  : min={realized_rates.min():.2f}, "
      f"max={realized_rates.max():.2f}, mean={realized_rates.mean():.2f}")
print(f"|requested - realized|: mean={np.mean(np.abs(requested_rates-realized_rates)):.3f}, "
      f"max={np.max(np.abs(requested_rates-realized_rates)):.3f}")

# ----- Figure 1: 10x10 grid of all 100 masks (sorted by realized rate) -----
order = np.argsort(realized_rates)                 # ascending: easiest first

fig1 = plt.figure(figsize=(20, 12))
fig1.suptitle(
    f"100 GE bursty masks (sorted by realized loss rate)\n"
    f"each tile: shape {SPEC_SHAPE} — black=lost, white=kept",
    fontsize=12,
)
gs = GridSpec(10, 10, figure=fig1, wspace=0.05, hspace=0.35)
for i, idx in enumerate(order):
    ax = fig1.add_subplot(gs[i // 10, i % 10])
    ax.imshow(masks[idx], aspect='auto', cmap='gray', vmin=0, vmax=1,
              interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"r={realized_rates[idx]:.2f}", fontsize=7)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("masks_grid_100.png", dpi=120, bbox_inches='tight')
print("\nSaved: masks_grid_100.png")

# ----- Figure 2: histogram of realized loss rates + coverage bar -----
fig2, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].hist(realized_rates, bins=20, range=(0.0, 1.0),
             edgecolor='black', alpha=0.7, label='realized')
axes[0].hist(requested_rates, bins=20, range=(0.0, 1.0),
             edgecolor='red', fill=False, linewidth=1.5, label='requested')
axes[0].set_xlabel("Loss rate")
axes[0].set_ylabel("Count")
axes[0].set_title("Distribution of loss rates across 100 masks")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].bar(bin_labels, hist, edgecolor='black', alpha=0.7)
axes[1].set_xlabel("Realized loss-rate bin")
axes[1].set_ylabel("Count")
axes[1].set_title("Coverage of 0.2-0.9 range (10% bins)")
axes[1].axhline(N_MASKS / 8, color='red', linestyle='--',
                label=f'uniform expectation ({N_MASKS // 8}/bin)')
axes[1].legend()
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("masks_loss_rate_distribution.png", dpi=120, bbox_inches='tight')
print("Saved: masks_loss_rate_distribution.png")

# ----- Figure 3: 8 representative masks, one per loss-rate bucket (0.2-0.9) -----
# Picks the mask whose realized rate is closest to each target bucket.
fig3, axes = plt.subplots(8, 1, figsize=(14, 12))
fig3.suptitle("One representative mask per loss-rate bucket (0.2 - 0.9)", fontsize=12)
for ax, target_r in zip(axes, LOSS_RATES):
    pick = int(np.argmin(np.abs(realized_rates - target_r)))
    ax.imshow(masks[pick], aspect='auto', cmap='gray', vmin=0, vmax=1,
              interpolation='nearest')
    ax.set_title(f"target r={target_r:.1f}, realized r={realized_rates[pick]:.3f}",
                 fontsize=10, loc='left')
    ax.set_xlabel("time (frames, 100 fps)" if ax is axes[-1] else "")
    ax.set_ylabel("mel bin")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("masks_one_per_bucket.png", dpi=120, bbox_inches='tight')
print("Saved: masks_one_per_bucket.png")

plt.show()
