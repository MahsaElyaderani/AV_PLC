# ablation_means.py
"""
Compute the mean of every metric across all packet-loss rates for a set of models,
and produce per-loss-rate bar plots so you can inspect individual rates too.
"""
import os
import csv
import re
from glob import glob
from collections import defaultdict
from statistics import mean

import numpy as np
import matplotlib.pyplot as plt

# --------- 1. models to compare ---------
# (display name, path)
MODELS = [
    ("Full (PMSQE + ASR + enc-loss)",
     "./logs/av_wide_masking_mlp_av_only_fusion_5loss_bursty2_plc_a0.05_v0.1_pesq_0.01_asr_0.1(lrs2)"),
    ("No PMSQE",
     "./logs/av_wide_masking_mlp_av_only_fusion_5loss_bursty_no_pmsqe_asr_0.1_enc_loss(lrs2)"),
    ("No ASR",
     "./logs/av_wide_masking_mlp_av_only_fusion_5loss_bursty_no_asr_pesq_0.01_enc_loss(lrs2)"),
    #("Fusion only (no enc-loss)",
    ("L1-Only", "./logs/av_wide_masking_mlp_av_only_fusion_5loss_bursty_l1_only_enc_loss(grid)")
]

METRICS = ["pesq", "stoi", "estoi", "plcmos", "psnr", "wer", "cer"]

# rows in the CSV that should be treated as the model's reconstruction
REC_HEADS    = {"fused"}   # , "av"
MASKED_HEADS = {"masked_input"}

PLOTS_DIR = "ablation_plots"


def collect(model_dir):
    """
    Walk every test_<rate>/ folder, take the latest test_metrics_*.csv inside,
    and accumulate metric values for the reconstructed (rec/fused) and masked rows.

    Returns:
        rates           : sorted list[int] of loss rates found
        rec             : {metric: [values]} pooled across all rates
        masked          : {metric: [values]} pooled across all rates
        rec_per_rate    : {rate: {metric: [values]}}
        masked_per_rate : {rate: {metric: [values]}}
    """
    rec    = defaultdict(list)
    masked = defaultdict(list)
    rec_per_rate    = defaultdict(lambda: defaultdict(list))
    masked_per_rate = defaultdict(lambda: defaultdict(list))

    pairs = []
    for d in glob(os.path.join(model_dir, "test_*")):
        m = re.match(r"test_(\d+)$", os.path.basename(d))
        if not m:
            continue
        csvs = sorted(glob(os.path.join(d, "test_metrics_*.csv")))
        if csvs:
            pairs.append((int(m.group(1)), csvs[-1]))    # take latest CSV
    pairs.sort()
    rates = [r for r, _ in pairs]

    for rate, csv_path in pairs:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                head = row.get("head", "")
                if head in REC_HEADS:
                    target, target_pr = rec, rec_per_rate[rate]
                elif head in MASKED_HEADS:
                    target, target_pr = masked, masked_per_rate[rate]
                else:
                    continue
                for met in METRICS:
                    try:
                        v = float(row.get(met, ""))
                    except ValueError:
                        continue    # skip empty / non-numeric cells
                    target[met].append(v)
                    target_pr[met].append(v)
    return rates, rec, masked, rec_per_rate, masked_per_rate


def safe_mean(xs):
    return mean(xs) if xs else float("nan")


def has_files(model_dir, pattern):
    return any(glob(os.path.join(model_dir, "test_*", pattern)))


# --------- 2. gather results ---------
# list of (name, rates, rec_mean, masked_mean, rec_per_rate_mean, masked_per_rate_mean)
results = []
for name, path in MODELS:
    if not os.path.isdir(path):
        print(f"[WARN] missing directory: {path}")
        results.append((name, [],
                        {m: float('nan') for m in METRICS},
                        {m: float('nan') for m in METRICS},
                        {}, {}))
        continue

    rates, rec, masked, rec_per_rate, masked_per_rate = collect(path)

    rec_mean    = {m: safe_mean(rec[m])    for m in METRICS}
    masked_mean = {m: safe_mean(masked[m]) for m in METRICS}

    rec_pr_mean = {
        r: {m: safe_mean(rec_per_rate[r][m]) for m in METRICS}
        for r in rates
    }
    masked_pr_mean = {
        r: {m: safe_mean(masked_per_rate[r][m]) for m in METRICS}
        for r in rates
    }
    results.append((name, rates, rec_mean, masked_mean, rec_pr_mean, masked_pr_mean))
    print(f"{name}: rates={rates}  ({len(rates)} loss rates)")

# --------- 3. pretty table (rows = models, cols = metrics) ---------
col_w = 10
header = f"{'model':<40}" + "".join(f"{m.upper():>{col_w}}" for m in METRICS)
print("\n=== Mean over all loss rates (reconstructed output) ===")
print(header)
print("-" * len(header))
for name, _, rec_mean, _, _, _ in results:
    line = f"{name:<40}" + "".join(f"{rec_mean[m]:>{col_w}.4f}" for m in METRICS)
    print(line)

# masked-input baseline
print("\n=== Masked-input baseline (mean over loss rates, first model) ===")
_, _, _, masked_mean0, _, _ = results[0]
print(f"{'masked_input':<40}" +
      "".join(f"{masked_mean0[m]:>{col_w}.4f}" for m in METRICS))

# --------- 3b. per-loss-rate tables (one table per model) ---------
print("\n=== Per-loss-rate values (reconstructed output) ===")
for name, rates, _, _, rec_pr_mean, _ in results:
    if not rates:
        continue
    print(f"\n-- {name} --")
    sub_header = f"{'rate':<8}" + "".join(f"{m.upper():>{col_w}}" for m in METRICS)
    print(sub_header)
    print("-" * len(sub_header))
    for r in rates:
        row_vals = rec_pr_mean[r]
        line = f"{r:<8}" + "".join(f"{row_vals[m]:>{col_w}.4f}" for m in METRICS)
        print(line)

# --------- 4. CSV dumps ---------
# 4a. means over all rates
out_csv = "ablation_means.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "n_rates"] + METRICS)
    for name, rates, rec_mean, _, _, _ in results:
        w.writerow([name, len(rates)] + [f"{rec_mean[m]:.6f}" for m in METRICS])
print(f"\nSaved -> {out_csv}")

# 4b. per-rate values for every model
per_rate_csv = "ablation_per_rate.csv"
with open(per_rate_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "loss_rate"] + METRICS)
    for name, rates, _, _, rec_pr_mean, _ in results:
        for r in rates:
            row_vals = rec_pr_mean[r]
            w.writerow([name, r] + [f"{row_vals[m]:.6f}" for m in METRICS])
print(f"Saved -> {per_rate_csv}")

# --------- 5. Bar plots: per-loss-rate, per-model ---------
all_rates = sorted({r for _, rates, _, _, _, _ in results for r in rates})

if all_rates:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    n_models = len(results)
    x = np.arange(len(all_rates))
    width = 0.8 / max(n_models, 1)

    # ---- one figure per metric ----
    for met in METRICS:
        fig, ax = plt.subplots(
            figsize=(max(6, 1.1 * len(all_rates) * n_models / 2), 4.2)
        )
        for i, (name, _, _, _, rec_pr_mean, _) in enumerate(results):
            vals = [rec_pr_mean.get(r, {}).get(met, float("nan")) for r in all_rates]
            offsets = x + (i - (n_models - 1) / 2) * width
            bars = ax.bar(offsets, vals, width, label=name)
            for b, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                            f"{v:.2f}", ha="center", va="bottom",
                            fontsize=7, rotation=0)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{r}%" for r in all_rates])
        ax.set_xlabel("Packet-loss rate")
        ax.set_ylabel(met.upper())
        ax.set_title(f"{met.upper()} per loss rate (reconstructed output)")
        ax.legend(fontsize=8, loc="best")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        fig.tight_layout()
        out_path = os.path.join(PLOTS_DIR, f"bar_{met}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")

    # ---- single grid figure with all metrics ----
    n = len(METRICS)
    cols = 3
    rows_ = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_, cols, figsize=(5 * cols, 3.2 * rows_))
    axes = np.array(axes).reshape(-1)
    for ax, met in zip(axes, METRICS):
        for i, (name, _, _, _, rec_pr_mean, _) in enumerate(results):
            vals = [rec_pr_mean.get(r, {}).get(met, float("nan")) for r in all_rates]
            offsets = x + (i - (n_models - 1) / 2) * width
            ax.bar(offsets, vals, width, label=name)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r}%" for r in all_rates], fontsize=8)
        ax.set_title(met.upper())
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    for ax in axes[len(METRICS):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(results),
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Per-loss-rate metrics by model (reconstructed)", y=1.02)
    fig.tight_layout()
    grid_path = os.path.join(PLOTS_DIR, "bar_all_metrics.png")
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {grid_path}")
else:
    print("No loss rates found - skipping bar plots.")
