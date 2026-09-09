# eval_avail_modes_means.py
"""
For a single model's run directory, walk every test_<rate>/ folder, take the
latest avail_mode_summary_*.csv, and compute the mean of every metric across
all loss rates — separately for each modality (audio / video / av / masked_input).
"""
import os
import csv
import re
from glob import glob
from collections import defaultdict
from statistics import mean

# --------- 1. config ---------
#MODEL_DIR = "./logs/av_wide_masking_mlp_av_only_fusion_5loss_bursty2_pesq_0.01_asr_0.1_enc_loss(grid)"
MODEL_DIR = "./logs/av_wide_masking_mlp_av_only_fusion_5loss_bursty2_plc_a0.05_v0.1_pesq_0.01_asr_0.1(voxceleb2)"
# Modes we want to summarise (in display order).
# "video" is only present at loss_rate=99 in your trainer, but we still average
# over whatever rates contain it.
MODES = ["audio", "video", "av", "masked_input"]

# Non-metric columns in the CSV (everything else is a metric).
NON_METRIC_COLS = {"loss_rate", "mode"}

# Optional preferred column ordering for printing; any extra metrics are appended.
PREFERRED_METRICS = ["pesq", "stoi", "estoi", "plcmos", "psnr", "wer", "cer"]


def find_latest_summaries(model_dir):
    """Return sorted list of (loss_rate, csv_path) — latest CSV per test_<rate>/."""
    pairs = []
    for d in glob(os.path.join(model_dir, "test_*")):
        m = re.match(r"test_(\d+)$", os.path.basename(d))
        if not m:
            continue
        csvs = sorted(glob(os.path.join(d, "avail_mode_summary_*.csv")))
        if csvs:
            pairs.append((int(m.group(1)), csvs[-1]))    # latest by filename (timestamp)
    pairs.sort()
    return pairs


def collect(model_dir):
    """
    Returns:
        rates_per_mode : {mode: [rates contributing]}
        values         : {mode: {metric: [values across rates]}}
        all_metrics    : sorted list of all metric column names found
    """
    values = {m: defaultdict(list) for m in MODES}
    rates_per_mode = {m: [] for m in MODES}
    metric_set = set()

    for rate, csv_path in find_latest_summaries(model_dir):
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            metric_set.update(c for c in (reader.fieldnames or []) if c not in NON_METRIC_COLS)
            for row in reader:
                mode = row.get("mode", "")
                if mode not in values:
                    continue
                rates_per_mode[mode].append(rate)
                for k, v in row.items():
                    if k in NON_METRIC_COLS or v in (None, ""):
                        continue
                    try:
                        values[mode][k].append(float(v))
                    except ValueError:
                        pass

    # ordered metrics: preferred first, then any extras alphabetically
    extras = sorted(metric_set - set(PREFERRED_METRICS))
    all_metrics = [m for m in PREFERRED_METRICS if m in metric_set] + extras
    return rates_per_mode, values, all_metrics


def safe_mean(xs):
    return mean(xs) if xs else float("nan")

# --------- 2. run ---------
if not os.path.isdir(MODEL_DIR):
    raise SystemExit(f"Missing directory: {MODEL_DIR}")

rates_per_mode, values, METRICS = collect(MODEL_DIR)

print(f"Model: {MODEL_DIR}")
for m in MODES:
    rs = sorted(set(rates_per_mode[m]))
    print(f"  mode={m:<13} rates={rs}  ({len(rs)} loss rates)")

# --------- 3. table ---------
col_w = 10
header = f"{'mode':<14}{'n_rates':>9}" + "".join(f"{m.upper():>{col_w}}" for m in METRICS)
print("\n=== Mean of each metric over all loss rates, per modality ===")
print(header)
print("-" * len(header))

means = {}
for mode in MODES:
    vals = values[mode]
    n_rates = len(set(rates_per_mode[mode]))
    means[mode] = {met: safe_mean(vals.get(met, [])) for met in METRICS}
    line = f"{mode:<14}{n_rates:>9}" + "".join(f"{means[mode][met]:>{col_w}.4f}" for met in METRICS)
    print(line)

# --------- 4. CSV dump ---------
out_csv = os.path.join(MODEL_DIR, "avail_mode_means_over_rates.csv")
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["mode", "n_rates"] + METRICS)
    for mode in MODES:
        n_rates = len(set(rates_per_mode[mode]))
        w.writerow([mode, n_rates] + [f"{means[mode][met]:.6f}" for met in METRICS])

print(f"\nSaved -> {out_csv}")
