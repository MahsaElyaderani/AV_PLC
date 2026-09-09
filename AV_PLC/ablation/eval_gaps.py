# eval_gaps.py
import os
import csv
import re
from glob import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, MultipleLocator

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
})
plt.rcParams.update({'font.size': 15})

def _dense_grid(ax, x_major=10, y_minor=5, x_minor=5):
    """
    Denser ticks + minor gridlines so values can be read from the plot.

    x_major : major x-tick spacing in 'loss rate %' units (set None to auto)
    y_minor : number of minor intervals per major y-tick
    x_minor : number of minor intervals per major x-tick
    """
    # X axis: loss rate is in known integer steps — pin majors every `x_major`
    if x_major is not None:
        ax.xaxis.set_major_locator(MultipleLocator(x_major))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    ax.xaxis.set_minor_locator(AutoMinorLocator(x_minor))

    # Y axis: let matplotlib pick majors, then subdivide
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.yaxis.set_minor_locator(AutoMinorLocator(y_minor))

    ax.grid(which="major", alpha=0.45, linewidth=0.8)
    ax.grid(which="minor", alpha=0.20, linewidth=0.5, linestyle=":")
    ax.tick_params(which="both", direction="in", length=4)
    ax.tick_params(which="minor", length=2)

# --------- 1. config ---------
DATASETS = ["grid", "lrs2", "voxceleb2"]
DATASET_LABELS = {"grid": "GRID", "lrs2": "LRS2", "voxceleb2": "VoxCeleb2"}

def a_dir(ds):  return f"./logs/audio_bursty_plc({ds})"
def av_dir(ds): return f"./logs/av_wide_masking_mlp_av_only_fusion_5loss_bursty2_plc_a0.05_v0.1_pesq_0.01_asr_0.1({ds})"

METRICS = ["pesq", "stoi", "estoi", "plcmos", "psnr", "wer", "cer"]
HIGHER_BETTER = {"pesq": True, "stoi": True, "estoi": True,
                 "plcmos": True, "psnr": True, "wer": False, "cer": False}


def collect(model_dir):
    rec    = defaultdict(list)
    masked = defaultdict(list)
    pairs = []
    for d in glob(os.path.join(model_dir, "test_*")):
        m = re.match(r"test_(\d+)$", os.path.basename(d))
        if not m:
            continue
        csvs = (sorted(glob(os.path.join(d, "test_metrics_*.csv")))
                or sorted(glob(os.path.join(d, "avail_mode_summary_*.csv"))))
        if csvs:
            pairs.append((int(m.group(1)), csvs[-1]))
    pairs.sort()
    rates = [r for r, _ in pairs]
    for _, csv_path in pairs:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get("head"):
                    continue
                target = rec if row["head"] in ("rec", "fused") else None
                if target is None:
                    continue
                for met in METRICS:
                    val = row.get(met, "")
                    try:
                        target[met].append(float(val))
                    except ValueError:
                        target[met].append(float("nan"))
    return rates, rec, masked


def collect_mode_summary(model_dir, mode):
    av_res = defaultdict(list)
    pairs = []
    for d in glob(os.path.join(model_dir, "test_*")):
        m = re.match(r"test_(\d+)$", os.path.basename(d))
        if not m:
            continue
        csvs = sorted(glob(os.path.join(d, "avail_mode_summary_*.csv")))
        if csvs:
            pairs.append((int(m.group(1)), csvs[-1]))
    pairs.sort()
    rates = [r for r, _ in pairs]
    for _, csv_path in pairs:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            row_found = False
            for row in reader:
                if row.get("mode") == mode:
                    for met in METRICS:
                        val = row.get(met, "")
                        try:
                            av_res[met].append(float(val))
                        except ValueError:
                            av_res[met].append(float("nan"))
                    row_found = True
                    break
            if not row_found:
                for met in METRICS:
                    av_res[met].append(float("nan"))
    return rates, av_res


# --------- 2. read all datasets ---------
data = {}
for ds in DATASETS:
    rates_a, rec_a, _       = collect(a_dir(ds))
    rates_av, rec_av        = collect_mode_summary(av_dir(ds), "av")
    rates_masked, masked_av = collect_mode_summary(av_dir(ds), "masked_input")
    data[ds] = dict(
        rates_a=rates_a, rec_a=rec_a,
        rates_av=rates_av, rec_av=rec_av,
        rates_masked=rates_masked, masked_av=masked_av,
    )
    print(f"\n=== {ds} ===")
    print("Input    rates :", rates_masked)
    print("audio_plc rates:", rates_a)
    print("av_plc    rates:", rates_av)
    print(f"{'metric':<8}{'input':>10}{'A-PLC':>10}{'AV-PLC':>10}{'AV-A':>10}")
    for met in METRICS:
        if not masked_av[met] or not rec_a[met] or not rec_av[met]:
            continue
        inp = sum(masked_av[met]) / len(masked_av[met])
        a   = sum(rec_a[met])    / len(rec_a[met])
        av  = sum(rec_av[met])   / len(rec_av[met])
        print(f"{met:<8}{inp:>10.4f}{a:>10.4f}{av:>10.4f}{av - a:>+10.4f}")

os.makedirs("comparison_figures", exist_ok=True)


def _ylabel(met):
    return met.upper() + (r"  ($\uparrow$)" if HIGHER_BETTER[met] else r"  ($\downarrow$)")


# --------- 3. per-dataset, per-metric individual plots ---------
for ds in DATASETS:
    d = data[ds]
    for met in METRICS:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(d["rates_masked"], d["masked_av"][met], "o--",
                label="Input",       color="gray")
        ax.plot(d["rates_a"],      d["rec_a"][met],     "o-",
                label="Audio-only",  color="tab:blue")
        ax.plot(d["rates_av"],     d["rec_av"][met],    "s-",
                label="Audio-Video", color="tab:red")

        ax.set_xlabel(r"Packet loss rate (%)")
        ax.set_ylabel(_ylabel(met))
        _dense_grid(ax) #ax.grid(True, alpha=0.3)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                  ncol=3, frameon=False, borderaxespad=0.0)

        fig.tight_layout()
        fig.savefig(
            f"comparison_figures/{met}_vs_loss_rate_bursty_{ds}.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)


# --------- 4. per-metric 3-row combined figure (rows = datasets) ---------
n_rows = len(DATASETS)
for met in METRICS:
    fig, axes = plt.subplots(n_rows, 1,
                             figsize=(6, 3.6 * n_rows),
                             squeeze=False)
    handles, labels = None, None
    for i, ds in enumerate(DATASETS):
        d = data[ds]
        ax = axes[i][0]
        l_in,  = ax.plot(d["rates_masked"], d["masked_av"][met], "o--",
                         label="Input",       color="gray")
        l_a,   = ax.plot(d["rates_a"],      d["rec_a"][met],     "o-",
                         label="Audio-only",  color="tab:blue")
        l_av,  = ax.plot(d["rates_av"],     d["rec_av"][met],    "s-",
                         label="Audio-Video", color="tab:red")

        _dense_grid(ax)  #ax.grid(True, alpha=0.3)
        ax.set_ylabel(_ylabel(met))
        if i == n_rows - 1:
            ax.set_xlabel(r"Packet loss rate (%)")

        # dataset row label on the left
        ax.annotate(DATASET_LABELS[ds],
                    xy=(-0.22, 0.5), xycoords="axes fraction",
                    ha="center", va="center", rotation=90,
                    fontsize=16, fontweight="bold")

        if handles is None:
            handles = [l_in, l_a, l_av]
            labels  = [h.get_label() for h in handles]

    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False)
    fig.tight_layout(rect=(0.04, 0.04, 1, 1))
    fig.savefig(
        f"comparison_figures/{met}_vs_loss_rate_bursty_all_datasets.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)


# --------- 5. combined PESQ + STOI figure: rows = datasets, cols = metrics ---------
combo_metrics = ["pesq", "stoi"]
n_cols = len(combo_metrics)
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(5.5 * n_cols, 3.8 * n_rows),
                         squeeze=False)

handles, labels = None, None
for i, ds in enumerate(DATASETS):
    d = data[ds]
    for j, met in enumerate(combo_metrics):
        ax = axes[i][j]
        l_in,  = ax.plot(d["rates_masked"], d["masked_av"][met], "o--",
                         label="Input",       color="gray")
        l_a,   = ax.plot(d["rates_a"],      d["rec_a"][met],     "o-",
                         label="Audio-only",  color="tab:blue")
        l_av,  = ax.plot(d["rates_av"],     d["rec_av"][met],    "s-",
                         label="Audio-Video", color="tab:red")

        _dense_grid(ax)  #ax.grid(True, alpha=0.3)
        if i == n_rows - 1:
            ax.set_xlabel(r"Packet loss rate (%)")
        ax.set_ylabel(_ylabel(met))

        if j == 0:
            ax.annotate(DATASET_LABELS[ds],
                        xy=(-0.28, 0.5), xycoords="axes fraction",
                        ha="center", va="center", rotation=90,
                        fontsize=16, fontweight="bold")

        if handles is None:
            handles = [l_in, l_a, l_av]
            labels  = [h.get_label() for h in handles]

fig.legend(handles, labels, loc="lower center",
           bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False)
fig.tight_layout(rect=(0.02, 0.04, 1, 1))
fig.savefig(
    "comparison_figures/pesq_stoi_vs_loss_rate_bursty_all_datasets.png",
    dpi=300, bbox_inches="tight",
)
plt.close(fig)

print("\nSaved figures to ./comparison_figures/")
