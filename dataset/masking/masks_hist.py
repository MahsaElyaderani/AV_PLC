import numpy as np, matplotlib.pyplot as plt
from shared.masking import generate_ge_mask_bursty

LOSS_RATES = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
SPEC = (80, 300); FPS = 100; N = 5000

def bursts_ms(m):
    t = 1 - m[0].astype(np.int8); out, c = [], 0
    for v in t:
        if v: c += 1
        elif c: out.append(c); c = 0
    if c: out.append(c)
    return [x*1000/FPS for x in out]

# all_d = []
# for r in LOSS_RATES:
#     for _ in range(N):
#         all_d += bursts_ms(generate_ge_mask_bursty(SPEC, r))
# d = np.array(all_d)
#
# fig, ax = plt.subplots(1, 2, figsize=(12,4))
# ax[0].hist(d, bins=np.arange(0, 2000, 50), edgecolor='k')
# ax[0].set_xlabel("Gap duration (ms)"); ax[0].set_ylabel("Count")
# ax[0].axvline(d.mean(), color='r', ls='--', label=f"mean={d.mean():.0f} ms")
# ax[0].axvline(np.median(d), color='g', ls='--', label=f"median={np.median(d):.0f} ms")
# ax[0].legend(); ax[0].set_title("Gap-duration PDF (all loss rates pooled)")
#
# xs = np.sort(d); ys = np.arange(1, len(xs)+1)/len(xs)
# ax[1].plot(xs, ys); ax[1].set_xlabel("Gap duration (ms)")
# ax[1].set_ylabel("CDF"); ax[1].set_xscale("log")
# for p in [50, 90, 95, 99]:
#     v = np.percentile(d, p); ax[1].axhline(p/100, color='gray', alpha=0.3)
#     ax[1].text(v, p/100, f" p{p}={v:.0f}", fontsize=8, va='bottom')
# ax[1].set_title("Gap-duration CDF")
# plt.tight_layout(); plt.savefig("gap_dist.png", dpi=140)

# data = []
# for r in LOSS_RATES:
#     ds = []
#     for _ in range(N):
#         ds += bursts_ms(generate_ge_mask_bursty(SPEC, r))
#     data.append(ds)
#
# fig, ax = plt.subplots(figsize=(9,4))
# parts = ax.violinplot(data, positions=range(len(LOSS_RATES)),
#                       showmedians=True, widths=0.8)
# ax.set_xticks(range(len(LOSS_RATES)))
# ax.set_xticklabels([f"{int(r*100)}%" for r in LOSS_RATES])
# ax.set_xlabel("Target loss rate"); ax.set_ylabel("Gap duration (ms)")
# ax.set_yscale("log"); ax.axhline(100,  color='gray', ls=':', alpha=.5)
# ax.axhline(1000, color='gray', ls=':', alpha=.5)
# ax.set_title("Gap-duration distribution per loss rate")
# plt.tight_layout(); plt.savefig("gap_violin.png", dpi=140)

fig, ax = plt.subplots(figsize=(8,4))
edges = np.array([0,100,200,300,500,800,1200,1600,3000])
centers = 0.5*(edges[:-1]+edges[1:])
for r in [0.2, 0.5, 0.9]:
    ds = []
    for _ in range(N):
        ds += bursts_ms(generate_ge_mask_bursty(SPEC, r))
    ds = np.array(ds)
    # weight each gap by its length -> "frame-time spent in gaps of this size"
    frame_time = np.zeros(len(edges)-1)
    for k in range(len(edges)-1):
        sel = (ds>=edges[k])&(ds<edges[k+1])
        frame_time[k] = ds[sel].sum()
    frame_time /= frame_time.sum()
    ax.plot(centers, frame_time*100, marker='o', label=f"r={r:.1f}")
ax.set_xlabel("Gap duration (ms)")
ax.set_ylabel("% of lost frames coming from gaps of this length")
ax.set_xscale("log"); ax.legend(); ax.grid(alpha=.3)
ax.set_title("Where does the loss come from?")
plt.tight_layout(); plt.savefig("gap_time_share.png", dpi=140)

