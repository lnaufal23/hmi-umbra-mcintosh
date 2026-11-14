#!/usr/bin/env python3
"""Threshold calibration figure: pixel-intensity histograms of the U-Net v4 mask classes + crossing points per year."""
import json, shutil
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hmi_umbra import paths

D = Path(paths.get("analysis_v4mask"))
FIG = Path(paths.get("paper_figs")); DESK = Path(paths.get("check_plots"))
z = np.load(D / "hist_class_ratio_v4mask.npz"); c = z["centers"]
years = sorted({int(k.split("_")[0]) for k in z.files if k != "centers"})
tot = {k: sum(z[f"{y}_{k}"] for y in years) for k in ("umb", "pen", "pho")}
J = json.load(open(D / "threshold_histcross_v4mask.json"))
T_up, T_pp = J["T_umbra_penumbra"], J["T_penumbra_photosphere"]
yt = pd.read_csv(D / "threshold_histcross_by_year_v4mask.csv")
ok = yt["n_umb_pix"] > 1e5

fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
for k, col, lab in [("umb", "#c0392b", "umbra"), ("pen", "#e1a100", "penumbra"), ("pho", "#4a6fa5", "photosphere")]:
    a.semilogy(c, tot[k] / tot[k].sum() / (c[1] - c[0]), color=col, lw=1.6, label=lab)
for T, nm in [(T_up, "umbra–penumbra"), (T_pp, "penumbra–photosphere")]:
    a.axvline(T, color="k", ls="--", lw=1)
    a.text(T + 0.008, a.get_ylim()[1] * 0.3 if False else 30, f"{nm}\n{T:.3f}", fontsize=8.5, va="top")
a.set_xlim(0.15, 1.15); a.set_ylim(1e-3, 60)
a.set_xlabel(r"$I/I_{qs}$"); a.set_ylabel("normalized pixel density")
a.set_title(f"(a) Pixel intensity by U-Net class ({J['n_days_sampled']} days, |θ| ≤ 60°)", fontsize=10)
a.legend(fontsize=9, loc="upper left"); a.grid(alpha=0.3)

for col_, nm, mk in [("T_up", "umbra–penumbra", "o"), ("T_pp", "penumbra–photosphere", "s")]:
    b.plot(yt["year"][ok], yt[col_][ok], mk + "-", color="k", ms=5, lw=1, label=nm)
    b.plot(yt["year"][~ok], yt[col_][~ok], mk, color="0.6", ms=5)
    m, s_ = yt[col_][ok].mean(), yt[col_][ok].std()
    b.axhspan(m - s_, m + s_, color="0.85", zorder=0)
b.set_xlabel("Year"); b.set_ylabel(r"threshold $I/I_{qs}$")
b.set_title("(b) Crossing points per year (grey: < 10⁵ umbral pixels)", fontsize=10)
b.legend(fontsize=8.5, loc="center right"); b.grid(alpha=0.3)
fig.tight_layout()
out = FIG / "fig_threshold_histcross_v4mask_en.png"
fig.savefig(out, dpi=200); DESK.mkdir(parents=True, exist_ok=True); shutil.copy(out, DESK / out.name)
print(f"[WRITE] {out}\n[COPY] {DESK / out.name}")
print(f"T_up {T_up:.3f} | T_pp {T_pp:.3f} | per year (n={int(ok.sum())}): T_up {yt['T_up'][ok].mean():.3f}±{yt['T_up'][ok].std():.3f}, "
      f"T_pp {yt['T_pp'][ok].mean():.3f}±{yt['T_pp'][ok].std():.3f}")
