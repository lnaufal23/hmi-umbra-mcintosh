#!/usr/bin/env python3
"""Paper figures 3-7 from the v4-mask CSVs: I_qs time series, threshold ratios, umbra/penumbra DN, daily spot count vs SILSO and their difference;
prints the class pixel percentiles used in Sec. 3.3."""
import shutil
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

from hmi_umbra import paths

FIGDIR = Path(paths.get("paper_figs")); FIGDIR.mkdir(parents=True, exist_ok=True)
DESK = Path(paths.get("check_plots")); DESK.mkdir(parents=True, exist_ok=True)
V4 = Path(paths.get("analysis_v4mask"))
CSV_THRESH, CSV_RATIO, CSV_SPOT = V4 / "threshold_timeseries_v4mask.csv", V4 / "ratio_iqs_timeseries_v4mask.csv", V4 / "spot_count_per_day_v4mask.csv"
CSV_SPOT_OLD = Path(paths.get("spot_count_legacy"))
CSV_SILSO = Path(paths.get("silso_daily"))
CYCLE = [("2014-04-01", "SC24 max"), ("2019-12-01", "SC25 start"), ("2023-09-01", "SC25 max (est.)")]
T_UMBRA, T_PENUMBRA = 0.503, 0.857
DATE_START, DATE_END, MA = "2012-01-01", "2026-04-30", 30
saved = []


def add_cycle(ax, ytext):
    for dt, lbl in CYCLE:
        ax.axvline(pd.Timestamp(dt), color="gray", ls=":", alpha=0.7, lw=1.5)
        ax.text(pd.Timestamp(dt), ytext, lbl, fontsize=8, color="gray", rotation=90, va="top")


def years(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.get_xticklabels(), rotation=45)


def save(fig, name):
    p = FIGDIR / name; fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); saved.append(p)


# ---------- Figure 7: umbra/penumbra DN ----------
df = pd.read_csv(CSV_THRESH)
df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
for c in ["umb_min", "umb_max", "pen_min", "pen_max"]:
    df[f"{c}_ma"] = df[c].rolling(180, center=True, min_periods=10).mean()
fig, ax = plt.subplots(figsize=(16, 6.5))
fig.suptitle("Raw DN intensity (PSF+LD): umbra vs penumbra | 180-day moving average | 2012-2026", fontsize=13, fontweight="bold")
for c, col in [("umb_min", "darkred"), ("umb_max", "red"), ("pen_min", "darkorange"), ("pen_max", "gold")]:
    ax.scatter(df["date"], df[c], s=2, alpha=0.15, color=col)
for c, col, lab, ls in [("umb_min_ma", "darkred", "Umbra min", "-"), ("umb_max_ma", "red", "Umbra max", "--"),
                        ("pen_min_ma", "darkorange", "Penumbra min", "-"), ("pen_max_ma", "gold", "Penumbra max", "--")]:
    ax.plot(df["date"], df[c], color=col, lw=2, ls=ls, label=f"{lab} (MA180d)")
add_cycle(ax, df["pen_max"].quantile(0.95))
ax.set_ylabel("Intensity (DN/s)"); ax.set_xlabel("Year"); ax.legend(fontsize=8, ncol=2, loc="upper right"); ax.grid(alpha=0.3); years(ax)
plt.tight_layout(); save(fig, "fig_intensity_dn_umbra_penumbra_v4mask_en.png")

# ---------- Figure 3: I_qs ----------
fig, ax = plt.subplots(figsize=(15, 5))
ax.set_title("Quiet-Sun intensity I_qs per day | 27-day moving average | 2012-2026", fontsize=13, fontweight="bold")
ax.scatter(df["date"], df["I_qs"], s=2, alpha=0.2, color="steelblue")
ax.plot(df["date"], df["I_qs"].rolling(27, center=True, min_periods=10).mean(), color="steelblue", lw=2, label="I_qs (MA27d)")
add_cycle(ax, df["I_qs"].quantile(0.95))
ax.set_ylabel("I_qs (DN/s)"); ax.set_xlabel("Year"); ax.legend(fontsize=10); ax.grid(alpha=0.3); years(ax)
plt.tight_layout(); save(fig, "fig_iqs_timeseries_v4mask_en.png")

# ---------- Figure 4: threshold ratios ----------
dr = pd.read_csv(CSV_RATIO); dr["date"] = pd.to_datetime(dr["date"], errors="coerce"); dr = dr.dropna(subset=["date"]).sort_values("date")
fig, ax = plt.subplots(figsize=(16, 6.5))
fig.suptitle("Threshold ratios relative to I_qs (PSF+LD) | MA180d | 2012-2026", fontsize=13, fontweight="bold")
for c, col in [("umb_min_ratio", "darkred"), ("umb_max_ratio", "red"), ("pen_min_ratio", "darkorange"), ("pen_max_ratio", "gold")]:
    ax.scatter(dr["date"], dr[c], s=2, alpha=0.12, color=col)
for c, col, lab, ls in [("umb_min_ratio_ma", "darkred", "Umbra min / I_qs", "-"), ("umb_max_ratio_ma", "red", "Umbra max / I_qs", "--"),
                        ("pen_min_ratio_ma", "darkorange", "Penumbra min / I_qs", "-"), ("pen_max_ratio_ma", "gold", "Penumbra max / I_qs", "--")]:
    ax.plot(dr["date"], dr[c], color=col, lw=2, ls=ls, label=f"{lab} (MA180d)")
ax.axhline(T_UMBRA, color="blue", ls=":", lw=2, alpha=0.8, label=f"Umbra-penumbra threshold = {T_UMBRA}")
ax.axhline(T_PENUMBRA, color="green", ls=":", lw=2, alpha=0.8, label=f"Penumbra-photosphere threshold = {T_PENUMBRA}")
add_cycle(ax, 0.95)
ax.set_ylabel("Ratio (DN / I_qs)"); ax.set_xlabel("Year"); ax.set_ylim(0, 1.1); ax.legend(fontsize=8, ncol=2, loc="center right"); ax.grid(alpha=0.3); years(ax)
plt.tight_layout(); save(fig, "fig_threshold_ratio_vs_time_v4mask_en.png")

# ---------- Figures 5-6: spots vs SILSO ----------
si = pd.read_csv(CSV_SILSO, sep=";", header=None, names=["year", "month", "day", "frac", "SN", "SNs", "n", "flag"])
si = si[si["SN"] >= 0].copy(); si["date"] = pd.to_datetime(si[["year", "month", "day"]])
si = si[(si["date"] >= DATE_START) & (si["date"] <= DATE_END)]
idx = pd.date_range(DATE_START, DATE_END, freq="D", name="date")
res = {}
for tag, f in [("legacy", CSV_SPOT_OLD), ("v4", CSV_SPOT)]:
    sp = pd.read_csv(f); sp["date"] = pd.to_datetime(sp["date"]); sp = sp[(sp["date"] >= DATE_START) & (sp["date"] <= DATE_END)]
    d2 = pd.DataFrame(index=idx)
    d2["spot"] = sp.set_index("date")["n_spot_total"]; d2["silso"] = si.set_index("date")["SN"]
    d2["umbra"] = sp.set_index("date")["n_umbra"]; d2["pen_only"] = sp.set_index("date")["n_penumbra_only"]
    d2 = d2.reset_index()
    d2["spot_ma"] = d2["spot"].rolling(MA, center=True, min_periods=1).mean()
    d2["silso_ma"] = d2["silso"].rolling(MA, center=True, min_periods=1).mean()
    d2["diff_ma"] = (d2["spot"] - d2["silso"]).rolling(MA, center=True, min_periods=1).mean()
    both = d2.dropna(subset=["spot", "silso"]); dd = both["spot"] - both["silso"]
    yr = both.assign(diff=dd).groupby(both["date"].dt.year)["diff"].mean()
    res[tag] = d2
    print(f"[{tag}] paired days {len(both)} | spots {int(both['spot'].sum()):,} | silso {int(both['silso'].sum()):,} | "
          f"difference {int(dd.sum()):+,} ({dd.mean():+.1f}/day) | umbra {int(both['umbra'].sum()):,} pen-only {int(both['pen_only'].sum()):,}")
    print(f"       Pearson MA{MA}d {d2[['spot_ma','silso_ma']].corr().iloc[0,1]:.4f} | daily {both[['spot','silso']].corr().iloc[0,1]:.4f}")
    print("       mean difference/day per year: " + ", ".join(f"{y}:{v:+.0f}" for y, v in yr.items()))
d2 = res["v4"]
fig, ax = plt.subplots(figsize=(18, 6.5))
fig.suptitle(f"Daily spot count (HMI segmentation) vs SILSO sunspot number | MA{MA}d", fontsize=13, fontweight="bold")
ax.plot(d2["date"], d2["spot_ma"], color="steelblue", lw=2.2, label=f"Spot total (HMI) MA{MA}d")
ax.plot(d2["date"], d2["silso_ma"], color="darkorange", lw=2.2, label=f"SILSO SN MA{MA}d")
add_cycle(ax, max(d2["spot_ma"].max(), d2["silso_ma"].max()) * 0.95)
ax.set_ylabel("Spot / SILSO SN per day"); ax.set_xlabel("Year"); ax.legend(fontsize=10, loc="upper left"); ax.grid(alpha=0.3); years(ax)
plt.tight_layout(); save(fig, "fig_val_spot_vs_silso_v4mask_en.png")
fig, ax = plt.subplots(figsize=(18, 5.5))
fig.suptitle(f"Daily spot-count difference (HMI - SILSO) | MA{MA}d | 2012-2026", fontsize=13, fontweight="bold")
ax.axhline(0, color="darkorange", lw=1.5)
ax.plot(d2["date"], d2["diff_ma"], color="steelblue", lw=2, label=f"HMI - SILSO MA{MA}d")
add_cycle(ax, d2["diff_ma"].max() * 0.95)
ax.set_ylabel("Spot-count difference per day"); ax.set_xlabel("Year"); ax.legend(fontsize=10, loc="upper left"); ax.grid(alpha=0.3); years(ax)
plt.tight_layout(); save(fig, "fig_val_spot_diff_v4mask_en.png")

# ---------- input for Sec. 3.3: pixel percentiles (IQR-filtered, zone <= 60 deg) / I_qs ----------
print("\nPixel percentiles of v4 mask classes / I_qs (daily median [p5, p95]):")
for c in ["umb_p50", "umb_p90", "pen_p10", "pen_p50", "pen_p90", "umb_mean", "pen_mean", "pho_mean"]:
    r = df[c] / df["I_qs"]
    print(f"  {c:9s} {r.median():.4f} [{r.quantile(.05):.4f}, {r.quantile(.95):.4f}]")
print(f"I_qs trend: {np.polyfit((df['date']-df['date'].min()).dt.days/365.25, df['I_qs'], 1)[0]:.1f} DN/year")

for p in saved:
    shutil.copy(p, DESK / p.name); print(f"[WRITE] {p}")
print(f"[COPY] {DESK}")
