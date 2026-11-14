#!/usr/bin/env python3
"""English SHAP figure for the paper, drawn from opt_v5/xgb/feature_importance_v7.csv (no retraining).
2x2 panels (joint, Z, p, c), top 15 descriptors by mean |SHAP| (% of total), colored by descriptor group."""
from pathlib import Path
import shutil
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hmi_umbra import paths

R = Path(paths.get("cls_root")) / "opt_v5"
CSV = R / "xgb/feature_importance_v7.csv"
OUT = R / "paper/feature_importance_en.png"
FIG = Path(paths.get("paper_figs")) / "feature_importance_en.png"

EN = {"grp_umb_deg2": "Total umbral area, group", "grp_pen_deg2": "Total penumbral area, group",
      "ls_umb_deg2": "Umbral area, largest spot", "ls_pen_deg2": "Penumbral area, largest spot",
      "ls_pen_umb_ratio": "Penumbra/umbra area ratio, largest spot",
      "grp_n_regions": "Number of spot regions", "grp_n_reg_umb": "Number of regions with umbra",
      "grp_n_reg_pen": "Number of regions with penumbra", "ls_n_umb": "Number of umbrae, largest spot",
      "pen_end_min": "Penumbra at group ends (smaller)", "pen_end_max": "Penumbra at group ends (larger)",
      "grp_major_deg": "Group length (major axis)", "grp_minor_deg": "Group width (minor axis)",
      "grp_frac_largest": "Area fraction of largest spot", "grp_bins_filled": "Fill fraction along group axis",
      "ls_major_deg": "Major axis, largest spot", "ls_minor_deg": "Minor axis, largest spot",
      "ls_axis_ratio": "Axis ratio, largest spot", "ls_ecc": "Eccentricity, largest spot",
      "ls_ray_cov": "Penumbral coverage (rays), largest spot", "ls_ray_gap_deg_ang": "Largest penumbral gap, largest spot",
      "ls_pen_width_med_deg": "Median penumbral width, largest spot",
      "ls_pen_width_p10_deg": "10th-pct. penumbral width, largest spot",
      "ls_trans_mean": "Umbra\u2013penumbra transitions per ray, largest spot",
      "ls_edge_ph_frac": "Umbral edge touching photosphere, largest spot",
      "ls_umb_offset_norm": "Umbral centroid offset, largest spot",
      "largest_pos_dev": "Position of largest spot in group", "has_umb": "Umbra present",
      "extent_max_deg": "Region extent"}
GRP = {"largest spot": "Largest spot", "group": "Group", "extent": "Extent", "other": "Other"}
COL = {"Largest spot": "#185FA5", "Group": "#3B6D11", "Extent": "#993C1D", "Other": "#888888"}
TITLE = {"joint": "Joint Zpc", "Z": "Zurich class (Z)", "p": "Penumbra class (p)", "c": "Compactness class (c)"}

df = pd.read_csv(CSV)
missing = sorted(set(df.feature) - set(EN))
assert not missing, f"no English label for: {missing}"
df["label"] = df.feature.map(EN); df["group_label"] = df.group.map(GRP)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
used = set()
for ax, comp in zip(axes.ravel(), ["joint", "Z", "p", "c"]):
    t = df[df.component == comp].sort_values("shap_%", ascending=False).head(15).iloc[::-1]
    ax.barh(t.label, t["shap_%"], color=[COL[g] for g in t.group_label])
    used |= set(t.group_label)
    ax.set_title(TITLE[comp], fontsize=12)
    ax.set_xlabel("Share of mean |SHAP| (%)", fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="x", alpha=0.3)
groups = [g for g in ["Largest spot", "Group", "Extent", "Other"] if g in used]
fig.legend([plt.Rectangle((0, 0), 1, 1, color=COL[g]) for g in groups], groups,
           loc="lower center", ncol=len(groups), fontsize=11, frameon=False)
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(OUT, dpi=200); plt.close()
shutil.copy(OUT, FIG)

for comp in ["p", "Z", "c"]:
    d = df[df.component == comp]
    print(f"{comp}: largest spot {d[d.group=='largest spot']['shap_%'].sum():.1f}% | "
          f"group {d[d.group=='group']['shap_%'].sum():.1f}% | extent {d[d.group=='extent']['shap_%'].sum():.1f}%")
print(f"legend groups: {groups}\nsaved: {OUT}\ncopied: {FIG}")
