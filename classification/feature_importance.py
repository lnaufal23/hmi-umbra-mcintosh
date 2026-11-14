#!/usr/bin/env python3
"""Feature importance of XGB standalone (morphological descriptors + extent) on v7: gain & mean|SHAP| (pred_contribs) on the test crops of each fold,
averaged over 7 folds, for joint/Z/p/c. 4-panel plot to opt_v5/xgb/vis. The paper figure is drawn by figures/plot_feature_importance_en.py."""
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hmi_umbra import paths

R = Path(paths.get("cls_root"))
src = open(Path(__file__).resolve().parent / "xgb_v7.py").read().split("lg = lambda p:")[0]
ns = {}
exec(compile(src, "xgb_defs", "exec"), ns)
D, COMP, target, weights, PARAMS = ns["D"], ns["COMP"], ns["target"], ns["weights"], ns["PARAMS"]
FEAT = list(ns["ns"]["FEAT"]) + ["extent_max_deg"]
OUT = R / "opt_v5/xgb"
DESC = {"grp_umb_deg2": "total umbral area of group", "grp_pen_deg2": "total penumbral area of group", "ls_umb_deg2": "umbral area of largest spot",
        "ls_pen_deg2": "penumbral area of largest spot", "ls_pen_umb_ratio": "penumbra/umbra ratio of largest spot",
        "grp_n_regions": "number of spot regions", "grp_n_reg_umb": "number of regions with umbra", "grp_n_reg_pen": "number of regions with penumbra",
        "ls_n_umb": "number of umbrae in largest spot", "pen_end_min": "penumbra at group ends (min)", "pen_end_max": "penumbra at group ends (max)",
        "grp_major_deg": "group length (major axis)", "grp_minor_deg": "group width (minor axis)", "grp_frac_largest": "area fraction of largest spot",
        "grp_bins_filled": "compactness along group axis", "ls_major_deg": "major axis of largest spot", "ls_minor_deg": "minor axis of largest spot",
        "ls_axis_ratio": "axis ratio of largest spot (elongation)", "ls_ecc": "eccentricity of largest spot", "ls_ray_cov": "penumbral coverage (rays)",
        "ls_ray_gap_deg_ang": "largest gap without penumbra", "ls_pen_width_med_deg": "median penumbral width",
        "ls_pen_width_p10_deg": "10th-percentile penumbral width", "ls_trans_mean": "umbra-penumbra transitions per ray",
        "ls_edge_ph_frac": "umbral edge touching photosphere", "ls_umb_offset_norm": "umbral center offset (symmetry)",
        "largest_pos_dev": "position of largest spot in group", "has_umb": "umbra present", "extent_max_deg": "AR extent (meta)"}
GROUP = lambda f: "largest spot" if f.startswith("ls_") else ("extent" if f == "extent_max_deg" else ("group" if f != "has_umb" else "other"))
COL = {"largest spot": "#185FA5", "group": "#3B6D11", "extent": "#993C1D", "other": "#888888"}

G = {c: np.zeros(len(FEAT)) for c in COMP}; S = {c: np.zeros(len(FEAT)) for c in COMP}
for k in range(7):
    tr, va, te = D[k]["train"], D[k]["val"], D[k]["test"]
    for comp, (i, cl) in COMP.items():
        ytr, yva = target(tr["y"], comp), target(va["y"], comp)
        present = np.unique(ytr); remap = {c: j for j, c in enumerate(present)}
        ytr_r = np.array([remap[v] for v in ytr]); keep = np.isin(yva, present); yva_r = np.array([remap[v] for v in yva[keep]])
        m = xgb.XGBClassifier(objective="multi:softprob", num_class=len(present), **PARAMS)
        m.fit(tr["feat"], ytr_r, sample_weight=weights(ytr_r, len(present)), eval_set=[(va["feat"][keep], yva_r)], verbose=False)
        b = m.get_booster()
        gain = b.get_score(importance_type="gain")
        g = np.array([gain.get(f"f{j}", 0.0) for j in range(len(FEAT))]); G[comp] += g / max(g.sum(), 1e-12) / 7
        b.set_param({"device": "cpu"})
        contrib = b.predict(xgb.DMatrix(te["feat"]), pred_contribs=True, iteration_range=(0, m.best_iteration + 1))
        sh = np.abs(contrib[..., :-1]).mean(axis=tuple(range(contrib.ndim - 1)))
        S[comp] += sh / max(sh.sum(), 1e-12) / 7
    print(f"fold {k} done", flush=True)

pd.set_option("display.width", 200)
rows = []
for comp in COMP:
    T = pd.DataFrame({"feature": FEAT, "description": [DESC[f] for f in FEAT], "group": [GROUP(f) for f in FEAT],
                      "shap_%": (100 * S[comp]).round(2), "gain_%": (100 * G[comp]).round(2)}).sort_values("shap_%", ascending=False)
    T.insert(0, "component", comp); rows.append(T)
    print(f"\n== {comp}: top 15 features (mean |SHAP|, % of total) ==")
    print(T.head(15)[["feature", "description", "group", "shap_%", "gain_%"]].to_string(index=False))
    tot = T.groupby("group")["shap_%"].sum().round(1).to_dict()
    print(f"   contribution per group (SHAP %): {tot}")
pd.concat(rows).to_csv(OUT / "feature_importance_v7.csv", index=False)

fig, axes = plt.subplots(1, 4, figsize=(24, 7))
for ax, comp in zip(axes, COMP):
    order = np.argsort(-S[comp])[:15][::-1]
    ax.barh([DESC[FEAT[j]] for j in order], 100 * S[comp][order], color=[COL[GROUP(FEAT[j])] for j in order])
    ax.set_title(f"{comp}: top 15 features (mean |SHAP|, %)", fontsize=11)
    ax.tick_params(axis="y", labelsize=8)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COL.values()]
fig.legend(handles, COL.keys(), loc="lower center", ncol=4)
plt.tight_layout(rect=[0, 0.05, 1, 1])
p = OUT / "vis/feature_importance_v7.png"
plt.savefig(p, dpi=90); plt.close()
print(f"\nsaved: {OUT / 'feature_importance_v7.csv'} | plot {p}")
