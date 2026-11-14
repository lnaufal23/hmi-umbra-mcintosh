#!/usr/bin/env python3
"""Paper classification figures (no titles) from the locked-test predictions: row-normalized Z/p/c confusion matrices of the main
XGBoost model (cm_components_xgb_en.png) and coverage versus accuracy of the ensemble confidence (coverage_accuracy_en.png)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from hmi_umbra import paths

R = Path(paths.get("cls_root")) / "opt_v5"
OUT = Path(paths.get("paper_figs"))
d = pd.read_csv(R / "final_lock/test_lock_predictions.csv")
t = d.true.astype(str).to_numpy()
px = d["XGB standalone factorized"].astype(str).to_numpy()

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, (i, cl, name) in zip(axes, ((0, list("ABCDEFH"), "(a) Z"), (1, list("xrsahk"), "(b) p"), (2, list("xoic"), "(c) c"))):
    cm = confusion_matrix([a[i] for a in t], [b[i] for b in px], labels=cl).astype(float)
    cmn = cm / cm.sum(1, keepdims=True).clip(1)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for r in range(len(cl)):
        for c in range(len(cl)):
            ax.text(c, r, f"{cmn[r, c]:.2f}\n({int(cm[r, c])})", ha="center", va="center", fontsize=8, color="white" if cmn[r, c] > 0.5 else "black")
    ax.set_xticks(range(len(cl))); ax.set_xticklabels(cl); ax.set_yticks(range(len(cl))); ax.set_yticklabels(cl)
    ax.set_xlabel("predicted"); ax.set_ylabel("audited label"); ax.set_title(name, loc="left")
plt.tight_layout(); plt.savefig(OUT / "cm_components_xgb_en.png", dpi=110); plt.close()

conf = d.confidence.to_numpy()
oke = d["ENSEMBLE a=0.6"].astype(str).to_numpy() == t
hs = np.linspace(0, 0.8, 33)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot([100 * (1 - h) for h in hs], [oke[conf > np.quantile(conf, h)].mean() if h else oke.mean() for h in hs], "o-", ms=3)
ax.set_xlabel("coverage (% of crops accepted)"); ax.set_ylabel("accuracy on accepted crops"); ax.invert_xaxis(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUT / "coverage_accuracy_en.png", dpi=110); plt.close()
print(f"n test = {len(t)} | XGB accuracy = {np.mean(px == t):.3f} | ensemble accuracy = {oke.mean():.3f}")
