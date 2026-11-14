#!/usr/bin/env python3
"""Ensemble of XGB standalone + v7 fusion: per-head (joint/Z/p/c) probability average with weight alpha, then factorized decoding w=0.5.
alpha per fold chosen on the 6 other folds (accuracy), compared with a fixed alpha of 0.5. 7-fold v7 OOF."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score

from hmi_umbra import paths

R = Path(paths.get("cls_root")) / "opt_v5"
Z_CL, P_CL, C_CL = list("ABCDEFH"), list("xrsahk"), list("xoic")
lg = lambda p: np.log(np.clip(p, 1e-12, None))
X = np.load(R / "xgb/oof_standalone.npz", allow_pickle=True)
xi = {s: i for i, s in enumerate(X["stem"])}
F, labels = {}, None
for k in range(7):
    a = np.load(R / f"fusion_v7/fold{k}_fusion_v7.npz", allow_pickle=True)
    labels = list(a["labels"]) if labels is None else labels
    assert list(a["labels"]) == labels
    idx = np.array([xi[s] for s in a["stem"]])
    F[k] = {"y": a["y"], "fus": [a["pj"], a["pz"], a["pp"], a["pc"]], "xgb": [X["pj"][idx], X["pz"][idx], X["pp"][idx], X["pc"][idx]]}
L = np.array(labels)
Zi = np.array([Z_CL.index(l[0]) for l in labels]); Pi = np.array([P_CL.index(l[1]) for l in labels]); Ci = np.array([C_CL.index(l[2]) for l in labels])


def pred(k, alpha):
    P = [alpha * x + (1 - alpha) * f for x, f in zip(F[k]["xgb"], F[k]["fus"])]
    return (0.5 * lg(P[0]) + 0.5 * (lg(P[1])[:, Zi] + lg(P[2])[:, Pi] + lg(P[3])[:, Ci])).argmax(1)


ALPHAS = np.round(np.arange(0, 1.01, 0.1), 2)
acc_of = {(k, a): np.mean(pred(k, a) == F[k]["y"]) for k in range(7) for a in ALPHAS}
n_of = {k: len(F[k]["y"]) for k in range(7)}
chosen = {}
for k in range(7):
    oth = [j for j in range(7) if j != k]
    chosen[k] = max(ALPHAS, key=lambda a: sum(acc_of[(j, a)] * n_of[j] for j in oth))
y = np.concatenate([F[k]["y"] for k in range(7)]); t = L[y]
P = {"FUSION v7": np.concatenate([pred(k, 0.0) for k in range(7)]), "XGB standalone": np.concatenate([pred(k, 1.0) for k in range(7)]),
     "Ensemble alpha=0.5": np.concatenate([pred(k, 0.5) for k in range(7)]),
     "Ensemble alpha cross-fold": np.concatenate([pred(k, chosen[k]) for k in range(7)])}


def metr(pi):
    p = L[pi]
    r = {"acc": np.mean(p == t), "macroF1": f1_score(t, p, average="macro", labels=labels, zero_division=0),
         "wF1": f1_score(t, p, average="weighted", labels=labels, zero_division=0)}
    for i, c in enumerate("Zpc"):
        r[c] = np.mean([a[i] == b[i] for a, b in zip(p, t)])
    return r


pd.set_option("display.width", 200)
print(f"OOF n={len(y)} | alpha chosen per fold: {[float(chosen[k]) for k in range(7)]}")
print(pd.DataFrame({k: metr(v) for k, v in P.items()}).T.round(4).to_string())
print("\nalpha curve (pooled OOF accuracy, informative only):")
print(" ".join(f"{a}:{np.mean(np.concatenate([pred(k, a) for k in range(7)]) == y):.4f}" for a in ALPHAS))
e = P["Ensemble alpha cross-fold"]
for name in ("FUSION v7", "XGB standalone"):
    b = P[name]
    n01, n10 = int(((b == y) & (e != y)).sum()), int(((b != y) & (e == y)).sum())
    print(f"McNemar ensemble vs {name}: only {name} correct {n01} | only ensemble correct {n10} | p {binomtest(n10, n01 + n10, 0.5).pvalue:.2e}")
