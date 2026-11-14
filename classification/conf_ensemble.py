#!/usr/bin/env python3
"""Confidence head on top of the ensemble (alpha=0.6, XGB standalone + v7 fusion). Signals: ensemble probabilities (MSP/margin/entropy of the
joint head, MSP of Z/p/c, factorized margin) + agreement of the two models (same top-1, same Z/p/c, JS divergence, ensemble=XGB, ensemble=fusion).
Logistic regression, cross-fold calibration. Evaluation: AUROC for wrong predictions, ECE, coverage-accuracy."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from hmi_umbra import paths

R = Path(paths.get("cls_root")) / "opt_v5"
Z_CL, P_CL, C_CL = list("ABCDEFH"), list("xrsahk"), list("xoic")
lg = lambda p: np.log(np.clip(p, 1e-12, None))
X = np.load(R / "xgb/oof_standalone.npz", allow_pickle=True)
xi = {s: i for i, s in enumerate(X["stem"])}
FOLD = {}
for k in range(7):
    a = np.load(R / f"fusion_v7/fold{k}_fusion_v7.npz", allow_pickle=True)
    labels = list(a["labels"]); idx = np.array([xi[s] for s in a["stem"]])
    FOLD[k] = {"stem": a["stem"], "y": a["y"], "fu": [a["pj"], a["pz"], a["pp"], a["pc"]],
               "xg": [X["pj"][idx], X["pz"][idx], X["pp"][idx], X["pc"][idx]]}
L = np.array(labels)
Zi = np.array([Z_CL.index(l[0]) for l in labels]); Pi = np.array([P_CL.index(l[1]) for l in labels]); Ci = np.array([C_CL.index(l[2]) for l in labels])


def score(P):
    return 0.5 * lg(P[0]) + 0.5 * (lg(P[1])[:, Zi] + lg(P[2])[:, Pi] + lg(P[3])[:, Ci])


def js(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (p * (lg(p) - lg(m))).sum(1) + 0.5 * (q * (lg(q) - lg(m))).sum(1)


NAMES = ["MSP", "margin", "neg_entropy", "MSP_Z", "MSP_p", "MSP_c", "factorized_margin", "agree_top1", "agree_Z", "agree_p",
         "agree_c", "neg_JS", "ens_eq_xgb", "ens_eq_fusion"]
for k in range(7):
    d = FOLD[k]
    E = [0.6 * x + 0.4 * f for x, f in zip(d["xg"], d["fu"])]
    se, sx, sf = score(E), score(d["xg"]), score(d["fu"])
    pe, px, pf = se.argmax(1), sx.argmax(1), sf.argmax(1)
    pj = np.sort(E[0], 1)[:, ::-1]; ss = np.sort(se, 1)[:, ::-1]
    sig = np.column_stack([pj[:, 0], pj[:, 0] - pj[:, 1], (E[0] * lg(E[0])).sum(1), E[1].max(1), E[2].max(1), E[3].max(1),
                           ss[:, 0] - ss[:, 1], px == pf] + [np.array([L[a][i] == L[b][i] for a, b in zip(px, pf)]) for i in range(3)] +
                          [-js(d["xg"][0], d["fu"][0]), pe == px, pe == pf]).astype(float)
    d.update({"sig": sig, "pred": pe, "ok": (pe == d["y"]).astype(float)})

rows = []
for k in range(7):
    oth = [j for j in range(7) if j != k]
    Xs = np.vstack([FOLD[j]["sig"] for j in oth]); yo = np.concatenate([FOLD[j]["ok"] for j in oth])
    mu, sd = Xs.mean(0), Xs.std(0) + 1e-6
    lr = LogisticRegression(max_iter=3000).fit((Xs - mu) / sd, yo)
    conf = lr.predict_proba((FOLD[k]["sig"] - mu) / sd)[:, 1]
    df = pd.DataFrame(FOLD[k]["sig"], columns=NAMES)
    df["stem"], df["fold"], df["true"], df["pred"], df["ok"], df["conf"] = FOLD[k]["stem"], k, L[FOLD[k]["y"]], L[FOLD[k]["pred"]], FOLD[k]["ok"], conf
    rows.append(df)
A = pd.concat(rows, ignore_index=True)
A.to_csv(R / "xgb/conf_ensemble_oof.csv", index=False)
ok = A.ok.to_numpy(); wrong = 1 - ok
print(f"OOF n={len(A)} | ensemble accuracy {ok.mean():.4f} | XGB & fusion agree on top-1 for {A.agree_top1.mean():.1%} of crops "
      f"(accuracy when agreeing {ok[A.agree_top1 == 1].mean():.4f}, when disagreeing {ok[A.agree_top1 == 0].mean():.4f})")
print("\nAUROC for detecting wrong predictions:")
for c in ["conf", "factorized_margin", "MSP", "margin", "neg_JS", "agree_top1", "neg_entropy"]:
    print(f"  {c:<18} {roc_auc_score(wrong, -A[c].to_numpy() if c != 'conf' else -A.conf.to_numpy()):.4f}")


def ece(conf, okk, nb=15):
    b = np.linspace(0, 1, nb + 1); e = 0.0
    for lo, hi in zip(b[:-1], b[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(okk[m].mean() - conf[m].mean())
    return e


print(f"\nECE: confidence head {ece(A.conf.to_numpy(), ok):.4f} (mean {A.conf.mean():.4f}) | ensemble MSP {ece(A.MSP.to_numpy(), ok):.4f} "
      f"(mean {A.MSP.mean():.4f}) | accuracy {ok.mean():.4f}")
cov = []
for hold in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    r = {"held_%": int(hold * 100)}
    for c in ["conf", "factorized_margin", "MSP"]:
        v = A[c].to_numpy()
        acc = v > np.quantile(v, hold) if hold > 0 else np.ones(len(v), bool)
        r[f"acc_{c}"] = round(float(ok[acc].mean()), 4)
    cov.append(r)
pd.set_option("display.width", 200)
print("\nCoverage vs accuracy (accuracy of ACCEPTED crops):")
print(pd.DataFrame(cov).to_string(index=False))
print(f"saved: {R / 'xgb/conf_ensemble_oof.csv'}")
