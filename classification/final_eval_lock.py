#!/usr/bin/env python3
"""Final evaluation on the locked test set (1394 crops, 35 classes). All components are trained on the pool (train/val) only:
fusion (frozen final CNN + 28 descriptors, 3 seeds), XGB standalone (joint/Z/p/c), ensemble alpha=0.6 (fixed from OOF), factorized decoding w=0.5,
confidence head (logistic regression trained on the 7-fold OOF signals without the locked-test crops)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report

from hmi_umbra import paths

R = Path(paths.get("cls_root"))
OUT = R / "opt_v5/final_lock"
src = open(Path(__file__).resolve().parent / "eval_variants.py").read()
src = src.split("ap = argparse.ArgumentParser()")[0] + "\nR = Path(paths.get('cls_root'))\n" + \
      src.split("a = ap.parse_args()")[1].split("sm = lambda x:")[0]
ev = {}
exec(compile(src, "eval_variants_defs", "exec"), ev)
FMAP = ev["FMAP"]
Z_CL, P_CL, C_CL = list("ABCDEFH"), list("xrsahk"), list("xoic")
lg = lambda p: np.log(np.clip(p, 1e-12, None))
sm = lambda x: np.exp(x - x.max(1, keepdims=True)) / np.exp(x - x.max(1, keepdims=True)).sum(1, keepdims=True)
v7 = pd.read_csv(R / "train_audit/v7_relabel_A/export_selected_v7.csv", dtype={"date": str})
EXT = dict(zip(v7.stem, v7.extent_max_deg))


def load(part):
    d = np.load(OUT / "emb" / f"{part}_outputs.npz", allow_pickle=True)
    labels = list(d["labels"]); lab = np.array(labels)[d["y"]]; t = torch.tensor
    return {"stem": d["stem"], "e": t(d["emb"]).float(), "f": t(np.stack([FMAP[s] for s in d["stem"]])), "y": t(d["y"]).long(),
            "z": t([Z_CL.index(l[0]) for l in lab]), "p": t([P_CL.index(l[1]) for l in lab]), "c": t([C_CL.index(l[2]) for l in lab]),
            "labels": labels, "cnn": [d["joint"], d["z"], d["p"], d["c"]],
            "xf": np.column_stack([np.stack([FMAP[s] for s in d["stem"]]), np.array([EXT[s] for s in d["stem"]])])}


Tr, Va, Te = load("train"), load("val"), load("test")
labels = Tr["labels"]; L = np.array(labels); n_cls = len(labels)
Zi = np.array([Z_CL.index(l[0]) for l in labels]); Pi = np.array([P_CL.index(l[1]) for l in labels]); Ci = np.array([C_CL.index(l[2]) for l in labels])
score = lambda P: 0.5 * lg(P[0]) + 0.5 * (lg(P[1])[:, Zi] + lg(P[2])[:, Pi] + lg(P[3])[:, Ci])
yt = Te["y"].numpy(); t = L[yt]
print(f"train {len(Tr['y'])} | val {len(Va['y'])} | locked test {len(yt)} | classes {n_cls}", flush=True)

FU = None
for sd in range(3):
    o = ev["train_fusion"](Tr, Va, Te, n_cls, 7000 + sd)
    FU = [x / 3 for x in o] if FU is None else [p + x / 3 for p, x in zip(FU, o)]
print("fusion done", flush=True)

PARAMS = dict(n_estimators=2000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.6, min_child_weight=2,
              reg_lambda=1.0, tree_method="hist", device="cuda", early_stopping_rounds=50, eval_metric="mlogloss", random_state=42)


def comp_target(y, i, cl):
    return y if i is None else np.array([cl.index(L[v][i]) for v in y])


XG = []
for i, cl in ((None, labels), (0, Z_CL), (1, P_CL), (2, C_CL)):
    ytr, yva = comp_target(Tr["y"].numpy(), i, cl), comp_target(Va["y"].numpy(), i, cl)
    present = np.unique(ytr); remap = {c: j for j, c in enumerate(present)}
    ytr_r = np.array([remap[v] for v in ytr]); keep = np.isin(yva, present); yva_r = np.array([remap[v] for v in yva[keep]])
    cnt = np.bincount(ytr_r, minlength=len(present)).astype(float); w = 1 / np.sqrt(np.maximum(cnt, 1)); w = w / w[ytr_r].mean()
    m = xgb.XGBClassifier(objective="multi:softprob", num_class=len(present), **PARAMS)
    m.fit(Tr["xf"], ytr_r, sample_weight=w[ytr_r], eval_set=[(Va["xf"][keep], yva_r)], verbose=False)
    P = np.zeros((len(yt), len(cl))); P[:, present] = m.predict_proba(Te["xf"]); XG.append(P)
print("XGB done", flush=True)

CN = [sm(x) for x in Te["cnn"]]
EN = [0.6 * x + 0.4 * f for x, f in zip(XG, FU)]
PRED = {"CNN (joint head)": L[CN[0].argmax(1)], "CNN factorized": L[score(CN).argmax(1)], "FUSION factorized": L[score(FU).argmax(1)],
        "XGB standalone factorized": L[score(XG).argmax(1)], "ENSEMBLE a=0.6": L[score(EN).argmax(1)]}
relab = set(pd.read_csv(R / "opt_v5/relabel_r1/ZPC_RELABEL_R1_export.csv").stem)
untouched = ~np.isin(Te["stem"], list(relab))


def metr(p, mask=None):
    mask = np.ones(len(t), bool) if mask is None else mask
    tt, pp = t[mask], p[mask]
    r = {"n": int(mask.sum()), "acc": np.mean(pp == tt), "macroF1": f1_score(tt, pp, average="macro", labels=labels, zero_division=0),
         "wF1": f1_score(tt, pp, average="weighted", labels=labels, zero_division=0)}
    for i, c in enumerate("Zpc"):
        a_, b_ = [x[i] for x in tt], [x[i] for x in pp]
        r[f"{c}_acc"] = np.mean(np.array(a_) == np.array(b_)); r[f"{c}_mF1"] = f1_score(a_, b_, average="macro", zero_division=0)
    return r


pd.set_option("display.width", 230)
print("\n== LOCKED TEST, 35 classes ==")
T1 = pd.DataFrame({k: metr(v) for k, v in PRED.items()}).T
print(T1.round(4).to_string())
print(f"\n== Sensitivity test: test crops untouched by relabeling ==")
print(pd.DataFrame({k: metr(v, untouched) for k, v in PRED.items()}).T.round(4).to_string())

OLD29 = ["Axx", "Bxi", "Bxo", "Cao", "Cko", "Cri", "Cro", "Cso", "Dac", "Dai", "Dao", "Dkc", "Dki", "Dko", "Dri", "Dro", "Dsi",
         "Dso", "Eai", "Eao", "Ekc", "Eko", "Esi", "Eso", "Fkc", "Hax", "Hhx", "Hrx", "Hsx"]
common = [c for c in labels if c in OLD29]; ci = np.array([labels.index(c) for c in common])
k28 = np.isin(t, common)
Zc = np.array([Z_CL.index(c[0]) for c in common]); Pc = np.array([P_CL.index(c[1]) for c in common]); Cc = np.array([C_CL.index(c[2]) for c in common])


def dec28(P):
    pj = P[0][k28][:, ci]; pj = pj / pj.sum(1, keepdims=True)
    return np.array(common)[(0.5 * lg(pj) + 0.5 * (lg(P[1][k28])[:, Zc] + lg(P[2][k28])[:, Pc] + lg(P[3][k28])[:, Cc])).argmax(1)]


t28 = t[k28]
row28 = lambda p: [np.mean(p == t28), f1_score(t28, p, average="macro", labels=common, zero_division=0),
                   f1_score(t28, p, average="weighted", labels=common, zero_division=0)] + [np.mean([a[i] == b[i] for a, b in zip(t28, p)]) for i in range(3)]
T28 = {"Earlier CNN (leaky split, n=637)": [0.6672, 0.6547, 0.6658, 0.8932, 0.7881, 0.9074], "Earlier XGB hybrid": [0.6374, 0.5773, 0.6322, 0.8870, 0.7802, 0.8854],
       "Earlier XGB standalone": [0.6060, 0.5614, 0.5974, 0.8367, 0.7410, 0.8540],
       f"FUSION (28 classes, n={k28.sum()})": row28(dec28(FU)), "XGB standalone (28 classes)": row28(dec28(XG)), "ENSEMBLE (28 classes)": row28(dec28(EN))}
print("\n== 28 shared classes vs the earlier pipeline ==")
print(pd.DataFrame(T28, index=["acc", "macroF1", "wF1", "Z", "p", "c"]).T.round(4).to_string())


def js(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (p * (lg(p) - lg(m))).sum(1) + 0.5 * (q * (lg(q) - lg(m))).sum(1)


def signals(E, X_, F_):
    se, sx, sf = score(E), score(X_), score(F_)
    pe, px, pf = se.argmax(1), sx.argmax(1), sf.argmax(1)
    pj = np.sort(E[0], 1)[:, ::-1]; ss = np.sort(se, 1)[:, ::-1]
    return np.column_stack([pj[:, 0], pj[:, 0] - pj[:, 1], (E[0] * lg(E[0])).sum(1), E[1].max(1), E[2].max(1), E[3].max(1),
                            ss[:, 0] - ss[:, 1], px == pf] + [np.array([L[a][i] == L[b][i] for a, b in zip(px, pf)]) for i in range(3)] +
                           [-js(X_[0], F_[0]), pe == px, pe == pf]).astype(float)


NAMES = ["MSP", "margin", "neg_entropy", "MSP_Z", "MSP_p", "MSP_c", "factorized_margin", "agree_top1", "agree_Z", "agree_p",
         "agree_c", "neg_JS", "ens_eq_xgb", "ens_eq_fusion"]
O = pd.read_csv(R / "opt_v5/xgb/conf_ensemble_oof.csv")
lock = set(pd.read_csv(R / "test_lock/test_locked.csv").stem)
O = O[~O.stem.isin(lock)]
Xs = O[NAMES].to_numpy(); mu, sdv = Xs.mean(0), Xs.std(0) + 1e-6
lr = LogisticRegression(max_iter=3000).fit((Xs - mu) / sdv, O.ok.to_numpy())
sig = signals(EN, XG, FU)
conf = lr.predict_proba((sig - mu) / sdv)[:, 1]
pe = PRED["ENSEMBLE a=0.6"]; ok = (pe == t).astype(float)
b = np.linspace(0, 1, 16); ece = sum(((conf > lo) & (conf <= hi)).mean() * abs(ok[(conf > lo) & (conf <= hi)].mean() - conf[(conf > lo) & (conf <= hi)].mean())
                                     for lo, hi in zip(b[:-1], b[1:]) if ((conf > lo) & (conf <= hi)).any())
print(f"\n== Confidence (head trained on OOF without the locked test, n train {len(O)}) ==")
print(f"AUROC wrong predictions: head {roc_auc_score(1 - ok, -conf):.4f} | ensemble MSP {roc_auc_score(1 - ok, -sig[:, 0]):.4f} | ECE head {ece:.4f} "
      f"(mean {conf.mean():.4f} vs accuracy {ok.mean():.4f})")
print("coverage vs accuracy: " + " | ".join(f"held {int(h * 100)}%: {ok[conf > np.quantile(conf, h)].mean():.4f}" if h else f"held 0%: {ok.mean():.4f}"
                                           for h in (0, 0.1, 0.2, 0.3, 0.4, 0.5)))
c_ = PRED["CNN (joint head)"]
n01, n10 = int(((c_ == t) & (pe != t)).sum()), int(((c_ != t) & (pe == t)).sum())
print(f"\nMcNemar ENSEMBLE vs CNN (joint head): only CNN correct {n01} | only ensemble correct {n10} | p {binomtest(n10, n01 + n10, 0.5).pvalue:.2e}")

rep = classification_report(t, pe, labels=labels, zero_division=0, digits=4)
(OUT / "classification_report_test_lock_ensemble.txt").write_text(rep)
np.savetxt(OUT / "confusion_matrix_test_lock_ensemble.txt", confusion_matrix(t, pe, labels=labels), fmt="%d")
rank = np.argsort(-score(EN), 1)
pd.DataFrame({"stem": Te["stem"], "true": t, **{k: v for k, v in PRED.items()}, "confidence": conf, "relabel_touched": ~untouched,
              "top5": [" ".join(L[rank[i, :5]]) for i in range(len(t))]}).to_csv(OUT / "test_lock_predictions.csv", index=False)
json.dump({k: {kk: float(vv) for kk, vv in r.items()} for k, r in T1.to_dict(orient="index").items()}, open(OUT / "summary_test_lock.json", "w"), indent=2)
print("\nPer class (ensemble, locked test):")
print(rep)
print(f"saved: {OUT}")
