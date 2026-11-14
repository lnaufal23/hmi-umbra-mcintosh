#!/usr/bin/env python3
"""XGBoost standalone (morphological descriptors + extent) and hybrid (descriptors + 528-d v7 CNN embedding) on the final v7 dataset, 7-fold OOF per AR group.
hybrid   : trained on train+val of fold k (embeddings from the fold-k CNN), early stopping on val.
hybrid_cf: trained on the TEST crops of the 6 other folds (honest OOF embeddings), early stopping on a 10% hold-out.
Per model: joint classifier (35) + Z/p/c; decisions from the joint head and from the factorized combination w=0.5. Compared with the v7 fusion and the earlier pipeline."""
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import binomtest
from sklearn.metrics import f1_score

from hmi_umbra import paths

R = Path(paths.get("cls_root"))
EMB = R / "opt_v5/v7_relabel_A"
OUT = R / "opt_v5/xgb"
Z_CL, P_CL, C_CL = list("ABCDEFH"), list("xrsahk"), list("xoic")
src = open(Path(__file__).resolve().parent / "conf_head.py").read().split("class Fusion")[0]
ns = {}
exec(compile(src, "feat_defs", "exec"), ns)
FMAP = ns["FMAP"]
v7 = pd.read_csv(R / "train_audit/v7_relabel_A/export_selected_v7.csv", dtype={"date": str})
EXT = dict(zip(v7.stem, v7.extent_max_deg))
PARAMS = dict(n_estimators=2000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.6, min_child_weight=2,
              reg_lambda=1.0, tree_method="hist", device="cuda", early_stopping_rounds=50, eval_metric="mlogloss", random_state=42)

D = {}
for k in range(7):
    D[k] = {}
    for part in ("train", "val", "test"):
        a = np.load(EMB / f"fold{k}" / f"{part}_outputs.npz", allow_pickle=True)
        st = a["stem"]
        feat = np.column_stack([np.stack([FMAP[s] for s in st]), np.array([EXT[s] for s in st])])
        D[k][part] = {"stem": st, "y": a["y"], "feat": feat, "emb": a["emb"], "labels": list(a["labels"])}
labels = D[0]["test"]["labels"]; L = np.array(labels)
for k in range(7):
    assert D[k]["test"]["labels"] == labels
COMP = {"joint": (None, labels), "Z": (0, Z_CL), "p": (1, P_CL), "c": (2, C_CL)}


def target(y, comp):
    i, cl = COMP[comp]
    return y if i is None else np.array([cl.index(L[v][i]) for v in y])


def weights(t, n):
    cnt = np.bincount(t, minlength=n).astype(float)
    w = 1 / np.sqrt(np.maximum(cnt, 1)); w = w / w[t].mean()
    return w[t]


def X_of(d, mode):
    return d["feat"] if mode == "standalone" else np.hstack([d["feat"], d["emb"]])


def fit_predict(Xtr, ytr, Xva, yva, Xte, n):
    present = np.unique(ytr)
    remap = {c: i for i, c in enumerate(present)}
    ytr_r = np.array([remap[v] for v in ytr]); keep = np.isin(yva, present)
    yva_r = np.array([remap[v] for v in yva[keep]])
    m = xgb.XGBClassifier(objective="multi:softprob", num_class=len(present), **PARAMS)
    m.fit(Xtr, ytr_r, sample_weight=weights(ytr_r, len(present)), eval_set=[(Xva[keep], yva_r)], verbose=False)
    P = np.zeros((len(Xte), n)); P[:, present] = m.predict_proba(Xte)
    return P, m.best_iteration


lg = lambda p: np.log(np.clip(p, 1e-12, None))
Zi = np.array([Z_CL.index(l[0]) for l in labels]); Pi = np.array([P_CL.index(l[1]) for l in labels]); Ci = np.array([C_CL.index(l[2]) for l in labels])
PRED = {}
for mode in ("standalone", "hybrid", "hybrid_cf"):
    stems, pj_, pz_, pp_, pc_, its = [], [], [], [], [], []
    for k in range(7):
        te = D[k]["test"]
        if mode == "hybrid_cf":
            oth = [j for j in range(7) if j != k]
            Xall = np.vstack([X_of(D[j]["test"], mode) for j in oth]); yall = np.concatenate([D[j]["test"]["y"] for j in oth])
            rng = np.random.default_rng(k); idx = rng.permutation(len(yall)); nv = len(idx) // 10
            Xtr, ytr, Xva, yva = Xall[idx[nv:]], yall[idx[nv:]], Xall[idx[:nv]], yall[idx[:nv]]
        else:
            Xtr = np.vstack([X_of(D[k]["train"], mode)]); ytr = D[k]["train"]["y"]
            Xva, yva = X_of(D[k]["val"], mode), D[k]["val"]["y"]
        out = {}
        for comp, (i, cl) in COMP.items():
            P, it = fit_predict(Xtr, target(ytr, comp), Xva, target(yva, comp), X_of(te, mode), len(cl))
            out[comp] = P; its.append(it)
        stems += list(te["stem"]); pj_.append(out["joint"]); pz_.append(out["Z"]); pp_.append(out["p"]); pc_.append(out["c"])
        print(f"{mode} fold {k} done", flush=True)
    pj, pz, pp, pc = (np.vstack(x) for x in (pj_, pz_, pp_, pc_))
    PRED[f"{mode}_joint"] = dict(zip(stems, L[pj.argmax(1)]))
    PRED[f"{mode}_factorized"] = dict(zip(stems, L[(0.5 * lg(pj) + 0.5 * (lg(pz)[:, Zi] + lg(pp)[:, Pi] + lg(pc)[:, Ci])).argmax(1)]))
    np.savez(OUT / f"oof_{mode}.npz", stem=np.array(stems), pj=pj, pz=pz, pp=pp, pc=pc)
    print(f"{mode}: mean best_iteration {np.mean(its):.0f}", flush=True)

for k in range(7):
    a = np.load(R / f"opt_v5/fusion_v7/fold{k}_fusion_v7.npz", allow_pickle=True)
    Ld = np.array(list(a["labels"]))
    s = 0.5 * lg(a["pj"]) + 0.5 * (lg(a["pz"])[:, Zi] + lg(a["pp"])[:, Pi] + lg(a["pc"])[:, Ci])
    PRED.setdefault("FUSION_v7_factorized", {}).update(dict(zip(a["stem"], Ld[s.argmax(1)])))
truth = {}
for k in range(7):
    truth.update(dict(zip(D[k]["test"]["stem"], L[D[k]["test"]["y"]])))
stems = sorted(truth)
t = np.array([truth[s] for s in stems])


def metr(P):
    p = np.array([P[s] for s in stems])
    r = {"acc": np.mean(p == t), "macroF1": f1_score(t, p, average="macro", labels=labels, zero_division=0),
         "wF1": f1_score(t, p, average="weighted", labels=labels, zero_division=0)}
    for i, c in enumerate("Zpc"):
        a_, b_ = [x[i] for x in t], [x[i] for x in p]
        r[f"{c}_acc"] = np.mean(np.array(a_) == np.array(b_)); r[f"{c}_mF1"] = f1_score(a_, b_, average="macro", zero_division=0)
    return r


res = {k: metr(v) for k, v in PRED.items()}
res["Earlier CNN (29 classes, leaky split)"] = dict(zip(["acc", "macroF1", "wF1", "Z_acc", "Z_mF1", "p_acc", "p_mF1", "c_acc", "c_mF1"],
                                            [0.6672, 0.6547, 0.6658, 0.8932, 0.8873, 0.7881, 0.8120, 0.9074, 0.8966]))
res["Earlier XGB hybrid"] = dict(zip(["acc", "macroF1", "wF1", "Z_acc", "Z_mF1", "p_acc", "p_mF1", "c_acc", "c_mF1"],
                                   [0.6374, 0.5773, 0.6322, 0.8870, 0.8876, 0.7802, 0.7457, 0.8854, 0.8670]))
res["Earlier XGB standalone"] = dict(zip(["acc", "macroF1", "wF1", "Z_acc", "Z_mF1", "p_acc", "p_mF1", "c_acc", "c_mF1"],
                                       [0.6060, 0.5614, 0.5974, 0.8367, 0.8353, 0.7410, 0.7431, 0.8540, 0.8438]))
T = pd.DataFrame(res).T[["acc", "macroF1", "wF1", "Z_acc", "p_acc", "c_acc", "Z_mF1", "p_mF1", "c_mF1"]]
pd.set_option("display.width", 220)
print(f"\nOOF 7-fold v7 n={len(stems)} (35 classes, split by AR group)")
print(T.round(4).to_string())
T.to_csv(OUT / "xgb_v7_results.csv")
best = max([k for k in PRED if k.startswith(("standalone", "hybrid"))], key=lambda k: res[k]["acc"])
f, b = np.array([PRED["FUSION_v7_factorized"][s] for s in stems]), np.array([PRED[best][s] for s in stems])
n01, n10 = int(((f == t) & (b != t)).sum()), int(((f != t) & (b == t)).sum())
print(f"\nMcNemar FUSION_v7 vs {best} [joint]: only fusion correct {n01} | only XGB correct {n10} | p {binomtest(n10, n01 + n10, 0.5).pvalue:.2e}")
print(f"saved: {OUT}")
