#!/usr/bin/env python3
"""Re-train and save the final classifier components exactly as in final_eval_lock.py (same data, recipe and seeds):
XGB standalone (joint/Z/p/c), fusion (frozen final CNN + 28 descriptors, seeds 7000-7002) and the logistic confidence head.
The locked-test metrics and per-crop predictions are recomputed and compared with the saved final_lock outputs,
which are not overwritten. Models are written to opt_v5/final_lock/models/."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from hmi_umbra import paths

R = Path(paths.get("cls_root"))
LOCK = R / "opt_v5/final_lock"
MOD = LOCK / "models"
MOD.mkdir(exist_ok=True)
src = open(Path(__file__).resolve().parent / "eval_variants.py").read()
src = src.split("ap = argparse.ArgumentParser()")[0] + "\nR = Path(paths.get('cls_root'))\n" + \
      src.split("a = ap.parse_args()")[1].split("sm = lambda x:")[0]
ev = {}
exec(compile(src, "eval_variants_defs", "exec"), ev)
exec(compile('''
def train_fusion_model(Tr, Va, n_cls, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    cnt = np.bincount(Tr["y"].numpy(), minlength=n_cls).astype(float)
    w = 1 / np.sqrt(np.maximum(cnt, 1)); w = torch.tensor(np.minimum(w / w.mean(), 5.0), dtype=torch.float32)
    m = Fusion(n_cls).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    ce_j, ce = nn.CrossEntropyLoss(weight=w.to(dev), label_smoothing=LS), nn.CrossEntropyLoss(label_smoothing=LS)
    best, bad, state, n = -1, 0, None, len(Tr["y"])
    for ep in range(EPOCHS):
        m.train()
        perm = torch.randperm(n)
        for s in range(0, n, BS):
            i = perm[s:s + BS]
            o = m(Tr["e"][i].to(dev), Tr["f"][i].to(dev), torch.rand(len(i), device=dev) < MOD_DROP)
            yy = [Tr[k][i].to(dev) for k in ("y", "z", "p", "c")]
            loss = 0.5 * ce_j(o[0], yy[0]) + 0.2 * sum(ce(o[j], yy[j]) for j in (1, 2, 3)) + 0.1 * sum(ce(o[j + 3], yy[j]) for j in (1, 2, 3))
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pv = m(Va["e"].to(dev), Va["f"].to(dev))[0].argmax(1).cpu().numpy()
        f1 = f1_score(Va["y"].numpy(), pv, average="macro", labels=list(range(n_cls)), zero_division=0)
        if f1 > best:
            best, bad, state = f1, 0, {kk: v.detach().clone() for kk, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    m.load_state_dict(state); m.eval()
    return m
''', "train_fusion_model", "exec"), ev)
FMAP, dev = ev["FMAP"], ev["dev"]
Z_CL, P_CL, C_CL = list("ABCDEFH"), list("xrsahk"), list("xoic")
lg = lambda p: np.log(np.clip(p, 1e-12, None))
sm = lambda x: np.exp(x - x.max(1, keepdims=True)) / np.exp(x - x.max(1, keepdims=True)).sum(1, keepdims=True)
v7 = pd.read_csv(R / "train_audit/v7_relabel_A/export_selected_v7.csv", dtype={"date": str})
EXT = dict(zip(v7.stem, v7.extent_max_deg))


def load(part):
    d = np.load(LOCK / "emb" / f"{part}_outputs.npz", allow_pickle=True)
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
    m = ev["train_fusion_model"](Tr, Va, n_cls, 7000 + sd)
    with torch.no_grad():
        o = m(Te["e"].to(dev), Te["f"].to(dev))
    o = [F.softmax(x, 1).cpu().numpy() for x in o[:4]]
    torch.save(m.state_dict(), MOD / f"fusion_seed{7000 + sd}.pt")
    FU = [x / 3 for x in o] if FU is None else [p + x / 3 for p, x in zip(FU, o)]
print("fusion done", flush=True)

PARAMS = dict(n_estimators=2000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.6, min_child_weight=2,
              reg_lambda=1.0, tree_method="hist", device="cuda", early_stopping_rounds=50, eval_metric="mlogloss", random_state=42)


def comp_target(y, i, cl):
    return y if i is None else np.array([cl.index(L[v][i]) for v in y])


XG, XMETA = [], {}
for name, i, cl in (("joint", None, labels), ("Z", 0, Z_CL), ("p", 1, P_CL), ("c", 2, C_CL)):
    ytr, yva = comp_target(Tr["y"].numpy(), i, cl), comp_target(Va["y"].numpy(), i, cl)
    present = np.unique(ytr); remap = {c: j for j, c in enumerate(present)}
    ytr_r = np.array([remap[v] for v in ytr]); keep = np.isin(yva, present); yva_r = np.array([remap[v] for v in yva[keep]])
    cnt = np.bincount(ytr_r, minlength=len(present)).astype(float); w = 1 / np.sqrt(np.maximum(cnt, 1)); w = w / w[ytr_r].mean()
    m = xgb.XGBClassifier(objective="multi:softprob", num_class=len(present), **PARAMS)
    m.fit(Tr["xf"], ytr_r, sample_weight=w[ytr_r], eval_set=[(Va["xf"][keep], yva_r)], verbose=False)
    P = np.zeros((len(yt), len(cl))); P[:, present] = m.predict_proba(Te["xf"]); XG.append(P)
    m.save_model(MOD / f"xgb_{name}.json")
    XMETA[name] = {"classes": list(cl), "present_class_index": present.tolist(), "best_iteration": int(m.best_iteration)}
print("XGB done", flush=True)

CN = [sm(x) for x in Te["cnn"]]
EN = [0.6 * x + 0.4 * f for x, f in zip(XG, FU)]
PRED = {"CNN (joint head)": L[CN[0].argmax(1)], "CNN factorized": L[score(CN).argmax(1)], "FUSION factorized": L[score(FU).argmax(1)],
        "XGB standalone factorized": L[score(XG).argmax(1)], "ENSEMBLE a=0.6": L[score(EN).argmax(1)]}


def metr(p):
    r = {"n": len(t), "acc": np.mean(p == t), "macroF1": f1_score(t, p, average="macro", labels=labels, zero_division=0),
         "wF1": f1_score(t, p, average="weighted", labels=labels, zero_division=0)}
    for i, c in enumerate("Zpc"):
        a_, b_ = [x[i] for x in t], [x[i] for x in p]
        r[f"{c}_acc"] = np.mean(np.array(a_) == np.array(b_)); r[f"{c}_mF1"] = f1_score(a_, b_, average="macro", zero_division=0)
    return r


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
joblib.dump({"model": lr, "mean": mu, "std": sdv, "signals": NAMES}, MOD / "conf_head_logreg.joblib")
conf = lr.predict_proba((signals(EN, XG, FU) - mu) / sdv)[:, 1]
ok = (PRED["ENSEMBLE a=0.6"] == t).astype(float)

json.dump({"labels": labels, "Z_classes": Z_CL, "p_classes": P_CL, "c_classes": C_CL,
           "xgb_features": list(ev["FEAT"]) + ["extent_max_deg"], "xgb_params": PARAMS, "xgb_components": XMETA,
           "fusion": {"seeds": [7000, 7001, 7002], "d_emb": 528, "n_descriptors": int(ev["KF"]), "class_def": "Fusion in eval_variants.py",
                      "cnn": "train_audit/final_lock_A/fold0/best_model.pt"},
           "ensemble_alpha_xgb": 0.6, "factorized_weight": 0.5,
           "confidence_head": "logistic regression on the 14 standardized signals in conf_head_logreg.joblib"},
          open(MOD / "models_meta.json", "w"), indent=2)

print("\n== Comparison with the saved locked-test results ==")
old = json.load(open(LOCK / "summary_test_lock.json"))
P0 = pd.read_csv(LOCK / "test_lock_predictions.csv").set_index("stem").loc[Te["stem"]]
for k, p in PRED.items():
    r = metr(p)
    d = max(abs(float(r[kk]) - float(old[k][kk])) for kk in old[k])
    print(f"{k:<27} acc {r['acc']:.4f} (old {old[k]['acc']:.4f}) | Zpc {r['Z_acc']:.4f}/{r['p_acc']:.4f}/{r['c_acc']:.4f} "
          f"| max|diff| all metrics {d:.2e} | per-crop agreement {np.mean(P0[k].to_numpy() == p):.4f}")
print(f"confidence: max|diff| vs saved {np.abs(P0['confidence'].to_numpy() - conf).max():.2e} | "
      f"AUROC wrong predictions {roc_auc_score(1 - ok, -conf):.4f}")
print(f"\nsaved models: {MOD}")
for f in sorted(MOD.iterdir()):
    print(f"  {f.stat().st_size:>10}  {f.name}")
