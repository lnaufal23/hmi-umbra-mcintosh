#!/usr/bin/env python3
"""Evaluate hyperparameter variants on the same folds: CNN (joint head, factorized) + fusion head (frozen CNN + 28 descriptors, 3 seeds)
+ overfitting indicator (train-val gap at the best epoch) + McNemar of the variant fusion vs v5.
Usage: eval_variants.py --runs v5_min40 hp_A hp_B --folds 0 1 2
Its definitions (features, Fusion, load, train_fusion) are also reused by fusion_A_def_rule.py, v7_eval.py and final_eval_lock.py."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import binomtest
from sklearn.metrics import f1_score

from hmi_umbra import paths

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--folds", nargs="+", type=int, required=True)
ap.add_argument("--seeds", type=int, default=3)
a = ap.parse_args()
R = Path(paths.get("cls_root"))
MORPH = R / "opt_v5/morph/morph_features.csv"
dev = torch.device("cuda")
EPOCHS, PATIENCE, BS, LR, WD, LS, MOD_DROP = 300, 30, 256, 1e-3, 1e-3, 0.12, 0.3
Z_CL, P_CL, C_CL = list("ABCDEFH"), list("xrsahk"), list("xoic")
LOG_COLS = ["grp_umb_deg2", "grp_pen_deg2", "ls_umb_deg2", "ls_pen_deg2", "ls_pen_umb_ratio",
            "grp_n_regions", "grp_n_reg_umb", "grp_n_reg_pen", "ls_n_umb", "pen_end_min", "pen_end_max"]
LIN_COLS = ["grp_major_deg", "grp_minor_deg", "grp_frac_largest", "grp_bins_filled", "ls_major_deg", "ls_minor_deg",
            "ls_axis_ratio", "ls_ecc", "ls_ray_cov", "ls_ray_gap_deg_ang", "ls_pen_width_med_deg", "ls_pen_width_p10_deg",
            "ls_trans_mean", "ls_edge_ph_frac", "ls_umb_offset_norm", "largest_pos_dev"]
M = pd.read_csv(MORPH)
M["pen_end_min"] = M[["grp_pen_endA_deg2", "grp_pen_endB_deg2"]].min(axis=1)
M["pen_end_max"] = M[["grp_pen_endA_deg2", "grp_pen_endB_deg2"]].max(axis=1)
M["largest_pos_dev"] = (M["grp_largest_pos"] - 0.5).abs()
X = pd.DataFrame({"stem": M["stem"]})
for c in LOG_COLS:
    X[c] = np.log1p(M[c].clip(lower=0))
for c in LIN_COLS:
    X[c] = M[c]
FEAT = LOG_COLS + LIN_COLS
for c in FEAT:
    med, iqr = X[c].median(), (X[c].quantile(0.75) - X[c].quantile(0.25)) or 1.0
    X[c] = ((X[c] - med) / iqr).clip(-5, 5).fillna(0.0)
X["has_umb"] = M["ls_ray_cov"].notna().astype(float)
FEAT += ["has_umb"]
FMAP = dict(zip(X["stem"], X[FEAT].to_numpy(np.float32)))
KF = len(FEAT)


class Fusion(nn.Module):
    def __init__(self, n_cls, d_emb=528):
        super().__init__()
        self.cnn = nn.Sequential(nn.Dropout(0.3), nn.Linear(d_emb, 64), nn.ReLU())
        self.fea = nn.Sequential(nn.Linear(KF, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.mix = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3))
        self.hj, self.hz, self.hp, self.hc = nn.Linear(128, n_cls), nn.Linear(128, 7), nn.Linear(128, 6), nn.Linear(128, 4)
        self.az, self.apz, self.ac = nn.Linear(64, 7), nn.Linear(64, 6), nn.Linear(64, 4)

    def forward(self, e, f, drop_cnn=None):
        hc = self.cnn(e)
        if drop_cnn is not None:
            hc = hc * (~drop_cnn).float().unsqueeze(1)
        hf = self.fea(f)
        h = self.mix(torch.cat([hc, hf], 1))
        return self.hj(h), self.hz(h), self.hp(h), self.hc(h), self.az(hf), self.apz(hf), self.ac(hf)


def load(run, k, part):
    d = np.load(R / "opt_v5" / run / f"fold{k}" / f"{part}_outputs.npz", allow_pickle=True)
    labels = list(d["labels"]); lab = np.array(labels)[d["y"]]
    t = torch.tensor
    return {"stem": d["stem"], "e": t(d["emb"]).float(), "f": t(np.stack([FMAP[s] for s in d["stem"]])), "y": t(d["y"]).long(),
            "z": t([Z_CL.index(l[0]) for l in lab]), "p": t([P_CL.index(l[1]) for l in lab]), "c": t([C_CL.index(l[2]) for l in lab]),
            "labels": labels, "cnn": [d["joint"], d["z"], d["p"], d["c"]]}


def train_fusion(Tr, Va, Te, n_cls, seed):
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
    with torch.no_grad():
        o = m(Te["e"].to(dev), Te["f"].to(dev))
    return [F.softmax(x, 1).cpu().numpy() for x in o[:4]]


sm = lambda x: np.exp(x - x.max(1, keepdims=True)) / np.exp(x - x.max(1, keepdims=True)).sum(1, keepdims=True)
lg = lambda p: np.log(np.clip(p, 1e-12, None))
rows, preds = [], {}
for run in a.runs:
    allp = {"stem": [], "true": [], "cnn_j": [], "cnn_f": [], "fus_j": [], "fus_f": []}
    gaps, bests = [], []
    for k in a.folds:
        Tr, Va, Te = load(run, k, "train"), load(run, k, "val"), load(run, k, "test")
        labels = Tr["labels"]; n_cls = len(labels); L = np.array(labels)
        Zi = np.array([Z_CL.index(l[0]) for l in labels]); Pi = np.array([P_CL.index(l[1]) for l in labels]); Ci = np.array([C_CL.index(l[2]) for l in labels])
        dec = lambda pj, pz, pp, pc: L[(0.5 * lg(pj) + 0.5 * (lg(pz)[:, Zi] + lg(pp)[:, Pi] + lg(pc)[:, Ci])).argmax(1)]
        cp = [sm(x) for x in Te["cnn"]]
        P = [np.zeros_like(x) for x in cp]
        for sd in range(a.seeds):
            out = train_fusion(Tr, Va, Te, n_cls, 1000 * k + sd)
            for i in range(4):
                P[i] += out[i] / a.seeds
        allp["stem"] += list(Te["stem"]); allp["true"] += list(L[Te["y"].numpy()])
        allp["cnn_j"] += list(L[cp[0].argmax(1)]); allp["cnn_f"] += list(dec(*cp))
        allp["fus_j"] += list(L[P[0].argmax(1)]); allp["fus_f"] += list(dec(*P))
        h = pd.read_csv(R / "train_audit" / run / f"fold{k}" / "training_history.csv")
        bi = h["val_f1_macro"].idxmax()
        gaps.append(h.loc[bi, "train_acc"] - h.loc[bi, "val_acc"]); bests.append(int(h.loc[bi, "epoch"]))
        print(f"{run} fold {k} done", flush=True)
    t = np.array(allp["true"])
    r = {"run": run, "best_ep": np.mean(bests), "gap_tr-va": np.mean(gaps)}
    for key in ("cnn_j", "cnn_f", "fus_j", "fus_f"):
        p = np.array(allp[key])
        r[f"{key}_acc"] = np.mean(p == t)
        r[f"{key}_mF1"] = f1_score(t, p, average="macro", zero_division=0)
    p = np.array(allp["fus_f"])
    for i, c in enumerate("Zpc"):
        r[f"fus_f_{c}"] = np.mean([x[i] == y[i] for x, y in zip(p, t)])
    rows.append(r)
    preds[run] = pd.DataFrame(allp).set_index("stem")

T = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(f"\nfolds {a.folds} | fusion seeds {a.seeds} | n = {len(preds[a.runs[0]])}")
print(T.round(4).to_string(index=False))
base = a.runs[0]
for run in a.runs[1:]:
    b, v = preds[base], preds[run].loc[preds[base].index]
    for name, i in [("joint", None), ("Z", 0), ("p", 1), ("c", 2)]:
        ok = lambda P: (P["fus_f"] == P["true"]) if i is None else (P["fus_f"].str[i] == P["true"].str[i])
        x, y = ok(b), ok(v)
        n01, n10 = int((x & ~y).sum()), int((~x & y).sum())
        print(f"McNemar fusion {run} vs {base} [{name:<5}] only {base} {n01:>4} | only {run} {n10:>4} | p {binomtest(n10, n01 + n10, 0.5).pvalue:.4f}")
out = R / "opt_v5/hp" / f"eval_{'_'.join(a.runs)}_f{''.join(map(str, a.folds))}.csv"
T.to_csv(out, index=False)
print(f"saved: {out}")
