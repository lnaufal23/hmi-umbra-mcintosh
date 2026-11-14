#!/usr/bin/env python3
"""Final variant-A fusion (frozen hp_A CNN + 28 descriptors, 3 seeds/fold) -> saves fusion probabilities per fold.
Test of rules that take the D/E/F part of Z from the CNN alone:
 R0 = factorized w=0.5 (Z from the fusion)
 R1 = D+E+F mass from the fusion, split among D/E/F by the CNN Z head
 R2 = R1 + w=0 (ignore the fusion joint head) for crops with predicted Z in D/E/F."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import binomtest
from sklearn.metrics import f1_score

sys.argv = [sys.argv[0], "--runs", "hp_A", "--folds", "0"]
import eval_variants as ev          # same Fusion, load, FMAP (the module's evaluation part also runs once for fold 0)
from hmi_umbra import paths

R = Path(paths.get("cls_root"))
OUT = R / "opt_v5/fusion_A"
SEEDS = 3
Z_CL, P_CL, C_CL = ev.Z_CL, ev.P_CL, ev.C_CL
DEF = [Z_CL.index(z) for z in "DEF"]
sm = lambda x: np.exp(x - x.max(1, keepdims=True)) / np.exp(x - x.max(1, keepdims=True)).sum(1, keepdims=True)
lg = lambda p: np.log(np.clip(p, 1e-12, None))

for k in range(7):
    f = OUT / f"fold{k}_fusionA.npz"
    if f.exists():
        continue
    Tr, Va, Te = ev.load("hp_A", k, "train"), ev.load("hp_A", k, "val"), ev.load("hp_A", k, "test")
    n_cls = len(Tr["labels"])
    P = None
    for sd in range(SEEDS):
        out = ev.train_fusion(Tr, Va, Te, n_cls, 1000 * k + sd)
        P = [o / SEEDS for o in out] if P is None else [p + o / SEEDS for p, o in zip(P, out)]
    np.savez(f, stem=Te["stem"], y=Te["y"].numpy(), labels=np.array(Tr["labels"]), pj=P[0], pz=P[1], pp=P[2], pc=P[3],
             cnn_joint=Te["cnn"][0], cnn_z=Te["cnn"][1], cnn_p=Te["cnn"][2], cnn_c=Te["cnn"][3])
    print(f"fold {k} fusion A saved", flush=True)

A = [np.load(OUT / f"fold{k}_fusionA.npz", allow_pickle=True) for k in range(7)]
labels = list(A[0]["labels"]); L = np.array(labels)
cat = lambda key: np.concatenate([a[key] for a in A])
y = cat("y"); tl = L[y]
pj, pz, pp, pc = cat("pj"), cat("pz"), cat("pp"), cat("pc")
cz = sm(cat("cnn_z"))
Zi = np.array([Z_CL.index(l[0]) for l in labels]); Pi = np.array([P_CL.index(l[1]) for l in labels]); Ci = np.array([C_CL.index(l[2]) for l in labels])

pz_r1 = pz.copy()
mass = pz[:, DEF].sum(1, keepdims=True)
pz_r1[:, DEF] = mass * cz[:, DEF] / cz[:, DEF].sum(1, keepdims=True).clip(1e-12)


def decode(pzz, w_vec):
    s = w_vec[:, None] * lg(pj) + (1 - w_vec[:, None]) * (lg(pzz)[:, Zi] + lg(pp)[:, Pi] + lg(pc)[:, Ci])
    return L[s.argmax(1)]


w05 = np.full(len(y), 0.5)
R0 = decode(pz, w05)
R1 = decode(pz_r1, w05)
w_r2 = np.where(np.isin(pz_r1.argmax(1), DEF), 0.0, 0.5)
R2 = decode(pz_r1, w_r2)


def metr(p):
    return {"joint": np.mean(p == tl), "macroF1": f1_score(tl, p, average="macro", labels=labels, zero_division=0),
            "Z": np.mean([a[0] == b[0] for a, b in zip(p, tl)]), "p": np.mean([a[1] == b[1] for a, b in zip(p, tl)]),
            "c": np.mean([a[2] == b[2] for a, b in zip(p, tl)])}


res = {"R0 fusion factorized": metr(R0), "R1 DEF from CNN": metr(R1), "R2 DEF from CNN + w0": metr(R2)}
pd.set_option("display.width", 220)
print(f"\nOOF 7-fold n={len(y)}")
print(pd.DataFrame(res).T.round(4).to_string())
defmask = np.isin(np.array([t[0] for t in tl]), list("DEF"))
print(f"\ncrops with true Z D/E/F: {defmask.sum()}")
for name, p in (("R0", R0), ("R1", R1), ("R2", R2)):
    sub = p[defmask]
    print(f"  {name}: joint {np.mean(sub == tl[defmask]):.4f} | Z {np.mean([a[0] == b[0] for a, b in zip(sub, tl[defmask])]):.4f}")
print("\nF1 per Z class:")
zt = [t[0] for t in tl]
print(pd.DataFrame({"Z": Z_CL, "n": [zt.count(z) for z in Z_CL],
                    **{n: f1_score(zt, [q[0] for q in p], labels=Z_CL, average=None, zero_division=0) for n, p in (("R0", R0), ("R1", R1), ("R2", R2))}}).round(4).to_string(index=False))
print("\nMcNemar vs R0:")
for name, p in (("R1", R1), ("R2", R2)):
    for comp, i in (("joint", None), ("Z", 0)):
        a = (R0 == tl) if i is None else np.array([q[i] == t[i] for q, t in zip(R0, tl)])
        b = (p == tl) if i is None else np.array([q[i] == t[i] for q, t in zip(p, tl)])
        n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
        pv = binomtest(n10, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        print(f"  {name} [{comp:<5}] only R0 {n01:>4} | only {name} {n10:>4} | p {pv:.4f}")
