#!/usr/bin/env python3
"""Evaluation of v7 (relabel R1 data, CNN variant A): fusion with 3 seeds/fold + factorized decoding -> v7 OOF metrics,
then an effect-separation matrix (model A vs v7) x (old vs new labels) on the common crops, and on the relabeled crops."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from hmi_umbra import paths

R = Path(paths.get("cls_root"))
src = open(Path(__file__).resolve().parent / "eval_variants.py").read()
src = src.split("ap = argparse.ArgumentParser()")[0] + "\nR = Path(paths.get('cls_root'))\n" + \
      src.split("a = ap.parse_args()")[1].split("sm = lambda x:")[0]
ev = {}
exec(compile(src, "eval_variants_defs", "exec"), ev)
OUT = R / "opt_v5/fusion_v7"
Z_CL, P_CL, C_CL = list("ABCDEFH"), list("xrsahk"), list("xoic")
lg = lambda p: np.log(np.clip(p, 1e-12, None))
sm = lambda x: np.exp(x - x.max(1, keepdims=True)) / np.exp(x - x.max(1, keepdims=True)).sum(1, keepdims=True)


def decode(labels, pj, pz, pp, pc):
    Zi = np.array([Z_CL.index(l[0]) for l in labels]); Pi = np.array([P_CL.index(l[1]) for l in labels]); Ci = np.array([C_CL.index(l[2]) for l in labels])
    return np.array(labels)[(0.5 * lg(pj) + 0.5 * (lg(pz)[:, Zi] + lg(pp)[:, Pi] + lg(pc)[:, Ci])).argmax(1)]


pred_v7 = {}
for k in range(7):
    f = OUT / f"fold{k}_fusion_v7.npz"
    if not f.exists():
        Tr, Va, Te = ev["load"]("v7_relabel_A", k, "train"), ev["load"]("v7_relabel_A", k, "val"), ev["load"]("v7_relabel_A", k, "test")
        n = len(Tr["labels"]); P = None
        for sd in range(3):
            o = ev["train_fusion"](Tr, Va, Te, n, 1000 * k + sd)
            P = [x / 3 for x in o] if P is None else [p + x / 3 for p, x in zip(P, o)]
        np.savez(f, stem=Te["stem"], y=Te["y"].numpy(), labels=np.array(Tr["labels"]), pj=P[0], pz=P[1], pp=P[2], pc=P[3])
        print(f"fold {k} fusion v7 saved", flush=True)
    a = np.load(f, allow_pickle=True)
    labels = list(a["labels"])
    for s, p in zip(a["stem"], decode(labels, a["pj"], a["pz"], a["pp"], a["pc"])):
        pred_v7[s] = p

pred_A = {}
for k in range(7):
    a = np.load(R / f"opt_v5/fusion_A/fold{k}_fusionA.npz", allow_pickle=True)
    labels = list(a["labels"])
    for s, p in zip(a["stem"], decode(labels, a["pj"], a["pz"], a["pp"], a["pc"])):
        pred_A[s] = p

v5 = pd.read_csv(R / "train_audit/v5_min40/export_selected_v5.csv", dtype={"date": str})
v7 = pd.read_csv(R / "train_audit/v7_relabel_A/export_selected_v7.csv", dtype={"date": str})
old = dict(zip(v5.stem, v5.label_audit.astype(str))); new = dict(zip(v7.stem, v7.label_audit.astype(str)))
relab = set(pd.read_csv(R / "opt_v5/relabel_r1/ZPC_RELABEL_R1_export.csv").query("status_relabel in ['kept','changed']").stem)
changed = set(pd.read_csv(R / "opt_v5/relabel_r1/ZPC_RELABEL_R1_export.csv").query("status_relabel == 'changed'").stem)


def metr(stems, P, Lmap):
    t = np.array([Lmap[s] for s in stems]); p = np.array([P[s] for s in stems])
    labs = sorted(set(t))
    return {"n": len(t), "joint": np.mean(p == t), "macroF1": f1_score(t, p, labels=labs, average="macro", zero_division=0),
            **{c: np.mean([a[i] == b[i] for a, b in zip(p, t)]) for i, c in enumerate("Zpc")}}


pd.set_option("display.width", 220)
own = sorted(pred_v7)
print("\n== Full v7 OOF (new labels, v7 set) ==")
print(pd.DataFrame({"v7": metr(own, pred_v7, new)}).T.round(4).to_string())
common = sorted(set(pred_v7) & set(pred_A))
print(f"\n== Effect-separation matrix on common crops (n={len(common)}) ==")
M = {"A x old labels": metr(common, pred_A, old), "A x new labels": metr(common, pred_A, new),
     "v7 x old labels": metr(common, pred_v7, old), "v7 x new labels": metr(common, pred_v7, new)}
T = pd.DataFrame(M).T.round(4)
print(T.to_string())
d = lambda a, b, c="joint": M[a][c] - M[b][c]
print(f"\neffect of test labels only (A: new - old): joint {d('A x new labels', 'A x old labels'):+.4f} | macroF1 {d('A x new labels', 'A x old labels', 'macroF1'):+.4f}")
print(f"effect of training on clean labels (new labels: v7 - A): joint {d('v7 x new labels', 'A x new labels'):+.4f} | macroF1 {d('v7 x new labels', 'A x new labels', 'macroF1'):+.4f}")
print(f"total (v7 x new - A x old): joint {d('v7 x new labels', 'A x old labels'):+.4f} | macroF1 {d('v7 x new labels', 'A x old labels', 'macroF1'):+.4f}")
for name, S in (("relabeled crops (kept+changed)", relab), ("crops whose label changed", changed), ("crops untouched by relabeling", set(common) - relab)):
    st = sorted(set(common) & S)
    print(f"\n== {name} (n={len(st)}) ==")
    print(pd.DataFrame({k: metr(st, pred_A if k[0] == "A" else pred_v7, old if "old" in k else new) for k in M}).T.round(4).to_string())
