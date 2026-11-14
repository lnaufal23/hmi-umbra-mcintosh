#!/usr/bin/env python3
"""Load best_model.pt of one fold, rebuild the identical split (manifest_used.csv), save the logits of the 4 heads + embeddings (val & test),
then test factorized decoding: w*logp_joint + (1-w)*(logp_Z+logp_p+logp_c) over the training classes. w is chosen on val and applied to test."""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score

import train_audit_native as ta
from hmi_umbra import paths

ap = argparse.ArgumentParser()
ap.add_argument("--run", default=str(Path(paths.get("cls_root")) / "train_audit/v5_min40"))
ap.add_argument("--fold", type=int, default=0)
a = ap.parse_args()
rd = Path(a.run) / f"fold{a.fold}"
out = Path(paths.get("cls_root")) / "opt_v5" / Path(a.run).name / f"fold{a.fold}"
out.mkdir(parents=True, exist_ok=True)
ta.set_seed(ta.SEED)

df = pd.read_csv(rd / "manifest_used.csv", dtype={"date": str})
ck = torch.load(rd / "best_model.pt", map_location="cpu", weights_only=False)
l2i = ck["label_to_idx"]; labels = sorted(l2i, key=l2i.get)
sgk = StratifiedGroupKFold(n_splits=ta.N_FOLDS, shuffle=True, random_state=ta.SEED)
rest_idx, test_idx = list(sgk.split(df, df["final_label"], groups=df["group_id"]))[a.fold]
rest = df.iloc[rest_idx]
tr_rel, va_rel = next(StratifiedGroupKFold(n_splits=ta.N_VAL_SPLITS, shuffle=True, random_state=ta.SEED)
                      .split(rest, rest["final_label"], groups=rest["group_id"]))
dva = df.iloc[rest_idx[va_rel]].reset_index(drop=True)
dte = df.iloc[test_idx].reset_index(drop=True)
print(f"split val {len(dva)} test {len(dte)} | classes {len(labels)}")

device = torch.device("cuda")
model = ta.ResNet18GNMasked(len(labels)).to(device)
model.load_state_dict(ck["model_state_dict"]); model.eval()


def dump(d, name):
    y = np.array([l2i[x] for x in d["final_label"]])
    dl = DataLoader(ta.AugDataset(d, l2i, ck["meta_mean"], ck["meta_std"]),
                    batch_sampler=ta.AugBucketSampler(list(zip(d.h_in, d.w_in)), y, None, False),
                    collate_fn=ta.collate, num_workers=ta.NUM_WORKERS, pin_memory=True)
    order = [i for b in dl.batch_sampler for (i, _) in b]
    LJ, LZ, LP, LC, EMB = [], [], [], [], []
    with torch.no_grad():
        for b in dl:
            x, m, meta = b["image"].to(device), b["mask"].to(device), b["meta"].to(device)
            f = model.backbone(x)
            mf = F.max_pool2d(m, kernel_size=32, stride=32, ceil_mode=True)
            hf, wf = f.shape[-2:]
            mf = mf[..., :hf, :wf]
            if mf.shape[-2] < hf or mf.shape[-1] < wf:
                mf = F.pad(mf, (0, wf - mf.shape[-1], 0, hf - mf.shape[-2]))
            feat = (f * mf).sum((2, 3)) / mf.sum((2, 3)).clamp(min=1.0)
            h = torch.cat([feat, model.meta_fc(meta)], 1)
            LJ.append(model.head_full(h).cpu()); LZ.append(model.head_z(h).cpu())
            LP.append(model.head_p(h).cpu()); LC.append(model.head_c(h).cpu()); EMB.append(h.cpu())
    dd = d.iloc[order].reset_index(drop=True)
    R = {k: torch.cat(v).numpy() for k, v in zip(["joint", "z", "p", "c", "emb"], [LJ, LZ, LP, LC, EMB])}
    np.savez(out / f"{name}_outputs.npz", stem=dd["stem"].to_numpy(), group_id=dd["group_id"].to_numpy(),
             y=np.array([l2i[x] for x in dd["final_label"]]), audit_status=dd["audit_status"].to_numpy(), labels=np.array(labels), **R)
    return dd, R


dva_o, RV = dump(dva, "val")
dte_o, RT = dump(dte, "test")
Zi = np.array([ta.Z2I[l[0]] for l in labels]); Pi = np.array([ta.P2I[l[1]] for l in labels]); Ci = np.array([ta.C2I[l[2]] for l in labels])


def scores(R, w):
    lj = torch.log_softmax(torch.tensor(R["joint"]), 1).numpy()
    lz = torch.log_softmax(torch.tensor(R["z"]), 1).numpy()
    lp = torch.log_softmax(torch.tensor(R["p"]), 1).numpy()
    lc = torch.log_softmax(torch.tensor(R["c"]), 1).numpy()
    return w * lj + (1 - w) * (lz[:, Zi] + lp[:, Pi] + lc[:, Ci])


def metr(d, pred):
    t = d["final_label"].to_numpy(); p = np.array([labels[i] for i in pred])
    return {"joint": np.mean(t == p), "macroF1": f1_score(t, p, average="macro", labels=labels, zero_division=0),
            "Z": np.mean([a[0] == b[0] for a, b in zip(t, p)]), "p": np.mean([a[1] == b[1] for a, b in zip(t, p)]),
            "c": np.mean([a[2] == b[2] for a, b in zip(t, p)])}


rows = []
for w in np.round(np.arange(0, 1.01, 0.1), 2):
    mv = metr(dva_o, scores(RV, w).argmax(1)); mt = metr(dte_o, scores(RT, w).argmax(1))
    rows.append({"w": w, **{f"val_{k}": round(v, 4) for k, v in mv.items()}, **{f"test_{k}": round(v, 4) for k, v in mt.items()}})
T = pd.DataFrame(rows)
T.to_csv(out / "factorized_decoding.csv", index=False)
pd.set_option("display.width", 250)
print(T.to_string(index=False))
wb = T.loc[T.val_macroF1.idxmax(), "w"]
print(f"\nSanity check w=1.0: test joint {T.loc[T.w == 1.0, 'test_joint'].item():.4f} (log: 0.4795)")
print(f"best w on val (macroF1): {wb} -> TEST " + " ".join(f"{k}={T.loc[T.w == wb, 'test_' + k].item():.4f}" for k in ["joint", "macroF1", "Z", "p", "c"]))
tl = dte_o["final_label"]
for c_, key, cls in (("Z", "z", ta.Z_CLASSES), ("p", "p", ta.P_CLASSES), ("c", "c", ta.C_CLASSES)):
    pos = "Zpc".index(c_)
    acc = np.mean([cls[i] == t[pos] for i, t in zip(RT[key].argmax(1), tl)])
    print(f"head {c_} alone (test): {acc:.4f}")
print(f"saved: {out}")
