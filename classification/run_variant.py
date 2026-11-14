#!/usr/bin/env python3
"""CNN hyperparameter tuning (data and split identical to v5 min40). Per fold: train the CNN variant -> dump train/val/test (fusion evaluated separately).
Variants: A (wd 5e-2 + dropout 0.6), W (wd 1e-1 + dropout 0.7), B (frozen conv1+layer1+layer2), C (mixup a=0.2),
E (LS 0.20 + LR 3e-5). Can be combined, e.g. AC. W overrides A if both are given.
Usage: run_variant.py A 0 1 2"""
import sys
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

import train_audit_native as ta
from hmi_umbra import paths

ROOT = Path(paths.get("cls_root"))
DATA = ROOT / "train_audit/v5_min40/export_selected_v5.csv"
PY = sys.executable
VAR = sys.argv[1]
FOLDS = [int(x) for x in sys.argv[2:]]
OUTROOT = ROOT / f"train_audit/hp_{VAR}"
BaseModel = ta.ResNet18GNMasked

HEAD_DROPOUT = None
if "A" in VAR:
    ta.WEIGHT_DECAY = 5e-2
    HEAD_DROPOUT = 0.6
if "W" in VAR:
    ta.WEIGHT_DECAY = 1e-1
    HEAD_DROPOUT = 0.7
if "E" in VAR:
    ta.LABEL_SMOOTHING = 0.20
    ta.LR = 3e-5


class VarModel(BaseModel):
    def __init__(self, n_classes, meta_dim=2):
        super().__init__(n_classes, meta_dim=meta_dim)
        if HEAD_DROPOUT is not None:
            self.head_full[2].p = HEAD_DROPOUT
        if "B" in VAR:
            for idx in (0, 1, 4, 5):            # conv1, gn1, layer1, layer2
                for p in self.backbone[idx].parameters():
                    p.requires_grad = False


ta.ResNet18GNMasked = VarModel

if "C" in VAR:
    MIX_ALPHA = 0.2

    def run_epoch_mix(model, loader, opt, ce_full, ce_aux, device, train):
        model.train(train)
        tot, yt, yp = 0.0, [], []
        with torch.set_grad_enabled(train):
            for b in loader:
                x, m, meta = b["image"].to(device), b["mask"].to(device), b["meta"].to(device)
                y, yz, ypp, yc = b["y"].to(device), b["yz"].to(device), b["yp"].to(device), b["yc"].to(device)
                if train:
                    opt.zero_grad()
                    lam = float(np.random.beta(MIX_ALPHA, MIX_ALPHA))
                    perm = torch.randperm(x.size(0), device=device)
                    xm = lam * x + (1 - lam) * x[perm]
                    mm = torch.maximum(m, m[perm])
                    metam = lam * meta + (1 - lam) * meta[perm]
                    of, oz, op, oc = model(xm, mm, metam)
                    L = lambda o, t, crit: lam * crit(o, t) + (1 - lam) * crit(o, t[perm])
                    loss = (ta.ALPHA_FULL * L(of, y, ce_full) + ta.ALPHA_Z * L(oz, yz, ce_aux)
                            + ta.ALPHA_P * L(op, ypp, ce_aux) + ta.ALPHA_C * L(oc, yc, ce_aux))
                    loss.backward()
                    opt.step()
                    with torch.no_grad():
                        of = model(x, m, meta)[0]
                else:
                    of, oz, op, oc = model(x, m, meta)
                    loss = (ta.ALPHA_FULL * ce_full(of, y) + ta.ALPHA_Z * ce_aux(oz, yz)
                            + ta.ALPHA_P * ce_aux(op, ypp) + ta.ALPHA_C * ce_aux(oc, yc))
                tot += loss.item() * x.size(0)
                yt += y.cpu().tolist()
                yp += of.argmax(1).cpu().tolist()
        return (tot / len(yt), accuracy_score(yt, yp), f1_score(yt, yp, average="macro", zero_division=0),
                f1_score(yt, yp, average="weighted", zero_division=0), np.array(yt), np.array(yp))

    ta.run_epoch = run_epoch_mix

print(f"VARIANT {VAR} | WD {ta.WEIGHT_DECAY} LR {ta.LR} LS {ta.LABEL_SMOOTHING} | dropout_head={HEAD_DROPOUT} frozen={'B' in VAR} "
      f"mixup={'C' in VAR} | folds {FOLDS} | out {OUTROOT}", flush=True)
for k in FOLDS:
    if not (OUTROOT / f"fold{k}" / "summary.json").exists():
        ta.EXPORT = DATA
        ta.MIN_SAMPLES = 40
        sys.argv = [sys.argv[0], "--fold", str(k), "--outroot", str(OUTROOT)]
        ta.main()
    torch.cuda.empty_cache()
    if not (ROOT / "opt_v5" / OUTROOT.name / f"fold{k}" / "train_outputs.npz").exists():
        subprocess.run([PY, str(Path(__file__).resolve().parent / "dump_all_splits.py"), "--run", str(OUTROOT), str(k)], check=True)
    print(f"[{VAR} fold {k}] DONE", flush=True)
print(f"ALL FOLDS DONE {VAR}", flush=True)
