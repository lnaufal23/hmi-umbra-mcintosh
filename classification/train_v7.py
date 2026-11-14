#!/usr/bin/env python3
"""Training on v7 (labels after relabeling round R1) with the final variant-A configuration (WD 5e-2, head dropout 0.6), min 40, then dump train/val/test.
Usage: train_v7.py 0 2 4 6"""
import sys
import subprocess
from pathlib import Path

import torch

import train_audit_native as ta
from hmi_umbra import paths

ROOT = Path(paths.get("cls_root"))
OUTROOT = ROOT / "train_audit/v7_relabel_A"
DATA = OUTROOT / "export_selected_v7.csv"
PY = sys.executable
BaseModel = ta.ResNet18GNMasked
ta.WEIGHT_DECAY = 5e-2


class ModelA(BaseModel):
    def __init__(self, n_classes, meta_dim=2):
        super().__init__(n_classes, meta_dim=meta_dim)
        self.head_full[2].p = 0.6


ta.ResNet18GNMasked = ModelA
folds = [int(x) for x in sys.argv[1:]]
print(f"v7 | WD {ta.WEIGHT_DECAY} dropout_head 0.6 | data {DATA} | folds {folds}", flush=True)
for k in folds:
    if not (OUTROOT / f"fold{k}" / "summary.json").exists():
        ta.EXPORT = DATA
        ta.MIN_SAMPLES = 40
        sys.argv = [sys.argv[0], "--fold", str(k), "--outroot", str(OUTROOT)]
        ta.main()
    torch.cuda.empty_cache()
    if not (ROOT / "opt_v5" / OUTROOT.name / f"fold{k}" / "train_outputs.npz").exists():
        subprocess.run([PY, str(Path(__file__).resolve().parent / "dump_all_splits.py"), "--run", str(OUTROOT), str(k)], check=True)
    print(f"[v7 fold {k}] DONE", flush=True)
print("ALL FOLDS DONE v7", flush=True)
