#!/usr/bin/env python3
"""Run several v5 min40 folds (data export_selected_v5.csv, MIN 40) in sequence, then dump OOF outputs + factorized decoding per fold.
Folds that already have summary.json skip training. Parent-process GPU memory is released before the dump subprocess (avoids OOM).
Usage: run_folds_v5.py 1 3 5"""
import gc
import sys
import subprocess
from pathlib import Path

import torch

import train_audit_native as ta
from hmi_umbra import paths

ROOT = Path(paths.get("cls_root"))
DATA = ROOT / "train_audit/v5_min40/export_selected_v5.csv"
OUTROOT = ROOT / "train_audit/v5_min40"
PY = sys.executable
folds = [int(x) for x in sys.argv[1:]]

for k in folds:
    od = OUTROOT / f"fold{k}"
    if (od / "summary.json").exists():
        print(f"[fold {k}] summary.json exists -> skip training", flush=True)
    else:
        ta.EXPORT = DATA
        ta.MIN_SAMPLES = 40
        sys.argv = [sys.argv[0], "--fold", str(k), "--outroot", str(OUTROOT)]
        ta.main()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"[fold {k}] parent GPU memory after release: {torch.cuda.memory_reserved() / 1e9:.2f} GB", flush=True)
    if not (ROOT / "opt_v5/v5_min40" / f"fold{k}" / "test_outputs.npz").exists():
        subprocess.run([PY, str(Path(__file__).resolve().parent / "dump_and_decode.py"), "--run", str(OUTROOT), "--fold", str(k)], check=True)
    print(f"[fold {k}] DONE", flush=True)
print("ALL FOLDS DONE:", folds, flush=True)
