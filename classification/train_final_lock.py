#!/usr/bin/env python3
"""Final variant-A CNN (WD 5e-2, head dropout 0.6) on the v7 pool, test = locked test (1394 crops, 35 classes).
Split from split_final_lock.csv: test_lock -> test; pool -> train/val (SGKF 6 per group, same seed); 'excluded' not used.
After training: dump the 528-d embedding + logits of the 4 heads for train/val/test (identical split)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold as RealSGKF

import train_audit_native as ta
from hmi_umbra import paths

R = Path(paths.get("cls_root"))
SPLIT = pd.read_csv(R / "opt_v5/final_lock/split_final_lock.csv")
SET = dict(zip(SPLIT.stem, SPLIT.set_))
OUTROOT = R / "train_audit/final_lock_A"
OUTROOT.mkdir(parents=True, exist_ok=True)
v7 = pd.read_csv(R / "train_audit/v7_relabel_A/export_selected_v7.csv", dtype={"date": str})
data = v7[v7.stem.map(SET).isin(["pool", "test_lock"])]
data.to_csv(OUTROOT / "export_final_lock.csv", index=False)
print(f"final data: {len(data)} crops | test_lock {int((data.stem.map(SET) == 'test_lock').sum())} | pool {int((data.stem.map(SET) == 'pool').sum())}", flush=True)


class LockSGKF:
    def __init__(self, n_splits, shuffle=True, random_state=None):
        self.n = n_splits; self.real = RealSGKF(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X, y, groups):
        if self.n == ta.N_FOLDS:
            is_test = X["stem"].map(SET).eq("test_lock").to_numpy()
            yield np.where(~is_test)[0], np.where(is_test)[0]
        else:
            yield from self.real.split(X, y, groups)


ta.StratifiedGroupKFold = LockSGKF
ta.WEIGHT_DECAY = 5e-2
Base = ta.ResNet18GNMasked


class ModelA(Base):
    def __init__(self, n_classes, meta_dim=2):
        super().__init__(n_classes, meta_dim=meta_dim)
        self.head_full[2].p = 0.6


ta.ResNet18GNMasked = ModelA
ta.EXPORT = OUTROOT / "export_final_lock.csv"
ta.MIN_SAMPLES = 40
if not (OUTROOT / "fold0" / "summary.json").exists():
    sys.argv = [sys.argv[0], "--fold", "0", "--outroot", str(OUTROOT)]
    ta.main()
torch.cuda.empty_cache()

# ---- dump embeddings with the identical split ----
rd = OUTROOT / "fold0"
out = R / "opt_v5/final_lock/emb"
out.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(rd / "manifest_used.csv", dtype={"date": str})
ck = torch.load(rd / "best_model.pt", map_location="cpu", weights_only=False)
l2i = ck["label_to_idx"]; labels = sorted(l2i, key=l2i.get)
is_test = df["stem"].map(SET).eq("test_lock").to_numpy()
rest_idx, test_idx = np.where(~is_test)[0], np.where(is_test)[0]
rest = df.iloc[rest_idx]
tr_rel, va_rel = next(RealSGKF(n_splits=ta.N_VAL_SPLITS, shuffle=True, random_state=ta.SEED).split(rest, rest["final_label"], groups=rest["group_id"]))
parts = {"train": df.iloc[rest_idx[tr_rel]], "val": df.iloc[rest_idx[va_rel]], "test": df.iloc[test_idx]}
print({k: len(v) for k, v in parts.items()}, flush=True)
device = torch.device("cuda")
model = Base(len(labels)).to(device)
model.load_state_dict(ck["model_state_dict"]); model.eval()
for name, d in parts.items():
    d = d.reset_index(drop=True)
    y = np.array([l2i[x] for x in d["final_label"]])
    dl = DataLoader(ta.AugDataset(d, l2i, ck["meta_mean"], ck["meta_std"]),
                    batch_sampler=ta.AugBucketSampler(list(zip(d.h_in, d.w_in)), y, None, False),
                    collate_fn=ta.collate, num_workers=ta.NUM_WORKERS, pin_memory=True)
    order = [i for b in dl.batch_sampler for (i, _) in b]
    Lh = {"joint": [], "z": [], "p": [], "c": [], "emb": []}
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
            for key, head in (("joint", model.head_full), ("z", model.head_z), ("p", model.head_p), ("c", model.head_c)):
                Lh[key].append(head(h).cpu())
            Lh["emb"].append(h.cpu())
    dd = d.iloc[order].reset_index(drop=True)
    np.savez(out / f"{name}_outputs.npz", stem=dd["stem"].to_numpy(), group_id=dd["group_id"].to_numpy(),
             y=np.array([l2i[x] for x in dd["final_label"]]), audit_status=dd["audit_status"].to_numpy(),
             labels=np.array(labels), **{kk: torch.cat(v).numpy() for kk, v in Lh.items()})
    print(f"{name}: {len(dd)} crops saved", flush=True)
print("FINAL CNN + DUMP DONE", flush=True)
