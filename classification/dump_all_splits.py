#!/usr/bin/env python3
"""Save the 528-d embedding + logits of the 4 CNN heads (frozen per-fold checkpoint) for the train/val/test splits (no augmentation).
The split is rebuilt identically (manifest_used.csv, same SGKF seed). Used to train the CNN + morphology fusion head."""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold

import train_audit_native as ta
from hmi_umbra import paths

ap = argparse.ArgumentParser()
ap.add_argument("--run", default=str(Path(paths.get("cls_root")) / "train_audit/v5_min40"))
ap.add_argument("folds", nargs="+", type=int)
a = ap.parse_args()
device = torch.device("cuda")

for k in a.folds:
    rd = Path(a.run) / f"fold{k}"
    out = Path(paths.get("cls_root")) / "opt_v5" / Path(a.run).name / f"fold{k}"
    out.mkdir(parents=True, exist_ok=True)
    ta.set_seed(ta.SEED)
    df = pd.read_csv(rd / "manifest_used.csv", dtype={"date": str})
    ck = torch.load(rd / "best_model.pt", map_location="cpu", weights_only=False)
    l2i = ck["label_to_idx"]; labels = sorted(l2i, key=l2i.get)
    sgk = StratifiedGroupKFold(n_splits=ta.N_FOLDS, shuffle=True, random_state=ta.SEED)
    rest_idx, test_idx = list(sgk.split(df, df["final_label"], groups=df["group_id"]))[k]
    rest = df.iloc[rest_idx]
    tr_rel, va_rel = next(StratifiedGroupKFold(n_splits=ta.N_VAL_SPLITS, shuffle=True, random_state=ta.SEED)
                          .split(rest, rest["final_label"], groups=rest["group_id"]))
    parts = {"train": df.iloc[rest_idx[tr_rel]], "val": df.iloc[rest_idx[va_rel]], "test": df.iloc[test_idx]}
    model = ta.ResNet18GNMasked(len(labels)).to(device)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    for name, d in parts.items():
        d = d.reset_index(drop=True)
        y = np.array([l2i[x] for x in d["final_label"]])
        dl = DataLoader(ta.AugDataset(d, l2i, ck["meta_mean"], ck["meta_std"]),
                        batch_sampler=ta.AugBucketSampler(list(zip(d.h_in, d.w_in)), y, None, False),
                        collate_fn=ta.collate, num_workers=ta.NUM_WORKERS, pin_memory=True)
        order = [i for b in dl.batch_sampler for (i, _) in b]
        L = {"joint": [], "z": [], "p": [], "c": [], "emb": []}
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
                    L[key].append(head(h).cpu())
                L["emb"].append(h.cpu())
        dd = d.iloc[order].reset_index(drop=True)
        np.savez(out / f"{name}_outputs.npz", stem=dd["stem"].to_numpy(), group_id=dd["group_id"].to_numpy(),
                 y=np.array([l2i[x] for x in dd["final_label"]]), audit_status=dd["audit_status"].to_numpy(),
                 labels=np.array(labels), **{kk: torch.cat(v).numpy() for kk, v in L.items()})
        print(f"[fold {k}] {name}: {len(dd)} crops saved", flush=True)
    del model
    torch.cuda.empty_cache()
print("DONE", flush=True)
