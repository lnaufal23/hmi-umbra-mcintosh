#!/usr/bin/env python3
"""Evaluate the production U-Net (<seg_runs>/unet_resnet34/best_model.pt) on its original split
with the preprocessing of train_seg_crops.py (CropSegDataset augment=False, pad 256 fill 245,
3 channels), Section 4.1.
Subsets: full validation, full test, date-clean test (dates absent from training), and test crops
at least 2 / 3 days from any training date. mIoU is computed from the confusion matrix (as in
training) and reported with and without the padding pixels."""
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_seg_crops as T
from hmi_umbra import paths

RUN = os.path.join(paths.get("seg_runs"), "unet_resnet34")
OUT = f"{RUN}/eval_dateclean.csv"
CLS = T.CLASS_NAMES


class EvalDS(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.base = T.CropSegDataset(self.df.to_dict("records"), augment=False)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        b = self.base[i]
        from PIL import Image
        h, w = np.array(Image.open(self.df.loc[i, "crop_png"]).convert("L")).shape
        valid = T.pad_to_square(np.ones((h, w), np.uint8), fill=0, target=T.PAD_SIZE)
        return b["image"], b["mask"], torch.from_numpy(valid.astype(bool))


def conf_metrics(cm):
    out, ious = {}, []
    for c in range(3):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        iou = tp / (tp + fp + fn + 1e-8); ious.append(iou); out[f"iou_{CLS[c]}"] = round(float(iou), 4)
    out["mIoU"] = round(float(np.mean(ious)), 4)
    return out


@torch.no_grad()
def evaluate(model, df, dev):
    cm_all = np.zeros((3, 3), np.int64); cm_val = np.zeros((3, 3), np.int64)
    for img, msk, valid in DataLoader(EvalDS(df), batch_size=32, shuffle=False, num_workers=6):
        pred = model(img.to(dev)).argmax(1).cpu()
        for ct in range(3):
            for cp in range(3):
                hit = (msk == ct) & (pred == cp)
                cm_all[ct, cp] += int(hit.sum()); cm_val[ct, cp] += int((hit & valid).sum())
    return conf_metrics(cm_all), conf_metrics(cm_val)


def main():
    dev = torch.device("cuda:0")
    model = T.build_model("unet").to(dev)
    model.load_state_dict(torch.load(f"{RUN}/best_model.pt", map_location=dev))
    model.eval()

    sp = {n: pd.read_csv(f"{RUN}/{n}_split.csv") for n in ["train", "val", "test"]}
    import os
    for n, d in sp.items():
        d["crop_png"] = d["crop_png"].map(lambda x: x if os.path.exists(x) else x.replace(paths.get("seg_crops"), paths.get("seg_crops_split")))
        for c in ("crop_png", "mask_png"):
            miss = ~d[c].map(os.path.exists)
            print(f"[CHECK] {n}.{c}: {int(miss.sum())}/{len(d)} missing", d.loc[miss, c].head(2).tolist())
            assert not miss.any(), "missing files, evaluation aborted"
    for d in sp.values():
        d["date"] = pd.to_datetime(d["stem"].astype(str).str.extract(r"(\d{8})_")[0], format="%Y%m%d")
    tr_days = np.array(sorted(sp["train"]["date"].unique()), dtype="datetime64[D]")
    te = sp["test"].copy()
    td = te["date"].values.astype("datetime64[D]")
    pos = np.searchsorted(tr_days, td)
    gap = np.full(len(td), np.inf)
    for k in (pos - 1, pos):
        ok = (k >= 0) & (k < len(tr_days))
        gap[ok] = np.minimum(gap[ok], np.abs((td[ok] - tr_days[k[ok]]).astype(int)))
    te["gap_days"] = gap

    subsets = {
        "full validation": sp["val"],
        "full test": te,
        "date-clean test": te[te["gap_days"] >= 1],
        "test gap >=2 days": te[te["gap_days"] >= 2],
        "test gap >=3 days": te[te["gap_days"] >= 3],
    }
    rows = []
    for name, df in subsets.items():
        if len(df) == 0:
            print(f"{name:22s}: 0 crops, skipped"); continue
        m_all, m_valid = evaluate(model, df, dev)
        n_dates = df["date"].nunique()
        for scope, m in (("with padding", m_all), ("without padding", m_valid)):
            rows.append({"subset": name, "scope": scope, "n_crop": len(df), "n_dates": n_dates, **m})
        print(f"{name:22s}: {len(df):5d} crops {n_dates:5d} days | mIoU {m_all['mIoU']:.4f} (umb {m_all['iou_umbra']:.4f} pen {m_all['iou_penumbra']:.4f} pho {m_all['iou_photosphere']:.4f})"
              f" | no-pad mIoU {m_valid['mIoU']:.4f} (umb {m_valid['iou_umbra']:.4f} pen {m_valid['iou_penumbra']:.4f} pho {m_valid['iou_photosphere']:.4f})", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\ndistribution of the distance from test to the nearest training date (days): {te['gap_days'].value_counts().sort_index().to_dict()}")
    print(f"[WRITE] {OUT}")


if __name__ == "__main__":
    main()
