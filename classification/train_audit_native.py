#!/usr/bin/env python3
"""
McIntosh Zpc training on the audited CVAT labels (ZPC_AUDIT_2012_2026), one fold per call.
Recipe:
- Data   : zpc_audit_export.csv; audit_status kept/changed used (dropped & multi_tag removed); min 40 per class.
- Groups : union-find over NOAA number + |dlat|<=3, |dCarr|<=7, |dt|<=15 days; StratifiedGroupKFold(7) + val SGKF(6) per group.
- Images : native rectangular 4096 JPG crops (longest side > 768 downscaled, scale factor recorded).
- Augment: 20 variants (4 flips x rotations 0,-10,-5,+5,+10 deg); classes below the median are oversampled up to
           ~median (max 20x), each copy a different variant; empty rotation corners filled with the photospheric
           median and excluded via the mask.
- Loss   : 0.5 CE joint (weight ~ 1/effective count, cap 20) + 0.2 Z + 0.2 p + 0.2 c, LS 0.12; no balanced sampler.
- Model  : ImageNet-pretrained ResNet18, BatchNorm -> GroupNorm(32), masked GAP, meta = extent_max_deg (z) + scale factor.
- Optim  : AdamW 5e-5, wd 1e-3, cosine 120, no freezing, patience 20 (val macro-F1), dynamic batch of 96*320*320 pixels.
Used as a module by run_folds_v5.py, dump_and_decode.py, aggregate_oof.py and train_final_lock.py.
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import json
import math
import random
import argparse
from pathlib import Path
from datetime import datetime as dt

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from astropy.time import Time
from sunpy.coordinates import sun

from hmi_umbra import paths

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

ROOT = Path(paths.get("cls_root"))
EXPORT = ROOT / "audit_cvat/export/zpc_audit_export.csv"

MAX_SIDE = 768
PIX_BUDGET = 96 * 320 * 320
BUCKET = 32
EPOCHS, PATIENCE = 120, 20
LR, WEIGHT_DECAY, LABEL_SMOOTHING, CW_CAP = 5e-5, 1e-3, 0.12, 20.0
SEED, NUM_WORKERS, MIN_SAMPLES = 42, 6, 40
N_FOLDS, N_VAL_SPLITS = 7, 6
ALPHA_FULL, ALPHA_Z, ALPHA_P, ALPHA_C = 0.5, 0.2, 0.2, 0.2
PHOTO_DN, PHOTO_FALLBACK = 200, 226
GN_GROUPS = 32
DLAT, DCARR, DT = 3.0, 7.0, 15
K_MAX = 20
ANGLES = [0, -10, -5, 5, 10]
FLIPS = [(0, 0), (1, 0), (0, 1), (1, 1)]
VARIANTS = [(fh, fv, a) for a in ANGLES for (fh, fv) in FLIPS]  # variant 0 = (0,0,0)

Z_CLASSES = ["A", "B", "C", "D", "E", "F", "H"]
P_CLASSES = ["x", "r", "s", "a", "h", "k"]
C_CLASSES = ["x", "o", "i", "c"]
Z2I = {c: i for i, c in enumerate(Z_CLASSES)}
P2I = {c: i for i, c in enumerate(P_CLASSES)}
C2I = {c: i for i, c in enumerate(C_CLASSES)}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def rot_shape(h, w, ang):
    if ang == 0:
        return h, w
    r = math.radians(abs(ang))
    return int(math.ceil(h * math.cos(r) + w * math.sin(r))), int(math.ceil(w * math.cos(r) + h * math.sin(r)))


def build_groups(df):
    df = df.copy()
    df["t"] = pd.to_datetime(df["date"], format="%Y%m%d")
    L0 = {d: float(sun.L0(Time(pd.Timestamp(d).isoformat())).deg) for d in sorted(df["date"].unique())}
    df["carr"] = (df["center_lon_deg"] + df["date"].map(L0)) % 360.0
    df["day"] = (df["t"] - df["t"].min()).dt.days
    df["noaa"] = pd.to_numeric(df["chosen_noaa"], errors="coerce")
    n = len(df)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for _, idx in df.dropna(subset=["noaa"]).groupby("noaa").groups.items():
        idx = list(idx)
        for j in idx[1:]:
            union(idx[0], j)
    order = np.argsort(df["day"].values, kind="stable")
    day, lat, carr = df["day"].values, df["center_lat_deg"].values, df["carr"].values
    for ap_ in range(n):
        a = order[ap_]
        for bp in range(ap_ + 1, n):
            b = order[bp]
            if day[b] - day[a] > DT:
                break
            if abs(lat[a] - lat[b]) > DLAT:
                continue
            dc = abs(carr[a] - carr[b]); dc = min(dc, 360 - dc)
            if dc <= DCARR:
                union(a, b)
    df["group_id"] = pd.Series([find(i) for i in range(n)]).astype("category").cat.codes.values
    return df.drop(columns=["t"])


def load_img(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    scale = 1.0
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    photo = img[img > PHOTO_DN]
    fill = float(np.median(photo)) if photo.size >= 50 else float(PHOTO_FALLBACK)
    return img, scale, fill


class AugDataset(Dataset):
    """Index = (row, variant)."""

    def __init__(self, df, l2i, mm, ms):
        self.df = df.reset_index(drop=True)
        self.l2i, self.mm, self.ms = l2i, float(mm), max(float(ms), 1e-6)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, key):
        i, v = key
        r = self.df.iloc[i]
        img, scale, fill = load_img(r["jpg_crop_abs"])
        fh, fv, ang = VARIANTS[v]
        if fh:
            img = img[:, ::-1]
        if fv:
            img = img[::-1, :]
        img = np.ascontiguousarray(img)
        mask = np.ones_like(img, dtype=np.uint8)
        if ang:
            h, w = img.shape
            H2, W2 = rot_shape(h, w, ang)
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, 1.0)
            M[0, 2] += (W2 - w) / 2.0
            M[1, 2] += (H2 - h) / 2.0
            img = cv2.warpAffine(img, M, (W2, H2), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=int(round(fill)))
            mask = cv2.warpAffine(mask, M, (W2, H2), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        lab = str(r["final_label"])
        return {"img": torch.from_numpy(img).float() / 255.0, "mask": torch.from_numpy(mask).float(), "fill": fill / 255.0,
                "meta": torch.tensor([(float(r["extent_max_deg"]) - self.mm) / self.ms, scale], dtype=torch.float32),
                "y": self.l2i[lab], "yz": Z2I[lab[0]], "yp": P2I[lab[1]], "yc": C2I[lab[2]]}


def collate(batch):
    H = max(b["img"].shape[0] for b in batch)
    W = max(b["img"].shape[1] for b in batch)
    x = torch.empty(len(batch), 1, H, W)
    m = torch.zeros(len(batch), 1, H, W)
    for k, b in enumerate(batch):
        h, w = b["img"].shape
        x[k].fill_(b["fill"])
        x[k, 0, :h, :w] = b["img"]
        m[k, 0, :h, :w] = b["mask"]
    x = (x.repeat(1, 3, 1, 1) - MEAN) / STD
    t = lambda key: torch.tensor([b[key] for b in batch], dtype=torch.long)
    return {"image": x, "mask": m, "meta": torch.stack([b["meta"] for b in batch]),
            "y": t("y"), "yz": t("yz"), "yp": t("yp"), "yc": t("yc")}


class AugBucketSampler(Sampler):
    """train: each row is repeated rep[class] times with different variants (variant 0 always included); eval: variant 0 only.
    Batches per shape bucket (multiples of 32) under a pixel budget."""

    def __init__(self, shapes, labels, reps=None, train=False, seed=SEED):
        self.shapes, self.labels, self.reps, self.train, self.seed, self.epoch = shapes, labels, reps, train, seed, 0

    def set_epoch(self, e):
        self.epoch = e

    def _batches(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        items = []
        for i, (h, w) in enumerate(self.shapes):
            r = int(self.reps[self.labels[i]]) if self.train else 1
            vs = [0] + (list(rng.choice(np.arange(1, len(VARIANTS)), size=r - 1, replace=False)) if r > 1 else [])
            for v in vs:
                items.append((i, int(v), rot_shape(h, w, VARIANTS[v][2])))
        keys = {}
        for i, v, (h, w) in items:
            keys.setdefault((math.ceil(h / BUCKET), math.ceil(w / BUCKET)), []).append((i, v))
        batches = []
        for (kh, kw), lst in sorted(keys.items()):
            if self.train:
                rng.shuffle(lst)
            per = max(1, PIX_BUDGET // (kh * BUCKET * kw * BUCKET))
            for s in range(0, len(lst), per):
                batches.append(lst[s:s + per])
        if self.train:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self):
        return len(self._batches())


def bn_to_gn(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            gn = nn.GroupNorm(GN_GROUPS, child.num_features, eps=child.eps, affine=True)
            gn.weight.data.copy_(child.weight.data)
            gn.bias.data.copy_(child.bias.data)
            setattr(module, name, gn)
        else:
            bn_to_gn(child)


class ResNet18GNMasked(nn.Module):
    def __init__(self, n_classes, meta_dim=2):
        super().__init__()
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)
        bn_to_gn(self.backbone)
        self.meta_fc = nn.Sequential(nn.Linear(meta_dim, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU())
        fused = 512 + 16
        self.head_full = nn.Sequential(nn.Linear(fused, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, n_classes))
        self.head_z, self.head_p, self.head_c = nn.Linear(fused, 7), nn.Linear(fused, 6), nn.Linear(fused, 4)

    def forward(self, x, mask, meta):
        f = self.backbone(x)
        mf = F.max_pool2d(mask, kernel_size=32, stride=32, ceil_mode=True)
        hf, wf = f.shape[-2:]
        mf = mf[..., :hf, :wf]
        if mf.shape[-2] < hf or mf.shape[-1] < wf:
            mf = F.pad(mf, (0, wf - mf.shape[-1], 0, hf - mf.shape[-2]))
        feat = (f * mf).sum((2, 3)) / mf.sum((2, 3)).clamp(min=1.0)
        h = torch.cat([feat, self.meta_fc(meta)], 1)
        return self.head_full(h), self.head_z(h), self.head_p(h), self.head_c(h)


def run_epoch(model, loader, opt, ce_full, ce_aux, device, train):
    model.train(train)
    tot, yt, yp = 0.0, [], []
    with torch.set_grad_enabled(train):
        for b in loader:
            x, m, meta = b["image"].to(device), b["mask"].to(device), b["meta"].to(device)
            y, yz, ypp, yc = b["y"].to(device), b["yz"].to(device), b["yp"].to(device), b["yc"].to(device)
            if train:
                opt.zero_grad()
            of, oz, op, oc = model(x, m, meta)
            loss = ALPHA_FULL * ce_full(of, y) + ALPHA_Z * ce_aux(oz, yz) + ALPHA_P * ce_aux(op, ypp) + ALPHA_C * ce_aux(oc, yc)
            if train:
                loss.backward()
                opt.step()
            tot += loss.item() * x.size(0)
            yt += y.cpu().tolist()
            yp += of.argmax(1).cpu().tolist()
    return (tot / len(yt), accuracy_score(yt, yp), f1_score(yt, yp, average="macro", zero_division=0),
            f1_score(yt, yp, average="weighted", zero_division=0), np.array(yt), np.array(yp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--outroot", default=str(ROOT / "train_audit/v1"))
    a = ap.parse_args()
    set_seed(SEED)
    outdir = Path(a.outroot) / f"fold{a.fold}"
    outdir.mkdir(parents=True, exist_ok=True)

    ex = pd.read_csv(EXPORT, dtype={"date": str})
    st = ex["audit_status"].value_counts().to_dict()
    df = ex[ex["audit_status"].isin(["kept", "changed"])].copy()
    df["final_label"] = df["label_audit"].astype(str)
    vc = df["final_label"].value_counts()
    keep = sorted(vc[vc >= MIN_SAMPLES].index)
    dropped_cls = sorted(set(vc.index) - set(keep))
    df = df[df["final_label"].isin(keep)].reset_index(drop=True)
    df = build_groups(df)
    shp = []
    for p in df["jpg_crop_abs"]:
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        h, w = im.shape
        s = min(1.0, MAX_SIDE / max(h, w))
        shp.append((max(1, round(h * s)), max(1, round(w * s))))
    df["h_in"], df["w_in"] = [s[0] for s in shp], [s[1] for s in shp]
    df.to_csv(outdir / "manifest_used.csv", index=False)

    labels = sorted(df["final_label"].unique())
    l2i = {c: i for i, c in enumerate(labels)}
    i2l = {i: c for c, i in l2i.items()}
    print("=" * 90)
    print(f"AUDIT NATIVE GN AUG | export status {st}")
    print(f"rows {len(df)} | classes {len(labels)} | classes < min{MIN_SAMPLES} dropped: {dropped_cls} | groups {df['group_id'].nunique()}")
    print("=" * 90, flush=True)

    sgk = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rest_idx, test_idx = list(sgk.split(df, df["final_label"], groups=df["group_id"]))[a.fold]
    rest = df.iloc[rest_idx]
    tr_rel, va_rel = next(StratifiedGroupKFold(n_splits=N_VAL_SPLITS, shuffle=True, random_state=SEED)
                          .split(rest, rest["final_label"], groups=rest["group_id"]))
    dtr = df.iloc[rest_idx[tr_rel]].reset_index(drop=True)
    dva = df.iloc[rest_idx[va_rel]].reset_index(drop=True)
    dte = df.iloc[test_idx].reset_index(drop=True)
    g_tr, g_va, g_te = set(dtr.group_id), set(dva.group_id), set(dte.group_id)
    assert not (g_tr & g_te) and not (g_tr & g_va) and not (g_va & g_te), "Group leakage!"
    print(f"split {len(dtr)}/{len(dva)}/{len(dte)} | AR {len(g_tr)}/{len(g_va)}/{len(g_te)} | group leakage 0 OK")

    ytr = np.array([l2i[x] for x in dtr["final_label"]])
    cnt = np.bincount(ytr, minlength=len(labels)).astype(float)
    T = float(np.median(cnt[cnt > 0]))
    reps = np.array([min(K_MAX, max(1, int(math.ceil(T / c)))) if c > 0 else 1 for c in cnt])
    eff = cnt * reps
    eff[eff == 0] = 1
    cw = torch.tensor(np.minimum(eff.mean() / eff, CW_CAP), dtype=torch.float32)
    tab = pd.DataFrame({"class": labels, "n_train": cnt.astype(int), "rep": reps, "effective": eff.astype(int),
                        "loss_w": cw.numpy().round(2)})
    tab.to_csv(outdir / "class_balance.csv", index=False)
    print(f"target median T={T:.0f} | effective samples/epoch {int(eff.sum())}")
    print(tab.to_string(index=False), flush=True)

    mm, ms = dtr["extent_max_deg"].mean(), dtr["extent_max_deg"].std()
    yva = np.array([l2i[x] for x in dva["final_label"]])
    yte = np.array([l2i[x] for x in dte["final_label"]])
    mk = lambda d, y, train: DataLoader(
        AugDataset(d, l2i, mm, ms),
        batch_sampler=AugBucketSampler(list(zip(d.h_in, d.w_in)), y, reps, train),
        collate_fn=collate, num_workers=NUM_WORKERS, pin_memory=True)
    dl_tr, dl_va, dl_te = mk(dtr, ytr, True), mk(dva, yva, False), mk(dte, yte, False)

    device = torch.device("cuda")
    model = ResNet18GNMasked(len(labels)).to(device)
    ce_full = nn.CrossEntropyLoss(weight=cw.to(device), label_smoothing=LABEL_SMOOTHING)
    ce_aux = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best, best_ep, bad, hist = -1.0, 0, 0, []
    for ep in range(1, EPOCHS + 1):
        dl_tr.batch_sampler.set_epoch(ep)
        t0 = dt.now()
        trl, tra, trf, _, _, _ = run_epoch(model, dl_tr, opt, ce_full, ce_aux, device, True)
        val, vaa, vaf, vafw, _, _ = run_epoch(model, dl_va, opt, ce_full, ce_aux, device, False)
        sch.step()
        sec = (dt.now() - t0).total_seconds()
        imp = vaf > best
        if imp:
            best, best_ep, bad = vaf, ep, 0
            torch.save({"model_state_dict": model.state_dict(), "label_to_idx": l2i, "meta_mean": float(mm),
                        "meta_std": float(ms), "epoch": ep, "val_f1_macro": float(vaf)}, outdir / "best_model.pt")
        else:
            bad += 1
        hist.append({"epoch": ep, "train_loss": trl, "train_acc": tra, "train_f1_macro": trf, "val_loss": val,
                     "val_acc": vaa, "val_f1_macro": vaf, "val_f1_weighted": vafw, "sec": sec,
                     "mem_GB": torch.cuda.max_memory_allocated() / 1e9})
        print(f"[fold {a.fold}][{ep:03d}] {'*' if imp else ' '} tr_acc={tra:.4f} tr_f1={trf:.4f} va_acc={vaa:.4f} "
              f"va_f1={vaf:.4f} va_wf1={vafw:.4f} bad={bad}/{PATIENCE} {sec:.0f}s mem={hist[-1]['mem_GB']:.1f}GB", flush=True)
        if bad >= PATIENCE:
            print(f"[fold {a.fold}] Early stop")
            break
    pd.DataFrame(hist).to_csv(outdir / "training_history.csv", index=False)

    model.load_state_dict(torch.load(outdir / "best_model.pt", map_location=device, weights_only=False)["model_state_dict"])
    tel, tea, tef, tefw, yt, yp = run_epoch(model, dl_te, opt, ce_full, ce_aux, device, False)
    order = [i for b in dl_te.batch_sampler for (i, _) in b]
    dte_o = dte.iloc[order].reset_index(drop=True)
    yt_l, yp_l = [i2l[int(i)] for i in yt], [i2l[int(i)] for i in yp]
    pred = pd.DataFrame({"stem": dte_o.stem, "group_id": dte_o.group_id, "date": dte_o.date,
                         "audit_status": dte_o.audit_status, "true_label": yt_l, "pred_label": yp_l})
    comp = {}
    for c, pos in (("Z", 0), ("p", 1), ("c", 2)):
        a_, b_ = [s[pos] for s in yt_l], [s[pos] for s in yp_l]
        pred[f"correct_{c}"] = np.array(a_) == np.array(b_)
        comp[c] = {"accuracy": float(accuracy_score(a_, b_)), "f1_weighted": float(f1_score(a_, b_, average="weighted", zero_division=0)),
                   "f1_macro": float(f1_score(a_, b_, average="macro", zero_division=0))}
        (outdir / f"classification_report_component_{c}.txt").write_text(classification_report(a_, b_, zero_division=0, digits=4))
    pred["correct_full"] = pred.true_label == pred.pred_label
    pred.to_csv(outdir / "test_predictions.csv", index=False)
    (outdir / "classification_report.txt").write_text(
        classification_report(yt, yp, labels=list(range(len(labels))), target_names=labels, zero_division=0, digits=4))
    np.savetxt(outdir / "confusion_matrix.txt", confusion_matrix(yt, yp, labels=list(range(len(labels)))), fmt="%d")
    mwf1 = float(np.mean([comp[c]["f1_weighted"] for c in "Zpc"]))
    summ = {"fold": a.fold, "rows": len(df), "n_classes": len(labels), "export_status": st, "dropped_classes": dropped_cls,
            "split": [len(dtr), len(dva), len(dte)], "best_epoch": best_ep, "best_val_f1_macro": best,
            "test_acc": float(tea), "test_f1_weighted": float(tefw), "test_f1_macro": float(tef),
            "component": comp, "mean_component_f1_weighted": mwf1}
    (outdir / "summary.json").write_text(json.dumps(summ, indent=2))
    print(f"[fold {a.fold}] TEST joint acc={tea:.4f} wF1={tefw:.4f} macroF1={tef:.4f} | Z={comp['Z']['accuracy']:.4f} "
          f"p={comp['p']['accuracy']:.4f} c={comp['c']['accuracy']:.4f} | mean wF1={mwf1:.4f} | best_epoch={best_ep}")
    for s_, g in pred.groupby("audit_status"):
        print(f"  audit_status={s_:<8} n={len(g):>5} | joint {g.correct_full.mean():.4f} | Z {g.correct_Z.mean():.4f} "
              f"| p {g.correct_p.mean():.4f} | c {g.correct_c.mean():.4f}")


if __name__ == "__main__":
    main()
