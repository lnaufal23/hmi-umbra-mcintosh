#!/usr/bin/env python3
"""Extraction of continuous morphological descriptors (no hard thresholds) for all audited v5 crops (kept/changed).
Definitions identical to the checked overlays: coarse groups = crop /4 INTER_AREA, DN<200, 8-connected components;
fine 4096 scale: umbra DN<92, penumbra 92<=DN<200; largest spot = coarse region with the largest fine-pixel area;
72 rays from the umbral center of the largest spot to the region boundary. Degrees from bar_2p5_px."""
import math
from pathlib import Path
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

from hmi_umbra import paths

R = Path(paths.get("cls_root"))
DATA = R / "train_audit/v5_min40/export_selected_v5.csv"
EXPORT = R / "audit_cvat/export/zpc_audit_export.csv"
OUT = R / "opt_v5/morph/morph_features.csv"
DN_UMB, DN_PEN, COARSE_THR = 92, 200, 200
N_RAYS, MIN_PEN_RUN, MIN_FINE_PX, N_BINS, END_FRAC = 72, 2, 4, 10, 0.2
WORKERS = 12


def axes_of(xs, ys):
    if len(xs) < 3:
        return 0.0, 0.0, 0.0, np.array([1.0, 0.0])
    ev, evec = np.linalg.eigh(np.cov(np.vstack([xs, ys])))
    ev = np.clip(ev, 0, None)
    a, b = 4 * math.sqrt(ev[1]), 4 * math.sqrt(ev[0])
    ecc = math.sqrt(1 - ev[0] / ev[1]) if ev[1] > 0 else 0.0
    return a, b, ecc, evec[:, 1]


def ray_stats(cls, region, cx, cy, ppd):
    h, w = cls.shape
    has, wid, trans = [], [], []
    for k in range(N_RAYS):
        t = 2 * math.pi * k / N_RAYS
        dx, dy = math.cos(t), math.sin(t)
        seq, s = [], 0.0
        while True:
            x, y = int(round(cx + s * dx)), int(round(cy + s * dy))
            if not (0 <= x < w and 0 <= y < h) or not region[y, x]:
                break
            seq.append(cls[y, x]); s += 1.0
        seq = np.array(seq, dtype=np.int8)
        if len(seq) == 0:
            has.append(False); wid.append(0.0); trans.append(0); continue
        u = np.where(seq == 2)[0]
        tail = seq[int(u.max()) + 1:] if len(u) else seq
        run = best = 0
        for v in tail:
            run = run + 1 if v == 1 else 0
            best = max(best, run)
        has.append(best >= MIN_PEN_RUN)
        wid.append(float((tail == 1).sum()) / ppd)
        nz = seq[seq > 0]
        trans.append(int(np.sum(nz[1:] != nz[:-1])) if len(nz) > 1 else 0)
    has = np.array(has)
    gap, run = 0, 0
    for v in np.concatenate([has, has]):          # longest circular gap without penumbra
        run = run + 1 if not v else 0
        gap = max(gap, min(run, N_RAYS))
    w_on = np.array(wid)[has]
    return {"ls_ray_cov": float(has.mean()), "ls_ray_gap_deg_ang": gap * 360.0 / N_RAYS,
            "ls_pen_width_med_deg": float(np.median(w_on)) if len(w_on) else 0.0,
            "ls_pen_width_p10_deg": float(np.percentile(w_on, 10)) if len(w_on) else 0.0,
            "ls_trans_mean": float(np.mean(trans))}


def feat(args):
    stem, path, ppd = args
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    f = {"stem": stem, "ok": 0}
    if img is None or not np.isfinite(ppd) or ppd <= 0:
        return f
    h, w = img.shape
    umb, pen = img < DN_UMB, (img >= DN_UMB) & (img < DN_PEN)
    fine = umb | pen
    cls = np.zeros(img.shape, np.int8); cls[pen] = 1; cls[umb] = 2
    small = cv2.resize(img, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    n, lab_s = cv2.connectedComponents((small < COARSE_THR).astype(np.uint8), connectivity=8)
    lab = cv2.resize(lab_s.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    a2 = ppd ** 2
    fine_cnt = np.bincount((lab * fine).ravel(), minlength=n); fine_cnt[0] = 0
    umb_cnt = np.bincount((lab * umb).ravel(), minlength=n); umb_cnt[0] = 0
    pen_cnt = np.bincount((lab * pen).ravel(), minlength=n); pen_cnt[0] = 0
    regs = np.where(fine_cnt >= MIN_FINE_PX)[0]
    f.update({"ok": 1, "grp_n_regions": len(regs), "grp_n_reg_umb": int((umb_cnt[regs] >= MIN_FINE_PX).sum()),
              "grp_n_reg_pen": int((pen_cnt[regs] >= MIN_FINE_PX * 4).sum()),
              "grp_umb_deg2": umb.sum() / a2, "grp_pen_deg2": pen.sum() / a2})
    ys, xs = np.nonzero(fine)
    ga, gb, _, gvec = axes_of(xs, ys)
    f.update({"grp_major_deg": ga / ppd, "grp_minor_deg": gb / ppd})
    if len(regs) == 0:
        return f
    big = int(regs[np.argmax(fine_cnt[regs])])
    f["grp_frac_largest"] = fine_cnt[big] / max(fine.sum(), 1)
    if len(xs) >= 3:
        proj = (xs - xs.mean()) * gvec[0] + (ys - ys.mean()) * gvec[1]
        lo, hi = proj.min(), proj.max()
        span = max(hi - lo, 1e-6)
        rel = (proj - lo) / span
        is_pen = pen[ys, xs]
        f["grp_pen_endA_deg2"] = float(is_pen[rel <= END_FRAC].sum()) / a2
        f["grp_pen_endB_deg2"] = float(is_pen[rel >= 1 - END_FRAC].sum()) / a2
        f["grp_bins_filled"] = len(np.unique(np.minimum((rel * N_BINS).astype(int), N_BINS - 1))) / N_BINS
        inbig = lab[ys, xs] == big
        f["grp_largest_pos"] = float(rel[inbig].mean()) if inbig.any() else np.nan
    region = lab == big
    m_u, m_p = region & umb, region & pen
    f.update({"ls_umb_deg2": m_u.sum() / a2, "ls_pen_deg2": m_p.sum() / a2,
              "ls_pen_umb_ratio": m_p.sum() / max(m_u.sum(), 1)})
    ry, rx = np.nonzero(region & fine)
    la, lb, lecc, _ = axes_of(rx, ry)
    f.update({"ls_major_deg": la / ppd, "ls_minor_deg": lb / ppd, "ls_axis_ratio": la / max(lb, 1e-6), "ls_ecc": lecc})
    nu, ulab = cv2.connectedComponents(m_u.astype(np.uint8), connectivity=8)
    usz = np.bincount(ulab.ravel())[1:]
    f["ls_n_umb"] = int((usz >= MIN_FINE_PX).sum())
    if m_u.sum() >= MIN_FINE_PX:
        uy, ux = np.nonzero(m_u)
        cx, cy = ux.mean(), uy.mean()
        f.update(ray_stats(cls, region, cx, cy, ppd))
        ub = m_u & ~cv2.erode(m_u.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        nb_ph = cv2.dilate((~fine).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        f["ls_edge_ph_frac"] = (ub & nb_ph).sum() / max(ub.sum(), 1)
        f["ls_umb_offset_norm"] = math.hypot(cx - rx.mean(), cy - ry.mean()) / max(la / 2, 1e-6)
    return f


if __name__ == "__main__":
    d = pd.read_csv(DATA, dtype={"date": str})
    d = d[d.audit_status.isin(["kept", "changed"])]
    ex = pd.read_csv(EXPORT, dtype={"date": str})[["stem", "bar_2p5_px"]]
    d = d.drop(columns=[c for c in ["bar_2p5_px"] if c in d.columns]).merge(ex, on="stem", how="left")
    print(f"crops {len(d)} | without bar_2p5_px: {d.bar_2p5_px.isna().sum()}")
    tasks = [(r.stem, str(r.jpg_crop_abs), float(r.bar_2p5_px) / 2.5) for r in d.itertuples()]
    with Pool(WORKERS) as p:
        rows = p.map(feat, tasks, chunksize=32)
    F = pd.DataFrame(rows)
    F = F.merge(d[["stem", "label_audit", "audit_status"]], on="stem", how="left")
    F.to_csv(OUT, index=False)
    print(f"saved {OUT} | rows {len(F)} | ok {int(F.ok.sum())} | feature columns {F.shape[1] - 4}")
    print("NaN per column (crops without umbra/region):")
    print(F.isna().sum()[F.isna().sum() > 0].to_string())
    num = F.select_dtypes("number").drop(columns=["ok"])
    print("\nfeature summary:")
    print(num.describe(percentiles=[0.1, 0.5, 0.9]).T[["mean", "10%", "50%", "90%"]].round(3).to_string())
    print("\nmedian per p component (check of feature direction):")
    F["p_comp"] = F["label_audit"].astype(str).str[1]
    cols = ["ls_ray_cov", "ls_ray_gap_deg_ang", "ls_pen_width_med_deg", "ls_trans_mean", "ls_edge_ph_frac", "ls_axis_ratio", "ls_pen_deg2", "ls_n_umb"]
    print(F.groupby("p_comp")[cols].median().round(3).to_string())
