#!/usr/bin/env python3
"""
Crop active regions from the 4096 x 4096 PSF+LD FITS using the final CVAT annotations (AR_POS_2012_2026).
Two versions per box (native size, no padding, no resizing):
  fits/<year>/<YYYYMMDD>_bbNN.fits : float32 I/I_qs + metadata header
  jpg/<year>/<YYYYMMDD>_bbNN.jpg   : 8-bit opt_F (identical to correction/make_optf.py), q=100
                                     + <YYYYMMDD>_bbNN.json sidecar
Orientation: data[y1:y2, x1:x2] without flipping (same as the opt_F PNG conversion).
--test N : process N dates + verify opt_F 4096 -> BOX/4 against the 1024 PNG used for annotation.
"""
import os
import sys
import csv
import json
import time
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from astropy.io import fits
from PIL import Image

from hmi_umbra import paths

ANN_CSV = os.path.join(paths.get("cls_root"), "final_annotations/ar_pos_final_2012_2026.csv")
IMG1024_ROOT = paths.get("ar_png_1024")
OUT_ROOT = os.path.join(paths.get("cls_root"), "crops_4096")
JPG_QUALITY = 100

# opt_F normalization parameters (identical to correction/make_optf.py)
QS_PERCENTILE_LO = 40
QS_PERCENTILE_HI = 70
PIECEWISE_BREAKS = [0.0, 0.30, 0.85, 1.05]
PIECEWISE_OUTPUTS = [0, 30, 200, 235]
OFF_DISK_VALUE = 245
DISK_MASK_SHRINK_FACTOR = 0.97


def compute_solar_disk_mask(data, header):
    h, w = data.shape
    crpix1 = float(header.get("CRPIX1", w / 2))
    crpix2 = float(header.get("CRPIX2", h / 2))
    cdelt = float(header.get("CDELT1", 0.504))
    rsun_obs = float(header.get("RSUN_OBS", 960.0))
    rsun_pix = rsun_obs / cdelt
    yy, xx = np.indices(data.shape)
    r = np.sqrt((xx - (crpix1 - 1)) ** 2 + (yy - (crpix2 - 1)) ** 2)
    return r <= (rsun_pix * DISK_MASK_SHRINK_FACTOR)


def compute_quiet_sun_median(data, disk_mask):
    disk_pix = data[disk_mask]
    p_low = np.percentile(disk_pix, QS_PERCENTILE_LO)
    p_high = np.percentile(disk_pix, QS_PERCENTILE_HI)
    quiet = disk_pix[(disk_pix >= p_low) & (disk_pix <= p_high)]
    return float(np.median(quiet))


def norm_F_piecewise(data, qs_median, disk_mask):
    ratio = data / max(qs_median, 1e-6)
    breaks = np.array(PIECEWISE_BREAKS, dtype=np.float32)
    outputs = np.array(PIECEWISE_OUTPUTS, dtype=np.float32)
    out = np.interp(ratio, breaks, outputs, left=0, right=outputs[-1])
    bright_mask = ratio > breaks[-1]
    if bright_mask.any():
        extra = (ratio[bright_mask] - breaks[-1]) / 0.15
        extra = np.clip(extra, 0, 1) * (255 - outputs[-1])
        out[bright_mask] = outputs[-1] + extra
    out[~disk_mask] = OFF_DISK_VALUE
    return np.clip(out, 0, 255).astype(np.uint8)


def load_fits(fp):
    with fits.open(fp, memmap=False) as hdul:
        hdu = None
        for h in hdul:
            if h.data is not None and h.data.ndim == 2 and h.data.shape[0] >= 1024:
                hdu = h
                break
        if hdu is None:
            raise ValueError("no valid 2D HDU")
        data = hdu.data.astype(np.float32)
        header = hdu.header.copy()
    if np.isnan(data).any():
        disk_pre = compute_solar_disk_mask(data, header)
        fill = float(np.nanmedian(data[disk_pre])) if disk_pre.any() else float(np.nanmedian(data))
        data = np.where(np.isnan(data), fill, data)
    return data, header


def process_date(task):
    t0 = time.time()
    date, year, fp, boxes, verify = task["date"], task["year"], task["fits_path"], task["boxes"], task["verify"]
    fdir = os.path.join(OUT_ROOT, "fits", str(year))
    jdir = os.path.join(OUT_ROOT, "jpg", str(year))
    os.makedirs(fdir, exist_ok=True)
    os.makedirs(jdir, exist_ok=True)

    todo = []
    for b in boxes:
        stem = f"{date}_bb{int(b['box_id']):02d}"
        paths = (os.path.join(fdir, stem + ".fits"), os.path.join(jdir, stem + ".jpg"), os.path.join(jdir, stem + ".json"))
        if not verify and all(os.path.exists(p) for p in paths):
            continue
        todo.append((stem, paths, b))
    if not todo and not verify:
        return {"date": date, "status": "SKIP", "n_saved": 0, "index": [], "elapsed": 0.0}

    try:
        data, header = load_fits(fp)
        disk = compute_solar_disk_mask(data, header)
        if not disk.any():
            raise ValueError("disk mask empty")
        qs = compute_quiet_sun_median(data, disk)
        if qs <= 0 or not np.isfinite(qs):
            raise ValueError(f"invalid qs_median={qs}")
        optf = norm_F_piecewise(data, qs, disk)
    except Exception as e:
        return {"date": date, "status": "FAIL", "error": f"{type(e).__name__}: {e}", "n_saved": 0, "index": [],
                "elapsed": round(time.time() - t0, 2)}

    res = {"date": date, "status": "OK", "n_saved": 0, "index": [], "qs_median": qs}
    if verify:
        ref_p = os.path.join(IMG1024_ROOT, str(year), f"{date}.png")
        ref = np.array(Image.open(ref_p).convert("L")).astype(np.int16)
        down = np.array(Image.fromarray(optf).resize((1024, 1024), Image.BOX)).astype(np.int16)
        d = np.abs(down - ref)
        flip = np.abs(np.array(Image.fromarray(optf[::-1]).resize((1024, 1024), Image.BOX)).astype(np.int16) - ref)
        res["verify"] = {"max_abs": int(d.max()), "mean_abs": float(d.mean()), "frac_gt1": float((d > 1).mean()),
                         "mean_abs_if_flipud": float(flip.mean())}

    ratio = data / qs
    for stem, (pf, pj, pjs), b in todo:
        x1, y1, x2, y2 = int(b["x1_4096"]), int(b["y1_4096"]), int(b["x2_4096"]), int(b["y2_4096"])
        crop_f = ratio[y1:y2, x1:x2].astype(np.float32)
        crop_j = optf[y1:y2, x1:x2]
        hdr = fits.Header()
        hdr["DATE_OBS"] = (str(header.get("DATE-OBS", header.get("T_OBS", date))), "from source FITS")
        hdr["AR_DATE"] = date
        hdr["AR_BOX"] = int(b["box_id"])
        hdr["AR_ZCLS"] = (b["label"], "Zurich class, CVAT annotation")
        hdr["X1_4096"] = x1; hdr["Y1_4096"] = y1; hdr["X2_4096"] = x2; hdr["Y2_4096"] = y2
        hdr["X1_1024"] = float(b["x1_1024"]); hdr["Y1_1024"] = float(b["y1_1024"])
        hdr["X2_1024"] = float(b["x2_1024"]); hdr["Y2_1024"] = float(b["y2_1024"])
        hdr["QS_MED"] = (qs, "quiet-sun median (40-70th pct) DN")
        hdr["BUNIT"] = ("I/I_qs", "intensity relative to quiet Sun")
        for k in ("CRPIX1", "CRPIX2", "CDELT1", "CDELT2", "RSUN_OBS", "CROTA2", "CRLN_OBS", "CRLT_OBS", "DSUN_OBS"):
            if k in header:
                hdr["SRC_" + k[:4] if len(k) > 4 else k] = header[k]
        hdr["SRC_FITS"] = os.path.basename(fp)[:68]
        n_off = int((~disk[y1:y2, x1:x2]).sum())
        hdr["N_OFFDSK"] = (n_off, "pixels outside 0.97 R_sun")
        tmp = pf + ".tmp"
        fits.PrimaryHDU(data=crop_f, header=hdr).writeto(tmp, overwrite=True)
        os.replace(tmp, pf)
        tmp = pj + ".tmp"
        Image.fromarray(crop_j, mode="L").save(tmp, format="JPEG", quality=JPG_QUALITY)
        os.replace(tmp, pj)
        side = {k: b[k] for k in b}
        side.update({"stem": stem, "qs_median": qs, "n_offdisk_pix": n_off,
                     "crop_native_size_wh": [x2 - x1, y2 - y1], "fits_crop": pf, "jpg_crop": pj,
                     "source_fits": fp, "jpg_quality": JPG_QUALITY,
                     "norm": {"method": "F_piecewise_qsanchored", "breaks": PIECEWISE_BREAKS,
                              "outputs": PIECEWISE_OUTPUTS, "off_disk": OFF_DISK_VALUE}})
        tmp = pjs + ".tmp"
        with open(tmp, "w") as f:
            json.dump(side, f, indent=2, default=str)
        os.replace(tmp, pjs)
        res["index"].append({"stem": stem, "year": year, "date": date, "box_id": int(b["box_id"]), "label": b["label"],
                             "x1_4096": x1, "y1_4096": y1, "x2_4096": x2, "y2_4096": y2,
                             "w_4096": x2 - x1, "h_4096": y2 - y1, "qs_median": round(qs, 2),
                             "n_offdisk_pix": n_off, "fits_crop": pf, "jpg_crop": pj})
        res["n_saved"] += 1
    res["elapsed"] = round(time.time() - t0, 2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--test", type=int, default=0, help="process N dates + verification against the 1024 PNG")
    a = ap.parse_args()

    by_date = defaultdict(list)
    meta = {}
    for r in csv.DictReader(open(ANN_CSV)):
        by_date[r["date"]].append(r)
        meta[r["date"]] = (int(r["year"]), r["fits_path"])
    dates = sorted(by_date)
    if a.test:
        idx = np.linspace(0, len(dates) - 1, a.test).astype(int)
        dates = [dates[i] for i in idx]
    tasks = [{"date": d, "year": meta[d][0], "fits_path": meta[d][1], "boxes": by_date[d], "verify": bool(a.test)}
             for d in dates]
    print(f"Dates: {len(tasks)} | boxes: {sum(len(t['boxes']) for t in tasks)} | workers {a.workers} | "
          f"mode {'TEST+VERIFY' if a.test else 'PRODUCTION'}", flush=True)

    os.makedirs(OUT_ROOT, exist_ok=True)
    idx_fn = os.path.join(OUT_ROOT, "crop_index_test.csv" if a.test else "crop_index.csv")
    idx_new = not os.path.exists(idx_fn) or a.test
    fi = open(idx_fn, "w" if a.test else "a", newline="")
    iw = None
    t0, n_ok, n_fail, n_saved, n_skip = time.time(), 0, 0, 0, 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(process_date, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r["status"] == "FAIL":
                n_fail += 1
                print(f"FAILED {r['date']}: {r['error']}", flush=True)
                continue
            if r["status"] == "SKIP":
                n_skip += 1
            else:
                n_ok += 1
            n_saved += r["n_saved"]
            if r["index"]:
                if iw is None:
                    iw = csv.DictWriter(fi, fieldnames=list(r["index"][0].keys()))
                    if idx_new:
                        iw.writeheader()
                iw.writerows(r["index"])
                fi.flush()
            if "verify" in r:
                v = r["verify"]
                print(f"[verify] {r['date']} | qs {r['qs_median']:.1f} | max|diff| {v['max_abs']} DN | "
                      f"mean {v['mean_abs']:.3f} | frac>1DN {v['frac_gt1']:.5f} | (if flipped: mean {v['mean_abs_if_flipud']:.1f})",
                      flush=True)
            if i % 100 == 0 or i == len(tasks):
                el = time.time() - t0
                print(f"{i}/{len(tasks)} dates | new crops {n_saved} | skip {n_skip} | failed {n_fail} | "
                      f"{el/i:.2f} s/date | ETA {(len(tasks)-i)*el/i/60:.1f} min", flush=True)
    fi.close()
    print(f"DONE. new crops {n_saved} | dates OK {n_ok} | skip {n_skip} | failed {n_fail} | index {idx_fn}", flush=True)


if __name__ == "__main__":
    main()
