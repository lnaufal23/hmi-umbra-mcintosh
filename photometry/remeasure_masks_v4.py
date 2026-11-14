#!/usr/bin/env python3
"""
Daily quantities from the full-disk U-Net masks on the clipped PSF+LD product (Sections 3.2, 4.2):
  threshold_timeseries_v4mask.csv : I_qs (DN) and per-class intensity statistics per day
  ratio_iqs_timeseries_v4mask.csv : class minima/maxima relative to I_qs with 180-day means
  spot_count_per_day_v4mask.csv   : daily number of segmented spots (umbrae + penumbrae
                                    without umbra, >= 4 px), split by hemisphere
I_qs is the median of the photospheric pixels (mask 255) within 60 degrees of disk center and
0.97 R_sun between the 40th and 70th percentiles. The final block compares with the previous
(tiled) masks.
"""
from __future__ import annotations
import os, sys, time, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd, cv2
from astropy.io import fits
from PIL import Image
from scipy.ndimage import maximum as ndi_maximum, sum as ndi_sum, center_of_mass

from hmi_umbra import paths

FITS_ROOT = Path(paths.get("psf_ld_fits"))
MASK_ROOT = Path(paths.get("masks_fulldisk"))
OUT_DIR = Path(paths.get("analysis_v4mask")); OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TS, OUT_RATIO, OUT_SPOT = OUT_DIR / "threshold_timeseries_v4mask.csv", OUT_DIR / "ratio_iqs_timeseries_v4mask.csv", OUT_DIR / "spot_count_per_day_v4mask.csv"
HELIO_LIMIT_DEG, DISK_SHRINK, IQR_MULT = 60.0, 0.97, 2.0
QS_LO, QS_HI = 40, 70
UMB, PEN = 0, 127
CONN, MIN_PEN_SIZE, Y_SPLIT = 8, 4, 2048
MA = 180


def zone_mask(data, header):
    H, W = data.shape
    crpix1 = float(header.get("CRPIX1", W / 2)); crpix2 = float(header.get("CRPIX2", H / 2))
    cdelt = float(header.get("CDELT1", 0.504)); rsun_obs = float(header.get("RSUN_OBS", 960.0))
    rsun_pix = rsun_obs / cdelt
    yy, xx = np.ogrid[:H, :W]
    r = np.sqrt((xx - (crpix1 - 1)) ** 2 + (yy - (crpix2 - 1)) ** 2)
    disk = r <= rsun_pix * DISK_SHRINK
    rho = np.sqrt(((xx - (crpix1 - 1)) * cdelt) ** 2 + ((yy - (crpix2 - 1)) * cdelt) ** 2)
    theta = np.degrees(np.arcsin(np.clip(rho / rsun_obs, 0, 1)))
    return disk & (theta <= HELIO_LIMIT_DEG)


def iqr_filter(a):
    if len(a) == 0:
        return a
    p25, p75 = np.percentile(a, [25, 75]); iqr = p75 - p25
    return a[(a >= p25 - IQR_MULT * iqr) & (a <= p75 + IQR_MULT * iqr)]


def stats(a):
    if len(a) == 0:
        return {k: np.nan for k in ["mean", "std", "p10", "p25", "p50", "p75", "p90"]} | {"n": 0}
    p10, p25, p50, p75, p90 = np.percentile(a, [10, 25, 50, 75, 90])
    return {"mean": float(a.mean()), "std": float(a.std()), "p10": float(p10), "p25": float(p25),
            "p50": float(p50), "p75": float(p75), "p90": float(p90), "n": int(len(a))}


def count_spots(mask):
    umbra_bin = (mask == UMB).astype(np.uint8)
    n_lbl, lbl = cv2.connectedComponents(umbra_bin, connectivity=CONN)
    n_umbra = n_lbl - 1; un = us = 0
    if n_umbra > 0:
        for cy, _ in center_of_mass(umbra_bin, labels=lbl, index=list(range(1, n_lbl))):
            un += cy < Y_SPLIT; us += cy >= Y_SPLIT
    pen_bin = (mask == PEN).astype(np.uint8)
    n_pl, pl = cv2.connectedComponents(pen_bin, connectivity=CONN)
    dil = cv2.dilate(umbra_bin, np.ones((3, 3), np.uint8), iterations=1) if n_umbra > 0 else np.zeros_like(umbra_bin)
    raw = filt = pn = ps = 0
    if n_pl > 1:
        idx = np.arange(1, n_pl)
        no_umb = np.asarray(ndi_maximum(dil, labels=pl, index=idx)) == 0
        keep = no_umb & (np.asarray(ndi_sum(pen_bin, labels=pl, index=idx)) >= MIN_PEN_SIZE)
        raw, filt = int(no_umb.sum()), int(keep.sum())
        kidx = idx[keep]
        if len(kidx):
            cents = center_of_mass(pen_bin, labels=pl, index=kidx.tolist())
            if not isinstance(cents, list):
                cents = [cents]
            for cy, _ in cents:
                pn += cy < Y_SPLIT; ps += cy >= Y_SPLIT
    return {"n_umbra": int(n_umbra), "n_umbra_north": int(un), "n_umbra_south": int(us),
            "n_penumbra_only_raw": raw, "n_penumbra_only": filt,
            "n_penumbra_only_north": int(pn), "n_penumbra_only_south": int(ps),
            "n_spot_total": int(n_umbra + filt), "n_spot_total_north": int(un + pn),
            "n_spot_total_south": int(us + ps), "n_spot_total_raw": int(n_umbra + raw)}


def process(mask_path):
    mp = Path(mask_path); stem = mp.name.replace("_pred.png", "")
    try:
        ds = stem.split(".")[2].split("_")[0]; year = int(ds[:4])
    except Exception:
        return {"stem": stem, "error": "parse_date"}, None
    try:
        mask = np.array(Image.open(mp).convert("L"))
        spot = {"stem": stem, "date": f"{ds[:4]}-{ds[4:6]}-{ds[6:8]}", "year": year, **count_spots(mask)}
    except Exception as e:
        return {"stem": stem, "error": f"mask:{e}"[:120]}, None
    fp = FITS_ROOT / str(year) / f"{stem}.continuum.psf_ld.fits"
    if not fp.exists():
        return {"stem": stem, "error": "fits_not_found"}, spot
    try:
        with fits.open(str(fp), memmap=True) as h:
            data = h[0].data.astype(np.float32); header = h[0].header
        if data.shape != mask.shape:
            return {"stem": stem, "error": "shape_mismatch"}, spot
        finite = np.isfinite(data); valid = zone_mask(data, header) & finite
        disk_pix = data[valid & (mask == 255)]
        if len(disk_pix) < 1000:
            return {"stem": stem, "error": "too_few_photosphere"}, spot
        lo, hi = np.percentile(disk_pix, [QS_LO, QS_HI])
        I_qs = float(np.median(disk_pix[(disk_pix >= lo) & (disk_pix <= hi)]))
        u = stats(iqr_filter(data[valid & (mask == UMB)])); p = stats(iqr_filter(data[valid & (mask == PEN)]))
        o = stats(iqr_filter(disk_pix))
        umb_all = data[finite & (mask == UMB)]; pen_all = data[finite & (mask == PEN)]
        row = {"stem": stem, "date": ds, "year": year, "month": int(ds[4:6]), "day": int(ds[6:8]),
               "I_qs": round(I_qs, 2), "T_umbra": round(0.50 * I_qs, 2), "T_penumbra": round(0.85 * I_qs, 2)}
        for pre, s in [("umb", u), ("pen", p)]:
            for k in ["mean", "std", "p10", "p25", "p50", "p75", "p90"]:
                row[f"{pre}_{k}"] = round(s[k], 2)
        row["n_umbra"] = u["n"]; row["n_penumbra"] = p["n"]
        row["pho_mean"] = round(o["mean"], 2); row["pho_std"] = round(o["std"], 2); row["pho_p50"] = round(o["p50"], 2)
        row["n_photosphere"] = o["n"]
        row["ratio_umb_Iqs"] = round(u["mean"] / I_qs, 4); row["ratio_pen_Iqs"] = round(p["mean"] / I_qs, 4)
        row["T_umb_formula"] = row["T_umbra"]; row["T_pen_formula"] = row["T_penumbra"]
        row["umb_min"] = float(umb_all.min()) if len(umb_all) else np.nan
        row["umb_max"] = float(umb_all.max()) if len(umb_all) else np.nan
        row["pen_min"] = float(pen_all.min()) if len(pen_all) else np.nan
        row["pen_max"] = float(pen_all.max()) if len(pen_all) else np.nan
        return row, spot
    except Exception as e:
        return {"stem": stem, "error": f"fits:{e}"[:120]}, spot


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=24); a = ap.parse_args()
    masks = sorted(str(p) for p in MASK_ROOT.glob("*/*_pred.png"))
    print(f"mask v4: {len(masks)} | workers {a.workers}", flush=True)
    ts, spots, errs = [], [], []; t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(process, m) for m in masks]
        for i, f in enumerate(as_completed(futs), 1):
            r, s = f.result()
            if s is not None:
                spots.append(s)
            (errs if "error" in r else ts).append(r)
            if i % 250 == 0:
                print(f"  {i}/{len(masks)} | {time.time()-t0:.0f}s | ts ok {len(ts)} err {len(errs)}", flush=True)
    print(f"done {time.time()-t0:.0f}s | threshold ok {len(ts)} | spot ok {len(spots)} | error {len(errs)}")
    from collections import Counter
    print("error types:", dict(Counter(e["error"].split(":")[0] for e in errs)))
    for e in errs[:10]:
        print("  ", e)

    df = pd.DataFrame(ts).sort_values("date").reset_index(drop=True); df.to_csv(OUT_TS, index=False)
    sp = pd.DataFrame(spots).sort_values("date").reset_index(drop=True); sp.to_csv(OUT_SPOT, index=False)

    r = pd.DataFrame({"date": pd.to_datetime(df["date"].astype(str), format="%Y%m%d"), "I_qs": df["I_qs"]})
    for c in ["umb_min", "umb_max", "pen_min", "pen_max"]:
        r[c] = df[c]; r[f"{c}_ratio"] = df[c] / df["I_qs"]
    for c in ["umb_min", "umb_max", "pen_min", "pen_max"]:
        r[f"{c}_ratio_ma"] = r[f"{c}_ratio"].rolling(MA, center=True, min_periods=10).mean()
    r.to_csv(OUT_RATIO, index=False)

    old = pd.read_csv(os.path.join(paths.get("analysis_legacy_masks"), "ratio_iqs_timeseries.csv"))
    print("\n=== BOUNDARIES (median of the daily ratio to I_qs): previous masks (masks_3class) vs v4 ===")
    for c in ["umb_min_ratio", "umb_max_ratio", "pen_min_ratio", "pen_max_ratio"]:
        print(f"  {c:15s} previous median {old[c].median():.4f} [p5 {old[c].quantile(.05):.4f}, p95 {old[c].quantile(.95):.4f}] | "
              f"v4 median {r[c].median():.4f} [p5 {r[c].quantile(.05):.4f}, p95 {r[c].quantile(.95):.4f}] | v4 mean of MA180 {r[c+'_ma'].mean():.4f}")
    print(f"  I_qs v4 vs previous: median relative difference {((df.set_index('stem')['I_qs'] / pd.read_csv(os.path.join(paths.get('analysis_legacy_masks'), 'threshold_timeseries.csv')).set_index('stem')['I_qs']) - 1).median():+.5f}")
    print(f"\n[WRITE] {OUT_TS}\n[WRITE] {OUT_RATIO}\n[WRITE] {OUT_SPOT}")


if __name__ == "__main__":
    main()
