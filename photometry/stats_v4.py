#!/usr/bin/env python3
"""Per-umbra and per-day tables from the full-disk U-Net masks on the clipped PSF+LD product
(Sections 3.1 and 4.3). I_qs is the median of the on-disk pixels between the 40th and 70th
percentiles within 0.97 R_sun. Per umbra: area (MSH, foreshortening-corrected), median
intensity after removing 1.5*IQR outliers, and darkest pixel. Per day: total umbral+penumbral
area of components with mu >= 0.7 and >= 1 MSH, and the darkest umbral pixel. No day is
excluded; the number of umbrae with a median above 0.503 is stored in n_umb_median_gt_0503.

Usage: python stats_v4.py [WORKERS]
"""
import glob
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from astropy.io import fits
from PIL import Image
from scipy import ndimage

from hmi_umbra import paths

Image.MAX_IMAGE_PIXELS = None
MASK_DIR = paths.get("masks_fulldisk")
FITS_DIR = paths.get("psf_ld_fits")
OUT = paths.get("stats_clipped_out")
MIN_BLOB_PIX, QS_LO, QS_HI, DISK_SHRINK = 4, 40, 70, 0.97
MU_MIN, AREA_MIN_MSH, MEDIAN_ART = 0.7, 1.0, 0.503
UMBRA, PENUMBRA = 0, 127
RX = re.compile(r"(hmi\.ic_720s\.(\d{8})_000000_TAI\.\d+)_pred\.png$")


def load_fits(fp):
    with fits.open(fp) as hdul:
        for h in hdul:
            if h.data is not None and h.data.ndim == 2 and h.data.shape[0] >= 1024:
                return h.data.astype(np.float64), h.header
    raise RuntimeError(f"no 2D HDU in {fp}")


def comps(lab, n, cx, cy, rsun, mpp):
    """(index, sl, sub, npix, mu, area_msh) for every component inside the limb."""
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        sub = lab[sl] == i
        npix = int(sub.sum())
        if npix == 0:
            continue
        ys, xs = np.where(sub)
        cyb, cxb = ys.mean() + sl[0].start, xs.mean() + sl[1].start
        frac = np.hypot(cxb - cx, cyb - cy) / rsun
        if frac >= 0.999:
            continue
        mu = np.sqrt(max(1 - frac ** 2, 1e-6))
        out.append((i, sl, sub, npix, mu, npix * mpp / mu, cxb, cyb))
    return out


def process(args):
    date, mp, fp = args
    try:
        mask = np.array(Image.open(mp)); mask = mask[..., 0] if mask.ndim == 3 else mask
        data, header = load_fits(fp)
        if mask.shape != data.shape:
            mask = np.array(Image.fromarray(mask).resize((data.shape[1], data.shape[0]), Image.NEAREST))
        h, w = data.shape
        cx, cy = float(header.get("CRPIX1", w / 2)) - 1.0, float(header.get("CRPIX2", h / 2)) - 1.0
        rsun = float(header.get("RSUN_OBS", 960.0)) / float(header.get("CDELT1", 0.504))
        yy, xx = np.indices(data.shape)
        px = data[np.hypot(xx - cx, yy - cy) <= rsun * DISK_SHRINK]
        lo, hi = np.percentile(px, QS_LO), np.percentile(px, QS_HI)
        iqs = float(np.median(px[(px >= lo) & (px <= hi)]))
        if not np.isfinite(iqs) or iqs <= 0:
            return [], None
        mpp = 1e6 / (2 * np.pi * rsun ** 2)
        lab_u, nu = ndimage.label(mask == UMBRA)
        lab_p, npn = ndimage.label(mask == PENUMBRA)
        cu, cp = comps(lab_u, nu, cx, cy, rsun, mpp), comps(lab_p, npn, cx, cy, rsun, mpp)
        blobs, umin, n_art = [], np.inf, 0
        for i, sl, sub, npix, mu, a, cxb, cyb in cu:
            vals = data[sl][sub]; vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            if npix >= MIN_BLOB_PIX:
                q1, q3 = np.percentile(vals, [25, 75]); iqr = q3 - q1
                cl = vals[(vals >= q1 - 1.5 * iqr) & (vals <= q3 + 1.5 * iqr)]
                med = float(np.median(cl)) / iqs if cl.size else float(np.median(vals)) / iqs
                blobs.append(dict(date=date, blob_id=i, cx=round(cxb, 1), cy=round(cyb, 1), mu=round(mu, 4), n_pix=npix,
                                  area_msh=round(a, 3), umb_min_ratio=round(float(vals.min()) / iqs, 4),
                                  umb_median_ratio=round(med, 4), I_qs=round(iqs, 2)))
                if mu >= MU_MIN and a >= AREA_MIN_MSH and med > MEDIAN_ART:
                    n_art += 1
            if mu >= MU_MIN and a >= AREA_MIN_MSH:
                umin = min(umin, float(vals.min()))
        au = sum(a for *_, a, _x, _y in [(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]) for c in cu] if False) or 0.0
        au = sum(c[5] for c in cu if c[4] >= MU_MIN and c[5] >= AREA_MIN_MSH)
        ap = sum(c[5] for c in cp if c[4] >= MU_MIN and c[5] >= AREA_MIN_MSH)
        day = None
        if au + ap > 0 and np.isfinite(umin):
            day = dict(date=date, area_ar_msh=round(au + ap, 3), area_umbra_msh=round(au, 3), area_penumbra_msh=round(ap, 3),
                       umb_min_ratio=round(umin / iqs, 4), I_qs=round(iqs, 2), n_umb_median_gt_0503=n_art)
        return blobs, day
    except Exception as e:
        return [dict(date=date, blob_id=-1, error=f"{type(e).__name__}: {e}")], None


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    days = []
    for mp in sorted(glob.glob(f"{MASK_DIR}/*/*_pred.png")):
        m = RX.search(os.path.basename(mp))
        if not m:
            continue
        fp = f"{FITS_DIR}/{os.path.basename(os.path.dirname(mp))}/{m.group(1)}.continuum.psf_ld.fits"
        if os.path.exists(fp):
            days.append((m.group(2), mp, fp))
    print(f"days (mask + FITS): {len(days)} | workers {workers}", flush=True)
    blobs, perday, t0 = [], [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(process, d) for d in days]
        for k, f in enumerate(as_completed(futs), 1):
            b, d = f.result(); blobs += b
            if d:
                perday.append(d)
            if k % 200 == 0 or k == len(futs):
                el = time.time() - t0
                print(f"  {k}/{len(futs)} | umbra {len(blobs)} | days {len(perday)} | {el / 60:.1f} min | ETA {(len(futs) - k) * el / k / 60:.1f} min", flush=True)
    B = pd.DataFrame(blobs).sort_values(["date", "blob_id"]); P = pd.DataFrame(perday).sort_values("date")
    B.to_csv(f"{OUT}/umbra_blob_intensity_area_v4.csv", index=False)
    P.to_csv(f"{OUT}/perday_ar_area_umbmin_v4mask.csv", index=False)
    q = B[(B.blob_id != -1) & (B.mu >= MU_MIN) & (B.area_msh >= AREA_MIN_MSH) & (B.umb_median_ratio <= MEDIAN_ART)]
    print(f"\nper-umbra: {len(B)} rows | passing the paper cuts (mu>=0.7, >=1 MSH, median<=0.503): {len(q)} | error {int((B.blob_id == -1).sum())}")
    print(f"per-day  : {len(P)} days | days with an umbra of median > 0.503: {(P.n_umb_median_gt_0503 > 0).sum()}")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
