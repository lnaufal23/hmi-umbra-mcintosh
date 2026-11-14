#!/usr/bin/env python3
"""Quiet-Sun RMS contrast and photospheric intensity distribution (Sections 3.1-3.3).

For every fifth day (1,041 days): RMS contrast (std/mean of I/I_qs) of the photospheric pixels
with mu >= 0.9 after dilating the spot mask by 10 px, on the clipped PSF+LD product and on the
raw image corrected only for limb darkening; and mean, median and peak of the photospheric
pixels within 60 degrees of disk center.

Usage: python rms_contrast.py [--workers N] [--n N_TEST_DAYS]
"""
import argparse
import glob
import os
import re
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
PSF_DIR = paths.get("psf_ld_fits")
RAW_DIR = paths.get("raw_fits")
OUT = paths.get("rms_contrast_out")
RX = re.compile(r"(hmi\.ic_720s\.(\d{8})_000000_TAI\.\d+)_pred\.png$")
QS_LO, QS_HI, DISK_SHRINK = 40, 70, 0.97
UMBRA, PENUMBRA = 0, 127
MU_RMS, MU_PHOT, DILATE_PX = 0.9, 0.5, 10
LD_MU_MIN, LD_N_BINS, LD_QUANTILE, LD_POLY_DEGREE, LD_CLIP_LOW_PCT, LD_CLIP_HIGH_PCT = 0.20, 140, 90, 3, 1, 99
HIST_EDGES = np.linspace(0.5, 1.5, 4001)


def load_fits(fp):
    with fits.open(fp, memmap=False) as hdul:
        for h in hdul:
            if h.data is not None and h.data.ndim == 2 and h.data.shape[0] >= 1024:
                return h.data.astype(np.float64), h.header.copy()
    raise RuntimeError(f"no 2D HDU in {fp}")


def geometry(shape, header):
    h, w = shape
    cx, cy = float(header.get("CRPIX1", w / 2)) - 1.0, float(header.get("CRPIX2", h / 2)) - 1.0
    rsun = float(header.get("RSUN_OBS", 960.0)) / float(header.get("CDELT1", 0.504))
    yy, xx = np.indices(shape)
    r = np.hypot(xx - cx, yy - cy) / rsun
    mu = np.zeros(shape)
    inside = r < 1.0
    mu[inside] = np.sqrt(1.0 - r[inside] ** 2)
    return r, mu, inside


def iqs_of(data, r):
    px = data[(r <= DISK_SHRINK) & np.isfinite(data)]
    lo, hi = np.percentile(px, QS_LO), np.percentile(px, QS_HI)
    return float(np.median(px[(px >= lo) & (px <= hi)]))


def ld_correct(image, mu, disk_mask):
    fit_mask = disk_mask & np.isfinite(image) & (image > 0) & (mu >= LD_MU_MIN)
    bins = np.linspace(LD_MU_MIN, 1.0, LD_N_BINS + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    profile = np.full(LD_N_BINS, np.nan)
    for i in range(LD_N_BINS):
        vals = image[fit_mask & (mu >= bins[i]) & (mu < bins[i + 1])]
        if vals.size:
            lo, hi = np.nanpercentile(vals, LD_CLIP_LOW_PCT), np.nanpercentile(vals, LD_CLIP_HIGH_PCT)
            v = vals[(vals >= lo) & (vals <= hi)]
            if v.size:
                profile[i] = np.nanpercentile(v, LD_QUANTILE)
    valid = np.isfinite(profile)
    coeff = np.polyfit(centers[valid], profile[valid], LD_POLY_DEGREE)
    inside = disk_mask & np.isfinite(image) & (mu > 0)
    out = np.full_like(image, np.nan)
    out[inside] = image[inside] / np.clip(np.polyval(coeff, mu[inside]), 1e-6, None) * float(np.polyval(coeff, 1.0))
    return out


def stats(q, sel):
    v = q[sel]
    v = v[np.isfinite(v)]
    return v


def process(args):
    date, mp, fp, rp = args
    t0 = time.time()
    try:
        data, header = load_fits(fp)
        r, mu, disk = geometry(data.shape, header)
        mask = np.array(Image.open(mp)); mask = mask[..., 0] if mask.ndim == 3 else mask
        if mask.shape != data.shape:
            mask = np.array(Image.fromarray(mask).resize((data.shape[1], data.shape[0]), Image.NEAREST))
        spot = (mask == UMBRA) | (mask == PENUMBRA)
        dil = ndimage.maximum_filter(spot.astype(np.uint8), size=2 * DILATE_PX + 1) > 0

        iqs = iqs_of(data, r)
        q = data / iqs
        sel_rms = disk & (mu >= MU_RMS) & ~dil
        v = stats(q, sel_rms)
        w = stats(q, disk & (mu >= MU_PHOT) & ~spot)
        hist, _ = np.histogram(w, HIST_EDGES)
        peak = float(0.5 * (HIST_EDGES[np.argmax(hist)] + HIST_EDGES[np.argmax(hist) + 1]))

        raw, rh = load_fits(rp)
        rr, rmu, rdisk = geometry(raw.shape, rh)
        raw_ld = ld_correct(raw, rmu, rdisk)
        qr = raw_ld / iqs_of(raw_ld, rr)
        vr = stats(qr, sel_rms) if qr.shape == q.shape else np.array([np.nan])

        return dict(date=date, iqs=round(iqs, 3), n_rms=int(v.size),
                    rms_psf=float(v.std() / v.mean()), sd_psf=float(v.std()), mean09_psf=float(v.mean()),
                    rms_raw=float(vr.std() / vr.mean()), sd_raw=float(vr.std()),
                    phot60_mean=float(w.mean()), phot60_median=float(np.median(w)), phot60_peak=peak,
                    t_s=round(time.time() - t0, 1)), None
    except Exception as e:
        return dict(date=date), f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--n", type=int, default=0, help="test run: number of first days only")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    days = []
    for k, mp in enumerate(sorted(glob.glob(f"{MASK_DIR}/*/*_pred.png"))):
        m = RX.search(os.path.basename(mp))
        if not m or k % 5:
            continue
        year = os.path.basename(os.path.dirname(mp))
        fp = f"{PSF_DIR}/{year}/{m.group(1)}.continuum.psf_ld.fits"
        rp = f"{RAW_DIR}/{year}/{m.group(1)}.continuum.fits"
        if os.path.exists(fp) and os.path.exists(rp):
            days.append((m.group(2), mp, fp, rp))
    if args.n:
        days = days[:args.n]
    print(f"sampled days: {len(days)} | workers {args.workers}", flush=True)
    rows, errs, t0 = [], [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process, d) for d in days]
        for k, f in enumerate(as_completed(futs), 1):
            row, err = f.result()
            if err:
                errs.append(f"{row['date']}: {err}"); print(f"  FAILED {row['date']}: {err}", flush=True)
            else:
                rows.append(row)
            if k % 50 == 0 or k == len(futs):
                el = time.time() - t0
                print(f"  {k}/{len(futs)} | failed {len(errs)} | {el / 60:.1f} min | ETA {(len(futs) - k) * el / k / 60:.1f} min", flush=True)
    tag = "_test" if args.n else ""
    D = pd.DataFrame(rows).sort_values("date")
    D.to_csv(f"{OUT}/rms_contrast{tag}.csv", index=False)
    with open(f"{OUT}/errors{tag}.txt", "w") as f:
        f.write("\n".join(errs))
    cols = ["rms_psf", "sd_psf", "mean09_psf", "rms_raw", "sd_raw", "phot60_mean", "phot60_median", "phot60_peak"]
    print(f"\nN days = {len(D)}")
    print(D[cols].describe(percentiles=[0.16, 0.5, 0.84]).round(4).to_string())
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
