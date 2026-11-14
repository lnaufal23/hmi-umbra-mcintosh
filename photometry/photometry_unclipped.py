#!/usr/bin/env python3
"""Umbral photometry on the unclipped deconvolution (Sections 3.1, 4.3, 5).

Each raw image is divided by its on-disk median, kept positive, deconvolved with the same PSF
and 20 Richardson-Lucy iterations without bounds on the output, rescaled, and corrected for limb
darkening as the clipped product. The full-disk U-Net masks are unchanged. Writes per-umbra and
per-day tables and, for every fifth day, class histograms of I/I_qs (mu >= 0.5).

Usage: python photometry_unclipped.py [--workers N] [--dates YYYYMMDD,...]
"""
import argparse
import glob
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from astropy.io import fits
from PIL import Image
from scipy import ndimage
from skimage.restoration import richardson_lucy

from hmi_umbra import paths

Image.MAX_IMAGE_PIXELS = None
RAW = paths.get("raw_fits")
MASK_DIR = paths.get("masks_fulldisk")
OLD_STATS = paths.get("stats_clipped_out")
OUT = paths.get("stats_unclipped_out")
PARTS, HIST = f"{OUT}/parts", f"{OUT}/hist"
RX = re.compile(r"(hmi\.ic_720s\.(\d{8})_000000_TAI\.\d+)_pred\.png$")

PSF_KERNEL_SIZE = 65
PSF_SIGMAS = (0.7, 1.4, 2.8, 5.5, 10.0)
PSF_WEIGHTS = (0.62, 0.22, 0.10, 0.045, 0.015)
PSF_ANISOTROPY = (0.00, 0.01, 0.02, 0.03, 0.04)
PSF_PHASE_DEG = (0.0, 20.0, 40.0, 70.0, 100.0)
RL_ITER, RL_EPS = 20, 1e-6
LD_MU_MIN, LD_N_BINS, LD_QUANTILE, LD_POLY_DEGREE, LD_CLIP_LOW_PCT, LD_CLIP_HIGH_PCT = 0.20, 140, 90, 3, 1, 99

MIN_BLOB_PIX, QS_LO, QS_HI, DISK_SHRINK = 4, 40, 70, 0.97
MU_MIN, AREA_MIN_MSH, MEDIAN_ART = 0.7, 1.0, 0.503
UMBRA, PENUMBRA = 0, 127
HIST_EDGES = np.linspace(0.0, 1.5, 1501)
HIST_MU_MIN = 0.5
ERODE_ITER, ERODE_MIN_PIX = 2, 10


def load_hmi_fits(path):
    with fits.open(path, memmap=False) as hdul:
        if len(hdul) > 1 and hdul[1].data is not None:
            return hdul[1].data.astype(np.float32), hdul[1].header.copy()
        return hdul[0].data.astype(np.float32), hdul[0].header.copy()


def build_solar_disk_geometry(data, header):
    ny, nx = data.shape
    x0, y0 = float(header["CRPIX1"]) - 1.0, float(header["CRPIX2"]) - 1.0
    rsun_pix = 0.5 * (float(header["RSUN_OBS"]) / abs(float(header["CDELT1"])) + float(header["RSUN_OBS"]) / abs(float(header["CDELT2"])))
    yy, xx = np.indices((ny, nx), dtype=np.float32)
    rho = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2) / rsun_pix
    mu = np.zeros_like(rho, dtype=np.float32)
    inside = rho <= 1.0
    mu[inside] = np.sqrt(np.clip(1.0 - rho[inside] ** 2, 0.0, 1.0))
    return mu, inside & np.isfinite(data)


def build_anisotropic_psf():
    n = PSF_KERNEL_SIZE
    ax = np.arange(-(n // 2), n // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    rr2, phi = xx ** 2 + yy ** 2, np.arctan2(yy, xx)
    psf = np.zeros((n, n), dtype=np.float64)
    for w, s, a, ph in zip(PSF_WEIGHTS, PSF_SIGMAS, PSF_ANISOTROPY, PSF_PHASE_DEG):
        mod = np.clip(1.0 + a * np.cos(2.0 * (phi - np.deg2rad(ph))), 0.0, None)
        g = np.exp(-0.5 * rr2 / (s ** 2)) * mod
        psf += w * g / g.sum()
    psf = np.clip(psf, 0.0, None)
    return (psf / psf.sum()).astype(np.float32)


def run_psf_noclip(data, disk_mask, psf_kernel):
    valid = disk_mask & np.isfinite(data) & (data > 0)
    med = float(np.nanmedian(data[valid]))
    img = data.astype(np.float32).copy()
    img[~np.isfinite(img)] = med
    img[~disk_mask] = med
    img = np.clip(img / med, 1e-6, None).astype(np.float32)
    out = richardson_lucy(image=img, psf=psf_kernel, num_iter=RL_ITER, clip=False, filter_epsilon=RL_EPS)
    out = (np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0) * med).astype(np.float32)
    out[~disk_mask] = data[~disk_mask]
    return out, med


def ld_correct(image, mu, disk_mask):
    fit_mask = disk_mask & np.isfinite(image) & np.isfinite(mu) & (image > 0) & (mu >= LD_MU_MIN) & (mu <= 1.0)
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
    if valid.sum() < LD_POLY_DEGREE + 2:
        raise RuntimeError(f"too few valid LD bins: {valid.sum()}")
    coeff = np.polyfit(centers[valid], profile[valid], LD_POLY_DEGREE)
    inside = disk_mask & np.isfinite(image) & np.isfinite(mu) & (mu > 0)
    i_ref = float(np.polyval(coeff, 1.0))
    corrected = np.full_like(image, np.nan, dtype=np.float32)
    corrected[inside] = (image[inside] / np.clip(np.polyval(coeff, mu[inside]), 1e-6, None)) * i_ref
    return corrected, coeff, i_ref


def comps(lab, cx, cy, rsun, mpp):
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
    date, stem, mp, rp, do_hist = args
    t0 = time.time()
    try:
        raw, header = load_hmi_fits(rp)
        mu_img, disk = build_solar_disk_geometry(raw, header)
        dpsf, med_raw = run_psf_noclip(raw, disk, build_anisotropic_psf())
        data, coeff, i_ref = ld_correct(dpsf, mu_img, disk)
        data = data.astype(np.float64)
        t_corr = time.time() - t0

        mask = np.array(Image.open(mp)); mask = mask[..., 0] if mask.ndim == 3 else mask
        if mask.shape != data.shape:
            mask = np.array(Image.fromarray(mask).resize((data.shape[1], data.shape[0]), Image.NEAREST))
        h, w = data.shape
        cx, cy = float(header.get("CRPIX1", w / 2)) - 1.0, float(header.get("CRPIX2", h / 2)) - 1.0
        rsun = float(header.get("RSUN_OBS", 960.0)) / float(header.get("CDELT1", 0.504))
        yy, xx = np.indices(data.shape)
        px = data[np.hypot(xx - cx, yy - cy) <= rsun * DISK_SHRINK]
        px = px[np.isfinite(px)]
        lo, hi = np.percentile(px, QS_LO), np.percentile(px, QS_HI)
        iqs = float(np.median(px[(px >= lo) & (px <= hi)]))
        if not np.isfinite(iqs) or iqs <= 0:
            raise RuntimeError(f"invalid I_qs: {iqs}")
        mpp = 1e6 / (2 * np.pi * rsun ** 2)
        lab_u, _ = ndimage.label(mask == UMBRA)
        lab_p, _ = ndimage.label(mask == PENUMBRA)
        cu, cp = comps(lab_u, cx, cy, rsun, mpp), comps(lab_p, cx, cy, rsun, mpp)
        blobs, umin, n_art = [], np.inf, 0
        for i, sl, sub, npix, mu, a, cxb, cyb in cu:
            vals = data[sl][sub]; vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            if npix >= MIN_BLOB_PIX:
                q1, q3 = np.percentile(vals, [25, 75]); iqr = q3 - q1
                cl = vals[(vals >= q1 - 1.5 * iqr) & (vals <= q3 + 1.5 * iqr)]
                med = float(np.median(cl)) / iqs if cl.size else float(np.median(vals)) / iqs
                vs = np.sort(vals)
                dark25 = float(vs[:max(1, int(np.ceil(0.25 * vs.size)))].mean()) / iqs
                sub_e = ndimage.binary_erosion(sub, iterations=ERODE_ITER)
                ve = data[sl][sub_e]; ve = ve[np.isfinite(ve)]
                med_e = float(np.median(ve)) / iqs if ve.size >= ERODE_MIN_PIX else np.nan
                blobs.append(dict(date=date, blob_id=i, cx=round(cxb, 1), cy=round(cyb, 1), mu=round(mu, 4), n_pix=npix,
                                  area_msh=round(a, 3), umb_min_ratio=round(float(vals.min()) / iqs, 4),
                                  umb_median_ratio=round(med, 4), umb_dark25_ratio=round(dark25, 4),
                                  umb_median_erode2_ratio=round(med_e, 4) if np.isfinite(med_e) else np.nan,
                                  n_pix_erode2=int(ve.size), I_qs=round(iqs, 2)))
                if mu >= MU_MIN and a >= AREA_MIN_MSH and med > MEDIAN_ART:
                    n_art += 1
            if mu >= MU_MIN and a >= AREA_MIN_MSH:
                umin = min(umin, float(vals.min()))
        au = sum(c[5] for c in cu if c[4] >= MU_MIN and c[5] >= AREA_MIN_MSH)
        ap = sum(c[5] for c in cp if c[4] >= MU_MIN and c[5] >= AREA_MIN_MSH)
        day = None
        if au + ap > 0 and np.isfinite(umin):
            day = dict(date=date, area_ar_msh=round(au + ap, 3), area_umbra_msh=round(au, 3), area_penumbra_msh=round(ap, 3),
                       umb_min_ratio=round(umin / iqs, 4), I_qs=round(iqs, 2), n_umb_median_gt_0503=n_art)

        if do_hist:
            sel = disk & (mu_img >= HIST_MU_MIN) & np.isfinite(data)
            ratio = data / iqs
            cls = {"umbra": sel & (mask == UMBRA), "penumbra": sel & (mask == PENUMBRA),
                   "photosphere": sel & (mask != UMBRA) & (mask != PENUMBRA)}
            np.savez_compressed(f"{HIST}/{date}.npz", edges=HIST_EDGES,
                                **{k: np.histogram(ratio[m], HIST_EDGES)[0] for k, m in cls.items()})

        cols = ["date", "blob_id", "cx", "cy", "mu", "n_pix", "area_msh", "umb_min_ratio", "umb_median_ratio",
                "umb_dark25_ratio", "umb_median_erode2_ratio", "n_pix_erode2", "I_qs"]
        pd.DataFrame(blobs, columns=cols).to_csv(f"{PARTS}/{date}.csv", index=False)
        meta = dict(date=date, stem=stem, I_qs=iqs, i_ref=i_ref, raw_disk_median=med_raw,
                    ld_c0=float(coeff[0]), ld_c1=float(coeff[1]), ld_c2=float(coeff[2]), ld_c3=float(coeff[3]),
                    n_blob=len(blobs), hist=bool(do_hist), t_corr_s=round(t_corr, 1), t_total_s=round(time.time() - t0, 1))
        with open(f"{PARTS}/{date}.json.tmp", "w") as f:
            json.dump(dict(meta=meta, day=day), f)
        os.replace(f"{PARTS}/{date}.json.tmp", f"{PARTS}/{date}.json")
        return date, None, time.time() - t0
    except Exception as e:
        return date, f"{type(e).__name__}: {e}", time.time() - t0


def list_days():
    days = []
    for mp in sorted(glob.glob(f"{MASK_DIR}/*/*_pred.png")):
        m = RX.search(os.path.basename(mp))
        if not m:
            continue
        year = os.path.basename(os.path.dirname(mp))
        rp = f"{RAW}/{year}/{m.group(1)}.continuum.fits"
        days.append((m.group(2), m.group(1), mp, rp))
    return days


def merge(dates=None):
    js = sorted(glob.glob(f"{PARTS}/*.json"))
    if dates:
        js = [j for j in js if os.path.basename(j)[:8] in dates]
    metas, perday, blobs = [], [], []
    for j in js:
        d = json.load(open(j)); metas.append(d["meta"])
        if d["day"]:
            perday.append(d["day"])
        b = pd.read_csv(j.replace(".json", ".csv"), dtype={"date": str})
        if len(b):
            blobs.append(b)
    B = pd.concat(blobs, ignore_index=True).sort_values(["date", "blob_id"]) if blobs else pd.DataFrame()
    P = pd.DataFrame(perday).sort_values("date") if perday else pd.DataFrame()
    M = pd.DataFrame(metas).sort_values("date")
    tag = "_test" if dates else ""
    B.to_csv(f"{OUT}/umbra_blob_intensity_area_noclipT1{tag}.csv", index=False)
    P.to_csv(f"{OUT}/perday_ar_area_umbmin_noclipT1{tag}.csv", index=False)
    M.to_csv(f"{OUT}/meta_noclipT1{tag}.csv", index=False)
    return B, P, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--dates", default=None, help="test run: comma-separated list of YYYYMMDD")
    args = ap.parse_args()
    os.makedirs(PARTS, exist_ok=True); os.makedirs(HIST, exist_ok=True)

    all_days = list_days()
    hist_dates = {d[0] for k, d in enumerate(all_days) if k % 5 == 0}
    missing = [d for d in all_days if not os.path.exists(d[3])]
    sel = [d for d in all_days if os.path.exists(d[3])]
    dates = set(args.dates.split(",")) if args.dates else None
    if dates:
        sel = [d for d in sel if d[0] in dates]
    todo = [(d[0], d[1], d[2], d[3], d[0] in hist_dates or bool(dates)) for d in sel if not os.path.exists(f"{PARTS}/{d[0]}.json")]
    print(f"mask v4: {len(all_days)} | raw missing: {len(missing)} | selected: {len(sel)} | already done: {len(sel) - len(todo)} | "
          f"to process: {len(todo)} | workers {args.workers}", flush=True)
    for d in missing[:20]:
        print(f"  raw missing: {d[3]}", flush=True)

    errs, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process, t) for t in todo]
        for k, f in enumerate(as_completed(futs), 1):
            date, err, dt = f.result()
            if err:
                errs.append(f"{date}: {err}"); print(f"  FAILED {date}: {err}", flush=True)
            if k % 50 == 0 or k == len(futs) or dates:
                el = time.time() - t0
                print(f"  {k}/{len(futs)} | failed {len(errs)} | {el / 60:.1f} min | ETA {(len(futs) - k) * el / k / 60:.1f} min | last {dt:.0f}s", flush=True)
    with open(f"{OUT}/errors{'_test' if dates else ''}.txt", "w") as f:
        f.write("\n".join(errs))

    B, P, M = merge(dates)
    print(f"\nper-umbra: {len(B)} rows | per-day: {len(P)} | meta: {len(M)} | median time/day {M.t_total_s.median():.1f}s")
    if len(B):
        q = B[(B.mu >= MU_MIN) & (B.area_msh >= AREA_MIN_MSH) & (B.umb_median_ratio <= MEDIAN_ART)]
        print(f"passing the paper cuts: {len(q)} | lowest per-umbra median {q.umb_median_ratio.min():.4f}")
    if dates and len(B):
        O = pd.read_csv(f"{OLD_STATS}/umbra_blob_intensity_area_v4.csv", dtype={"date": str})
        O = O[O.date.isin(dates) & (O.blob_id != -1)]
        J = O.merge(B, on=["date", "blob_id"], suffixes=("_old", "_new"))
        print(f"\n=== check against stats_v4 (test dates) ===")
        print(f"umbrae old {len(O)} | new {len(B)} | matched blob_id {len(J)} | n_pix equal {int((J.n_pix_old == J.n_pix_new).sum())} | "
              f"area equal {int((J.area_msh_old == J.area_msh_new).sum())} | mu equal {int((J.mu_old == J.mu_new).sum())}")
        print(f"I_qs old vs new (median ratio): {(J.I_qs_new / J.I_qs_old).median():.5f}")
        g = J[(J.mu_old >= MU_MIN) & (J.area_msh_old >= AREA_MIN_MSH)].copy()
        g["abin"] = pd.cut(g.area_msh_old, [1, 3, 10, 30, 131, 1e9], right=False)
        print(g.groupby("abin", observed=True).agg(N=("blob_id", "size"), med_old=("umb_median_ratio_old", "median"),
                                                    med_new=("umb_median_ratio_new", "median"),
                                                    min_old=("umb_min_ratio_old", "median"), min_new=("umb_min_ratio_new", "median"),
                                                    dark25=("umb_dark25_ratio", "median"), med_erode2=("umb_median_erode2_ratio", "median")).round(4).to_string())
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
