#!/usr/bin/env python3
"""
Intensity boundaries implied by the U-Net masks (Section 3.3): crossing points of the class
histograms of I/I_qs within 60 degrees of disk center and 0.97 R_sun, on the clipped product.
  T_up : penumbral = umbral pixel count (between the umbral and penumbral peaks)
  T_pp : photospheric = penumbral pixel count (between the penumbral and photospheric peaks)
Here I_qs is the median of the photospheric pixels (mask 255) in this zone between the 40th and
70th percentiles, as in remeasure_masks_v4.py. Every STEP-th day is sampled.
"""
import json, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
from astropy.io import fits
from PIL import Image

from hmi_umbra import paths

FITS_ROOT = Path(paths.get("psf_ld_fits"))
MASK_ROOT = Path(paths.get("masks_fulldisk"))
OUT = Path(paths.get("analysis_v4mask"))
STEP, WORKERS = 5, 24
EDGES = np.arange(0.0, 1.2001, 0.002); CEN = 0.5 * (EDGES[1:] + EDGES[:-1])
QS_LO, QS_HI, DISK_SHRINK, HELIO = 40, 70, 0.97, 60.0


def zone(data, h):
    H, W = data.shape
    c1, c2 = float(h.get("CRPIX1", W / 2)), float(h.get("CRPIX2", H / 2))
    cd, rs = float(h.get("CDELT1", 0.504)), float(h.get("RSUN_OBS", 960.0))
    yy, xx = np.ogrid[:H, :W]
    r = np.sqrt((xx - (c1 - 1)) ** 2 + (yy - (c2 - 1)) ** 2)
    th = np.degrees(np.arcsin(np.clip(r * cd / rs, 0, 1)))
    return (r <= rs / cd * DISK_SHRINK) & (th <= HELIO)


def one(mp):
    mp = Path(mp); stem = mp.name.replace("_pred.png", ""); year = int(stem.split(".")[2][:4])
    fp = FITS_ROOT / str(year) / f"{stem}.continuum.psf_ld.fits"
    if not fp.exists():
        return None
    with fits.open(str(fp), memmap=True) as hd:
        d = hd[0].data.astype(np.float32); h = hd[0].header
    m = np.array(Image.open(mp).convert("L"))
    v = zone(d, h) & np.isfinite(d)
    pho = d[v & (m == 255)]
    lo, hi = np.percentile(pho, [QS_LO, QS_HI]); iqs = float(np.median(pho[(pho >= lo) & (pho <= hi)]))
    return year, [np.histogram(d[v & (m == k)] / iqs, EDGES)[0] for k in (0, 127, 255)]


def cross(a, b, lo, hi):
    """First ratio in (lo, hi) at which b >= a (a dominates before it)."""
    sel = np.where((CEN > lo) & (CEN < hi))[0]
    for i in sel:
        if b[i] >= a[i] and a[i] > 0:
            return float(CEN[i])
    return np.nan


def main():
    masks = sorted(str(p) for p in MASK_ROOT.glob("*/*_pred.png"))[::STEP]
    print(f"sample of {len(masks)} days (every {STEP}th)", flush=True)
    H = {}; t0 = time.time()
    with ProcessPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(one, m) for m in masks]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r:
                y, hs = r; H.setdefault(y, [np.zeros(len(CEN)) for _ in range(3)])
                for k in range(3):
                    H[y][k] += hs[k]
            if i % 100 == 0:
                print(f"  {i}/{len(masks)} | {time.time()-t0:.0f}s", flush=True)
    tot = [sum(H[y][k] for y in H) for k in range(3)]
    um, pe, ph = tot
    pk_u, pk_p, pk_f = (CEN[np.argmax(x)] for x in tot)
    T_up = cross(um, pe, pk_u, pk_p); T_pp = cross(pe, ph, pk_p, pk_f)
    print(f"\nGLOBAL | peaks: umbra {pk_u:.3f} penumbra {pk_p:.3f} photosphere {pk_f:.3f}")
    print(f"  T_umbra-penumbra      = {T_up:.3f}  (previous 0.503)")
    print(f"  T_penumbra-photosphere= {T_pp:.3f}  (previous 0.857)")
    for nm, x in [("umbra", um), ("penumbra", pe)]:
        c = np.cumsum(x) / x.sum()
        print(f"  {nm:8s} pixels: p5 {CEN[np.searchsorted(c,.05)]:.3f} p50 {CEN[np.searchsorted(c,.5)]:.3f} p95 {CEN[np.searchsorted(c,.95)]:.3f}")
    fu = um[CEN >= T_up].sum() / um.sum(); fp_ = pe[CEN < T_up].sum() / pe.sum(); fpp = pe[CEN >= T_pp].sum() / pe.sum()
    print(f"  umbral pixels above T_up {fu:.3%} | penumbral below T_up {fp_:.3%} | penumbral above T_pp {fpp:.3%}")
    rows = []
    print("\nPER YEAR:")
    for y in sorted(H):
        u, p, f = H[y]
        a, b = cross(u, p, CEN[np.argmax(u)], CEN[np.argmax(p)]), cross(p, f, CEN[np.argmax(p)], CEN[np.argmax(f)])
        rows.append({"year": y, "T_up": a, "T_pp": b, "n_umb_pix": int(u.sum()), "n_pen_pix": int(p.sum())})
        print(f"  {y}: T_up {a:.3f} | T_pp {b:.3f} | umbral pixels {int(u.sum()):>9,} penumbral {int(p.sum()):>10,}")
    yt = pd.DataFrame(rows); ok = yt[yt["n_umb_pix"] > 1e5]
    print(f"\nStability (years with >1e5 umbral pixels, n={len(ok)}): T_up {ok['T_up'].mean():.3f} ± {ok['T_up'].std():.3f} | "
          f"T_pp {ok['T_pp'].mean():.3f} ± {ok['T_pp'].std():.3f}")
    np.savez(OUT / "hist_class_ratio_v4mask.npz", centers=CEN, **{f"{y}_{k}": H[y][i] for y in H for i, k in enumerate(["umb", "pen", "pho"])})
    yt.to_csv(OUT / "threshold_histcross_by_year_v4mask.csv", index=False)
    json.dump({"T_umbra_penumbra": T_up, "T_penumbra_photosphere": T_pp, "peaks": [pk_u, pk_p, pk_f],
               "n_days_sampled": len(masks), "step": STEP}, open(OUT / "threshold_histcross_v4mask.json", "w"), indent=2)
    print(f"[WRITE] {OUT}/threshold_histcross_v4mask.json, threshold_histcross_by_year_v4mask.csv, hist_class_ratio_v4mask.npz")


if __name__ == "__main__":
    main()
