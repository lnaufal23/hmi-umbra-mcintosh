#!/usr/bin/env python3
"""PSF+LD FITS -> full-disk 4096x4096 8-bit opt_F PNG, the input of the U-Net (Section 4.1).

The quiet-Sun level is the median of the pixels between the 40th and 70th percentiles within
0.97 R_sun; I/I_qs is mapped piecewise 0/0.30/0.85/1.05 -> 0/30/200/235, values above 1.05
are stretched to 255, and off-disk pixels are set to 245.
Output: <optf_png>/{year}/hmi.ic_720s.YYYYMMDD_000000_TAI.N.png. Existing files are skipped.

Usage: python make_optf.py [WORKERS]
"""
import glob
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from astropy.io import fits
from PIL import Image

from hmi_umbra import params as C
from hmi_umbra import paths

SRC = paths.get("psf_ld_fits")
OUT = paths.get("optf_png")
QS_LO, QS_HI = C.QS_PERCENTILE_LO, C.QS_PERCENTILE_HI
BREAKS, OUTPUTS = C.PIECEWISE_BREAKS, C.PIECEWISE_OUTPUTS
OFF_DISK, SHRINK = C.OFF_DISK_VALUE, C.DISK_MASK_SHRINK_FACTOR
RX = re.compile(r"(hmi\.ic_720s\.\d{8}_000000_TAI\.\d+)\.continuum\.psf_ld\.fits$")


def disk_mask(data, header):
    h, w = data.shape
    crpix1, crpix2 = float(header.get("CRPIX1", w / 2)), float(header.get("CRPIX2", h / 2))
    rsun_pix = float(header.get("RSUN_OBS", 960.0)) / float(header.get("CDELT1", 0.504))
    yy, xx = np.indices(data.shape)
    return np.sqrt((xx - (crpix1 - 1)) ** 2 + (yy - (crpix2 - 1)) ** 2) <= rsun_pix * SHRINK


def do(fp):
    m = RX.search(os.path.basename(fp))
    year = os.path.basename(os.path.dirname(fp))
    out = os.path.join(OUT, year, m.group(1) + ".png")
    if os.path.exists(out):
        return None
    try:
        with fits.open(fp, memmap=False) as hdul:
            hdu = next(h for h in hdul if h.data is not None and h.data.ndim == 2 and h.data.shape[0] >= 1024)
            data, header = hdu.data.astype(np.float32), hdu.header.copy()
        if np.isnan(data).any():
            pre = disk_mask(data, header)
            fill = float(np.nanmedian(data[pre])) if pre.any() else float(np.nanmedian(data))
            data = np.where(np.isnan(data), fill, data)
        dm = disk_mask(data, header)
        p = data[dm]
        lo, hi = np.percentile(p, QS_LO), np.percentile(p, QS_HI)
        qs = float(np.median(p[(p >= lo) & (p <= hi)]))
        ratio = data / max(qs, 1e-6)
        b, o = np.array(BREAKS, np.float32), np.array(OUTPUTS, np.float32)
        img = np.interp(ratio, b, o, left=0, right=o[-1])
        br = ratio > b[-1]
        if br.any():
            img[br] = o[-1] + np.clip((ratio[br] - b[-1]) / 0.15, 0, 1) * (255 - o[-1])
        img[~dm] = OFF_DISK
        os.makedirs(os.path.dirname(out), exist_ok=True)
        Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode="L").save(out + ".tmp", format="PNG", compress_level=C.PNG_COMPRESS_LEVEL)
        os.replace(out + ".tmp", out)
        return None
    except Exception as e:
        return f"{fp}: {type(e).__name__}: {e}"


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    files = sorted(f for f in glob.glob(f"{SRC}/*/*.continuum.psf_ld.fits") if RX.search(os.path.basename(f)))
    print(f"FITS found: {len(files)} | workers {workers}", flush=True)
    t0, errs = time.time(), []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(do, f) for f in files]
        for i, fu in enumerate(as_completed(futs), 1):
            e = fu.result()
            if e:
                errs.append(e)
            if i % 200 == 0 or i == len(futs):
                el = time.time() - t0
                print(f"  {i}/{len(futs)} | failed {len(errs)} | {el / 60:.1f} min | ETA {(len(futs) - i) * el / i / 60:.1f} min", flush=True)
    with open(os.path.join(OUT, "optf_errors.txt"), "w") as f:
        f.write("\n".join(errs))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
