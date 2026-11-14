#!/usr/bin/env python3
"""
correct_psf_ld.py — PSF deconvolution (five anisotropic Gaussians, Richardson-Lucy) and
limb-darkening correction of one SDO/HMI continuum image (Section 3.1).

This produces the clipped PSF+LD product on which the training masks were drawn and on
which segmentation and classification operate. Parameters are in hmi_umbra/params.py.

Usage: python correct_psf_ld.py RAW.continuum.fits [OUT.psf_ld.fits]
"""

import time
import logging
from pathlib import Path

import numpy as np
from astropy.io import fits
from skimage.restoration import richardson_lucy

from hmi_umbra import params as C

log = logging.getLogger("scientific_daily.psf_ld")


# ============================================================================
# Helpers
# ============================================================================
def load_hmi_fits(path):
    with fits.open(path, memmap=False) as hdul:
        if len(hdul) > 1 and hdul[1].data is not None:
            data = hdul[1].data.astype(np.float32)
            header = hdul[1].header.copy()
        else:
            data = hdul[0].data.astype(np.float32)
            header = hdul[0].header.copy()
    return data, header


def build_solar_disk_geometry(data, header):
    ny, nx = data.shape
    crpix1 = float(header["CRPIX1"])
    crpix2 = float(header["CRPIX2"])
    cdelt1 = float(header["CDELT1"])
    cdelt2 = float(header["CDELT2"])
    rsun_obs = float(header["RSUN_OBS"])

    x0 = crpix1 - 1.0
    y0 = crpix2 - 1.0

    rsun_pix_x = rsun_obs / abs(cdelt1)
    rsun_pix_y = rsun_obs / abs(cdelt2)
    rsun_pix = 0.5 * (rsun_pix_x + rsun_pix_y)

    yy, xx = np.indices((ny, nx), dtype=np.float32)
    dx = xx - x0
    dy = yy - y0
    rr_pix = np.sqrt(dx**2 + dy**2)
    rho = rr_pix / rsun_pix

    mu = np.zeros_like(rho, dtype=np.float32)
    inside = rho <= 1.0
    mu[inside] = np.sqrt(np.clip(1.0 - rho[inside]**2, 0.0, 1.0))

    disk_mask = inside & np.isfinite(data)

    geom = {
        "nx": nx, "ny": ny, "crpix1": crpix1, "crpix2": crpix2,
        "x0_numpy": x0, "y0_numpy": y0, "cdelt1": cdelt1, "cdelt2": cdelt2,
        "rsun_obs_arcsec": rsun_obs, "rsun_pix_x": rsun_pix_x, "rsun_pix_y": rsun_pix_y,
        "rsun_pix": rsun_pix, "n_disk_pixels": int(disk_mask.sum()),
    }
    return rr_pix, rho, mu, disk_mask, geom


def build_anisotropic_psf(kernel_size=65, sigmas=C.PSF_SIGMAS, weights=C.PSF_WEIGHTS,
                          anisotropy=C.PSF_ANISOTROPY, phase_deg=C.PSF_PHASE_DEG):
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    ax = np.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    rr2 = xx**2 + yy**2
    phi = np.arctan2(yy, xx)

    psf = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    for w, s, a, ph in zip(weights, sigmas, anisotropy, phase_deg):
        ph_rad = np.deg2rad(ph)
        mod = 1.0 + a * np.cos(2.0 * (phi - ph_rad))
        mod = np.clip(mod, 0.0, None)
        g = np.exp(-0.5 * rr2 / (s**2)) * mod
        g /= g.sum()
        psf += w * g

    psf = np.clip(psf, 0.0, None)
    psf /= psf.sum()
    return psf.astype(np.float32)


def run_psf_correction_safe(data, disk_mask, psf_kernel):
    valid_disk = disk_mask & np.isfinite(data) & (data > 0)
    if valid_disk.sum() == 0:
        raise RuntimeError("valid_disk is empty; check disk_mask and data.")

    disk_median = float(np.nanmedian(data[valid_disk]))
    rl_lo, rl_hi = np.nanpercentile(data[valid_disk], [C.RL_LOW_PCT, C.RL_HIGH_PCT])
    if not np.isfinite(rl_lo) or not np.isfinite(rl_hi) or rl_hi <= rl_lo:
        raise RuntimeError("Invalid RL normalization values.")

    img_for_rl = data.astype(np.float32).copy()
    img_for_rl[~np.isfinite(img_for_rl)] = disk_median
    img_for_rl[~disk_mask] = disk_median

    img_clip = np.clip(img_for_rl, rl_lo, rl_hi)
    img_rl_in = (img_clip - rl_lo) / (rl_hi - rl_lo)
    img_rl_in = np.clip(img_rl_in, 1e-6, 1.0).astype(np.float32)

    psf_safe = psf_kernel.astype(np.float32).copy()
    psf_safe[~np.isfinite(psf_safe)] = 0
    psf_safe = np.clip(psf_safe, 0, None)
    psf_safe = psf_safe / psf_safe.sum()

    data_psf_norm = richardson_lucy(
        image=img_rl_in, psf=psf_safe, num_iter=C.RL_ITER,
        clip=False, filter_epsilon=C.RL_FILTER_EPSILON,
    )
    data_psf_norm = np.nan_to_num(data_psf_norm, nan=0.0, posinf=1.0, neginf=0.0)
    data_psf_norm = np.clip(data_psf_norm, 0.0, 2.0)

    data_psf = data_psf_norm * (rl_hi - rl_lo) + rl_lo
    data_psf = data_psf.astype(np.float32)
    data_psf[~disk_mask] = data[~disk_mask]

    psf_info = {
        "valid_disk_pixels": int(valid_disk.sum()),
        "disk_median": float(disk_median),
        "rl_lo": float(rl_lo), "rl_hi": float(rl_hi),
        "psf_sum": float(psf_safe.sum()),
        "rl_iter": int(C.RL_ITER), "rl_filter_epsilon": float(C.RL_FILTER_EPSILON),
    }
    return data_psf, psf_info


def compute_quiet_sun_ld_profile(image, mu, disk_mask, mu_min=C.LD_MU_MIN, n_bins=C.LD_N_BINS,
                                 quantile=C.LD_QUANTILE, clip_low_pct=C.LD_CLIP_LOW_PCT,
                                 clip_high_pct=C.LD_CLIP_HIGH_PCT):
    fit_mask = (
        disk_mask & np.isfinite(image) & np.isfinite(mu)
        & (image > 0) & (mu >= mu_min) & (mu <= 1.0)
    )
    bins = np.linspace(mu_min, 1.0, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    profile = np.full(n_bins, np.nan, dtype=np.float64)
    n_pix = np.zeros(n_bins, dtype=np.int64)

    for i in range(n_bins):
        m = fit_mask & (mu >= bins[i]) & (mu < bins[i + 1])
        vals = image[m]
        if vals.size > 0:
            lo = np.nanpercentile(vals, clip_low_pct)
            hi = np.nanpercentile(vals, clip_high_pct)
            vals_clip = vals[(vals >= lo) & (vals <= hi)]
            if vals_clip.size > 0:
                profile[i] = np.nanpercentile(vals_clip, quantile)
                n_pix[i] = vals_clip.size

    valid = np.isfinite(profile) & (n_pix > 0)
    return centers, profile, n_pix, valid


def fit_ld_polynomial(mu_centers, profile, valid, degree=C.LD_POLY_DEGREE):
    if valid.sum() < degree + 2:
        raise RuntimeError(f"Too few valid points for degree={degree}. Valid={valid.sum()}")
    coeff = np.polyfit(mu_centers[valid], profile[valid], degree)
    fit_curve = np.polyval(coeff, mu_centers)
    return coeff, fit_curve


def apply_limb_darkening_correction(image, mu, disk_mask, coeff):
    ld_model = np.full_like(image, np.nan, dtype=np.float32)
    inside = disk_mask & np.isfinite(image) & np.isfinite(mu) & (mu > 0)
    ld_vals = np.polyval(coeff, mu[inside])
    ld_vals = np.clip(ld_vals, 1e-6, None)
    ld_model[inside] = ld_vals.astype(np.float32)

    i_ref = float(np.polyval(coeff, 1.0))
    corrected = np.full_like(image, np.nan, dtype=np.float32)
    corrected[inside] = (image[inside] / ld_model[inside]) * i_ref
    return corrected, ld_model, i_ref


def _write_psf_ld_fits(out_path, image, header, source_fits, psf_info, ld_coeff, ld_i_ref):
    """Write the PSF+LD FITS with the processing parameters in its header."""
    hdr = header.copy()
    hdr["BUNIT"] = "DN"
    hdr["SRCFITS"] = Path(source_fits).name[:68]
    hdr["PROCSTG"] = "PSF_LD_CORRECTED"
    hdr["PSFSIZE"] = (int(C.PSF_KERNEL_SIZE), "PSF kernel size")
    hdr["PSFNCMP"] = (5, "Number of Gaussian PSF components")
    hdr["PSFSIG"] = (",".join(map(str, C.PSF_SIGMAS))[:68], "PSF Gaussian sigmas")
    hdr["PSFWGT"] = (",".join(map(str, C.PSF_WEIGHTS))[:68], "PSF Gaussian weights")
    hdr["PSFANI"] = (",".join(map(str, C.PSF_ANISOTROPY))[:68], "PSF anisotropy")
    hdr["PSFPHA"] = (",".join(map(str, C.PSF_PHASE_DEG))[:68], "PSF anisotropy phase")
    hdr["PSFRLIT"] = (int(C.RL_ITER), "Richardson-Lucy iterations")
    hdr["RLFILT"] = (float(C.RL_FILTER_EPSILON), "RL filter epsilon")
    hdr["RLLO"] = (float(psf_info["rl_lo"]), "RL normalization low")
    hdr["RLHI"] = (float(psf_info["rl_hi"]), "RL normalization high")
    hdr["LDMUMIN"] = (float(C.LD_MU_MIN), "LD minimum mu for fit")
    hdr["LDNBINS"] = (int(C.LD_N_BINS), "LD number of mu bins")
    hdr["LDQ"] = (float(C.LD_QUANTILE), "LD quiet-sun quantile")
    hdr["LDPDEG"] = (int(C.LD_POLY_DEGREE), "LD polynomial degree")
    hdr["LDIREF"] = (float(ld_i_ref), "LD reference intensity at mu=1")
    for i, c in enumerate(ld_coeff):
        hdr[f"LDC{i}"] = (float(c), f"LD polynomial coefficient {i}")
    fits.PrimaryHDU(image.astype(np.float32), header=hdr).writeto(out_path, overwrite=C.OVERWRITE_FITS)


# ============================================================================
# PUBLIC: run PSF+LD on one FITS file
# ============================================================================
def correct_psf_ld(raw_fits, out_psf_ld_fits):
    """Raw continuum FITS -> PSF+LD FITS (off-disk pixels NaN). Returns (path, info)."""
    out_psf_ld_fits = Path(out_psf_ld_fits)
    out_psf_ld_fits.parent.mkdir(parents=True, exist_ok=True)

    data, header = load_hmi_fits(raw_fits)
    rr_pix, rho, mu, disk_mask, geom = build_solar_disk_geometry(data, header)

    psf_kernel = build_anisotropic_psf(
        kernel_size=C.PSF_KERNEL_SIZE, sigmas=C.PSF_SIGMAS, weights=C.PSF_WEIGHTS,
        anisotropy=C.PSF_ANISOTROPY, phase_deg=C.PSF_PHASE_DEG)

    t0 = time.time()
    data_psf, psf_info = run_psf_correction_safe(data=data, disk_mask=disk_mask, psf_kernel=psf_kernel)
    t_psf = time.time() - t0
    log.info("[B] PSF (RL %d iter) done: %.1fs | rl_lo=%.1f rl_hi=%.1f",
             C.RL_ITER, t_psf, psf_info["rl_lo"], psf_info["rl_hi"])

    t0 = time.time()
    mu_centers, ld_profile, ld_npix, ld_valid = compute_quiet_sun_ld_profile(
        image=data_psf, mu=mu, disk_mask=disk_mask, mu_min=C.LD_MU_MIN, n_bins=C.LD_N_BINS,
        quantile=C.LD_QUANTILE, clip_low_pct=C.LD_CLIP_LOW_PCT, clip_high_pct=C.LD_CLIP_HIGH_PCT)
    ld_coeff, ld_fit_curve = fit_ld_polynomial(mu_centers, ld_profile, ld_valid, degree=C.LD_POLY_DEGREE)
    data_psf_ld, ld_model, ld_i_ref = apply_limb_darkening_correction(
        image=data_psf, mu=mu, disk_mask=disk_mask, coeff=ld_coeff)
    t_ld = time.time() - t0
    log.info("[B] LD (degree %d, %d/%d valid bins) done: %.1fs | i_ref=%.2f",
             C.LD_POLY_DEGREE, int(ld_valid.sum()), len(ld_valid), t_ld, ld_i_ref)

    # data_psf_ld is NaN off the disk
    _write_psf_ld_fits(out_psf_ld_fits, data_psf_ld, header, raw_fits, psf_info, ld_coeff, ld_i_ref)
    log.info("[B] psf_ld FITS -> %s", out_psf_ld_fits)

    info = {"psf_sec": round(t_psf, 1), "ld_sec": round(t_ld, 1),
            "rl_lo": psf_info["rl_lo"], "rl_hi": psf_info["rl_hi"],
            "ld_i_ref": ld_i_ref, "ld_coeff": [float(c) for c in ld_coeff],
            "ld_valid_bins": int(ld_valid.sum()), "geom": geom}
    return str(out_psf_ld_fits), info


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    raw = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else raw.replace(".continuum.fits", ".psf_ld.fits")
    correct_psf_ld(raw, out)
