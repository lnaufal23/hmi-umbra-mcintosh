#!/usr/bin/env python3
"""
Before/after visualization of the PSF + LD correction (2014-10-23, AR 12192).

Panels:
  Panel A : full disk, raw vs psf_ld (side by side)
  Panel B : zoom on AR 12192 (400x400 px); panel_B_zoom.png is the paper figure
  Panel C : 1D intensity profile across the umbra
  Panel D : gradient magnitude before/after
  Panel E : intensity histogram before/after

Input in <psf_demo_dir>: the raw FITS of the day (input/) and its PSF-only and PSF+LD products from a demo run of the
correction with the parameters of hmi_umbra/params.py (output/).
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pathlib import Path

from hmi_umbra import paths

# === CONFIGURATION ===
DEMO        = Path(paths.get("psf_demo_dir"))
RAW_PATH    = DEMO / "input/hmi.ic_720s.20141023_000000_TAI.1.continuum.fits"
PSF_PATH    = DEMO / "output/hmi.ic_720s.20141023_000000_TAI.continuum.psf.fits"
PSF_LD_PATH = DEMO / "output/hmi.ic_720s.20141023_000000_TAI.continuum.psf_ld.fits"
OUT_DIR     = DEMO / "output"

# Zoom region for AR 12192 (approximate center on 2014-10-23 00:00 UTC)
# AR 12192 is near S14 W20 -> conversion to HMI 4096x4096 pixel coordinates:
# solar center = (2047.5, 2047.5), R_sun ~1920 px (varies per day)
# x ~ 2047.5 + R_sun * cos(latitude) * sin(longitude)
# y ~ 2047.5 - R_sun * sin(latitude)
# For S14 W20:   x ~ 2047 + 1920 * cos(14) * sin(20) ~ 2047 + 637 = 2684
#                y ~ 2047 + 1920 * sin(14)            ~ 2047 + 464 = 2511 (FITS y axis usually flipped)
# NOTE: HMI coordinate orientation: y positive to the north, so S14 means y < center
# Corrected automatically from the darkest region if it does not match
ZOOM_CENTER  = (2684, 2511)   # (x, y) initial guess for AR 12192
ZOOM_SIZE    = 400            # pixels (400x400 region)

# Length of the line profile (across the umbra)
PROFILE_LENGTH = 200          # pixels

OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_fits_2d(path):
    """Load FITS 2D data + header."""
    with fits.open(path) as hdul:
        hdul.verify('silentfix')
        for h in hdul:
            if h.data is not None and h.data.ndim == 2:
                return np.asarray(h.data, dtype=np.float32), h.header.copy()
    raise RuntimeError(f"No 2D data in {path}")


def find_darkest_region(data, hint_center, search_radius=200):
    """
    Refine the zoom center: find the darkest pixel within a search window around the hint.
    Useful if the hint coordinates are slightly off.
    """
    x0, y0 = hint_center
    h, w = data.shape
    x_lo, x_hi = max(0, x0 - search_radius), min(w, x0 + search_radius)
    y_lo, y_hi = max(0, y0 - search_radius), min(h, y0 + search_radius)
    window = data[y_lo:y_hi, x_lo:x_hi]
    # Position of the minimum (darkest umbra)
    iy, ix = np.unravel_index(np.nanargmin(window), window.shape)
    new_center = (ix + x_lo, iy + y_lo)
    print(f"  refined center: {hint_center} -> {new_center} "
          f"(min intensity = {np.nanmin(window):.1f} DN/s)")
    return new_center


def crop_region(data, center, size):
    """Crop a size x size region centered on center=(x,y)."""
    x, y = center
    h = size // 2
    return data[y-h : y+h, x-h : x+h]


def compute_gradient_magnitude(image):
    """Gradient magnitude |∇I| (Sobel-like, numpy gradient)."""
    gy, gx = np.gradient(image.astype(np.float32))
    return np.sqrt(gx**2 + gy**2)


def plot_figure_a_fulldisk(raw, psf_ld, out_path):
    """Panel A: Full-disk before vs after."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Use the 1-99.5 percentiles for vmin/vmax (avoid saturation by outliers)
    vmin_r, vmax_r = np.nanpercentile(raw,    [1, 99.5])
    vmin_p, vmax_p = np.nanpercentile(psf_ld, [1, 99.5])

    axes[0].imshow(raw,    cmap='gray', origin='lower', vmin=vmin_r, vmax=vmax_r)
    axes[0].set_title("(a) Raw HMI Ic_720s — 2014-10-23 00:00 UTC", fontsize=12)
    axes[0].set_xlabel("x (pixel)"); axes[0].set_ylabel("y (pixel)")

    axes[1].imshow(psf_ld, cmap='gray', origin='lower', vmin=vmin_p, vmax=vmax_p)
    axes[1].set_title("(b) After PSF deconvolution + Limb-darkening correction", fontsize=12)
    axes[1].set_xlabel("x (pixel)"); axes[1].set_ylabel("y (pixel)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  saved: {out_path}")


def plot_figure_b_zoom(raw_crop, psf_crop, psf_ld_crop, out_path):
    """Panel B: 3-panel zoom (raw, psf, psf_ld) of AR 12192."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Same intensity scale for a fair comparison
    vmin = min(np.nanpercentile(raw_crop,    1),
               np.nanpercentile(psf_crop,    1),
               np.nanpercentile(psf_ld_crop, 1))
    vmax = max(np.nanpercentile(raw_crop,    99.5),
               np.nanpercentile(psf_crop,    99.5),
               np.nanpercentile(psf_ld_crop, 99.5))

    titles = ["(a) Raw", "(b) PSF deconvolved", "(c) PSF + LD corrected"]
    for ax, img, title in zip(axes, [raw_crop, psf_crop, psf_ld_crop], titles):
        im = ax.imshow(img, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(title + " — AR 12192 zoom", fontsize=12)
        ax.set_xlabel("x (pixel)"); ax.set_ylabel("y (pixel)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Intensity (DN/s)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  saved: {out_path}")


def plot_figure_c_profile(raw_crop, psf_ld_crop, out_path):
    """Panel C: horizontal 1D intensity profile across the umbra."""
    h, w = raw_crop.shape
    # Row containing the darkest umbra (minimum)
    row_idx = int(np.nanargmin(np.nanmean(psf_ld_crop, axis=1)))
    # Profile along that row
    x_axis = np.arange(w)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_axis, raw_crop[row_idx, :],    label='Raw',                 color='gray',  linewidth=1.5)
    ax.plot(x_axis, psf_ld_crop[row_idx, :], label='PSF + LD corrected', color='red',   linewidth=1.5)
    ax.set_xlabel("Pixel position (along x)")
    ax.set_ylabel("Intensity (DN/s)")
    ax.set_title(f"1D intensity profile across umbra (row y={row_idx})", fontsize=12)
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  saved: {out_path}")
    return row_idx


def plot_figure_d_gradient(raw_crop, psf_ld_crop, out_path):
    """Panel D: gradient magnitude before/after (motivation for the gradient pipeline)."""
    grad_raw    = compute_gradient_magnitude(raw_crop)
    grad_psf_ld = compute_gradient_magnitude(psf_ld_crop)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    vmax = max(np.nanpercentile(grad_raw, 99), np.nanpercentile(grad_psf_ld, 99))

    im0 = axes[0].imshow(grad_raw,    cmap='inferno', origin='lower', vmin=0, vmax=vmax)
    axes[0].set_title("(a) |∇I| Raw", fontsize=12)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(grad_psf_ld, cmap='inferno', origin='lower', vmin=0, vmax=vmax)
    axes[1].set_title("(b) |∇I| PSF + LD corrected", fontsize=12)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("x (pixel)"); ax.set_ylabel("y (pixel)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  saved: {out_path}")

    # Quantitative statistics
    print(f"\n  Gradient stats (zoom region):")
    print(f"    Raw    : median = {np.nanmedian(grad_raw):.2f}, "
          f"p99 = {np.nanpercentile(grad_raw, 99):.2f}, "
          f"max = {np.nanmax(grad_raw):.2f}")
    print(f"    PSF+LD : median = {np.nanmedian(grad_psf_ld):.2f}, "
          f"p99 = {np.nanpercentile(grad_psf_ld, 99):.2f}, "
          f"max = {np.nanmax(grad_psf_ld):.2f}")
    print(f"    Ratio (PSF+LD / Raw): "
          f"median = {np.nanmedian(grad_psf_ld)/np.nanmedian(grad_raw):.2f}x, "
          f"p99 = {np.nanpercentile(grad_psf_ld, 99)/np.nanpercentile(grad_raw, 99):.2f}x")



def plot_figure_e_histogram(raw_crop, psf_ld_crop, out_path):
    """Panel E: histogram of the intensity distribution before/after in the AR 12192 region."""
    raw_vals    = raw_crop[np.isfinite(raw_crop)].ravel()
    psf_ld_vals = psf_ld_crop[np.isfinite(psf_ld_crop)].ravel()

    lo = min(np.nanpercentile(raw_vals, 0.5), np.nanpercentile(psf_ld_vals, 0.5))
    hi = max(np.nanpercentile(raw_vals, 99.5), np.nanpercentile(psf_ld_vals, 99.5))
    bins = np.linspace(lo, hi, 120)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(raw_vals,    bins=bins, color='gray', alpha=0.6, label='Raw', density=True)
    ax.hist(psf_ld_vals, bins=bins, color='red',  alpha=0.5, label='PSF + LD corrected', density=True)
    ax.set_xlabel("Intensity (DN/s)")
    ax.set_ylabel("Normalized frequency")
    ax.set_title("Intensity distribution before vs after correction \u2014 AR 12192", fontsize=12)
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  saved: {out_path}")

    print(f"\n  Intensity stats (zoom region):")
    print(f"    Raw    : median = {np.nanmedian(raw_vals):.1f}, std = {np.nanstd(raw_vals):.1f}")
    print(f"    PSF+LD : median = {np.nanmedian(psf_ld_vals):.1f}, std = {np.nanstd(psf_ld_vals):.1f}")


def main():
    print("=" * 70)
    print("Before/after PSF + LD correction — AR 12192 (2014-10-23)")
    print("=" * 70)

    # --- Load all data ---
    print("\n[Load FITS]")
    raw,    _ = load_fits_2d(RAW_PATH);    print(f"  raw    : {raw.shape}")
    psf,    _ = load_fits_2d(PSF_PATH);    print(f"  psf    : {psf.shape}")
    psf_ld, _ = load_fits_2d(PSF_LD_PATH); print(f"  psf_ld : {psf_ld.shape}")

    # --- Refine the zoom center (darkest umbra around the hint) ---
    print("\n[Refine zoom center]")
    zoom_center = find_darkest_region(psf_ld, ZOOM_CENTER, search_radius=200)

    # --- Crop ---
    raw_crop    = crop_region(raw,    zoom_center, ZOOM_SIZE)
    psf_crop    = crop_region(psf,    zoom_center, ZOOM_SIZE)
    psf_ld_crop = crop_region(psf_ld, zoom_center, ZOOM_SIZE)
    print(f"  crop shape: {raw_crop.shape}")

    # --- Generate figures ---
    print("\n[Generate Panel A: Full-disk before/after]")
    plot_figure_a_fulldisk(raw, psf_ld, OUT_DIR / "panel_A_fulldisk.png")

    print("\n[Generate Panel B: Zoom-in 3-panel]")
    plot_figure_b_zoom(raw_crop, psf_crop, psf_ld_crop, OUT_DIR / "panel_B_zoom.png")

    print("\n[Generate Panel C: 1D intensity profile]")
    plot_figure_c_profile(raw_crop, psf_ld_crop, OUT_DIR / "panel_C_profile.png")

    print("\n[Generate Panel D: Gradient magnitude]")
    plot_figure_d_gradient(raw_crop, psf_ld_crop, OUT_DIR / "panel_D_gradient.png")

    print("\n[Generate Panel E: Intensity histogram]")
    plot_figure_e_histogram(raw_crop, psf_ld_crop, OUT_DIR / "panel_E_histogram.png")

    print(f"\n[DONE] All figures saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
