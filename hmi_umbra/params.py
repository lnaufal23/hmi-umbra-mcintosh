"""Processing parameters of the image correction (PSF, Richardson-Lucy, limb darkening)
and of the 8-bit opt_F rendering used as segmentation input.

Values are identical to the production configuration of the pipeline.
"""

# ============================================================================
# PSF (five anisotropic Gaussian components) + Richardson-Lucy (Table 1 of the paper)
# ============================================================================
PSF_KERNEL_SIZE = 65
PSF_SIGMAS      = (0.7, 1.4, 2.8, 5.5, 10.0)
PSF_WEIGHTS     = (0.62, 0.22, 0.10, 0.045, 0.015)
PSF_ANISOTROPY  = (0.00, 0.01, 0.02, 0.03, 0.04)
PSF_PHASE_DEG   = (0.0, 20.0, 40.0, 70.0, 100.0)

RL_ITER           = 20
RL_LOW_PCT        = 0.1
RL_HIGH_PCT       = 99.9
RL_FILTER_EPSILON = 1e-6

# ============================================================================
# Limb-darkening correction
# ============================================================================
LD_MU_MIN        = 0.20
LD_N_BINS        = 140
LD_QUANTILE      = 90
LD_POLY_DEGREE   = 3
LD_CLIP_LOW_PCT  = 1
LD_CLIP_HIGH_PCT = 99

# Off-disk pixels of the PSF+LD FITS are NaN.
OVERWRITE_FITS = True

# ============================================================================
# opt_F 8-bit PNG (segmentation and classification input)
# ============================================================================
QS_PERCENTILE_LO        = 40
QS_PERCENTILE_HI        = 70
PIECEWISE_BREAKS        = [0.0, 0.30, 0.85, 1.05]
PIECEWISE_OUTPUTS       = [0, 30, 200, 235]
OFF_DISK_VALUE          = 245
DISK_MASK_SHRINK_FACTOR = 0.97
PNG_COMPRESS_LEVEL      = 3
