# hmi-umbra-mcintosh

Code for the paper *Umbral Darkness and Interpretable McIntosh Classification of Solar Active
Regions from 14 Years of PSF-Corrected SDO/HMI Continuum Images* (Naufal, Herdiwijaya, Irwanto,
Satrya, Dermawan, Soegiartini, Vierdayanti, Kesumaningrum, Iwata, Alvyanita, Utomo).

The pipeline corrects SDO/HMI `hmi.Ic_720s` continuum images for the instrumental PSF and limb
darkening, segments umbrae and penumbrae with a U-Net, measures umbral intensity against area,
and classifies active regions in the McIntosh (Zpc) scheme from geometric descriptors, a CNN and
their ensemble. The data products (umbra catalog, daily I_qs, audited labels, descriptors, model
predictions and trained models) are released on Zenodo:
[doi:10.5281/zenodo.24051054](https://doi.org/10.5281/zenodo.24051054).

## Installation

```bash
conda env create -f environment.yml
conda activate hmi-umbra
pip install -e .
```

`pip install -e .` makes the shared package `hmi_umbra` (parameters and path configuration)
importable from every script.

## Configuration

All data and output locations are read from `config.yaml` at the repository root. Edit the
entries under `paths:` for your machine, or point the environment variable `HMI_UMBRA_CONFIG` to
a copy of the file. Entries under `hosts.<hostname>:` override the common entries on that host.
The defaults are the locations used for the paper.

The PSF, Richardson-Lucy, limb-darkening and opt_F parameters are in `hmi_umbra/params.py`.

Scripts are run from the repository root, e.g. `python photometry/stats_v4.py`. Scripts in
`classification/` import sibling modules from their own folder and must be run as files
(`python classification/train_v7.py 0 1 2`), not imported.

## Pipeline and paper sections

### 1. Correction (Section 3.1)

| Script | Purpose |
|---|---|
| `correction/correct_psf_ld.py` | PSF deconvolution (five anisotropic Gaussians, Richardson-Lucy) and limb-darkening correction (clipped PSF+LD product) |
| `correction/make_optf.py` | PSF+LD FITS to 8-bit opt_F PNG, the U-Net input |

### 2. Segmentation (Section 4.1)

| Script | Purpose |
|---|---|
| `segmentation/train_seg_crops.py` | U-Net (ResNet-34) training on AR crops, three classes |
| `segmentation/infer_masks_fulldisk.py` | Full-disk masks in a single pass |
| `segmentation/eval_unet_dateclean.py` | Test-set and date-clean evaluation |

### 3. Photometry and calibration (Sections 3.1-3.3, 4.2, 4.3)

| Script | Purpose |
|---|---|
| `photometry/stats_v4.py` | Per-umbra and per-day tables on the clipped product |
| `photometry/remeasure_masks_v4.py` | Daily I_qs, class statistics and spot counts from the masks |
| `photometry/threshold_histcross_v4.py` | Umbra-penumbra and penumbra-photosphere boundaries (Section 3.3) |
| `photometry/photometry_unclipped.py` | Umbral photometry on the unclipped product |
| `photometry/boundaries_unclipped.py` | Boundaries on the unclipped product |
| `photometry/rms_contrast.py` | Quiet-Sun RMS contrast and photospheric distribution |
| `photometry/silso_correlation.py` | Daily spot count versus SILSO (Section 4.2) |

### 4. Umbral darkness versus area (Section 5, Appendix A)

| Script | Purpose |
|---|---|
| `darkness_fits/biexp_final.py` | Main per-umbra fit (Table 2) |
| `darkness_fits/fit_variants.py` | Fit variants and model comparison |
| `darkness_fits/profile_c.py` | Profile of the asymptotic level c |
| `darkness_fits/null_tests_perday.py` | Per-day relation and extreme-value null tests (Section 5.2) |
| `darkness_fits/robust_table_noclip.py` | Robustness table (Appendix A) |
| `darkness_fits/secular_trend.py` | Secular trend at fixed area (Section 3.2) |

### 5. McIntosh classification (Section 6)

The scripts read internal working tables (CVAT annotation exports and audit tables) that are not
released. Their content is released on Zenodo in consolidated form: `ar_crops_audited.csv` holds,
for every crop, the box, the SRS and audited labels, the partition, the 28 descriptors and the
out-of-fold and locked-test predictions, and `models/` holds the trained segmentation, CNN,
XGBoost, fusion and confidence-head models. The scripts that built the working tables (CVAT tasks
and their export) are not part of this repository.

| Script | Purpose |
|---|---|
| `classification/crop_fits_4096.py` | AR crops (FITS and opt_F JPG) from the annotated boxes (Section 6.1) |
| `classification/srs_download.py` | NOAA SRS download and parsing (module) |
| `classification/srs_crossmatch.py` | SRS labels, position and extent per crop (Section 2.2) |
| `classification/extract_morph.py` | The geometric descriptors (Section 6.3) |
| `classification/train_audit_native.py` | CNN definition and training, one fold (module) |
| `classification/run_folds_v5.py`, `dump_and_decode.py`, `dump_all_splits.py` | Baseline CNN folds on v5 and their outputs |
| `classification/run_variant.py`, `eval_variants.py` | Hyperparameter search (variant A selected) |
| `classification/fusion_A_def_rule.py`, `conf_head.py` | Variant-A fusion and confidence head on v5; `conf_head.py` also holds the descriptor definitions reused by `xgb_v7.py` |
| `classification/train_v7.py` | CNN variant A on the final labels (v7), seven folds |
| `classification/v7_eval.py` | v7 fusion and the relabeling-effect matrix |
| `classification/xgb_v7.py` | XGBoost on the descriptors (main model), seven-fold OOF |
| `classification/ensemble_xgb_fusion.py` | Ensemble weight from cross-validation |
| `classification/conf_ensemble.py` | Confidence estimate (fourteen signals) |
| `classification/feature_importance.py` | SHAP attributions (Section 6.7) |
| `classification/check_lock.py` | Final split around the locked test set |
| `classification/train_final_lock.py` | Final CNN on the pool |
| `classification/final_eval_lock.py` | Locked-test evaluation (Section 6.6, Table 3) |
| `classification/export_final_models.py` | Rebuilds and saves the locked-test XGBoost, fusion and confidence-head models released on Zenodo |

Order on the locked test set: `check_lock.py` -> `train_final_lock.py` -> `final_eval_lock.py`
(requires `extract_morph.py`, `xgb_v7.py`, `v7_eval.py` and `conf_ensemble.py` outputs).

### 6. Figures and tables

| Script | Output |
|---|---|
| `figures/psf_visualize_20141023.py` | PSF+LD correction panels (Figure 1 = `panel_B_zoom.png`) |
| `figures/regen_figs_v4mask.py` | I_qs time series, spot count versus SILSO |
| `figures/fig_threshold_histcross_v4mask.py` | Boundary calibration |
| `figures/fig_paper_noclip.py` | Darkness-area fits and null tests |
| `figures/fig_classification.py` | Confusion matrices and coverage-accuracy |
| `figures/plot_feature_importance_en.py` | SHAP figure |
| `figures/make_paper_tables.py` | Classification tables (LaTeX) |
| `figures/supplementary/` | Supplementary plots not in the paper: clipped versus unclipped intensities, and the three fitted models on linear and logarithmic area axes |

## Input data

- SDO/HMI `hmi.Ic_720s` continuum images from JSOC, one per day at 00:00 TAI (2012-2026).
- NOAA SWPC Solar Region Summaries.
- SILSO daily total sunspot number (`SN_d_tot_V2.0.csv`).
- From the Zenodo release: the umbra catalog (`umbra_catalog.csv`), the daily quiet-Sun
  intensity (`daily_iqs.csv`), the audited AR crop table (`ar_crops_audited.csv`) and the trained
  models (`models/`).

## License

MIT, see `LICENSE`.
