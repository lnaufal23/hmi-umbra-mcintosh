"""Supplementary plot (not in the manuscript): per-umbra intensity versus umbral area on the
clipped (v4) and unclipped photometric products, in 0.1 dex area bins.
(a) per-umbra median with IQR bands, the power-law fit to the unclipped medians and the mean
clipping floor (0.2985 I_qs); (b) median, eroded median (2 px), darkest 25% and minimum on the
unclipped product, with the minimum on the clipped product."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, os
from hmi_umbra import paths
def load(p):
    d = pd.read_csv(p)
    return d[(d.mu >= 0.7) & (d.area_msh >= 1) & (d.n_pix >= 4) & (d.umb_median_ratio <= 0.509)]
o = load(os.path.join(paths.get('stats_clipped_out'), 'umbra_blob_intensity_area_v4.csv'))
n = load(os.path.join(paths.get('stats_unclipped_out'), 'umbra_blob_intensity_area_noclipT1.csv'))
e = 10 ** np.arange(0, np.log10(max(o.area_msh.max(), n.area_msh.max())) + 0.1, 0.1)
def bmed(d, col, nmin=15):
    x, y, lo, hi = [], [], [], []
    for a, b in zip(e[:-1], e[1:]):
        m = (d.area_msh >= a) & (d.area_msh < b)
        v = d.loc[m, col].dropna()
        if len(v) >= nmin:
            x.append(np.median(d.area_msh[m])); y.append(np.median(v))
            lo.append(np.percentile(v, 25)); hi.append(np.percentile(v, 75))
    return map(np.array, (x, y, lo, hi))
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
for d, lab, col in ((o, 'clipped (v4)', 'C0'), (n, 'unclipped', 'C3')):
    x, y, lo, hi = bmed(d, 'umb_median_ratio')
    ax[0].fill_between(x, lo, hi, color=col, alpha=0.15)
    ax[0].plot(x, y, 'o-', color=col, ms=4, label=lab + ' (N=%d)' % len(d))
x, y, _, _ = bmed(n, 'umb_median_ratio')
ax[0].plot(x, 0.5623 * x ** -0.2018, 'k--', lw=1, label='0.562 A$^{-0.202}$ (unclipped fit)')
ax[0].axhline(0.2985, color='grey', ls=':', lw=1, label='mean clipping floor 0.2985')
ax[0].set_title('(a) per-umbra median, 0.1 dex bins, IQR band')
for col, lab, c in (('umb_median_ratio', 'median', 'C3'), ('umb_median_erode2_ratio', 'median erode-2', 'C1'), ('umb_dark25_ratio', 'dark25', 'C2'), ('umb_min_ratio', 'minimum', 'C4')):
    x, y, _, _ = bmed(n, col)
    ax[1].plot(x, y, 'o-', ms=4, color=c, label=lab + ' (unclipped)')
x, y, _, _ = bmed(o, 'umb_min_ratio')
ax[1].plot(x, y, 's--', ms=4, color='C0', label='minimum (clipped)')
ax[1].axhline(0.2985, color='grey', ls=':', lw=1)
ax[1].set_title('(b) per-umbra intensity statistics, 0.1 dex bins')
for a in ax:
    a.set_xscale('log'); a.set_xlabel('umbral area (MSH)'); a.grid(alpha=0.3); a.legend(fontsize=8)
ax[0].set_ylabel('I / I$_{qs}$')
plt.tight_layout()
out = os.path.join(paths.get('check_plots'), 'compare_median_area_clipped_vs_unclipped.png'); os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=130); print('saved:', out)
