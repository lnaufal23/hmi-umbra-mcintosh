"""Supplementary plot (not in the manuscript): single exponential, bi-exponential and power law
fitted to the 5 MSH binned medians of the per-umbra median intensity on the unclipped product
(1-131 MSH), shown with the clipped binned medians, the unfitted tail above 131 MSH (0.1 dex
bins, >= 5 umbrae) and the mean clipping floor, on linear and logarithmic area axes."""
import numpy as np, pandas as pd, os, warnings
from scipy.optimize import curve_fit
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from hmi_umbra import paths
def load(p):
    d = pd.read_csv(p)
    return d[(d.mu >= 0.7) & (d.area_msh >= 1) & (d.n_pix >= 4) & (d.umb_median_ratio <= 0.509)]
o = load(os.path.join(paths.get('stats_clipped_out'), 'umbra_blob_intensity_area_v4.csv'))
n = load(os.path.join(paths.get('stats_unclipped_out'), 'umbra_blob_intensity_area_noclipT1.csv'))
def b5(d):
    A, Y = d.area_msh.values, d.umb_median_ratio.values
    e = np.arange(1, A.max() + 5, 5.0); x, y = [], []
    for lo, hi in zip(e[:-1], e[1:]):
        m = (A >= lo) & (A < hi)
        if m.sum() >= 15: x.append(0.5 * (lo + hi)); y.append(np.median(Y[m]))
    return np.array(x), np.array(y)
def tail(d):
    A, Y = d.area_msh.values, d.umb_median_ratio.values
    e = 10 ** np.arange(np.log10(131), np.log10(A.max()) + 0.1, 0.1); x, y, k = [], [], []
    for lo, hi in zip(e[:-1], e[1:]):
        m = (A >= lo) & (A < hi)
        if m.sum() >= 5: x.append(np.median(A[m])); y.append(np.median(Y[m])); k.append(m.sum())
    return np.array(x), np.array(y), k
ex1 = lambda A, A1, t1, c: A1 * np.exp(-A / t1) + c
ex2 = lambda A, A1, t1, A2, t2, c: A1 * np.exp(-A / t1) + A2 * np.exp(-A / t2) + c
pl = lambda A, a, b, c: a * A ** (-b) + c
xn, yn = b5(n); xo, yo = b5(o); xt, yt, kt = tail(n)
F = {}
F['single exponential'] = (ex1, curve_fit(ex1, xn, yn, p0=[0.2, 25, 0.22], bounds=([0, 0.1, 0], [5, 1e4, 1]), maxfev=50000)[0])
F['bi-exponential'] = (ex2, curve_fit(ex2, xn, yn, p0=[0.19, 17, 0.27, 500, 0.0], bounds=([0, 0.1, 0, 5, 0], [5, 200, 5, 1e5, 1]), maxfev=50000)[0])
F['power law'] = (pl, curve_fit(pl, xn, yn, p0=[0.56, 0.2, 0.0], bounds=([0, 0, 0], [5, 5, 1]), maxfev=50000)[0])
for k, (f, p) in F.items():
    r = yn - f(xn, *p); print('%-22s %s | RSS %.5f' % (k, ' '.join('%.4g' % v for v in p), np.sum(r ** 2)))
print('tail >131 MSH (log bins, >=5 umbrae):', ', '.join('%.0f MSH: %.3f (n=%d)' % z for z in zip(xt, yt, kt)))
rng = np.random.default_rng(0); s = n.sample(min(8000, len(n)), random_state=0)
Ain, Aout = np.linspace(1, 131, 400), np.linspace(131, 250, 200)
col = {'single exponential': 'C2', 'bi-exponential': 'C3', 'power law': 'k'}
fig, ax = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
for i, a in enumerate(ax):
    a.scatter(s.area_msh, s.umb_median_ratio, s=2, c='0.75', alpha=0.4, rasterized=True)
    a.plot(xo, yo, 's', ms=4, mfc='none', mec='C0', label='5 MSH binned median, clipped')
    a.plot(xn, yn, 'o', ms=4, color='C1', label='5 MSH binned median, unclipped (fitted)')
    a.plot(xt, yt, 'D', ms=6, mfc='none', mec='C1', mew=1.5, label='median >131 MSH, unclipped (not fitted)')
    for k, (f, p) in F.items():
        a.plot(Ain, f(Ain, *p), color=col[k], lw=1.6, label=k)
        a.plot(Aout, f(Aout, *p), color=col[k], lw=1, ls=':')
    a.axhline(0.2985, color='C0', ls='--', lw=0.8, label='clipping floor 0.2985')
    a.axvline(131, color='0.5', ls='-.', lw=0.8)
    a.set_xlabel('umbral area (MSH)'); a.grid(alpha=0.3); a.set_ylim(0.05, 0.52)
ax[0].set_xlim(0, 250); ax[0].set_title('linear area axis')
ax[1].set_xscale('log'); ax[1].set_xlim(0.9, 300); ax[1].set_title('logarithmic area axis')
ax[0].set_ylabel('per-umbra median I / I$_{qs}$'); ax[0].legend(fontsize=7.5, loc='upper right')
plt.tight_layout(); f = os.path.join(paths.get('check_plots'), 'fit_models_on_unclipped_data.png'); os.makedirs(os.path.dirname(f), exist_ok=True); plt.savefig(f, dpi=130); print('saved:', f)
