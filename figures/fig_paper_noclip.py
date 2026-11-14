"""Paper figures on the unclipped product: (1) per-umbra median intensity vs umbral area and daily minimum vs total AR area,
with bi-exponential fits (fig_perblob_perday_noclip_en.png); (2) the four extreme-value null tests (fig_null_perday_noclip_umb1msh_en.png)."""
import numpy as np, pandas as pd, os, warnings
from scipy.optimize import curve_fit
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from hmi_umbra import paths
OUT = paths.get('paper_figs'); DESK = paths.get('check_plots')
NC = paths.get('stats_unclipped_out')
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'legend.fontsize': 9})

def biexp(A, a1, t1, a2, t2, c): return a1 * np.exp(-A / t1) + a2 * np.exp(-A / t2) + c
def fitbe(x, y, starts, bd):
    best = None
    for p0 in starts:
        try: p, cov = curve_fit(biexp, x, y, p0=p0, bounds=bd, maxfev=100000)
        except Exception: continue
        rss = np.sum((y - biexp(x, *p)) ** 2)
        if best is None or rss < best[1]: best = (p, rss)
    p = best[0]
    if p[1] > p[3]: p = np.array([p[2], p[3], p[0], p[1], p[4]])
    return p, 1 - best[1] / np.sum((y - y.mean()) ** 2)

B = pd.read_csv(NC + '/umbra_blob_intensity_area_noclipT1.csv')
B = B[(B.mu >= 0.7) & (B.area_msh >= 1) & (B.n_pix >= 4) & (B.umb_median_ratio <= 0.509)]
A, Y = B.area_msh.values, B.umb_median_ratio.values
e = np.arange(1, A.max() + 5, 5.0); xb, yb = [], []
for lo, hi in zip(e[:-1], e[1:]):
    m = (A >= lo) & (A < hi)
    if m.sum() >= 15: xb.append(0.5 * (lo + hi)); yb.append(np.median(Y[m]))
xb, yb = np.array(xb), np.array(yb)
pu, r2u = fitbe(xb, yb, [[0.16, 5.7, 0.03, 82, 0.3], [0.19, 17, 0.27, 500, 0.0], [0.15, 10, 0.1, 150, 0.15]], ([0, 0.1, 0, 5, 0], [5, 200, 5, 1e5, 1]))
et = 10 ** np.arange(np.log10(131), np.log10(A.max()) + 0.1, 0.1); xt, yt = [], []
for lo, hi in zip(et[:-1], et[1:]):
    m = (A >= lo) & (A < hi)
    if m.sum() >= 5: xt.append(np.median(A[m])); yt.append(np.median(Y[m]))
print('per-umbra N=%d bins=%d | A1=%.4f t1=%.2f A2=%.4f t2=%.1f c=%.4g R2=%.4f' % (len(B), len(xb), *pu, r2u))

P = pd.read_csv(NC + '/perday_ar_area_umbmin_noclipT1.csv', dtype={'date': str})
Bm = pd.read_csv(NC + '/umbra_blob_intensity_area_noclipT1.csv', dtype={'date': str})
Bm = Bm[(Bm.blob_id != -1) & (Bm.mu >= 0.7) & (Bm.area_msh >= 1)]
P = P.set_index('date').join(Bm.groupby('date').umb_min_ratio.min().rename('ymin'), how='inner').reset_index()
x, y = P.area_ar_msh.values, P.ymin.values
e20 = np.arange(0, x.max() + 20, 20.0); xd, yd = [], []
for lo, hi in zip(e20[:-1], e20[1:]):
    m = (x >= lo) & (x < hi)
    if m.sum() >= 10: xd.append(0.5 * (lo + hi)); yd.append(np.median(y[m]))
xd, yd = np.array(xd), np.array(yd)
pd_, r2d = fitbe(xd, yd, [[0.1, 30, 0.05, 900, 0.29], [0.1, 30, 0.05, 900, 0.10]], ([0, 1, 0, 50, 0], [1, 1000, 1, 20000, 1]))
print('per-day N=%d bins=%d | A1=%.4f t1=%.1f A2=%.4f t2=%.0f c=%.4f R2=%.4f' % (len(P), len(xd), *pd_, r2d))

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
ax[0].scatter(A, Y, s=1.5, c='0.75', alpha=0.35, rasterized=True, label='individual umbrae')
ax[0].plot(xb, yb, 'ko', ms=4, label='binned median (5 MSH)')
ax[0].plot(xt, yt, 'D', ms=6, mfc='none', mec='k', mew=1.2, label='median, umbrae > 131 MSH (not fitted)')
Ai = np.linspace(1, 131, 400); Ao = np.linspace(131, 260, 200)
ax[0].plot(Ai, biexp(Ai, *pu), 'C3', lw=1.8, label='bi-exponential fit ($R^2$ = %.3f)' % r2u)
ax[0].plot(Ao, biexp(Ao, *pu), 'C3', lw=1.2, ls=':')
ax[0].axvline(131, color='0.5', ls='-.', lw=0.8)
ax[0].set_xlim(0, 260); ax[0].set_ylim(0.05, 0.55)
ax[0].set_xlabel('Umbral area [MSH]'); ax[0].set_ylabel(r'$I_{\rm umb,med}/I_{qs}$')
ax[0].legend(loc='upper right', frameon=False); ax[0].text(0.02, 0.96, '(a)', transform=ax[0].transAxes, fontsize=13, va='top')
ax[1].scatter(x, y, s=2, c='0.75', alpha=0.5, rasterized=True, label='daily minimum')
ax[1].plot(xd, yd, 'ko', ms=3.5, label='binned median (20 MSH)')
Ad = np.linspace(xd.min(), xd.max(), 600)
ax[1].plot(Ad, biexp(Ad, *pd_), 'C3', lw=1.8, label='bi-exponential fit ($R^2$ = %.3f)' % r2d)
ax[1].set_xlabel('Total AR area per day [MSH]'); ax[1].set_ylabel(r'$I_{\rm umb,min}/I_{qs}$')
ax[1].legend(loc='upper right', frameon=False); ax[1].text(0.02, 0.96, '(b)', transform=ax[1].transAxes, fontsize=13, va='top')
for a_ in ax: a_.grid(alpha=0.25)
plt.tight_layout()
for d in (OUT, DESK): os.makedirs(d, exist_ok=True); plt.savefig(d + '/fig_perblob_perday_noclip_en.png', dpi=300)
plt.close()

ND = paths.get('null_tests_out') + '/noclip_umb1msh'
kinds = ['null1_count_only', 'null2_area_alltime', 'null3_area_sameyear', 'null4_area_same27d']
titles = ['(1) whole sample', '(2) similar area, any epoch', '(3) similar area, same year', '(4) similar area, same 27-day block']
Bl = pd.read_csv(NC + '/umbra_blob_intensity_area_noclipT1.csv', dtype={'date': str})
q = Bl[(Bl.blob_id != -1) & (Bl.mu >= 0.7) & (Bl.area_msh >= 1)]
ymin = q.groupby('date').umb_min_ratio.min()
Pd = P.copy()
fig, axes = plt.subplots(2, 2, figsize=(12, 8.6), sharex=True, sharey=True)
for a_, k, tt in zip(axes.ravel(), kinds, titles):
    b = pd.read_csv('%s/bands_%s.csv' % (ND, k))
    a_.scatter(Pd.area_ar_msh, Pd.ymin, s=2, c='0.8', rasterized=True)
    a_.fill_between(b.x_bin, b.null_lo, b.null_hi, color='tab:blue', alpha=0.3, label='null, 95% range')
    a_.plot(b.x_bin, b.null_med, color='tab:blue', lw=1, label='null, median')
    a_.plot(b.x_bin, b.obs, 'ko', ms=3, label='observed binned median')
    a_.set_title(tt, fontsize=11); a_.grid(alpha=0.25)
axes[0, 0].legend(frameon=False, loc='upper right')
for a_ in axes[1]: a_.set_xlabel('Total AR area per day [MSH]')
for a_ in axes[:, 0]: a_.set_ylabel(r'$I_{\rm umb,min}/I_{qs}$')
plt.tight_layout()
for d in (OUT, DESK): plt.savefig(d + '/fig_null_perday_noclip_umb1msh_en.png', dpi=300)
print('saved in', OUT, 'and', DESK)
