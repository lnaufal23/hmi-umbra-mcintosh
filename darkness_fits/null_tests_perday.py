"""Per-day relation and its extreme-value null tests (Section 5.2, Figure 7).

Main fit: bi-exponential to the binned daily darkest umbral intensity versus total AR area (20-MSH
bins with >= 10 days). Null tests: every umbra is replaced by one drawn at random from the whole
sample (Null 1) or from umbrae of similar area (25 logarithmic area bins) in any epoch (Null 2),
the same year (Null 3) or the same 27-day block (Null 4); the daily minima and the fit are
recomputed (bins of >= 3 days at the median area, 500 replicates).

Usage: python null_tests_perday.py --blob PER_UMBRA.csv --pday PER_DAY.csv --tag NAME [--amin 1] [--nrep 500] [--workers N]
"""
import os, sys, json, argparse
import numpy as np, pandas as pd
from scipy.optimize import curve_fit
from multiprocessing import Pool
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

from hmi_umbra import paths

ap = argparse.ArgumentParser()
ap.add_argument('--blob', required=True); ap.add_argument('--pday', required=True)
ap.add_argument('--amin', type=float, default=1.0); ap.add_argument('--tag', required=True)
ap.add_argument('--nrep', type=int, default=500); ap.add_argument('--workers', type=int, default=28)
a = ap.parse_args()
OUT = os.path.join(paths.get('null_tests_out'), a.tag); os.makedirs(OUT, exist_ok=True)
MU_MIN, BIN_W, MIN_PER_BIN, N_AREA_BINS, BLOCK_DAYS, SEED = 0.7, 20.0, 3, 25, 27, 42
AREA_RANGES = [(0, 100), (100, 500), (500, 1000), (1000, 1e9)]

def biexp(A, a1, t1, a2, t2, c): return a1 * np.exp(-A / t1) + a2 * np.exp(-A / t2) + c
P0S = [[0.1, 30.0, 0.05, 900.0, 0.29], [0.1, 30.0, 0.05, 900.0, 0.10]]
BOUNDS = ([0, 1, 0, 50, 0], [1, 1000, 1, 20000, 1])

def binned(x, y, edges, nmin, where):
    idx = np.digitize(x, edges) - 1; xc, ym, n = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() >= nmin:
            xc.append(np.median(x[m]) if where == 'median' else 0.5 * (edges[b] + edges[b + 1])); ym.append(np.median(y[m])); n.append(int(m.sum()))
    return np.array(xc), np.array(ym), np.array(n)

def fit(xb, yb):
    best = None
    for p0 in P0S:
        try:
            p, cov = curve_fit(biexp, xb, yb, p0=p0, bounds=BOUNDS, maxfev=20000)
        except Exception:
            continue
        rss = np.sum((yb - biexp(xb, *p)) ** 2)
        if best is None or rss < best[2]: best = (p, cov, rss)
    if best is None: return None, None, None
    p, cov, rss = best; perr = np.sqrt(np.diag(cov))
    if p[1] > p[3]:
        p = np.array([p[2], p[3], p[0], p[1], p[4]]); perr = perr[[2, 3, 0, 1, 4]]
    return p, perr, 1 - rss / np.sum((yb - yb.mean()) ** 2)

bl = pd.read_csv(os.path.expanduser(a.blob)); bl = bl[bl['blob_id'] != -1].copy()
bl['date'] = bl['date'].astype(int).astype(str)
q = bl[(bl['mu'] >= MU_MIN) & (bl['area_msh'] >= a.amin)].copy()
pdd = pd.read_csv(os.path.expanduser(a.pday)); pdd['date'] = pdd['date'].astype(int).astype(str)
pdd = pdd[pdd['area_ar_msh'].notna()].copy()
q = q[q['date'].isin(set(pdd['date']))].sort_values('date').reset_index(drop=True)
days = q['date'].unique(); pdd = pdd.set_index('date').loc[days].reset_index()
starts = np.searchsorted(q['date'].values, days)
x_day = pdd['area_ar_msh'].values.astype(float); umin = q['umb_min_ratio'].values.astype(float)
area_blob = q['area_msh'].values.astype(float); y_obs = np.minimum.reduceat(umin, starts)
print('[%s] A_min=%g | days %d | umbrae %d | max difference vs per-day CSV %.4f' % (a.tag, a.amin, len(days), len(q), np.abs(y_obs - pdd['umb_min_ratio'].values).max()))

e20 = np.arange(0.0, x_day.max() + BIN_W, BIN_W)
print('\n== Main fit (20-MSH bins, >=10 days) ==')
for where in ('center', 'median'):
    xb, yb, nb = binned(x_day, y_obs, e20, 10, where); p, e, r2 = fit(xb, yb)
    k = 5; n = len(yb); rss = np.sum((yb - biexp(xb, *p)) ** 2)
    print('  point=%-6s n=%d R2=%.3f | A1=%.3f±%.3f t1=%.1f±%.1f A2=%.3f±%.3f t2=%.0f±%.0f c=%.3f±%.3f | AIC=%.2f' % (
        where, n, r2, p[0], e[0], p[1], e[1], p[2], e[2], p[3], e[3], p[4], e[4], n * np.log(rss / n) + 2 * k))

t = pd.to_datetime(q['date'], format='%Y%m%d'); year = t.dt.year.values
block = ((t - t.min()).dt.days.values // BLOCK_DAYS).astype(int)
log_edges = np.quantile(np.log(area_blob), np.linspace(0, 1, N_AREA_BINS + 1)); log_edges[0] -= 1e-9; log_edges[-1] += 1e-9
abin = np.digitize(np.log(area_blob), log_edges) - 1
def make_groups(keys):
    s = pd.Series(np.arange(len(keys))); return [g.values for _, g in s.groupby(keys)]
groups = {'null2_area_alltime': make_groups(abin),
          'null3_area_sameyear': make_groups(pd.Series(year).astype(str) + '_' + pd.Series(abin).astype(str)),
          'null4_area_same27d': make_groups(pd.Series(block).astype(str) + '_' + pd.Series(abin).astype(str))}

xb_obs, yb_obs, nb_obs = binned(x_day, y_obs, e20, MIN_PER_BIN, 'median')
p_obs, e_obs, r2_obs = fit(xb_obs, yb_obs)
print('\n== Observed, null-test binning (>=3 days, median point) == bins=%d R2=%.3f t1=%.1f±%.1f t2=%.0f±%.0f c=%.3f±%.3f' % (
    len(xb_obs), r2_obs, p_obs[1], e_obs[1], p_obs[3], e_obs[3], p_obs[4], e_obs[4]))

def one(args):
    kind, seed = args; rng = np.random.default_rng(seed)
    if kind == 'null1_count_only':
        draw = umin[rng.integers(0, len(umin), size=len(umin))]
    else:
        draw = np.empty_like(umin)
        for mem in groups[kind]: draw[mem] = umin[mem[rng.integers(0, len(mem), size=len(mem))]]
    ys = np.minimum.reduceat(draw, starts)
    xb, yb, _ = binned(x_day, ys, e20, MIN_PER_BIN, 'median')
    p, e, r2 = fit(xb, yb)
    return np.interp(xb_obs, xb, yb), (list(p) + [r2] if p is not None else None)

names = ['A1', 'tau1', 'A2', 'tau2', 'c', 'R2']; obs_vals = list(p_obs) + [r2_obs]
KINDS = ['null1_count_only', 'null2_area_alltime', 'null3_area_sameyear', 'null4_area_same27d']
results, bands = {}, {}
with Pool(a.workers) as pool:
    for ki, kind in enumerate(KINDS):
        res = pool.map(one, [(kind, SEED * 100000 + ki * 10000 + r) for r in range(a.nrep)])
        curves = np.array([r[0] for r in res]); params = np.array([r[1] for r in res if r[1] is not None]); fails = sum(r[1] is None for r in res)
        lo, med, hi = np.percentile(curves, [2.5, 50, 97.5], axis=0); bands[kind] = (lo, med, hi)
        pct = {n: [float(v) for v in np.percentile(params[:, i], [2.5, 50, 97.5])] for i, n in enumerate(names)}
        rank = {n: float(np.mean(params[:, i] <= obs_vals[i])) for i, n in enumerate(names)}
        by_range = []
        for lo_, hi_ in AREA_RANGES:
            m = (xb_obs >= lo_) & (xb_obs < hi_)
            if m.sum() == 0: continue
            d = yb_obs[m] - med[m]; out = (yb_obs[m] < lo[m]) | (yb_obs[m] > hi[m])
            by_range.append(dict(range_msh=('%g-%g' % (lo_, hi_)) if hi_ < 1e9 else '>%g' % lo_, n_bins=int(m.sum()), n_days=int(nb_obs[m].sum()),
                                 mean_obs_minus_null=float(d.mean()), frac_outside_95=float(out.mean()), frac_obs_darker=float((d < 0).mean())))
        results[kind] = dict(fit_fail=int(fails), param_pct=pct, obs_quantile_in_null=rank,
                             frac_bins_outside_95=float(np.mean((yb_obs < lo) | (yb_obs > hi))), by_area_range=by_range)
        print('\n[%s] failed fits %d/%d' % (kind, fails, a.nrep))
        for n in ['tau1', 'tau2', 'c']:
            x0, m_, x1 = pct[n]; print('  %-5s null %9.3f [95%%: %9.3f - %9.3f] | obs %9.3f (quantile %.3f)' % (n, m_, x0, x1, obs_vals[names.index(n)], rank[n]))
        print('  bins outside the 95%% band: %.3f' % results[kind]['frac_bins_outside_95'])
        for r_ in by_range:
            print('  %10s bins %3d days %5d | mean obs-null %+.4f | outside 95%% %.2f | obs darker %.2f' % (
                r_['range_msh'], r_['n_bins'], r_['n_days'], r_['mean_obs_minus_null'], r_['frac_outside_95'], r_['frac_obs_darker']))

titles = {'null1_count_only': 'Null 1: sampling only', 'null2_area_alltime': 'Null 2: area-matched, all epochs',
          'null3_area_sameyear': 'Null 3: area-matched, same year', 'null4_area_same27d': 'Null 4: area-matched, same 27-day block'}
fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
for ax, kind in zip(axes.ravel(), titles):
    lo, med, hi = bands[kind]
    ax.scatter(x_day, y_obs, s=2, color='0.85'); ax.fill_between(xb_obs, lo, hi, color='tab:blue', alpha=0.3, label='null 95%')
    ax.plot(xb_obs, med, color='tab:blue', lw=1); ax.plot(xb_obs, yb_obs, 'ko', ms=3, label='observed binned median')
    ax.set_title(titles[kind], fontsize=10); ax.legend(fontsize=8)
for ax in axes[1]: ax.set_xlabel('Total AR area per day [MSH]')
for ax in axes[:, 0]: ax.set_ylabel(r'$I_{\rm umb,min}/I_{qs}$')
fig.suptitle('Extreme-value null tests (%s, A$_{min}$=%g MSH, N=%d days)' % (a.tag, a.amin, len(days))); fig.tight_layout()
fig.savefig(os.path.join(OUT, 'null_tests_4panel.png'), dpi=140)
os.makedirs(paths.get('check_plots'), exist_ok=True); fig.savefig(os.path.join(paths.get('check_plots'), 'null_4panel_%s.png' % a.tag), dpi=140)
for kind, (lo, med, hi) in bands.items():
    pd.DataFrame(dict(x_bin=xb_obs, obs=yb_obs, n_days=nb_obs, null_lo=lo, null_med=med, null_hi=hi)).to_csv(os.path.join(OUT, 'bands_%s.csv' % kind), index=False)
json.dump(dict(blob=a.blob, pday=a.pday, amin=a.amin, n_days=int(len(days)), n_umbrae=int(len(q)), observed=dict(zip(names, map(float, obs_vals))),
               observed_err=list(map(float, e_obs)), n_bins=int(len(xb_obs)), nulls=results), open(os.path.join(OUT, 'summary.json'), 'w'), indent=2)
print('\n[WRITE] %s' % OUT)
