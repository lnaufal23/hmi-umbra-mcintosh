"""Main per-umbra fit (Section 5.1, Table 2): power law, single exponential and bi-exponential on
the binned medians (5-MSH bins with >= 15 umbrae, bin centers, equal weights) with R^2, AIC, BIC
and AICc, the extrapolation to the largest umbrae, and a 27-day block bootstrap (500 replicates)
of the bi-exponential, for the unclipped ('noclip') and clipped products."""
import numpy as np, pandas as pd, os, warnings
from scipy.optimize import curve_fit
from multiprocessing import Pool
warnings.filterwarnings('ignore')
from hmi_umbra import paths
SRC = {'noclip': os.path.join(paths.get('stats_unclipped_out'), 'umbra_blob_intensity_area_noclipT1.csv'),
       'clipped': os.path.join(paths.get('stats_clipped_out'), 'umbra_blob_intensity_area_v4.csv')}
OUT = os.path.join(paths.get('fits_out'), 'biexp_final'); os.makedirs(OUT, exist_ok=True)
NB = 500

def load(p):
    d = pd.read_csv(p)
    d = d[(d.mu >= 0.7) & (d.area_msh >= 1) & (d.n_pix >= 4) & (d.umb_median_ratio <= 0.509)].copy()
    t = pd.to_datetime(d['date'].astype(str)); d['blk'] = ((t - t.min()).dt.days // 27).astype(int)
    return d

def b5(A, Y):
    e = np.arange(1, A.max() + 5, 5.0); x, y = [], []
    for lo, hi in zip(e[:-1], e[1:]):
        m = (A >= lo) & (A < hi)
        if m.sum() >= 15: x.append(0.5 * (lo + hi)); y.append(np.median(Y[m]))
    return np.array(x), np.array(y)

ex1 = lambda A, A1, t1, c: A1 * np.exp(-A / t1) + c
ex2 = lambda A, A1, t1, A2, t2, c: A1 * np.exp(-A / t1) + A2 * np.exp(-A / t2) + c
pl = lambda A, a, b, c: a * A ** (-b) + c
M = {'power': (pl, [[0.3, 0.8, 0.3], [0.56, 0.2, 0.0]], ([0, 0, 0], [5, 5, 1]), ['a', 'b', 'c']),
     'ex1': (ex1, [[0.15, 8, 0.3], [0.22, 25, 0.22]], ([0, 0.1, 0], [5, 1e4, 1]), ['A1', 't1', 'c']),
     'ex2': (ex2, [[0.16, 5.7, 0.03, 82, 0.3], [0.19, 17, 0.27, 500, 0.0], [0.15, 10, 0.1, 150, 0.15]],
             ([0, 0.1, 0, 5, 0], [5, 200, 5, 1e5, 1]), ['A1', 't1', 'A2', 't2', 'c'])}

def fit(x, y, name):
    f, starts, bd, _ = M[name]; best = None
    for p0 in starts:
        try:
            p, cov = curve_fit(f, x, y, p0=p0, bounds=bd, maxfev=100000)
        except Exception:
            continue
        rss = np.sum((y - f(x, *p)) ** 2)
        if best is None or rss < best[2]: best = (p, cov, rss)
    if best and best[1] is not None:
        p = best[0]
        if name == 'ex2' and p[1] > p[3]:
            p = np.array([p[2], p[3], p[0], p[1], p[4]]); best = (p, best[1][[2, 3, 0, 1, 4]][:, [2, 3, 0, 1, 4]], best[2])
    return best

D = {k: load(v) for k, v in SRC.items()}
for tag, d in D.items():
    x, y = b5(d.area_msh.values, d.umb_median_ratio.values); n = len(y)
    print('\n===== %s | N umbrae %d | 5-MSH bins %d, range 1-%.0f MSH =====' % (tag, len(d), n, x.max() + 2.5))
    for name in M:
        p, cov, rss = fit(x, y, name); k = len(p)
        e = np.sqrt(np.diag(cov)); r2 = 1 - rss / np.sum((y - y.mean()) ** 2)
        aic = n * np.log(rss / n) + 2 * k; bic = n * np.log(rss / n) + k * np.log(n); aicc = aic + 2 * k * (k + 1) / (n - k - 1)
        print('%-5s k=%d R2=%.4f AIC=%.2f BIC=%.2f AICc=%.2f | %s' % (name, k, r2, aic, bic, aicc,
              ' '.join('%s=%.4g±%.2g' % z for z in zip(M[name][3], p, e))))
        if name == 'ex2':
            f2 = p
    print('model ex2 at 150 MSH %.3f, 200 MSH %.3f' % (ex2(150, *f2), ex2(200, *f2)))
    A, Y = d.area_msh.values, d.umb_median_ratio.values
    for lo, hi in ((131, 175), (175, 250), (250, 1e4)):
        m = (A >= lo) & (A < hi)
        if m.sum(): print('  data %3.0f-%4.0f MSH: median %.3f (n=%d)' % (lo, min(hi, A.max()), np.median(Y[m]), m.sum()))

def boot(args):
    tag, seed = args; d = D[tag]; rng = np.random.default_rng(seed)
    ub = d.blk.unique(); pick = rng.choice(ub, len(ub), replace=True)
    s = pd.concat([d[d.blk == b] for b in pick])
    x, y = b5(s.area_msh.values, s.umb_median_ratio.values)
    r = fit(x, y, 'ex2')
    return [tag] + (list(r[0]) if r else [np.nan] * 5)

with Pool(28) as pool:
    B = pool.map(boot, [(t, s) for t in D for s in range(1, NB + 1)])
B = pd.DataFrame(B, columns=['data', 'A1', 't1', 'A2', 't2', 'c'])
B.to_csv(OUT + '/boot_ex2_lin5_equal.csv', index=False)
print('\n===== 27-day block bootstrap, %d replicates, ex2 on 5-MSH bins, equal weights =====' % NB)
for tag, g in B.groupby('data', sort=False):
    print('-- %s (failed %d)' % (tag, g.c.isna().sum()))
    for c in ['A1', 't1', 'A2', 't2', 'c']:
        md, lo, hi = np.nanpercentile(g[c], [50, 2.5, 97.5])
        print('   %-3s median %.4g  95%% [%.4g, %.4g]' % (c, md, lo, hi))
    print('   c=0 (<1e-3) in %.1f %% of replicates | 95%% upper bound on c (95th percentile) = %.3f' % (100 * np.mean(g.c < 1e-3), np.nanpercentile(g.c, 95)))
print('\nOutput:', OUT)
