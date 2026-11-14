"""Robustness of the per-umbra bi-exponential fit (Appendix A, Table 5): minimum umbral areas of
1, 5 and 10 MSH, over the whole period and per solar cycle (SC24: 2012-2019; SC25: 2020-2026),
with Delta AIC/BIC against the single exponential and the power law and a 27-day block bootstrap
(500 replicates) of tau_1 and c."""
import numpy as np, pandas as pd, os, warnings
from scipy.optimize import curve_fit
from multiprocessing import Pool
warnings.filterwarnings('ignore')
from hmi_umbra import paths
NB = 500
d = pd.read_csv(os.path.join(paths.get('stats_unclipped_out'), 'umbra_blob_intensity_area_noclipT1.csv'))
d = d[(d.mu >= 0.7) & (d.area_msh >= 1) & (d.n_pix >= 4) & (d.umb_median_ratio <= 0.509)].copy()
t = pd.to_datetime(d['date'].astype(str)); d['yr'] = t.dt.year; d['blk'] = ((t - t.min()).dt.days // 27).astype(int)

ex1 = lambda A, A1, t1, c: A1 * np.exp(-A / t1) + c
ex2 = lambda A, A1, t1, A2, t2, c: A1 * np.exp(-A / t1) + A2 * np.exp(-A / t2) + c
pl = lambda A, a, b, c: a * A ** (-b) + c
M = {'pl': (pl, [[0.3, 0.8, 0.3], [0.56, 0.2, 0.0]], ([0, 0, 0], [5, 5, 1])),
     'ex1': (ex1, [[0.15, 8, 0.3], [0.22, 25, 0.22]], ([0, 0.1, 0], [5, 1e4, 1])),
     'ex2': (ex2, [[0.16, 5.7, 0.03, 82, 0.3], [0.19, 17, 0.27, 500, 0.0], [0.15, 10, 0.1, 150, 0.15]], ([0, 0.1, 0, 5, 0], [5, 200, 5, 1e5, 1]))}

def b5(A, Y, a0):
    e = np.arange(a0, A.max() + 5, 5.0); x, y = [], []
    for lo, hi in zip(e[:-1], e[1:]):
        m = (A >= lo) & (A < hi)
        if m.sum() >= 15: x.append(0.5 * (lo + hi)); y.append(np.median(Y[m]))
    return np.array(x), np.array(y)

def fit(x, y, name):
    f, starts, bd = M[name]; best = None
    for p0 in starts:
        try: p, cov = curve_fit(f, x, y, p0=p0, bounds=bd, maxfev=100000)
        except Exception: continue
        rss = np.sum((y - f(x, *p)) ** 2)
        if best is None or rss < best[2]: best = (p, cov, rss)
    p, cov, rss = best; e = np.sqrt(np.abs(np.diag(cov)))
    if name == 'ex2' and p[1] > p[3]: p = p[[2, 3, 0, 1, 4]]; e = e[[2, 3, 0, 1, 4]]
    n, k = len(y), len(p)
    return p, e, rss, 1 - rss / np.sum((y - y.mean()) ** 2), n * np.log(rss / n) + 2 * k, n * np.log(rss / n) + k * np.log(n)

SUB = {'all': lambda s: s, 'SC24': lambda s: s[s.yr <= 2019], 'SC25': lambda s: s[s.yr >= 2020]}
JOBS = [(am, sb) for am in (1, 5, 10) for sb in ('all', 'SC24', 'SC25')]

def boot(args):
    am, sb, seed = args; s = SUB[sb](d[d.area_msh >= am]); rng = np.random.default_rng(seed)
    ub = s.blk.unique(); pick = rng.choice(ub, len(ub), replace=True)
    r = pd.concat([s[s.blk == b] for b in pick]); x, y = b5(r.area_msh.values, r.umb_median_ratio.values, float(am))
    try: p = fit(x, y, 'ex2')[0]; return (am, sb, p[1], p[4])
    except Exception: return (am, sb, np.nan, np.nan)

rows = []
for am, sb in JOBS:
    s = SUB[sb](d[d.area_msh >= am]); x, y = b5(s.area_msh.values, s.umb_median_ratio.values, float(am))
    R = {k: fit(x, y, k) for k in M}
    p, e, _, r2, aic, bic = R['ex2']
    rows.append(dict(Amin=am, sub=sb, N=len(s), nbin=len(y), blocks=s.blk.nunique(), t1=p[1], t1e=e[1], c=p[4], ce=e[4], R2=r2,
                     dAIC_ex1=aic - R['ex1'][4], dBIC_ex1=bic - R['ex1'][5], dAIC_pl=aic - R['pl'][4], dBIC_pl=bic - R['pl'][5],
                     ex1_t1=R['ex1'][0][1], ex1_c=R['ex1'][0][2], pl_b=R['pl'][0][1], pl_c=R['pl'][0][2]))
T = pd.DataFrame(rows)
with Pool(28) as pool:
    Bs = pd.DataFrame(pool.map(boot, [(am, sb, 1000 * i + r) for i, (am, sb) in enumerate(JOBS) for r in range(NB)]), columns=['Amin', 'sub', 't1', 'c'])
g = Bs.groupby(['Amin', 'sub'])
Q = g.agg(t1_lo=('t1', lambda v: np.nanpercentile(v, 2.5)), t1_md=('t1', 'median'), t1_hi=('t1', lambda v: np.nanpercentile(v, 97.5)),
          c95=('c', lambda v: np.nanpercentile(v, 95)), c0=('c', lambda v: 100 * np.mean(v < 1e-3)), fail=('c', lambda v: int(v.isna().sum()))).reset_index()
T = T.merge(Q, on=['Amin', 'sub'])
ROB = os.path.join(paths.get('fits_out'), 'robust'); os.makedirs(ROB, exist_ok=True)
T.to_csv(ROB + '/robust_table_noclip.csv', index=False); Bs.to_csv(ROB + '/robust_boot_noclip.csv', index=False)
pd.set_option('display.width', 250)
print(T.to_string(index=False, float_format=lambda v: '%.4g' % v))
