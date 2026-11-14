"""Fits of the per-umbra median intensity versus umbral area (Section 5.1) and their variants:
power law and bi-exponential on 5-MSH bins with different point locations and weights,
logarithmic bins, minimum areas, a central-meridian band, a non-parametric table of medians per
area range, and an unbinned L1 regression with a 27-day block bootstrap.

Usage: python fit_variants.py --csv PER_UMBRA_TABLE.csv --tag NAME [--nboot N] [--workers N] [--thr 0.509]
"""
import argparse, os, sys, json
import numpy as np, pandas as pd
from scipy.optimize import curve_fit, minimize
from multiprocessing import Pool

from hmi_umbra import paths

ap = argparse.ArgumentParser()
ap.add_argument('--csv', required=True)
ap.add_argument('--tag', required=True)
ap.add_argument('--nboot', type=int, default=200)
ap.add_argument('--workers', type=int, default=28)
ap.add_argument('--thr', type=float, default=0.509)
args = ap.parse_args()

OUT = os.path.join(paths.get('fits_out'), args.tag)
os.makedirs(OUT, exist_ok=True)
XC, R_TYP, BAND_DEG = 2047.5, 1915.0, 6.6
MU_MIN, A_MIN0, MIN_PIX, BLOCK = 0.7, 1.0, 4, 27

raw = pd.read_csv(args.csv)
print('file:', args.csv)
print('columns:', raw.columns.tolist())
print('total rows:', len(raw))
base = raw[(raw.mu >= MU_MIN) & (raw.area_msh >= A_MIN0)]
if 'n_pix' in raw.columns:
    base = base[base.n_pix >= MIN_PIX]
print('mu>=%.1f, A>=1 MSH, n_pix>=%d: %d | largest median %.4f' % (MU_MIN, MIN_PIX, len(base), base.umb_median_ratio.max()))
d = base[base.umb_median_ratio <= args.thr].copy()
print('passing the paper cuts (median <= %.3f): N = %d' % (args.thr, len(d)))
t = pd.to_datetime(d['date'].astype(str))
d['block'] = ((t - t.min()).dt.days // BLOCK).astype(int)
A_all = d.area_msh.values.astype(float)
Y_all = d.umb_median_ratio.values.astype(float)
B_all = d.block.values


def pl(A, a, b, c):
    return a * A ** (-b) + c


def be(A, A1, t1, A2, t2, c):
    return A1 * np.exp(-A / t1) + A2 * np.exp(-A / t2) + c


MODELS = {
    'power': (pl, [0.3, 0.8, 0.3], ([0, 0, 0], [5, 5, 1]), ['a', 'b', 'c']),
    'biexp': (be, [0.16, 5.0, 0.03, 80.0, 0.3], ([0, 0.1, 0, 5, 0], [5, 100, 5, 1e5, 1]), ['A1', 't1', 'A2', 't2', 'c']),
}


def sigma_med(y):
    mad = np.median(np.abs(y - np.median(y)))
    return max(1.2533 * 1.4826 * mad / np.sqrt(len(y)), 1e-6)


def bins_lin(A, Y, width=5.0, nmin=15, a0=1.0):
    edges = np.arange(a0, A.max() + width, width)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (A >= lo) & (A < hi)
        if m.sum() >= nmin:
            rows.append((0.5 * (lo + hi), np.median(A[m]), np.median(Y[m]), m.sum(), sigma_med(Y[m])))
    return pd.DataFrame(rows, columns=['A_mid', 'A_med', 'y', 'n', 'sig'])


def bins_log(A, Y, dex=0.1, nmin=15, a0=1.0):
    edges = 10 ** np.arange(np.log10(a0), np.log10(A.max()) + dex, dex)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (A >= lo) & (A < hi)
        if m.sum() >= nmin:
            rows.append((np.sqrt(lo * hi), np.median(A[m]), np.median(Y[m]), m.sum(), sigma_med(Y[m])))
    return pd.DataFrame(rows, columns=['A_mid', 'A_med', 'y', 'n', 'sig'])


def ic(rss, n, k):
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)
    aicc = aic + 2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else np.nan
    return aic, bic, aicc


def fit_binned(df, xcol, weight):
    x, y = df[xcol].values, df.y.values
    if weight == 'equal':
        s = None; w = np.ones_like(y)
    elif weight == 'n':
        s = 1 / np.sqrt(df.n.values); w = df.n.values / df.n.values.mean()
    else:
        s = df.sig.values; w = (1 / s ** 2) / (1 / s ** 2).mean()
    res = {}
    for name, (f, p0, bd, pn) in MODELS.items():
        try:
            p, cov = curve_fit(f, x, y, p0=p0, bounds=bd, sigma=s, absolute_sigma=False, maxfev=200000)
            e = np.sqrt(np.diag(cov))
            r = y - f(x, *p)
            wrss = np.sum(w * r ** 2)
            r2 = 1 - np.sum(r ** 2) / np.sum((y - y.mean()) ** 2)
            aic, bic, aicc = ic(wrss, len(y), len(p))
            res[name] = dict(p=p, e=e, r2=r2, aic=aic, bic=bic, aicc=aicc, n=len(y), names=pn)
        except Exception as ex:
            res[name] = dict(err=str(ex), n=len(y))
    return res


def l1_fit(A, Y, name, p_start):
    f, p0, bd, pn = MODELS[name]
    lo, hi = np.array(bd[0], float), np.array(bd[1], float)

    def obj(q):
        p = np.clip(q, lo, hi)
        return np.sum(np.abs(Y - f(A, *p))) + 1e3 * np.sum((q - p) ** 2)
    best = None
    for s0 in (np.asarray(p_start, float), np.asarray(p0, float)):
        r = minimize(obj, s0, method='Nelder-Mead', options=dict(maxiter=20000, maxfev=20000, xatol=1e-6, fatol=1e-8))
        r = minimize(obj, r.x, method='Nelder-Mead', options=dict(maxiter=20000, maxfev=20000, xatol=1e-7, fatol=1e-9))
        p = np.clip(r.x, lo, hi); s = np.sum(np.abs(Y - f(A, *p)))
        if best is None or s < best[1]:
            best = (p, s)
    return best


def l1_both(A, Y, starts):
    out = {}
    for name in MODELS:
        p, s = l1_fit(A, Y, name, starts[name])
        n, k = len(Y), len(p)
        out[name] = dict(p=p, sad=s, aic=2 * n * np.log(s / n) + 2 * k, bic=2 * n * np.log(s / n) + k * np.log(n))
    return out


STARTS = None


def boot_one(seed):
    rng = np.random.default_rng(seed)
    ub = np.unique(B_all)
    pick = rng.choice(ub, size=len(ub), replace=True)
    idx = np.concatenate([np.where(B_all == b)[0] for b in pick])
    o = l1_both(A_all[idx], Y_all[idx], STARTS)
    return np.concatenate([o['power']['p'], o['biexp']['p'], [o['biexp']['aic'] - o['power']['aic']]])


def fmt(res):
    lines = []
    for name in MODELS:
        r = res[name]
        if 'err' in r:
            lines.append('  %-6s FAILED: %s' % (name, r['err'])); continue
        ps = ' '.join('%s=%.4g±%.2g' % (a, b, c) for a, b, c in zip(r['names'], r['p'], r['e']))
        lines.append('  %-6s n=%d R2=%.4f AIC=%.2f BIC=%.2f AICc=%.2f | %s' % (name, r['n'], r['r2'], r['aic'], r['bic'], r['aicc'], ps))
    if 'err' not in res['power'] and 'err' not in res['biexp']:
        lines.append('  dAIC(bi-pl)=%.2f dBIC=%.2f dAICc=%.2f' % (res['biexp']['aic'] - res['power']['aic'], res['biexp']['bic'] - res['power']['bic'], res['biexp']['aicc'] - res['power']['aicc']))
    return '\n'.join(lines)


rows = []


def record(sec, label, res):
    for name in MODELS:
        r = res[name]
        if 'err' in r:
            continue
        row = dict(section=sec, label=label, model=name, n=r['n'], R2=r['r2'], AIC=r['aic'], BIC=r['bic'], AICc=r['aicc'])
        for a, b, c in zip(r['names'], r['p'], r['e']):
            row[a] = b; row[a + '_err'] = c
        rows.append(row)


print('\n=== 1. Table 2: 5-MSH bins (>=15 umbrae), A >= 1 MSH ===')
b5 = bins_lin(A_all, Y_all)
b5.to_csv(OUT + '/bins_5msh.csv', index=False)
print('bins: %d | range %.1f-%.1f MSH | umbrae in bins %d (%.1f %%)' % (len(b5), b5.A_mid.min() - 2.5, b5.A_mid.max() + 2.5, b5.n.sum(), 100 * b5.n.sum() / len(A_all)))
for xcol in ('A_mid', 'A_med'):
    for w in ('equal', 'n', 'sigma'):
        res = fit_binned(b5, xcol, w)
        print('-- point=%s weight=%s' % (xcol, w)); print(fmt(res)); record('lin5', xcol + '/' + w, res)

print('\n=== 2. Logarithmic 0.1-dex bins (>=15 umbrae) ===')
bl = bins_log(A_all, Y_all)
bl.to_csv(OUT + '/bins_log01.csv', index=False)
print('bins: %d' % len(bl))
for w in ('equal', 'n', 'sigma'):
    res = fit_binned(bl, 'A_med', w)
    print('-- point=A_med weight=%s' % w); print(fmt(res)); record('log01', 'A_med/' + w, res)

print('\n=== 3. Variation of A_min (5-MSH bins, point A_mid) ===')
for amin in (1, 2, 3, 5, 10):
    m = A_all >= amin
    bb = bins_lin(A_all[m], Y_all[m], a0=float(amin))
    for w in ('equal', 'n'):
        res = fit_binned(bb, 'A_mid', w)
        print('-- A_min=%g N=%d weight=%s' % (amin, m.sum(), w)); print(fmt(res)); record('amin', '%g/%s' % (amin, w), res)

print('\n=== 4. Central-meridian band |dx| <= R sin(%.1f deg) ===' % BAND_DEG)
if 'cx' in d.columns:
    mb = np.abs(d.cx.values - XC) <= R_TYP * np.sin(np.radians(BAND_DEG))
    bb = bins_lin(A_all[mb], Y_all[mb])
    print('N band = %d, bins = %d' % (mb.sum(), len(bb)))
    for w in ('equal', 'n'):
        res = fit_binned(bb, 'A_mid', w)
        print('-- weight=%s' % w); print(fmt(res)); record('band', w, res)
else:
    print('column cx missing, skipped')

print('\n=== 5. Non-parametric table per area range ===')
edges = [1, 1.5, 2, 3, 5, 7, 10, 15, 25, 50, 131]
extra = [c for c in ('umb_dark25_ratio', 'umb_median_erode2_ratio', 'umb_min_ratio') if c in d.columns]
nrows = []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (A_all >= lo) & (A_all < hi)
    y = Y_all[m]
    rr = dict(A_lo=lo, A_hi=hi, N=int(m.sum()), med=np.median(y), p16=np.percentile(y, 16), p84=np.percentile(y, 84), q25=np.percentile(y, 25), q75=np.percentile(y, 75))
    for c in extra:
        v = d[c].values[m]; v = v[np.isfinite(v)]
        rr[c + '_med'] = np.median(v) if len(v) else np.nan
    nrows.append(rr)
m = A_all >= 131
nrows.append(dict(A_lo=131, A_hi=np.inf, N=int(m.sum()), med=np.median(Y_all[m]) if m.any() else np.nan))
npt = pd.DataFrame(nrows)
npt.to_csv(OUT + '/nonparam_table.csv', index=False)
print(npt.to_string(index=False, float_format=lambda v: '%.4f' % v))

print('\n=== 6. Unbinned L1 regression + %d-day block bootstrap (%d replicates) ===' % (BLOCK, args.nboot))
lsq = fit_binned(b5, 'A_mid', 'equal')
STARTS = {k: (lsq[k]['p'] if 'p' in lsq[k] else np.array(MODELS[k][1])) for k in MODELS}
l1 = l1_both(A_all, Y_all, STARTS)
STARTS = {k: l1[k]['p'] for k in MODELS}
for k in MODELS:
    print('  %-6s SAD=%.4f AIC_L=%.2f BIC_L=%.2f | %s' % (k, l1[k]['sad'], l1[k]['aic'], l1[k]['bic'], ' '.join('%s=%.4g' % (a, b) for a, b in zip(MODELS[k][3], l1[k]['p']))))
print('  dAIC_L(bi-pl)=%.2f dBIC_L=%.2f' % (l1['biexp']['aic'] - l1['power']['aic'], l1['biexp']['bic'] - l1['power']['bic']))
with Pool(args.workers) as pool:
    bs = np.array(pool.map(boot_one, range(1, args.nboot + 1)))
cols = ['pl_a', 'pl_b', 'pl_c', 'be_A1', 'be_t1', 'be_A2', 'be_t2', 'be_c', 'dAIC_L']
bsd = pd.DataFrame(bs, columns=cols)
bsd.to_csv(OUT + '/l1_bootstrap.csv', index=False)
for c in cols:
    lo, md, hi = np.percentile(bsd[c], [2.5, 50, 97.5])
    print('  %-7s median %.4g  95%% [%.4g, %.4g]' % (c, md, lo, hi))
print('  fraction of replicates favoring the bi-exponential (dAIC_L<0): %.1f %%' % (100 * np.mean(bsd.dAIC_L < 0)))

pd.DataFrame(rows).to_csv(OUT + '/fits_binned.csv', index=False)
json.dump({k: dict(p=list(map(float, l1[k]['p'])), sad=float(l1[k]['sad']), aic=float(l1[k]['aic']), bic=float(l1[k]['bic'])) for k in l1}, open(OUT + '/l1_full.json', 'w'), indent=1)
print('\nOutput: ' + OUT)
print('DONE')
