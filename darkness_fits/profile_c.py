"""Profile of the asymptotic level c (Section 5.1): free single- and bi-exponential fits, the fit
quality as a function of a fixed c, and a 27-day block bootstrap (200 replicates) of tau_1 and c,
for the clipped and unclipped products."""
import numpy as np, pandas as pd, os, warnings
from scipy.optimize import curve_fit
from multiprocessing import Pool
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from hmi_umbra import paths
SRC = {'clipped': os.path.join(paths.get('stats_clipped_out'), 'umbra_blob_intensity_area_v4.csv'),
       'noclip': os.path.join(paths.get('stats_unclipped_out'), 'umbra_blob_intensity_area_noclipT1.csv')}
OUT = os.path.join(paths.get('fits_out'), 'profile_c'); os.makedirs(OUT, exist_ok=True)
CGRID = np.round(np.arange(0, 0.3001, 0.01), 3)

def load(p):
    d = pd.read_csv(p)
    d = d[(d.mu >= 0.7) & (d.area_msh >= 1) & (d.n_pix >= 4) & (d.umb_median_ratio <= 0.509)].copy()
    t = pd.to_datetime(d['date'].astype(str)); d['blk'] = ((t - t.min()).dt.days // 27).astype(int)
    return d

def bins(A, Y, kind):
    e = np.arange(1, A.max() + 5, 5.0) if kind == 'lin5' else 10 ** np.arange(0, np.log10(A.max()) + 0.1, 0.1)
    x, y, n = [], [], []
    for lo, hi in zip(e[:-1], e[1:]):
        m = (A >= lo) & (A < hi)
        if m.sum() >= 15:
            x.append(0.5 * (lo + hi) if kind == 'lin5' else np.median(A[m])); y.append(np.median(Y[m])); n.append(m.sum())
    return np.array(x), np.array(y), np.array(n, float)

def ex1(A, A1, t1, c): return A1 * np.exp(-A / t1) + c
def ex2(A, A1, t1, A2, t2, c): return A1 * np.exp(-A / t1) + A2 * np.exp(-A / t2) + c
MOD = {'ex1': (ex1, [0.15, 10, 0.25], ([0, 0.1, 0], [5, 1e4, 1])),
       'ex2': (ex2, [0.15, 5, 0.05, 80, 0.25], ([0, 0.1, 0, 5, 0], [5, 200, 5, 1e5, 1]))}

def fit(x, y, w, name, cfix=None):
    f, p0, (lo, hi) = MOD[name]
    s = 1 / np.sqrt(w)
    best = None
    for sc in (1.0, 0.5, 2.0, 4.0):
        q0 = list(p0); q0[1] *= sc
        if name == 'ex2': q0[3] *= sc
        try:
            if cfix is None:
                p, _ = curve_fit(f, x, y, p0=q0, bounds=(lo, hi), sigma=s, maxfev=50000)
            else:
                g = lambda A, *q: f(A, *q, cfix)
                p, _ = curve_fit(g, x, y, p0=q0[:-1], bounds=(lo[:-1], hi[:-1]), sigma=s, maxfev=50000)
                p = np.append(p, cfix)
        except Exception:
            continue
        chi = np.sum(w * (y - f(x, *p)) ** 2)
        if best is None or chi < best[1]: best = (p, chi)
    return best

def ic(chi, w, n, k):
    return n * np.log(chi / w.sum() * 1.0) + 2 * k

D = {k: load(v) for k, v in SRC.items()}
SETS = [('lin5', 'equal'), ('lin5', 'n'), ('log01', 'n')]
rows, prof = [], []
for tag, d in D.items():
    A, Y = d.area_msh.values, d.umb_median_ratio.values
    for kind, wt in SETS:
        x, y, n = bins(A, Y, kind); w = np.ones_like(n) if wt == 'equal' else n / n.mean()
        for name in MOD:
            p, chi = fit(x, y, w, name)
            k = len(p); N = len(y)
            rows.append(dict(data=tag, bin=kind, w=wt, model=name, nbin=N, chi=chi, AIC=N * np.log(chi / N) + 2 * k,
                             **{('p%d' % i): v for i, v in enumerate(p)}))
            for c in CGRID:
                r = fit(x, y, w, name, cfix=c)
                prof.append(dict(data=tag, bin=kind, w=wt, model=name, c=c, chi=r[1] if r else np.nan, chi_free=chi, nbin=N))
R = pd.DataFrame(rows); P = pd.DataFrame(prof)
R.to_csv(OUT + '/fit_free.csv', index=False); P.to_csv(OUT + '/profile_c.csv', index=False)
pd.set_option('display.width', 220)
print('=== Free fits (ex1: A1,t1,c | ex2: A1,t1,A2,t2,c) ===')
print(R.to_string(index=False, float_format=lambda v: '%.4g' % v))

print('\n=== Supported range of c: dAIC_profile = N ln(chi_c/chi_free) < 2 and < 4 ===')
P['dA'] = P.nbin * np.log(P.chi / P.chi_free)
for (tag, kind, wt, name), g in P.groupby(['data', 'bin', 'w', 'model'], sort=False):
    ok2 = g.c[g.dA < 2]; ok4 = g.c[g.dA < 4]
    print('%-6s %-5s %-5s %-3s | best c %.2f | dA<2: %s | dA<4: %s' % (tag, kind, wt, name, g.c[g.dA.idxmin()],
          '%.2f-%.2f' % (ok2.min(), ok2.max()) if len(ok2) else '-', '%.2f-%.2f' % (ok4.min(), ok4.max()) if len(ok4) else '-'))

def boot(seed):
    rng = np.random.default_rng(seed); out = []
    for tag, d in D.items():
        ub = d.blk.unique(); pick = rng.choice(ub, len(ub), replace=True)
        s = pd.concat([d[d.blk == b] for b in pick])
        x, y, n = bins(s.area_msh.values, s.umb_median_ratio.values, 'lin5'); w = n / n.mean()
        for name in MOD:
            r = fit(x, y, w, name)
            out.append((tag, name, r[0][1], r[0][-1]) if r else (tag, name, np.nan, np.nan))
    return out

with Pool(28) as pool:
    B = pool.map(boot, range(1, 201))
B = pd.DataFrame([z for b in B for z in b], columns=['data', 'model', 't1', 'c'])
B.to_csv(OUT + '/boot_t1_c.csv', index=False)
print('\n=== 27-day block bootstrap, 200 replicates, 5-MSH bins weighted by n ===')
for (tag, name), g in B.groupby(['data', 'model'], sort=False):
    q = lambda v: '%.3g [%.3g, %.3g]' % tuple(np.nanpercentile(v, [50, 2.5, 97.5]))
    print('%-6s %-3s | t1 %s | c %s | c=0 in %.0f %% of replicates' % (tag, name, q(g.t1), q(g.c), 100 * np.mean(g.c < 1e-3)))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
for i, (kind, wt) in enumerate(SETS):
    for (tag, name), g in P[(P.bin == kind) & (P.w == wt)].groupby(['data', 'model']):
        ax[i].plot(g.c, g.dA, ('-' if tag == 'noclip' else '--'), marker='o', ms=3, label='%s %s' % (tag, name))
    ax[i].axhline(2, color='grey', ls=':'); ax[i].axhline(4, color='grey', ls=':')
    ax[i].set_title('%s, weight %s' % (kind, wt)); ax[i].set_xlabel('fixed c'); ax[i].set_ylim(-0.5, 30); ax[i].grid(alpha=0.3); ax[i].legend(fontsize=8)
ax[0].set_ylabel('profile dAIC relative to free c')
plt.tight_layout(); f = os.path.join(paths.get('check_plots'), 'profile_c.png'); os.makedirs(os.path.dirname(f), exist_ok=True); plt.savefig(f, dpi=130); print('\nsaved:', f)
