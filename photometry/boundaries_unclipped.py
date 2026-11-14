"""Umbra-penumbra and penumbra-photosphere crossing points on the unclipped photometric product
(Section 3.3: 0.508 and 0.843), from the class histograms written by photometry_unclipped.py,
for all days together and per year. Crossings are given for raw counts and for normalized
distributions."""
import numpy as np, glob, os, collections
from hmi_umbra import paths
D = os.path.join(paths.get('stats_unclipped_out'), 'hist')
fs = sorted(glob.glob(D + '/*.npz'))
e = np.load(fs[0])['edges']; c = 0.5*(e[1:]+e[:-1])
tot = {k: np.zeros(1500) for k in ('umbra','penumbra','photosphere')}
per = collections.defaultdict(lambda: {k: np.zeros(1500) for k in tot})
for f in fs:
    d = np.load(f); y = os.path.basename(f)[:4]
    for k in tot:
        tot[k] += d[k]; per[y][k] += d[k]

def cross(a, b, lo, hi):
    m = (c >= lo) & (c <= hi); x = c[m]; g = a[m] - b[m]
    idx = np.where((g[:-1] > 0) & (g[1:] <= 0))[0]
    if len(idx) == 0: return np.nan
    i = idx[-1]
    return x[i] + (x[i+1]-x[i]) * g[i]/(g[i]-g[i+1])

def both(h):
    u, p, q = h['umbra'], h['penumbra'], h['photosphere']
    raw = (cross(u, p, 0.2, 0.8), cross(p, q, 0.6, 1.1))
    un, pn, qn = u/u.sum(), p/p.sum(), q/q.sum()
    nrm = (cross(un, pn, 0.2, 0.8), cross(pn, qn, 0.6, 1.1))
    return raw, nrm

raw, nrm = both(tot)
u, p = tot['umbra'], tot['penumbra']
print('days:', len(fs), '| pixels umbra %.3e penumbra %.3e photosphere %.3e' % (u.sum(), p.sum(), tot['photosphere'].sum()))
for name, (t1, t2) in (('raw', raw), ('normalized', nrm)):
    fu = u[c > t1].sum()/u.sum()*100; fp = p[c < t1].sum()/p.sum()*100
    print('%-10s U-P %.4f  P-F %.4f | umbra > T1 %.2f %% | penumbra < T1 %.2f %%' % (name, t1, t2, fu, fp))
print('median I/Iqs umbra %.4f penumbra %.4f' % (c[np.searchsorted(np.cumsum(u), u.sum()/2)], c[np.searchsorted(np.cumsum(p), p.sum()/2)]))
print('\nper year (* marks years with > 1e5 umbral pixels)')
R = {'raw': [], 'normalized': []}
for y in sorted(per):
    h = per[y]; (a1, a2), (b1, b2) = both(h); n = h['umbra'].sum(); ok = n > 1e5
    print('%s %s n_umb %.2e | raw %.4f %.4f | normalized %.4f %.4f' % (y, '*' if ok else ' ', n, a1, a2, b1, b2))
    if ok: R['raw'].append((a1, a2)); R['normalized'].append((b1, b2))
for k in R:
    a = np.array(R[k]); print('%s: U-P %.4f +- %.4f | P-F %.4f +- %.4f (%d years)' % (k, np.nanmean(a[:,0]), np.nanstd(a[:,0]), np.nanmean(a[:,1]), np.nanstd(a[:,1]), len(a)))
