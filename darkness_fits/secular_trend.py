"""Secular trend of the per-umbra median intensity at fixed area (Section 3.2): linear trend in
time per decade within area ranges, controlling for ln A, with 27-day block-bootstrap errors
(500 replicates), for the clipped and unclipped products."""
import numpy as np, pandas as pd, os
from multiprocessing import Pool
from hmi_umbra import paths
SRC = {'clipped': os.path.join(paths.get('stats_clipped_out'), 'umbra_blob_intensity_area_v4.csv'), 'noclip': os.path.join(paths.get('stats_unclipped_out'), 'umbra_blob_intensity_area_noclipT1.csv')}
BINS = [(1, 3), (3, 10), (10, 30), (30, 131), (10, 131)]
D = {}
for k, p in SRC.items():
    d = pd.read_csv(p); d = d[(d.mu >= 0.7) & (d.area_msh >= 1) & (d.n_pix >= 4) & (d.umb_median_ratio <= 0.509)].copy()
    t = pd.to_datetime(d['date'].astype(str)); d['ty'] = t.dt.year + (t.dt.dayofyear - 1) / 365.25
    d['blk'] = ((t - t.min()).dt.days // 27).astype(int); D[k] = d
def slope(s):
    x = s.ty.values - 2019.0; y = s.umb_median_ratio.values; la = np.log(s.area_msh.values)
    X = np.column_stack([np.ones_like(x), x, la - la.mean()])
    return 10 * np.linalg.lstsq(X, y, rcond=None)[0][1]
def boot(a):
    k, lo, hi, seed = a; s = D[k][(D[k].area_msh >= lo) & (D[k].area_msh < hi)]; rng = np.random.default_rng(seed)
    ub = s.blk.unique(); pick = rng.choice(ub, len(ub), replace=True)
    return (k, lo, hi, slope(pd.concat([s[s.blk == b] for b in pick])))
jobs = [(k, lo, hi, 100000 * i + r) for i, (k, (lo, hi)) in enumerate([(k, b) for k in D for b in BINS]) for r in range(500)]
with Pool(28) as pool: B = pd.DataFrame(pool.map(boot, jobs), columns=['data', 'lo', 'hi', 's'])
print('trend of the median I/Iqs per decade, controlling for log(area) within each range; error = sd of a 27-day block bootstrap (500)')
for k in D:
    for lo, hi in BINS:
        s = D[k][(D[k].area_msh >= lo) & (D[k].area_msh < hi)]; b = B[(B.data == k) & (B.lo == lo) & (B.hi == hi)].s
        print('%-6s %3g-%-4g MSH N=%6d | trend %+.4f +- %.4f per decade (%.1f sigma) | 95%% [%+.4f, %+.4f]' % (
            k, lo, hi, len(s), slope(s), b.std(), abs(slope(s)) / b.std(), *np.percentile(b, [2.5, 97.5])))
