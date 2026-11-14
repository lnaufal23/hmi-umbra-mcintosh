#!/usr/bin/env python3
"""Correlation between the daily HMI spot count and the SILSO sunspot number (Section 4.2):
Pearson and Spearman coefficients for the daily values of the days covered by both series, and for
non-overlapping 27-day means (blocks with at least 14 paired days). No p-values are computed,
since the series are autocorrelated."""
import pandas as pd
from pathlib import Path

from hmi_umbra import paths

V4 = Path(paths.get("analysis_v4mask"))
CSV_SPOT = V4 / "spot_count_per_day_v4mask.csv"
CSV_SILSO = Path(paths.get("silso_daily"))
DATE_START, DATE_END, BLOCK, MIN_DAYS, MA = "2012-01-01", "2026-04-30", 27, 14, 30

si = pd.read_csv(CSV_SILSO, sep=";", header=None, names=["year", "month", "day", "frac", "SN", "SNs", "n", "flag"])
si = si[si["SN"] >= 0].copy()
si["date"] = pd.to_datetime(si[["year", "month", "day"]])
sp = pd.read_csv(CSV_SPOT)
sp["date"] = pd.to_datetime(sp["date"])

idx = pd.date_range(DATE_START, DATE_END, freq="D", name="date")
d = pd.DataFrame(index=idx)
d["spot"] = sp.set_index("date")["n_spot_total"]
d["silso"] = si.set_index("date")["SN"]
d = d.reset_index()
both = d.dropna(subset=["spot", "silso"]).copy()

print(f"paired days: {len(both)} (of {len(d)} calendar days)")
print(f"daily        : Pearson r = {both['spot'].corr(both['silso']):.3f} | Spearman rho = {both['spot'].corr(both['silso'], method='spearman'):.3f}")

both["blk"] = (both["date"] - pd.Timestamp(DATE_START)).dt.days // BLOCK
g = both.groupby("blk").agg(n=("spot", "size"), spot=("spot", "mean"), silso=("silso", "mean"))
g = g[g["n"] >= MIN_DAYS]
print(f"{BLOCK}-day blocks (>= {MIN_DAYS} paired days): {len(g)} blocks | Pearson r = {g['spot'].corr(g['silso']):.3f} | Spearman rho = {g['spot'].corr(g['silso'], method='spearman'):.3f}")

d["spot_ma"] = d["spot"].rolling(MA, center=True, min_periods=1).mean()
d["silso_ma"] = d["silso"].rolling(MA, center=True, min_periods=1).mean()
print(f"reference, MA{MA}d: Pearson r = {d['spot_ma'].corr(d['silso_ma']):.3f}")
