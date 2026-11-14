#!/usr/bin/env python3
"""Check the composition of the locked test set against the final v7 dataset before the final training: v7 labels available,
relabel R1, min-40 classes, and non-test crops in the same group (union-find train_audit_native.build_groups) as locked-test crops.
Writes split_final_lock.csv (set_: test_lock | pool | excluded)."""
import sys
from pathlib import Path

import pandas as pd

import train_audit_native as ta
from hmi_umbra import paths

R = Path(paths.get("cls_root"))
lock = pd.read_csv(R / "test_lock/test_locked.csv")
v7 = pd.read_csv(R / "train_audit/v7_relabel_A/export_selected_v7.csv", dtype={"date": str})
v7 = v7[v7.audit_status.isin(["kept", "changed"])].copy()
v7["final_label"] = v7["label_audit"].astype(str)
vc = v7.final_label.value_counts(); keep_cls = set(vc[vc >= 40].index)
relab = pd.read_csv(R / "opt_v5/relabel_r1/ZPC_RELABEL_R1_export.csv")
rl_touched = set(relab.stem); rl_changed = set(relab[relab.status_relabel == "changed"].stem)

L = set(lock.stem)
print(f"locked test: {len(L)} crops | per year: {lock.stem.str[:4].value_counts().sort_index().to_dict()}")
inv7 = v7[v7.stem.isin(L)]
print(f"  with a v7 label (kept/changed): {len(inv7)} | per year: {inv7.stem.str[:4].value_counts().sort_index().to_dict()}")
print(f"  not in v7 (e.g. 2013 job>518 / dropped): {len(L - set(inv7.stem))}")
inc = inv7[inv7.final_label.isin(keep_cls)]
print(f"  in the 35 min40 classes: {len(inc)} | outside (rare classes): {len(inv7) - len(inc)}")
print(f"  in relabel R1: {len(L & rl_touched)} (changed {len(L & rl_changed)}) | untouched by relabel & in 35 classes: "
      f"{len(set(inc.stem) - rl_touched)}")

d = v7[v7.final_label.isin(keep_cls)].reset_index(drop=True)
d = ta.build_groups(d)
g_lock = set(d[d.stem.isin(L)].group_id)
same = d[~d.stem.isin(L) & d.group_id.isin(g_lock)]
pool = d[~d.stem.isin(L) & ~d.group_id.isin(g_lock)]
print(f"\nv7 dataset, 35 classes: {len(d)} crops | groups: {d.group_id.nunique()}")
print(f"  locked test (35 classes): {d.stem.isin(L).sum()} crops in {len(g_lock)} groups")
print(f"  non-test crops in a group with locked-test crops (excluded from training): {len(same)} | per year: "
      f"{same.stem.str[:4].value_counts().sort_index().to_dict()}")
print(f"  final training pool (train+val): {len(pool)} crops | groups {pool.group_id.nunique()}")
cmp = pd.DataFrame({"test_lock": d[d.stem.isin(L)].final_label.value_counts(), "pool": pool.final_label.value_counts()}).fillna(0).astype(int)
print("\nclass distribution (locked test vs pool):")
print(cmp.sort_values("test_lock").T.to_string())
d.assign(set_=lambda x: x.stem.isin(L).map({True: "test_lock", False: "pool"}).where(~(~x.stem.isin(L) & x.group_id.isin(g_lock)), "excluded")) \
 .to_csv(R / "opt_v5/final_lock/split_final_lock.csv", index=False)
print(f"saved: {R / 'opt_v5/final_lock/split_final_lock.csv'}")
