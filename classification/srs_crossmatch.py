#!/usr/bin/env python3
"""
srs_crossmatch.py: stages 1-2 of the McIntosh classification pipeline (SRS labels) for the AR crops.
Matching rules:
  - nearest issued SRS (-1/0/+1 day), Stonyhurst at issue time -> HPC at observation time -> 4096 pixel;
    match if the NOAA point lies inside the box or misses it by <= 0.5*min(w,h); label rules below.
  - KEEP status {noaa_single, noaa_multi_ok, auto_axx, auto_bxo},
    |lat|,|lon| <= 60 (center = chosen NOAA location, else box center), extent_max_deg from the 4 corners.
Notes:
  - Boxes = final CVAT annotations AR_POS_2012_2026 (x_1024 * 4), not model detections.
  - The CVAT-corrected Zurich class is used for the 0-NOAA rule and the multi-match check.
  - Multi-match picks the NOAA region with the largest area_msh (an earlier version used the key 'area',
    which was always 0, i.e. SRS order). Column label_if_old_bug records the label under that old rule.
Output: <cls_root>/srs_match/
"""
import sys
import json
import time
from pathlib import Path
from datetime import timedelta
from collections import Counter

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from sunpy.coordinates import frames, Helioprojective

from srs_download import build_srs_dataframe  # noqa: E402
from hmi_umbra import paths  # noqa: E402

ANN_CSV = Path(paths.get("cls_root")) / "final_annotations/ar_pos_final_2012_2026.csv"
CROP_INDEX = Path(paths.get("cls_root")) / "crops_4096/crop_index.csv"
WCS_CACHE = Path(paths.get("fits_wcs_cache"))
OUTDIR = Path(paths.get("cls_root")) / "srs_match"

NATIVE_SIZE = 4096
SCALE = 4.0
MISS_TOLERANCE_FRAC = 0.5
KEEP_STATUS = {"noaa_single", "noaa_multi_ok", "auto_axx", "auto_bxo"}
LAT_LIMIT_DEG = 60.0
LON_LIMIT_DEG = 60.0
RSUN = 696000 * u.km


def parse_issued(issued_str, date_str):
    try:
        return Time(pd.to_datetime(issued_str))
    except Exception:
        return Time(date_str + "T00:30:00")


def pick_best_srs_date(srs_by_date, obs_time):
    obs_dt = obs_time.datetime
    cands = []
    for delta in (-1, 0, 1):
        d = (obs_dt + timedelta(days=delta)).strftime("%Y-%m-%d")
        sub = srs_by_date.get(d)
        if sub is None or len(sub) == 0:
            continue
        issued = parse_issued(sub.iloc[0]["Issued Time"], d)
        cands.append((abs((issued - obs_time).to(u.hour).value), d, issued, sub))
    if not cands:
        return None, None, None, None
    cands.sort(key=lambda c: c[0])
    dt_h, d, issued, sub = cands[0]
    return d, issued, sub.reset_index(drop=True), dt_h


def hpc_to_pixel(tx, ty, w):
    x = w["CRPIX1"] + (tx - w["CRVAL1"]) / w["CDELT1"] - 1
    y = w["CRPIX2"] - (ty - w["CRVAL2"]) / w["CDELT2"] - 1
    return x, y


def pixel_to_hpc(x, y, w):
    tx = w["CRVAL1"] + (x - w["CRPIX1"] + 1) * w["CDELT1"]
    ty = w["CRVAL2"] - (y - w["CRPIX2"] + 1) * w["CDELT2"]
    return tx, ty


def hpc_to_hgs(tx, ty, obs_time):
    hpc = SkyCoord(np.atleast_1d(tx) * u.arcsec, np.atleast_1d(ty) * u.arcsec,
                   frame=Helioprojective(obstime=obs_time, observer="earth", rsun=RSUN))
    hgs = hpc.transform_to(frames.HeliographicStonyhurst(obstime=obs_time))
    return hgs.lat.to(u.deg).value, hgs.lon.to(u.deg).value


def find_matches(bb, pts):
    x1, y1, x2, y2 = bb
    tol = MISS_TOLERANCE_FRAC * min(x2 - x1, y2 - y1)
    out = []
    for p in pts:
        cx = max(x1, min(p["x_pix"], x2))
        cy = max(y1, min(p["y_pix"], y2))
        md = float(np.hypot(p["x_pix"] - cx, p["y_pix"] - cy))
        if md <= tol:
            out.append({"noaa": p, "miss_dist": md})
    return out


def label_rule(matches, zcls):
    """Return (final_label, status, flag_reason, chosen, label_if_old_bug)."""
    if len(matches) == 0:
        if zcls == "a":
            return "Axx", "auto_axx", None, None, "Axx"
        if zcls == "b":
            return "Bxo", "auto_bxo", None, None, "Bxo"
        return None, "skip", None, None, None
    if len(matches) == 1:
        n = matches[0]["noaa"]
        return n["z"], "noaa_single", None, n, n["z"]
    old_bug = matches[0]["noaa"]["z"]  # old: sort key 'area' always 0 -> SRS order
    chosen = sorted(matches, key=lambda m: -m["noaa"]["area_msh"])[0]["noaa"]
    letter = chosen["z"][0].upper() if chosen["z"] else "?"
    letters = {(m["noaa"]["z"][0].upper() if m["noaa"]["z"] else "?") for m in matches}
    consistent = len(letters) == 1
    zmatch = zcls.upper() == letter
    if consistent and zmatch:
        return chosen["z"], "noaa_multi_ok", None, chosen, old_bug
    if not consistent and not zmatch:
        reason = "noaa_z_inconsistent_and_zclass_mismatch"
    elif not consistent:
        reason = "noaa_z_inconsistent"
    else:
        reason = "zclass_mismatch"
    return chosen["z"], "noaa_multi_flag", reason, chosen, old_bug


def main():
    t0 = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ann = pd.read_csv(ANN_CSV, dtype={"date": str})
    idx = pd.read_csv(CROP_INDEX, dtype={"date": str})[["stem", "jpg_crop", "fits_crop", "w_4096", "h_4096"]]
    wcs = pd.read_csv(WCS_CACHE, dtype={"date": str}).drop_duplicates("date").set_index("date")
    dates = sorted(ann["date"].unique())

    need = set()
    for d in dates:
        t = pd.Timestamp(d)
        for k in (-2, -1, 0, 1, 2):
            need.add((t + pd.Timedelta(days=k)).strftime("%Y-%m-%d"))
    srs_all, src = build_srs_dataframe(sorted(need), allow_download=False)
    srs_all["Date"] = srs_all["Date"].astype(str)
    srs_by_date = {d: g for d, g in srs_all.groupby("Date")}
    print(f"[SRS] rows {len(srs_all)} | date sources: {dict(Counter(src.values()))}", flush=True)

    rows = []
    for i, d in enumerate(dates, 1):
        w = wcs.loc[d].to_dict()
        obs = Time(w["DATE_OBS"], scale="utc")
        best_date, issued, srs_today, dt_h = pick_best_srs_date(srs_by_date, obs)

        pts = []
        if srs_today is not None and len(srs_today):
            locs = []
            for _, r in srs_today.iterrows():
                loc = str(r["Location"]).strip()
                if len(loc) < 6 or loc[0] not in "NS" or loc[3] not in "EW":
                    continue
                lat = float(loc[1:3]) * (1 if loc[0] == "N" else -1)
                lon = float(loc[4:]) * (1 if loc[3] == "W" else -1)
                locs.append((r, loc, lat, lon))
            if locs:
                sc = SkyCoord([l[3] for l in locs] * u.deg, [l[2] for l in locs] * u.deg,
                              frame=frames.HeliographicStonyhurst, obstime=issued, observer="earth")
                hp = sc.transform_to(Helioprojective(obstime=obs, observer="earth", rsun=RSUN))
                txs, tys = hp.Tx.to(u.arcsec).value, hp.Ty.to(u.arcsec).value
                for (r, loc, lat, lon), tx, ty in zip(locs, txs, tys):
                    if not (np.isfinite(tx) and np.isfinite(ty)):
                        continue
                    xp, yp = hpc_to_pixel(tx, ty, w)
                    if not (0 <= xp < NATIVE_SIZE and 0 <= yp < NATIVE_SIZE):
                        continue
                    try:
                        area = float(r.get("Area", 0))
                    except (TypeError, ValueError):
                        area = 0.0
                    pts.append({"noaa_nmbr": str(int(r["Nmbr"])) if pd.notna(r["Nmbr"]) else "?",
                                "location": loc, "z": str(r.get("Z", "")).strip(),
                                "mag_type": str(r.get("Mag Type", "")).strip(),
                                "lat_stony_deg": lat, "lon_stony_deg": lon,
                                "x_pix": float(xp), "y_pix": float(yp),
                                "area_msh": area if np.isfinite(area) else 0.0})

        sub = ann[ann["date"] == d]
        bbs = np.stack([sub["x1_1024"].values * SCALE, sub["y1_1024"].values * SCALE,
                        sub["x2_1024"].values * SCALE, sub["y2_1024"].values * SCALE], axis=1)
        cx, cy = (bbs[:, 0] + bbs[:, 2]) / 2, (bbs[:, 1] + bbs[:, 3]) / 2
        tcx, tcy = pixel_to_hpc(cx, cy, w)
        blat, blon = hpc_to_hgs(tcx, tcy, obs)
        corners_x = np.stack([bbs[:, 0], bbs[:, 2], bbs[:, 2], bbs[:, 0]], axis=1).ravel()
        corners_y = np.stack([bbs[:, 1], bbs[:, 1], bbs[:, 3], bbs[:, 3]], axis=1).ravel()
        ctx, cty = pixel_to_hpc(corners_x, corners_y, w)
        clat, clon = hpc_to_hgs(ctx, cty, obs)
        clat, clon = clat.reshape(-1, 4), clon.reshape(-1, 4)

        for k, (_, a) in enumerate(sub.iterrows()):
            bb = bbs[k]
            zcls = str(a["label"]).lower()
            matches = find_matches(bb, pts)
            final_label, status, reason, chosen, old_bug = label_rule(matches, zcls)

            if chosen is not None:
                lat_c, lon_c, csrc = chosen["lat_stony_deg"], chosen["lon_stony_deg"], "noaa_chosen"
            elif matches:
                lat_c, lon_c, csrc = matches[0]["noaa"]["lat_stony_deg"], matches[0]["noaa"]["lon_stony_deg"], "noaa_first"
            else:
                lat_c, lon_c, csrc = float(blat[k]), float(blon[k]), "bbox_center"

            la, lo = clat[k], clon[k]
            if np.all(np.isfinite(la)) and np.all(np.isfinite(lo)):
                lon_rng = lo.max() - lo.min()
                if lon_rng > 180:
                    lo2 = np.where(lo < 0, lo + 360, lo)
                    lon_rng = lo2.max() - lo2.min()
                lat_rng = la.max() - la.min()
                ext_max, ext_lat, ext_lon = float(max(lat_rng, lon_rng)), float(lat_rng), float(lon_rng)
            else:
                ext_max = ext_lat = ext_lon = np.nan

            if status not in KEEP_STATUS:
                reject = f"status_{status}"
            elif not (np.isfinite(lat_c) and np.isfinite(lon_c)):
                reject = "center_helio_fail"
            elif abs(lat_c) > LAT_LIMIT_DEG or abs(lon_c) > LON_LIMIT_DEG:
                reject = "beyond_60deg"
            elif not np.isfinite(ext_max):
                reject = "extent_compute_fail"
            else:
                reject = ""

            rows.append({
                "stem": f"{d}_bb{int(a['box_id']):02d}", "year": int(a["year"]), "date": d, "box_id": int(a["box_id"]),
                "zclass_cvat": zcls,
                "bbox_x1_4096": bb[0], "bbox_y1_4096": bb[1], "bbox_x2_4096": bb[2], "bbox_y2_4096": bb[3],
                "bbox_w_pix": bb[2] - bb[0], "bbox_h_pix": bb[3] - bb[1], "bbox_max_side_pix": max(bb[2] - bb[0], bb[3] - bb[1]),
                "status": status, "flag_reason": reason, "final_label": final_label, "label_if_old_bug": old_bug,
                "n_noaa_match": len(matches), "chosen_noaa": chosen["noaa_nmbr"] if chosen else None,
                "chosen_area_msh": chosen["area_msh"] if chosen else None,
                "chosen_mag_type": chosen["mag_type"] if chosen else None,
                "noaa_matches": json.dumps([{"nmbr": m["noaa"]["noaa_nmbr"], "z": m["noaa"]["z"], "area": m["noaa"]["area_msh"],
                                             "loc": m["noaa"]["location"], "miss": round(m["miss_dist"], 1)} for m in matches]),
                "srs_date_used": best_date, "srs_dt_hours": round(dt_h, 2) if dt_h is not None else None,
                "obs_time": str(obs.isot),
                "center_lat_deg": lat_c, "center_lon_deg": lon_c, "center_source": csrc,
                "bbox_center_lat_deg": float(blat[k]), "bbox_center_lon_deg": float(blon[k]),
                "extent_max_deg": ext_max, "extent_lat_deg": ext_lat, "extent_lon_deg": ext_lon,
                "keep": reject == "", "reject_reason": reject,
            })
        if i % 500 == 0 or i == len(dates):
            print(f"  {i}/{len(dates)} dates | {time.time()-t0:.0f} s", flush=True)

    df = pd.DataFrame(rows).merge(idx, on="stem", how="left")
    df.to_csv(OUTDIR / "srs_match_all.csv", index=False)
    kept = df[df["keep"]].copy()
    kept.to_csv(OUTDIR / "srs_match_kept.csv", index=False)

    vc = kept["final_label"].value_counts()
    multi = df[df["status"].isin(["noaa_multi_ok", "noaa_multi_flag"])]
    bug_diff = multi[multi["final_label"] != multi["label_if_old_bug"]]
    summary = {
        "n_total": len(df), "n_kept": len(kept),
        "status_counts": dict(Counter(df["status"])),
        "flag_reason_counts": dict(Counter(df["flag_reason"].dropna())),
        "reject_counts": dict(Counter(df.loc[~df["keep"], "reject_reason"])),
        "kept_status_counts": dict(Counter(kept["status"])),
        "kept_label_counts": {k: int(v) for k, v in vc.items()},
        "n_classes_kept": int(vc.size), "n_classes_ge100": int((vc >= 100).sum()), "n_classes_ge40": int((vc >= 40).sum()),
        "n_unique_noaa_kept": int(kept["chosen_noaa"].dropna().nunique()),
        "multi_match_total": len(multi), "multi_label_changed_by_bugfix": len(bug_diff),
        "srs_dt_hours_max": float(df["srs_dt_hours"].max()),
        "missing_crop_link": int(df["jpg_crop"].isna().sum()),
    }
    json.dump(summary, open(OUTDIR / "srs_match_summary.json", "w"), indent=2)

    print("\n================ CROSS-MATCH SUMMARY ================")
    for k in ("n_total", "n_kept", "status_counts", "flag_reason_counts", "reject_counts", "kept_status_counts",
              "n_classes_kept", "n_classes_ge100", "n_classes_ge40", "n_unique_noaa_kept",
              "multi_match_total", "multi_label_changed_by_bugfix", "srs_dt_hours_max", "missing_crop_link"):
        print(f"{k:<32}: {summary[k]}")
    print("\nLabels (kept), by count:")
    print(", ".join(f"{k}:{v}" for k, v in vc.items()))
    print(f"\nOutput: {OUTDIR}/srs_match_all.csv, srs_match_kept.csv, srs_match_summary.json | {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
