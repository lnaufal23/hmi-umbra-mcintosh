#!/usr/bin/env python3
"""
srs_download.py: download and parse NOAA SRS for the daily cross-match.

- Parser (raw .txt -> rows in the srs_all_fresh.csv schema) VERIFIED identical to the
  builder of srs_all_fresh.csv: sunpy read_srs + reconstruction of Location (E = positive lon).
  Tested by reproducing srs_all_fresh.csv for 2017-04-29 EXACTLY (all cross-match columns match).
- Download of the current SRS from NOAA SWPC: https://services.swpc.noaa.gov/text/srs.txt
  (valid only for the latest SRS, i.e. today); past dates come from the per-date FTP archive.
  Saved to srs_raw_files/{YYYYMMDD}SRS.txt (archive); skipped if already present.
- build_srs_dataframe(dates): combines srs_all_fresh.csv (up to 2026-04-30) with parsed raw
  files for dates that are not in the CSV.
"""

import re
import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sunpy.io.special.srs import read_srs

from hmi_umbra import paths

log = logging.getLogger("hmi_umbra.srs")

SRS_RAW_DIR  = Path(paths.get("srs_raw_dir"))
SRS_CSV      = Path(paths.get("srs_csv"))
# /text/srs.txt = the CURRENT SRS only (today). Historical dates REQUIRE the per-date archive.
SWPC_SRS_URL = "https://services.swpc.noaa.gov/text/srs.txt"
# per-date archive (FTP warehouse), the authoritative source for past dates
WAREHOUSE_FTP = "ftp://ftp.swpc.noaa.gov/pub/warehouse/{year}/SRS/{ymd}SRS.txt"

_ISSUED_RE = re.compile(r":Issued:\s*(\d{4})\s+(\w{3})\s+(\d{1,2})")
_MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


def _curl(url, timeout=30):
    """Fetch a URL (http/https/ftp) with curl. Return the text or None."""
    try:
        r = subprocess.run(["curl", "-s", "-m", str(timeout), url],
                           capture_output=True, text=True, timeout=timeout + 10)
        return r.stdout if r.returncode == 0 else None
    except Exception as e:
        log.warning("[SRS] curl error %s: %s", url, e)
        return None


def _is_valid_srs(text):
    return bool(text) and ":Product:" in text and "Regions with Sunspots" in text


def _issued_date(text):
    """Issue date from the :Issued: line -> 'YYYY-MM-DD' or None."""
    m = _ISSUED_RE.search(text or "")
    if not m:
        return None
    y, mon, d = m.groups()
    mi = _MONTHS.get(mon)
    if not mi:
        return None
    return f"{int(y):04d}-{mi:02d}-{int(d):02d}"

CSV_COLUMNS = ["Nmbr", "Location", "Lo", "Area", "Z", "LL", "NN",
               "Mag Type", "Date", "Issued Time", "Latitude", "Longitude"]


def _raw_path_for(date_str):
    """date_str 'YYYY-MM-DD' -> path srs_raw_files/{YYYYMMDD}SRS.txt."""
    return SRS_RAW_DIR / f"{date_str.replace('-', '')}SRS.txt"


def download_srs(date_str, overwrite=False):
    """
    Download the SRS for date_str to {YYYYMMDD}SRS.txt. Return (path, status).
    status: 'exists' | 'archive' | 'current' | 'failed'.

    SOURCES (so that historical dates get the CORRECT SRS):
      1. Per-date FTP warehouse archive (authoritative for any date).
      2. Fallback to /text/srs.txt ONLY if its issue date == date_str
         (a request for the current SRS that is not archived yet). Content
         with a different issue date is NOT saved (prevents wrong-date SRS).
    """
    SRS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = _raw_path_for(date_str)
    if path.exists() and not overwrite:
        return str(path), "exists"

    ymd = date_str.replace("-", "")
    year = date_str[:4]

    # 1) per-date archive (FTP), correct for past dates
    txt = _curl(WAREHOUSE_FTP.format(year=year, ymd=ymd))
    if _is_valid_srs(txt):
        tmp = str(path) + ".tmp"
        Path(tmp).write_text(txt)
        Path(tmp).replace(path)
        log.info("[SRS] archive -> %s (%d bytes)", path, len(txt))
        return str(path), "archive"

    # 2) fallback to the current SRS, only if its issue date matches date_str
    cur = _curl(SWPC_SRS_URL)
    if _is_valid_srs(cur):
        iss = _issued_date(cur)
        if iss == date_str:
            tmp = str(path) + ".tmp"
            Path(tmp).write_text(cur)
            Path(tmp).replace(path)
            log.info("[SRS] current -> %s (issued %s)", path, iss)
            return str(path), "current"
        else:
            log.info("[SRS] %s not archived yet & /text/srs.txt issued %s (!=) -> not available yet",
                     date_str, iss)
    log.warning("[SRS] SRS %s not available (archive & current failed)", date_str)
    return None, "failed"


def parse_srs_file(path, date_str):
    """
    Raw SRS .txt -> DataFrame in the srs_all_fresh.csv schema (Section I / Regions with Sunspots).
    Recipe VERIFIED identical to the builder of srs_all_fresh.csv (sunpy read_srs + Location E=lon+).
    date_str = date used for the 'Date' column (= SRS issue date).
    """
    t = read_srs(str(path))
    rows = []

    def _gv(x):
        try:
            return float(x.value)
        except Exception:
            try:
                return float(x)
            except Exception:
                return None

    for r in t:
        if str(r["ID"]) != "I":
            continue
        lat_q, lon_q = r["Latitude"], r["Longitude"]
        # skip regions without valid (masked) coordinates
        if lat_q is None or lon_q is None:
            continue
        if getattr(lat_q, "mask", False) or getattr(lon_q, "mask", False):
            continue
        lat = float(lat_q.value)
        lon = float(lon_q.value)  # sunpy: W positive, E negative
        # reconstruct Location: lon>=0 -> 'E' (flip as in the srs_all_fresh.csv builder)
        loc = (f"{'N' if lat >= 0 else 'S'}{abs(int(round(lat))):02d}"
               f"{'E' if lon >= 0 else 'W'}{abs(int(round(lon))):02d}")
        nn = _gv(r["Number of Sunspots"])
        rows.append({
            "Nmbr": int(r["Number"]),
            "Location": loc,
            "Lo": _gv(r["Carrington Longitude"]),
            "Area": _gv(r["Area"]),
            "Z": str(r["Z"]),
            "LL": _gv(r["Longitudinal Extent"]),
            "NN": int(nn) if nn is not None else None,
            "Mag Type": str(r["Mag Type"]),
            "Date": date_str,
            "Issued Time": f"{date_str} 00:30:00",
            "Latitude": lat,
            "Longitude": lon,
        })
    return pd.DataFrame(rows, columns=CSV_COLUMNS)


def build_srs_dataframe(dates, allow_download=True):
    """
    Build srs_all (DataFrame in the CSV schema) for the required dates
    (e.g. [D-1, D, D+1]). Source priority:
      1. srs_all_fresh.csv (if the date is there)
      2. archive srs_raw_files/{YYYYMMDD}SRS.txt (parsed)
      3. download (if allow_download and not present yet) -> saved to the archive
    Return (DataFrame, info dict per date: 'csv'|'raw'|'downloaded'|'missing').
    """
    csv_df = pd.read_csv(SRS_CSV) if SRS_CSV.exists() else pd.DataFrame(columns=CSV_COLUMNS)
    csv_df["Date"] = csv_df["Date"].astype(str)
    csv_dates = set(csv_df["Date"].unique())

    frames = []
    src_info = {}
    for d in dates:
        if d in csv_dates:
            frames.append(csv_df[csv_df["Date"] == d])
            src_info[d] = "csv"
            continue
        raw = _raw_path_for(d)
        if not raw.exists() and allow_download:
            _, st = download_srs(d)
            if st in ("downloaded",):
                src_info[d] = "downloaded"
        if raw.exists():
            try:
                frames.append(parse_srs_file(raw, d))
                src_info.setdefault(d, "raw")
            except Exception as e:
                log.warning("[SRS] parse failed %s: %s", raw, e)
                src_info[d] = "missing"
        else:
            src_info.setdefault(d, "missing")

    if frames:
        out = pd.concat(frames, ignore_index=True)
    else:
        out = pd.DataFrame(columns=CSV_COLUMNS)
    out["Date"] = out["Date"].astype(str)
    return out, src_info


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    d = sys.argv[1] if len(sys.argv) > 1 else "2017-04-29"
    df, info = build_srs_dataframe([d], allow_download=False)
    print("source:", info)
    print(df.to_string())
