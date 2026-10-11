#!/usr/bin/env python3
"""
Lightning recorder - saves Environment Canada's real-time lightning (CLDN)
to small CSV files, because GeoMet only keeps the last ~3 hours.

Layer: Lightning_2.5km_Density  (flashes per km2 per minute, 10-minute steps,
cloud-to-ground + intra-cloud). Each run fetches every 10-min step of the last
~3 hours that we have not saved yet, converts density to flash COUNTS, adds
them up onto the 0.1-degree grid used by the model, and keeps only cells with
lightning.

Outputs (in --out):
  flashes_YYYY-MM-DD.csv   columns: t (UTC step), LAT, LON (0.1 deg cell centre), flashes, pixels
  steps_YYYY-MM.csv        columns: t, status (ok), n_active_cells, fetched_at
                           -> a step listed here with no rows in flashes_* means
                              "recorded, no lightning". A step missing from steps_* means
                              "no data" (never treat that as zero).

Usage:  python record_lightning.py --out data
"""
import argparse, io, os, sys, time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

BASE = "https://geo.weather.gc.ca/geomet"
LAYER = "Lightning_2.5km_Density"
X0, X1, Y0, Y1 = -141.0, -52.5, 41.7, 66.0      # Canada, south border to ~66N
CELL = 0.1                                       # model grid, degrees
NATIVE_DEG = 0.0225                              # ~2.5 km
STEP_MIN = 10
WINDOW_STEPS = 19                                # ~3 hours kept by the server


def fetch_tiff(t, width, height):
    p = dict(service="WCS", version="2.0.1", request="GetCoverage", coverageid=LAYER,
             format="image/tiff", time=t.strftime("%Y-%m-%dT%H:%M:%SZ"),
             subset=[f"x({X0},{X1})", f"y({Y0},{Y1})"], scalesize=f"x({width}),y({height})")
    r = requests.get(BASE, params=p, timeout=180)
    ct = r.headers.get("content-type", "")
    if r.status_code == 200 and ("tif" in ct or r.content[:2] in (b"II", b"MM")):
        return r.content
    return None


def tiff_to_cells(content, t):
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.transform import xy
    with MemoryFile(content) as mf, mf.open() as ds:
        a = ds.read(1).astype("float64")
        tr = ds.transform
        nodata = ds.nodata
    if nodata is not None:
        a[a == nodata] = 0.0
    a[~np.isfinite(a)] = 0.0
    a[a < 0] = 0.0
    rows, cols = np.nonzero(a > 0)
    if len(rows) == 0:
        return pd.DataFrame(columns=["t", "LAT", "LON", "flashes", "pixels"]), 0
    lon, lat = xy(tr, rows, cols)               # pixel centres
    lon, lat = np.asarray(lon), np.asarray(lat)
    dlon, dlat = abs(tr.a), abs(tr.e)
    area = (dlat * 111.32) * (dlon * 111.32 * np.cos(np.radians(lat)))   # km2 per pixel
    flashes = a[rows, cols] * STEP_MIN * area                              # density * minutes * area
    df = pd.DataFrame({
        "LAT": np.round(np.round(lat / CELL) * CELL, 1),
        "LON": np.round(np.round(lon / CELL) * CELL, 1),
        "flashes": flashes, "pixels": 1})
    # NOTE: "flashes" = density * 10 min * pixel area. The server's smallest non-zero level
    # looks like one flash, so the absolute unit may be off by a constant factor; we rank-match
    # later, so only the ORDER matters. "pixels" = number of lit 2.5 km pixels (unit-free).
    df = df.groupby(["LAT", "LON"], as_index=False)[["flashes", "pixels"]].sum()
    df["flashes"] = df["flashes"].round(3)
    df.insert(0, "t", t.strftime("%Y-%m-%dT%H:%M:%SZ"))
    return df, len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--steps", type=int, default=WINDOW_STEPS)
    args, _ = ap.parse_known_args()          # ignore extra args (Colab/Jupyter adds some)
    if "ipykernel" in sys.modules and args.out == "data":
        args.out = "/content/rec_test"        # default for notebooks
    os.makedirs(args.out, exist_ok=True)

    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    now -= timedelta(minutes=now.minute % STEP_MIN)
    wanted = [now - timedelta(minutes=STEP_MIN * k) for k in range(0, args.steps)]

    # which steps do we already have?
    seen = set()
    for t in wanted:
        f = os.path.join(args.out, f"steps_{t:%Y-%m}.csv")
        if os.path.exists(f) and f not in seen:
            seen.add(f)
    have = set()
    for f in seen:
        have |= set(pd.read_csv(f)["t"])

    W = int(round((X1 - X0) / NATIVE_DEG)); H = int(round((Y1 - Y0) / NATIVE_DEG))
    ok = fail = 0
    for t in sorted(wanted):
        ts = t.strftime("%Y-%m-%dT%H:%M:%SZ")
        if ts in have:
            continue
        content = fetch_tiff(t, W, H)
        if content is None and (W, H) != (1780, 570):
            content = fetch_tiff(t, 1780, 570)           # fallback (coarser) - logged below
            res_note = "coarse"
        else:
            res_note = "native"
        if content is None:
            fail += 1
            continue
        df, n = tiff_to_cells(content, t)
        if n:
            f = os.path.join(args.out, f"flashes_{t:%Y-%m-%d}.csv")
            df.to_csv(f, mode="a", header=not os.path.exists(f), index=False)
        f = os.path.join(args.out, f"steps_{t:%Y-%m}.csv")
        pd.DataFrame([{"t": ts, "status": "ok", "n_active_cells": n, "res": res_note,
                       "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}]).to_csv(
            f, mode="a", header=not os.path.exists(f), index=False)
        ok += 1
        time.sleep(0.5)                                   # be polite to the server
    print(f"done: {ok} new steps saved, {fail} failed (older steps often fall out of the 3h window), "
          f"{len(have)} already had")


if __name__ == "__main__":
    main()
