#!/usr/bin/env python3
"""
Live weather + rain for every cell of the static grid (Environment Canada GeoMet).

Output columns per cell (same names the model uses):
  precip_on_day     mm  rain in the LAST 24 h        (RDPA 6-hour rain analysis, 4 slices)
  precip_prior_7d   mm  rain in the 7 days before that (7 x 24 h blocks)
  days_since_rain   days since a 24 h block had > 1 mm (7 if none)  [same rule as training]
  temp_on_day       C   mean of the NEXT 24 h forecast (HRDPS 2 m temperature)
  rel_humidity_pct  %   from mean temperature + mean dew point (Magnus), like the ERA5 training
  wind_speed_ms     m/s mean of the NEXT 24 h forecast wind speed (HRDPS 10 m)

NOTE (honest approximations): training used ERA5 daily means for the same UTC day;
live weather has no history, so temperature/humidity/wind come from the next-24 h
forecast, and rain from the last 24 h of observed analysis. Windows are "rolling from
now", not UTC calendar days.

Usage (Colab):
  %run weather_live.py --grid /content/drive/MyDrive/wildfire_project/lightning_project/live_static_grid.csv --out /content/wx
"""
import argparse, os, sys, re, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

BASE = "https://geo.weather.gc.ca/geomet"
CELL = 0.1
# bbox shifted half a cell so pixel centres sit on the 0.1-degree cell centres
BX0, BX1, BY0, BY1 = -141.05, -52.45, 41.65, 66.05
W = int(round((BX1 - BX0) / CELL)); H = int(round((BY1 - BY0) / CELL))


def coverage(layer, t):
    p = dict(service="WCS", version="2.0.1", request="GetCoverage", coverageid=layer,
             format="image/tiff", time=t.strftime("%Y-%m-%dT%H:%M:%SZ"),
             subset=[f"x({BX0},{BX1})", f"y({BY0},{BY1})"], scalesize=f"x({W}),y({H})",
             subsettingcrs="EPSG:4326", outputcrs="EPSG:4326")   # without these the rain grid came back flat
    for attempt in range(3):
        try:
            r = requests.get(BASE, params=p, timeout=120)
            ct = r.headers.get("content-type", "")
            if r.status_code == 200 and ("tif" in ct or r.content[:2] in (b"II", b"MM")):
                return r.content
        except Exception:
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


def sample(content, lats, lons):
    from rasterio.io import MemoryFile
    with MemoryFile(content) as mf, mf.open() as ds:
        a = ds.read(1).astype("float64")
        nod = ds.nodata
        tr = ds.transform
    # pixel index by plain arithmetic (works on every rasterio version, handles arrays)
    cols = np.floor((lons - tr.c) / tr.a).astype(int)
    rows = np.floor((lats - tr.f) / tr.e).astype(int)
    ok = (rows >= 0) & (rows < a.shape[0]) & (cols >= 0) & (cols < a.shape[1])
    out = np.full(len(lats), np.nan)
    out[ok] = a[rows[ok], cols[ok]]
    if nod is not None:
        out[out == nod] = np.nan
    out[np.abs(out) > 1e10] = np.nan
    return out


def layer_units(layer, t):
    """Read the layer title (it contains the unit, e.g. [kts]) with one point query."""
    p = dict(service="WMS", version="1.3.0", request="GetFeatureInfo", layers=layer, query_layers=layer,
             crs="EPSG:4326", bbox="49.95,-100.05,50.05,-99.95", width=101, height=101, i=50, j=50,
             info_format="application/json", time=t.strftime("%Y-%m-%dT%H:%M:%SZ"))
    try:
        r = requests.get(BASE, params=p, timeout=60).json()
        return r["features"][0]["properties"].get("title_en", "")
    except Exception:
        return ""


def fetch_many(layer, times, lats, lons, workers=6):
    def one(t):
        c = coverage(layer, t)
        return sample(c, lats, lons) if c is not None else None
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, times))
    return res


def run(grid_path, out_dir, asof=None):
    os.makedirs(out_dir, exist_ok=True)
    g = pd.read_csv(grid_path)
    lats, lons = g["LAT"].values.astype(float), g["LON"].values.astype(float)
    now = asof or datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    log = {}

    # ---------- RAIN: last 24 h and 7 earlier 24 h blocks from RDPA 6-hour analysis ----------
    # newest available slice = latest 6-hour stamp at least ~7 h old (analysis lag); we probe backwards
    base6 = now.replace(hour=(now.hour // 6) * 6)
    t_last, probe = None, base6
    for _ in range(6):
        c = coverage("RDPA.6P_PR", probe)
        if c is not None:
            t_last = probe; break
        probe -= timedelta(hours=6)
    if t_last is None:
        raise RuntimeError("could not find a recent RDPA 6-hour rain slice")
    log["rain_latest_slice"] = t_last.strftime("%Y-%m-%dT%H:%MZ")
    slice_times = [t_last - timedelta(hours=6 * k) for k in range(32)]      # 8 blocks x 4 slices
    rain = fetch_many("RDPA.6P_PR", slice_times, lats, lons)
    nfail = sum(r is None for r in rain)
    log["rain_slices_failed"] = nfail
    rain_arr = np.vstack([np.full(len(g), np.nan) if r is None else r for r in rain])    # 32 x cells
    rain_arr = np.where(rain_arr < 0, 0, rain_arr)
    blocks = np.nansum(rain_arr.reshape(8, 4, -1), axis=1)                            # 8 x cells (mm per 24 h)
    allnan = np.isnan(rain_arr.reshape(8, 4, -1)).all(axis=1)
    blocks[allnan] = np.nan
    g["precip_on_day"] = blocks[0]
    g["precip_prior_7d"] = np.nansum(blocks[1:8], axis=0)
    dsr = np.full(len(g), 7.0)
    for L in range(7, 0, -1):                                                       # same rule as training
        wet = blocks[L] > 1.0
        dsr = np.where(wet, L - 1, dsr)
    g["days_since_rain"] = dsr

    # ---------- WEATHER: next 24 h forecast from HRDPS ----------
    first = now + timedelta(hours=1)
    ftimes = [first + timedelta(hours=k) for k in range(24)]
    tt = fetch_many("HRDPS.CONTINENTAL_TT", ftimes, lats, lons)
    td = fetch_many("HRDPS.CONTINENTAL_TD", ftimes, lats, lons)
    ws = fetch_many("HRDPS.CONTINENTAL_WSPD", ftimes, lats, lons)   # wind speed (UU is arrows only, no WCS)
    wd = fetch_many("HRDPS.CONTINENTAL_WD", ftimes, lats, lons)     # wind direction (degrees)
    log["tt_failed"] = sum(x is None for x in tt); log["td_failed"] = sum(x is None for x in td)
    log["wspd_failed"] = sum(x is None for x in ws); log["wd_failed"] = sum(x is None for x in wd)

    def mean_of(lst):
        arr = [x for x in lst if x is not None]
        return np.nanmean(np.vstack(arr), axis=0) if arr else np.full(len(g), np.nan)
    T = mean_of(tt); TD = mean_of(td)
    g["temp_on_day"] = T
    if np.isfinite(TD).any():                                        # humidity like training: Magnus on mean T and Td
        g["rel_humidity_pct"] = 100 * np.exp(17.625 * TD / (243.04 + TD)) / np.exp(17.625 * T / (243.04 + T))
        log["humidity_source"] = "TT+TD (Magnus)"
    else:
        hr = mean_of(fetch_many("HRDPS.CONTINENTAL_HR", ftimes, lats, lons))
        g["rel_humidity_pct"] = hr * (100 if np.nanmax(hr) <= 1.5 else 1)
        log["humidity_source"] = "HR layer (fallback)"
    title = layer_units("HRDPS.CONTINENTAL_WSPD", first)
    log["wind_layer_title"] = title
    if "kts" in title or "kt]" in title:
        f = 0.514444
    elif "km/h" in title:
        f = 1 / 3.6
    elif "m/s" in title:
        f = 1.0
    else:
        log["wind_unit_warning"] = "unit not recognised - assumed km/h"
        f = 1 / 3.6
    us, vs, sp = [], [], []
    for s_, d_ in zip(ws, wd):
        if s_ is None:
            continue
        s_ = s_ * f
        sp.append(s_)
        if d_ is not None:                                   # u,v from speed + direction ("coming from")
            rad = np.radians(d_)
            us.append(-s_ * np.sin(rad)); vs.append(-s_ * np.cos(rad))
    if us:                                                   # same definition as training: speed of the MEAN wind vector
        U = np.sqrt(np.nanmean(np.vstack(us), axis=0) ** 2 + np.nanmean(np.vstack(vs), axis=0) ** 2)
        log["wind_method"] = "vector mean of hourly u,v (like ERA5 training)"
    elif sp:
        U = np.nanmean(np.vstack(sp), axis=0)
        log["wind_method"] = "scalar mean speed (direction missing) - runs higher than training"
    else:
        U = np.full(len(g), np.nan); log["wind_method"] = "FAILED"
    if sp:
        log["wind_scalar_mean_median_ms"] = round(float(np.nanmedian(np.nanmean(np.vstack(sp), axis=0))), 2)
    g["wind_speed_ms"] = U
    g["asof_utc"] = now.strftime("%Y-%m-%dT%H:%MZ")

    out = os.path.join(out_dir, f"live_weather_{now:%Y%m%dT%H}Z.csv")
    g.to_csv(out, index=False)
    return g, log, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="/content/drive/MyDrive/wildfire_project/lightning_project/live_static_grid.csv")
    ap.add_argument("--out", default="/content/drive/MyDrive/wildfire_project/lightning_project/live_weather")
    a, _ = ap.parse_known_args()
    g, log, out = run(a.grid, a.out)
    print("saved", out, g.shape)
    for k, v in log.items():
        print(" ", k, ":", v)
    cols = ["precip_on_day", "precip_prior_7d", "days_since_rain", "temp_on_day", "rel_humidity_pct", "wind_speed_ms"]
    print("\nfilled share and summary:")
    print(pd.DataFrame({"filled": g[cols].notna().mean().round(3), "min": g[cols].min().round(1),
                        "median": g[cols].median().round(1), "max": g[cols].max().round(1)}))
