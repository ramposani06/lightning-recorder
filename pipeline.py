#!/usr/bin/env python3
"""
Live lightning risk pipeline for GitHub Actions (no Colab needed).

Reads   static/   live_static_grid.csv, live_ndvi.csv, live_model_no_cg1.pkl, live_calibrator_platt.pkl,
                  live_features_list.pkl, requirements.txt       (uploaded once from Colab)
        data/     lightning recorded every 2 h by record_lightning.py
Fetches live weather + rain (GeoMet) and recent fires (CWFIS) every run.
Writes  docs/index.html (the map), docs/latest_scores.csv, archive/scores_<asof>.csv.gz (for later testing),
        cache/live_recent_fires.csv (fallback if CWFIS is down).
If something important fails, it STOPS with an error so the last good map stays online.
"""
import glob, io, os, pickle, time
from datetime import datetime, timezone
import numpy as np, pandas as pd, requests

ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC, DATA, DOCS, ARCH, CACHE = [f"{ROOT}/{x}" for x in ("static", "data", "docs", "archive", "cache")]
for p in (DOCS, ARCH, CACHE):
    os.makedirs(p, exist_ok=True)

LIGHTNING_SCALE = 0.35        # assumed ground-flash share of all flashes (to be calibrated later)
NLAGS = 15
LAT0, LON0, STEP = 41.7, -141.0, 0.1
NLAT, NLON = int(round((66.0 - LAT0) / STEP)) + 1, int(round((-52.5 - LON0) / STEP)) + 1
WFS = "https://geoserver.cwfif.nrcan.gc.ca/geoserver/wfs"
PROPS = "national_fire_id,agency_code,latitude,longitude,situation_report_date,status_date,fire_was_prescribed,national_fire_cause"
now_utc = datetime.now(timezone.utc)


def key(df):
    df = df.copy(); df["LAT"] = df["LAT"].round(1); df["LON"] = df["LON"].round(1); return df


# ---------------------------------------------------------------- 1. weather + rain
import weather_live
g, wlog, _ = weather_live.run(f"{STATIC}/live_static_grid.csv", f"{ROOT}/tmp_weather")
print("weather:", wlog)
wxc = ["precip_on_day", "precip_prior_7d", "days_since_rain", "temp_on_day", "rel_humidity_pct", "wind_speed_ms"]
share = g[wxc].notna().mean()
print("weather filled share:\n", share.round(3).to_string())
if (share < 0.9).any():
    raise RuntimeError("weather is incomplete - stopping so the last good map stays online")
d = key(g)
weather_asof = g["asof_utc"].iloc[0]


# ---------------------------------------------------------------- 2. recent fires (CWFIS)
def cwfis_year(year, page=10000):
    parts, start = [], 0
    while True:
        p = dict(service="WFS", version="2.0.1", request="GetFeature", outputFormat="csv",
                 typeName="public:cwfif_national_reportedfires", propertyName=PROPS,
                 sortBy="status_date A,id A", CQL_FILTER=f"status_year={year}", count=page, startIndex=start)
        for attempt in range(4):
            r = requests.get(WFS, params=p, timeout=300)
            if r.status_code == 200 and b"national_fire_id" in r.content[:2000]:
                break
            time.sleep(4)
        else:
            raise RuntimeError(f"CWFIS failed at row {start}: {r.status_code}")
        df = pd.read_csv(io.BytesIO(r.content)); parts.append(df)
        if len(df) < page:
            break
        start += page
    return pd.concat(parts, ignore_index=True)


def first_reports(ts):
    ts = ts.copy()
    for c in ["situation_report_date", "status_date"]:
        ts[c] = pd.to_datetime(ts[c], errors="coerce")
    gr = ts.sort_values("status_date").groupby("national_fire_id")
    f = gr.agg(lat=("latitude", "first"), lon=("longitude", "first"),
               first_sitrep=("situation_report_date", "min")).reset_index()
    return f.dropna(subset=["lat", "lon", "first_sitrep"])


def count_recent(fires, lats, lons, cand_day, radius=50_000, lookback=30):
    from scipy.spatial import cKDTree
    from pyproj import Transformer
    tr = Transformer.from_crs(4326, 3978, always_xy=True)
    fx, fy = tr.transform(fires["lon"].values, fires["lat"].values)
    fd = fires["first_sitrep"].values.astype("datetime64[D]")
    tree = cKDTree(np.column_stack([fx, fy]))
    cx, cy = tr.transform(lons, lats)
    out = np.zeros(len(lats), dtype="int32")
    lo, hi = cand_day - np.timedelta64(lookback, "D"), cand_day
    for i, nb in enumerate(tree.query_ball_point(np.column_stack([cx, cy]), r=radius)):
        if nb:
            dd = fd[np.array(nb)]
            out[i] = ((dd >= lo) & (dd < hi)).sum()
    return out


fires_note = "live"
try:
    f = first_reports(cwfis_year(now_utc.year))
    # a fire counts if first reported in [today-30, today+1)  (same as step 29: "count through today")
    day = np.datetime64(now_utc.date()) + np.timedelta64(1, "D")
    d["recent_fires_50km_30d"] = count_recent(f, d["LAT"].values, d["LON"].values, day)
    d[["LAT", "LON", "recent_fires_50km_30d"]].to_csv(f"{CACHE}/live_recent_fires.csv", index=False)
    print("fires: live,", len(f), "fires this year, newest report", f["first_sitrep"].max())
except Exception as e:
    cp = f"{CACHE}/live_recent_fires.csv"
    if not os.path.exists(cp):
        raise
    print("WARNING: CWFIS failed (", e, ") - using the last saved fire counts")
    d = d.merge(key(pd.read_csv(cp)), on=["LAT", "LON"], how="left")
    fires_note = "OLD (CWFIS was unreachable)"

# ---------------------------------------------------------------- 3. NDVI (uploaded from Colab)
nd = key(pd.read_csv(f"{STATIC}/live_ndvi.csv"))
ndvi_start = str(nd["composite_start"].iloc[0])
d = d.merge(nd[["LAT", "LON", "ndvi"]], on=["LAT", "LON"], how="left")
print("ndvi filled:", round(d["ndvi"].notna().mean(), 3), "| composite", ndvi_start)

# ---------------------------------------------------------------- 4. lightning (recorded in data/)
steps = pd.concat([pd.read_csv(p) for p in glob.glob(f"{DATA}/steps_*.csv")])
steps["t"] = pd.to_datetime(steps["t"], utc=True)
steps = steps[steps["status"] == "ok"].drop_duplicates("t")
asof = steps["t"].max()
age_h = (asof - steps["t"]).dt.total_seconds() / 3600
steps["k"] = np.floor(age_h / 24).astype(int)
steps = steps[steps["k"] < NLAGS]
cov = steps.groupby("k").size().reindex(range(NLAGS), fill_value=0) / 144.0
history_days = int((cov >= 0.5).sum())
print("lightning steps:", len(steps), "| newest", asof, "| history days:", history_days, "of", NLAGS)
lag_h = (now_utc - asof.to_pydatetime()).total_seconds() / 3600
if lag_h > 8:
    raise RuntimeError(f"newest lightning step is {lag_h:.1f} h old - recorder problem, stopping")

fl = pd.concat([pd.read_csv(p) for p in glob.glob(f"{DATA}/flashes_*.csv")])
fl["t"] = pd.to_datetime(fl["t"], utc=True)
fl = fl[fl["t"] <= asof].copy()
fl["k"] = np.floor(((asof - fl["t"]).dt.total_seconds() / 3600) / 24).astype(int)
fl = fl[fl["k"] < NLAGS]
P = np.zeros((NLAT + 2, NLON + 2, NLAGS))
ii = np.round((fl["LAT"].values - LAT0) / STEP).astype(int); jj = np.round((fl["LON"].values - LON0) / STEP).astype(int)
okk = (ii >= 0) & (ii < NLAT) & (jj >= 0) & (jj < NLON)
np.add.at(P, (ii[okk] + 1, jj[okk] + 1, fl["k"].values[okk]), fl["pixels"].values[okk])
box = np.zeros((NLAT, NLON, NLAGS))
for dy in (-1, 0, 1):
    for dx in (-1, 0, 1):
        box += P[1 + dy:1 + dy + NLAT, 1 + dx:1 + dx + NLON, :]
gi = np.round((d["LAT"].values - LAT0) / STEP).astype(int); gj = np.round((d["LON"].values - LON0) / STEP).astype(int)
inb = (gi >= 0) & (gi < NLAT) & (gj >= 0) & (gj < NLON)
for k in range(NLAGS):
    v = np.full(len(d), np.nan)
    if cov.iloc[k] >= 0.5:
        v[inb] = box[gi[inb], gj[inb], k] * LIGHTNING_SCALE / cov.iloc[k]
    d[f"cg3_lag{k}"] = v
d["lightning_24h_pixels"] = np.where(inb, box[np.clip(gi, 0, NLAT - 1), np.clip(gj, 0, NLON - 1), 0], 0)

# ---------------------------------------------------------------- 5. score
doy = asof.dayofyear
d["doy_sin"] = np.sin(2 * np.pi * doy / 365.25); d["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
FEATS = pickle.load(open(f"{STATIC}/live_features_list.pkl", "rb"))
p7 = d["precip_prior_7d"]
for c in [f for f in FEATS if f.startswith("stormtype_")]:
    if "<2.5" in c:   d[c] = (p7 < 2.5).astype(int)
    elif ">10" in c:  d[c] = (p7 > 10).astype(int)
    else:             d[c] = ((p7 >= 2.5) & (p7 <= 10)).astype(int)
missing = [f for f in FEATS if f not in d.columns]
assert not missing, f"features missing: {missing}"
model = pickle.load(open(f"{STATIC}/live_model_no_cg1.pkl", "rb"))
plat = pickle.load(open(f"{STATIC}/live_calibrator_platt.pkl", "rb"))
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p)).reshape(-1, 1)
d["scored"] = d["lightning_24h_pixels"] > 0
d["prob"] = np.nan
if d["scored"].any():
    raw = model.predict_proba(d.loc[d["scored"], FEATS])[:, 1]
    d.loc[d["scored"], "prob"] = plat.predict_proba(logit(raw))[:, 1]
d["lightning_history_days"] = history_days
d["flag_north"] = d["region"].eq("North")
d["flag_short_history"] = history_days < NLAGS
d["flag_missing_inputs"] = d[["temp_on_day", "precip_on_day", "ndvi", "recent_fires_50km_30d", "wind_speed_ms"]].isna().any(axis=1)
d["asof_utc"] = asof.strftime("%Y-%m-%dT%H:%MZ")
s = d[d["scored"]].copy()
print("scored cells:", len(s), "| prob summary:", s["prob"].describe().round(4).to_dict() if len(s) else "none")

keep = ["LAT", "LON", "province", "region", "prob", "lightning_24h_pixels", "lightning_history_days",
        "flag_north", "flag_short_history", "flag_missing_inputs", "asof_utc"]
s[keep].to_csv(f"{DOCS}/latest_scores.csv", index=False)
s[keep + FEATS].to_csv(f"{ARCH}/scores_{asof:%Y%m%dT%H%M}Z.csv.gz", index=False, compression="gzip")   # for later testing

# ---------------------------------------------------------------- 6. map
import folium
BANDS = [(0.005, "#fee08b", "under 0.5%"), (0.01, "#fdae61", "0.5 - 1%"), (0.02, "#f46d43", "1 - 2%"),
         (0.05, "#d73027", "2 - 5%"), (0.10, "#a50026", "5 - 10%"), (9, "#4d0013", "over 10%")]
def color(p):
    for lim, c, _ in BANDS:
        if p < lim:
            return c
    return BANDS[-1][1]

TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"
if len(s):
    span = max((s.LAT.max() - s.LAT.min()) * 1.5, s.LON.max() - s.LON.min())
    zoom = 7 if span < 3 else 6 if span < 6 else 5 if span < 12 else 4 if span < 24 else 3
    center = [float(s.LAT.median()), float(s.LON.median())]
else:
    center, zoom = [55.0, -96.0], 4
m = folium.Map(location=center, zoom_start=zoom, tiles=None, min_zoom=3, max_bounds=True)
folium.TileLayer(TILES, attr="Tiles &copy; Esri", no_wrap=True).add_to(m)
for r in s.itertuples(index=False):
    tip = (f"<b>{r.prob*100:.2f}%</b> chance a fire follows<br>{r.province} | lat {r.LAT:.1f}, lon {r.LON:.1f}"
           f"<br>lightning pixels (24 h): {r.lightning_24h_pixels:.0f}"
           + ("<br><i>North of 52N: low confidence</i>" if r.flag_north else "")
           + ("<br><i>Some inputs missing</i>" if r.flag_missing_inputs else ""))
    folium.Rectangle([[r.LAT - .05, r.LON - .05], [r.LAT + .05, r.LON + .05]], color=color(r.prob), weight=1, fill=True,
                     fill_color=color(r.prob), fill_opacity=0.75, dash_array="3" if r.flag_north else None).add_to(m)
    folium.CircleMarker([r.LAT, r.LON], radius=6, color="black", weight=1, fill=True, fill_color=color(r.prob),
                        fill_opacity=0.9, tooltip=folium.Tooltip(tip)).add_to(m)
warn = ""
if history_days < NLAGS:
    warn += (f"<div style='color:#b00;font-weight:bold;margin-top:4px'>WARNING: short lightning history "
             f"({history_days} of 15 days). Probabilities are NOT reliable yet.</div>")
if fires_note != "live":
    warn += f"<div style='color:#b00;margin-top:4px'>Fire counts: {fires_note}</div>"
empty = "" if len(s) else "<div style='margin-top:6px'><b>No lightning detected in the last 24 h.</b></div>"
legend = "".join(f"<div><span style='background:{c};display:inline-block;width:14px;height:14px;margin-right:6px'></span>{t}</div>"
                 for _, c, t in BANDS)
panel = f"""
<div style="position:fixed;top:10px;left:55px;z-index:9999;background:white;padding:10px 14px;border:1px solid #999;
 border-radius:6px;max-width:340px;font:13px Arial">
<b>Lightning risk map</b><br>
<i>Conditional on detected lightning. This is not overall wildfire risk.</i><br>
Each dot = a 0.1&deg; cell with lightning in the last 24 h. Colour = chance that a lightning fire follows.
<div style="margin-top:6px">{legend}</div>
<div style="margin-top:6px;font-size:12px">Dashed box = north of 52&deg;N (low confidence). Covers Canada south of 66&deg;N.</div>
{empty}{warn}
<div style="margin-top:6px;font-size:11px;color:#555">Lightning to {asof:%Y-%m-%d %H:%M} UTC | weather {weather_asof}<br>
NDVI composite {ndvi_start} | page built {now_utc:%Y-%m-%d %H:%M} UTC</div>
</div>"""
m.get_root().html.add_child(folium.Element(panel))
m.save(f"{DOCS}/index.html")
print("map saved:", len(s), "cells")
