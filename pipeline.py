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

LIGHTNING_SCALE = 0.35        # fallback only; normally replaced by percentile matching to CanCPLD ground counts
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
def load_ndvi(grid):
    """Newest MODIS 16-day composite via Earth Engine (service account in the EE_KEY secret).
    Only re-downloads when a NEWER composite exists; otherwise reuses cache/live_ndvi.csv.
    Falls back to the old file if Earth Engine can't be used."""
    cache = f"{CACHE}/live_ndvi.csv"
    try:
        import ee, json
        kjson = os.environ.get("EE_KEY", "").strip()
        if not kjson:
            raise RuntimeError("no EE_KEY secret set")
        info = json.loads(kjson)
        creds = ee.ServiceAccountCredentials(info["client_email"], key_data=kjson)
        ee.Initialize(creds, project=os.environ.get("EE_PROJECT") or info.get("project_id"))
        coll = ee.ImageCollection("MODIS/061/MOD13A2")
        starts = pd.DatetimeIndex(sorted(set(pd.Timestamp(ms, unit="ms")
                                             for ms in coll.aggregate_array("system:time_start").getInfo())))
        today = pd.Timestamp(now_utc.date())
        period = None
        for st in reversed([x for x in starts if x <= today]):
            n = coll.filterDate(st.strftime("%Y-%m-%d"), (st + pd.Timedelta(days=16)).strftime("%Y-%m-%d")).size().getInfo()
            if n > 0:
                period = st; break
        if period is None:
            raise RuntimeError("no MODIS composite found")
        ps = period.strftime("%Y-%m-%d")
        if os.path.exists(cache):
            c = pd.read_csv(cache)
            if str(c["composite_start"].iloc[0]) == ps and len(c) == len(grid):
                return key(c), "auto, cached"
        img = coll.filterDate(ps, (period + pd.Timedelta(days=16)).strftime("%Y-%m-%d")).mosaic().select(["NDVI"]).multiply(0.0001).unmask(-9999.0)
        # same method as training (step 35c): one map at the native 0.1 degree grid = mean NDVI of each cell square
        L1, LO0 = 66.0, -141.0
        GRIDX = {"dimensions": {"width": NLON, "height": NLAT},
                 "affineTransform": {"scaleX": 0.1, "shearX": 0, "translateX": LO0 - 0.05, "shearY": 0, "scaleY": -0.1, "translateY": L1 + 0.05},
                 "crsCode": "EPSG:4326"}
        arr = ee.data.computePixels({"expression": img, "fileFormat": "NUMPY_NDARRAY", "grid": GRIDX})["NDVI"].astype("float64")
        ri = np.clip(np.rint((L1 - grid["LAT"].values) / 0.1).astype(int), 0, NLAT - 1)
        rj = np.clip(np.rint((grid["LON"].values - LO0) / 0.1).astype(int), 0, NLON - 1)
        vals = arr[ri, rj]; vals[vals < -9990] = np.nan
        out = pd.DataFrame({"LAT": grid["LAT"].values, "LON": grid["LON"].values, "ndvi": vals, "composite_start": ps})
        if np.isfinite(vals).mean() < 0.9:
            raise RuntimeError(f"NDVI only {np.isfinite(vals).mean():.2f} filled")
        out.to_csv(cache, index=False)
        return key(out), "auto, new composite"
    except Exception as e:
        print("NDVI auto-refresh not used:", repr(e)[:300])
        for p in (cache, f"{STATIC}/live_ndvi.csv"):
            if os.path.exists(p):
                return key(pd.read_csv(p)), "monthly Colab file"
        raise


nd, ndvi_src = load_ndvi(d.reset_index(drop=True))
ndvi_start = str(nd["composite_start"].iloc[0]) + f" ({ndvi_src})"
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

# --- convert live lit-pixel sums into "ground-flash-like" counts (what the model was trained on) ---
# Preferred: match storm sizes percentile-to-percentile with CanCPLD ground counts of the same month
# (static/ground_ref_quantiles.csv, built from real data). Needs >= 10 covered live days and >= 5000 active
# cell-days; otherwise falls back to the fixed LIGHTNING_SCALE.
SCALE_MODE = f"fixed {LIGHTNING_SCALE}"
convert = lambda x: x * LIGHTNING_SCALE
REF = f"{STATIC}/ground_ref_quantiles.csv"
try:
    if os.path.exists(REF):
        ref = pd.read_csv(REF); ref = ref[ref["month"] == asof.month].sort_values("q")
        pool = [box[:, :, k][box[:, :, k] > 0] / cov.iloc[k] for k in range(NLAGS) if cov.iloc[k] >= 0.5]
        live_sorted = np.sort(np.concatenate(pool)) if pool else np.array([])
        if len(ref) >= 20 and len(pool) >= 10 and len(live_sorted) >= 5000:
            qg, gg = ref["q"].values, ref["ground3"].values
            def convert(x, live_sorted=live_sorted, qg=qg, gg=gg):
                x = np.asarray(x, dtype=float)
                p = np.searchsorted(live_sorted, x, side="right") / len(live_sorted)
                return np.where(x > 0, np.interp(p, qg, gg), 0.0)
            SCALE_MODE = f"matched to CanCPLD ground, month {asof.month}"
        else:
            print(f"matching not ready (days {len(pool)}, cell-days {len(live_sorted)}, ref rows {len(ref)}) -> fixed scale")
    else:
        print("no ground_ref_quantiles.csv -> fixed scale")
except Exception as e:
    print("matching failed, using fixed scale:", repr(e))
print("lightning scale mode:", SCALE_MODE)
for k in range(NLAGS):
    v = np.full(len(d), np.nan)
    if cov.iloc[k] >= 0.5:
        v[inb] = convert(box[gi[inb], gj[inb], k] / cov.iloc[k])
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
if os.path.exists(f"{STATIC}/live_model_v7.pkl") and os.path.exists(f"{STATIC}/live_calibrator_platt_v7.pkl"):
    MODEL_VERSION = "v7"        # step 36: all positive cases, Canada-only rows, weighted, physical rules
    model = pickle.load(open(f"{STATIC}/live_model_v7.pkl", "rb"))
    plat = pickle.load(open(f"{STATIC}/live_calibrator_platt_v7.pkl", "rb"))
elif os.path.exists(f"{STATIC}/live_model_ca.pkl") and os.path.exists(f"{STATIC}/live_calibrator_platt_ca.pkl"):
    MODEL_VERSION = "ca"        # step 34: Canada-only training rows, physical rules, no road rule, Canada-only calibration
    model = pickle.load(open(f"{STATIC}/live_model_ca.pkl", "rb"))
    plat = pickle.load(open(f"{STATIC}/live_calibrator_platt_ca.pkl", "rb"))
elif os.path.exists(f"{STATIC}/live_model_mono.pkl") and os.path.exists(f"{STATIC}/live_calibrator_platt_mono.pkl"):
    MODEL_VERSION = "mono"      # model with physical rules (step 30/31)
    model = pickle.load(open(f"{STATIC}/live_model_mono.pkl", "rb"))
    plat = pickle.load(open(f"{STATIC}/live_calibrator_platt_mono.pkl", "rb"))
else:
    MODEL_VERSION = "base"      # first model (no rules)
    model = pickle.load(open(f"{STATIC}/live_model_no_cg1.pkl", "rb"))
    plat = pickle.load(open(f"{STATIC}/live_calibrator_platt.pkl", "rb"))
print("model version:", MODEL_VERSION)
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
d["model_version"] = MODEL_VERSION
d["lightning_scale_mode"] = SCALE_MODE
s = d[d["scored"]].copy()
print("scored cells:", len(s), "| prob summary:", s["prob"].describe().round(4).to_dict() if len(s) else "none")

keep = ["LAT", "LON", "province", "region", "prob", "lightning_24h_pixels", "lightning_history_days",
        "flag_north", "flag_short_history", "flag_missing_inputs", "asof_utc", "model_version", "lightning_scale_mode"]
s[keep].to_csv(f"{DOCS}/latest_scores.csv", index=False)
s[keep + FEATS].to_csv(f"{ARCH}/scores_{asof:%Y%m%dT%H%M}Z.csv.gz", index=False, compression="gzip")   # for later testing

# ---------------------------------------------------------------- 5b. data health (shown on the map and saved as docs/health.json)
import json as _json
def _status(ok, warn=False):
    return "OK" if ok else ("CHECK" if warn else "PROBLEM")
H_ROWS = []
def _add(name, source, updated, filled, status, note=""):
    H_ROWS.append(dict(input=name, source=source, updated=str(updated), filled=("" if filled is None else f"{filled*100:.1f}%"), status=status, note=note))
_wx_cols = ["precip_on_day", "precip_prior_7d", "days_since_rain", "temp_on_day", "rel_humidity_pct", "wind_speed_ms"]
_rain_slice = wlog.get("rain_latest_slice", "?")
try:
    _rain_age = (now_utc - pd.Timestamp(_rain_slice).to_pydatetime()).total_seconds() / 3600
except Exception:
    _rain_age = None
_add("Rain (24 h + 7 days)", "Environment Canada RDPA 6-hour analysis", _rain_slice, float(d["precip_prior_7d"].notna().mean()),
     _status(d["precip_prior_7d"].notna().mean() >= 0.99 and (_rain_age is not None and _rain_age < 24), True), f"newest slice {_rain_age:.0f} h old" if _rain_age is not None else "")
_add("Temperature, humidity, wind", "Environment Canada HRDPS forecast (next 24 h)", weather_asof, float(d[["temp_on_day", "rel_humidity_pct", "wind_speed_ms"]].notna().all(axis=1).mean()),
     _status(d[["temp_on_day", "rel_humidity_pct", "wind_speed_ms"]].notna().all(axis=1).mean() >= 0.99, True), "forecast, not the daily mean used in training")
try:
    _nd_age = (pd.Timestamp(now_utc.date()) - pd.Timestamp(str(nd["composite_start"].iloc[0]))).days
except Exception:
    _nd_age = None
_add("Vegetation (NDVI)", "MODIS 16-day composite", str(nd["composite_start"].iloc[0]) if "composite_start" in nd else "?", float(d["ndvi"].notna().mean()),
     _status(d["ndvi"].notna().mean() >= 0.98 and _nd_age is not None and _nd_age <= 55, True), f"{_nd_age} days old" if _nd_age is not None else "age unknown")
_add("Recent fires (50 km, 30 days)", "CWFIS", now_utc.strftime("%Y-%m-%d %H:%MZ"), float(d["recent_fires_50km_30d"].notna().mean()),
     _status(fires_note == "live" and d["recent_fires_50km_30d"].notna().mean() >= 0.99, True), fires_note)
_add("Lightning (15-day memory)", "Recorded from Environment Canada (ground + cloud)", asof.strftime("%Y-%m-%d %H:%MZ"), history_days / NLAGS,
     _status(history_days >= NLAGS and lag_h <= 6, True), f"{history_days} of {NLAGS} days recorded; newest step {lag_h:.1f} h old; scale: {SCALE_MODE}")
_stat_cols = ["dist_to_road_m", "pop_within_10km", "pop_within_25km", "elevation", "slope_deg", "ruggedness_m"]
_add("Roads, population, terrain", "Static grid (NRN, Kontur, SRTM/Copernicus)", "static", float(d[_stat_cols].notna().all(axis=1).mean()),
     _status(d[_stat_cols].notna().all(axis=1).mean() >= 0.99, True))
_feat_ok = all(f in d.columns for f in FEATS) and len(FEATS) == 34
_s_probs = s["prob"].dropna()
_prob_ok = bool(len(_s_probs) == 0 or (_s_probs.between(0, 1).all() and _s_probs.max() < 0.95))
_scored_nan = [f for f in FEATS if len(s) and s[f].isna().all()]
_add("Model", f"{MODEL_VERSION}, {len(FEATS)} inputs", MODEL_VERSION, None,
     _status(_feat_ok and _prob_ok and not _scored_nan and MODEL_VERSION == "v7", True),
     "all 34 inputs present" if (_feat_ok and not _scored_nan) else f"inputs missing/empty: {_scored_nan or 'schema'}")
HEALTH_OVERALL = "OK" if all(r["status"] == "OK" for r in H_ROWS) else ("PROBLEM" if any(r["status"] == "PROBLEM" for r in H_ROWS) else "CHECK")
with open(f"{DOCS}/health.json", "w") as _f:
    _json.dump(dict(built_utc=now_utc.strftime("%Y-%m-%dT%H:%MZ"), overall=HEALTH_OVERALL, rows=H_ROWS), _f, indent=1)
print("data health:", HEALTH_OVERALL)
for r in H_ROWS: print("  ", r["status"], "|", r["input"], "|", r["filled"], "|", r["note"])
_hc = {"OK": "#1a7f37", "CHECK": "#b7791f", "PROBLEM": "#b00020"}
HEALTH_HTML = ("<details style='margin-top:6px'><summary style='cursor:pointer'><b>Data health: <span style='color:%s'>%s</span></b></summary>"
               "<table style='font-size:11px;border-collapse:collapse;margin-top:4px'>" % (_hc[HEALTH_OVERALL], HEALTH_OVERALL)
               + "".join("<tr><td style='padding:1px 4px;vertical-align:top'><span style='color:%s'><b>%s</b></span></td><td style='padding:1px 4px'>%s<br><span style='color:#555'>%s %s</span></td></tr>"
                         % (_hc[r["status"]], r["status"], r["input"], r["filled"], r["note"]) for r in H_ROWS)
               + "</table></details>")

# ---------------------------------------------------------------- 6. map
import folium
BANDS = [(0.005, "#ffffb2", "under 0.5% (real: about 0.2%)"), (0.009, "#fecc5c", "0.5 - 0.9% (real: about 0.8%)"),
         (0.016, "#fd8d3c", "0.9 - 1.6% (real: about 1.6%)"), (0.035, "#f03b20", "1.6 - 3.5% (real: about 2.7%)"),
         (0.08, "#bd0026", "3.5 - 8% (real: about 5.5%)"), (0.23, "#7a0177", "8 - 23% (real: about 11%)"), (9, "#240046", "over 23% (real: about 26%)")]
# "real" = measured fire rate inside each band on the 2023-24 test years (step 38). Bands 8%+ show ~1.25-1.35x too high, so real is lower.
# bands from static/risk_bands_v7.json (written by Colab step 39 from 21 years of out-of-sample scores); the list above is the fallback
try:
    import json as _json
    _rb = _json.load(open(f"{STATIC}/risk_bands_v7.json"))
    BANDS = [(float(b["upper"]), b["color"], b["label"]) for b in _rb["bands"]]
    assert len(BANDS) >= 5 and BANDS[-1][0] >= 1
    print("risk bands: from risk_bands_v7.json,", len(BANDS), "bands")
except Exception as _e:
    print("risk bands: built-in fallback (", type(_e).__name__, ")")
# colours that stay visible on the light-grey base map (fill, outline); by band index, whatever the bands file says
VIS = [("#ffe14d", "#a88b00"), ("#ffb000", "#8a5c00"), ("#ff6a00", "#8a3300"), ("#e0201b", "#6e0d0b"),
       ("#a00060", "#4d002d"), ("#5a1a8a", "#2a0a42"), ("#1a0033", "#000000")]
BANDS = [(u, VIS[min(i, len(VIS) - 1)][0], t) for i, (u, _, t) in enumerate(BANDS)]
def outline(p):
    for i, (lim, _, _t) in enumerate(BANDS):
        if p < lim:
            return VIS[min(i, len(VIS) - 1)][1]
    return VIS[-1][1]
def color(p):
    for lim, c, _ in BANDS:
        if p < lim:
            return c
    return BANDS[-1][1]

TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"
center, zoom = [55.0, -96.0], 4        # always start on Canada; zoom in with +
m = folium.Map(location=center, zoom_start=zoom, tiles=None, min_zoom=2, max_bounds=True,
               min_lat=38.0, max_lat=72.0, min_lon=-150.0, max_lon=-40.0)     # cannot pan far from Canada
m.fit_bounds([[41.5, -141.5], [66.5, -52.0]])                                 # Canada fills the screen
folium.TileLayer(TILES, attr="Tiles &copy; Esri", no_wrap=True).add_to(m)
def _f(v, fmt="{:.1f}"):
    try:
        return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)
    except Exception:
        return "n/a"
dots = folium.FeatureGroup(name="dots", show=False)                       # round dots, only shown when zoomed out (cells are tiny there)
for r in s.itertuples(index=False):
    tip = (f"<b>{r.prob*100:.2f}%</b> chance a lightning fire follows (next 3 days, in the 3x3 cells around)"
           f"<br>{r.province} | lat {r.LAT:.1f}, lon {r.LON:.1f}"
           f"<hr style='margin:3px 0'>"
           f"Lightning (last 24 h): {r.lightning_24h_pixels:.0f} lit pixels, about {_f(r.cg3_lag0)} ground-like flashes (3x3 cells)<br>"
           f"Rain: {_f(r.precip_on_day)} mm last 24 h, {_f(r.precip_prior_7d)} mm in the 7 days before; {_f(r.days_since_rain, '{:.0f}')} days since rain<br>"
           f"Temp {_f(r.temp_on_day)} &deg;C | humidity {_f(r.rel_humidity_pct, '{:.0f}')}% | wind {_f(r.wind_speed_ms)} m/s (forecast)<br>"
           f"NDVI {_f(r.ndvi, '{:.2f}')} (composite {str(nd['composite_start'].iloc[0])[:10]}) | fires within 50 km, 30 days: {_f(r.recent_fires_50km_30d, '{:.0f}')}<br>"
           f"Model {r.model_version} | inputs as of {r.asof_utc}"
           + ("<br><i>Far north: less certain (fewer fires to learn from)</i>" if r.flag_north else "")
           + ("<br><i>Some inputs missing</i>" if r.flag_missing_inputs else ""))
    folium.Rectangle([[r.LAT - .05, r.LON - .05], [r.LAT + .05, r.LON + .05]], color="#333333" if r.flag_north else outline(r.prob),
                     weight=1.5 if r.flag_north else 0.5, fill=True, fill_color=color(r.prob), fill_opacity=0.9,
                     dash_array="3" if r.flag_north else None, tooltip=folium.Tooltip(tip)).add_to(m)
    folium.CircleMarker([r.LAT, r.LON], radius=3.2, color=outline(r.prob), weight=0.7, fill=True, fill_color=color(r.prob), fill_opacity=0.95,
                        tooltip=folium.Tooltip(tip)).add_to(dots)
warn = ""
if history_days < NLAGS:
    warn += (f"<div style='color:#b00;font-weight:bold;margin-top:4px'>WARNING: short lightning history "
             f"({history_days} of 15 days). Probabilities are NOT reliable yet.</div>")
if fires_note != "live":
    warn += f"<div style='color:#b00;margin-top:4px'>Fire counts: {fires_note}</div>"
n_north_unscored = 0   # grid now reaches 66N; lightning north of 66N is outside the box
try:
    ndvi_age = (pd.Timestamp(now_utc.date()) - pd.Timestamp(str(nd["composite_start"].iloc[0]))).days
except Exception:
    ndvi_age = None
if ndvi_age is not None and ndvi_age > 55:
    warn += (f"<div style='color:#b00;font-weight:bold;margin-top:4px'>WARNING: vegetation (NDVI) data is {ndvi_age} days old. "
             f"Run the monthly refresh.</div>")
print("lit cells north of 52N not scored:", n_north_unscored, "| ndvi age (days):", ndvi_age)
empty = "" if len(s) else "<div style='margin-top:6px'><b>No lightning detected in the last 24 h.</b></div>"
legend = "".join(f"<div><span style='background:{c};display:inline-block;width:14px;height:14px;margin-right:6px;border:1px solid #555'></span>{t}</div>"
                 for _, c, t in BANDS)
short = "<div style='color:#b00;font-weight:bold;font-size:12px;margin-top:4px'>Short lightning history - probabilities not reliable yet</div>" if history_days < NLAGS else ""
panel = f"""
<div style="position:fixed;top:10px;left:55px;z-index:9999;background:white;padding:6px 10px;border:1px solid #999;
 border-radius:6px;max-width:300px;max-height:80vh;overflow:auto;font:13px Arial">
<details>
<summary style="cursor:pointer"><b>Lightning risk map</b> <span style="color:#555">(tap to open / close)</span></summary>
<i>Conditional on detected lightning. This is not overall wildfire risk.</i><br>
Each square = a 0.1&deg; cell (about 11 km) with lightning in the last 24 h. Colour = chance that a <b>reported lightning-caused fire</b> follows in the 3x3 cells around it within the next 3 days. The % is <b>conditional on lightning in that cell</b>: a reported lightning-caused fire anywhere in the 3x3 cells (about 33 km) within 3 days counts. It is <b>not</b> the chance that one strike starts a fire, not the chance of any wildfire, and not human-caused fires. About 1 in 4 lightning fires (more in the far north) have no ground lightning nearby beforehand, so this map cannot show them. Hover a square for its inputs.
<div style="margin-top:6px">{legend}</div>
<div style="margin-top:6px;font-size:12px">Covers Canadian land up to 66&deg;N. Lightning north of 66&deg;N and outside Canada is not scored. Dashed outline = Yukon/NT/Nunavut (less certain; the model was tested there with fewer fires). Tested 2023-24: in the North, Saskatchewan and Yukon the shown % runs about 1.4-1.9x too high; in October it runs too low. Treat those as rough.</div>
{empty}{warn}
{HEALTH_HTML}
<div style="margin-top:6px;font-size:11px;color:#555">Lightning to {asof:%Y-%m-%d %H:%M} UTC | weather {weather_asof}<br>
NDVI composite {ndvi_start} | page built {now_utc:%Y-%m-%d %H:%M} UTC</div>
</details>
{short}
</div>"""
dots.add_to(m)
_js = """<script>
window.addEventListener('load', function () {
  var mp = %s, dots = %s;
  function upd() { if (mp.getZoom() >= 7) { if (mp.hasLayer(dots)) mp.removeLayer(dots); } else { if (!mp.hasLayer(dots)) mp.addLayer(dots); } }
  mp.on('zoomend', upd); upd();
});
</script>""" % (m.get_name(), dots.get_name())
m.get_root().html.add_child(folium.Element(_js))
m.get_root().html.add_child(folium.Element(panel))
m.save(f"{DOCS}/index.html")
print("map saved:", len(s), "cells")
