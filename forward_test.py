#!/usr/bin/env python3
"""
FORWARD TEST: did the saved live scores match the fires that really happened afterwards?

Reads   archive/scores_*.csv.gz  (saved by pipeline.py every 3 h: only cells that had lightning)
Fetches fires from CWFIS (this year and last year) and takes each fire's first report date + latest cause.
Rule (same as training): a scored cell-day D is a "fire" if a fire was first reported within 50 km
between day D and day D+3.  Only days at least 7 days old are judged, so fire causes have time to settle.
Writes  docs/forward_test.csv   (one summary row per group, overwritten each run)
        docs/forward_test_history.csv  (appended: one line per run, to see the trend)
IMPORTANT: out of fire season there are few or no lightning fires, so early results are mostly about
calibration and coverage, not skill. Real skill needs many fires (May-Sept).
"""
import glob, io, os, time
from datetime import datetime, timezone
import numpy as np, pandas as pd, requests

ROOT = os.path.dirname(os.path.abspath(__file__))
WFS = "https://geoserver.cwfif.nrcan.gc.ca/geoserver/wfs"
PROPS = "national_fire_id,latitude,longitude,situation_report_date,status_date,national_fire_cause"
RADIUS, FORWARD, SETTLE = 50_000, 3, 7


def cwfis_year(year, page=10000):
    parts, start = [], 0
    while True:
        p = dict(service="WFS", version="2.0.1", request="GetFeature", outputFormat="csv",
                 typeName="public:cwfif_national_reportedfires", propertyName=PROPS,
                 sortBy="status_date A,id A", CQL_FILTER=f"status_year={year}", count=page, startIndex=start)
        for _ in range(4):
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


def fires_table(ts):
    ts = ts.copy()
    for c in ["situation_report_date", "status_date"]:
        ts[c] = pd.to_datetime(ts[c], errors="coerce")
    ts = ts.sort_values("status_date")
    g = ts.groupby("national_fire_id")
    f = g.agg(lat=("latitude", "first"), lon=("longitude", "first"),
              fire_date=("situation_report_date", "min"), cause=("national_fire_cause", "last")).reset_index()
    f = f.dropna(subset=["lat", "lon", "fire_date"])
    f["fire_date"] = f["fire_date"].dt.normalize()
    return f


def _xy(lat, lon):
    from pyproj import Transformer
    return Transformer.from_crs(4326, 3978, always_xy=True).transform(np.asarray(lon), np.asarray(lat))


def positives(cells, fires):
    """cells: LAT, LON, date (Timestamp). True if a fire was first reported within 50 km in [date, date+3]."""
    from scipy.spatial import cKDTree
    if len(fires) == 0:
        return np.zeros(len(cells), dtype=bool)
    fx, fy = _xy(fires["lat"].values, fires["lon"].values)
    tree = cKDTree(np.column_stack([fx, fy]))
    fd = fires["fire_date"].values.astype("datetime64[D]")
    cx, cy = _xy(cells["LAT"].values, cells["LON"].values)
    cd = cells["date"].values.astype("datetime64[D]")
    out = np.zeros(len(cells), dtype=bool)
    for i, nb in enumerate(tree.query_ball_point(np.column_stack([cx, cy]), r=RADIUS)):
        if nb:
            d = fd[np.array(nb)]
            out[i] = ((d >= cd[i]) & (d <= cd[i] + np.timedelta64(FORWARD, "D"))).any()
    return out


def metrics(df, y, label):
    from sklearn.metrics import roc_auc_score, average_precision_score
    n, pos = len(df), int(y.sum())
    row = dict(group=label, cell_days=n, fire_cell_days=pos, fire_rate_pct=round(100 * pos / n, 3) if n else np.nan,
               mean_predicted_pct=round(100 * df["prob"].mean(), 3) if n else np.nan,
               roc_auc=np.nan, pr_auc=np.nan, top10pct_recall=np.nan)
    if pos >= 5 and n - pos >= 5:
        row["roc_auc"] = round(roc_auc_score(y, df["prob"]), 3)
        row["pr_auc"] = round(average_precision_score(y, df["prob"]), 4)
        k = max(1, int(round(0.1 * n)))
        top = np.argsort(-df["prob"].values)[:k]
        row["top10pct_recall"] = round(float(y[top].sum() / pos), 3)
    row["enough_fires_to_judge"] = pos >= 30
    return row


def evaluate(scores, fires, today):
    s = scores.copy()
    s["date"] = pd.to_datetime(s["asof_utc"].str[:10])
    s = s[s["date"] <= pd.Timestamp(today) - pd.Timedelta(days=SETTLE)]
    if s.empty:
        return pd.DataFrame(), {"note": "no scored days old enough yet"}
    s["LAT"] = s["LAT"].round(1); s["LON"] = s["LON"].round(1)
    cd = s.groupby(["LAT", "LON", "date"], as_index=False).agg(
        prob=("prob", "max"), north=("flag_north", "max"), short=("flag_short_history", "max"))
    rows = []
    for name, causes in [("natural only (N)", ["N"]), ("natural or unknown (N,U)", ["N", "U"])]:
        fs = fires[fires["cause"].isin(causes)]
        y = positives(cd, fs)
        rows.append(metrics(cd, y, f"{name} | all"))
        for lab, m in [("history >= 15 days", ~cd["short"]), ("short history", cd["short"]),
                       ("south of 52N-line cells flagged North", cd["north"])]:
            if m.sum():
                rows.append(metrics(cd[m], y[m.values], f"{name} | {lab}"))
    # coverage: how many natural fires had a scored cell within 50 km in the 3 days before they were reported?
    from scipy.spatial import cKDTree
    nat = fires[(fires["cause"] == "N") & (fires["fire_date"] >= cd["date"].min()) &
                (fires["fire_date"] <= pd.Timestamp(today) - pd.Timedelta(days=SETTLE))]
    cov = {"natural_fires_in_period": int(len(nat))}
    if len(nat):
        cx, cy = _xy(cd["LAT"].values, cd["LON"].values); tree = cKDTree(np.column_stack([cx, cy]))
        fx, fy = _xy(nat["lat"].values, nat["lon"].values)
        cdd = cd["date"].values.astype("datetime64[D]"); fdd = nat["fire_date"].values.astype("datetime64[D]")
        hit = 0
        for i, nb in enumerate(tree.query_ball_point(np.column_stack([fx, fy]), r=RADIUS)):
            if nb:
                d = cdd[np.array(nb)]
                hit += int(((d <= fdd[i]) & (d >= fdd[i] - np.timedelta64(FORWARD, "D"))).any())
        cov["natural_fires_with_a_scored_cell_nearby"] = hit
        cov["coverage_pct"] = round(100 * hit / len(nat), 1)
    return pd.DataFrame(rows), cov


def main():
    files = sorted(glob.glob(f"{ROOT}/archive/scores_*.csv.gz"))
    if not files:
        print("no archived scores yet"); return
    scores = pd.concat([pd.read_csv(f, usecols=["LAT", "LON", "prob", "asof_utc", "flag_north", "flag_short_history"])
                        for f in files])
    today = datetime.now(timezone.utc).date()
    ts = pd.concat([cwfis_year(today.year), cwfis_year(today.year - 1)], ignore_index=True)
    fires = fires_table(ts)
    res, cov = evaluate(scores, fires, today)
    os.makedirs(f"{ROOT}/docs", exist_ok=True)
    print("archived snapshots:", len(files), "| days judged only if >=", SETTLE, "days old")
    print("coverage:", cov)
    if res.empty:
        print("nothing to judge yet"); return
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    print(res.to_string(index=False))
    res.assign(run_date=str(today)).to_csv(f"{ROOT}/docs/forward_test.csv", index=False)
    hist = f"{ROOT}/docs/forward_test_history.csv"
    top = res.iloc[0].to_dict(); top.update(cov); top["run_date"] = str(today)
    pd.concat([pd.read_csv(hist) if os.path.exists(hist) else pd.DataFrame(), pd.DataFrame([top])]).to_csv(hist, index=False)
    if not res["enough_fires_to_judge"].any():
        print("\nNOTE: fewer than 30 fire cell-days. Not enough to judge skill. Check calibration (mean predicted vs rate) only.")


if __name__ == "__main__":
    main()
