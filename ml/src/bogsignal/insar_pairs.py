"""NISAR InSAR pair selection, milestones M1 (discovery) and M2 (hydrological context) of the E1 plan.

    python -m bogsignal.insar_pairs --out ../../runs/insar_e1

M1 queries the ASF archive (no login needed) for NISAR L-band GSLC scenes over the pilot sites, pairs scenes with
compatible geometry and records, for every pair, whether NASA's ready-made interferogram (GUNW) exists.
M2 adds proxy hydrological context from the ERA5 reanalysis (Open-Meteo archive API) and a deterministic ranking.
Neither milestone establishes that a pair carries useful phase; that is M3 (``bogsignal.insar_flooded``).

Outputs: scenes.parquet, pairs.parquet, rejected_pairs.parquet, pairs.geojson, selected_pairs.json.
Missing metadata never counts as matching: such pairs get status "needs_verification".
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from itertools import combinations
from pathlib import Path

import asf_search as asf
import pandas as pd
import requests
from shapely import wkt as shapely_wkt
from shapely.geometry import box, mapping, shape

# Pilot sites: bounding boxes (west, south, east, north) of the OpenStreetMap protected-area / reserve objects.
SITES = {
    "Cheremske": (25.512, 51.509, 25.596, 51.570),
    "Somyne": (26.835, 51.352, 27.018, 51.478),
    "Syra Pohonia": (27.116, 51.447, 27.307, 51.582),
}
TEMPORAL = [("preferred", 12), ("secondary", 24), ("exploratory", 36)]
MUST_MATCH = ["track", "direction", "frame", "range_bandwidth", "beam_mode", "collection"]  # collection separates provisional from validated products


def scene_row(p, site: str, aoi) -> dict:
    q = p.properties
    footprint = shape(p.geometry) if p.geometry else None
    start = q.get("startTime")
    pols = q.get("polarization") or q.get("mainBandPolarization")
    pols = pols if isinstance(pols, list) else ([pols] if pols else [])
    crid = q.get("crid")
    return {
        "site": site, "granule": q.get("sceneName") or q.get("fileID"), "product": q.get("processingLevel"),
        "start_utc": start, "track": q.get("pathNumber"), "frame": q.get("frameNumber"), "direction": q.get("flightDirection"),
        "radar_band": "L" if (q.get("sensor") or "").upper().startswith("L") or "LSAR" in (q.get("sceneName") or "") or "_L2_" in (q.get("sceneName") or "") else None,
        "frequency_group": "A", "polarizations": ",".join(sorted(str(x) for x in pols)) or None,
        "range_bandwidth": json.dumps(q.get("rangeBandwidth")) if q.get("rangeBandwidth") is not None else None,
        "beam_mode": q.get("beamModeType"), "crid": crid, "crid_family": crid[:1] if crid else None,  # P = provisional, V/A = validated ...
        "collection": q.get("collectionName"), "concept_id": q.get("conceptID"), "url": q.get("url"),
        "processing_date": q.get("processingDate"), "bytes": json.dumps(q.get("bytes")),
        "aoi_overlap_fraction": float(footprint.intersection(aoi).area / aoi.area) if footprint is not None else None,
        "footprint_wkt": footprint.wkt if footprint is not None else None,
    }


def discover(site: str, bounds, level: str) -> list[dict]:
    aoi = box(*bounds)
    results = asf.search(intersectsWith=aoi.wkt, dataset=asf.DATASET.NISAR, processingLevel=level, maxResults=2000)
    return [scene_row(p, site, aoi) for p in results if (p.properties.get("url") or "").endswith(".h5")]


def dedupe(scenes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One product per site/track/frame/acquisition: the most recently processed; the others are kept as provenance."""
    key = ["site", "product", "track", "frame", "direction", "start_utc"]
    ordered = scenes.sort_values("processing_date", na_position="first")
    keep = ordered.drop_duplicates(key, keep="last")
    return keep.sort_values(["site", "track", "frame", "start_utc"]).reset_index(drop=True), ordered.drop(keep.index)


def gunw_index(gunw: pd.DataFrame) -> dict:
    """(site, track, frame, reference date, secondary date) -> GUNW url, parsed from the product name"""
    index = {}
    for r in gunw.itertuples():
        d = re.findall(r"_(\d{8})T\d{6}", r.granule or "")
        if len(d) >= 4:
            index[(r.site, r.track, r.frame, d[0], d[2])] = r.url
    return index


def make_pairs(scenes: pd.DataFrame, gunw: dict, tolerance: int, min_overlap: float):
    accepted, rejected = [], []
    for (site, track, frame, direction), group in scenes.groupby(["site", "track", "frame", "direction"], dropna=False):
        rows = group.sort_values("start_utc").to_dict("records")
        for a, b in combinations(rows, 2):
            ta, tb = pd.Timestamp(a["start_utc"]), pd.Timestamp(b["start_utc"])
            days = (tb - ta).total_seconds() / 86400
            base = {"site": site, "track": track, "frame": frame, "direction": direction, "reference": a["granule"], "secondary": b["granule"],
                    "reference_utc": a["start_utc"], "secondary_utc": b["start_utc"], "temporal_separation_days": round(days, 2),
                    "reference_url": a["url"], "secondary_url": b["url"]}
            tier = next((name for name, target in TEMPORAL if abs(days - target) <= tolerance), None)
            if tier is None:
                if days <= TEMPORAL[-1][1] + tolerance:
                    rejected.append({**base, "reason": "temporal separation outside the 12/24/36-day tolerances"})
                continue
            reasons, unknown = [], []
            for f in MUST_MATCH:
                if a[f] is None or b[f] is None or pd.isna(a[f]) or pd.isna(b[f]):
                    unknown.append(f)
                elif a[f] != b[f]:
                    reasons.append(f"{f} differs ({a[f]} vs {b[f]})")
            for s in (a, b):
                if not s["polarizations"]:
                    unknown.append("polarizations")
                elif "HH" not in s["polarizations"]:
                    reasons.append("no HH polarization")
                if s["radar_band"] != "L":
                    unknown.append("radar_band")
            footprint = shapely_wkt.loads(a["footprint_wkt"]).intersection(shapely_wkt.loads(b["footprint_wkt"]))
            overlap = float(footprint.intersection(box(*SITES[site])).area / box(*SITES[site]).area)
            if overlap < min_overlap:
                reasons.append(f"common AOI coverage {overlap:.2f} below {min_overlap}")
            if reasons:
                rejected.append({**base, "reason": "; ".join(reasons)})
                continue
            month = ta.month
            ref_d, sec_d = ta.strftime("%Y%m%d"), tb.strftime("%Y%m%d")
            product = gunw.get((site, track, frame, ref_d, sec_d))
            # The archive metadata carries no acquisition-mode field. A NASA interferogram for exactly this pair shows
            # the two scenes were processed together, which settles mode compatibility; anything else stays unverified.
            by_product = bool(product) and set(unknown) <= {"beam_mode"}
            accepted.append({**base, "temporal_tier": tier, "common_aoi_coverage": round(overlap, 3),
                             "status": "compatible" if (not unknown or by_product) else "needs_verification",
                             "unverified_fields": None if by_product else (",".join(sorted(set(unknown))) or None),
                             "verified_by": "NASA GUNW interferogram exists for this pair" if (by_product and unknown) else None,
                             "grid_registration": "co-registered by NASA in the GUNW product" if product else "not checked (M3)",
                             "season_preference": month in (3, 4, 5, 9, 10, 11), "gunw_url": product, "common_footprint_wkt": footprint.wkt})
    return pd.DataFrame(accepted), pd.DataFrame(rejected)


def weather(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Daily ERA5 reanalysis at the site centre (proxy evidence only: ~9-25 km cells, not local water levels)."""
    daily = "precipitation_sum,snowfall_sum,temperature_2m_min,temperature_2m_mean"
    for model in ("era5_land", "era5", None):
        params = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end, "daily": daily, "timezone": "UTC"}
        if model:
            params["models"] = model
        r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=60)
        if r.ok:
            d = pd.DataFrame(r.json()["daily"])
            if d["precipitation_sum"].notna().mean() > 0.8:
                d["time"] = pd.to_datetime(d["time"])
                d.attrs["model"] = model or "best_match"
                return d.set_index("time")
    return pd.DataFrame()


def add_hydrology(pairs: pd.DataFrame) -> pd.DataFrame:
    out = []
    for site, group in pairs.groupby("site"):
        w, s, e, n = SITES[site]
        t0 = pd.to_datetime(group["reference_utc"]).min().tz_localize(None) - pd.Timedelta(days=16)
        t1 = min(pd.to_datetime(group["secondary_utc"]).max().tz_localize(None) + pd.Timedelta(days=1), pd.Timestamp(dt.date.today()) - pd.Timedelta(days=6))
        wx = weather((s + n) / 2, (w + e) / 2, t0.strftime("%Y-%m-%d"), t1.strftime("%Y-%m-%d"))
        for row in group.to_dict("records"):
            a = pd.Timestamp(row["reference_utc"]).tz_localize(None).normalize()
            b = pd.Timestamp(row["secondary_utc"]).tz_localize(None).normalize()
            h = {"hydrology_source": None, "precip_between_mm": None, "antecedent7_ref_mm": None, "antecedent14_ref_mm": None,
                 "antecedent7_sec_mm": None, "antecedent14_sec_mm": None, "min_temp_ref_c": None, "min_temp_sec_c": None,
                 "snowfall_window_cm": None, "gauge_stage_difference": None}
            if len(wx) and a in wx.index and b in wx.index:
                p = wx["precipitation_sum"]
                ante = lambda d, k: float(p[d - pd.Timedelta(days=k): d - pd.Timedelta(days=1)].sum())  # noqa: E731
                h.update(hydrology_source=f"Open-Meteo archive, {wx.attrs.get('model')}", precip_between_mm=float(p[a + pd.Timedelta(days=1): b].sum()),
                         antecedent7_ref_mm=ante(a, 7), antecedent14_ref_mm=ante(a, 14), antecedent7_sec_mm=ante(b, 7), antecedent14_sec_mm=ante(b, 14),
                         min_temp_ref_c=float(wx["temperature_2m_min"][a - pd.Timedelta(days=1): a].min()),
                         min_temp_sec_c=float(wx["temperature_2m_min"][b - pd.Timedelta(days=1): b].min()),
                         snowfall_window_cm=float(wx["snowfall_sum"][a - pd.Timedelta(days=3): b].sum()))
            known = h["precip_between_mm"] is not None
            # deterministic proxy rules; thresholds are provisional and documented here
            wet_both = known and h["antecedent14_ref_mm"] >= 15 and h["antecedent14_sec_mm"] >= 15
            change = known and (h["precip_between_mm"] >= 20 or abs(h["antecedent14_ref_mm"] - h["antecedent14_sec_mm"]) >= 20)
            frost = known and (min(h["min_temp_ref_c"], h["min_temp_sec_c"]) < 0 or h["snowfall_window_cm"] > 0)
            h.update(hydrology_evidence="proxy" if known else "unknown", proxy_wet_at_both_dates=bool(wet_both) if known else None,
                     proxy_water_change=bool(change) if known else None, frost_or_snow_flag=bool(frost) if known else None)
            out.append({**row, **h})
    df = pd.DataFrame(out)
    tier = df["temporal_tier"].map({"preferred": 3, "secondary": 2, "exploratory": 1})
    hydro = df["proxy_wet_at_both_dates"].fillna(False).astype(int) + df["proxy_water_change"].fillna(False).astype(int)
    df["rank_key"] = list(zip(-(df["status"] == "compatible").astype(int), -tier, df["frost_or_snow_flag"].fillna(False).astype(int), -hydro,
                              -df["season_preference"].astype(int), -df["gunw_url"].notna().astype(int)))
    df = df.sort_values(["site", "rank_key", "reference_utc"]).reset_index(drop=True)
    df["rank_in_site"] = df.groupby("site").cumcount() + 1
    return df.drop(columns="rank_key")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tolerance-days", type=int, default=2)
    ap.add_argument("--min-overlap", type=float, default=0.95, help="smallest share of the site covered by both scenes")
    ap.add_argument("--no-hydrology", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    gslc = pd.DataFrame([r for site, b in SITES.items() for r in discover(site, b, "GSLC")])
    gunw = pd.DataFrame([r for site, b in SITES.items() for r in discover(site, b, "GUNW")])
    scenes, superseded = dedupe(gslc)
    scenes.to_parquet(args.out / "scenes.parquet")
    superseded.to_parquet(args.out / "superseded_scenes.parquet")
    gunw.to_parquet(args.out / "gunw_products.parquet")
    pairs, rejected = make_pairs(scenes, gunw_index(gunw), args.tolerance_days, args.min_overlap)
    if not args.no_hydrology and len(pairs):
        pairs = add_hydrology(pairs)
    pairs.to_parquet(args.out / "pairs.parquet")
    rejected.to_parquet(args.out / "rejected_pairs.parquet")
    features = [{"type": "Feature", "geometry": mapping(shapely_wkt.loads(r["common_footprint_wkt"])),
                 "properties": {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in r.items() if k != "common_footprint_wkt"}}
                for r in pairs.to_dict("records")]
    (args.out / "pairs.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}, default=str))
    selected = pairs[(pairs["status"] == "compatible") & (pairs["temporal_tier"] == "preferred") & pairs["gunw_url"].notna()] if len(pairs) else pairs
    manifest = {"generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "sites": SITES,
                "rules": {"temporal": TEMPORAL, "tolerance_days": args.tolerance_days, "must_match": MUST_MATCH, "min_common_aoi_coverage": args.min_overlap,
                          "selection": "compatible, 12-day, with an existing GUNW interferogram; ordered by rank_in_site"},
                "pairs": selected.drop(columns=["common_footprint_wkt"]).to_dict("records")}
    (args.out / "selected_pairs.json").write_text(json.dumps(manifest, indent=1, default=str))

    print(f"GSLC scenes {len(gslc)} ({len(superseded)} superseded), GUNW products {len(gunw)}, pairs {len(pairs)}, rejected {len(rejected)}")
    for site in SITES:
        s, p = scenes[scenes["site"] == site], pairs[pairs["site"] == site] if len(pairs) else pairs
        print(f"\n{site}: {len(s)} scenes, {s['start_utc'].min()[:10] if len(s) else '-'} to {s['start_utc'].max()[:10] if len(s) else '-'}")
        for (track, frame, direction), g in s.groupby(["track", "frame", "direction"]):
            q = p[(p["track"] == track) & (p["frame"] == frame)] if len(p) else p
            print(f"  track {track} frame {frame} {direction}: {len(g)} scenes, site coverage {g['aoi_overlap_fraction'].min():.2f}-{g['aoi_overlap_fraction'].max():.2f}, "
                  f"pairs {dict(q['temporal_tier'].value_counts()) if len(q) else {}}, status {dict(q['status'].value_counts()) if len(q) else {}}, with GUNW {int(q['gunw_url'].notna().sum()) if len(q) else 0}")
    if len(rejected):
        print("\nrejection reasons:", dict(rejected["reason"].str.replace(r"\(.*?\)", "", regex=True).value_counts().head(8)))


if __name__ == "__main__":
    main()
