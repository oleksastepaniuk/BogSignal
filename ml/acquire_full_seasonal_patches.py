"""Acquire month-matched Sentinel chips for three full 20 m prediction patches."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import ee
import numpy as np
import rasterio
import requests
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/dataset/s1s2nisar_256_v4"
OUTPUT = ROOT / "data/inference/full_seasonal_june_oct_2026_v2"
SELECTED = {"cheremske": "r016896_c009984", "somyne": "r016640_c014848", "syra_pohonia": "r015360_c015616"}
MONTHS = {"june": ("2026-06-11", "2026-07-01", "2026-06-26"),
          "october": ("2026-10-01", "2026-10-28", "2026-10-12")}
S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
S1_BANDS = ["VV", "VH", "angle"]
NODATA_S2 = 65535
NODATA_S1 = -9999.0


def inventory(collection):
    def record(image):
        image = ee.Image(image)
        return ee.Dictionary({"id": image.get("system:index"), "time_ms": image.get("system:time_start")})
    return ee.List(collection.toList(collection.size()).map(record)).getInfo()


def ranked_sources(rows, target):
    target_ms = datetime.fromisoformat(target).replace(tzinfo=timezone.utc).timestamp() * 1000
    return sorted(rows, key=lambda r: (-abs(r["time_ms"] - target_ms), r["id"]))


def fetch(image, path, transform, band_count, dtype, nodata):
    if path.exists():
        with rasterio.open(path) as ds:
            assert ds.count == band_count and ds.width == ds.height == 256
            assert ds.crs.to_string() == "EPSG:3035" and ds.transform.almost_equals(transform)
            assert ds.dtypes[0] == dtype
        return "cached"
    url = image.getDownloadURL({"crs": "EPSG:3035", "crs_transform": list(transform)[:6],
                                "dimensions": [256, 256], "format": "GEO_TIFF"})
    response = requests.get(url, timeout=(30, 240))
    response.raise_for_status()
    temporary = path.with_suffix(".part.tif")
    temporary.write_bytes(response.content)
    with rasterio.open(temporary) as ds:
        assert ds.count == band_count and ds.width == ds.height == 256
        assert ds.crs.to_string() == "EPSG:3035" and ds.transform.almost_equals(transform)
        assert ds.dtypes[0] == dtype
        data = ds.read()
        assert np.isfinite(data).all()
    temporary.replace(path)
    return "downloaded"


def s2_chip(geometry, projection, start, end, target, folder, transform):
    collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterDate(start, end).filterBounds(geometry)
    rows = ranked_sources(inventory(collection), target)
    if not rows:
        raise ValueError("No Sentinel-2 scenes")
    codes = {r["id"]: i + 1 for i, r in enumerate(rows)}
    ranks = ee.Dictionary({r["id"]: i for i, r in enumerate(rows)})
    ids = ee.Dictionary(codes)
    cloud = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    collection = collection.filter(ee.Filter.inList("system:index", list(codes))).linkCollection(cloud, ["cs_cdf"])

    def prepare(image):
        reflectance = image.select(S2_BANDS).reproject(projection)
        scl = image.select("SCL").reproject(projection)
        score = image.select("cs_cdf").reproject(projection)
        valid = reflectance.mask().reduce(ee.Reducer.min())
        valid = valid.And(scl.remap([1, 3, 8, 9, 10, 11], [0]*6, 1)).And(score.gte(0.6))
        zero = reflectance.select(0).multiply(0)
        day = ee.Number(image.get("system:time_start")).divide(86400000).floor()
        index = ee.Number(ids.get(image.get("system:index")))
        result = reflectance.addBands(zero.add(day).rename("observation_day"))
        result = result.addBands(zero.add(index).rename("source_index"))
        return result.toUint16().updateMask(valid).set("priority", ranks.get(image.get("system:index")))

    background = ee.Image.constant([NODATA_S2]*8).rename(S2_BANDS + ["observation_day", "source_index"]).toUint16().set("priority", -1)
    image = ee.ImageCollection([background]).merge(collection.map(prepare)).sort("priority").mosaic().unmask(NODATA_S2, sameFootprint=False).toUint16()
    path = folder / "sentinel2.tif"
    status = fetch(image, path, transform, 8, "uint16", NODATA_S2)
    with rasterio.open(path) as ds:
        data = ds.read()
    valid = np.all(data[:6] != NODATA_S2, axis=0) & (data[7] != NODATA_S2)
    return {"collection": "COPERNICUS/S2_SR_HARMONIZED", "cloud_score": "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED", "cloud_score_threshold": 0.6,
            "excluded_SCL": [1, 3, 8, 9, 10, 11], "selection": "nearest clear pixel to target date within calendar month", "target": target,
            "sources": [{"index": codes[r["id"]], "id": r["id"], "time_ms": r["time_ms"]} for r in rows],
            "used_source_indices": np.unique(data[7, valid]).astype(int).tolist(), "valid_pixels": int(valid.sum()), "status": status,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def s1_chip(direction, geometry, projection, start, end, target, folder, transform):
    collection = ee.ImageCollection("COPERNICUS/S1_GRD_FLOAT").filterDate(start, end).filterBounds(geometry)
    collection = collection.filter(ee.Filter.eq("orbitProperties_pass", direction)).filter(ee.Filter.eq("instrumentMode", "IW"))
    collection = collection.filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    collection = collection.filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    rows = ranked_sources(inventory(collection), target)
    if not rows:
        raise ValueError(f"No Sentinel-1 {direction} scenes")
    codes = {r["id"]: i + 1 for i, r in enumerate(rows)}
    ranks = ee.Dictionary({r["id"]: i for i, r in enumerate(rows)})
    ids = ee.Dictionary(codes)
    collection = collection.filter(ee.Filter.inList("system:index", list(codes)))

    def prepare(image):
        bands = image.select(S1_BANDS).reproject(projection)
        power = bands.select(["VV", "VH"])
        angle = bands.select("angle")
        valid = bands.mask().reduce(ee.Reducer.min())
        valid = valid.And(power.reduce(ee.Reducer.min()).gt(0)).And(power.reduce(ee.Reducer.max()).lt(1e30))
        valid = valid.And(angle.gte(0)).And(angle.lte(90))
        zero = power.select(0).multiply(0)
        day = ee.Number(image.get("system:time_start")).divide(86400000).floor()
        index = ee.Number(ids.get(image.get("system:index")))
        result = bands.addBands(zero.add(day).rename("observation_day"))
        result = result.addBands(zero.add(index).rename("source_index"))
        return result.toFloat().updateMask(valid).set("priority", ranks.get(image.get("system:index")))

    background = ee.Image.constant([NODATA_S1]*5).rename(S1_BANDS + ["observation_day", "source_index"]).toFloat().set("priority", -1)
    image = ee.ImageCollection([background]).merge(collection.map(prepare)).sort("priority").mosaic().unmask(NODATA_S1, sameFootprint=False).toFloat()
    path = folder / f"sentinel1_{direction.lower()}.tif"
    status = fetch(image, path, transform, 5, "float32", NODATA_S1)
    with rasterio.open(path) as ds:
        data = ds.read()
    valid = np.all(data[:2] > 0, axis=0) & (data[4] > 0)
    return {"collection": "COPERNICUS/S1_GRD_FLOAT", "pass": direction, "selection": "nearest valid pixel to target date within calendar month", "target": target,
            "sources": [{"index": codes[r["id"]], "id": r["id"], "time_ms": r["time_ms"]} for r in rows],
            "used_source_indices": np.unique(data[4, valid]).astype(int).tolist(), "valid_pixels": int(valid.sum()), "status": status,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    ee.Initialize(project="pysheets-384513")
    ee.data.setDeadline(180000)
    records = {json.loads(line)["id"]: json.loads(line) for line in (DATASET / "index.jsonl").read_text().splitlines()}
    plan = json.loads((DATASET / "plan.json").read_text())
    grid = plan["grid"]
    transform = rasterio.Affine(*grid["transform"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for site, patch_id in SELECTED.items():
        r = records[patch_id]
        window = Window(r["col"], r["row"], 256, 256)
        patch_transform = rasterio.windows.transform(window, transform)
        bounds = rasterio.windows.bounds(window, transform)
        geometry = ee.Geometry.Rectangle(bounds, proj="EPSG:3035", geodesic=False)
        projection = ee.Projection("EPSG:3035", list(patch_transform)[:6])
        for month, (start, end, target) in MONTHS.items():
            folder = OUTPUT / site / month
            folder.mkdir(parents=True, exist_ok=True)
            manifest = {"site": site, "patch_id": patch_id, "grid_crs": "EPSG:3035", "transform": list(patch_transform)[:6],
                        "size": [256, 256], "date_window": [start, end], "target_date": target}
            manifest["sentinel2"] = s2_chip(geometry, projection, start, end, target, folder, patch_transform)
            for direction in ["ASCENDING", "DESCENDING"]:
                manifest[f"sentinel1_{direction.lower()}"] = s1_chip(direction, geometry, projection, start, end, target, folder, patch_transform)
            (folder / "sentinel_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(json.dumps({"site": site, "month": month, "S2_valid": manifest["sentinel2"]["valid_pixels"],
                              "S1A_valid": manifest["sentinel1_ascending"]["valid_pixels"],
                              "S1D_valid": manifest["sentinel1_descending"]["valid_pixels"]}), flush=True)


if __name__ == "__main__":
    main()
