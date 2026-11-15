"""Audit provisional temporal patch coverage of the three mapped site boxes."""
import json
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds

ROOT = Path(__file__).resolve().parents[1]
ARGOS = ROOT.parent / "Argos"
sites = json.loads((ARGOS / "data/targets/polesia_restoration_sites_bbox.geojson").read_text())["features"]
plan = json.loads((ARGOS / "data/temporal_sources/provisional_v1/series_plan.json").read_text())
grid = plan["groups"][0]["grid"]
affine = rasterio.Affine(*grid["transform"])
records = {}
for group in plan["groups"]:
    for record in group["records"]:
        records.setdefault(record["id"], []).append(group["key"][0])
index = [json.loads(line) for line in (ROOT / "data/temporal/provisional_v1/T0/index.jsonl").read_text().splitlines()]
for site in sites:
    bounds = transform_bounds("EPSG:4326", grid["crs"], *site["bbox"])
    hits = []
    for record in index:
        window = rasterio.windows.Window(record["col"], record["row"], 256, 256)
        left, bottom, right, top = rasterio.windows.bounds(window, affine)
        if left < bounds[2] and right > bounds[0] and bottom < bounds[3] and top > bounds[1]:
            hits.append(record)
    print(json.dumps({"site": site["id"], "patches": len(hits),
                      "split_counts": {s: sum(r["split"] == s for r in hits) for s in ["train", "validation", "test"]},
                      "ids": [r["id"] for r in hits]}))
