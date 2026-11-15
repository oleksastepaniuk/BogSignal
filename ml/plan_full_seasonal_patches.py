"""Rank 256-pixel patches by mapped site overlap and provisional NISAR coverage."""
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import transform_bounds, transform_geom

ROOT = Path(__file__).resolve().parents[1]
ARGOS = ROOT.parent / "Argos"
DATASET = ROOT / "data/dataset/s1s2nisar_256_v4"
SERIES = ARGOS / "data/temporal_sources/provisional_v1"
sites = json.loads((ARGOS / "data/targets/polesia_restoration_sites_bbox.geojson").read_text())["features"]
records = [json.loads(line) for line in (DATASET / "index.jsonl").read_text().splitlines()]
jobs = json.loads((SERIES / "series_plan.json").read_text())["groups"]
lookup = {}
for job in jobs:
    sources = json.loads((Path(job["folder"]) / "complete.json").read_text())["sources"]
    by_month = {int(s["acquired_start"][5:7]): s for s in sources}
    if 6 in by_month and 10 in by_month:
        for i, r in enumerate(job["records"]):
            lookup.setdefault((r["id"], job["key"][0]), []).append((i, by_month))

for site in sites:
    candidates = []
    grid = jobs[0]["grid"]
    affine = rasterio.Affine(*grid["transform"])
    site_bounds = transform_bounds("EPSG:4326", grid["crs"], *site["bbox"])
    for r in records:
        if any((r["id"], d) not in lookup for d in ["ASCENDING", "DESCENDING"]):
            continue
        patch_bounds = rasterio.windows.bounds(rasterio.windows.Window(r["col"], r["row"], 256, 256), affine)
        if patch_bounds[0] >= site_bounds[2] or patch_bounds[2] <= site_bounds[0] or patch_bounds[1] >= site_bounds[3] or patch_bounds[3] <= site_bounds[1]:
            continue
        with rasterio.open(DATASET / r["images"]) as ds:
            geom = transform_geom("EPSG:4326", ds.crs, site["geometry"])
            overlap = geometry_mask([geom], out_shape=(256, 256), transform=ds.transform, invert=True)
        if overlap.any():
            candidates.append((int(overlap.sum()), r))
    candidates.sort(key=lambda item: (-item[0], item[1]["id"]))
    top = []
    for overlap, r in candidates[:8]:
        counts = {}
        for month in [6, 10]:
            valid = np.ones((256, 256), bool)
            for direction in ["ASCENDING", "DESCENDING"]:
                pass_valid = np.zeros((256, 256), bool)
                for i, by_month in lookup[(r["id"], direction)]:
                    p = np.load(by_month[month]["cache"], mmap_mode="r")[i]
                    pass_valid |= np.isfinite(p).all(0) & (p > 0).all(0)
                valid &= pass_valid
            counts[str(month)] = int(valid.sum())
        top.append({"id": r["id"], "site_pixels": overlap, "full_patch_pixels": 65536, "nisar_joint_valid": counts})
    print(json.dumps({"site": site["id"], "top": top}))
