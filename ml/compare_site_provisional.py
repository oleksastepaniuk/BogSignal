"""Exploratory June/October NISAR-only sensitivity of frozen model D at site boxes."""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import transform_bounds, transform_geom
from PIL import Image

from bogsignal.inference.pipeline import model_input, probabilities
from bogsignal.inference.torch_adapter import load_predictor

ROOT = Path(__file__).resolve().parents[1]
ARGOS = ROOT.parent / "Argos"
DATASET = ROOT / "data/dataset/s1s2nisar_256_v4"
SERIES = ARGOS / "data/temporal_sources/provisional_v1"
OUTPUT = ROOT / "data/inference/site_provisional_june_oct_2026"


def main(export_only=False):
    if OUTPUT.exists() and not export_only:
        raise FileExistsError(OUTPUT)
    sites = json.loads((ARGOS / "data/targets/polesia_restoration_sites_bbox.geojson").read_text())["features"]
    jobs = json.loads((SERIES / "series_plan.json").read_text())["groups"]
    records = {json.loads(line)["id"]: json.loads(line) for line in (DATASET / "index.jsonl").read_text().splitlines()}
    specs = json.loads((DATASET / "normalization.json").read_text())
    lookup = {}
    for job in jobs:
        complete = json.loads((Path(job["folder"]) / "complete.json").read_text())
        by_month = {int(s["acquired_start"][5:7]): s for s in complete["sources"]}
        if 6 not in by_month or 10 not in by_month:
            continue
        for i, r in enumerate(job["records"]):
            lookup.setdefault((r["id"], job["key"][0]), []).append((i, by_month))
    predictor = load_predictor(ROOT / "data/experiments/D_s2s1nisar_resnet34_seed17/best.pt")
    if not export_only:
        OUTPUT.mkdir(parents=True)
    export_dir = ROOT / "web/public/seasonal"
    selected = {}
    exports = []
    if export_only:
        previous = json.loads((OUTPUT / "report.json").read_text())
        for site_id, result in previous["sites"].items():
            eligible = [r for r in result["patches"] if r.get("joint_valid_pixels", 0) >= 10000]
            chosen = sorted(eligible, key=lambda r: (-abs(r["october_wetland_pixels"] - r["june_wetland_pixels"]), r["id"]))[:2]
            if len(chosen) != 2:
                raise ValueError(f"Need two sufficiently covered patches for {site_id}")
            selected[site_id] = {r["id"] for r in chosen}
        export_dir.mkdir(parents=True, exist_ok=True)
    report = {"method": "Frozen validated-product model D; frozen v4 normalization and S2/S1; provisional June and October NISAR substituted; patch-only coverage", "sites": {}}
    for site in sites:
        rows = []
        transition = np.zeros((5, 5), dtype="int64")
        total_site, compared = 0, 0
        grid = jobs[0]["grid"]
        affine = rasterio.Affine(*grid["transform"])
        site_bounds = transform_bounds("EPSG:4326", grid["crs"], *site["bbox"])
        for r in records.values():
            if export_only and r["id"] not in selected[site["id"]]:
                continue
            patch_bounds = rasterio.windows.bounds(rasterio.windows.Window(r["col"], r["row"], 256, 256), affine)
            if patch_bounds[0] >= site_bounds[2] or patch_bounds[2] <= site_bounds[0] or patch_bounds[1] >= site_bounds[3] or patch_bounds[3] <= site_bounds[1]:
                continue
            with rasterio.open(DATASET / r["images"]) as ds:
                geom = transform_geom("EPSG:4326", ds.crs, site["geometry"])
                roi = geometry_mask([geom], out_shape=(256, 256), transform=ds.transform, invert=True)
                if not roi.any():
                    continue
                image = ds.read().astype("float32")
            total_site += int(roi.sum())
            if any((r["id"], direction) not in lookup for direction in ["ASCENDING", "DESCENDING"]):
                rows.append({"id": r["id"], "reason": "missing both-pass provisional series", "site_pixels": int(roi.sum())})
                continue
            with rasterio.open(DATASET / r["input_validity"]) as ds:
                base_valid = ds.read() == 1
            predictions = {}
            masks = {}
            dates = {}
            for month in [6, 10]:
                values = image.copy()
                valid = base_valid.copy()
                dates[month] = {}
                for d, direction in enumerate(["ASCENDING", "DESCENDING"]):
                    month_power = np.full((2, 256, 256), np.nan, dtype="float32")
                    dates[month][direction] = []
                    for i, by_month in lookup[(r["id"], direction)]:
                        source = by_month[month]
                        p = np.load(source["cache"], mmap_mode="r")[i]
                        good = np.isfinite(p).all(0) & (p > 0).all(0)
                        if np.isfinite(month_power[:, good]).any():
                            raise ValueError("Overlapping provisional source groups")
                        month_power[:, good] = p[:, good]
                        dates[month][direction].append(source["acquired_start"][:10])
                    good = np.isfinite(month_power).all(0) & (month_power > 0).all(0)
                    sl = slice(10 + 2*d, 12 + 2*d)
                    valid[sl] = good & base_valid[sl]
                    db = 10 * np.log10(np.maximum(month_power, 1e-12))
                    values[sl] = np.clip(db, np.array([-40, -45])[:, None, None], np.array([5, 0])[:, None, None])
                    values[sl] = np.where(valid[sl], values[sl], -9999)
                x = model_input(values, valid, specs)
                predictions[month] = probabilities(predictor(x[None]), 256, "logits")
                masks[month] = valid.all(0)
            common = roi & masks[6] & masks[10]
            if not common.any():
                rows.append({"id": r["id"], "reason": "no joint-valid site pixels", "site_pixels": int(roi.sum())})
                continue
            def classes(p):
                five = np.concatenate((p[:1], p[1:3].sum(0, keepdims=True), p[3:]), axis=0)
                return five.argmax(0)
            a, b = classes(predictions[6]), classes(predictions[10])
            if export_only:
                colors = np.array([[185,185,170], [25,123,73], [40,121,185], [207,154,75], [139,102,139], [255,255,255]], dtype="uint8")
                for name, array in [("june", a), ("october", b)]:
                    Image.fromarray(colors[np.where(common, array, 5)]).save(export_dir / f"{site['id']}-{r['id']}-{name}.png")
                change = np.full((256, 256, 3), [221,228,216], dtype="uint8")
                change[common & (a != 1) & (b == 1)] = [52,111,173]
                change[common & (a == 1) & (b != 1)] = [199,103,60]
                change[~common] = [255,255,255]
                Image.fromarray(change).save(export_dir / f"{site['id']}-{r['id']}-change.png")
                exports.append({"site": site["id"], "id": r["id"], "compared_pixels": int(common.sum()),
                                "june_wetland_pixels": int((a[common] == 1).sum()),
                                "october_wetland_pixels": int((b[common] == 1).sum()),
                                "source_dates": dates})
            counts = np.bincount(5*a[common] + b[common], minlength=25).reshape(5, 5)
            # Combined Swamp+Marsh has class index 1 in this five-class decision.
            compared += int(common.sum())
            transition += counts
            rows.append({"id": r["id"], "site_pixels": int(roi.sum()), "joint_valid_pixels": int(common.sum()),
                         "june_wetland_pixels": int((a[common] == 1).sum()), "october_wetland_pixels": int((b[common] == 1).sum()),
                         "changed_pixels": int((a[common] != b[common]).sum()), "source_dates": dates})
        report["sites"][site["id"]] = {"site_box_pixels_in_sampled_patches": total_site,
            "joint_valid_pixels": compared, "joint_valid_area_km2": compared * 0.0004,
            "june_wetland_pixels": int(transition[1].sum()), "october_wetland_pixels": int(transition[:, 1].sum()),
            "changed_pixels": int(transition.sum() - np.trace(transition)), "transition_counts": transition.tolist(), "patches": rows}
        print(site["id"], report["sites"][site["id"]]["joint_valid_area_km2"], flush=True)
    if export_only:
        if len(exports) != 6:
            raise ValueError(f"Expected six exported patches, got {len(exports)}")
        (export_dir / "manifest.json").write_text(json.dumps({"selection": "Per site, two sampled patches with at least 10,000 jointly valid site-box pixels and largest absolute June-to-October change in predicted Swamp/Marsh pixel count; illustrative, not representative", "examples": exports}, indent=2) + "\n")
    else:
        (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main(export_only="--export-selected" in sys.argv[1:])
