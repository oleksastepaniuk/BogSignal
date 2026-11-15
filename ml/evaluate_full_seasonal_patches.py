"""Run frozen model D on full June/October S2, S1 and provisional NISAR chips."""
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom

from bogsignal.inference.pipeline import model_input, probabilities
from bogsignal.inference.torch_adapter import load_predictor
from acquire_full_seasonal_patches import DATASET, OUTPUT, SELECTED

ROOT = Path(__file__).resolve().parents[1]
ARGOS = ROOT.parent / "Argos"
SERIES = ARGOS / "data/temporal_sources/provisional_v1"
COLORS = np.array([[185, 185, 170], [25, 123, 73], [40, 121, 185], [207, 154, 75], [139, 102, 139], [255, 255, 255]], dtype="uint8")
CHANNELS = ["S2_B2", "S2_B3", "S2_B4", "S2_B8", "S2_B11", "S2_B12",
            "S1_ASCENDING_VV", "S1_ASCENDING_VH", "S1_DESCENDING_VV", "S1_DESCENDING_VH",
            "NISAR_ASCENDING_HH", "NISAR_ASCENDING_HV", "NISAR_DESCENDING_HH", "NISAR_DESCENDING_HV"]


def checked_raster(path, digest):
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Source changed: {path}")
    with rasterio.open(path) as ds:
        return ds.read(), ds.profile


def save_raster(path, data, profile, nodata, descriptions):
    data = np.asarray(data)
    settings = profile.copy()
    settings.update(driver="GTiff", count=data.shape[0], dtype=str(data.dtype), nodata=nodata,
                    compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(path, "w", **settings) as ds:
        ds.write(data)
        ds.descriptions = tuple(descriptions)


def source_lookup():
    jobs = json.loads((SERIES / "series_plan.json").read_text())["groups"]
    lookup = {}
    for job in jobs:
        completed = json.loads((Path(job["folder"]) / "complete.json").read_text())
        sources = {int(s["acquired_start"][5:7]): s for s in completed["sources"]}
        for i, record in enumerate(job["records"]):
            if record["id"] in SELECTED.values():
                lookup.setdefault((record["id"], job["key"][0]), []).append((i, sources))
    return lookup


def provisional_nisar(patch_id, month, lookup):
    power = np.full((4, 256, 256), np.nan, dtype="float32")
    sources = []
    for d, direction in enumerate(["ASCENDING", "DESCENDING"]):
        if (patch_id, direction) not in lookup:
            raise ValueError(f"Missing provisional {direction} source for {patch_id}")
        for index, months in lookup[(patch_id, direction)]:
            source = months[month]
            if hashlib.sha256(Path(source["cache"]).read_bytes()).hexdigest() != source["cache_sha256"]:
                raise ValueError(f"NISAR cache changed: {source['id']}")
            chip = np.load(source["cache"], mmap_mode="r")[index]
            valid = np.isfinite(chip).all(0) & (chip > 0).all(0)
            sl = slice(2*d, 2*d+2)
            if np.isfinite(power[sl, valid]).any():
                raise ValueError("Overlapping NISAR anchor groups")
            power[sl, valid] = chip[:, valid]
            sources.append({"pass": direction, "id": source["id"], "acquired_start": source["acquired_start"],
                            "cache_sha256": source["cache_sha256"]})
    return power, sources


def display_gray(values, valid, low, high):
    scaled = (np.clip((values-low)/(high-low), 0, 1)*255).astype("uint8")
    rgb = np.repeat(scaled[:, :, None], 3, axis=2)
    rgb[~valid] = [255, 255, 255]
    return rgb


def prepare(site, patch_id, month_name, month_number, lookup, stats, site_geometry):
    folder = OUTPUT / site / month_name
    manifest = json.loads((folder / "sentinel_manifest.json").read_text())
    s2, profile = checked_raster(folder / "sentinel2.tif", manifest["sentinel2"]["sha256"])
    s1a, s1_profile = checked_raster(folder / "sentinel1_ascending.tif", manifest["sentinel1_ascending"]["sha256"])
    s1d, s1d_profile = checked_raster(folder / "sentinel1_descending.tif", manifest["sentinel1_descending"]["sha256"])
    if not (profile["transform"] == s1_profile["transform"] == s1d_profile["transform"]):
        raise ValueError("Sentinel grids differ")
    power, nisar_sources = provisional_nisar(patch_id, month_number, lookup)
    values = np.full((14, 256, 256), -9999.0, dtype="float32")
    valid = np.zeros((14, 256, 256), bool)
    s2_valid = (s2[7] > 0) & (s2[7] < 65535) & np.all(s2[:6] != 65535, axis=0)
    values[:6] = s2[:6].astype("float32") * 0.0001
    valid[:6] = s2_valid
    for d, source in enumerate([s1a, s1d]):
        for p in range(2):
            band = 6 + 2*d + p
            ok = (source[p] > 0) & np.isfinite(source[p]) & (source[4] > 0)
            valid[band] = ok
            db = 10 * np.log10(np.maximum(source[p], 1e-12))
            bounds = (-40, 5) if p == 0 else (-45, 0)
            values[band] = np.where(ok, np.clip(db, *bounds), -9999)
    for p in range(4):
        band = 10 + p
        ok = np.isfinite(power[p]) & (power[p] > 0)
        valid[band] = ok
        bounds = (-40, 5) if p % 2 == 0 else (-45, 0)
        values[band] = np.where(ok, np.clip(10*np.log10(np.maximum(power[p], 1e-12)), *bounds), -9999)
    values[~valid] = -9999
    if stats["channels"] != CHANNELS:
        raise ValueError("Frozen model channel order differs")
    save_raster(folder / "model_inputs.tif", values, profile, -9999.0, CHANNELS)
    save_raster(folder / "input_validity.tif", valid.astype("uint8"), profile, 255, CHANNELS)
    save_raster(folder / "nisar_linear_power.tif", np.where(np.isfinite(power), power, -9999), profile, -9999.0,
                CHANNELS[10:])
    keep = valid.all(0)
    rgb = (np.clip(values[[2, 1, 0]].transpose(1, 2, 0)/0.3, 0, 1)**0.7*255).astype("uint8")
    rgb[~s2_valid] = [255, 255, 255]
    Image.fromarray(rgb).save(folder / "sentinel2_rgb.png")
    Image.fromarray(display_gray(values[6], valid[6], -30, 0)).save(folder / "sentinel1_ascending_vv.png")
    Image.fromarray(display_gray(values[10], valid[10], -25, 5)).save(folder / "nisar_ascending_hh.png")
    Image.fromarray(display_gray(values[11], valid[11], -35, 0)).save(folder / "nisar_ascending_hv.png")
    composite = np.empty((256, 256, 3), dtype="uint8")
    composite[:, :, 0] = display_gray(values[10], valid[10], -25, 5)[:, :, 0]
    composite[:, :, 1] = display_gray(values[11], valid[11], -35, 0)[:, :, 0]
    composite[:, :, 2] = display_gray(values[10]-values[11], valid[10] & valid[11], 0, 25)[:, :, 0]
    composite[~(valid[10] & valid[11])] = [255, 255, 255]
    Image.fromarray(composite).save(folder / "nisar_hh_hv_false_color.png")
    return values, valid, keep, profile, manifest, nisar_sources


def summarize(a, b, mask):
    wet_a, wet_b = a == 1, b == 1
    return {"pixels": int(mask.sum()), "area_km2": round(float(mask.sum())*0.0004, 4),
            "june_wetland_pixels": int((wet_a & mask).sum()), "october_wetland_pixels": int((wet_b & mask).sum()),
            "gain_pixels": int((~wet_a & wet_b & mask).sum()), "loss_pixels": int((wet_a & ~wet_b & mask).sum()),
            "changed_class_pixels": int(((a != b) & mask).sum())}


def main():
    stats = json.loads((DATASET / "normalization.json").read_text())
    sites = {s["id"]: s for s in json.loads((ARGOS / "data/targets/polesia_restoration_sites_bbox.geojson").read_text())["features"]}
    lookup = source_lookup()
    predictor = load_predictor(ROOT / "data/experiments/D_s2s1nisar_resnet34_seed17/best.pt")
    results = {"method": "Full 256x256 patches; June and October S2, S1 and provisional NISAR all updated; same frozen validated-product model D and v4 normalization", "sites": {}}
    for site, patch_id in SELECTED.items():
        by_month = {}
        source_info = {}
        for name, month in [("june", 6), ("october", 10)]:
            values, valid, keep, profile, manifest, nisar_sources = prepare(site, patch_id, name, month, lookup, stats, sites[site]["geometry"])
            p = probabilities(predictor(model_input(values, valid, stats)[None]), 256, "logits")
            combined = np.concatenate((p[:1], p[1:3].sum(0, keepdims=True), p[3:]), axis=0)
            classes = combined.argmax(0).astype("uint8")
            save_raster(OUTPUT / site / name / "probabilities.tif", np.where(keep[None], p, -9999).astype("float32"), profile, -9999.0,
                        ["Other", "Swamp", "Marsh", "Permanent water", "Peatland cropland", "Peatland non-cropland"])
            save_raster(OUTPUT / site / name / "classes.tif", np.where(keep, classes, 255)[None].astype("uint8"), profile, 255,
                        ["Other", "Swamp or Marsh", "Permanent water", "Peatland cropland", "Peatland non-cropland"][:1])
            by_month[name] = (classes, keep, profile)
            source_info[name] = {"sentinel": manifest, "nisar": nisar_sources, "valid_pixels": int(keep.sum())}
        june, june_valid, profile = by_month["june"]
        october, oct_valid, _ = by_month["october"]
        common = june_valid & oct_valid
        geom = transform_geom("EPSG:4326", profile["crs"], sites[site]["geometry"])
        site_mask = geometry_mask([geom], out_shape=(256, 256), transform=profile["transform"], invert=True)
        save_raster(OUTPUT / site / "joint_valid.tif", common[None].astype("uint8"), profile, 255, ["Joint valid"])
        for name, classes in [("june", june), ("october", october)]:
            Image.fromarray(COLORS[np.where(common, classes, 5)]).save(OUTPUT / site / f"{name}_classes.png")
        change = np.full((256, 256, 3), [221, 228, 216], dtype="uint8")
        change[common & (june != 1) & (october == 1)] = [52, 111, 173]
        change[common & (june == 1) & (october != 1)] = [199, 103, 60]
        change[~common] = [255, 255, 255]
        Image.fromarray(change).save(OUTPUT / site / "wetland_change.png")
        results["sites"][site] = {"patch_id": patch_id, "site_box_pixels_in_patch": int(site_mask.sum()),
                                   "full_patch": summarize(june, october, common), "within_site_box": summarize(june, october, common & site_mask),
                                   "sources": source_info}
        print(json.dumps({"site": site, "patch": patch_id, "full": results["sites"][site]["full_patch"],
                          "site_box": results["sites"][site]["within_site_box"]}), flush=True)
    (OUTPUT / "report.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
