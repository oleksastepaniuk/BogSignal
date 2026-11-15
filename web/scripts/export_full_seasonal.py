"""Publish display PNGs and dated provenance for three full seasonal patches."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/inference/full_seasonal_june_oct_2026_v2"
DEST = ROOT / "web/public/seasonal_full"
NAMES = {"cheremske": "Cheremske", "somyne": "Somyne", "syra_pohonia": "Syra Pohonia"}
VIEWS = {"prediction": "classes", "s2": "sentinel2_rgb", "s1": "sentinel1_ascending_vv",
         "nisar": "nisar_hh_hv_false_color", "hh": "nisar_ascending_hh", "hv": "nisar_ascending_hv"}


def used_dates(manifest, key):
    record = manifest[key]
    used = set(record["used_source_indices"])
    return sorted({datetime.fromtimestamp(source["time_ms"] / 1000, timezone.utc).date().isoformat()
                   for source in record["sources"] if source["index"] in used})


def main():
    report = json.loads((SOURCE / "report.json").read_text())
    DEST.mkdir(parents=True, exist_ok=True)
    examples = []
    for site, record in report["sites"].items():
        if site not in NAMES:
            raise ValueError(f"Unexpected site {site}")
        dates = {}
        files = {}
        for month in ["june", "october"]:
            folder = SOURCE / site / month
            manifest = json.loads((folder / "sentinel_manifest.json").read_text())
            dates[month] = {"s2": used_dates(manifest, "sentinel2"),
                            "s1_ascending": used_dates(manifest, "sentinel1_ascending"),
                            "s1_descending": used_dates(manifest, "sentinel1_descending"),
                            "nisar_ascending": sorted({s["acquired_start"][:10] for s in record["sources"][month]["nisar"] if s["pass"] == "ASCENDING"}),
                            "nisar_descending": sorted({s["acquired_start"][:10] for s in record["sources"][month]["nisar"] if s["pass"] == "DESCENDING"})}
            for view, source_name in VIEWS.items():
                source = SOURCE / site / (f"{month}_{source_name}.png" if view == "prediction" else f"{month}/{source_name}.png")
                if not source.exists():
                    raise FileNotFoundError(source)
                with Image.open(source) as image:
                    if image.size != (256, 256):
                        raise ValueError(f"Unexpected display size {source}")
                filename = f"{site}-{month}-{view}.png"
                shutil.copyfile(source, DEST / filename)
                files[f"{month}_{view}"] = {"filename": filename, "sha256": hashlib.sha256((DEST / filename).read_bytes()).hexdigest()}
        change = SOURCE / site / "wetland_change.png"
        with Image.open(change) as image:
            if image.size != (256, 256):
                raise ValueError(f"Unexpected change size {change}")
        filename = f"{site}-change.png"
        shutil.copyfile(change, DEST / filename)
        files["change"] = {"filename": filename, "sha256": hashlib.sha256((DEST / filename).read_bytes()).hexdigest()}
        examples.append({"site": site, "name": NAMES[site], "patch_id": record["patch_id"],
                         "patch_size_pixels": [256, 256], "pixel_size_m": 20,
                         "site_box_pixels_in_patch": record["site_box_pixels_in_patch"],
                         "full_patch": record["full_patch"], "within_site_box": record["within_site_box"],
                         "dates": dates, "files": files})
    if len(examples) != 3:
        raise ValueError("Expected three sites")
    data = {"selection": "One existing dataset-grid patch per site, ranked by number of pixels inside the saved site bounding box among patches with June/October provisional NISAR in both passes; selection uses geography and source coverage, not predictions",
            "comparison": "All six Sentinel-2, four Sentinel-1 and four provisional NISAR channels updated between June and October; frozen model D and v4 normalization",
            "caution": "Single autumn-centered training mosaic; June input is a seasonal distribution shift. Provisional NISAR. Model predictions do not establish wetland or restoration change.",
            "display": "S2 model-grid RGB (B4/B3/B2), fixed reflectance 0-0.3 and gamma 0.7; S1 ascending VV grayscale -30 to 0 dB; NISAR ascending HH grayscale -25 to 5 dB, HV -35 to 0 dB; false color R=HH, G=HV, B=HH-HV (0 to 25 dB), fixed stretches across months",
            "examples": examples}
    (DEST / "manifest.json").write_text(json.dumps(data, indent=2) + "\n")
    print("Exported", len(examples), "full seasonal site examples")


if __name__ == "__main__":
    main()
