"""Build a matched, same-track NISAR repeat-observation pilot from frozen v4."""
import argparse
from collections import defaultdict, Counter
from contextlib import ExitStack
import json
from pathlib import Path

import h5py
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling

from .preprocessing import IDENTITY, text_scalar, validate_nisar_identity
from .pipeline import sha, save
from .sampling import mixture_weights

PASSES = ["ASCENDING", "DESCENDING"]


def repeat_features(power):
    """Mean in linear power; spread/quantiles from clipped per-date dB.

    Power shape is dates x 2 polarizations x pixels, with complete positive data.
    """
    power = np.asarray(power, dtype="float64")
    if power.ndim != 3 or power.shape[0] < 2 or power.shape[1] != 2 or not np.isfinite(power).all() or (power <= 0).any():
        raise ValueError("Temporal features need >=2 complete positive HH/HV power observations")
    low, high = np.array([-40., -45.])[:, None], np.array([5., 0.])[:, None]
    mean = np.clip(10 * np.log10(np.maximum(power.mean(0), 1e-12)), low, high)
    db = np.clip(10 * np.log10(np.maximum(power, 1e-12)), low[None], high[None])
    extras = np.stack([db.std(0, ddof=0), np.percentile(db, 10, axis=0), np.percentile(db, 90, axis=0)])
    return mean.astype("float32"), extras.astype("float32")


def source_key(source):
    return tuple(source[k] for k in ["orbit_pass", "relative_orbit", "frame", "mode", "polarization_code"])


def inventory(data_root, archive_root, output):
    data_root, archive_root, output = map(Path, [data_root, archive_root, output])
    output.mkdir(parents=True, exist_ok=True)
    dataset = data_root / "dataset/s1s2nisar_256_v4"
    revision = data_root / "inputs/preprocessing_v4"
    raw = archive_root / "data/dataset/raw/NISAR"
    sources = json.loads((raw / "source_lookup.json").read_text())["sources"]
    groups = defaultdict(list)
    for source in sources:
        paths = [archive_root / "data/h5" / (source["id"] + ".h5"), raw / "source_h5" / (source["id"] + ".h5")]
        path = next(p for p in paths if p.exists())
        with h5py.File(path, "r") as h:
            group = h["science/LSAR/identification"]
            identity = {k: text_scalar(group[k][()]) for k in IDENTITY}
        validate_nisar_identity(source, identity)
        signature = tuple(identity[k] for k in ["compositeReleaseId", "productVersion", "productSpecificationVersion"])
        item = dict(source, native_identity=identity, signature=signature,
                    archive=str(path), native_cache=str(raw / "native_joint" / (source["id"] + ".tif")))
        groups[source_key(source)].append(item)
    repeated = {k: sorted(v, key=lambda s: s["acquired_start"]) for k, v in groups.items()
                if len({s["acquired_start"][:10] for s in v}) >= 2}
    for key, group in repeated.items():
        if len({tuple(s["signature"]) for s in group}) != 1:
            raise ValueError(f"Mixed processing within repeat group {key}")
        if not all(Path(s["native_cache"]).exists() for s in group):
            raise FileNotFoundError(f"Missing native cache for {key}")
    manifest = json.loads((revision / "manifest.json").read_text())
    lookups = {p: json.loads((revision / f"NISAR_{p}_source_lookup.json").read_text())["sources"] for p in PASSES}
    eligible_indices = {p: [s["source_index"] for s in lookups[p] if source_key(s) in repeated] for p in PASSES}
    records = [json.loads(line) for line in (dataset / "index.jsonl").read_text().splitlines()]
    retained, totals = [], Counter()
    with ExitStack() as stack:
        rasters = {p: stack.enter_context(rasterio.open(data_root / manifest["auxiliary"][f"NISAR_{p}"]["rasters"]["source_index"])) for p in PASSES}
        masks = {p: stack.enter_context(rasterio.open(data_root / manifest["auxiliary"][f"NISAR_{p}"]["valid_mask"])) for p in PASSES}
        for r in records:
            window = Window(r["col"], r["row"], 256, 256)
            joint = np.ones((256, 256), dtype=bool)
            for p in PASSES:
                joint &= np.isin(rasters[p].read(1, window=window), eligible_indices[p]) & (masks[p].read(1, window=window) == 1)
            fraction = float(joint.mean())
            totals[r["split"]] += 1
            if fraction >= .95:
                retained.append(dict(r, repeat_geometry_fraction=fraction))
    report = dict(stage="geometry-only repeat feasibility; new-date validity not yet tested",
        window_utc=[min(s["acquired_start"] for v in repeated.values() for s in v),
                    max(s["acquired_start"] for v in repeated.values() for s in v)],
        local_source_count=len(sources), repeat_group_count=len(repeated),
        repeat_groups=[dict(key=list(k), sources=v) for k, v in repeated.items()],
        original_splits=dict(totals), geometry_candidate_splits=dict(Counter(r["split"] for r in retained)),
        geometry_candidates=retained,
        source_lookups=lookups, input_manifest=str(revision / "manifest.json"),
        dataset=str(dataset), dataset_plan_sha256=sha(dataset / "plan.json"),
        policy=">=95% anchor pixels have >=2 dates in same pass/track/frame/mode/polarization and processing signature; no cross-track temporal averaging")
    save(output / "inventory.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ["repeat_groups", "geometry_candidates", "source_lookups"]}, indent=2), flush=True)
    return report


def write_raster(path, array, profile, nodata, descriptions=None):
    array = array if array.ndim == 3 else array[None]
    profile = dict(profile, count=len(array), dtype=str(array.dtype), nodata=nodata,
                   compress="DEFLATE", predictor=3 if array.dtype.kind == "f" else 2,
                   tiled=True, blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(array)
        if descriptions:
            ds.descriptions = tuple(descriptions)
    with rasterio.open(path) as ds:
        if not np.array_equal(ds.read(), array):
            raise ValueError(f"Raster readback mismatch: {path}")


def build(data_root, archive_root, output):
    data_root, archive_root, output = map(lambda p: Path(p).resolve(), [data_root, archive_root, output])
    report = inventory(data_root, archive_root, output)
    if (output / "build_complete.json").exists():
        raise FileExistsError("Temporal build already completed")
    dataset, revision = Path(report["dataset"]), Path(report["input_manifest"]).parent
    manifest = json.loads(Path(report["input_manifest"]).read_text())
    groups = {tuple(group["key"]): group["sources"] for group in report["repeat_groups"]}
    lookup = {p: {s["source_index"]: s for s in report["source_lookups"][p]} for p in PASSES}
    channels = json.loads((dataset / "plan.json").read_text())["channels"]
    base_names = [c["name"] for c in channels]
    extra_names = [f"NISAR_{p}_{pol}_{stat}" for stat in ["STD_DB", "P10_DB", "P90_DB"]
                   for p in PASSES for pol in ["HH", "HV"]]
    roots = {name: output / name for name in ["T0", "T1", "T2"]}
    if any(root.exists() for root in roots.values()):
        raise FileExistsError("Retain incomplete runs and choose a new temporal output directory")
    for root in roots.values():
        root.mkdir()
    indices = {name: [] for name in roots}
    moments = {name: [np.zeros(26 if name == "T2" else 14, "float64") for _ in range(3)] for name in roots}
    accepted, rejected, source_use = [], [], Counter()
    native_paths = sorted({s["native_cache"] for group in groups.values() for s in group})
    source_hashes = {path: sha(Path(path)) for path in native_paths}
    signatures = {tuple(s["signature"]) for group in groups.values() for s in group}
    if len(signatures) != 1:
        raise ValueError("This pilot requires one NISAR processing signature across all repeat groups")
    with ExitStack() as stack:
        anchors = {p: stack.enter_context(rasterio.open(data_root / manifest["auxiliary"][f"NISAR_{p}"]["rasters"]["source_index"])) for p in PASSES}
        sample = anchors[PASSES[0]]
        vrts = {}
        for path in native_paths:
            source = stack.enter_context(rasterio.open(path))
            vrts[path] = stack.enter_context(WarpedVRT(source, crs=sample.crs, transform=sample.transform,
                width=sample.width, height=sample.height, resampling=Resampling.nearest,
                src_nodata=np.nan, nodata=np.nan))
        for number, r in enumerate(report["geometry_candidates"]):
            win = Window(r["col"], r["row"], 256, 256)
            with rasterio.open(dataset / r["images"]) as ds:
                original, profile = ds.read(), ds.profile
            with rasterio.open(dataset / r["input_validity"]) as ds:
                old_valid = ds.read() == 1
            with rasterio.open(dataset / r["labels"]) as ds:
                labels = ds.read(1)
            means = np.full((4, 256, 256), -9999, dtype="float32")
            extras = np.full((3, 4, 256, 256), -9999, dtype="float32")
            repeated_valid = np.zeros((2, 256, 256), dtype=bool)
            per_patch_sources = set()
            for direction, p in enumerate(PASSES):
                anchor_ids = anchors[p].read(1, window=win)
                for source_index in np.unique(anchor_ids[old_valid[10 + 2 * direction]]):
                    anchor = lookup[p].get(int(source_index))
                    if not anchor or source_key(anchor) not in groups:
                        continue
                    group = groups[source_key(anchor)]
                    if anchor["id"] not in {s["id"] for s in group}:
                        raise ValueError("Anchor is not in its own repeat group")
                    arrays = np.stack([vrts[s["native_cache"]].read(window=win) for s in group])
                    valid = np.isfinite(arrays[:, :2]).all(1) & (arrays[:, :2] > 0).all(1)
                    valid &= (arrays[:, 2] > 0) & (arrays[:, 2] < 255)
                    selected = (anchor_ids == source_index) & old_valid[10 + 2 * direction:12 + 2 * direction].all(0) & valid.all(0)
                    if not selected.any():
                        continue
                    values, statistics = repeat_features(arrays[:, :2, selected])
                    means[2 * direction:2 * direction + 2, selected] = values
                    extras[:, 2 * direction:2 * direction + 2, selected] = statistics
                    repeated_valid[direction] |= selected
                    per_patch_sources.update(s["id"] for s in group)
            common = old_valid.all(0) & repeated_valid.all(0)
            eligible = common & (labels != 255)
            if common.mean() < .95 or eligible.mean() < .80:
                rejected.append(dict(id=r["id"], split=r["split"], common_fraction=float(common.mean()), label_fraction=float(eligible.mean())))
                continue
            labels = labels.copy()
            labels[~common] = 255
            counts = np.bincount(labels[labels != 255], minlength=6).tolist()
            raw = original.copy()
            averaged = original.copy()
            averaged[10:14] = means
            augmented = np.concatenate([averaged, extras.reshape(12, 256, 256)])
            for name, image in [("T0", raw), ("T1", averaged), ("T2", augmented)]:
                names = base_names + (extra_names if name == "T2" else [])
                image[:, ~common] = -9999
                valid = np.broadcast_to(common, image.shape).astype("uint8")
                folder = roots[name] / r["split"]
                paths = dict(images=f"{r['split']}/images/{r['id']}.tif",
                    labels=f"{r['split']}/labels/{r['id']}.tif",
                    input_validity=f"{r['split']}/input_validity/{r['id']}.tif")
                write_raster(roots[name] / paths["images"], image, profile, -9999, names)
                write_raster(roots[name] / paths["labels"], labels, profile, 255)
                write_raster(roots[name] / paths["input_validity"], valid, profile, 255)
                indices[name].append(dict(r, **paths, counts=counts, temporal_sources=sorted(per_patch_sources),
                    temporal_common_pixels=int(common.sum()), valid_input_pixels=[int(common.sum())] * len(image)))
                if r["split"] == "train":
                    values = image[:, common].astype("float64")
                    moments[name][0] += values.sum(1)
                    moments[name][1] += np.square(values).sum(1)
                    moments[name][2] += values.shape[1]
            accepted.append(dict(id=r["id"], split=r["split"], counts=counts))
            source_use.update(per_patch_sources)
            if number % 50 == 0:
                save(output / "progress.json", dict(candidate=number, total=len(report["geometry_candidates"]),
                    accepted=dict(Counter(s["split"] for s in accepted)), rejected=len(rejected)))
                print(json.dumps(dict(candidate=number, accepted=len(accepted), rejected=len(rejected))), flush=True)
    train_records = [r for r in indices["T0"] if r["split"] == "train"]
    weights = mixture_weights([r["counts"] for r in train_records])
    weight_by_id = dict(zip([r["id"] for r in train_records], map(float, weights)))
    for name, root in roots.items():
        records = indices[name]
        for r in records:
            if r["split"] == "train":
                r["sampling_weight"] = weight_by_id[r["id"]]
        (root / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        names = base_names + (extra_names if name == "T2" else [])
        sums, squares, count = moments[name]
        mean = sums / count
        std = np.sqrt(np.maximum(squares / count - mean ** 2, 0))
        save(root / "normalization.json", dict(channels=names, mean=mean.tolist(), std=std.tolist(),
            valid_pixel_counts=count.astype("int64").tolist(), scope="matched training pixels only"))
        save(root / "training_sampling.json", dict(patch_ids=[r["id"] for r in train_records],
            weights=weights.tolist(), index_sha256=sha(root / "index.jsonl"),
            policy="Same frozen 75/25 mixture rule recomputed on matched temporal training labels"))
        summary = {split: dict(patches=sum(r["split"] == split for r in records),
            class_pixels=np.sum([r["counts"] for r in records if r["split"] == split], axis=0).tolist())
            for split in ["train", "validation", "test"]}
        save(root / "plan.json", dict(channels=[dict(name=n, units="physical values; see temporal policy") for n in names],
            patches=records, summary=summary, temporal_variant=name, base_dataset_plan_sha256=report["dataset_plan_sha256"],
            temporal_policy="Fixed source anchor track/frame per pixel; >=2 valid dates in both passes; shared labels/masks/splits; two-date pilot"))
        save(root / "audit.json", dict(channels=len(names), summary=summary,
            checks="Every exported TIFF read back exactly; inherited v4 geography; identical shared label/mask policy; training-only normalization",
            source_native_cache_sha256=source_hashes, groups=report["repeat_groups"],
            quality="v4 reviewed spatial exclusions retained across dates; GCOV/positive/finite validity applied per date; not certified RFI-free"))
        save(root / "complete.json", dict(patches=len(records), summary=summary,
            plan_sha256=sha(root / "plan.json"), index_sha256=sha(root / "index.jsonl")))
    if sha(dataset / "plan.json") != report["dataset_plan_sha256"]:
        raise ValueError("Base dataset changed during temporal build")
    for path, value in source_hashes.items():
        if sha(Path(path)) != value:
            raise ValueError("Native repeat source cache changed during build")
    result = dict(variants={name: str(root) for name, root in roots.items()},
        summary=json.loads((roots["T0"] / "plan.json").read_text())["summary"], rejected=rejected,
        used_sources=dict(source_use), window_utc=report["window_utc"],
        limitation="Two observations per track/frame; short-repeat averaging/representation pilot, not seasonal inference")
    save(output / "build_complete.json", result)
    print(json.dumps(dict(summary=result["summary"], limitation=result["limitation"]), indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="../data")
    parser.add_argument("--archive-root", default="/home/krk/projects/Argos")
    parser.add_argument("--output", default="../data/temporal/validated_repeat_pilot_v1")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    (build if args.build else inventory)(args.data_root, args.archive_root, args.output)


if __name__ == "__main__":
    main()
