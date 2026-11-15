"""Matched provisional-only single-date/mean/variability datasets for experiment C."""
import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np
import rasterio

from .pipeline import sha, save
from .sampling import mixture_weights
from .temporal import write_raster


def summarize(power, minimum_observations=5):
    """Return latest dB, dB(mean power), dB spread/quantiles, counts and validity."""
    power = np.asarray(power, dtype="float64")
    if power.ndim != 4 or power.shape[1] != 2 or minimum_observations < 2:
        raise ValueError("Expected [dates, HH/HV, height, width] and >=2 minimum dates")
    joint = np.isfinite(power).all(1) & (power > 0).all(1)
    count = joint.sum(0)
    keep = count >= minimum_observations
    latest = np.full(power.shape[1:], -9999, dtype="float32")
    mean = latest.copy()
    extras = np.full((3, *latest.shape), -9999, dtype="float32")
    if keep.any():
        samples = power[:, :, keep].copy()
        valid = joint[:, keep]
        samples[~np.broadcast_to(valid[:, None], samples.shape)] = np.nan
        low, high = np.array([-40., -45.])[:, None], np.array([5., 0.])[:, None]
        db = np.clip(10 * np.log10(np.maximum(samples, 1e-12)), low[None], high[None])
        last = len(samples) - 1 - valid[::-1].argmax(0)
        latest[:, keep] = db[last, :, np.arange(len(last))].T
        mean[:, keep] = np.clip(10 * np.log10(np.maximum(np.nanmean(samples, axis=0), 1e-12)), low, high)
        # NaNs sort after finite dates. Vectorized order-statistic interpolation
        # avoids np.nanpercentile's Python loop over hundreds of thousands of pixels.
        ordered = np.sort(db, axis=0)
        observations = valid.sum(0)
        def quantile(q):
            position = (observations - 1) * q
            lower, upper = np.floor(position).astype(int), np.ceil(position).astype(int)
            fraction = position - lower
            lo = ordered[lower, :, np.arange(len(lower))].T
            hi = ordered[upper, :, np.arange(len(upper))].T
            return lo * (1 - fraction) + hi * fraction
        extras[:, :, keep] = np.stack([np.nanstd(db, axis=0, ddof=0),
            quantile(.1), quantile(.9)])
    return latest, mean, extras, count.astype("uint8"), keep


def build(data_root, series_root, output, minimum_observations=3):
    data_root, series_root, output = map(lambda p: Path(p).resolve(), [data_root, series_root, output])
    if not (series_root / "acquisition_complete.json").exists():
        raise ValueError("Temporal acquisition must complete before freezing matched datasets")
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((series_root / "series_plan.json").read_text())
    original = data_root / "dataset/s1s2nisar_256_v4"
    original_hashes = {name: sha(original / name) for name in
        ["plan.json", "index.jsonl", "normalization.json", "training_sampling.json", "complete.json", "audit.json"]}
    records = [json.loads(line) for line in (original / "index.jsonl").read_text().splitlines()]
    groups, lookup, sources = [], {}, {}
    for job in plan["groups"]:
        marker = Path(job["folder"]) / "complete.json"
        completed = json.loads(marker.read_text())
        group = dict(job, completed=completed, record_index={r["id"]: i for i, r in enumerate(job["records"])})
        groups.append(group)
        for source in completed["sources"]:
            sources[source["cache"]] = source["cache_sha256"]
        for r in job["records"]:
            lookup.setdefault((job["key"][0], r["id"]), []).append(group)
    for path, digest in sources.items():
        if sha(Path(path)) != digest:
            raise ValueError("Temporal source cache checksum mismatch")
    roots = {name: output / name for name in ["T0", "T1", "T2"]}
    for root in roots.values():
        root.mkdir()
    base_names = [c["name"] for c in json.loads((original / "plan.json").read_text())["channels"]]
    extra_names = [f"NISAR_{p}_{pol}_{stat}" for stat in ["STD_DB", "P10_DB", "P90_DB"]
                   for p in ["ASCENDING", "DESCENDING"] for pol in ["HH", "HV"]]
    indices = {name: [] for name in roots}
    moments = {name: [np.zeros(26 if name == "T2" else 14, "float64") for _ in range(3)] for name in roots}
    rejected, observation_histogram = [], Counter()
    for number, r in enumerate(records):
        with rasterio.open(original / r["images"]) as ds:
            image, profile = ds.read(), ds.profile
        with rasterio.open(original / r["input_validity"]) as ds:
            original_validity = ds.read() == 1
            common = original_validity.all(0)
        with rasterio.open(original / r["labels"]) as ds:
            labels = ds.read(1)
        latest = np.full((4, 256, 256), -9999, dtype="float32")
        mean = latest.copy()
        statistics = np.full((3, 4, 256, 256), -9999, dtype="float32")
        count = np.zeros((2, 256, 256), dtype="uint8")
        temporal_validity = np.zeros((4, 256, 256), bool)
        used = set()
        for direction, p in enumerate(["ASCENDING", "DESCENDING"]):
            pass_keep = np.zeros((256, 256), bool)
            for group in lookup.get((p, r["id"]), []):
                i = group["record_index"][r["id"]]
                arrays = np.stack([np.load(s["cache"], mmap_mode="r")[i] for s in group["completed"]["sources"]])
                last, average, extras, n, valid = summarize(arrays, minimum_observations)
                if (pass_keep & valid).any():
                    raise ValueError("Two temporal track/frame groups claim the same anchor pixel")
                sl = slice(direction * 2, direction * 2 + 2)
                latest[sl, valid] = last[:, valid]
                mean[sl, valid] = average[:, valid]
                statistics[:, sl, valid] = extras[:, :, valid]
                count[direction, valid] = n[valid]
                pass_keep |= valid
                used.update(s["id"] for s in group["completed"]["sources"])
            common &= pass_keep
            temporal_validity[direction * 2:direction * 2 + 2] = pass_keep
        eligible = common & (labels != 255)
        if common.mean() < .95 or eligible.mean() < .80:
            rejected.append(dict(id=r["id"], split=r["split"], common_fraction=float(common.mean()), label_fraction=float(eligible.mean())))
            continue
        labels = labels.copy()
        labels[~common] = 255
        counts = np.bincount(labels[labels != 255], minlength=6).tolist()
        observation_histogram.update(map(int, count[:, common].ravel()))
        raw, averaged = image.copy(), image.copy()
        raw[10:14], averaged[10:14] = latest, mean
        augmented = np.concatenate([averaged, statistics.reshape(12, 256, 256)])
        base_validity = original_validity.copy()
        base_validity[10:14] = temporal_validity
        for name, values in [("T0", raw), ("T1", averaged), ("T2", augmented)]:
            names = base_names + (extra_names if name == "T2" else [])
            # Preserve optical/S1 observations and source-specific masks even at
            # temporally ineligible pixels. Otherwise a Sentinel-only control
            # would receive a hidden NISAR availability mask through its inputs.
            validity = np.concatenate([base_validity, np.tile(temporal_validity, (3, 1, 1))]) if name == "T2" else base_validity
            values[~validity] = -9999
            validity = validity.astype("uint8")
            paths = dict(images=f"{r['split']}/images/{r['id']}.tif", labels=f"{r['split']}/labels/{r['id']}.tif",
                         input_validity=f"{r['split']}/input_validity/{r['id']}.tif")
            write_raster(roots[name] / paths["images"], values, profile, -9999, names)
            write_raster(roots[name] / paths["labels"], labels, profile, 255)
            write_raster(roots[name] / paths["input_validity"], validity, profile, 255)
            indices[name].append(dict(r, **paths, counts=counts, temporal_sources=sorted(used),
                temporal_common_pixels=int(common.sum()), valid_input_pixels=validity.reshape(len(values), -1).sum(1).tolist()))
            if r["split"] == "train":
                for band in range(len(values)):
                    valid_values = values[band, validity[band] == 1].astype("float64")
                    moments[name][0][band] += valid_values.sum()
                    moments[name][1][band] += np.square(valid_values).sum()
                    moments[name][2][band] += len(valid_values)
        write_raster(output / "observation_counts" / r["split"] / f"{r['id']}.tif", count, profile, 0,
                     ["Ascending valid dates", "Descending valid dates"])
        if number % 50 == 0:
            status = dict(candidate=number, total=len(records), accepted=len(indices["T0"]), rejected=len(rejected))
            save(output / "progress.json", status)
            print(json.dumps(status), flush=True)
    train = [r for r in indices["T0"] if r["split"] == "train"]
    weights = mixture_weights([r["counts"] for r in train])
    weight_by_id = dict(zip([r["id"] for r in train], map(float, weights)))
    summaries = {}
    for name, root in roots.items():
        rows = indices[name]
        for r in rows:
            if r["split"] == "train":
                r["sampling_weight"] = weight_by_id[r["id"]]
        (root / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        names = base_names + (extra_names if name == "T2" else [])
        sums, squares, count = moments[name]
        means = sums / count
        std = np.sqrt(np.maximum(squares / count - means ** 2, 0))
        shared = 10 if name == "T1" else 14 if name == "T2" else 0
        if shared:
            reference = json.loads((roots["T0" if name == "T1" else "T1"] / "normalization.json").read_text())
            for actual, field in [(means, "mean"), (std, "std")]:
                expected = np.asarray(reference[field])[:shared]
                if not np.allclose(actual[:shared], expected, atol=1e-10, rtol=1e-10):
                    raise ValueError("Shared physical predictors unexpectedly changed normalization")
                actual[:shared] = expected  # identical floating-point normalization for fixed shared channels
        save(root / "normalization.json", dict(channels=names, mean=means.tolist(), std=std.tolist(),
            valid_pixel_counts=count.astype("int64").tolist(), scope="matched training pixels only"))
        save(root / "training_sampling.json", dict(patch_ids=[r["id"] for r in train], weights=weights.tolist(),
            index_sha256=sha(root / "index.jsonl"), policy="Frozen 75/25 mixture rule, recomputed on matched temporal training labels"))
        summary = {s: dict(patches=sum(r["split"] == s for r in rows),
            class_pixels=np.sum([r["counts"] for r in rows if r["split"] == s], axis=0).tolist()) for s in ["train", "validation", "test"]}
        summaries[name] = summary
        policy = dict(collection="NISAR_L2_GCOV_PROVISIONAL_V1", signature=plan["native_processing_signature"],
            window_utc=plan["window_utc"], minimum_observations=minimum_observations,
            baseline="Latest valid provisional date per anchor pixel", average="Mean valid linear power, then dB and baseline clipping",
            variability="Population std and p10/p90 of per-date clipped dB", valid_pixel_policy="Same v4 spatial exclusions, complete HH/HV dates, common pixels/labels/masks in all variants",
            geography="Inherited v4 folds, groups and buffers; no geographical reshuffle",
            sources=sha(series_root / "series_plan.json"), original_dataset_hashes=original_hashes)
        save(root / "plan.json", dict(channels=[dict(name=n) for n in names], patches=rows, summary=summary,
            temporal_variant=name, temporal_policy=policy))
        save(root / "audit.json", dict(summary=summary, channels=len(names), source_cache_sha256=sources,
            checks="Every exported TIFF read back exactly; shared inherited geography/labels/masks; training-only normalization",
            observation_histogram=dict(observation_histogram), policy=policy,
            limitations="Provisional data; v4 exclusions retained but not a comprehensive multitemporal RFI mask; historical weak labels"))
        save(root / "complete.json", dict(patches=len(rows), summary=summary, plan_sha256=sha(root / "plan.json")))
    if any(sha(original / name) != digest for name, digest in original_hashes.items()):
        raise ValueError("Original v4 metadata changed")
    if summaries["T0"] != summaries["T1"] or summaries["T1"] != summaries["T2"]:
        raise ValueError("Temporal variants differ in geography or target counts")
    result = dict(variants={name: str(root) for name, root in roots.items()}, summary=summaries["T0"],
        window_utc=plan["window_utc"], minimum_observations=minimum_observations,
        observation_histogram=dict(observation_histogram), rejected=rejected, test_evaluated=False)
    save(output / "build_complete.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "rejected"}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="../data")
    parser.add_argument("--series-root", default="/home/krk/projects/Argos/data/temporal_sources/provisional_v1")
    parser.add_argument("--output", default="../data/temporal/provisional_v1")
    parser.add_argument("--minimum-observations", type=int, default=3)
    args = parser.parse_args()
    build(args.data_root, args.series_root, args.output, args.minimum_observations)


if __name__ == "__main__":
    main()
