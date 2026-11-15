"""Per-pixel LightGBM baseline and channel-usefulness check for the Polissia patches.

    uv run python -m bogsignal.baseline_gbm --data ../../s1s2nisar_256_v4 --out runs/gbm

Trains on a stride-8 pixel grid of every training patch and scores validation on
the same stride-4 grid the U-Net evaluation uses, so average precision is
directly comparable with ``train_unet`` (IoU here is on that pixel subsample).

It answers three questions:
  * how far a model with no spatial context gets (``all raw+idx``),
  * whether local context helps a pixel model (``all raw+idx+ctx``),
  * which sensors and channels carry the signal (sensor subsets, permutation importance).
"""

from __future__ import annotations

import argparse
import json
import warnings
from multiprocessing import Pool
from pathlib import Path

import lightgbm as lgb
import numpy as np
import rasterio
from scipy.ndimage import uniform_filter

from bogsignal.train_unet import CLASSES, IGNORE, SENSOR_CHANNELS, average_precision, class_iou, read_index, to_model_units

RAW = ["S2_B2", "S2_B3", "S2_B4", "S2_B8", "S2_B11", "S2_B12", "S1A_VV", "S1A_VH", "S1D_VV", "S1D_VH",
       "NISAR_A_HH", "NISAR_A_HV", "NISAR_D_HH", "NISAR_D_HV"]
SENSOR_OF = {c: s for s, chans in SENSOR_CHANNELS.items() for c in chans}
# name, sensors it needs, function of the 14 raw channels
nd = lambda a, b: (a - b) / (a + b + 1e-6)  # noqa: E731
INDICES = [
    ("NDVI", {"s2"}, lambda x: nd(x[3], x[2])), ("NDWI", {"s2"}, lambda x: nd(x[1], x[3])),
    ("MNDWI", {"s2"}, lambda x: nd(x[1], x[4])), ("NDMI", {"s2"}, lambda x: nd(x[3], x[4])),
    ("NBR2", {"s2"}, lambda x: nd(x[4], x[5])),
    ("S1A_VV-VH", {"s1"}, lambda x: x[6] - x[7]), ("S1D_VV-VH", {"s1"}, lambda x: x[8] - x[9]),
    ("NISAR_A_HH-HV", {"nisar"}, lambda x: x[10] - x[11]), ("NISAR_D_HH-HV", {"nisar"}, lambda x: x[12] - x[13]),
    ("L_HH-C_VV", {"s1", "nisar"}, lambda x: np.nanmean([x[10], x[12]], 0) - np.nanmean([x[6], x[8]], 0)),
]
CTX = (5, 15)  # box-mean window sizes in pixels (100 m, 300 m)


def feature_table() -> list[tuple[str, set, str]]:
    """(name, sensors needed, kind) for every column produced by ``patch_features``."""
    cols = [(n, {SENSOR_OF[i]}, "raw") for i, n in enumerate(RAW)]
    cols += [(n, s, "idx") for n, s, _ in INDICES]
    cols += [(f"{n}_mean{w}", {SENSOR_OF[i]}, "ctx") for w in CTX for i, n in enumerate(RAW)]
    return cols


def patch_features(job):
    root, rec, linear_sar, step, offset = job
    with rasterio.open(root / rec["images"]) as ds:
        x = to_model_units(ds.read(), linear_sar)
    with rasterio.open(root / rec["labels"]) as ds:
        y = ds.read(1)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels where a sensor is missing
        idx = np.stack([f(x) for _, _, f in INDICES])
        fill = np.nan_to_num(np.nanmean(x, (1, 2), keepdims=True))
    filled = np.where(np.isnan(x), fill, x)
    ctx = [uniform_filter(filled, (1, w, w), mode="nearest") for w in CTX]
    f = np.concatenate([x, idx, *ctx])[:, offset::step, offset::step]
    y = y[offset::step, offset::step].reshape(-1)
    keep = y != IGNORE
    return f.reshape(len(f), -1).T[keep].astype(np.float32), y[keep]


def extract(root: Path, split: str, step: int, offset: int, workers: int, cache: Path):
    path = cache / f"{root.name}_{split}_s{step}o{offset}.npz"
    if path.exists():
        d = np.load(path)
        return d["x"], d["y"]
    units = [c["units"] for c in json.loads((root / "plan.json").read_text())["channels"]][6:]
    linear = units[0].startswith("linear_")
    jobs = [(root, r, linear, step, offset) for r in read_index(root).values() if r["split"] == split]
    with Pool(workers) as pool:
        parts = pool.map(patch_features, jobs, chunksize=16)
    x, y = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    np.savez(path, x=x, y=y)
    return x, y


def score(p: np.ndarray, y: np.ndarray) -> dict:
    wet = (y == 1) | (y == 2)
    iou = class_iou(p.argmax(1), y.astype(np.int64))
    hits = wet[np.argsort(-(p[:, 1] + p[:, 2]), kind="stable")]
    out = {
        "ap": {c: round(average_precision(p[:, i], y == i), 4) for i, c in enumerate(CLASSES)},
        "iou_sub": dict(zip(CLASSES, iou.round(4).tolist())),
        "miou_sub": round(float(iou.mean()), 4),
        "wetland_ap": round(average_precision(p[:, 1] + p[:, 2], wet), 4),
        "wetland_recall_at_area": {str(b): round(float(hits[: round(len(hits) * b / 100)].sum() / max(hits.sum(), 1)), 4) for b in (1, 5, 10, 20)},
    }
    out["ap_score"] = round((out["ap"]["Swamp"] + out["ap"]["Marsh"]) / 2, 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    xt, yt = extract(args.data, "train", 8, 3, args.workers, args.out)
    xv, yv = extract(args.data, "validation", 4, 2, args.workers, args.out)
    cols = feature_table()
    print(f"train {xt.shape}, validation {xv.shape}, features {len(cols)}", flush=True)
    params = dict(objective="multiclass", num_class=len(CLASSES), learning_rate=0.1, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.5, bagging_freq=1, max_bin=63, num_threads=args.threads,
                  seed=args.seed, verbose=-1)

    def fit(sensors: set, kinds: set):
        use = [i for i, (_, need, kind) in enumerate(cols) if need <= sensors and kind in kinds]
        model = lgb.train(params, lgb.Dataset(xt[:, use], yt, feature_name=[cols[i][0] for i in use]), args.rounds)
        return model, use

    results = {}
    everything = {"s2", "s1", "nisar"}
    runs = [("+".join(sorted(s)) + " raw+idx", s, {"raw", "idx"}) for s in
            ({"s2"}, {"s1"}, {"nisar"}, {"s2", "s1"}, {"s2", "nisar"}, {"s1", "nisar"}, everything)]
    runs += [("all raw", everything, {"raw"}), ("all raw+idx+ctx", everything, {"raw", "idx", "ctx"})]
    for name, sensors, kinds in runs:
        model, use = fit(sensors, kinds)
        m = score(model.predict(xv[:, use]), yv)
        m["features"] = len(use)
        results[name] = m
        print(f"{name:26s} AP swamp {m['ap']['Swamp']:.3f} marsh {m['ap']['Marsh']:.3f} wetland {m['wetland_ap']:.3f} "
              f"water {m['ap']['Permanent water']:.3f} peat-crop {m['ap']['Peatland cropland']:.3f} "
              f"peat-other {m['ap']['Peatland non-cropland']:.3f} | mIoU {m['miou_sub']:.3f}", flush=True)
        if name == "all raw":  # permutation importance of each raw channel on a validation subsample
            rng = np.random.default_rng(args.seed)
            sub = rng.choice(len(xv), min(len(xv), 600_000), replace=False)
            xs, ys = xv[sub][:, use], yv[sub]
            ref = score(model.predict(xs), ys)
            gain = model.feature_importance("gain")
            imp = {}
            for j, i in enumerate(use):
                shuffled = xs.copy()
                shuffled[:, j] = rng.permutation(shuffled[:, j])
                s = score(model.predict(shuffled), ys)
                imp[cols[i][0]] = {"gain_share": round(float(gain[j] / gain.sum()), 4),
                                   "swamp_ap_drop": round(ref["ap"]["Swamp"] - s["ap"]["Swamp"], 4),
                                   "marsh_ap_drop": round(ref["ap"]["Marsh"] - s["ap"]["Marsh"], 4),
                                   "wetland_ap_drop": round(ref["wetland_ap"] - s["wetland_ap"], 4)}
                print(f"  permute {cols[i][0]:12s} gain {imp[cols[i][0]]['gain_share']:.3f} "
                      f"AP drop swamp {imp[cols[i][0]]['swamp_ap_drop']:+.3f} marsh {imp[cols[i][0]]['marsh_ap_drop']:+.3f} "
                      f"wetland {imp[cols[i][0]]['wetland_ap_drop']:+.3f}", flush=True)
            results["channel_importance"] = imp
        (args.out / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
