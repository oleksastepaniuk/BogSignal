"""Export the "season in motion" assets: NISAR 12-day interferometric coherence through one season.

    ml/.venv/bin/python web/scripts/export_change.py --gunw ../nisar_gunw --run ../runs/insar_e1/track100_frame28 \
        --data ../s1s2nisar_256_v4 [--pairs ../runs/insar_e1/pairs.parquet]

Reads the NISAR L2 GUNW interferograms of one track/frame, the 80 m label mosaic and per-class summary written by
``bogsignal.insar_flooded``, and the patch dataset. Writes web/public/change/:

  change-<n>.png   surface change between two passes (1 - coherence), 40 m, one per interferogram
  labels.png       weak reference classes over the same window
  s2.png, hh.png, hv.png, pol.png   Sentinel-2 colour and NISAR backscatter (HH, HV, HH/HV/ratio composite)
  change.json      dates, window, per-class coherence per pair (whole frame and window), rainfall where known

The window is chosen automatically: the fully observed square with the most balanced mix of swamp/marsh and
peatland cropland. Coherence is a property of a pair of dates, not of one date.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import rasterio
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.image import imsave
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform, transform_bounds

GRID = "science/LSAR/GUNW/grids/frequencyA/wrappedInterferogram/HH"
CLASSES = ["Other", "Swamp", "Marsh", "Permanent water", "Peatland cropland", "Peatland non-cropland"]
CLASS_COLORS = ["#b9b9aa", "#197b49", "#197b49", "#2879b9", "#cf9a4b", "#8b668b"]  # as on the rest of the site
SIZE80, FINE = 256, 2  # window of 256 x 256 pixels at 80 m (20.5 km), exported at 40 m
CHANGE_RANGE = (0.25, 0.85)  # 1 - coherence shown between these limits
CHANGE_CMAP = LinearSegmentedColormap.from_list("change", ["#f4f1e8", "#f0c987", "#d9722c", "#8f2a14", "#3d0d0a"])


def dates(name: str) -> tuple[str, str]:
    d = re.findall(r"_(\d{8})T\d{6}", name)
    return f"{d[0][:4]}-{d[0][4:6]}-{d[0][6:]}", f"{d[2][:4]}-{d[2][4:6]}-{d[2][6:]}"


def box_sum(a: np.ndarray, n: int) -> np.ndarray:
    """sum over every n x n window (top-left indexed)"""
    c = np.pad(a.astype(np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n]


def choose_window(labels: np.ndarray, valid: np.ndarray) -> tuple[int, int]:
    n = SIZE80
    full = box_sum(valid & (labels != 255), n) >= 0.97 * n * n
    wet, crop = box_sum(np.isin(labels, (1, 2)), n), box_sum(labels == 4, n)
    score = np.where(full, np.minimum(wet, crop), -1)
    r, c = np.unravel_index(score.argmax(), score.shape)
    if score[r, c] <= 0:
        raise SystemExit("no fully observed window with both wetland and peatland cropland")
    return int(r), int(c)


def patch_mosaic(data: Path, bands: list[int], dst_transform, epsg: int, size: int, resampling) -> np.ndarray:
    """Dataset patches (image bands) warped onto the window grid; NaN where no patch."""
    out = np.full((len(bands), size, size), np.nan, np.float32)
    w, s, e, n = rasterio.transform.array_bounds(size, size, dst_transform)
    for line in open(data / "index.jsonl"):
        rec = json.loads(line)
        with rasterio.open(data / rec["images"]) as src:
            b = transform_bounds(src.crs, f"EPSG:{epsg}", *src.bounds)
            if b[2] < w or b[0] > e or b[3] < s or b[1] > n:
                continue
            x = src.read([k + 1 for k in bands]).astype(np.float32)
            x[(x == src.nodata) | ~np.isfinite(x)] = np.nan
            tmp = np.full_like(out, np.nan)
            reproject(x, tmp, src_transform=src.transform, src_crs=src.crs, src_nodata=np.nan, dst_transform=dst_transform,
                      dst_crs=f"EPSG:{epsg}", dst_nodata=np.nan, resampling=resampling)
        out = np.where(np.isfinite(tmp), tmp, out)
    return out


def stretch(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((a - lo) / (hi - lo), 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gunw", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True, help="output folder of bogsignal.insar_flooded for the same frame")
    ap.add_argument("--data", type=Path, required=True, help="patch dataset root")
    ap.add_argument("--pairs", type=Path, help="pairs.parquet of bogsignal.insar_pairs, for rainfall between passes")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "public" / "change")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    summary = json.loads((args.run / "summary.json").read_text())
    g = summary["grid"]
    x0, y0, spacing, epsg = g["x0"], g["y0"], g["spacing_m"], g["epsg"]
    labels = np.load(next(args.run.glob("labels_*.npy")))
    files = sorted(p for p in args.gunw.glob("*.h5") if (p.parent / (p.name + ".done")).exists())
    valid = np.all([np.load(args.run / (p.stem + "_ml4.npz"))["valid"] for p in files], 0)
    r, c = choose_window(labels, valid)
    lab = labels[r:r + SIZE80, c:c + SIZE80]
    looks = int(summary["looks"])  # 80 m pixel = looks x looks native pixels
    step, size = looks // FINE, SIZE80 * FINE
    wx, wy, fine = x0 + c * spacing, y0 - r * spacing, spacing / FINE
    dst_transform = from_origin(wx, wy, fine, fine)

    lab_fine = np.repeat(np.repeat(lab, FINE, 0), FINE, 1)
    rgba = np.zeros((size, size, 4), np.float32)
    for k, colour in enumerate(CLASS_COLORS):
        rgba[lab_fine == k] = [int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5)] + [1]
    imsave(args.out / "labels.png", rgba)

    rain = {}
    if args.pairs and args.pairs.exists():
        import pandas as pd

        p = pd.read_parquet(args.pairs)
        p = p[p["precip_between_mm"].notna()]
        for _, row in p.iterrows():
            rain.setdefault((str(row["reference_utc"])[:10], str(row["secondary_utc"])[:10]), []).append(float(row["precip_between_mm"]))

    by_file = {e["file"]: e for e in summary["pairs"]}
    pairs = []
    for n, path in enumerate(files):
        with h5py.File(path, "r") as h:
            coh = h[f"{GRID}/coherenceMagnitude"][r * looks:(r + SIZE80) * looks, c * looks:(c + SIZE80) * looks]
        coh = np.where(np.isfinite(coh) & (coh > 0), coh, np.nan)
        with np.errstate(invalid="ignore"):
            coh = np.nanmean(coh.reshape(size, step, size, step), (1, 3))
        img = CHANGE_CMAP(stretch(1 - coh, *CHANGE_RANGE))
        img[..., 3] = np.isfinite(coh)
        imsave(args.out / f"change-{n}.png", img)
        d0, d1 = dates(path.name)
        days = (np.datetime64(d1) - np.datetime64(d0)).astype(int)
        frame = by_file.get(path.name, {}).get("mean_coherence", {})
        window = {cl: (float(np.nanmean(coh[lab_fine == k])) if (lab_fine == k).any() else None) for k, cl in enumerate(CLASSES)}
        pairs.append({"image": f"change-{n}.png", "start": d0, "end": d1, "days": int(days), "frame_coherence": frame,
                      "window_coherence": window, "rain_mm": float(np.mean(rain[(d0, d1)])) if (d0, d1) in rain else None})
        print(d0, d1, days, "window coherence", {k[:10]: None if v is None else round(v, 2) for k, v in window.items()})

    # context layers from the patch dataset: Sentinel-2 colour and NISAR backscatter in both polarizations
    plan = json.loads((args.data / "plan.json").read_text())
    names = [ch["name"] for ch in plan["channels"]]
    ix = {n: names.index(n) for n in ("S2_B4", "S2_B3", "S2_B2", "NISAR_ASCENDING_HH", "NISAR_ASCENDING_HV", "NISAR_DESCENDING_HH", "NISAR_DESCENDING_HV")}
    m = patch_mosaic(args.data, list(ix.values()), dst_transform, epsg, size, Resampling.bilinear)
    alpha = np.isfinite(m[0]).astype(np.float32)
    imsave(args.out / "s2.png", np.dstack([stretch(np.nan_to_num(m[i]), 0.0, 0.14) ** 0.8 for i in range(3)] + [alpha]))
    hh = np.where(np.isfinite(m[3]), m[3], m[5])
    hv = np.where(np.isfinite(m[4]), m[4], m[6])
    if plan["channels"][ix["NISAR_ASCENDING_HH"]].get("units", "").startswith("db"):
        hh_db, hv_db = hh, hv
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            hh_db, hv_db = 10 * np.log10(np.maximum(hh, 1e-6)), 10 * np.log10(np.maximum(hv, 1e-6))
    a = np.isfinite(hh_db).astype(np.float32)
    gray = colormaps["gray"]
    for name, db, lo, hi in (("hh", hh_db, -20, -4), ("hv", hv_db, -27, -10)):
        im = gray(stretch(np.nan_to_num(db, nan=lo), lo, hi))
        im[..., 3] = a
        imsave(args.out / f"{name}.png", im)
    ratio = hh_db - hv_db
    imsave(args.out / "pol.png", np.dstack([stretch(np.nan_to_num(hh_db, nan=-20), -20, -4), stretch(np.nan_to_num(hv_db, nan=-27), -27, -10),
                                            stretch(np.nan_to_num(ratio, nan=3), 3, 12), a]))
    backscatter = {cl: {"hh_db": float(np.nanmean(hh_db[lab_fine == k])), "hv_db": float(np.nanmean(hv_db[lab_fine == k]))}
                   for k, cl in enumerate(CLASSES) if (lab_fine == k).any()}

    lon, lat = transform(f"EPSG:{epsg}", "EPSG:4326", [wx, wx + size * fine], [wy, wy - size * fine])
    out = {"pairs": pairs, "classes": CLASSES, "class_colors": CLASS_COLORS, "polarization": "HH", "pixel_m": fine, "size_px": size,
           "window_km": size * fine / 1000, "bounds": [[lat[1], lon[0]], [lat[0], lon[1]]], "center": [float(np.mean(lat)), float(np.mean(lon))],
           "change_range": CHANGE_RANGE, "window_class_share": {cl: float((lab == k).mean()) for k, cl in enumerate(CLASSES)},
           "frame_labelled_pixels_80m": summary["labelled_pixels"], "window_backscatter": backscatter,
           "source": "NISAR L2 GUNW (provisional), frequency A, HH, track 100 frame 28; coherenceMagnitude of the wrapped interferogram, 20 m averaged to 40 m",
           "labels_source": f"{args.data.name} weak reference labels on the 80 m radar grid",
           "files": [p.name for p in files]}
    (args.out / "change.json").write_text(json.dumps(out, indent=1))
    print("window row/col", r, c, "centre", out["center"], "class share", {k[:10]: round(v, 3) for k, v in out["window_class_share"].items()})


if __name__ == "__main__":
    main()
