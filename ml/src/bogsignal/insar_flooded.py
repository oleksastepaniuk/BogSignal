"""Flooded-vegetation candidates from NISAR L-band interferograms (GUNW), after Hübinger et al. 2026.

    python -m bogsignal.insar_flooded --gunw ../../nisar_gunw --data ../../s1s2nisar_256_v4 --out ../../runs/insar

Hübinger, Fluet-Chouinard, Escobar & Jaramillo (Remote Sensing of Environment 332, 115086) map water under
vegetation by combining InSAR fringe detection with local phase homogeneity and by randomly sampling the
processing parameters to obtain a probability map. Only the abstract of that paper was available when this was
written, so the two measures below use their standard definitions and the parameter ranges are our own choices;
this is an implementation of the idea, not a reproduction of the paper's exact workflow.

For every interferogram (one 12-day pair):
  1. the wrapped complex interferogram is multilooked (default 4 x 4, 20 m -> 80 m);
  2. local phase homogeneity = | mean of unit phasors | in a small window: 1 where neighbouring phases agree
     (a coherent return, as from double bounce between water and stems), ~0 where phase is noise (open water,
     decorrelated canopy);
  3. fringe measure = | phase - regional background phase |, the background being the phase of the complex mean
     over a large window (removes atmosphere, ionosphere and orbit ramps); a water-level change under vegetation
     between the two dates shows up as a local phase offset or fringe;
  4. a pixel is a flooded-vegetation candidate when homogeneity and fringe both exceed thresholds; window sizes
     and thresholds are drawn at random N times and the share of draws that flag the pixel is its probability.

Reference labels from the patch dataset are warped onto the radar grid only to summarise the result by class.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from scipy.ndimage import uniform_filter

GRID = "science/LSAR/GUNW/grids/frequencyA/wrappedInterferogram"
CLASSES = ["Other", "Swamp", "Marsh", "Permanent water", "Peatland cropland", "Peatland non-cropland"]


def pair_dates(path: Path) -> tuple[str, str]:
    """reference and secondary acquisition dates from the product name"""
    d = re.findall(r"_(\d{8})T\d{6}", path.name)
    return f"{d[0][:4]}-{d[0][4:6]}-{d[0][6:]}", f"{d[2][:4]}-{d[2][4:6]}-{d[2][6:]}"


def load_pair(path: Path, looks: int, cache: Path, pol: str = "HH"):
    """Multilooked complex interferogram, mean coherence, validity, and the grid (x0, y0, spacing, epsg)."""
    out = cache / (path.stem + f"_ml{looks}.npz")
    with h5py.File(path, "r") as h:
        g = h[f"{GRID}/{pol}"]
        x, y, epsg = g["xCoordinates"][:], g["yCoordinates"][:], int(g["projection"][()])
        grid = (float(x[0] - (x[1] - x[0]) / 2), float(y[0] - (y[1] - y[0]) / 2), float(abs(x[1] - x[0])) * looks, epsg)
        if out.exists():
            d = np.load(out)
            return d["ifg"], d["coh"], d["valid"], grid
        rows, cols = (len(y) // looks) * looks, (len(x) // looks) * looks
        ifg = np.zeros((rows // looks, cols // looks), np.complex64)
        coh = np.zeros(ifg.shape, np.float32)
        count = np.zeros(ifg.shape, np.float32)
        step = 256 * looks
        for r in range(0, rows, step):
            n = min(step, rows - r)
            z = g["wrappedInterferogram"][r:r + n, :cols]
            c = g["coherenceMagnitude"][r:r + n, :cols]
            ok = np.isfinite(z.real) & np.isfinite(z.imag) & (z != 0) & np.isfinite(c)
            z, c = np.where(ok, z, 0), np.where(ok, c, 0)
            block = lambda a: a.reshape(n // looks, looks, cols // looks, looks).sum((1, 3))  # noqa: E731
            sl = slice(r // looks, (r + n) // looks)
            ifg[sl], coh[sl], count[sl] = block(z), block(c), block(ok.astype(np.float32))
    valid = count >= looks * looks / 2
    coh = np.where(valid, coh / np.maximum(count, 1), 0).astype(np.float32)
    np.savez(out, ifg=ifg, coh=coh, valid=valid)
    return ifg, coh, valid, grid


def box(a: np.ndarray, size: int) -> np.ndarray:
    if np.iscomplexobj(a):
        return uniform_filter(a.real, size, mode="nearest") + 1j * uniform_filter(a.imag, size, mode="nearest")
    return uniform_filter(a, size, mode="nearest")


def phase_measures(ifg: np.ndarray, valid: np.ndarray, window: int, background: int):
    """local phase homogeneity (0-1) and fringe measure (absolute phase offset from the regional background, rad)"""
    unit = np.where(valid, ifg / np.maximum(np.abs(ifg), 1e-12), 0).astype(np.complex64)
    weight = np.maximum(box(valid.astype(np.float32), window), 1e-6)
    local = box(unit, window) / weight
    regional = box(unit, background) / np.maximum(box(valid.astype(np.float32), background), 1e-6)
    homogeneity = np.abs(local).astype(np.float32)
    fringe = np.abs(np.angle(local * np.conj(regional))).astype(np.float32)
    return np.where(valid, homogeneity, 0), np.where(valid, fringe, 0)


def probability(ifg: np.ndarray, valid: np.ndarray, draws: int, rng: np.random.Generator, spacing: float):
    """Share of random parameter draws in which a pixel has both a homogeneous and an offset phase."""
    hits = np.zeros(ifg.shape, np.float32)
    params = []
    for _ in range(draws):
        window = int(rng.choice([3, 5, 7, 9]))
        background = int(rng.integers(round(3000 / spacing), round(8000 / spacing))) | 1  # 3-8 km
        h_thr, f_thr = float(rng.uniform(0.8, 0.95)), float(rng.uniform(0.3, 1.0))
        homogeneity, fringe = phase_measures(ifg, valid, window, background)
        hits += (homogeneity > h_thr) & (fringe > f_thr)
        params.append({"window": window, "background": background, "homogeneity_threshold": round(h_thr, 3), "fringe_threshold": round(f_thr, 3)})
    return np.where(valid, hits / draws, np.nan).astype(np.float32), params


def paper_iteration(phase: np.ndarray, valid: np.ndarray, sigma: float, low: float, high: float, window: int, var_thr: float,
                    kernel: int, buffer: int = 3) -> np.ndarray:
    """One pass of the Hübinger et al. (2026) workflow, section 2.2 / Fig. 4 A-F.

    A-B  Canny edge detection on the Gaussian-smoothed wrapped phase finds fringe turnovers (the -pi/pi jump);
    C    the fringe lines are buffered by ``buffer`` pixels (three in the paper);
    D    homogeneous-phase regions = local circular variance of the phase (1 - |mean unit phasor|) below ``var_thr`` in
         a ``window`` moving window, cleaned with a morphological opening of size ``kernel``. The paper only says
         "local variance"; the circular form is used so that the -pi/pi wrap itself does not count as noise;
    E-F  only connected homogeneous groups that touch a buffered fringe are kept.
    """
    from scipy.ndimage import binary_dilation, binary_opening, label
    from skimage.feature import canny
    from skimage.morphology import disk

    edges = canny(phase, sigma=sigma, low_threshold=low, high_threshold=high, mask=valid)
    fringe = binary_dilation(edges, disk(buffer))
    unit = np.where(valid, np.exp(1j * phase), 0).astype(np.complex64)
    variance = 1 - np.abs(box(unit, window)) / np.maximum(box(valid.astype(np.float32), window), 1e-6)
    homogeneous = binary_opening((variance < var_thr) & valid, np.ones((kernel, kernel), bool))
    groups, n = label(homogeneous)
    keep = np.zeros(n + 1, bool)
    keep[np.unique(groups[fringe & homogeneous])] = True
    keep[0] = False
    return keep[groups]


def remove_regional_phase(ifg: np.ndarray, valid: np.ndarray, pixels: int) -> np.ndarray:
    """Interferogram with the phase of its complex mean over a ``pixels`` wide window removed.

    Not part of the paper's workflow. Atmosphere, ionosphere and orbit errors put several broad fringes across a
    NISAR frame; the paper names such fringes as a source of false detections and suggests a secondary filter."""
    unit = np.where(valid, ifg / np.maximum(np.abs(ifg), 1e-12), 0).astype(np.complex64)
    regional = box(unit, pixels | 1)
    return ifg * np.conj(regional / np.maximum(np.abs(regional), 1e-12))


def paper_probability(ifg: np.ndarray, valid: np.ndarray, draws: int, rng: np.random.Generator, detrend_pixels: int = 0):
    """Frequency with which the paper's workflow flags a pixel over random parameter draws (Fig. 4 G; 100 in the paper).

    The paper's parameter ranges are in its supplementary Table S2, which we do not have: the ranges here are ours."""
    if detrend_pixels:
        ifg = remove_regional_phase(ifg, valid, detrend_pixels)
    phase = np.where(valid, np.angle(ifg), 0).astype(np.float32)
    hits = np.zeros(ifg.shape, np.float32)
    params = []
    for _ in range(draws):
        sigma = float(rng.uniform(1.5, 3.0))
        # Canny thresholds as a share of the response of a full -pi/pi turnover after smoothing (Sobel: 8 * 2pi / (sigma * sqrt(2pi))),
        # so that edges between coherent and noisy ground, or within smoothed noise, are not taken for fringes
        high = float(rng.uniform(0.4, 0.7)) * 8 * np.sqrt(2 * np.pi) / sigma
        q = {"sigma": sigma, "low": high * float(rng.uniform(0.4, 0.6)), "high": high, "window": int(rng.choice([5, 7, 9, 11])),
             "var_thr": float(rng.uniform(0.05, 0.3)), "kernel": int(rng.choice([3, 5]))}
        hits += paper_iteration(phase, valid, **q)
        params.append(q)
    return np.where(valid, hits / draws, np.nan).astype(np.float32), params


def flood_regime(masks: list[np.ndarray], dates: list[tuple[str, str]], valid: np.ndarray):
    """Section 2.3 of the paper: flooded vegetation per acquisition date and the seasonal regime.

    An interferogram shows water present on both of its dates, so the extent at one date is the union of the masks
    of the two consecutive interferograms that share it (first and last dates cannot be inferred). The regime is
    2 = permanently flooded (at every inferred date), 1 = seasonally flooded (at some), 0 = never, 255 = no data.
    """
    by_date = {}
    for (a, b), (c, d), m0, m1 in zip(dates[:-1], dates[1:], masks[:-1], masks[1:]):
        if b == c:  # consecutive pairs only
            by_date[b] = m0 | m1
    if not by_date:
        return {}, np.full(valid.shape, 255, np.uint8)
    count = np.sum(list(by_date.values()), 0)
    regime = np.where(count == len(by_date), 2, np.where(count > 0, 1, 0)).astype(np.uint8)
    regime[~valid] = 255
    return by_date, regime


def write_tif(path: Path, array: np.ndarray, grid, nodata) -> None:
    x0, y0, spacing, epsg = grid
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1], count=1, dtype=array.dtype.name,
                       crs=f"EPSG:{epsg}", transform=from_origin(x0, y0, spacing, spacing), nodata=nodata, compress="deflate") as dst:
        dst.write(array, 1)


def label_mosaic(data: Path, shape, grid, cache: Path) -> np.ndarray:
    """Reference labels of every dataset patch warped (nearest) onto the radar grid; 255 where there is no patch."""
    out = cache / f"labels_{shape[0]}x{shape[1]}_{int(grid[2])}m.npy"
    if out.exists():
        return np.load(out)
    x0, y0, spacing, epsg = grid
    dst_transform = from_origin(x0, y0, spacing, spacing)
    mosaic = np.full(shape, 255, np.uint8)
    for line in open(data / "index.jsonl"):
        rec = json.loads(line)
        with rasterio.open(data / rec["labels"]) as src:
            tmp = np.full(shape, 255, np.uint8)
            b = rasterio.warp.transform_bounds(src.crs, f"EPSG:{epsg}", *src.bounds)
            c0, c1 = int((b[0] - x0) / spacing) - 1, int((b[2] - x0) / spacing) + 2
            r0, r1 = int((y0 - b[3]) / spacing) - 1, int((y0 - b[1]) / spacing) + 2
            if c1 < 0 or r1 < 0 or c0 >= shape[1] or r0 >= shape[0]:
                continue
            r0, c0, r1, c1 = max(r0, 0), max(c0, 0), min(r1, shape[0]), min(c1, shape[1])
            win = np.full((r1 - r0, c1 - c0), 255, np.uint8)
            reproject(src.read(1), win, src_transform=src.transform, src_crs=src.crs, src_nodata=255,
                      dst_transform=from_origin(x0 + c0 * spacing, y0 - r0 * spacing, spacing, spacing), dst_crs=f"EPSG:{epsg}",
                      dst_nodata=255, resampling=Resampling.nearest)
            view = mosaic[r0:r1, c0:c1]
            view[win != 255] = win[win != 255]
            del tmp
    np.save(out, mosaic)
    return mosaic


def export_patch_features(data: Path, out: Path, layers: dict[str, np.ndarray], valid_any: np.ndarray, grid, min_cover: float = 0.95) -> int:
    """Warp real-valued season summaries onto each dataset patch grid (bilinear) and save them as <id>.npy.

    Only derived real-valued layers are resampled, never wrapped phase. Patches with less than ``min_cover`` radar
    coverage are skipped; ids.txt lists the exported patches and channels.json the layer order."""
    x0, y0, spacing, epsg = grid
    src_transform = from_origin(x0, y0, spacing, spacing)
    out.mkdir(parents=True, exist_ok=True)
    names = list(layers)
    stack = np.stack([np.where(valid_any, layers[n], np.nan).astype(np.float32) for n in names])
    ids = []
    for line in open(data / "index.jsonl"):
        rec = json.loads(line)
        with rasterio.open(data / rec["labels"]) as src:
            b = rasterio.warp.transform_bounds(src.crs, f"EPSG:{epsg}", *src.bounds)
            if b[2] < x0 or b[0] > x0 + stack.shape[2] * spacing or b[3] < y0 - stack.shape[1] * spacing or b[1] > y0:
                continue
            dst = np.full((len(names), src.height, src.width), np.nan, np.float32)
            reproject(stack, dst, src_transform=src_transform, src_crs=f"EPSG:{epsg}", src_nodata=np.nan,
                      dst_transform=src.transform, dst_crs=src.crs, dst_nodata=np.nan, resampling=Resampling.bilinear)
        if np.isfinite(dst[0]).mean() >= min_cover:
            np.save(out / f"{rec['id']}.npy", dst.astype(np.float16))
            ids.append(rec["id"])
    (out / "ids.txt").write_text("\n".join(ids) + "\n")
    (out / "channels.json").write_text(json.dumps(names))
    return len(ids)


def site_diagnostic(out: Path, name: str, bounds, grid, pairs: list[dict], labels: np.ndarray) -> None:
    """Preview per pair for one site: wrapped phase, coherence, fringe measure and candidate probability."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from rasterio.warp import transform_bounds

    x0, y0, spacing, epsg = grid
    w, s_, e, n = transform_bounds("EPSG:4326", f"EPSG:{epsg}", *bounds)
    pad = 4000
    c0, c1 = int((w - pad - x0) / spacing), int((e + pad - x0) / spacing)
    r0, r1 = int((y0 - n - pad) / spacing), int((y0 - s_ + pad) / spacing)
    if r0 < 0 or c0 < 0 or r1 > labels.shape[0] or c1 > labels.shape[1]:
        return
    sl = (slice(r0, r1), slice(c0, c1))
    if not pairs[0]["valid"][sl].any():
        return
    fig, axes = plt.subplots(len(pairs), 5, figsize=(15, 3.05 * len(pairs)), squeeze=False)
    cmap = ListedColormap(["#c6c5b5", "#2f7d4f", "#8fcf6b", "#3b7fc4", "#dba545", "#9a6aa6", "#ffffff"])
    for row, p in zip(axes, pairs):
        phase = np.where(p["valid"][sl], np.angle(p["ifg"][sl]), np.nan)
        row[0].imshow(phase, cmap="hsv", vmin=-np.pi, vmax=np.pi, interpolation="nearest")
        row[1].imshow(np.where(p["valid"][sl], p["coh"][sl], np.nan), cmap="gray", vmin=0, vmax=1)
        row[2].imshow(np.where(p["valid"][sl], p["fringe"][sl], np.nan), cmap="magma", vmin=0, vmax=2)
        row[3].imshow(p["prob"][sl], cmap="Blues", vmin=0, vmax=1)
        row[4].imshow(np.where(labels[sl] == 255, 6, labels[sl]), cmap=cmap, vmin=0, vmax=6, interpolation="nearest")
        row[0].set_ylabel(f"{p['dates'][0]}\n-> {p['dates'][1]}", fontsize=9)
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
    for ax, t in zip(axes[0], ["Wrapped phase (HH)", "Coherence", "Fringe measure (rad)", "Flooded-vegetation probability", "Reference labels"]):
        ax.set_title(t, fontsize=10)
    fig.suptitle(f"{name}: NISAR 12-day interferograms, {spacing:.0f} m", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    (out / "diagnostics").mkdir(exist_ok=True)
    fig.savefig(out / "diagnostics" / f"{name.lower().replace(' ', '_')}.png", dpi=110)
    plt.close(fig)


def average_precision(score: np.ndarray, positive: np.ndarray) -> float:
    order = np.argsort(-score, kind="stable")
    hits = positive[order]
    return float((np.cumsum(hits)[hits] / (np.flatnonzero(hits) + 1)).mean()) if hits.any() else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gunw", type=Path, required=True, help="folder with NISAR L2 GUNW .h5 files of one track and frame")
    ap.add_argument("--data", type=Path, required=True, help="patch dataset root (for reference labels)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--looks", type=int, default=4)
    ap.add_argument("--draws", type=int, default=30)
    ap.add_argument("--method", default="offset", choices=["offset", "paper"],
                    help="offset: homogeneity and phase offset from the regional background (ours); paper: Canny fringe turnovers joined to homogeneous groups (Hübinger et al. 2026, section 2.2)")
    ap.add_argument("--detrend-km", type=float, default=0.0,
                    help="paper method: remove the regional phase over a window this wide first (0 = as in the paper, no removal)")
    ap.add_argument("--flooded-threshold", type=float, default=0.8, help="paper method: share of draws that must flag a pixel (0.8 in the paper)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--export-patches", type=Path, help="write per-patch season-summary feature arrays here (for the E1 model)")
    ap.add_argument("--per-pair", action="store_true", help="export coherence, homogeneity and fringe of every pair instead of six season summaries")
    ap.add_argument("--sites", default="", help="JSON object {name: [west, south, east, north]} of sites to draw diagnostics for")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in args.gunw.glob("*.h5") if (p.parent / (p.name + ".done")).exists() or not list(p.parent.glob("*.done")))
    rng = np.random.default_rng(args.seed)
    labels, summary, stack = None, {"pairs": [], "looks": args.looks, "draws": args.draws, "method": args.method, "detrend_km": args.detrend_km}, []
    kept, qa = [], []
    for path in files:
        ifg, coh, valid, grid = load_pair(path, args.looks, args.out)
        if labels is None:
            labels = label_mosaic(args.data, ifg.shape, grid, args.out)
            summary["grid"] = {"x0": grid[0], "y0": grid[1], "spacing_m": grid[2], "epsg": grid[3], "shape": list(ifg.shape)}
            summary["labelled_pixels"] = {c: int((labels == k).sum()) for k, c in enumerate(CLASSES)}
        prob, params = paper_probability(ifg, valid, args.draws, rng, int(round(args.detrend_km * 1000 / grid[2]))) if args.method == "paper" else probability(ifg, valid, args.draws, rng, grid[2])
        homogeneity, fringe = phase_measures(ifg, valid, 5, int(round(5000 / grid[2])) | 1)  # fixed mid-range setting, used as features
        kept.append({"ifg": ifg if args.sites else None, "coh": coh, "valid": valid, "homogeneity": homogeneity, "fringe": fringe, "prob": prob, "dates": pair_dates(path)})
        cv = coh[valid]
        qa.append({"file": path.name, "dates": pair_dates(path), "valid_fraction_of_frame_grid": float(valid.mean()),
                   "coherence_percentiles": {str(q): float(np.percentile(cv, q)) for q in (5, 25, 50, 75, 95)},
                   "coherent_area_fraction": {str(t): float((cv >= t).mean()) for t in (0.3, 0.5, 0.7)},
                   "candidate_area_fraction": float((prob[valid] >= 0.5).mean())})
        np.save(args.out / (path.stem + "_prob.npy"), prob.astype(np.float16))
        ok = (labels != 255) & valid
        y, p, c = labels[ok], prob[ok], coh[ok]
        wet = (y == 1) | (y == 2)
        entry = {"file": path.name, "reference_date": pair_dates(path)[0], "secondary_date": pair_dates(path)[1],
                 "valid_share_of_labelled": float(ok.sum() / max((labels != 255).sum(), 1)),
                 "mean_probability": {cl: float(p[y == k].mean()) for k, cl in enumerate(CLASSES)},
                 "flagged_share": {cl: float((p[y == k] >= 0.5).mean()) for k, cl in enumerate(CLASSES)},
                 "mean_coherence": {cl: float(c[y == k].mean()) for k, cl in enumerate(CLASSES)},
                 "wetland_ap_probability": average_precision(p[::7], wet[::7]), "wetland_ap_coherence": average_precision(c[::7], wet[::7]),
                 "wetland_prevalence": float(wet.mean()), "parameters": params}
        summary["pairs"].append(entry)
        stack.append(np.nan_to_num(prob))
        print(f"{entry['reference_date']} -> {entry['secondary_date']}: mean probability "
              + ", ".join(f"{cl[:10]} {entry['mean_probability'][cl]:.3f}" for cl in CLASSES)
              + f" | wetland AP {entry['wetland_ap_probability']:.3f} (coherence alone {entry['wetland_ap_coherence']:.3f}, prevalence {entry['wetland_prevalence']:.3f})", flush=True)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    (args.out / "qa_metrics.json").write_text(json.dumps(qa, indent=1))
    if args.sites:
        for name, bounds in json.loads(args.sites).items():
            site_diagnostic(args.out, name, bounds, grid, kept, labels)
    if args.export_patches and kept:
        valid_any = np.all([k["valid"] for k in kept], 0)
        if args.per_pair:  # continuous measures of every pair: the thresholded probability and season summaries lose most of the class signal
            layers = {f"{name}_{k['dates'][0]}": k[key] for key, name in (("coh", "coherence"), ("homogeneity", "homogeneity"), ("fringe", "fringe")) for k in kept}
        else:
            cohs, fr, pr = np.stack([k["coh"] for k in kept]), np.stack([k["fringe"] for k in kept]), np.stack([np.nan_to_num(k["prob"]) for k in kept])
            layers = {"coherence_mean": cohs.mean(0), "coherence_min": cohs.min(0), "fringe_mean": fr.mean(0), "fringe_max": fr.max(0),
                      "probability_mean": pr.mean(0), "probability_max": pr.max(0)}
        n = export_patch_features(args.data, args.export_patches, layers, valid_any, grid)
        summary["exported_patches"] = n
        print("exported feature arrays for", n, "patches")
    if args.method == "paper" and len(kept) > 1:  # per-date extent and permanent / seasonal / never regime (paper section 2.3)
        valid_all = np.all([k["valid"] for k in kept], 0)
        masks = [np.nan_to_num(k["prob"]) >= args.flooded_threshold for k in kept]
        by_date, regime = flood_regime(masks, [k["dates"] for k in kept], valid_all)
        km2 = (grid[2] / 1000) ** 2
        ok = (labels != 255) & valid_all
        summary["flooded_vegetation"] = {
            "threshold": args.flooded_threshold, "valid_area_km2": float(valid_all.sum() * km2),
            "pair_extent_km2": {f"{k['dates'][0]}_{k['dates'][1]}": float((m & valid_all).sum() * km2) for k, m in zip(kept, masks)},
            "date_extent_km2": {d: float((m & valid_all).sum() * km2) for d, m in by_date.items()},
            "regime_area_km2": {name: float((regime == v).sum() * km2) for v, name in enumerate(["never", "seasonal", "permanent"])},
            "regime_share_by_reference_class": {cl: {name: float((regime[ok & (labels == k)] == v).mean()) for v, name in enumerate(["never", "seasonal", "permanent"])}
                                                for k, cl in enumerate(CLASSES) if (ok & (labels == k)).any()}}
        np.save(args.out / "flood_regime.npy", regime)
        write_tif(args.out / "flood_regime.tif", regime, grid, 255)
        write_tif(args.out / "flooded_probability_mean.tif", np.where(valid_all, np.mean([np.nan_to_num(k["prob"]) for k in kept], 0), np.nan).astype(np.float32), grid, np.nan)
        print("flooded vegetation:", json.dumps(summary["flooded_vegetation"], indent=1), flush=True)
    if len(stack) > 1:  # season-long view: how often a pixel is flagged, and the single strongest pair
        mean = np.mean(stack, 0).astype(np.float32)
        np.save(args.out / "probability_mean.npy", mean.astype(np.float16))
        ok = labels != 255
        y, wet = labels[ok], np.isin(labels[ok], (1, 2))
        summary["season"] = {"mean_probability": {cl: float(mean[ok][y == k].mean()) for k, cl in enumerate(CLASSES)},
                             "wetland_ap_mean_probability": average_precision(mean[ok][::7], wet[::7])}
        print("season mean:", summary["season"])
        (args.out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
