"""Build the self-contained NISAR interferometry experiment report from completed runs.

    cd ml && python build_insar_report.py [--runs ../../runs] [--data ../../s1s2nisar_256_v4] [--web-map ../web/public/map]

Reads only finished artefacts: k-fold ``final.json`` files, out-of-fold prediction maps (``bogsignal.oof_maps``),
the interferogram caches and summaries written by ``bogsignal.insar_flooded`` (both methods), and the website's
regional layers. Writes docs/insar_experiment_report.html and docs/insar_experiment_report_data.json.
Nothing is trained here and the locked test split is not scored.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import html
import io
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Circle, Patch
from pyproj import Transformer
from rasterio.transform import from_bounds, from_origin
from rasterio.warp import Resampling, reproject
from scipy.ndimage import rotate

from bogsignal.insar_flooded import flood_regime, phase_measures

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
SOURCES: list[dict] = []
# same stylesheet as docs/final_experiment_report.html
CSS = "*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f1f4f5;color:#1a2b33;font:16px/1.65 system-ui,-apple-system,Segoe UI,sans-serif}main{max-width:1160px;margin:32px auto;background:white;padding:44px 48px;border-radius:12px}h1{font-size:38px;line-height:1.18;letter-spacing:-1px}h2{font-size:25px;line-height:1.35;margin-top:0}h3{font-size:19px}header{border-bottom:3px solid #20836a;margin-bottom:28px;padding-bottom:16px}.eyebrow{color:#366d64;letter-spacing:.1em;text-transform:uppercase;font-size:12px;font-weight:700}.subtitle{font-size:20px;color:#53646c}.summary{background:#ecf6f1;border-left:5px solid #20836a;padding:24px!important;border-radius:6px}.note{border-left:3px solid #ba913e;background:#fff9e9;padding:12px 16px}nav{display:flex;flex-wrap:wrap;gap:10px 22px;margin:28px 0}a{color:#176b58}section{padding-top:20px;margin:25px 0 38px}table{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}td,th{padding:11px 10px;text-align:left;border-bottom:1px solid #dce5e5;vertical-align:top}th{background:#eaf0f0;font-weight:650}tbody tr:nth-child(even){background:#f7fafa}.table-wrap{overflow-x:auto;margin:16px 0}figure{margin:24px 0}img{display:block;width:100%;height:auto}figcaption,.small{font-size:13px;color:#52646c}figcaption{margin-top:9px}li{margin-bottom:10px}code{font:12px/1.6 ui-monospace,Consolas,monospace;overflow-wrap:anywhere}details{margin:20px 0}summary{cursor:pointer;font-weight:600}@media(max-width:700px){main{margin:0;padding:24px 16px;border-radius:0}h1{font-size:30px}h2{font-size:22px}}@media print{body{background:white;font-size:10pt}main{margin:0;padding:0;max-width:none}nav{display:none}section{break-before:auto}tr,img{break-inside:avoid}h2,h3{break-after:avoid}.table-wrap{overflow:visible}details{display:none}a{color:inherit}}"

SIX = ["Other", "Swamp", "Marsh", "Permanent water", "Peatland cropland", "Peatland non-cropland"]
FIVE = ["Other", "Swamp/marsh", "Permanent water", "Peatland cropland", "Peatland non-cropland"]
TO_FIVE = np.array([0, 1, 1, 2, 3, 4] + [255] * 250, np.uint8)
COLORS = ["#eceee6", "#009e73", "#0072b2", "#e69f00", "#cc79a7"]
CMAP5 = ListedColormap(COLORS + ["#ffffff"])  # 5 = excluded from scoring / no data
VARIANTS = [("base", "Baseline", "Sentinel-2, Sentinel-1 and NISAR backscatter", 14),
            ("pp", "+ InSAR, 27 layers", "Baseline plus coherence, phase homogeneity and phase offset of each of the nine pairs", 41),
            ("coh", "+ coherence, 9 layers", "Baseline plus the coherence of each of the nine pairs", 23),
            ("merge", "One wetland class", "Baseline inputs; Swamp and Marsh trained as one class, scored on the original labels", 14)]
NAME = {v[0]: v[1] for v in VARIANTS}
MON = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH = ["", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
esc = html.escape


def fd(s: str) -> str:
    return f"{int(s[8:])} {MON[int(s[5:7])]}"


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def source(path: Path) -> Path:
    SOURCES.append({"path": os.path.relpath(path, ROOT.parent), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()})
    return path


def read(path: Path):
    return json.loads(source(path).read_text())


def fig_html(fig, caption: str) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f'<figure><img alt="{esc(caption, quote=True)}" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"><figcaption>{esc(caption)}</figcaption></figure>'


def table(head: list[str], rows: list[list[str]]) -> str:
    return ('<div class="table-wrap"><table><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr>"
            + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table></div>")


def class_legend(fig, extra=(), ncol=None, y=-0.01):
    handles = [Patch(facecolor=c, edgecolor="#9aa59c", label=n) for c, n in zip(COLORS, FIVE)] + list(extra)
    fig.legend(handles=handles, loc="lower center", ncol=ncol or len(handles), frameon=False, fontsize=9, bbox_to_anchor=(0.5, y))


def pair_dates(name: str) -> tuple[str, str]:
    d = re.findall(r"_(\d{8})T\d{6}", name)
    return f"{d[0][:4]}-{d[0][4:6]}-{d[0][6:]}", f"{d[2][:4]}-{d[2][4:6]}-{d[2][6:]}"


# ---------------------------------------------------------------------------------------------- feature test

FEATURE_SETS = ["First thresholded detector, mean over pairs", "Six season summaries", "Coherence of each pair (9)", "Coherence, homogeneity and phase offset of each pair (27)"]


def feature_test(run: Path, cache: Path) -> dict:
    """Do the InSAR measures separate the reference classes? Spatially blocked cross-validation on a pixel sample."""
    if cache.exists():
        return read(cache)
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score

    lab = np.load(next(run.glob("labels_*.npy")))
    files = sorted(run.glob("*_ml4.npz"))
    ok = np.all([np.load(f)["valid"] for f in files], 0) & (lab != 255)
    rng = np.random.default_rng(0)
    rc = np.argwhere(ok)[rng.choice(int(ok.sum()), 400000, replace=False)]

    def pick(a):
        return a[rc[:, 0], rc[:, 1]]

    C, Hm, O = [], [], []
    for f in files:  # one interferogram in memory at a time
        z = np.load(f)
        h, o = phase_measures(z["ifg"], z["valid"], 5, 63)  # 5 x 5 window, 5 km background at 80 m
        C.append(pick(z["coh"])); Hm.append(pick(h)); O.append(pick(o))
    C, Hm, O = (np.stack(s, 1) for s in (C, Hm, O))
    y = pick(lab)
    sets = {FEATURE_SETS[0]: None,
            FEATURE_SETS[1]: np.stack([C.mean(1), C.min(1), Hm.mean(1), Hm.min(1), O.mean(1), O.max(1)], 1),
            FEATURE_SETS[2]: C, FEATURE_SETS[3]: np.concatenate([C, Hm, O], 1)}
    thresholded = np.nan_to_num(pick(np.load(run / "probability_mean.npy").astype(np.float32)))
    block = (rc[:, 0] // 250) * 100 + rc[:, 1] // 250  # 20 km blocks
    ub = np.unique(block); rng.shuffle(ub)
    lut = dict(zip(ub.tolist(), (np.arange(len(ub)) % 4).tolist()))
    fold = np.array([lut[b] for b in block.tolist()])
    targets = {"Swamp or marsh": np.isin(y, (1, 2)), "Swamp": y == 1, "Peatland cropland": y == 4, "Permanent water": y == 3}
    out = {"sample_pixels": int(len(y)), "pairs": len(files), "block_km": 20, "folds": 4, "results": {}}
    for tn, t in targets.items():
        out["results"][tn] = {"prevalence": float(t.mean())}
        for sn, X in sets.items():
            if X is None:
                s = thresholded
            else:
                s = np.zeros(len(y))
                for k in range(4):
                    m = HistGradientBoostingClassifier(max_iter=150, class_weight="balanced", random_state=0).fit(X[fold != k], t[fold != k])
                    s[fold == k] = m.predict_proba(X[fold == k])[:, 1]
            out["results"][tn][sn] = {"auc": float(roc_auc_score(t, s)), "ap": float(average_precision_score(t, s))}
            print(f"feature test: {tn} / {sn}: AUC {out['results'][tn][sn]['auc']:.3f}", flush=True)
    coherent = C > 0.4
    out["phase_offset_rad"] = {}
    for k, c in enumerate(SIX):
        v = O[(y == k)[:, None] & coherent]
        out["phase_offset_rad"][c] = {"median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "share_over_quarter_fringe": float((v > np.pi / 2).mean())}
    out["cm_per_rad_line_of_sight"] = 23.8 / (4 * np.pi)  # L-band wavelength 23.8 cm, two-way path
    cache.write_text(json.dumps(out, indent=1))
    return read(cache)


# ---------------------------------------------------------------------------------------------- model runs

METRICS = {"Swamp/marsh average precision": lambda m: m["wetland_ap"], "Swamp average precision": lambda m: m["ap"]["Swamp"], "Marsh average precision": lambda m: m["ap"]["Marsh"],
           "Swamp/marsh recall at 5% area": lambda m: m["wetland_recall_at_area"]["5"], "Swamp/marsh IoU, most likely class": lambda m: m["wetland_iou"],
           "Peatland cropland average precision": lambda m: m["ap"]["Peatland cropland"], "Permanent water average precision": lambda m: m["ap"]["Permanent water"],
           "Six-class macro IoU": lambda m: m["miou"]}
W = "Swamp/marsh average precision"


def model_section(exp: Path, oof: Path):
    present = [v for v in VARIANTS if all((exp / f"insar_{v[0]}_f{k}" / "final.json").exists() for k in range(5))]
    finals = {v[0]: [read(exp / f"insar_{v[0]}_f{k}" / "final.json") for k in range(5)] for v in present}
    result = {v: {n: [float(f(r["last"]["validation"])) for r in rs] for n, f in METRICS.items()} for v, rs in finals.items()}
    rows = []
    for n in METRICS:
        row = [n]
        for v, *_ in present:
            xs = result[v][n]
            cell = pct(np.mean(xs))
            if v != "base":
                up = sum(a > b for a, b in zip(xs, result["base"][n]))
                cell += f'<br><span class="small">{100 * (np.mean(xs) - np.mean(result["base"][n])):+.1f} points; higher in {up} of 5 folds</span>'
            row.append(cell)
        rows.append(row)
    main = table(["Metric, mean of five folds"] + [esc(v[1]) for v in present], rows)
    folds = table([W] + [f"Fold {k + 1}" for k in range(5)] + ["Mean", "Range"],
                  [[esc(v[1])] + [pct(x) for x in result[v[0]][W]] + [pct(np.mean(result[v[0]][W])), f"{100 * min(result[v[0]][W]):.1f}–{100 * max(result[v[0]][W]):.1f}%"] for v in present])
    run_rows = []
    for v in present:
        logs = [[json.loads(line) for line in open(exp / f"insar_{v[0]}_f{k}" / "metrics.jsonl")] for k in range(5)]
        secs = [sum(e.get("seconds", e.get("time", 0)) for e in lg) for lg in logs]
        run_rows.append([esc(v[1]), str(v[3]), "5", ", ".join(str(r["last"]["epoch"]) for r in finals[v[0]]), ", ".join(str(r["best"]["epoch"]) for r in finals[v[0]]),
                         f"{sum(secs) / 60:.1f}" if all(secs) else "not recorded"])
    runs = table(["Variant", "Input channels", "Fold models", "Epochs completed per fold", "Epoch of best.pt per fold (not used for scores)", "Recorded training minutes, all folds"], run_rows)
    mapped = [v for v in present if (oof / v[0] / "offsets.json").exists()]
    return present, mapped, result, main, folds, runs


def decision_and_examples(mapped, oof: Path, data: Path):
    idx = {r["id"]: r for r in map(json.loads, open(data / "index.jsonl")) if "split" in r}
    ids = sorted(p.stem for p in (oof / mapped[0][0]).glob("*.npz"))
    labels: dict[str, np.ndarray] = {}

    def lab(i):
        if i not in labels:
            with rasterio.open(data / idx[i]["labels"]) as d:
                labels[i] = TO_FIVE[d.read(1)]
        return labels[i]

    def pred(v, i, key="cls_tuned"):
        return np.load(oof / v / f"{i}.npz")[key]

    def rgb(i):
        with rasterio.open(data / idx[i]["images"]) as d:
            x = d.read([3, 2, 1]).astype(np.float32)
        return np.moveaxis(np.clip(np.nan_to_num(x) / 0.16, 0, 1) ** 0.8, 0, -1)

    def show(m):
        return np.where(m == 255, 5, m)

    # class areas under both decision rules
    def tally(get):
        c, tp, npred, nref = np.zeros(5), 0, 0, 0
        for i in ids:
            y = lab(i); ok = y != 255; q = get(i)
            c += np.bincount(q[ok], minlength=5)[:5]; tp += ((q == 1) & (y == 1)).sum(); npred += ((q == 1) & ok).sum(); nref += (y == 1).sum()
        return (c / c.sum()).tolist(), float(tp / max(nref, 1)), float(tp / max(npred, 1))

    shares = {"Reference labels": tally(lab)}
    for v, name, *_ in mapped:
        shares[f"{name}: most likely class"] = tally(lambda i, v=v: pred(v, i, "cls"))
        shares[f"{name}: fitted offsets"] = tally(lambda i, v=v: pred(v, i))
    share_table = table(["Decision"] + [f"{c} area" for c in FIVE] + ["Swamp/marsh recall", "Swamp/marsh reference precision"],
                        [[esc(n)] + [pct(x) for x in sh] + (["—", "—"] if n == "Reference labels" else [pct(rc), pct(pc)]) for n, (sh, rc, pc) in shares.items()])
    offsets = {v[0]: read(oof / v[0] / "offsets.json")["offsets_by_fold"] for v in mapped}

    # per-patch swamp/marsh IoU and example selection by rule
    stats = {}
    for i in ids:
        y = lab(i); ok = y != 255; w = (y == 1) & ok
        row = {"reference_share": float(w.sum() / max(ok.sum(), 1))}
        for v, *_ in mapped:
            p = pred(v, i) == 1; u = ((p | w) & ok).sum()
            row[v] = float((p & w).sum() / u) if u else float("nan")
        stats[i] = row
    cand = [i for i in ids if stats[i]["reference_share"] >= 0.05]
    ref = "pp" if any(v[0] == "pp" for v in mapped) else mapped[-1][0]
    delta = {i: stats[i][ref] - stats[i]["base"] for i in cand}
    order = sorted(cand, key=lambda i: delta[i])
    picks = [("Largest gain", order[-1]), ("Second largest gain", order[-2]), ("Smallest change", min(cand, key=lambda i: (abs(delta[i]), -stats[i]["reference_share"]))),
             ("Second largest loss", order[1]), ("Largest loss", order[0])]
    ncol = 2 + len(mapped)
    fig, axes = plt.subplots(len(picks), ncol, figsize=(2.55 * ncol, 2.72 * len(picks)), squeeze=False)
    for r, (title, i) in enumerate(picks):
        axes[r][0].imshow(rgb(i)); axes[r][1].imshow(show(lab(i)), cmap=CMAP5, vmin=0, vmax=5, interpolation="nearest")
        axes[r][0].set_ylabel(f"{title}\n{i}", fontsize=8)
        for c, (v, *_) in enumerate(mapped):
            axes[r][2 + c].imshow(pred(v, i), cmap=CMAP5, vmin=0, vmax=5, interpolation="nearest")
            axes[r][2 + c].set_xlabel(f"swamp/marsh IoU {100 * stats[i][v]:.0f}%", fontsize=8)
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])
    for ax, t in zip(axes[0], ["Sentinel-2 RGB", "Reference labels"] + [v[1] for v in mapped]):
        ax.set_title(t, fontsize=9)
    class_legend(fig); fig.tight_layout(rect=(0, 0.025, 1, 1))
    examples = fig_html(fig, "Same five locations, same reference labels: held-out predictions of every variant with the fitted-offset decision. White pixels are excluded from scoring.")

    # largest gap-free, roughly square block of patches, weighted towards reference swamp/marsh
    rows = np.array([idx[i]["row"] for i in ids]) // 256; cols = np.array([idx[i]["col"] for i in ids]) // 256
    r0, c0 = int(rows.min()), int(cols.min())
    G = np.zeros((rows.max() - r0 + 1, cols.max() - c0 + 1), bool); G[rows - r0, cols - c0] = True
    Wt = np.zeros(G.shape); Wt[rows - r0, cols - c0] = [stats[i]["reference_share"] for i in ids]
    at = {(int(r), int(c)): i for i, r, c in zip(ids, rows - r0, cols - c0)}
    best, hgt = None, np.zeros(G.shape[1], int)
    for r in range(G.shape[0]):
        hgt = np.where(G[r], hgt + 1, 0)
        for c in range(G.shape[1]):
            h = int(hgt[c])
            for c2 in range(c, G.shape[1]):
                h = min(h, int(hgt[c2]))
                if h == 0:
                    break
                w = c2 - c + 1
                if min(h, w) < 4 or min(h, w) / max(h, w) < 0.6:
                    continue
                key = (h * w * (0.02 + Wt[r - h + 1:r + 1, c:c2 + 1].mean()), h * w)
                if best is None or key > best[0]:
                    best = (key, r - h + 1, c, h, w)
    _, br, bc, bh, bw = best
    S, n = 4, 64

    def mosaic(get, colour=False):
        m = np.zeros((bh * n, bw * n) + ((3,) if colour else ()), np.float32 if colour else np.uint8)
        for r in range(bh):
            for c in range(bw):
                m[r * n:(r + 1) * n, c * n:(c + 1) * n] = get(at[(br + r, bc + c)])[::S, ::S]
        return m

    def block_iou(v, key):
        inter = union = 0
        for r in range(bh):
            for c in range(bw):
                i = at[(br + r, bc + c)]; y = lab(i); ok = y != 255; p = pred(v, i, key) == 1; w = y == 1
                inter += (p & w & ok).sum(); union += ((p | w) & ok).sum()
        return float(inter / max(union, 1))

    panels = [("Sentinel-2 RGB", mosaic(rgb, True), None), ("Reference labels", show(mosaic(lab)), None),
              (f"{mapped[0][1]}: most likely class", mosaic(lambda i: pred(mapped[0][0], i, "cls")), block_iou(mapped[0][0], "cls"))]
    panels += [(f"{name}: fitted offsets", mosaic(lambda i, v=v: pred(v, i)), block_iou(v, "cls_tuned")) for v, name, *_ in mapped]
    nc = 4 if len(panels) > 6 else 3 if len(panels) > 4 else len(panels); nr = -(-len(panels) // nc)
    fig, axes = plt.subplots(nr, nc, figsize=(3.7 * nc, (3.7 * bh / bw + 0.55) * nr), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (t, m, iou) in zip(axes.ravel(), panels):
        ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
        if m.ndim == 3:
            ax.imshow(m)
        else:
            ax.imshow(m, cmap=CMAP5, vmin=0, vmax=5, interpolation="nearest")
        ax.set_title(t + (f"\nswamp/marsh IoU {100 * iou:.0f}%" if iou is not None else ""), fontsize=9)
    class_legend(fig, y=-0.02); fig.tight_layout(rect=(0, 0.03, 1, 1))
    block_info = {"patches": int(bh * bw), "width_km": bw * 5.12, "height_km": bh * 5.12, "reference_swamp_marsh_share": float(Wt[br:br + bh, bc:bc + bw].mean()),
                  "swamp_marsh_iou": {t: iou for t, _, iou in panels if iou is not None}}
    block = fig_html(fig, f"One continuous block of {bh * bw} neighbouring patches, {bw * 5.12:.0f} by {bh * 5.12:.0f} km, shown at 80 m. Every patch is predicted by the fold model that did not train on it.")
    info = {"class_areas": shares, "offsets_by_fold": offsets, "example_rule_variant": ref, "example_patches": [{"rule": t, "id": i, **stats[i]} for t, i in picks], "block": block_info}
    return share_table, examples, block, info


# ---------------------------------------------------------------------------------------------- fringe method

def upright_angle(valid: np.ndarray) -> float:
    """Rotation (degrees, scipy.ndimage.rotate convention) that turns the tilted radar footprint upright."""
    small = valid[::32, ::32].astype(np.uint8)

    def fill(a):
        r = rotate(small, a, reshape=True, order=0, cval=0, prefilter=False)
        rr, cc = np.where(r)
        return r.sum() / ((np.ptp(rr) + 1) * (np.ptp(cc) + 1))

    coarse = max(np.arange(-45, 45.5, 1.0), key=fill)
    return float(max(np.arange(coarse - 1, coarse + 1.01, 0.1), key=fill))


def north_vector(angle: float) -> tuple[float, float]:
    """Direction of grid north in a rotated image, as (x to the right, y downwards), found with a probe."""
    probe = np.zeros((201, 201), np.uint8); probe[98:103, 98:103] = 1; probe[18:23, 98:103] = 2
    r = rotate(probe, angle, reshape=True, order=0, cval=0, prefilter=False)
    d = np.argwhere(r == 2).mean(0) - np.argwhere(r == 1).mean(0)
    return float(d[1] / np.hypot(*d)), float(d[0] / np.hypot(*d))


def north_arrow(ax, shape, vec) -> None:
    h, w = shape[:2]; L = 0.055 * min(h, w); x0, y0 = w - 2.6 * L, 2.6 * L
    ax.add_patch(Circle((x0, y0), 2.1 * L, facecolor="white", edgecolor="none", alpha=0.85, zorder=5))
    ax.annotate("", xy=(x0 + 1.2 * L * vec[0], y0 + 1.2 * L * vec[1]), xytext=(x0 - 1.5 * L * vec[0], y0 - 1.5 * L * vec[1]),
                arrowprops=dict(arrowstyle="-|>", color="#16332b", lw=1.6), zorder=6)
    ax.text(x0 - 0.9 * L * vec[1], y0 + 0.9 * L * vec[0], "N", ha="center", va="center", fontsize=8, color="#16332b", zorder=6)


def fringe_section(run_paper: Path, run_detrend: Path, run_base: Path, web_map: Path):
    from PIL import Image

    S = 2  # figures at 160 m
    lab = np.load(next(run_paper.glob("labels_*.npy")))
    files = sorted(run_paper.glob("*_prob.npy"), key=lambda p: pair_dates(p.name)[0])
    ds = [pair_dates(p.name) for p in files]
    probs = [np.load(p).astype(np.float32) for p in files]
    valid = np.all([np.isfinite(p) for p in probs], 0)
    masks = [np.nan_to_num(p) >= 0.8 for p in probs]
    by_date, regime = flood_regime(masks, ds, valid)
    summary = read(run_paper / "summary.json")
    ok = valid & (lab != 255)
    names3 = ((2, "every_date"), (1, "some_dates"), (0, "never"))
    by_class = {SIX[k]: {"flagged_mean": float(np.mean([m[ok & (lab == k)].mean() for m in masks])), **{n: float((regime[ok & (lab == k)] == v).mean()) for v, n in names3}} for k in range(6)}
    km2 = (summary["grid"]["spacing_m"] / 1000) ** 2
    area = {n: float((regime == v).sum() * km2) for v, n in names3}
    per_pair = [{"reference": a, "secondary": b, "days": int((np.datetime64(b) - np.datetime64(a)).astype(int)), "flagged_share": float(m[valid].mean())} for (a, b), m in zip(ds, masks)]

    detrend = None
    dfiles = sorted(run_detrend.glob("*_prob.npy")) if run_detrend.exists() else []
    if dfiles and (run_detrend / "summary.json").exists():
        dv = [np.load(p).astype(np.float32) for p in dfiles]
        detrend = {"pairs": len(dfiles), "km": read(run_detrend / "summary.json").get("detrend_km"),
                   "flagged_share": [float((np.nan_to_num(p)[np.isfinite(p)] >= 0.8).mean()) for p in dv],
                   "any_run_share": [float((np.nan_to_num(p)[np.isfinite(p)] > 0).mean()) for p in dv]}
        del dv

    # the website's regional layers (reference labels, model D, analysed area) on the radar grid
    man = read(web_map / "manifest.json"); (la0, lo0), (la1, lo1) = man["bounds"]
    t = Transformer.from_crs(4326, 3857, always_xy=True)
    (mx0, my0), (mx1, my1) = t.transform(lo0, la0), t.transform(lo1, la1)
    src_tr = from_bounds(mx0, my0, mx1, my1, man["width"], man["height"])
    g = summary["grid"]

    def web(name, colours):
        a = np.array(Image.open(source(web_map / f"{name}.png")).convert("RGBA")); code = np.zeros(a.shape[:2], np.uint8)
        for rgb, k in colours.items():
            code[(a[..., :3] == rgb).all(-1) & (a[..., 3] > 0)] = k
        dst = np.zeros(regime.shape, np.uint8)
        reproject(code, dst, src_transform=src_tr, src_crs="EPSG:3857", dst_transform=from_origin(g["x0"], g["y0"], g["spacing_m"], g["spacing_m"]), dst_crs=f"EPSG:{g['epsg']}", resampling=Resampling.nearest)
        return dst

    WEBC = {(255, 220, 0): 1, (40, 121, 185): 2, (207, 154, 75): 3, (139, 102, 139): 4}
    w_pred, w_lab, w_cov = web("prediction", WEBC), web("labels", WEBC), web("coverage", {(46, 116, 128): 1, (80, 80, 80): 2})
    w_pred = np.where(w_cov == 1, w_pred, 5).astype(np.uint8)
    against_model = {FIVE[k]: float((regime[valid & (w_pred == k)] == 2).mean()) for k in range(5) if (valid & (w_pred == k)).any()}

    angle = upright_angle(valid); vec = north_vector(angle)
    v_r = rotate(valid[::S, ::S].astype(np.uint8), angle, reshape=True, order=0, cval=0, prefilter=False)
    rr, cc = np.where(v_r); pad = int(0.012 * v_r.shape[0])
    frame = (int(np.percentile(rr, 0.5)) + pad, int(np.percentile(rr, 99.5)) - pad, int(np.percentile(cc, 0.5)) + pad, int(np.percentile(cc, 99.5)) - pad)

    def T(a, fill):
        return rotate(a[::S, ::S], angle, reshape=True, order=0, cval=fill, prefilter=False)[frame[0]:frame[1], frame[2]:frame[3]]

    r_reg, r_lab, r_pred, r_cov = T(regime, 255), T(w_lab, 0), T(w_pred, 5), T(w_cov, 0)
    cr, cc2 = np.where(r_cov == 1)
    box = (int(np.percentile(cr, 1)), int(np.percentile(cr, 99)), int(np.percentile(cc2, 1)), int(np.percentile(cc2, 99)))

    def crop(a):
        return a[box[0]:box[1], box[2]:box[3]]

    REG = ListedColormap(["#fafaf6", "#7fb2e0", "#124e96", "#d6dbd1"])
    CM6 = ListedColormap(COLORS + ["#d6dbd1"])
    hh, ww = box[1] - box[0], box[3] - box[2]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 13.5 / 3 * hh / ww + 1.1))
    axes[0].imshow(np.where(crop(r_reg) == 255, 3, crop(r_reg)), cmap=REG, vmin=0, vmax=3, interpolation="nearest"); axes[0].set_title("Fringe method: water under vegetation", fontsize=10)
    axes[1].imshow(crop(r_lab), cmap=CM6, vmin=0, vmax=5, interpolation="nearest"); axes[1].set_title("Reference labels", fontsize=10)
    axes[2].imshow(crop(r_pred), cmap=CM6, vmin=0, vmax=5, interpolation="nearest"); axes[2].set_title("Model D, all three sensors", fontsize=10)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); north_arrow(ax, (hh, ww), vec)
    class_legend(fig, extra=[Patch(facecolor="#d6dbd1", label="Not analysed by model D"), Patch(facecolor="#124e96", label="Flagged at every date"), Patch(facecolor="#7fb2e0", label="Flagged at some dates"),
                             Patch(facecolor="#fafaf6", edgecolor="#9aa59c", label="Never flagged")], ncol=5, y=-0.03)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    three = fig_html(fig, f"The same {ww * S * g['spacing_m'] / 1000:.0f} km wide area in three views: fringe method, reference labels and model D. Maps are turned {abs(angle):.0f} degrees so the tilted radar footprint fills the frame; the arrow points to grid north.")

    BLUE = LinearSegmentedColormap.from_list("b", ["#eceee6", "#7fb2e0", "#124e96"])
    fig, axes = plt.subplots(1, len(probs), figsize=(1.55 * len(probs), 2.15), squeeze=False)
    for ax, p, (a, b), m in zip(axes[0], probs, ds, masks):
        ax.imshow(T(np.nan_to_num(p), 0)[::2, ::2], cmap=BLUE, vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{fd(a)} – {fd(b)}\n{100 * m[valid].mean():.0f}% flagged", fontsize=8)
    fig.tight_layout()
    strip = fig_html(fig, "Share of the 100 runs that flagged each pixel, whole radar frame, one panel per interferogram. Darker blue is a higher share.")
    del probs

    # what the same interferograms do carry: coherence
    coh_files = sorted(run_base.glob("*_ml4.npz"))
    coh_mean = np.mean([np.load(f)["coh"] for f in coh_files], 0)
    base = read(run_base / "summary.json")
    by = {e["reference_date"]: e["mean_coherence"] for e in base["pairs"]}
    wpx = base["labelled_pixels"]
    grp = {"Swamp + Marsh": ["Swamp", "Marsh"], "Peatland non-cropland": ["Peatland non-cropland"], "Peatland cropland": ["Peatland cropland"], "Permanent water": ["Permanent water"], "Other": ["Other"]}
    dates = sorted(by)
    series = {n: [sum(by[d][c] * wpx[c] for c in cl) / sum(wpx[c] for c in cl) for d in dates] for n, cl in grp.items()}
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13.5, 5.3), gridspec_kw={"width_ratios": [1, 1.3]})
    CH = LinearSegmentedColormap.from_list("c", ["#3d0d0a", "#8f2a14", "#d9722c", "#f0c987", "#f4f1e8"])
    cm_img = np.ma.masked_where(T(valid.astype(np.uint8), 0) == 0, T(coh_mean, 0))
    im = a0.imshow(cm_img, cmap=CH, vmin=0.15, vmax=0.75); a0.set_xticks([]); a0.set_yticks([]); north_arrow(a0, cm_img.shape, vec)
    fig.colorbar(im, ax=a0, fraction=0.046, pad=0.02, label="Mean coherence")
    a0.set_title(f"Mean coherence of {len(coh_files)} pairs, whole frame", fontsize=10)
    for (n, v), col, ls in zip(series.items(), ["#009e73", "#cc79a7", "#e69f00", "#0072b2", "#8a958c"], ["-", "--", ":", "-.", "-"]):
        a1.plot(range(len(v)), v, color=col, ls=ls, lw=2, marker="o", ms=4, label=n)
    a1.set_xticks(range(len(dates))); a1.set_xticklabels([fd(d) for d in dates], fontsize=8); a1.set_ylabel("Mean coherence"); a1.grid(axis="y", color="#dce5e5"); a1.set_axisbelow(True)
    for sp in ("top", "right"):
        a1.spines[sp].set_visible(False)
    a1.legend(frameon=False, fontsize=9, ncol=2); a1.set_title("Mean coherence by reference class; dates mark the first pass of each pair", fontsize=10)
    fig.tight_layout()
    coh_fig = fig_html(fig, "Coherence between consecutive NISAR passes. Left: frame mean, turned upright; dark means the surface changed between passes. Right: mean per reference class and pair.")

    info = {"files": [e["file"] for e in summary["pairs"]], "pairs": per_pair, "runs_per_pair": summary["draws"], "threshold": 0.8, "area_km2": area, "by_reference_class": by_class,
            "every_date_share_by_model_d_class": against_model, "inferred_dates": list(by_date), "regional_phase_removed": detrend, "rotation_degrees": angle,
            "swamp_marsh_ap_per_pair": [e["wetland_ap_probability"] for e in summary["pairs"]], "swamp_marsh_prevalence": summary["pairs"][0]["wetland_prevalence"],
            "coherence": {"dates": dates, "mean_by_class": series, "class_mean": {n: float(np.mean(v)) for n, v in series.items()}}}
    return three, strip, coh_fig, info


# ---------------------------------------------------------------------------------------------- report

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=ROOT.parent / "runs")
    ap.add_argument("--data", type=Path, default=ROOT.parent / "s1s2nisar_256_v4")
    ap.add_argument("--web-map", type=Path, default=ROOT / "web" / "public" / "map")
    ap.add_argument("--tests-passed", type=int, help="number of repository tests that passed, stated in the report")
    ap.add_argument("--fragment", type=Path, help="also write the report without the html/head/body wrapper")
    args = ap.parse_args()
    e1 = args.runs / "insar_e1"
    ft = feature_test(e1 / "track100_frame28", e1 / "feature_test.json")
    present, mapped, result, main_table, fold_table, run_table = model_section(args.runs / "exp", e1 / "oof")
    share_table, examples, block, dec = decision_and_examples(mapped, e1 / "oof", args.data)
    three, strip, coh_fig, fr = fringe_section(e1 / "track100_frame28_paper", e1 / "track100_frame28_paper_detrend5", e1 / "track100_frame28", args.web_map)

    have = {v[0] for v in present}
    missing = [v[1] for v in VARIANTS if v[0] not in have]
    mean = lambda v, n=W: float(np.mean(result[v][n]))  # noqa: E731
    ups = lambda v, n=W: sum(a > b for a, b in zip(result[v][n], result["base"][n]))  # noqa: E731
    SW = "Swamp average precision"
    tot = sum(fr["area_km2"].values())
    cm = ft["cm_per_rad_line_of_sight"]; sw = ft["phase_offset_rad"]["Swamp"]
    ftr = ft["results"]["Swamp or marsh"]
    areas = dec["class_areas"]
    today = dt.date.today()

    findings = [f"<strong>Interferometric coherence added useful weak-reference signal for swamp.</strong> Adding the InSAR layers to the 14-channel U-Net raised swamp/marsh average precision from {pct(mean('base'))} to {pct(mean('pp'))} "
                f"({100 * (mean('pp') - mean('base')):+.1f} points; higher in {ups('pp')} of 5 folds) and swamp average precision from {pct(mean('base', SW))} to {pct(mean('pp', SW))} (higher in {ups('pp', SW)} of 5 folds). Marsh and permanent water did not improve."]
    if "coh" in have:
        findings.append(f"<strong>Coherence alone did at least as well as all 27 InSAR layers.</strong> Nine coherence layers gave {pct(mean('coh'))} swamp/marsh average precision against {pct(mean('pp'))} for the 27-layer variant; phase homogeneity and phase offset added little beyond it.")
    if "merge" in have:
        findings.append(f"<strong>Training Swamp and Marsh as one class did not improve the ranking.</strong> Swamp/marsh average precision was {pct(mean('merge'))} against {pct(mean('base'))} for the baseline (higher in {ups('merge')} of 5 folds). "
                        f"It raised the IoU of the most-likely-class map from {pct(mean('base', 'Swamp/marsh IoU, most likely class'))} to {pct(mean('merge', 'Swamp/marsh IoU, most likely class'))} (higher in {ups('merge', 'Swamp/marsh IoU, most likely class')} of 5 folds), mainly by drawing more wetland. Its Marsh score is not meaningful, because Marsh is no longer a separate training class.")
    findings += [
        f"<strong>The fringe method of Hübinger et al. (2026) did not produce a wetland map from 12-day summer pairs.</strong> It flagged {pct(fr['area_km2']['every_date'] / tot, 0)} of the radar frame as water under vegetation at every inferred date. "
        f"Reference Swamp was flagged as often as Other ({pct(fr['by_reference_class']['Swamp']['flagged_mean'], 0)} against {pct(fr['by_reference_class']['Other']['flagged_mean'], 0)}); only Permanent water differed ({pct(fr['by_reference_class']['Permanent water']['flagged_mean'], 0)}). "
        f"The method needs a water-level change of a full fringe, about {2 * np.pi * cm:.0f} cm along the line of sight; the median local phase change over coherent Swamp was {sw['median']:.2f} rad, about {sw['median'] * cm:.1f} cm.",
        f"<strong>The most-likely-class map under-draws every class except Other.</strong> It assigns {pct(areas['Baseline: most likely class'][0][0])} of scored pixels to Other against {pct(areas['Reference labels'][0][0])} in the reference. "
        f"Cross-fitted per-class offsets restore the class areas and raise baseline swamp/marsh recall from {pct(areas['Baseline: most likely class'][1])} to {pct(areas['Baseline: fitted offsets'][1])}, with reference precision falling from {pct(areas['Baseline: most likely class'][2])} to {pct(areas['Baseline: fitted offsets'][2])}.",
    ]
    recommendation = ("<strong>Recommendation:</strong> keep coherence as the candidate InSAR input for the next model version" + (", in its nine-layer form" if "coh" in have else "")
                      + ". Do not use the fringe method on 12-day pairs; retest it only on pairs at least 36 days apart or across the spring flood. Confirm the swamp gain with more seeds and on further radar frames before changing the published model D.")
    caveat = "These are five-fold validation results against historical/inferred weak labels, not independent contemporary wetland accuracy. Every variant uses one training seed and the 1,061 train and validation patches inside one radar frame. The test set remains unscored."

    design = [["Interferograms", "NISAR L2 GUNW, provisional release; L-band, HH; track 100, frame 28; nine consecutive pairs, 21 June to 19 October 2026; eight span 12 days and one 24"],
              ["Interferogram processing", "Wrapped phase and coherence averaged 4 × 4 from 20 m to 80 m; pixels need at least half of the looks valid"],
              ["Base dataset", "V4: 3,753 patches; 256 × 256 pixels; 20 m; EPSG:3035"],
              ["Patch pool", "1,087 patches with at least 95% interferogram coverage in every pair: 1,061 train and validation patches used, 26 locked test patches not scored"],
              ["Base channels", "Sentinel-2 B2, B3, B4, B8, B11, B12; S1 ascending/descending VV, VH; NISAR ascending/descending HH, HV backscatter"],
              ["InSAR layers", "Per pair: coherence; local phase homogeneity (length of the mean unit phasor, 5 × 5 window); absolute phase offset from a 5 km background. Warped bilinearly to each patch grid; missing values become zero after normalization"],
              ["Model", "U-Net, ImageNet-pretrained ResNet-34 encoder; six logits; first convolution sized to the input channels"],
              ["Optimization", "Seed 0; batch 16; learning rate 3 × 10⁻⁴; weight decay 10⁻⁴; cross-entropy plus Dice, ignore 255; 20 epochs; weight EMA 0.998"],
              ["Evaluation", "Five geographic folds over connected wetland blocks, identical for every variant; training patches touching a held-out patch are dropped. Scores use the final-epoch weights (<code>last.pt</code>, no checkpoint selection) with horizontal-flip averaging"]]
    variant_rows = [[esc(v[1]), esc(v[2]), str(v[3]), "Complete, five folds" if v[0] in have else "Not complete when this report was built"] for v in VARIANTS]
    ft_rows = [[esc(s)] + [f"{ft['results'][t][s]['auc']:.3f} / {pct(ft['results'][t][s]['ap'])}" for t in ft["results"]] for s in FEATURE_SETS]
    ph_rows = [[esc(c), f"{v['median']:.2f} rad / {v['median'] * cm:.1f} cm", f"{v['p90']:.2f} rad / {v['p90'] * cm:.1f} cm", pct(v["share_over_quarter_fringe"], 2)] for c, v in ft["phase_offset_rad"].items()]
    pair_rows = [[f"{fd(p['reference'])} – {fd(p['secondary'])} 2026", str(p["days"]), pct(p["flagged_share"]), pct(a)] for p, a in zip(fr["pairs"], fr["swamp_marsh_ap_per_pair"])]
    cls_rows = [[esc(c), pct(v["flagged_mean"]), pct(v["every_date"]), pct(v["some_dates"]), pct(v["never"])] for c, v in fr["by_reference_class"].items()]
    md = fr["every_date_share_by_model_d_class"]
    det = fr["regional_phase_removed"]
    det_text = (f"Removing the regional phase over a {det['km']:.0f} km window before fringe detection ({det['pairs']} of {len(fr['pairs'])} interferograms processed) left " + ("no land" if max(det['flagged_share']) == 0 else f"{pct(max(det['flagged_share']), 2)} or less of land") + f" flagged in at least 80 of 100 runs, "
                f"while {pct(min(det['any_run_share']), 0)} to {pct(max(det['any_run_share']), 0)} was flagged in at least one run: single runs still mark almost everything or nothing, and no stable local detections remain."
                if det else "The variant that removes the regional phase first had not produced results when this report was built.")
    off_rows = [[esc(NAME[k])] + [f"{a:g}" for a in np.mean(list(o.values()), 0)] for k, o in dec["offsets_by_fold"].items()]
    ex_ids = ", ".join(f"<code>{p['id']}</code> ({p['rule'].lower()})" for p in dec["example_patches"])
    tests = f" The repository test suite passed <strong>{args.tests_passed} tests</strong>, including the fringe workflow on synthetic interferograms." if args.tests_passed else ""

    body = f"""<header><p class="eyebrow">BogSignal / Argos · {today.day} {MONTH[today.month]} {today.year}</p><h1>Wetland segmentation:<br>NISAR interferometry experiments</h1><p class="subtitle">Report for the completed seed-0, five-fold InSAR experiments</p></header>
<section class="summary"><h2>Executive findings</h2>{''.join(f'<p>{f}</p>' for f in findings)}<p>{recommendation}</p><p class="note">{caveat}</p></section>
<nav aria-label="Report contents"><a href="#design">Design</a><a href="#features">Feature test</a><a href="#models">InSAR layers</a><a href="#decision">Decision rule</a><a href="#examples">Examples</a><a href="#fringe">Fringe method</a><a href="#coherence">Coherence</a><a href="#limits">Limits</a><a href="#artifacts">Artifacts</a></nav>
<section id="design"><h2>1. Study question and fixed design</h2><p>The earlier A–D experiments used NISAR backscatter. This study adds NISAR interferograms, which record for two passes how the radar phase changed and how alike the passes are (coherence). The objective was to assess whether measures derived from interferograms improve the six-class U-Net beyond its 14 input channels, and whether the unsupervised fringe workflow of Hübinger, Fluet-Chouinard, Escobar and Jaramillo (Remote Sensing of Environment 332, 115086) maps water under vegetation in Polissia.</p>
{table(['Item', 'Specification'], design)}{table(['Variant', 'Inputs and supervision', 'Input channels', 'Status'], variant_rows)}
<p>Normalization is fitted on each fold's training patches only. The same fold assignment, seed and recipe are used for every variant, so differences come from the inputs or the supervision named in the table.</p>
<h3>How to read the metrics</h3><p>For combined wetland results, the Swamp and Marsh probabilities are summed first. The main tables rank pixels by that probability; the hard class maps are treated separately in section 4.</p><ul>
<li><strong>Average precision:</strong> quality of the ranking of pixels by swamp/marsh probability. With {pct(fr['swamp_marsh_prevalence'])} reference prevalence, a random ranking scores about that value.</li>
<li><strong>Recall at 5% area:</strong> fraction of reference swamp/marsh pixels recovered by inspecting the highest-scoring 5% of the scored area.</li>
<li><strong>IoU:</strong> overlap of a hard class map with the reference. Recall and reference precision in section 4 describe the same maps.</li>
<li><strong>AUC</strong> (feature test only): ranking quality where 50% is chance.</li>
<li><strong>Higher in k of 5 folds:</strong> number of folds in which a variant beat the baseline. Folds differ far more from each other than variants do within a fold.</li></ul></section>
<section id="features"><h2>2. Do the InSAR measures separate the reference classes?</h2><p>Before any training, <strong>{ft['sample_pixels']:,} labelled pixels</strong> were sampled from the frame and each class was ranked from InSAR measures alone, using gradient-boosted trees with four-fold cross-validation over {ft['block_km']} km blocks. The first row is the earlier hand-set threshold on homogeneity and phase offset, with no learning. Cells show AUC / average precision.</p>
{table(['InSAR input'] + [f"{esc(t)}<br><span class='small'>prevalence {pct(ft['results'][t]['prevalence'])}</span>" for t in ft['results']], ft_rows)}
<p>The thresholded detector is close to chance for every class (swamp/marsh AUC {ftr[FEATURE_SETS[0]]['auc']:.2f}). Season summaries recover part of the signal and keeping each pair separate recovers most of it ({ftr[FEATURE_SETS[3]]['auc']:.2f}). Coherence by itself carries the larger part ({ftr[FEATURE_SETS[2]]['auc']:.2f}). This motivated passing per-pair layers to the U-Net instead of a thresholded map.</p></section>
<section id="models"><h2>3. U-Net with InSAR layers: five geographic folds</h2><p>Each variant was trained five times, each time holding out one geographic fold of the patch pool. Cells show the mean over the five held-out folds; the second line compares with the baseline.</p>
{main_table}<p>The folds are not interchangeable. The table below shows the headline metric for every fold.</p>{fold_table}
<p>The swamp gain of the InSAR variants appears in most folds; the other differences are within the fold-to-fold spread and should remain inconclusive.{(' Not complete when this report was built: ' + ', '.join(esc(m) for m in missing) + '.') if missing else ''}</p></section>
<section id="decision"><h2>4. Decision rule: most likely class versus fitted offsets</h2><p>Choosing the single most likely class per pixel under-draws every class except Other, because Other is far more common and is chosen whenever the model is unsure. The training code's <code>tune_bias</code> adds one offset per class to the log-probabilities before the choice. Here the offsets are <strong>cross-fitted</strong>: those applied to a fold's patches were fitted on the held-out predictions of the other four folds, so no patch helped choose its own rule.</p>
{share_table}{table(['Mean fitted offset'] + FIVE, off_rows)}
<p>The fitted offsets bring each class area close to the reference. They recover more reference swamp/marsh and confirm a smaller share of what they map; the choice depends on whether missing wetland or mapping too much is the costlier error. Peatland stays the weakest class under both rules.</p></section>
<section id="examples"><h2>5. Matched prediction examples</h2><p>The five patches are selected by rule among held-out patches with at least 5% reference swamp/marsh: the two largest gains and the two largest losses in swamp/marsh IoU of “{esc(NAME[dec['example_rule_variant']])}” over the baseline, and the smallest change. They are illustrations chosen by performance difference, not a representative accuracy sample. All predictions use the fitted-offset decision and the same fixed RGB display stretch.</p>
{examples}<p class="small">Patch IDs: {ex_ids}.</p>
<p>The block below is the largest roughly square rectangle of neighbouring patches without gaps, weighted towards reference swamp/marsh ({pct(dec['block']['reference_swamp_marsh_share'], 0)} of its pixels). The dataset samples patches and does not cover the region wall to wall, so a view of the whole pool has holes. Neighbouring patches can come from different fold models, which leaves visible seams.</p>{block}</section>
<section id="fringe"><h2>6. The fringe method of Hübinger et al. on 12-day pairs</h2><p>The published workflow (sections 2.2–2.3 of the paper) was implemented as <code>--method paper</code>: Canny edge detection on the Gaussian-smoothed wrapped phase finds fringe turnovers; fringe lines are buffered by three pixels; homogeneous-phase regions come from a local variance filter with a morphological clean-up; only connected homogeneous regions that touch a fringe are kept. This is repeated <strong>{fr['runs_per_pair']} times</strong> per interferogram with random parameters. Pixels flagged in at least {pct(fr['threshold'], 0)} of runs form the pair's mask, consecutive pairs are joined to give the extent at each date, and pixels are sorted into flagged at every date, at some dates, or never. The method uses no labels.</p>
<p class="note">Two choices are ours because the paper's main text leaves them open: the parameter ranges (the paper's are in its supplementary Table S2) and a circular form of the local variance, so that the −π/π wrap itself does not count as noise. Canny thresholds are set as a share of the response of a full turnover.</p>
{three}<p>“Never flagged” is not “dry”: open water returns no stable phase, so the method cannot flag it. Reference labels and model D in this figure are the website's regional layers at about 340 m, warped to the radar grid.</p>
{table(['Interferogram', 'Days apart', 'Land flagged in at least 80 of 100 runs', 'Swamp/marsh average precision of the flag frequency'], pair_rows)}{strip}
{table(['Reference class', 'Flagged, mean of pairs', 'Flagged at every date', 'At some dates', 'Never'], cls_rows)}
<p>Over the {len(fr['inferred_dates'])} inferred dates, <strong>{fr['area_km2']['every_date']:,.0f} of {tot:,.0f} km²</strong> were flagged at every date. Where model D predicts swamp/marsh, the method flags every date for {pct(md.get('Swamp/marsh', float('nan')), 0)} of pixels; where model D predicts Other, for {pct(md.get('Other', float('nan')), 0)}. {det_text}</p>
<h3>Why the method has nothing to detect here</h3>
{table(['Reference class', 'Median local phase change', '90th percentile', 'Pixels above a quarter fringe'], ph_rows)}
<p>Local phase change is the absolute offset from a 5 km background on pixels with coherence above 0.4; one radian is {cm:.1f} cm along the line of sight. Without local fringes, the only turnovers are broad atmospheric and orbital ramps. Over 12 days nearly all land stays coherent and forms one connected homogeneous region, so a single such turnover marks all of it. The paper used pairs 46 to 219 days apart over a tropical floodplain, where non-flooded forest loses coherence and water levels move by much more.</p></section>
<section id="coherence"><h2>7. What the same interferograms do carry: coherence</h2><p>Coherence needs no water-level change. Across the nine pairs, Swamp + Marsh was the steadiest reference class (mean {fr['coherence']['class_mean']['Swamp + Marsh']:.2f}), Peatland cropland changed most among land classes ({fr['coherence']['class_mean']['Peatland cropland']:.2f}) and Permanent water stayed near the noise level ({fr['coherence']['class_mean']['Permanent water']:.2f}).</p>{coh_fig}
<p>This is the signal the InSAR layers pass to the U-Net. Low coherence over farmed peat is consistent with mowing, harvest and ploughing between passes, but coherence also falls with rain and with longer gaps, so it is evidence of surface change, not of its cause.</p></section>
<section id="limits"><h2>8. Interpretation, limitations and stopping point</h2><ul>
<li><strong>Weak references:</strong> labels are historical/inferred and can be incomplete, wrong or temporally mismatched. Other is not verified dry land. Peatland classes outside Poland derive from a product with about 1 km detail.</li>
<li><strong>One seed, one frame:</strong> no repeated-seed uncertainty was estimated, and the patch pool lies inside a single radar frame. The direction of the swamp gain is supported by most folds; its size is not established.</li>
<li><strong>Fold spread:</strong> the same recipe ranges widely across folds. Differences of a few points between variants should remain inconclusive.</li>
<li><strong>Decision offsets:</strong> cross-fitting keeps each patch out of its own rule, but the offsets are fitted and scored against the same weak labels.</li>
<li><strong>Fringe method scope:</strong> it was applied with our own parameter ranges to NASA's standard 12-day product. The result says nothing about longer pairs, spring flood dates or the paper's own setting.</li>
<li><strong>Provisional products:</strong> the interferograms are provisional NISAR releases and may be reprocessed.</li>
<li><strong>Stopping point:</strong> conclude this phase with coherence as a candidate input. Repeating with three seeds, extending to track 172 (catalogued, not downloaded) and building pairs of 36 days or more from single-pass scenes belong to follow-up work.</li></ul></section>
<section id="artifacts"><h2>9. Reproducibility and retained artifacts</h2><p>All fold checkpoints are retained under <code>runs/exp/insar_*</code> with their per-epoch logs and final scores; held-out prediction maps and fitted offsets are under <code>runs/insar_e1/oof</code>, and the fringe-method outputs, including GeoTIFFs of the regime and mean flag frequency, under <code>runs/insar_e1</code>.{tests} Retention is on one machine; this report does not claim that an off-machine backup was created.</p>
{run_table}
<p class="small">Timing is operational context, not a controlled speed benchmark; runs shared the machine with other jobs.</p>
<p>From <code>ml/</code>: <code>python -m bogsignal.insar_pairs</code> builds the pair catalogue; <code>python -m bogsignal.insar_flooded --per-pair --export-patches DIR</code> exports the InSAR layers; <code>python -m bogsignal.insar_flooded --method paper --draws 100</code> runs the fringe method; <code>python -m bogsignal.train_unet --folds 5 --fold K --ids IDS [--extra DIR] [--train-merge 2:1]</code> trains a fold model; <code>python -m bogsignal.oof_maps</code> writes held-out maps and offsets; <code>python build_insar_report.py</code> builds this report.</p>
<p class="small">Machine-readable values: <a href="insar_experiment_report_data.json">insar_experiment_report_data.json</a>. Interferograms: {', '.join(f'<code>{esc(f)}</code>' for f in fr['files'])}.</p>
<details><summary>Source files and SHA-256 ({len(SOURCES)})</summary>{table(['Path', 'SHA-256'], [[f'<code>{esc(s["path"])}</code>', f'<code>{s["sha256"]}</code>'] for s in SOURCES])}</details></section>"""
    title = "Wetland segmentation — NISAR interferometry experiment report"
    page = f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>{CSS}</style></head><body><main>{body}</main></body></html>'
    (DOCS / "insar_experiment_report.html").write_text(page)
    if args.fragment:
        args.fragment.write_text(f"<title>InSAR Experiment Report</title><style>{CSS}</style><main>{body}</main>")
    data = {"generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "seed": 0, "folds": 5, "test_evaluated": False,
            "variants": {v[0]: {"name": v[1], "inputs": v[2], "channels": v[3], "complete": v[0] in have} for v in VARIANTS},
            "kfold_validation_last_epoch": result, "feature_test": ft, "decision": dec, "fringe_method": fr, "sources": SOURCES}
    (DOCS / "insar_experiment_report_data.json").write_text(json.dumps(data, indent=1))
    print(f"wrote docs/insar_experiment_report.html ({len(page) // 1024} KB); variants {sorted(have)}; mapped {[v[0] for v in mapped]}")


if __name__ == "__main__":
    main()
