"""Build the plain-language prediction viewer from out-of-fold maps of the InSAR k-fold runs.

    cd ml && python build_prediction_viewer.py [--runs ../../runs] [--data ../../s1s2nisar_256_v4]

Reads only finished artefacts: the out-of-fold prediction maps written by ``bogsignal.oof_maps`` for the four
variants (base, pp, coh, merge), their k-fold ``final.json`` files and the dataset's label and image patches.
Writes docs/prediction_viewer.html: one map with a layer switch (satellite photo, existing maps, each model as
land types, confidence, or hits and misses), a score table and five close-ups. Nothing is trained here and the
locked test split is not scored.

The map area is chosen by rule: among rectangles of neighbouring patches with none missing, at least 6 patches on
each side and 1.0 to 1.8 times wider than tall, the one holding the most reference wetland. The close-ups sit at
the 95th, 75th, 50th, 25th and 5th percentile of wetland overlap (mean of the four models) among patches with at
least 5% reference wetland and at most 5% unlabelled pixels.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
VARS = [("base", "Baseline", "satellite photos + radar brightness"),
        ("pp", "+ InSAR, 27 layers", "adds how the ground changed between radar passes"),
        ("coh", "+ coherence, 9 layers", "adds only the simplest change measure"),
        ("merge", "One wetland class", "baseline, with swamp and marsh learned as one")]
SHORT = {"base": "Baseline", "pp": "+ InSAR", "coh": "+ coherence", "merge": "One class"}
DEFAULT = "coh"  # layer shown when the page opens
PATCH = 256
OTHER, GAP, UNLABELLED = (236, 238, 230), (203, 210, 199), (250, 250, 246)
CLASS_COLOURS = [OTHER, (0, 158, 115), (0, 114, 178), (230, 159, 0), (204, 121, 167), UNLABELLED]
SCORE_COLOURS = [OTHER, (169, 214, 194), (111, 191, 159), (47, 159, 122), (11, 122, 90), (6, 81, 60)]
AGREE_COLOURS = [OTHER, (0, 140, 102), (213, 94, 0), (122, 82, 199), UNLABELLED]
SCORE_EDGES = np.array([0.05, 0.10, 0.20, 0.35, 0.55]) * 255  # wetland probability bins
TO_FIVE = np.array([0, 1, 1, 2, 3, 4] + [255] * 250, np.uint8)  # six label classes -> other, wetland, water, peat cropland, peat non-cropland
PICKS = [("One of the best", 0.95), ("Good", 0.75), ("Typical", 0.50), ("Poor", 0.25), ("One of the worst", 0.05)]


def data_uri(data: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def indexed_png(codes: np.ndarray, colours: list) -> bytes:
    image = Image.fromarray(codes, "P")
    image.putpalette([x for colour in colours + [GAP] for x in colour])
    buffer = io.BytesIO()
    image.save(buffer, "PNG", optimize=True)
    return buffer.getvalue()


def jpeg(array: np.ndarray, quality: int) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, "JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def read_labels(data: Path, record: dict) -> np.ndarray:
    with rasterio.open(data / record["labels"]) as src:
        return TO_FIVE[src.read(1)]


def read_rgb(data: Path, record: dict) -> np.ndarray:
    with rasterio.open(data / record["images"]) as src:
        x = src.read([3, 2, 1]).astype(np.float32)
    x[~np.isfinite(x) | (x < -1000)] = 0
    return (np.moveaxis(np.clip(x / 0.16, 0, 1) ** 0.8, 0, -1) * 255).astype(np.uint8)


def agreement(prediction: np.ndarray, wetland: np.ndarray, labelled: np.ndarray) -> np.ndarray:
    """0 neither, 1 both, 2 reference only, 3 model only, 4 unlabelled."""
    out = np.zeros(prediction.shape, np.uint8)
    out[prediction & wetland] = 1
    out[~prediction & wetland] = 2
    out[prediction & ~wetland] = 3
    out[~labelled] = 4
    return out


def patch_stats(data: Path, oof: Path, index: dict, ids: list[str]) -> tuple[list[dict], dict, dict]:
    """Per-patch wetland overlap for every variant, and frame-wide pixel counts (tp, model wetland, reference wetland)."""
    rows = np.array([index[i]["row"] for i in ids]) // PATCH
    cols = np.array([index[i]["col"] for i in ids]) // PATCH
    r0, c0 = rows.min(), cols.min()
    stats = []
    total = {v: np.zeros(3, np.int64) for v, _, _ in VARS}
    total_plain = {v: np.zeros(3, np.int64) for v, _, _ in VARS}
    for i, r, c in zip(ids, rows - r0, cols - c0):
        y = read_labels(data, index[i])
        ok = y != 255
        wet = (y == 1) & ok
        row = dict(id=i, r=int(r), c=int(c), wet=float(wet.sum() / max(ok.sum(), 1)),
                   unknown=1 - sum(index[i]["counts"]) / PATCH**2)
        for v, _, _ in VARS:
            z = np.load(oof / v / f"{i}.npz")
            p = z["cls_tuned"] == 1
            inter, union = int((p & wet).sum()), int(((p | wet) & ok).sum())
            row[v] = inter / union if union else -1.0
            row["fold"] = int(z["fold"])
            total[v] += [inter, int((p & ok).sum()), int(wet.sum())]
            plain = z["cls"] == 1
            total_plain[v] += [int((plain & wet).sum()), int((plain & ok).sum()), int(wet.sum())]
        stats.append(row)
    return stats, total, total_plain


def choose_block(stats: list[dict]) -> tuple[int, int, int, int]:
    """Gap-free rectangle of patches holding the most reference wetland: top row, left column, height, width."""
    rows, cols = max(s["r"] for s in stats) + 1, max(s["c"] for s in stats) + 1
    present, wet = np.zeros((rows, cols), bool), np.zeros((rows, cols))
    for s in stats:
        present[s["r"], s["c"]] = True
        wet[s["r"], s["c"]] = s["wet"]
    best, height = None, np.zeros(cols, int)
    for r in range(rows):
        height = np.where(present[r], height + 1, 0)
        for c in range(cols):
            h = height[c]
            for c2 in range(c, cols):
                h = min(h, height[c2])
                if h == 0:
                    break
                w = c2 - c + 1
                for hh in range(6, h + 1):
                    if w < 6 or not 1.0 <= w / hh <= 1.8:
                        continue
                    key = (hh * w * wet[r - hh + 1:r + 1, c:c + w].mean(), hh * w)
                    if best is None or key > best[0]:
                        best = (key, r - hh + 1, c, hh, w)
    if best is None:
        raise SystemExit("no gap-free block of at least 6 x 6 patches")
    return best[1:]


def block_layers(data: Path, oof: Path, index: dict, stats: list[dict], block: tuple[int, int, int, int]):
    """Full-detail mosaics of the block: data-URI layers, the patches in reading order, and overlap per variant."""
    top, left, bh, bw = block
    at = {(s["r"], s["c"]): s for s in stats}
    shape = (bh * PATCH, bw * PATCH)
    ref, rgb = np.zeros(shape, np.uint8), np.zeros(shape + (3,), np.uint8)
    cls = {v: np.zeros(shape, np.uint8) for v, _, _ in VARS}
    score = {v: np.zeros(shape, np.uint8) for v, _, _ in VARS}
    agree = {v: np.zeros(shape, np.uint8) for v, _, _ in VARS}
    total = {v: np.zeros(3, np.int64) for v, _, _ in VARS}
    patches = []
    for r in range(bh):
        for c in range(bw):
            s = at[(top + r, left + c)]
            window = (slice(r * PATCH, (r + 1) * PATCH), slice(c * PATCH, (c + 1) * PATCH))
            y = read_labels(data, index[s["id"]])
            ok = y != 255
            wet = (y == 1) & ok
            ref[window] = np.where(ok, y, 5)
            rgb[window] = read_rgb(data, index[s["id"]])
            for v, _, _ in VARS:
                z = np.load(oof / v / f"{s['id']}.npz")
                p = z["cls_tuned"] == 1
                cls[v][window] = z["cls_tuned"]
                score[v][window] = np.searchsorted(SCORE_EDGES, z["wetland"], side="right").astype(np.uint8)
                agree[v][window] = agreement(p, wet, ok)
                total[v] += [int((p & wet).sum()), int((p & ok).sum()), int(wet.sum())]
            patches.append([s["id"], r, c, s["fold"], round(100 * s["wet"], 1)]
                           + [(-1 if s[v] < 0 else round(100 * s[v])) for v, _, _ in VARS])
    layers = {"s2": data_uri(jpeg(rgb, 80), "image/jpeg"), "ref": data_uri(indexed_png(ref, CLASS_COLOURS), "image/png")}
    for v, _, _ in VARS:
        layers[f"{v}.cls"] = data_uri(indexed_png(cls[v], CLASS_COLOURS), "image/png")
        layers[f"{v}.score"] = data_uri(indexed_png(score[v], SCORE_COLOURS), "image/png")
        layers[f"{v}.agree"] = data_uri(indexed_png(agree[v], AGREE_COLOURS), "image/png")
    overlap = {v: t[0] / (t[1] + t[2] - t[0]) for v, t in total.items()}
    return layers, patches, overlap


def close_ups(data: Path, oof: Path, index: dict, stats: list[dict]) -> tuple[str, int]:
    candidates = [s for s in stats if s["wet"] >= 0.05 and s["unknown"] <= 0.05]
    candidates.sort(key=lambda s: np.mean([s[v] for v, _, _ in VARS]))
    out = ""
    for title, quantile in PICKS:
        s = candidates[int(round(quantile * (len(candidates) - 1)))]
        i = s["id"]
        y = read_labels(data, index[i])
        y = np.where(y == 255, 5, y).astype(np.uint8)
        cells = (f'<figure><img src="{data_uri(jpeg(read_rgb(data, index[i]), 86), "image/jpeg")}" alt="Sentinel-2 image of patch {i}" '
                 f'width="256" height="256"><figcaption>Satellite photo</figcaption></figure>')
        cells += (f'<figure><img src="{data_uri(indexed_png(y, CLASS_COLOURS), "image/png")}" alt="Reference labels, patch {i}" '
                  f'width="256" height="256"><figcaption>Existing maps</figcaption></figure>')
        for v, name, _ in VARS:
            p = np.load(oof / v / f"{i}.npz")["cls_tuned"]
            cells += (f'<figure><img src="{data_uri(indexed_png(p, CLASS_COLOURS), "image/png")}" alt="{name} class map, patch {i}" width="256" height="256">'
                      f'<figcaption>{name}<br><span class="num">match {100*s[v]:.0f}%</span></figcaption></figure>')
        out += (f'<div class="ex"><h3>{title} <span class="pid">{100*s["wet"]:.0f}% wetland on existing maps · square {i}</span></h3>'
                f'<div class="cells">{cells}</div></div>')
    return out, len(candidates)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=ROOT.parent / "runs")
    ap.add_argument("--data", type=Path, default=ROOT.parent / "s1s2nisar_256_v4")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "prediction_viewer.html")
    ap.add_argument("--fragment", type=Path, help="also write the page without the html/head/body wrapper")
    args = ap.parse_args()
    oof, exp = args.runs / "insar_e1" / "oof", args.runs / "exp"
    with open(args.data / "index.jsonl") as f:
        index = {r["id"]: r for r in map(json.loads, f) if "split" in r}
    ids = sorted(p.stem for p in (oof / VARS[0][0]).glob("*.npz"))
    for v, _, _ in VARS[1:]:
        if sorted(p.stem for p in (oof / v).glob("*.npz")) != ids:
            raise SystemExit(f"{oof / v} does not cover the same patches as {oof / VARS[0][0]}")

    stats, total, total_plain = patch_stats(args.data, oof, index, ids)
    block = choose_block(stats)
    _, _, bh, bw = block
    layers, patches, block_overlap = block_layers(args.data, oof, index, stats, block)
    examples, n_candidates = close_ups(args.data, oof, index, stats)

    width, height = bw * PATCH, bh * PATCH
    names = {v: name for v, name, _ in VARS}
    alt = {"s2": "Satellite photo of the whole area", "ref": "Existing wetland maps of the whole area"}
    view_alt = {"cls": "land types", "score": "wetland confidence", "agree": "hits and misses against existing maps"}
    images = ""
    for key, uri in layers.items():
        text = alt.get(key) or f'{names[key.split(".")[0]]}: {view_alt[key.split(".")[1]]} across the whole area'
        # only the opening layer is loaded up front; the page sets src on the others when they are first picked
        source = f'data-key="{key}" src="{uri}"' if key == f"{DEFAULT}.cls" else f'hidden data-key="{key}" data-src="{uri}"'
        images += f'<img {source} alt="{text}" width="{width}" height="{height}">'

    def folds(v: str) -> list[float]:
        return [json.loads((exp / f"insar_{v}_f{k}" / "final.json").read_text())["last"]["validation"]["wetland_ap"] for k in range(5)]

    ap_folds = {v: folds(v) for v, _, _ in VARS}
    hard = {v: (t[0] / t[2], t[0] / t[1], t[0] / (t[1] + t[2] - t[0])) for v, t in total.items()}
    columns = [{v: hard[v][k] for v in hard} for k in range(3)] + [{v: float(np.mean(ap_folds[v])) for v in hard}]
    table = ""
    for v, name, desc in VARS:
        def cell(column: dict, text: str, extra: str = "") -> str:
            return f'<td class="n{" best" if column[v] == max(column.values()) else ""}">{text}{extra}</td>'
        up = "" if v == "base" else f'<small>better than baseline in {sum(a > b for a, b in zip(ap_folds[v], ap_folds["base"]))} of 5 regions</small>'
        table += (f'<tr><td>{name}<small>{desc[0].upper() + desc[1:]}</small></td>' + cell(columns[0], f"{100*hard[v][0]:.0f}%")
                  + cell(columns[1], f"{100*hard[v][1]:.0f}%") + cell(columns[2], f"{100*hard[v][2]:.0f}%")
                  + cell(columns[3], f"{np.mean(ap_folds[v]):.2f}", up) + "</tr>")
    buttons = ('<button type="button" id="layer-s2" data-layer="s2" aria-pressed="false">Satellite photo</button>'
               '<button type="button" id="layer-ref" data-layer="ref" aria-pressed="false">Existing maps<small>what older maps show</small></button>')
    for v, name, desc in VARS:
        buttons += (f'<button type="button" id="layer-{v}" data-layer="{v}" aria-pressed="{"true" if v == DEFAULT else "false"}">{name}'
                    f'<small>{desc}</small></button>')

    tp, n_pred, n_ref = total_plain["base"]
    page = (Path(__file__).parent / "prediction_viewer_template.html").read_text()
    fill = {
        "{{NPATCH}}": f"{len(stats):,}", "{{LAYER_BUTTONS}}": buttons, "{{LAYER_IMAGES}}": images, "{{TABLE}}": table,
        "{{KMW}}": f"{bw * 5.12:.0f}", "{{KMH}}": f"{bh * 5.12:.0f}", "{{ASPECT}}": f"{bw / bh:.4f}", "{{NBLOCK}}": str(bh * bw),
        "{{BLOCK_BASE}}": f"{100 * block_overlap['base']:.0f}%", "{{BLOCK_PP}}": f"{100 * block_overlap['pp']:.0f}%",
        "{{PLAIN_FOUND}}": f"{100*tp/n_ref:.0f}%", "{{PLAIN_CORRECT}}": f"{100*tp/n_pred:.0f}%",
        "{{NCAND}}": str(n_candidates), "{{EXAMPLES}}": examples, "{{C}}": str(bw), "{{R}}": str(bh), "{{FULLW}}": str(width),
        "{{MODELS}}": json.dumps([[v, SHORT[v]] for v, _, _ in VARS]), "{{PATCHES}}": json.dumps(patches, separators=(",", ":")),
        "{{DEFAULT}}": DEFAULT,
    }
    for key, value in fill.items():
        if key not in page:
            raise SystemExit(f"template has no {key}")
        page = page.replace(key, value)
    left_over = set(re.findall(r"\{\{[A-Z_]+\}\}", page))
    if left_over:
        raise SystemExit(f"unfilled template fields: {sorted(left_over)}")
    if args.fragment:
        args.fragment.write_text(page)

    # standalone document: title, fonts and styles go in the head
    cut = page.index('<div class="wrap">')
    head = page[:cut]
    head = head.replace("<style>\n", "<style>\n:root { color-scheme:light; }\nbody { margin:0; }\nimg { max-width:100%; }\n", 1)
    args.out.write_text('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
                        + head + "</head>\n<body>\n" + page[cut:] + "\n</body>\n</html>\n")
    print(f"wrote {args.out} ({args.out.stat().st_size // 1024} KB): {bh * bw} patches on the map, "
          f"{bw * 5.12:.0f} by {bh * 5.12:.0f} km, close-ups from {n_candidates} candidates")


if __name__ == "__main__":
    main()
