"""Figures for U-Net runs (PNG).

    uv run python -m bogsignal.plot_unet curves --runs "v3=runs/unet_r34_all" --out runs/plots/curves.png
    uv run python -m bogsignal.plot_unet bars --evals "v3=a.json" "v4=b.json" --title "..." --out runs/plots/bars.png
    uv run python -m bogsignal.plot_unet confusion --eval a.json --title "..." --out runs/plots/confusion.png
    uv run python -m bogsignal.plot_unet recall --evals "S2=a.json" "S2 + S1=b.json" "All three=d.json" --baseline "S2 + S1" --out runs/plots/recall.png
    uv run python -m bogsignal.plot_unet maps --run runs/unet_r34_all --data ../../s1s2nisar_256_v3 --out runs/plots/maps.png
    uv run python -m bogsignal.plot_unet examples runs/exp/base_s*/last.pt runs/exp/tm_s*/last.pt --data ../../s1s2nisar_256_v4 --out runs/plots/examples.png
    uv run python -m bogsignal.plot_unet compare --root runs/exp --configs "U-Net=base" "TerraMind=tm" --out runs/plots/compare.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch

from bogsignal.train_unet import CLASSES, IGNORE

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SWAMP, MARSH, WATER, RUN_A, RUN_B = "#eb6834", "#1baf7a", "#2a78d6", "#6250d6", "#898781"
BLUES = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
# map legend: both peatland classes share one neutral so the three wetland/water hues stay distinguishable
MAP_COLORS = ["#f0efec", SWAMP, MARSH, WATER, "#c3c2b7", "#c3c2b7", "#52514e"]
MAP_LEGEND = [("Swamp", SWAMP), ("Marsh", MARSH), ("Permanent water", WATER), ("Peatland (both classes)", "#c3c2b7"),
              ("Other", "#f0efec"), ("No label", "#52514e")]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "legend.frameon": False,
})


def pairs(items: list[str]) -> list[tuple[str, Path]]:
    return [(k, Path(v)) for k, v in (item.split("=", 1) for item in items)]


def save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print("wrote", out)


def curves(args) -> None:
    runs = pairs(args.runs)
    fig, axes = plt.subplots(1, len(runs), figsize=(5.6 * len(runs), 3.8), sharey=True, squeeze=False)
    series = [("Swamp + marsh merged", lambda m: m["wetland_iou"], INK2), ("Marsh", lambda m: m["iou"]["Marsh"], MARSH),
              ("Swamp", lambda m: m["iou"]["Swamp"], SWAMP)]
    for ax, (name, run) in zip(axes[0], runs):
        rows = [json.loads(line) for line in open(run / "metrics.jsonl")]
        epochs = [r["epoch"] for r in rows]
        best = max(rows, key=lambda r: r.get("select", r["score"]))
        ax.axvline(best["epoch"], color=AXIS, lw=1)
        ax.text(best["epoch"] + 0.5, 0.01, f"saved checkpoint\n(epoch {best['epoch']})", color=MUTED, fontsize=8, va="bottom")
        for label, get, color in series:
            ax.plot(epochs, [get(r) for r in rows], color=color, lw=2, solid_capstyle="round", label=label)
            ax.plot(best["epoch"], get(best), "o", color=color, ms=7, mec=SURFACE, mew=2)
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylim(0, None)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("Validation IoU")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 1.0), ncols=3, fontsize=9)
    fig.suptitle("Validation IoU by epoch", x=0.01, y=0.985, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, args.out)


def bars(args) -> None:
    evals = [(k, json.loads(p.read_text())) for k, p in pairs(args.evals)]
    metric = args.metric
    names = ["Swamp", "Marsh", "Swamp + marsh merged", "Permanent water", "Peatland cropland", "Peatland non-cropland", "Other"]

    def value(m, name):
        return m[f"wetland_{metric}"] if name.startswith("Swamp +") else m[metric][name]

    colors = [RUN_A, RUN_B, WATER][: len(evals)]
    h = 0.8 / len(evals)
    fig, ax = plt.subplots(figsize=(7.5, 0.5 + 0.62 * len(names) * max(1, len(evals) / 2)))
    y0 = np.arange(len(names))[::-1].astype(float)
    for i, ((label, m), color) in enumerate(zip(evals, colors)):
        vals = [value(m, n) for n in names]
        ys = y0 + 0.4 - h * (i + 0.5)
        ax.barh(ys, vals, height=h * 0.86, color=color, label=label)
        for y, v in zip(ys, vals):
            ax.text(v + 0.008, y, f"{v:.2f}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y0, names)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1)
    ax.set_xlabel({"iou": "IoU", "ap": "Average precision"}[metric])
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_title(args.title, pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=len(evals), fontsize=9, handlelength=1, borderaxespad=0.2)
    save(fig, args.out)


def confusion(args) -> None:
    cm = np.array(json.loads(args.eval.read_text())["confusion"], float)
    share = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    ax.imshow(share, cmap=BLUES, vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{share[i, j]:.2f}", ha="center", va="center", fontsize=9, color="#ffffff" if share[i, j] > 0.5 else INK)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=30, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Reference class")
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(args.title + "\nshare of each reference class's pixels", fontsize=10.5)
    save(fig, args.out)


def maps(args) -> None:
    import torch

    from bogsignal.train_unet import Patches, load_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids = set(args.ids.read_text().split()) if args.ids else None
    paths = [args.run / f"{args.checkpoint}.pt"] + [Path(m) for m in args.members]  # more than one = averaged probabilities
    members = []
    for path in paths:
        ckpt, model, stats = load_checkpoint(path, device)
        members.append((model.eval(), Patches(args.data, args.split, ckpt["channels"], stats, ids=ids),
                        ckpt["args"].get("loss", "softmax") == "ovr", ckpt))
    model, ds, _, ckpt = members[0]
    mean, std = ds.stats
    bias = np.array(ckpt["bias"], np.float32)[:, None, None] if args.tuned and "bias" in ckpt and len(members) == 1 else 0.0
    by = lambda c: sorted(range(len(ds)), key=lambda i: -ds.records[i]["counts"][c])  # noqa: E731
    chosen = list(dict.fromkeys([i for trio in zip(by(1), by(2)) for i in trio]))[: args.n]
    pos = {c: i for i, c in enumerate(ckpt["channels"])}
    cmap = ListedColormap(MAP_COLORS)
    titles = ["Sentinel-2 (SWIR / NIR / red)", "NISAR HH (dB)" if 10 in pos else "Sentinel-1 VV (dB)", "Reference labels",
              "Model prediction", "Swamp + marsh probability"]
    fig, axes = plt.subplots(len(chosen), 5, figsize=(13.5, 2.75 * len(chosen) + 0.6), squeeze=False)
    for row, i in zip(axes, chosen):
        x, y = ds[i]
        with torch.no_grad():
            p = 0.0
            for net, data, sigmoid, _ in members:
                z = net(data[i][0][None].to(device)).float()
                p = p + (z.sigmoid() if sigmoid else z.softmax(1))[0].cpu().numpy() / len(members)
        raw = x.numpy() * std[:, None, None] + mean[:, None, None]
        rgb = np.stack([raw[pos[c]] for c in (4, 3, 2)], -1)
        lo, hi = np.percentile(rgb, [2, 98], axis=(0, 1))
        y = np.where(y.numpy() == IGNORE, 6, y.numpy())
        row[0].imshow(np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1))
        row[1].imshow(raw[pos[10 if 10 in pos else 6]], cmap="gray", vmin=-20, vmax=0)
        row[2].imshow(y, cmap=cmap, vmin=0, vmax=6, interpolation="nearest")
        row[3].imshow((np.log(np.maximum(p, 1e-6)) + bias).argmax(0), cmap=cmap, vmin=0, vmax=6, interpolation="nearest")
        im = row[4].imshow(np.minimum(p[1] + p[2], 1), cmap=BLUES, vmin=0, vmax=1)
        row[0].set_ylabel(ds.records[i]["id"], fontsize=8, color=MUTED)
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_color(AXIS)
    for ax, t in zip(axes[0], titles):
        ax.set_title(t, fontsize=10)
    fig.legend(handles=[Patch(facecolor=c, edgecolor=AXIS, label=n) for n, c in MAP_LEGEND], loc="lower left",
               bbox_to_anchor=(0.01, 0.0), ncols=6, fontsize=9)
    cax = fig.add_axes([0.815, 0.012, 0.16, 0.012])
    fig.colorbar(im, cax=cax, orientation="horizontal").outline.set_visible(False)
    cax.tick_params(labelsize=8, length=0)
    fig.suptitle(f"{args.title} — each patch is 5.12 km across", x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0.035, 1, 0.985))
    save(fig, args.out)


# all six classes distinguishable: Other, Swamp, Marsh, Permanent water, Peatland cropland, Peatland non-cropland, no label
TRIPTYCH_COLORS = ["#c6c5b5", "#2f7d4f", "#8fcf6b", "#3b7fc4", "#dba545", "#9a6aa6", "#ffffff"]


def examples(args) -> None:
    """Sentinel-2 true colour, weak reference labels and the prediction side by side, one patch per row.

    Several checkpoints are averaged. Class decisions use per-class offsets tuned on validation (plain argmax
    under-predicts the rare classes); --no-tune shows plain argmax instead.
    """
    import torch
    from torch.utils.data import DataLoader

    from bogsignal.train_unet import Patches, evaluate, load_checkpoint, tune_bias

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids = set(args.ids.read_text().split()) if args.ids else None
    members, val = [], 0.0
    for path in args.checkpoints:
        ckpt, model, stats = load_checkpoint(path, device)
        loss = ckpt["args"].get("loss", "softmax")
        if not args.no_tune and not args.offsets:
            loader = DataLoader(Patches(args.data, "validation", ckpt["channels"], stats), 32, num_workers=6)
            _, pv, yv = evaluate(model, loader, device, loss, arrays=True)
            val = val + pv / len(args.checkpoints)
        members.append((model.eval(), Patches(args.data, args.split, ckpt["channels"], stats, ids=ids), loss == "ovr"))
    if args.offsets:  # given explicitly, e.g. the offsets fitted by bogsignal.eval_combined
        bias = np.array([float(v) for v in args.offsets.split(",")], np.float32)
    else:
        bias = np.zeros(len(CLASSES), np.float32) if args.no_tune else np.array(tune_bias(val, yv), np.float32)
    print("class offsets", bias.tolist())
    ds = members[0][1]
    mean, std = ds.stats
    pos = {c: i for i, c in enumerate(ds.channels)}
    def predict(i):
        p = 0.0
        with torch.no_grad():
            for net, data, sigmoid in members:
                x = data[i][0][None].to(device)
                prob = (lambda z: z.float().sigmoid()) if sigmoid else (lambda z: z.float().softmax(1))
                p = p + ((prob(net(x)) + prob(net(x.flip(-1))).flip(-1)) / 2)[0].cpu().numpy() / len(members)
        return (np.log(np.maximum(p, 1e-6)) + bias[:, None, None]).argmax(0)

    by = lambda c: sorted(range(len(ds)), key=lambda i: -ds.records[i]["counts"][c])  # noqa: E731
    note: dict[int, str] = {}
    if ids:
        chosen = list(range(len(ds)))[: args.n]
    elif args.select == "detections":
        # patches where the model predicts the most swamp and marsh (chosen by prediction, not by agreement with the
        # reference), alternating the two classes and skipping neighbours of patches already chosen
        area = {}
        for i in range(len(ds)):
            pred = predict(i)
            valid = ds[i][1].numpy() != IGNORE
            area[i] = [int(((pred == c) & valid).sum()) for c in (1, 2)]
        ranked = [sorted(area, key=lambda i: -area[i][k]) for k in (0, 1)]
        chosen, near = [], lambda a, b: abs(ds.records[a]["row"] - ds.records[b]["row"]) <= 512 and abs(ds.records[a]["col"] - ds.records[b]["col"]) <= 512  # noqa: E731
        for i in [i for pair in zip(*ranked) for i in pair]:
            if len(chosen) < args.n and i not in chosen and not any(near(i, j) for j in chosen):
                chosen.append(i)
    elif args.select in ("best", "weak"):
        # highest / lowest per-patch IoU of combined swamp+marsh, among patches where the reference or the prediction
        # has at least --min-share wetland; neighbours of already chosen patches are skipped
        scored = []
        for i in range(len(ds)):
            y = ds[i][1].numpy()
            if max(ds.records[i]["counts"][1] + ds.records[i]["counts"][2], 0) / 256**2 < args.min_share / 4:
                pred = None  # cheap pre-filter: little reference wetland; still check the prediction below
            pred = predict(i)
            valid = y != IGNORE
            ref, det = np.isin(y, (1, 2)) & valid, np.isin(pred, (1, 2)) & valid
            if max(ref.sum(), det.sum()) / 256**2 >= args.min_share:
                scored.append(((ref & det).sum() / max((ref | det).sum(), 1), i))
        scored.sort(reverse=args.select == "best")
        note.update({i: f"IoU {100 * v:.0f}%" for v, i in scored})
        chosen, near = [], lambda a, b: abs(ds.records[a]["row"] - ds.records[b]["row"]) <= 256 and abs(ds.records[a]["col"] - ds.records[b]["col"]) <= 256  # noqa: E731
        for _, i in scored:
            if len(chosen) < args.n and not any(near(i, j) for j in chosen):
                chosen.append(i)
        print(f"{len(scored)} candidate patches; median wetland IoU {100 * float(np.median([v for v, _ in scored])):.0f}%")
    else:  # alternate the patches richest in reference swamp and marsh
        chosen = list(dict.fromkeys([i for pair in zip(by(1), by(2)) for i in pair]))[args.skip: args.skip + args.n]
    print("patches", [ds.records[i]["id"] for i in chosen])
    cmap = ListedColormap(TRIPTYCH_COLORS)
    per_row = max(1, args.per_row)
    rows = -(-len(chosen) // per_row)
    with plt.rc_context({"figure.facecolor": "#ffffff", "savefig.facecolor": "#ffffff", "axes.facecolor": "#ffffff"}):
        fig, axes = plt.subplots(rows, 3 * per_row, figsize=(8.6 * per_row, 2.85 * rows + 0.7), squeeze=False,
                                 gridspec_kw={"wspace": 0.04, "hspace": 0.05})
        for ax in axes.ravel():
            ax.axis("off")
        for k, i in enumerate(chosen):
            row = axes[k // per_row, 3 * (k % per_row): 3 * (k % per_row) + 3]
            x, y = ds[i]
            y = y.numpy()
            raw = x.numpy() * std[:, None, None] + mean[:, None, None]
            rgb = np.clip(np.stack([raw[pos[b]] for b in (2, 1, 0)], -1) / args.rgb_max, 0, 1) ** 0.8
            pred = np.where(y == IGNORE, 6, predict(i))
            row[0].imshow(rgb)
            row[1].imshow(np.where(y == IGNORE, 6, y), cmap=cmap, vmin=0, vmax=6, interpolation="nearest")
            row[2].imshow(pred, cmap=cmap, vmin=0, vmax=6, interpolation="nearest")
            if not args.hide_ids:
                row[0].set_ylabel(ds.records[i]["id"], fontsize=7.5, color=INK2)
            if i in note:  # per-patch score, inside the first panel so it never collides with a neighbour
                row[0].text(0.03, 0.96, note[i], transform=row[0].transAxes, ha="left", va="top", fontsize=9, color=INK,
                            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#ffffff", "edgecolor": "none", "alpha": 0.85})
            for ax in row:
                ax.axis("on")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                for sp in ax.spines.values():
                    sp.set_visible(True)
                    sp.set_color(INK2)
        for ax, t in zip(axes[0], ["Sentinel-2 RGB", "Weak reference", "Prediction"] * per_row):
            ax.set_title(t, fontsize=10.5, fontweight="normal", loc="center")
        names = CLASSES + ["No label (not scored)"]
        fig.legend(handles=[Patch(facecolor=TRIPTYCH_COLORS[k], edgecolor=AXIS, label=names[k]) for k in (1, 2, 3, 4, 5, 0, 6)],
                   loc="lower center", bbox_to_anchor=(0.5, 0.0), ncols=4, fontsize=9)
        if args.title:
            fig.suptitle(args.title, fontsize=11.5, fontweight="bold")
        fig.subplots_adjust(left=0.03 if not args.hide_ids else 0.01, right=0.99, top=1 - (0.75 if args.title else 0.4) / fig.get_figheight(),
                            bottom=0.75 / fig.get_figheight())
        save(fig, args.out)


def compare(args) -> None:
    """One row per configuration, one dot per seed, a bar at the mean, for swamp, marsh and merged-wetland AP."""
    metrics = [("Swamp", lambda m: m["ap"]["Swamp"], SWAMP), ("Marsh", lambda m: m["ap"]["Marsh"], MARSH),
               ("Swamp + marsh merged", lambda m: m["wetland_ap"], INK2)]
    rows = []
    for label, config in (item.split("=", 1) for item in args.configs):
        runs = [json.loads(p.read_text())[args.checkpoint] for p in sorted(args.root.glob(f"{config}_[sf][0-9]*/final.json"))]
        rows.append((label, [r[args.split] for r in runs if args.split in r]))
    for item in args.gbm:
        label, spec = item.split("=", 1)
        path, name = spec.split(":", 1)
        entry = json.loads(Path(path).read_text())[name]
        rows.append((label, list(entry.values()) if name == "members" else [entry]))  # "members": one dot per ensemble member
    rows = [r for r in rows if r[1]]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 0.9 + 0.42 * len(rows)), sharey=True)
    ys = np.arange(len(rows))[::-1]
    for ax, (name, get, color) in zip(axes, metrics):
        for y, (_, runs) in zip(ys, rows):
            vals = np.array([get(m) for m in runs])
            ax.plot(vals, np.full(len(vals), y), "o", color=color, ms=5, alpha=0.45, mec="none")
            ax.plot([vals.mean()] * 2, [y - 0.28, y + 0.28], color=color, lw=2.5, solid_capstyle="round")
            ax.text(vals.max() + 0.012, y, f"{vals.mean():.2f}", va="center", fontsize=8.5, color=INK2)
        ax.set_title(name)
        ax.set_xlabel("Average precision")
        ax.set_xlim(0, max(0.6, max(get(m) for _, runs in rows for m in runs) + 0.08))
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(ys, [f"{label}  (n={len(runs)})" for label, runs in rows])
    axes[0].set_ylim(-0.6, len(rows) - 0.4)
    fig.suptitle(args.title or f"Average precision on {args.split}: dots are individual runs, bars are the mean",
                 x=0.01, y=0.99, ha="left", fontweight="bold", fontsize=13)
    if args.subtitle:
        fig.text(0.01, 0.925 - 0.02 * (8 / max(len(rows), 8)), args.subtitle, fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93 if args.subtitle else 0.97))
    save(fig, args.out)


DARK = {"figure.facecolor": "#1a1a19", "axes.facecolor": "#1a1a19", "savefig.facecolor": "#1a1a19", "text.color": "#f0efec",
        "axes.labelcolor": "#c3c2b7", "xtick.labelcolor": "#c3c2b7", "ytick.labelcolor": "#c3c2b7", "grid.color": "#2c2c2a",
        "axes.edgecolor": "#383835"}
SENSOR_COLORS = ["#3987e5", "#199e70", "#d95926", "#c98500"]  # dark-surface steps


def recall(args) -> None:
    """Recall of reference swamp+marsh pixels at area budgets, one line per sensor set, plus two headline tiles."""
    from matplotlib.patches import FancyBboxPatch

    evals = [(k, json.loads(p.read_text())) for k, p in pairs(args.evals)]
    budgets = list(evals[0][1]["wetland_recall_at_area"])
    base = next(m for k, m in evals if k == args.baseline) if args.baseline else evals[0][1]
    top_name, top = evals[-1]
    with plt.rc_context(DARK):
        fig = plt.figure(figsize=(9.2, 6.9))
        grid = fig.add_gridspec(2, 2, height_ratios=[4.2, 1.25], hspace=0.42, wspace=0.04)
        ax = fig.add_subplot(grid[0, :])
        for (label, m), color in zip(evals, SENSOR_COLORS):
            vals = [100 * m["wetland_recall_at_area"][b] for b in budgets]
            ax.plot(range(len(budgets)), vals, color=color, lw=2, marker="o", ms=8, mec="#1a1a19", mew=2, label=label,
                    solid_capstyle="round", zorder=3 if m is top else 2)
        for i, b in enumerate(budgets):
            ax.annotate(f"{100 * top['wetland_recall_at_area'][b]:.0f}%", (i, 100 * top["wetland_recall_at_area"][b]),
                        xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9, color="#f0efec")
        ax.set_xticks(range(len(budgets)), [f"{b}%" for b in budgets])
        ax.set_yticks(range(0, 101, 25), [f"{v}%" for v in range(0, 101, 25)])
        ax.set_ylim(0, 105)
        ax.set_xlim(-0.15, len(budgets) - 0.85)
        ax.set_xlabel("Share of area selected, ranked by swamp + marsh probability")
        ax.set_ylabel("Reference swamp + marsh pixels recovered")
        ax.grid(axis="x", visible=False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(length=0)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=len(evals), fontsize=10, handlelength=1.2)
        fig.suptitle(args.title or "Recall of reference swamp and marsh pixels across selected area budgets",
                     x=0.02, y=0.975, ha="left", fontsize=12.5, fontweight="bold")
        fig.text(0.02, 0.925, args.subtitle, fontsize=9.5, color="#c3c2b7")
        a, b = 100 * base["wetland_ap"], 100 * top["wetland_ap"]
        tiles = [(f"{top_name} · top 5% of area", f"{100 * top['wetland_recall_at_area']['5']:.1f}%", "of reference swamp + marsh pixels recovered"),
                 (f"Average precision · {args.baseline or evals[0][0]} → {top_name}", f"{a:.1f} → {b:.1f}%", f"{b - a:+.1f} percentage points")]
        for col, (label, value, note) in enumerate(tiles):
            t = fig.add_subplot(grid[1, col])
            t.axis("off")
            t.add_patch(FancyBboxPatch((0.02, 0.04), 0.96, 0.92, boxstyle="round,pad=0,rounding_size=0.05", transform=t.transAxes,
                                       facecolor="#1a1a19", edgecolor="#383835", lw=1))
            t.text(0.07, 0.78, label, transform=t.transAxes, fontsize=9.5, color="#c3c2b7", va="center")
            t.text(0.07, 0.47, value, transform=t.transAxes, fontsize=24, fontweight="bold", va="center")
            t.text(0.07, 0.17, note, transform=t.transAxes, fontsize=9.5, color="#c3c2b7", va="center")
        fig.subplots_adjust(top=0.86, left=0.09, right=0.98, bottom=0.03)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=160)
        print("wrote", args.out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("curves", "bars", "confusion", "maps", "recall"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--title", default="")
    sub.choices["curves"].add_argument("--runs", nargs="+", required=True, help="label=run_dir")
    sub.choices["bars"].add_argument("--evals", nargs="+", required=True, help="label=eval.json (up to 3)")
    sub.choices["bars"].add_argument("--metric", default="iou", choices=["iou", "ap"])
    sub.choices["confusion"].add_argument("--eval", type=Path, required=True)
    r = sub.choices["recall"]
    r.add_argument("--evals", nargs="+", required=True, help="label=eval.json, weakest sensor set first (up to 4)")
    r.add_argument("--baseline", help="label of the run the average-precision tile compares against (default: first)")
    r.add_argument("--subtitle", default="")
    m = sub.choices["maps"]
    m.add_argument("--run", type=Path, required=True)
    m.add_argument("--data", type=Path, required=True)
    m.add_argument("--split", default="validation")
    m.add_argument("--ids", type=Path)
    m.add_argument("--n", type=int, default=6)
    m.add_argument("--checkpoint", default="best", choices=["best", "last"])
    m.add_argument("--members", nargs="*", default=[], help="extra checkpoint files whose probabilities are averaged with --run")
    m.add_argument("--tuned", action="store_true", help="apply the checkpoint's validation-tuned class offsets to the predicted map")
    w = sub.add_parser("examples")
    w.add_argument("checkpoints", nargs="+", type=Path, help="checkpoint files; probabilities are averaged")
    w.add_argument("--out", type=Path, required=True)
    w.add_argument("--title", default="")
    w.add_argument("--data", type=Path, required=True)
    w.add_argument("--split", default="test")
    w.add_argument("--ids", type=Path, help="text file of patch ids to show instead of the swamp/marsh-richest patches")
    w.add_argument("--n", type=int, default=6)
    w.add_argument("--skip", type=int, default=0, help="skip this many of the richest patches (to page through examples)")
    w.add_argument("--rgb-max", type=float, default=0.16, help="reflectance shown as white in the true-colour column")
    w.add_argument("--select", default="reference", choices=["reference", "detections", "best", "weak"],
                   help="reference: patches richest in labelled swamp/marsh; detections: patches where the model predicts the most; "
                        "best / weak: highest / lowest per-patch swamp+marsh IoU")
    w.add_argument("--min-share", type=float, default=0.05, help="best/weak: smallest wetland share of a patch (reference or predicted)")
    w.add_argument("--offsets", help="six comma-separated class offsets to use instead of tuning on validation")
    w.add_argument("--hide-ids", action="store_true", help="do not print patch ids beside the rows")
    w.add_argument("--per-row", type=int, default=1, help="patches shown side by side in each row")
    w.add_argument("--no-tune", action="store_true", help="plain argmax instead of validation-tuned class offsets")
    c = sub.add_parser("compare")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--title", default="")
    c.add_argument("--root", type=Path, required=True, help="directory of <config>_s<seed> runs")
    c.add_argument("--configs", nargs="+", required=True, help="label=config, in display order")
    c.add_argument("--gbm", nargs="*", default=[], help="label=file.json:key rows read from a results file (ensemble.py output: key 'ensemble' or 'members')")
    c.add_argument("--split", default="validation", choices=["validation", "test"])
    c.add_argument("--checkpoint", default="last", choices=["best", "last"])
    c.add_argument("--subtitle", default="")
    args = ap.parse_args()
    {"curves": curves, "bars": bars, "confusion": confusion, "maps": maps, "recall": recall, "compare": compare, "examples": examples}[args.cmd](args)


if __name__ == "__main__":
    main()
