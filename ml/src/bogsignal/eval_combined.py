"""Score train_unet checkpoints with the metrics and scoring area of the A-D report.

    python -m bogsignal.eval_combined --data ../../s1s2nisar_256_v4 --out ../../runs/exp/analysis/combined_validation.json \
        --group "U-Net all=../../runs/exp/base_s0/last.pt,../../runs/exp/base_s1/last.pt" \
        --ensemble "Ensemble=../../runs/exp/base_s0/last.pt,../../runs/exp/tm_s0/last.pt"

Every member of a ``--group`` is scored on its own (the report takes the mean); an ``--ensemble`` averages its
members' class probabilities first. Definitions follow ``bogsignal.training``:

* scoring area: label-eligible pixels that are valid in all 14 predictors, at full resolution;
* six-class macro IoU from the plain six-class argmax;
* combined wetland class: argmax over [pOther, pSwamp + pMarsh, pWater, pPeatCropland, pPeatNonCropland];
* candidate ranking by pSwamp + pMarsh: average precision and recall/precision at 1, 5, 10 and 20% of scored area.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset

from bogsignal.train_unet import CLASSES, IGNORE, load_checkpoint, read_index, to_model_units

BUDGETS = (0.01, 0.05, 0.10, 0.20)


class RawPatches(Dataset):
    """All 14 channels in model units (NaN where missing), labels, and the all-14-valid input mask."""

    def __init__(self, root: Path, split: str):
        self.root = root
        self.records = [r for r in read_index(root).values() if r["split"] == split]
        units = [c["units"] for c in json.loads((root / "plan.json").read_text())["channels"]][6:]
        self.linear_sar = units[0].startswith("linear_")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        with rasterio.open(self.root / rec["images"]) as ds:
            x = to_model_units(ds.read(), self.linear_sar)
        with rasterio.open(self.root / rec["labels"]) as ds:
            y = ds.read(1)
        with rasterio.open(self.root / rec["input_validity"]) as ds:
            valid = (ds.read() == 1).all(0)
        return torch.from_numpy(x), torch.from_numpy(y).long(), torch.from_numpy(valid)


def tune_offsets(p5: np.ndarray, y5: np.ndarray, rounds: int = 2) -> list[float]:
    """Log-probability offsets for the five combined classes that maximise their macro IoU (coordinate ascent).
    Fit on validation only; the hard map otherwise under-predicts the rare classes."""
    logp = np.log(np.maximum(p5, 1e-6))

    def macro(off):
        cm = np.bincount(y5 * 5 + (logp + off).argmax(1), minlength=25).reshape(5, 5).astype(float)
        return (np.diag(cm) / np.maximum(cm.sum(0) + cm.sum(1) - np.diag(cm), 1)).mean()

    off = np.zeros(5, np.float32)
    best = macro(off)
    for _ in range(rounds):
        for c in range(1, 5):
            for b in np.linspace(-2, 4, 25, dtype=np.float32):
                trial = off.copy()
                trial[c] = b
                score = macro(trial)
                if score > best:
                    best, off = score, trial
    return off.round(3).tolist()


class Scorer:
    def __init__(self, offsets: list[float] | None = None, keep_sample: bool = False):
        self.offsets, self.keep_sample, self.sample_p, self.sample_y = offsets, keep_sample, [], []
        self.cm6 = np.zeros((6, 6), np.int64)
        self.cm5 = np.zeros((5, 5), np.int64)
        self.scores, self.positive = [], []

    def add(self, p: torch.Tensor, y: torch.Tensor, ok: torch.Tensor):
        """p: (B, 6, H, W) probabilities; y labels; ok: scored pixels"""
        merged = torch.stack([p[:, 0], p[:, 1] + p[:, 2], p[:, 3], p[:, 4], p[:, 5]], 1)
        y5 = torch.tensor([0, 1, 1, 2, 3, 4], device=y.device)[y.clamp(max=5)]
        if self.keep_sample:  # stride-4 pixel grid, enough to fit five offsets
            sub = ok[:, 2::4, 2::4]
            self.sample_p.append(merged[:, :, 2::4, 2::4].permute(0, 2, 3, 1)[sub].float().cpu().numpy())
            self.sample_y.append(y5[:, 2::4, 2::4][sub].cpu().numpy())
        pred6, pred5 = p.argmax(1), merged.argmax(1)
        if self.offsets is not None:
            o5 = torch.tensor(self.offsets, device=p.device).view(1, 5, 1, 1)
            pred5 = (merged.clamp(min=1e-6).log() + o5).argmax(1)
            pred6 = (p.clamp(min=1e-6).log() + o5[:, [0, 1, 1, 2, 3, 4]]).argmax(1)
        self.cm6 += torch.bincount(y[ok] * 6 + pred6[ok], minlength=36).view(6, 6).cpu().numpy()
        self.cm5 += torch.bincount(y5[ok] * 5 + pred5[ok], minlength=25).view(5, 5).cpu().numpy()
        self.scores.append(merged[:, 1][ok].float().cpu().numpy().astype(np.float32))
        self.positive.append((y5[ok] == 1).cpu().numpy())

    def result(self) -> dict:
        cm6, cm5 = self.cm6.astype(float), self.cm5.astype(float)
        iou6 = np.diag(cm6) / np.maximum(cm6.sum(0) + cm6.sum(1) - np.diag(cm6), 1)
        tp, ref, pred = cm5[1, 1], cm5[1].sum(), cm5[:, 1].sum()
        r, pr = tp / ref, tp / max(pred, 1)
        score, positive = np.concatenate(self.scores), np.concatenate(self.positive)
        order = np.argsort(-score, kind="stable")
        hits = positive[order]
        cum = np.cumsum(hits)
        n, npos = len(hits), int(hits.sum())
        ap = float((cum[hits] / (np.flatnonzero(hits) + 1)).mean())
        budgets = []
        for b in BUDGETS:
            k = int(np.floor(b * n))
            budgets.append({"budget_fraction": b, "flagged_pixels": k, "recall": float(cum[k - 1] / npos), "precision": float(cum[k - 1] / k)})
        return {
            "scored_pixels": int(cm6.sum()), "positive_pixels": npos, "reference_prevalence": npos / n,
            "macro_iou": float(iou6.mean()), "class_iou": dict(zip(CLASSES, iou6.round(4).tolist())),
            "confusion_matrix": self.cm6.tolist(), "combined_confusion_matrix": self.cm5.tolist(),
            "wetland": {"recall": float(r), "precision": float(pr), "f1": float(2 * pr * r / (pr + r)), "f2": float(5 * pr * r / (4 * pr + r)),
                        "iou": float(tp / (ref + pred - tp)), "predicted_area_fraction": float(pred / cm5.sum())},
            "ranking": {"average_precision": ap, "area_budgets": budgets},
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--group", action="append", default=[], help="label=ckpt,ckpt,... scored one by one")
    ap.add_argument("--ensemble", action="append", default=[], help="label=ckpt,ckpt,... probabilities averaged")
    ap.add_argument("--tune", action="store_true", help="also fit decision offsets for each ensemble on this split (use validation)")
    ap.add_argument("--offsets", type=Path, help="output of a --tune run: apply its ensemble offsets here without refitting")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    device = torch.device("cuda")
    loader = DataLoader(RawPatches(args.data, args.split), 16, num_workers=args.workers)
    out = {"split": args.split, "patches": len(loader.dataset), "groups": {}, "ensembles": {},
           "definitions": {"scoring_area": "label != 255 and all 14 input channels valid, full resolution",
                           "combined_decision": "argmax([pOther, pSwamp+pMarsh, pWater, pPeatCropland, pPeatNonCropland])",
                           "test_time_augmentation": "mean of probabilities for the patch and its horizontal flip"}}

    def run(paths: list[str], offsets=None, keep_sample=False):
        """Score each checkpoint alone and their average; returns [member results..., ensemble result]."""
        members = []
        for path in paths:
            ckpt, model, (mean, std) = load_checkpoint(Path(path), device)
            members.append((model.eval(), ckpt["channels"], torch.tensor(mean, device=device).view(1, -1, 1, 1),
                            torch.tensor(std, device=device).view(1, -1, 1, 1), ckpt["args"].get("loss", "softmax") == "ovr"))
        scorers = [Scorer() for _ in members] + [Scorer(offsets, keep_sample)]
        with torch.no_grad():
            for x, y, valid in loader:
                x, y, valid = x.to(device), y.to(device), valid.to(device)
                ok = (y != IGNORE) & valid
                total = 0.0
                for scorer, (model, channels, mean, std, sigmoid) in zip(scorers, members):
                    z = torch.nan_to_num((x[:, channels] - mean) / std)
                    prob = (lambda t: t.float().sigmoid()) if sigmoid else (lambda t: t.float().softmax(1))
                    with torch.autocast("cuda"):
                        p = (prob(model(z)) + prob(model(z.flip(-1))).flip(-1)) / 2
                    scorer.add(p, y, ok)
                    total = total + p
                scorers[-1].add(total / len(members), y, ok)
        del members
        torch.cuda.empty_cache()
        if keep_sample:
            return [s.result() for s in scorers], np.concatenate(scorers[-1].sample_p), np.concatenate(scorers[-1].sample_y)
        return [s.result() for s in scorers]

    def show(label, m):
        w = m["wetland"]
        print(f"{label:34s} macroIoU {m['macro_iou']:.3f} | recall {w['recall']:.3f} precision {w['precision']:.3f} F2 {w['f2']:.3f} IoU {w['iou']:.3f} | "
              f"AP {m['ranking']['average_precision']:.3f} R@5% {m['ranking']['area_budgets'][1]['recall']:.3f} | px {m['scored_pixels']}", flush=True)

    for item in args.group:
        label, paths = item.split("=", 1)
        results = run(paths.split(","))
        out["groups"][label] = {"checkpoints": paths.split(","), "members": results[:-1]}
        for path, m in zip(paths.split(","), results[:-1]):
            show(f"{label} [{Path(path).parent.name}]", m)
        args.out.write_text(json.dumps(out, indent=1))
    for item in args.ensemble:
        label, paths = item.split("=", 1)
        entry = {"checkpoints": paths.split(",")}
        if args.tune:
            results, p5, y5 = run(paths.split(","), keep_sample=True)
            entry["offsets"] = tune_offsets(p5, y5)
            entry["offsets_fit_on"] = args.split
        else:
            results = run(paths.split(","))
            if args.offsets:
                fitted = json.loads(args.offsets.read_text())
                entry["offsets"], entry["offsets_fit_on"] = fitted["ensembles"][label]["offsets"], fitted["split"]
        entry["result"] = results[-1]
        show(label, results[-1])
        if "offsets" in entry:
            entry["result_tuned"] = run(paths.split(","), offsets=entry["offsets"])[-1]
            show(f"{label} (offsets {entry['offsets_fit_on']})", entry["result_tuned"])
        out["ensembles"][label] = entry
        args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
