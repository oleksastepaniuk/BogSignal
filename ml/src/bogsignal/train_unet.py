"""Train a segmentation model on the Polissia 14-channel patches (6 classes, ignore 255).

    uv run python -m bogsignal.train_unet --data ../../s1s2nisar_256_v4 --out runs/unet_all

SAR channels stored as linear power are converted to dB (units come from the
dataset's plan.json); normalization statistics are computed on training patches
after that conversion. Missing inputs (NoData) become zero after normalization.

Validation scores an exponential moving average of the weights. ``best.pt`` is
the epoch with the best ``--select`` metric (default: mean swamp/marsh average
precision, which is far less noisy than argmax IoU on this validation set);
``last.pt`` involves no selection at all and is the fairest one to compare runs.

Variants:
    --arch terramind                 TerraMind-small encoder instead of the ResNet U-Net
    --loss ovr                       one sigmoid per class; peat is "unknown" under swamp/marsh/water
    --train-ignore 4,5               drop classes from the training loss only (still scored)
    --train-merge 2:1                train marsh as swamp, i.e. one wetland class (scored on the original labels)
    --sensors s2,s1                  sensor ablation
    --jitter 0.15 --zoom 0.6         radiometric and scale augmentation
    --folds 5 --fold 0               geographic k-fold over train+validation blocks
    --extra DIR --ids DIR/ids.txt    extra per-patch input channels (e.g. InSAR features) on a restricted patch pool

Score a checkpoint on chosen patches, optionally against another dataset
version's labels, or with each sensor/channel blanked out:

    uv run python -m bogsignal.train_unet --data ../../s1s2nisar_256_v3 --out runs/x \
        --eval runs/unet_r34_all/best.pt --ids ids.txt --labels ../../s1s2nisar_256_v4
    uv run python -m bogsignal.train_unet --data ../../s1s2nisar_256_v4 --out runs/x \
        --eval runs/unet_all/last.pt --knockout
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

CLASSES = ["Other", "Swamp", "Marsh", "Permanent water", "Peatland cropland", "Peatland non-cropland"]
IGNORE = 255
SENSOR_CHANNELS = {"s2": range(0, 6), "s1": range(6, 10), "nisar": range(10, 14)}
SAR_START = 6  # channels from here on are linear backscatter
DB_FLOOR, DB_CEIL = -40.0, 15.0
AREA_BUDGETS = (1, 5, 10, 20)  # percent of labelled area
PATCH = 256


NODATA = -9999.0


def to_model_units(x: np.ndarray, linear_sar: bool) -> np.ndarray:
    """Physical units -> S2 reflectance clipped to [0, 1], SAR in clipped dB, missing as NaN."""
    x = x.astype(np.float32)
    missing = ~np.isfinite(x) | (x == NODATA)
    x[missing] = 1.0
    x[:SAR_START] = np.clip(x[:SAR_START], 0.0, 1.0)
    if linear_sar:
        x[SAR_START:] = 10.0 * np.log10(np.maximum(x[SAR_START:], 1e-6))
    x[SAR_START:] = np.clip(x[SAR_START:], DB_FLOOR, DB_CEIL)
    x[missing] = np.nan
    return x


def read_index(root: Path) -> dict[str, dict]:
    with open(root / "index.jsonl") as f:
        return {r["id"]: r for r in map(json.loads, f) if "split" in r}


def policy_weights(records: list[dict], minority_share: float = 0.75, presence: float = 0.05) -> list[float]:
    """The dataset's sampling policy, recomputed for an arbitrary set of patches (used for k-fold).

    ``minority_share`` of draws come from an equal mixture of the five minority classes, each patch weighted by
    the class's share of its trainable pixels when the class covers at least ``presence`` of the patch; the rest
    are uniform over other patches. Reproduces the ``sampling_weight`` values stored in index.jsonl.
    """
    counts = np.array([r["counts"] for r in records], float)
    qual = np.where(counts[:, 1:] / PATCH**2 >= presence, counts[:, 1:] / counts.sum(1, keepdims=True), 0.0)
    present = qual.sum(0) > 0
    w = (qual[:, present] / qual[:, present].sum(0)).sum(1) * minority_share / max(present.sum(), 1)
    background = qual.sum(1) == 0
    w[background] = (1 - minority_share) / max(background.sum(), 1)
    return w.tolist()


def fold_records(root: Path, fold: int, folds: int, ids: set[str] | None = None) -> tuple[list[dict], list[dict]]:
    """Geographic k-fold over the train+validation patches; the test split is never touched.

    Whole split groups (connected wetland blocks) go to one fold, balanced greedily on swamp, marsh and patch
    counts. Training patches that touch a held-out patch are dropped so neighbours never straddle the boundary.
    """
    records = [r for r in read_index(root).values() if r["split"] in ("train", "validation") and (ids is None or r["id"] in ids)]
    feats = json.loads((root / "splits.geojson").read_text())["features"]
    group_of = {tuple(f["properties"]["block"]): f["properties"]["group"] for f in feats}
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(group_of.get(tuple(r["block"]), str(r["block"])), []).append(r)
    load = lambda rs: np.array([sum(r["counts"][1] for r in rs), sum(r["counts"][2] for r in rs), len(rs)], float)  # noqa: E731
    total = load(records)
    sums, members = np.zeros((folds, 3)), [[] for _ in range(folds)]
    for name in sorted(groups, key=lambda g: -(load(groups[g]) / total).sum()):
        k = int(np.argmin((sums / total).sum(1)))
        sums[k] += load(groups[name])
        members[k] += groups[name]
    held = members[fold]
    cells = {(r["row"], r["col"]) for r in held}
    touching = lambda r: any((r["row"] + dr, r["col"] + dc) in cells for dr in (-PATCH, 0, PATCH) for dc in (-PATCH, 0, PATCH))  # noqa: E731
    train = [r for k in range(folds) if k != fold for r in members[k] if not touching(r)]
    return train, held


class Patches(Dataset):
    def __init__(self, root: Path, split: str | None, channels: list[int], stats=None, augment=False,
                 ids: set[str] | None = None, label_root: Path | None = None, records: list[dict] | None = None,
                 sensor_dropout: float = 0.0, train_ignore: tuple[int, ...] = (), jitter: float = 0.0, zoom: float = 0.0,
                 extra_root: Path | None = None, train_merge: dict[int, int] | None = None):
        self.root, self.channels, self.stats, self.augment = root, channels, stats, augment
        self.records = records if records is not None else [
            r for r in read_index(root).values() if (split is None or r["split"] == split) and (ids is None or r["id"] in ids)]
        if not self.records:
            raise ValueError(f"no matching patches in {root / 'index.jsonl'}")
        units = [c["units"] for c in json.loads((root / "plan.json").read_text())["channels"]][SAR_START:]
        if not all(u.startswith(("linear_", "db_")) for u in units) or len({u[:2] for u in units}) != 1:
            raise ValueError(f"unexpected SAR units in plan.json: {units}")
        self.linear_sar = units[0].startswith("linear_")
        self.label_root = label_root or root
        self.label_index = read_index(label_root) if label_root else None
        self.sensor_dropout, self.train_ignore, self.jitter, self.zoom = sensor_dropout, train_ignore, jitter, zoom
        # optional extra input channels stored per patch as <extra_root>/<id>.npy (C, H, W), NaN where missing
        self.extra_root = extra_root
        self.train_merge = train_merge or {}  # class id -> class id it is trained as (labels used for scoring are untouched)
        self.n_extra = len(json.loads((extra_root / "channels.json").read_text())) if extra_root else 0
        if extra_root:
            self.records = [r for r in self.records if (extra_root / f"{r['id']}.npy").exists()]
            if not self.records:
                raise ValueError(f"no patches with extra channels in {extra_root}")
        self.sensor_positions = [p for p in ([channels.index(c) for c in chans if c in channels]
                                             for chans in SENSOR_CHANNELS.values()) if p]

    def __len__(self):
        return len(self.records)

    def sampling_weights(self) -> list[float]:
        key = next((k for k in ("sampling_weight", "training_weight", "weight") if k in self.records[0]), None)
        if key is None or any(key not in r for r in self.records):
            return policy_weights(self.records)
        return [float(r[key]) for r in self.records]

    def __getitem__(self, i):
        rec = self.records[i]
        with rasterio.open(self.root / rec["images"]) as ds:
            x = to_model_units(ds.read(), self.linear_sar)[self.channels]
        if self.extra_root:
            x = np.concatenate([x, np.load(self.extra_root / f"{rec['id']}.npy").astype(np.float32)])
        labels = self.label_index[rec["id"]]["labels"] if self.label_index else rec["labels"]
        with rasterio.open(self.label_root / labels) as ds:
            y = ds.read(1)
        if self.stats is not None:
            x = (x - self.stats[0][:, None, None]) / self.stats[1][:, None, None]
            if self.augment and self.jitter:  # per-channel gain and offset in z-units: mosaic dates and orbits differ by region
                c = (len(x), 1, 1)
                x = x * np.exp(np.random.normal(0, self.jitter, c)).astype(np.float32) + np.random.normal(0, self.jitter, c).astype(np.float32)
            x = np.nan_to_num(x)
        for c, into in self.train_merge.items():
            y[y == c] = into
        for c in self.train_ignore:
            y[y == c] = IGNORE
        if self.augment:
            k = np.random.randint(4)
            x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k)
            if np.random.rand() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
            if self.zoom and np.random.rand() < 0.5:  # random crop enlarged back to full size (up to 1/zoom magnification)
                n = np.random.randint(int(PATCH * self.zoom), PATCH + 1)
                r, c = np.random.randint(0, PATCH - n + 1, 2)
                xt = torch.from_numpy(np.ascontiguousarray(x[:, r:r + n, c:c + n]))[None]
                yt = torch.from_numpy(np.ascontiguousarray(y[r:r + n, c:c + n]))[None, None].float()
                x = F.interpolate(xt, size=PATCH, mode="bilinear", align_corners=False)[0].numpy()
                y = F.interpolate(yt, size=PATCH, mode="nearest")[0, 0].numpy().astype(np.uint8)
            if len(self.sensor_positions) > 1 and np.random.rand() < self.sensor_dropout:
                x = x.copy()
                x[self.sensor_positions[np.random.randint(len(self.sensor_positions))]] = 0.0  # as if the sensor were missing
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y)).long()


def channel_stats(ds: Patches, workers: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std per channel over a stride-4 pixel subsample of every training patch."""
    n, s, ss = 0, 0.0, 0.0
    for x, _ in DataLoader(ds, batch_size=64, num_workers=workers):
        x = x[:, :, ::4, ::4].double()
        n = n + x.isfinite().sum((0, 2, 3))
        s = s + x.nansum((0, 2, 3))
        ss = ss + (x * x).nansum((0, 2, 3))
    mean = s / n
    return mean.numpy().astype(np.float32), (ss / n - mean**2).sqrt().numpy().astype(np.float32)


def dice_loss(logits, y, eps=1.0):
    valid = (y != IGNORE).unsqueeze(1)
    p = logits.float().softmax(1) * valid
    onehot = F.one_hot(torch.where(y == IGNORE, 0, y), logits.shape[1]).permute(0, 3, 1, 2) * valid
    inter, total = (p * onehot).sum((0, 2, 3)), (p + onehot).sum((0, 2, 3))
    present = onehot.sum((0, 2, 3)) > 0
    return (1 - (2 * inter + eps) / (total + eps))[present].mean()


def softmax_loss(logits, y):
    return F.cross_entropy(logits.float(), y, ignore_index=IGNORE) + dice_loss(logits, y)


def ovr_loss(logits, y, eps=1.0):
    """One sigmoid per class (BCE + Dice). The reference labels are exclusive, but peat is a soil property that
    the label hierarchy hides under swamp, marsh and water, so those pixels are unknown for the two peat outputs
    instead of negative."""
    logits = logits.float()
    valid = y != IGNORE
    yy = torch.where(valid, y, 0)
    onehot = F.one_hot(yy, logits.shape[1]).permute(0, 3, 1, 2).float()
    mask = valid.unsqueeze(1).repeat(1, logits.shape[1], 1, 1)
    mask[:, 4:] &= ~((yy >= 1) & (yy <= 3)).unsqueeze(1)
    mask = mask.float()
    onehot = onehot * mask
    bce = (F.binary_cross_entropy_with_logits(logits, onehot, reduction="none") * mask).sum() / mask.sum().clamp(min=1)
    p = logits.sigmoid() * mask
    inter, total = (p * onehot).sum((0, 2, 3)), (p + onehot).sum((0, 2, 3))
    present = onehot.sum((0, 2, 3)) > 0
    return bce + (1 - (2 * inter + eps) / (total + eps))[present].mean()


LOSSES = {"softmax": softmax_loss, "ovr": ovr_loss}


def average_precision(score: np.ndarray, positive: np.ndarray) -> float:
    order = np.argsort(-score)
    hits = positive[order]
    if not hits.any():
        return float("nan")
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(precision[hits].mean())


def class_iou(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    k = len(CLASSES)
    cm = np.bincount(y * k + pred, minlength=k * k).reshape(k, k).astype(np.float64)
    return np.diag(cm) / np.maximum(cm.sum(0) + cm.sum(1) - np.diag(cm), 1)


def tune_bias(p: np.ndarray, y: np.ndarray, rounds: int = 2) -> list[float]:
    """Per-class log-probability offsets that maximise mIoU of argmax(log p + bias) (coordinate ascent).

    Fit on validation only; argmax of the raw probabilities under-predicts the rare classes.
    """
    logp = np.log(np.maximum(p, 1e-6))
    y = y.astype(np.int64)
    bias = np.zeros(p.shape[1], np.float32)
    best = class_iou((logp + bias).argmax(1), y).mean()
    for _ in range(rounds):
        for c in range(1, p.shape[1]):
            for b in np.linspace(-2, 4, 13, dtype=np.float32):
                trial = bias.copy()
                trial[c] = b
                score = class_iou((logp + trial).argmax(1), y).mean()
                if score > best:
                    best, bias = score, trial
    return bias.round(3).tolist()


@torch.no_grad()
def evaluate(model, loader, device, loss: str = "softmax", bias: list[float] | None = None, zero: list[int] | None = None,
             arrays: bool = False):
    """Metrics on a loader. ``bias`` adds tuned-argmax IoU; ``zero`` blanks input channels (sensor knockout)."""
    model.eval()
    k = len(CLASSES)
    cm = torch.zeros(k, k, dtype=torch.long, device=device)
    wet = torch.zeros(2, 2, dtype=torch.long, device=device)  # swamp or marsh vs rest
    probs, labels = [], []
    to_prob = (lambda z: z.float().sigmoid()) if loss == "ovr" else (lambda z: z.float().softmax(1))
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if zero:
            x[:, zero] = 0.0
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            p = (to_prob(model(x)) + to_prob(model(x.flip(-1)).flip(-1))) / 2  # flip test-time augmentation
        pred, ok = p.argmax(1), y != IGNORE
        cm += torch.bincount(y[ok] * k + pred[ok], minlength=k * k).view(k, k)
        wet += torch.bincount(((y[ok] == 1) | (y[ok] == 2)) * 2 + ((pred[ok] == 1) | (pred[ok] == 2)), minlength=4).view(2, 2)
        probs.append(p[:, :, 2::4, 2::4].permute(0, 2, 3, 1).reshape(-1, k).half().cpu())
        labels.append(y[:, 2::4, 2::4].reshape(-1).cpu())
    cm = cm.cpu().numpy().astype(np.float64)
    wet = wet.cpu().numpy().astype(np.float64)
    tp = np.diag(cm)
    iou = tp / np.maximum(cm.sum(0) + cm.sum(1) - tp, 1)
    p, y = torch.cat(probs).float().numpy(), torch.cat(labels).numpy()
    p, y = p[y != IGNORE], y[y != IGNORE]
    out = {
        "iou": dict(zip(CLASSES, iou.round(4).tolist())),
        "precision": dict(zip(CLASSES, (tp / np.maximum(cm.sum(0), 1)).round(4).tolist())),
        "recall": dict(zip(CLASSES, (tp / np.maximum(cm.sum(1), 1)).round(4).tolist())),
        "ap": {c: round(average_precision(p[:, i], y == i), 4) for i, c in enumerate(CLASSES)},
        "miou": round(float(iou.mean()), 4),
        "confusion": cm.astype(int).tolist(),  # rows = reference class, columns = predicted
        "wetland_iou": round(float(wet[1, 1] / max(wet.sum() - wet[0, 0], 1)), 4),
        "wetland_ap": round(average_precision(p[:, 1] + p[:, 2], (y == 1) | (y == 2)), 4),
    }
    out["score"] = round((out["iou"]["Swamp"] + out["iou"]["Marsh"]) / 2, 4)
    out["ap_score"] = round((out["ap"]["Swamp"] + out["ap"]["Marsh"]) / 2, 4)
    # share of reference swamp/marsh pixels inside the top N% of area ranked by swamp+marsh probability
    hits = ((y == 1) | (y == 2))[np.argsort(-(p[:, 1] + p[:, 2]), kind="stable")]
    out["wetland_recall_at_area"] = {str(b): round(float(hits[: round(len(hits) * b / 100)].sum() / max(hits.sum(), 1)), 4)
                                     for b in AREA_BUDGETS}
    if bias is not None:  # stride-4 pixel subsample, so compare with iou_sub rather than iou
        tuned = class_iou((np.log(np.maximum(p, 1e-6)) + np.array(bias, np.float32)).argmax(1), y.astype(np.int64))
        plain = class_iou(p.argmax(1), y.astype(np.int64))
        out["bias"] = bias
        out["iou_sub"] = dict(zip(CLASSES, plain.round(4).tolist()))
        out["iou_tuned"] = dict(zip(CLASSES, tuned.round(4).tolist()))
        out["miou_sub"], out["miou_tuned"] = round(float(plain.mean()), 4), round(float(tuned.mean()), 4)
    return (out, p, y) if arrays else out


SELECT = {"ap": "ap_score", "iou": "score", "wetland_ap": "wetland_ap"}


class Ema:
    """Exponential moving average of the weights; BatchNorm statistics are copied."""

    def __init__(self, model, decay: float):
        self.model, self.decay, self.n = copy.deepcopy(model).eval(), decay, 0
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.lerp_(p.detach(), 1 - d)
        for e, b in zip(self.model.buffers(), model.buffers()):
            e.copy_(b)


def build_model(arch: str, encoder: str, channels: list[int], stats, pretrained: bool, head_dropout: float = 0.0, n_extra: int = 0):
    if arch == "terramind":
        if n_extra:
            raise ValueError("extra channels are only supported with --arch unet")
        from bogsignal.terramind_seg import TerraMindSeg

        model = TerraMindSeg(channels, stats[0], stats[1], len(CLASSES), variant=encoder, pretrained=pretrained)
        model.head_dropout = head_dropout
        return model
    if arch != "unet":
        raise ValueError(f"unknown arch {arch}")
    model = smp.Unet(encoder, encoder_weights="imagenet" if pretrained else None, in_channels=len(channels) + n_extra, classes=len(CLASSES))
    if head_dropout:  # a hook keeps the state dict identical to a plain smp.Unet
        model.segmentation_head.register_forward_pre_hook(lambda m, inp: (F.dropout2d(inp[0], head_dropout, m.training),))
    return model


def load_checkpoint(path: Path, device):
    ckpt = torch.load(path, map_location=device)
    a = ckpt["args"]
    stats = (np.array(ckpt["mean"], np.float32), np.array(ckpt["std"], np.float32))
    model = build_model(a.get("arch", "unet"), a["encoder"], ckpt["channels"], stats, pretrained=False, n_extra=ckpt.get("n_extra", 0)).to(device)
    model.load_state_dict(ckpt["model"])
    return ckpt, model, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="dataset root containing index.jsonl")
    ap.add_argument("--out", type=Path, required=True, help="run directory")
    ap.add_argument("--sensors", default="s2,s1,nisar", help="comma list of s2,s1,nisar (for ablations)")
    ap.add_argument("--arch", default="unet", choices=["unet", "terramind"])
    ap.add_argument("--encoder", default=None, help="default resnet34 (unet) or terramind_v1_small (terramind)")
    ap.add_argument("--loss", default="softmax", choices=list(LOSSES))
    ap.add_argument("--train-ignore", default="", help="comma list of class ids left out of the training loss only")
    ap.add_argument("--train-merge", default="", help="train some classes as others, e.g. 2:1 trains marsh as swamp (one wetland class); scoring labels are unchanged")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--backbone-lr-scale", type=float, default=0.2, help="terramind: backbone lr = lr * this")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--head-dropout", type=float, default=0.0, help="spatial dropout before the final classifier")
    ap.add_argument("--sensor-dropout", type=float, default=0.0, help="probability of blanking one random sensor per sample")
    ap.add_argument("--jitter", type=float, default=0.0, help="std of per-channel random gain/offset in normalised units")
    ap.add_argument("--zoom", type=float, default=0.0, help="smallest random-crop fraction that is enlarged to full size, e.g. 0.6")
    ap.add_argument("--ema", type=float, default=0.998, help="weight EMA decay per step; 0 disables")
    ap.add_argument("--select", default="ap", choices=list(SELECT), help="validation metric that picks best.pt")
    ap.add_argument("--folds", type=int, default=0, help="geographic k-fold over train+validation instead of the fixed split")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test", action="store_true", help="also evaluate best.pt and last.pt on the test split at the end")
    ap.add_argument("--eval", type=Path, help="checkpoint to score instead of training")
    ap.add_argument("--split", default="validation", help="split to score with --eval ('any' for all)")
    ap.add_argument("--ids", type=Path, help="text file of patch ids to restrict to (with --eval, or the pool for --folds)")
    ap.add_argument("--extra", type=Path, help="folder of per-patch extra input channels (<id>.npy and channels.json), e.g. InSAR features")
    ap.add_argument("--labels", type=Path, help="with --eval: dataset root whose labels to score against")
    ap.add_argument("--knockout", action="store_true", help="with --eval: also score with each sensor and channel blanked")
    args = ap.parse_args()
    args.encoder = args.encoder or ("terramind_v1_small" if args.arch == "terramind" else "resnet34")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.eval:
        ckpt, model, stats = load_checkpoint(args.eval, device)
        loss = ckpt["args"].get("loss", "softmax")
        ds = Patches(args.data, None if args.split == "any" else args.split, ckpt["channels"], stats,
                     ids=set(args.ids.read_text().split()) if args.ids else None, label_root=args.labels)
        loader = DataLoader(ds, 32, num_workers=args.workers)
        m = evaluate(model, loader, device, loss, bias=ckpt.get("bias"))
        m["patches"] = len(ds)
        if args.knockout:
            names = json.loads((args.data / "plan.json").read_text())["channels"]
            targets = {s: [ckpt["channels"].index(c) for c in chans if c in ckpt["channels"]] for s, chans in SENSOR_CHANNELS.items()}
            targets.update({names[c]["name"]: [p] for p, c in enumerate(ckpt["channels"])})
            m["knockout"] = {}
            for name, pos in targets.items():
                if pos:
                    k = evaluate(model, loader, device, loss, zero=pos)
                    m["knockout"][name] = {"ap": k["ap"], "wetland_ap": k["wetland_ap"], "ap_score": k["ap_score"], "miou": k["miou"]}
                    print(f"without {name:22s} AP swamp {k['ap']['Swamp']:.3f} marsh {k['ap']['Marsh']:.3f} wetland {k['wetland_ap']:.3f}", flush=True)
        name = f"eval_{args.eval.parent.name}-{args.eval.stem}_{args.data.name}_labels-{(args.labels or args.data).name}.json"
        (args.out / name).write_text(json.dumps(m, indent=1))
        print(name, json.dumps(m))
        return
    channels = [c for s in args.sensors.split(",") for c in SENSOR_CHANNELS[s]]
    train_ignore = tuple(int(c) for c in args.train_ignore.split(",") if c)
    if args.folds:
        pool = set(args.ids.read_text().split()) if args.ids else None
        train_records, val_records = fold_records(args.data, args.fold, args.folds, pool)
        print(f"fold {args.fold}/{args.folds}: {len(train_records)} train, {len(val_records)} held-out patches", flush=True)
    else:
        train_records = val_records = None

    mean, std = channel_stats(Patches(args.data, "train", channels, records=train_records, extra_root=args.extra), args.workers)
    stats = (mean, std)
    print("channels", channels, "\nmean", mean.round(3), "\nstd ", std.round(3), flush=True)

    train = Patches(args.data, "train", channels, stats, augment=True, records=train_records,
                    sensor_dropout=args.sensor_dropout, train_ignore=train_ignore, jitter=args.jitter, zoom=args.zoom, extra_root=args.extra,
                    train_merge={int(a): int(b) for a, b in (m.split(":") for m in args.train_merge.split(",") if m)})
    sampler = WeightedRandomSampler(train.sampling_weights(), len(train), replacement=True)
    seed_worker = lambda i: np.random.seed(args.seed * 1000 + i)  # noqa: E731
    loaders = {
        "train": DataLoader(train, args.batch_size, sampler=sampler, num_workers=args.workers, drop_last=True,
                            persistent_workers=True, pin_memory=True, worker_init_fn=seed_worker),
        "validation": DataLoader(Patches(args.data, "validation", channels, stats, records=val_records, extra_root=args.extra), 32, num_workers=args.workers),
    }

    model = build_model(args.arch, args.encoder, channels, stats, pretrained=True, head_dropout=args.head_dropout, n_extra=train.n_extra).to(device)
    if args.arch == "terramind":
        backbone = list(model.backbone.parameters())
        for p in backbone:
            p.requires_grad_(args.backbone_lr_scale > 0)  # 0 freezes the pretrained encoder
        ids = {id(p) for p in backbone}
        groups = [{"params": backbone, "lr": args.lr * args.backbone_lr_scale},
                  {"params": [p for p in model.parameters() if id(p) not in ids], "lr": args.lr}]
    else:
        groups = [{"params": model.parameters(), "lr": args.lr}]
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    steps = args.epochs * len(loaders["train"])
    warmup = len(loaders["train"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    ema = Ema(model, args.ema) if args.ema else None
    loss_fn = LOSSES[args.loss]
    meta = {"channels": channels, "mean": mean.tolist(), "std": std.tolist(), "classes": CLASSES, "args": {k: str(v) for k, v in vars(args).items()},
            "n_extra": train.n_extra}
    key = SELECT[args.select]

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for x, y in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema:
                ema.update(model)
            running += loss.item()
        scored = ema.model if ema else model
        m = evaluate(scored, loaders["validation"], device, args.loss)
        m.update(epoch=epoch, train_loss=round(running / len(loaders["train"]), 4), seconds=round(time.time() - t0, 1), select=m[key])
        with open(args.out / "metrics.jsonl", "a") as f:
            f.write(json.dumps(m) + "\n")
        state = {"model": scored.state_dict(), "epoch": epoch, "metrics": m, **meta}
        torch.save(state, args.out / "last.pt")
        if m[key] > best:
            best = m[key]
            torch.save(state, args.out / "best.pt")
        print(f"ep {epoch:3d} loss {m['train_loss']:.3f} | IoU swamp {m['iou']['Swamp']:.3f} marsh {m['iou']['Marsh']:.3f} "
              f"wetland {m['wetland_iou']:.3f} mIoU {m['miou']:.3f} | AP swamp {m['ap']['Swamp']:.3f} marsh {m['ap']['Marsh']:.3f} "
              f"wetland {m['wetland_ap']:.3f} | {m['seconds']:.0f}s{' *' if m[key] == best else ''}", flush=True)

    # decision offsets fitted on validation, stored in the checkpoints and applied unchanged to test
    final = {}
    for name in ("best", "last"):
        ckpt, scored, _ = load_checkpoint(args.out / f"{name}.pt", device)
        _, p, y = evaluate(scored, loaders["validation"], device, args.loss, arrays=True)
        ckpt["bias"] = tune_bias(p, y)
        torch.save(ckpt, args.out / f"{name}.pt")
        final[name] = {"validation": evaluate(scored, loaders["validation"], device, args.loss, bias=ckpt["bias"]), "epoch": ckpt["epoch"]}
        if args.test and not args.folds:
            test = DataLoader(Patches(args.data, "test", channels, stats), 32, num_workers=args.workers)
            final[name]["test"] = evaluate(scored, test, device, args.loss, bias=ckpt["bias"])
    (args.out / "final.json").write_text(json.dumps(final, indent=1))
    v = final["last"]["validation"]
    print(f"last epoch: AP swamp {v['ap']['Swamp']:.3f} marsh {v['ap']['Marsh']:.3f} wetland {v['wetland_ap']:.3f} | "
          f"mIoU argmax {v['miou_sub']:.3f} tuned {v['miou_tuned']:.3f} (tuned on this same validation set) | "
          f"best epoch {final['best']['epoch']}", flush=True)
    if "test" in final["last"]:
        (args.out / "test_metrics.json").write_text(json.dumps(final["best"]["test"], indent=1))


if __name__ == "__main__":
    main()
