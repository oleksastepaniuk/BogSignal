"""Out-of-fold prediction maps for k-fold runs: every patch is predicted by the model that did not train on it.

    python -m bogsignal.oof_maps --data ../../s1s2nisar_256_v4 --runs ../../runs/exp --prefix insar_base --folds 5 \
        --ids ../../runs/insar_e1/features_track100_perpair/ids.txt --out ../../runs/insar_e1/oof/base

For each fold k it loads ``<runs>/<prefix>_f<k>/last.pt``, rebuilds the same held-out patch list as training
(``fold_records``) and writes ``<out>/<id>.npz`` with
  cls        combined decision argmax([other, swamp+marsh, water, peat cropland, peat non-cropland]), uint8 0-4
  cls_tuned  the same decision with per-class offsets that lower the bar for the rare classes (see below)
  prob       the five combined probabilities, uint8 0-255
  wetland    swamp+marsh probability, uint8 0-255

The plain argmax under-predicts everything except "other". ``cls_tuned`` is argmax(log p + offset), with the offsets
fitted by ``tune_bias`` to maximise mean IoU. They are cross-fitted: the offsets applied to fold k's patches are fitted
only on the held-out predictions of the other folds, so no patch helps choose its own decision rule.

Predictions use the same horizontal-flip averaging as scoring. Extra input channels are taken from the checkpoint's
own ``--extra`` folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import json

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader

from bogsignal.train_unet import Patches, fold_records, load_checkpoint, read_index, tune_bias

TO_FIVE = np.array([0, 1, 1, 2, 3, 4] + [255] * 250, np.uint8)  # six label classes -> other, wetland, water, peat cropland, peat non-cropland


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--ids", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--checkpoint", default="last.pt")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool = set(args.ids.read_text().split()) if args.ids else None
    args.out.mkdir(parents=True, exist_ok=True)
    total = 0
    for k in range(args.folds):
        ckpt, model, stats = load_checkpoint(args.runs / f"{args.prefix}_f{k}" / args.checkpoint, device)
        model.eval()
        a = ckpt["args"]
        extra = Path(a["extra"]) if a.get("extra") not in (None, "", "None") else None
        _, held_out = fold_records(args.data, k, args.folds, pool)
        ds = Patches(args.data, None, ckpt["channels"], stats, records=held_out, extra_root=extra)
        prob = (lambda z: z.float().sigmoid()) if a.get("loss") == "ovr" else (lambda z: z.float().softmax(1))
        i = 0
        with torch.no_grad():
            for x, _ in DataLoader(ds, 32, num_workers=4):
                x = x.to(device)
                with torch.autocast(device.type, enabled=device.type == "cuda"):
                    p = (prob(model(x)) + prob(model(x.flip(-1)).flip(-1))) / 2
                combined = torch.stack([p[:, 0], p[:, 1] + p[:, 2], p[:, 3], p[:, 4], p[:, 5]], 1)
                cls = combined.argmax(1).to(torch.uint8).cpu().numpy()
                wet = (combined[:, 1].clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
                pr = (combined.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
                for c, w, q in zip(cls, wet, pr):
                    np.savez_compressed(args.out / f"{ds.records[i]['id']}.npz", cls=c, wetland=w, prob=q, fold=k)
                    i += 1
        total += i
        print(f"fold {k}: {i} held-out patches", flush=True)
    print("wrote", total, "patches to", args.out)

    # cross-fitted decision offsets
    index = read_index(args.data)
    files = sorted(args.out.glob("*.npz"))
    fold, ps, ys = [], [], []
    for f in files:
        z = np.load(f)
        with rasterio.open(args.data / index[f.stem]["labels"]) as src:
            y = TO_FIVE[src.read(1)][2::8, 2::8].reshape(-1)
        ok = y != 255
        ps.append(z["prob"][:, 2::8, 2::8].reshape(5, -1).T[ok].astype(np.float32) / 255)
        ys.append(y[ok])
        fold.append(int(z["fold"]))
    fold = np.array(fold)
    offsets = {}
    for k in range(args.folds):
        others = np.flatnonzero(fold != k)
        offsets[k] = tune_bias(np.concatenate([ps[i] for i in others]), np.concatenate([ys[i] for i in others]))
        print(f"fold {k}: offsets fitted on the other folds {offsets[k]}", flush=True)
    for f, k in zip(files, fold):
        z = dict(np.load(f))
        logp = np.log(np.maximum(z["prob"].astype(np.float32) / 255, 1e-6)) + np.array(offsets[int(k)], np.float32)[:, None, None]
        z["cls_tuned"] = logp.argmax(0).astype(np.uint8)
        np.savez_compressed(f, **z)
    (args.out / "offsets.json").write_text(json.dumps({"classes": ["other", "wetland", "water", "peat cropland", "peat non-cropland"], "offsets_by_fold": offsets}, indent=1))


if __name__ == "__main__":
    main()
