"""Score the average of several checkpoints' class probabilities.

    uv run python -m bogsignal.ensemble --data ../../s1s2nisar_256_v4 --out runs/exp/ens.json \
        runs/exp/base_s0/last.pt runs/exp/base_s1/last.pt runs/exp/tm_s0/last.pt

Members may differ in architecture and sensors. Metrics use the same stride-4
pixel grid as ``train_unet`` validation, so average precision is comparable.
Each member is also scored alone, because an ensemble gain says nothing about
which input or model produced it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from bogsignal.baseline_gbm import score
from bogsignal.train_unet import Patches, evaluate, load_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+", type=Path)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total, members, y = 0.0, {}, None
    for path in args.checkpoints:
        ckpt, model, stats = load_checkpoint(path, device)
        loader = DataLoader(Patches(args.data, args.split, ckpt["channels"], stats), 32, num_workers=args.workers)
        _, p, y = evaluate(model, loader, device, ckpt["args"].get("loss", "softmax"), arrays=True)
        total = total + p
        members[str(path)] = score(p, y)
        m = members[str(path)]
        print(f"{str(path):40s} AP swamp {m['ap']['Swamp']:.3f} marsh {m['ap']['Marsh']:.3f} wetland {m['wetland_ap']:.3f}", flush=True)
    m = score(total / len(args.checkpoints), y)
    print(f"{'ensemble of ' + str(len(members)):40s} AP swamp {m['ap']['Swamp']:.3f} marsh {m['ap']['Marsh']:.3f} wetland {m['wetland_ap']:.3f}")
    if args.out:
        args.out.write_text(json.dumps({"ensemble": m, "members": members}, indent=1))


if __name__ == "__main__":
    main()
