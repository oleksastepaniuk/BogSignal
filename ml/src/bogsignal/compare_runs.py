"""Summarise training runs as mean ± std over seeds or folds.

    uv run python -m bogsignal.compare_runs runs/exp            # last.pt on validation
    uv run python -m bogsignal.compare_runs runs/exp --split test --checkpoint best

Runs named ``<config>_s<seed>`` or ``<config>_f<fold>`` are grouped by ``<config>``.
The spread over seeds is the noise floor: differences inside it are not findings.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

COLUMNS = [
    ("swampAP", lambda m: m["ap"]["Swamp"]), ("marshAP", lambda m: m["ap"]["Marsh"]), ("wetAP", lambda m: m["wetland_ap"]),
    ("waterAP", lambda m: m["ap"]["Permanent water"]), ("peatCrAP", lambda m: m["ap"]["Peatland cropland"]),
    ("peatNcAP", lambda m: m["ap"]["Peatland non-cropland"]), ("R@5%", lambda m: m["wetland_recall_at_area"]["5"]),
    ("mIoU", lambda m: m["miou_sub"]), ("mIoUtuned", lambda m: m["miou_tuned"]),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--checkpoint", default="last", choices=["last", "best"])
    ap.add_argument("--json", type=Path, help="also write the table here")
    args = ap.parse_args()

    groups: dict[str, list[dict]] = {}
    for path in sorted(args.root.glob("*/final.json")):
        final = json.loads(path.read_text())[args.checkpoint]
        if args.split in final:
            groups.setdefault(re.sub(r"_[sf]\d+$", "", path.parent.name), []).append(final[args.split])
    print(f"{'config':24s} {'n':>2s} " + " ".join(f"{name:>13s}" for name, _ in COLUMNS))
    table = {}
    for name, runs in groups.items():
        vals = np.array([[get(m) for _, get in COLUMNS] for m in runs])
        mean, std = vals.mean(0), vals.std(0, ddof=1) if len(runs) > 1 else np.zeros(len(COLUMNS))
        table[name] = {"n": len(runs), **{c: [round(float(a), 4), round(float(b), 4)] for (c, _), a, b in zip(COLUMNS, mean, std)}}
        print(f"{name:24s} {len(runs):2d} " + " ".join(f"{a:6.3f} ±{b:5.3f}" for a, b in zip(mean, std)))
    if args.json:
        args.json.write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
