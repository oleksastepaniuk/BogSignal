"""Report matched provisional temporal variants and their static Sentinel control."""
import argparse
import json
from pathlib import Path

import numpy as np

from .core import build_cache, sha256
from .report import write_report, percent


def compare(runs, output):
    output = Path(output).resolve()
    records, configs, policies, signatures, links = [], [], [], [], []
    for run in map(lambda p: Path(p).resolve(), runs):
        report = write_report(run, output.parent / f"{run.name}.md")
        records.append(json.loads(report.with_name(report.stem + "_metrics.json").read_text()))
        config = json.loads((run / "manifest.json").read_text())["config"]
        configs.append(config)
        policies.append(json.loads((Path(config["dataset"]) / "plan.json").read_text())["temporal_policy"])
        cache = build_cache(config["dataset"], config["channels"]) / "validation"
        signatures.append({name: sha256(cache / name) for name in ["ids.json", "target.npy", "evaluation.npy"]})
        links.append(report.name)
    if [r["experiment"].split("_")[0] for r in records] != ["T0", "T1", "T2", "TB"]:
        raise ValueError("Supply T0, T1, T2 and TB in that order")
    recipes = [{k: v for k, v in c.items() if k not in ["experiment", "dataset", "output", "channels"]} for c in configs]
    if any(r != recipes[0] for r in recipes) or any(s != signatures[0] for s in signatures) or any(p != policies[0] for p in policies):
        raise ValueError("Temporal comparison requires matched recipes, observations, geography, labels and scoring masks")
    names = ["T0: latest provisional date", "T1: temporal mean", "T2: mean + variability", "TB: static S2 + S1 control"]
    wetland = [r["combined_class_evaluation"]["combined_class"]["classes"]["Wetland (Swamp + Marsh)"] for r in records]
    rows, deltas = [], []
    for i, r in enumerate(records):
        rows.append("| " + " | ".join([f"[{names[i]}]({links[i]})", percent(r["validation"]["macro_iou"])] +
            [percent(wetland[i][key]) for key in ["recall", "precision", "dice", "f2", "iou"]]) + " |")
    for new, old, label in [(1, 0, "T1 − T0: averaging"), (2, 1, "T2 − T1: variability features"), (2, 0, "T2 − T0: full temporal representation")]:
        values = [records[new]["validation"]["macro_iou"] - records[old]["validation"]["macro_iou"]]
        values += [wetland[new][key] - wetland[old][key] for key in ["recall", "precision", "f2", "iou"]]
        deltas.append("| " + " | ".join([label] + [f"{100 * v:+.2f}" for v in values]) + " |")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(4)
    for shift, key, label in [(-.25, "recall", "Recall"), (0, "precision", "Reference precision"), (.25, "f2", "F2")]:
        axes[0].bar(x + shift, [100 * w[key] for w in wetland], width=.25, label=label)
    axes[0].set(title="Combined swamp/marsh: ordinary pixel classification", ylabel="Weak-reference score (%)", ylim=(0, 100))
    axes[0].legend(fontsize=8)
    axes[1].bar(x, [100 * r["validation"]["macro_iou"] for r in records], color="#487ca4")
    axes[1].set(title="Six-class segmentation", ylabel="Macro IoU (%)", ylim=(0, 100))
    for ax in axes:
        ax.set(xticks=x, xticklabels=["T0", "T1", "T2", "TB"])
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.tight_layout()
    figure = output.with_suffix(".png")
    fig.savefig(figure, dpi=150)
    plt.close(fig)
    policy = policies[0]
    original = Path(configs[0]["dataset"]).parent
    build = json.loads((original / "build_complete.json").read_text())
    output.write_text(f"""# Monthly provisional temporal comparison — seed 17

## Interpretation

For the recall-focused objective, retain T1 (temporal mean) as the candidate
to confirm across additional seeds. Relative to T0, its combined wetland recall
changes by {100 * (wetland[1]['recall'] - wetland[0]['recall']):+.2f} percentage points,
precision by {100 * (wetland[1]['precision'] - wetland[0]['precision']):+.2f},
F2 by {100 * (wetland[1]['f2'] - wetland[0]['f2']):+.2f} and IoU by
{100 * (wetland[1]['iou'] - wetland[0]['iou']):+.2f}.
In practical terms, T1 recovers about {100 * wetland[1]['recall']:.0f} of every
100 reference wetland pixels, versus {100 * wetland[0]['recall']:.0f} for T0.
Its reference precision means about {100 * wetland[1]['precision']:.0f} of every
100 pixels predicted wetland overlap the weak wetland labels; the remainder
cannot all be assumed to be real errors because those labels are incomplete.

T2 has the highest reference precision, but misses more reference wetland pixels
at the fixed combined-class decision. Its combined wetland average precision
(optional ranking) can be higher while its ordinary-map recall is lower; these
measure different uses of the scores. No threshold was tuned to reverse that
tradeoff. The 0.05-point six-class macro-IoU gap between T1 and T2 provides no
reliable basis for claiming one general segmentation model is better.

Subtype classification remains difficult. T1's original six-class Swamp recall
is {percent(records[1]['validation']['classes']['Swamp']['recall'])} and Marsh
recall is {percent(records[1]['validation']['classes']['Marsh']['recall'])}.
These require the correct subtype and therefore differ from combined wetland
detection. Repeat the matched T0/T1 contrast with additional seeds and review
spatially separated wetland/non-wetland samples before claiming a reliable gain
or opening the locked test set. T2 is a precision-oriented alternative to retain,
rather than the preferred recall-oriented representation in this pilot.

## Matched experiment

Five calendar-month observations per pass/track/frame group were selected from
June–October, using catalogue timing/duration only. Collection:
`{policy['collection']}`; signature `{policy['signature']}`.
Window: {policy['window_utc'][0]} through {policy['window_utc'][1]}.
At least {policy['minimum_observations']} jointly valid HH/HV observations per pixel
and pass are required. Tracks, frames and ascending/descending remain separate.
Original v4 spatial exclusions are retained; they do not certify all dates RFI-free.

Matched patches: {build['summary']['train']['patches']:,} train,
{build['summary']['validation']['patches']:,} validation,
{build['summary']['test']['patches']:,} locked test. All four runs use exactly the
same validation patch IDs, label arrays and scoring masks, checked byte-for-byte.
The common validation mask has {records[0]['validation']['scored_pixels']:,} pixels.
Geographic folds/buffers are inherited from v4. All use seed 17 and the original
training recipe; training-only sampling weights are recalculated on matched labels.
Optical/S1 physical inputs and normalization remain fixed across variants.

T0 uses the latest valid provisional date per anchor pixel. T1 averages valid
linear power before dB conversion and clipping. T2 adds per-date clipped-dB
population standard deviation and p10/p90, computed from the same observations.
TB is a matched static Sentinel reference; it changes modalities relative to the
C-style variants, so a T0/T1/T2 versus TB gap is not a pure incremental NISAR test.
The original validated C results are not directly comparable with this changed
processing regime and eligible footprint. No validated/provisional mixing occurs.

## Ordinary per-pixel results

Wetland means `argmax([pOther, pSwamp+pMarsh, pWater, pPeatCrop, pPeatNonCrop])`.
Every pixel receives a class. There is no ranking or area budget in this table.

| Run | Six-class macro IoU | Wetland recall | Reference precision | Dice / F1 | F2 | Wetland IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

![Temporal comparison]({figure.name})

## Temporal differences, percentage points

| Contrast | Six-class macro IoU | Wetland recall | Reference precision | Wetland F2 | Wetland IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(deltas)}

Checkpoint selection remains six-class validation macro IoU. These are one-seed
weak-reference validation point estimates, not significance or independent
contemporary wetland accuracy. Historical labels, residual provisional artifacts,
irregular observation timing and only 3–5 usable dates limit interpretation.
Percentiles are feature representations, not validated annual hydrological extremes.
The test fold remains unscored. Numerical GPU reproduction checks and actual
per-class deviations are recorded in the individual reports.

Observed date-count histogram across retained pixels/passes:
`{json.dumps(build['observation_histogram'], sort_keys=True)}`.
Individual reports include six-class and combined-class confusion matrices,
learning curves, prediction examples, source hashes and optional ranking analyses.
""")
    output.with_suffix(".json").write_text(json.dumps(dict(records=records, validation_cache_sha256=signatures[0],
        temporal_policy=policy, build_summary=build["summary"], test_evaluated=False), indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs=4, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(compare(args.runs, args.output))


if __name__ == "__main__":
    main()
