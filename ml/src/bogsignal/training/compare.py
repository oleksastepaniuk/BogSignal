"""Generate matched A-D validation reports and a sensor-comparison summary."""
import argparse
import json
from pathlib import Path

import numpy as np

from .report import percent, write_report
from .core import build_cache, sha256


def validate_comparison(records):
    if len(records) != 4 or [r["experiment"].split("_")[0] for r in records] != list("ABCD"):
        raise ValueError("Supply exactly A, B, C and D in order")
    first = records[0]
    for row in records:
        if row["dataset_hashes"] != first["dataset_hashes"] or row["seed"] != first["seed"]:
            raise ValueError("Comparison requires matching dataset hashes and seed")
        if ([c["support"] for c in row["validation"]["classes"].values()] !=
                [c["support"] for c in first["validation"]["classes"].values()]):
            raise ValueError("Different reference support/scoring masks")
        if row["combined_class_evaluation"]["combined_class"]["scored_pixels"] != first["validation"]["scored_pixels"]:
            raise ValueError("Combined evaluation changed the scoring area")
        for name, rank in row["recall_evaluation"]["ranking"].items():
            expected = first["recall_evaluation"]["ranking"][name]
            if (rank["scored_pixels"] != expected["scored_pixels"]
                    or rank["positive_pixels"] != expected["positive_pixels"]
                    or [p["budget_fraction"] for p in rank["area_budgets"]] !=
                       [p["budget_fraction"] for p in expected["area_budgets"]]):
                raise ValueError("Ranking comparison requires matching masks, targets and budgets")


def compare_runs(runs, output):
    output = Path(output).resolve()
    records, links, recipes, cache_signatures = [], [], [], []
    for run in map(lambda p: Path(p).resolve(), runs):
        report = write_report(run, output.parent / f"{run.name}.md")
        records.append(json.loads(report.with_name(report.stem + "_metrics.json").read_text()))
        links.append(report.name)
        config = json.loads((run / "manifest.json").read_text())["config"]
        recipes.append({k: v for k, v in config.items() if k not in ["experiment", "output", "channels"]})
        cache = build_cache(config["dataset"], config["channels"]) / "validation"
        cache_signatures.append({name: sha256(cache / name) for name in ["ids.json", "target.npy", "evaluation.npy"]})
    validate_comparison(records)
    if any(r != recipes[0] for r in recipes) or any(s != cache_signatures[0] for s in cache_signatures):
        raise ValueError("Comparison requires identical recipes, pixel references, patch IDs and scoring masks")
    labels = ["A: S2", "B: S2 + S1", "C: S2 + NISAR", "D: S2 + S1 + NISAR"]
    wet = [r["combined_class_evaluation"]["combined_class"]["classes"]["Wetland (Swamp + Marsh)"] for r in records]
    main_rows, subtype_rows, ranking_rows = [], [], []
    for i, row in enumerate(records):
        main_rows.append("| " + " | ".join([f"[{labels[i]}]({links[i]})", percent(row["validation"]["macro_iou"])] +
            [percent(wet[i][m]) for m in ["recall", "precision", "dice", "f2", "iou"]]) + " |")
        classes = row["validation"]["classes"]
        subtype_rows.append("| " + " | ".join([labels[i]] + [percent(classes[name][m])
            for name in ["Swamp", "Marsh"] for m in ["recall", "precision", "iou"]]) + " |")
        rank = row["recall_evaluation"]["ranking"]["Swamp or marsh"]
        ranking_rows.append("| " + " | ".join([labels[i], percent(rank["average_precision"])] +
            [percent(p["recall"]) for p in rank["area_budgets"]]) + " |")
    delta_rows = []
    for added, base, label in [(3, 1, "D − B: NISAR beyond S2 + S1"), (2, 0, "C − A: NISAR beyond S2"),
                               (1, 0, "B − A: S1 beyond S2"), (3, 2, "D − C: S1 beyond S2 + NISAR")]:
        values = [records[added]["validation"]["macro_iou"] - records[base]["validation"]["macro_iou"]]
        values += [wet[added][m] - wet[base][m] for m in ["recall", "precision", "f2", "iou"]]
        delta_rows.append("| " + " | ".join([label] + [f"{v * 100:+.2f}" for v in values]) + " |")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(4)
    for offset, key, name in [(-.25, "recall", "Recall"), (0, "precision", "Reference precision"), (.25, "f2", "F2")]:
        axes[0].bar(x + offset, [100 * w[key] for w in wet], width=.25, label=name)
    axes[0].set(title="Combined wetland class: normal pixel classification", ylabel="Weak-reference score (%)", ylim=(0, 100))
    axes[0].legend(fontsize=8)
    axes[1].bar(x, [100 * r["validation"]["macro_iou"] for r in records], color="#487ca4")
    axes[1].set(title="Original six-class segmentation", ylabel="Macro IoU (%)", ylim=(0, 100))
    for ax in axes:
        ax.set(xticks=x, xticklabels=list("ABCD"))
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.tight_layout()
    figure = output.with_name(output.stem + ".png")
    fig.savefig(figure, dpi=150)
    plt.close(fig)
    text = f"""# A-D sensor comparison: seed {records[0]['seed']}

All four checkpoints use the same v4 dataset, geographic folds, normalization,
sampling recipe and common scoring mask: **{records[0]['validation']['scored_pixels']:,}
validation pixels**. Each best checkpoint is selected by six-class validation
macro IoU. Cached validation patch IDs, label arrays and pixel masks are verified
byte-for-byte identical across runs; training configurations differ only in the
sensor list, experiment name and output path. Frozen-checkpoint inference is checked against training-time validation;
the largest recorded per-class metric/macro-IoU numerical difference across all
passes is **{100 * max(check['max_absolute_metric_difference'] for r in records for check in r['numerical_reproduction_checks'].values()):.4f}
percentage points**. Exact and tolerance-based matches are distinguished in each
report; original checkpoint hashes and scoring support must match. No retraining or threshold tuning was
performed for this evaluation. Test remains unscored.

## Main comparison: ordinary per-pixel classification

The wetland column sums Swamp + Marsh probabilities before argmax against Other,
water and the two separate peatland classes. Every pixel is classified; this
table has no ranking or area budget. Reference Swamp/Marsh labels are merged in
the same way. Macro IoU still describes the original six-class model output;
the remaining columns describe the combined wetland class.

| Run | Six-class macro IoU | Wetland recall | Reference precision | Dice / F1 | F2 | Wetland IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(main_rows)}

![Sensor comparison]({figure.name})

## Sensor differences (percentage points)

| Contrast | Six-class macro IoU | Wetland recall | Reference precision | Wetland F2 | Wetland IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(delta_rows)}

**D versus B is the primary NISAR comparison.** These are single-seed validation
point estimates, with no confidence intervals or significance claim. Macro-IoU
improvement does not establish improved wetland recall or independent accuracy.
No universal winner is declared across objectives; inspect recall together with
precision and overlap. Repeated seeds and independent reviewed references remain
necessary for headline scientific conclusions.

## Original subtype classification

| Run | Swamp recall | Swamp precision | Swamp IoU | Marsh recall | Marsh precision | Marsh IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(subtype_rows)}

## Optional candidate-ranking analysis

This is separate from ordinary pixel classification above. Pixels are ranked by
the Swamp + Marsh score and selected under a fixed maximum fraction of the common
scored area. It is not accuracy on the full AOI. Budgets and tie policies match
across all four models; AP is exact and non-interpolated.

| Run | Weak-reference AP | Recall at 1% area | Recall at 5% area | Recall at 10% area | Recall at 20% area |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(ranking_rows)}

All metrics use historical/inferred weak labels. Other is not verified dry land;
unmapped wetlands can appear as false positives. The grouped wetland definition
includes only Swamp and Marsh; water and peat classes remain separate. The
comparison does not establish restoration suitability or discovery accuracy.

Individual linked reports contain precision/recall curves, confusion matrices,
class metrics, example predictions, frozen configurations and provenance hashes.
"""
    output.write_text(text)
    output.with_suffix(".json").write_text(json.dumps(dict(
        seed=records[0]["seed"], dataset_hashes=records[0]["dataset_hashes"],
        validation_cache_sha256=cache_signatures[0],
        test_evaluated=False, reports=links,
        runs=[dict(experiment=r["experiment"], validation=r["validation"], combined_wetland=wet[i],
                   checkpoint_sha256=r["checkpoint_sha256"]) for i, r in enumerate(records)]), indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs=4, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(compare_runs(args.runs, args.output))


if __name__ == "__main__":
    main()
