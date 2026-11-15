"""Publish a compact, reproducible validation report from a completed run."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from .core import CLASS_NAMES, metadata_hashes, metrics, sha256, verify_reproduction


def percent(value):
    return f"{100 * value:.2f}%" if value is not None else "n/a"


def completed_version(run, prefix):
    candidates = [p for p in run.glob(prefix + "_v*")
                  if p.name.rsplit("_v", 1)[-1].isdigit() and (p / "complete.json").exists()]
    return max(candidates, key=lambda p: int(p.name.rsplit("_v", 1)[-1])) if candidates else run / (prefix + "_v1")


def combined_section(val, combined, stem):
    if combined is None:
        return ""
    scores = combined["combined_class"]
    wetland = scores["classes"]["Wetland (Swamp + Marsh)"]
    old = val["wetland_detection"]
    rows = []
    for label, key in [("Recall", "recall"), ("Reference precision", "precision"),
                       ("Dice / F1", "dice"), ("F2", "f2"), ("IoU", "iou")]:
        rows.append(f"| {label} | {percent(old[key])} | {percent(wetland[key])} |")
    classes = []
    for name, row in scores["classes"].items():
        classes.append("| " + " | ".join([name] + [percent(row[k]) for k in
                       ["iou", "dice", "precision", "recall", "f2"]]) + " |")
    return f"""## Combined wetland class: normal per-pixel inference

Every pixel receives one of **five classes**: Other, Wetland (Swamp + Marsh),
Permanent water, Peatland cropland or Peatland non-cropland. The decision is:

`argmax([pOther, pSwamp + pMarsh, pWater, pPeatCropland, pPeatNonCropland])`

There is **no ranking, area budget or extra confidence threshold**. Scores are
combined before choosing the class. Reference Swamp and Marsh labels are also
merged into Wetland; water and peat remain separate, and ignore 255 stays ignored.
The same original checkpoint and {scores['scored_pixels']:,} scoring pixels are
used, without retraining or changing the original six-class result.

The comparison below distinguishes merging labels *after* the original six-class
decision from merging scores *before* the five-class decision:

| Wetland metric | Original argmax, then merge Swamp/Marsh | Sum scores, then argmax |
| --- | ---: | ---: |
{chr(10).join(rows)}
| Area classified Wetland | {percent(old['flagged_area_fraction'])} | {percent(scores['wetland_area_fraction'])} |

The new five-class map has macro IoU **{percent(scores['macro_iou'])}** and
macro Dice **{percent(scores['macro_dice'])}**. These averages use five classes
and must not be directly compared with the original six-class macro averages.

| Combined inference class | IoU | Dice / F1 | Reference precision | Recall | F2 |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(classes)}

![Combined-class confusion matrix]({stem}_combined_confusion.png)

This is an inference-only change, not a model trained on five-class labels.
Precision, recall and overlap scores still measure agreement with weak references.
The original six-class checkpoint-selection criterion and the unscored test fold
remain unchanged. This combined per-pixel map is distinct from the optional
candidate-ranking analysis below.

"""


def recall_section(val, evaluation, stem):
    grouped = val["wetland_detection"]
    rows = []
    for name in ["Swamp", "Marsh"]:
        c = val["classes"][name]
        rows.append(f"| {name}, correct subtype | {percent(c['recall'])} | {percent(c['precision'])} | {percent(c['f2'])} |")
    rows.append(f"| Swamp or marsh, either subtype accepted | {percent(grouped['recall'])} | {percent(grouped['precision'])} | {percent(grouped['f2'])} |")
    section = f"""## Recall-focused weak-reference evaluation

These are **known-reference-positive recall** and **weak-reference precision/F2**,
not accuracy against independently established contemporary wetlands. Missing
wetlands in Other can be counted as apparent false positives. Wrong, historical
or geographically biased positive labels can also distort recall. Higher recall
alone is insufficient: flagging everything would recover all mapped positives.

### Original six-class argmax decisions

| Target | Recall | Reference precision | F2 |
| --- | ---: | ---: | ---: |
{chr(10).join(rows)}

Swamp/marsh macro subtype recall (equal weight per class):
**{percent(val['swamp_marsh_macro_recall'])}**. Their mean subtype IoU is
**{percent(val['swamp_marsh_mean_iou'])}**, excluding Other, water and peatland
classes. Grouped detection counts a Swamp-to-Marsh prediction as a detection,
while subtype metrics count it as an error. Water and both peatland classes
remain separately reported; they are not automatically current-wetland positives.
The argmax grouped map flags **{percent(grouped['flagged_area_fraction'])}** of
the scored area. F2 is `5 TP / (5 TP + 4 FN + FP)` and emphasizes omissions;
it does not correct label errors.

"""
    if evaluation is None:
        return section + "Ranking metrics have not been computed. Run the checkpoint evaluation command in the training instructions to add AP, PR curves and area-budget recall.\n\n"
    budgets = next(iter(evaluation["ranking"].values()))["area_budgets"]
    budget_text = ", ".join(f"{100 * p['budget_fraction']:g}%" for p in budgets)
    headers = [f"Recall at {100 * p['budget_fraction']:g}% area" for p in budgets]
    table = ["| Target score | Reference prevalence | Average precision | " + " | ".join(headers) + " |",
             "| --- | ---: | ---: | " + " | ".join(["---:"] * len(headers)) + " |"]
    for name, value in evaluation["ranking"].items():
        table.append("| " + " | ".join([name, percent(value["reference_prevalence"]),
                      percent(value["average_precision"])] + [percent(p["recall"]) for p in value["area_budgets"]]) + " |")
    operating = []
    for p in evaluation["ranking"]["Swamp or marsh"]["area_budgets"]:
        operating.append("| " + " | ".join([percent(p["budget_fraction"]), percent(p["flagged_area_fraction"]),
            percent(p["recall"]), percent(p["precision"]), percent(p["f2"]),
            f"{p['threshold']:.6f}" if p["threshold"] is not None else "select none"]) + " |")
    return section + f"""### Scores ranked at fixed review-area budgets

The **same original epoch-{evaluation['checkpoint_epoch']} checkpoint** was run
in inference mode, without retraining or selecting another checkpoint. Its
six-class confusion matrix is checked against the original result. Each score
is the corresponding softmax output; the grouped score sums Swamp + Marsh.
Grouped thresholding differs from grouping the original argmax decisions, so
these are different operating points on the same model. Softmax scores have not
been independently calibrated as real-world wetland probabilities.

{chr(10).join(table)}

Budgets of {budget_text} were fixed for the evaluation, not selected to
maximize recall. Each budget applies separately to each target score and counts
only label-eligible pixels on the fixed common scoring mask, not the entire AOI.
The threshold uses the score distribution without consulting labels. Whole tied
score groups are included only if they fit; actual flagged area never exceeds
the budget. The same footprint and budgets must be used for A-D comparisons.

For the combined Swamp-or-Marsh candidate map:

| Maximum area | Actual flagged area | Known-positive recall | Reference precision | F2 | Score threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(operating)}

AP is the exact non-interpolated sum of precision times recall increments over
all distinct score thresholds. Only the plotted curve is downsampled. AP, F2 and
precision remain weak-reference agreement measures; no ground-truth precision
claim follows from them.

![Recall evaluation]({stem}_recall_evaluation.png)

**Recall at an independently reviewed precision floor is not available:** no
reviewed evaluation labels have been supplied. Site/object recall is also not
computed from patch IDs. A reviewed evaluation sample must include detections,
predicted negatives, known wetlands and dry controls, with geographically
separate tuning/evaluation samples and sampling-aware analysis. Record uncertain
cases explicitly. Do not choose a precision floor or label corrections using the
locked test fold.

The original macro-IoU checkpoint selection and training recipe remain fixed for
the initial sensor comparison. Recall-focused checkpoint selection or a changed
loss/supervision policy belongs to a separately declared experiment.

"""


def write_report(run, output, evaluation_path=None):
    run, output = Path(run).resolve(), Path(output).resolve()
    complete = json.loads((run / "complete.json").read_text())
    if sha256(run / "results.json") != complete["results_sha256"]:
        raise ValueError("Results changed since completion")
    if sha256(run / "best.pt") != complete["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed since completion")
    manifest = json.loads((run / "manifest.json").read_text())
    result = json.loads((run / "results.json").read_text())
    figures_path = run
    if result.get("post_training_evaluation_deferred"):
        figures_path = completed_version(run, "validation_evaluation")
        finalized = json.loads((figures_path / "complete.json").read_text())
        if (finalized["checkpoint_sha256"] != complete["checkpoint_sha256"]
                or finalized["original_results_sha256"] != complete["results_sha256"]
                or finalized["dataset_hashes"] != complete["dataset_hashes"]
                or sha256(figures_path / "results.json") != finalized["results_sha256"]):
            raise ValueError("Final validation provenance mismatch")
        result = json.loads((figures_path / "results.json").read_text())
        if result.get("post_training_evaluation_deferred") or not result.get("checkpoint_predictions_reverified"):
            raise ValueError("Final validation is incomplete")
    history = json.loads((run / "history.json").read_text())
    config = manifest["config"]
    experiment_id = config["experiment"].split("_")[0]
    sensor_names = " + ".join(sensor for prefix, sensor in
        [("S2_", "Sentinel-2"), ("S1_", "Sentinel-1"), ("NISAR_", "NISAR")]
        if any(name.startswith(prefix) for name in config["channels"]))
    config_name = next((run / "source").glob("*.json")).name
    if metadata_hashes(config["dataset"]) != complete["dataset_hashes"]:
        raise ValueError("Dataset metadata changed since experiment")
    recomputed = metrics(result["validation"]["confusion_matrix"])
    if abs(recomputed["macro_iou"] - result["validation"]["macro_iou"]) > 1e-12:
        raise ValueError("Saved metric does not match confusion counts")
    val = dict(result["validation"], **recomputed)
    majority = dict(result["majority_other_baseline"], **metrics(result["majority_other_baseline"]["confusion_matrix"]))
    evaluation_path = Path(evaluation_path).resolve() if evaluation_path else completed_version(run, "recall_evaluation")
    evaluation = None
    if evaluation_path.exists():
        e_complete = json.loads((evaluation_path / "complete.json").read_text())
        if (e_complete["checkpoint_sha256"] != complete["checkpoint_sha256"]
                or e_complete["original_results_sha256"] != complete["results_sha256"]
                or sha256(evaluation_path / "results.json") != e_complete["results_sha256"]):
            raise ValueError("Recall evaluation provenance mismatch")
        evaluation = json.loads((evaluation_path / "results.json").read_text())
        verify_reproduction(val["confusion_matrix"], evaluation["validation"]["confusion_matrix"])
        if evaluation["provenance"]["dataset_hashes"] != complete["dataset_hashes"]:
            raise ValueError("Recall evaluation uses a different benchmark")
        evaluation["checkpoint_epoch"] = result["best_epoch"]
    combined = None
    combined_path = completed_version(run, "combined_evaluation")
    if combined_path.exists():
        c_complete = json.loads((combined_path / "complete.json").read_text())
        if (c_complete["checkpoint_sha256"] != complete["checkpoint_sha256"]
                or c_complete["original_results_sha256"] != complete["results_sha256"]
                or sha256(combined_path / "results.json") != c_complete["results_sha256"]):
            raise ValueError("Combined evaluation provenance mismatch")
        combined = json.loads((combined_path / "results.json").read_text())
        verify_reproduction(val["confusion_matrix"], combined["original_six_class_confusion"])
        if (combined["provenance"]["dataset_hashes"] != complete["dataset_hashes"]
                or combined["combined_class"]["scored_pixels"] != val["scored_pixels"]):
            raise ValueError("Combined evaluation uses a different benchmark")
    training_reference = json.loads((run / "results.json").read_text())["validation"]["confusion_matrix"]
    numerical_checks = {"final_validation": verify_reproduction(training_reference, val["confusion_matrix"])}
    if evaluation is not None:
        numerical_checks["ranking"] = verify_reproduction(training_reference, evaluation["validation"]["confusion_matrix"])
    if combined is not None:
        numerical_checks["combined"] = verify_reproduction(training_reference, combined["original_six_class_confusion"])
    maximum_drift = max(c["max_absolute_metric_difference"] for c in numerical_checks.values())
    exact_count = sum(c["exact_confusion_match"] for c in numerical_checks.values())
    plan = json.loads((Path(config["dataset"]) / "plan.json").read_text())
    split_counts = {s: sum(r["split"] == s for r in plan["patches"]) for s in ["train", "validation", "test"]}
    temporal = plan.get("temporal_policy")
    temporal_text = ""
    if temporal:
        temporal_text = f"""## Temporal input policy

Variant: **{plan['temporal_variant']}**, on inherited v4 geographic groups/folds.
Collection: `{temporal['collection']}`, processing signature
`{temporal['signature']}`. Window: {temporal['window_utc'][0]} through
{temporal['window_utc'][1]}. At least **{temporal['minimum_observations']} valid
joint HH/HV dates per pixel and pass** are required. Ascending/descending and
anchor tracks/frames remain separate. All temporal variants share patch IDs,
labels, masks and sampling weights. Optical and Sentinel-1 values remain fixed.

T0 uses the latest valid provisional observation per pixel. T1 takes the mean
of valid linear power before dB conversion/clipping. T2 adds population standard
deviation and p10/p90 of per-date clipped dB. Normalization is fitted only on the
matched training fold. Original validated C is a different dataset/processing
regime and must not serve as the direct temporal baseline. Historical weak-label
and residual radar-quality limitations still apply. No validated/provisional
products are mixed in the temporal stack.

"""
    comparison_text = ("Primary temporal contrasts are T1 minus T0 (averaging) and T2 minus T1 (variability representation); the matched static B control is separate."
                       if temporal else "The primary NISAR contrast is D versus B.")
    if experiment_id == "TD":
        comparison_text = ("Compare TD with the existing D checkpoint on common validation pixels. "
                           "Processing versions and training eligibility differ, so this is not an isolated test of averaging.")
    validation_records = [r for r in plan["patches"] if r["split"] == "validation"]
    eligible_labels = sum(sum(r["counts"]) for r in validation_records)
    rows = []
    for name in CLASS_NAMES:
        c = val["classes"][name]
        rows.append("| " + " | ".join([name] + [f"{100 * c[m]:.2f}%" if c[m] is not None else "n/a"
                     for m in ["iou", "dice", "precision", "recall", "f2"]] + [f"{c['support']:,}"]) + " |")
    epoch_seconds = [r["train"]["seconds"] + r["validation"]["seconds"] for r in history]
    output.parent.mkdir(parents=True, exist_ok=True)
    stem = output.stem
    for figure in ["learning_curves", "confusion_matrix", "validation_examples"]:
        shutil.copy2(figures_path / f"{figure}.png", output.parent / f"{stem}_{figure}.png")
    if evaluation is not None:
        shutil.copy2(evaluation_path / "recall_evaluation.png", output.parent / f"{stem}_recall_evaluation.png")
    if combined is not None:
        shutil.copy2(combined_path / "combined_confusion.png", output.parent / f"{stem}_combined_confusion.png")
    review_metrics = dict(result, validation=val, majority_other_baseline=majority,
                          numerical_reproduction_checks=numerical_checks,
                          checkpoint_sha256=complete["checkpoint_sha256"],
                          dataset_hashes=complete["dataset_hashes"],
                          manifest_sha256=sha256(run / "manifest.json"))
    if figures_path != run:
        review_metrics["final_validation_results_sha256"] = sha256(figures_path / "results.json")
    if evaluation is not None:
        review_metrics["recall_evaluation"] = dict(
            ranking={name: {k: v for k, v in value.items() if k != "pr_curve"}
                     for name, value in evaluation["ranking"].items()},
            provenance=evaluation["provenance"],
            numerical_reproduction=evaluation.get("numerical_reproduction"),
            evaluation_results_sha256=sha256(evaluation_path / "results.json"),
            independently_reviewed_precision_floor=evaluation["independently_reviewed_precision_floor"],
            site_detection=evaluation["site_detection"])
    if combined is not None:
        review_metrics["combined_class_evaluation"] = dict(combined,
            evaluation_results_sha256=sha256(combined_path / "results.json"))
    (output.parent / f"{stem}_metrics.json").write_text(json.dumps(review_metrics, indent=2) + "\n")
    report = f"""# Experiment {experiment_id}: {sensor_names} ResNet-34 U-Net

Training completed: {complete['finished_utc']}. This is a **one-seed screening
experiment**, evaluated against v4 weak references on the geographic validation
fold. The test fold was not scored. Sensor comparisons, seed uncertainty and
independent contemporary wetland accuracy are not established by this run.

## Result

| Validation metric | U-Net | Always predict Other |
| --- | ---: | ---: |
| Six-class macro IoU | {100 * val['macro_iou']:.2f}% | {100 * majority['macro_iou']:.2f}% |
| Six-class macro Dice | {100 * val['macro_dice']:.2f}% | {100 * majority['macro_dice']:.2f}% |
| Pixel accuracy | {100 * val['accuracy']:.2f}% | {100 * majority['accuracy']:.2f}% |

Best checkpoint: epoch **{result['best_epoch']}**, selected only by validation
macro IoU; {result['epochs_completed']} epochs completed, termination:
`{result['stopped']}`. Reloaded checkpoint validation was checked against the selected
epoch's recorded results. The always-Other baseline uses the same
scored pixels and demonstrates why pixel accuracy alone is insufficient.

Numerical reproduction: **{exact_count} of {len(numerical_checks)} inference checks
match the training-time confusion counts exactly**. The largest absolute
per-class metric/macro-IoU difference is **{maximum_drift * 100:.4f} percentage
points**, below the 0.1-point engineering tolerance. Identical checkpoint hashes,
reference support and scored area are required. BF16/cuDNN benchmarking is not
bitwise deterministic; non-exact confusion counts are retained in the metrics
JSON and are not described as exact matches. Original training artifacts and
incomplete earlier attempts are preserved. This tolerance is an artifact
reproduction check, not a confidence interval or model-selection criterion.

Macro IoU improves by {100 * (val['macro_iou'] - majority['macro_iou']):.2f}
percentage points over always-Other. Pixel accuracy differs by
{100 * (val['accuracy'] - majority['accuracy']):+.2f} percentage points.
Wetland-class recall remains weak: {100 * np.asarray(val['confusion_matrix'])[1, 0] / val['classes']['Swamp']['support']:.1f}%
of reference Swamp pixels and {100 * np.asarray(val['confusion_matrix'])[2, 0] / val['classes']['Marsh']['support']:.1f}%
of reference Marsh pixels are predicted as Other. This establishes a working
sensor baseline with substantial minority-class errors; it is not a strong
wetland delineation result.

| Class | IoU | Dice / F1 | Reference precision | Recall | F2 | Reference pixels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

The common scoring mask retains **{val['scored_pixels']:,} pixels**
({val['scored_pixels'] * .0004:,.2f} km² at 20 m), or
{100 * val['scored_pixels'] / eligible_labels:.2f}% of label-eligible validation
pixels. Both unknown labels and pixels missing any of the 14 baseline predictors
are excluded from scores. Radar validity is used solely for this shared scoring
area. Model inputs contain only the configured sensors and their validity masks.

{combined_section(val, combined, stem)}
{recall_section(val, evaluation, stem)}
{temporal_text}
## Frozen setup

- Dataset: `{Path(config['dataset']).name}`; {split_counts['train']:,} training patches,
  {split_counts['validation']:,} validation patches, {split_counts['test']:,} locked test patches.
  Geographic groups and buffers inherited from v4.
- Inputs: {', '.join(config['channels'])}, plus their corresponding validity masks:
  **{manifest['input_channels']} model channels**.
  Dataset training-only normalization; missing normalized values zero-filled.
- U-Net, ResNet-34 ImageNet encoder, decoder widths 256/128/64/32/16;
  **{manifest['parameters']:,} parameters**, six output logits. RGB kernels
  repeated/scaled for imagery; mask kernels zero-initialized; all layers trainable.
- Seed {config['seed']}; batch {config['batch_size']}; bfloat16; AdamW encoder LR
  {config['encoder_lr']}, decoder LR {config['decoder_lr']}, weight decay
  {config['weight_decay']}; cosine LR decay; gradient norm clipped at 1.0.
- Unweighted cross-entropy ignores 255. Frozen training patch-weighted sampling
  with replacement; flips and quarter-turns. Validation uniform and unaugmented.
- Maximum {config['epochs']} epochs, minimum {config['min_epochs']}, patience
  {config['patience']}. No hyperparameter changes were made in response to scores.
- {manifest['gpu']}; PyTorch {manifest['versions']['torch']}, torchvision
  {manifest['versions']['torchvision']}, SMP {manifest['versions']['segmentation-models-pytorch']}.
  Seeded CUDA/cuDNN benchmarking is not a guarantee of bitwise determinism.

Recorded training-run time:
**{result['training_seconds'] / 60:.2f} minutes** (excludes setup, cache creation,
the separate smoke check, deferred evaluation and plot rendering). A's original
run time also includes its immediate checkpoint reverification.
Median train-plus-validation epoch:
**{float(np.median(epoch_seconds)):.2f} seconds**; peak allocated GPU memory:
**{result['peak_vram_gib']:.2f} GiB** (allocator peak, not total device reservation).

## Figures and interpretation

![Learning curves]({stem}_learning_curves.png)

Training curves use weighted/augmented sampled training patches; validation uses
the natural geographic holdout. Their scores are not directly comparable as an
unbiased estimate of the train/validation gap.

![Validation confusion matrix]({stem}_confusion_matrix.png)

![Reference-selected validation examples]({stem}_validation_examples.png)

Examples are chosen by reference class abundance (one rich patch for each
minority class, plus a background-rich patch), not prediction quality. RGB has a
fixed display stretch. White label/prediction pixels are outside the scoring mask.
Class colors: gray Other, dark green Swamp, light green Marsh, blue Permanent
water, ochre Peatland cropland, purple Peatland non-cropland.

Historical/inferred weak labels can disagree with actual 2026 conditions. Peatland
cropland and non-cropland are separate targets, not automatically current wetlands
or restoration candidates. Use swamp/marsh error patterns to guide a separate
independent annotation effort; do not revise held-out labels based on predictions
and then present the revised scores as an untouched benchmark.

## Reproduction and artifacts

See [training instructions](../../ml/TRAINING.md) and the frozen
[configuration](../../ml/configs/{config_name}).
Local run: `{run}`. It contains source snapshots, package/config manifests,
checkpoint, full learning history and validation predictions. Generated models
and caches remain ignored by Git. [Reviewable metrics]({stem}_metrics.json)
include the confusion counts and dataset/checkpoint hashes.

Use the same splits, labels, sampling policy, recipe
and scoring mask. {comparison_text} Single-seed validation
differences do not establish statistical significance or independent accuracy.
"""
    output.write_text(report)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluation", help="Optional separately versioned recall evaluation folder")
    args = parser.parse_args()
    print(write_report(args.run, args.output, args.evaluation))


if __name__ == "__main__":
    main()
