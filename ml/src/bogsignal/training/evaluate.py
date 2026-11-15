"""Recall-focused validation of a frozen checkpoint; never retrains or scores test."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from .core import CachedPatches, build_cache, make_model, metadata_hashes, metrics, sha256, verify_reproduction
from .train import loader, run_epoch, save_json, seed_everything

SCORE_NAMES = ["Swamp", "Marsh", "Swamp or marsh"]
DEFAULT_BUDGETS = [.01, .05, .10, .20]


def ranking_metrics(labels, scores, budgets=DEFAULT_BUDGETS):
    """Exact non-interpolated AP and conservative whole-tie area budgets.

    Inputs already contain only common valid, non-ignored pixels. All equal-score
    pixels are inseparable by a threshold. Include whole tie groups only when
    they fit the budget; never break ties using labels or spatial ordering.
    """
    labels, scores = np.asarray(labels), np.asarray(scores)
    if (labels.ndim != 1 or scores.shape != labels.shape or not np.isin(labels, [0, 1]).all()
            or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any()):
        raise ValueError("Expected finite 1-D binary labels and probability scores")
    if not budgets or any(not 0 < b <= 1 for b in budgets):
        raise ValueError("Area budgets must be in (0, 1]")
    n, positives = scores.size, int(labels.sum())
    if not n:
        return dict(status="no_scored_pixels", average_precision=None,
                    scored_pixels=0, positive_pixels=0, area_budgets=[], pr_curve=[])
    order = np.argsort(scores, kind="stable")[::-1]
    ordered_scores = scores[order]
    cumulative_tp = np.cumsum(labels[order], dtype="int64")
    ends = np.r_[np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]), n - 1]
    predicted = ends + 1
    tp = cumulative_tp[ends]
    precision = tp / predicted
    recall = tp / positives if positives else np.zeros_like(precision)
    ap = float(np.sum(precision * np.diff(np.r_[0., recall]))) if positives else None
    points = []
    for budget in budgets:
        limit = int(np.floor(n * budget))
        position = int(np.searchsorted(predicted, limit, side="right")) - 1
        selected, found = (int(predicted[position]), int(tp[position])) if position >= 0 else (0, 0)
        points.append(dict(budget_fraction=float(budget), max_pixels=limit,
            flagged_pixels=selected, flagged_area_fraction=selected / n,
            threshold=float(ordered_scores[ends[position]]) if position >= 0 else None,
            decision="score >= threshold" if position >= 0 else "select_none",
            recall=found / positives if positives else None,
            precision=found / selected if selected else None,
            f2=5 * found / (4 * positives + selected) if 4 * positives + selected else None,
            tp=found, fp=selected - found, fn=positives - found))
    # Downsample only the displayed curve, never AP or budget calculations.
    displayed = np.unique(np.linspace(0, len(ends) - 1, min(1024, len(ends)), dtype=int))
    curve = [dict(recall=float(recall[i]), precision=float(precision[i]),
                  threshold=float(ordered_scores[ends[i]])) for i in displayed] if positives else []
    return dict(status="complete" if positives else "no_reference_positives",
                average_precision=ap, scored_pixels=int(n), positive_pixels=positives,
                reference_prevalence=positives / n, area_budgets=points, pr_curve=curve,
                ap_definition="sum of precision at each distinct threshold times recall increment; no interpolation",
                tie_policy="Whole equal-score groups only; flagged pixels never exceed floor(budget * scored pixels)")


def plot_ranking(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, value in result["ranking"].items():
        curve = value["pr_curve"]
        if curve:
            axes[0].plot([0.] + [p["recall"] for p in curve],
                         [1.] + [p["precision"] for p in curve],
                         label=f"{name}: AP {value['average_precision']:.3f}")
        points = value["area_budgets"]
        axes[1].plot([100 * p["budget_fraction"] for p in points],
                     [100 * p["recall"] if p["recall"] is not None else np.nan for p in points],
                     marker="o", label=name)
    axes[0].set(xlabel="Known-positive recall", ylabel="Weak-reference precision",
                title="Validation precision–recall (weak labels)", xlim=(0, 1), ylim=(0, 1))
    axes[1].set(xlabel="Maximum flagged fraction of scored area (%)",
                ylabel="Known-positive recall (%)", title="Recall at fixed area budgets", ylim=(0, 100))
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "recall_evaluation.png", dpi=150)
    plt.close(fig)


def evaluate_run(run, output=None, budgets=DEFAULT_BUDGETS):
    run = Path(run).resolve()
    output = Path(output).resolve() if output else run / "recall_evaluation_v1"
    completed = json.loads((run / "complete.json").read_text())
    original = json.loads((run / "results.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    if sha256(run / "best.pt") != completed["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    if sha256(run / "results.json") != completed["results_sha256"]:
        raise ValueError("Original results hash mismatch")
    config = manifest["config"]
    if metadata_hashes(config["dataset"]) != completed["dataset_hashes"]:
        raise ValueError("Dataset metadata changed since training")
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
    sources = [Path(__file__), Path(__file__).with_name("core.py"), Path(__file__).with_name("train.py")]
    (output / "source").mkdir()
    for source in sources:
        shutil.copy2(source, output / "source" / source.name)
    provenance = dict(checkpoint_sha256=completed["checkpoint_sha256"],
        original_results_sha256=completed["results_sha256"], dataset_hashes=completed["dataset_hashes"],
        source_hashes={p.name: sha256(p) for p in sources}, budgets=list(budgets),
        score_channels=SCORE_NAMES, group_score="softmax(Swamp) + softmax(Marsh)",
        decision_policy="Area budgets apply only to the fixed common scoring mask",
        label_status="Historical/inferred weak references; Other is not verified dry land",
        checkpoint_selection="Original macro-IoU checkpoint retained; no reselection, retraining or threshold optimization",
        test_evaluated=False)
    save_json(output / "manifest.json", provenance)
    seed_everything(config["seed"])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Frozen inference requires CUDA with bfloat16 support")
    cache = build_cache(config["dataset"], config["channels"])
    ds = CachedPatches(cache, "validation")
    model = make_model(len(config["channels"]), config["decoder_channels"], pretrained=False)
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.cuda().to(memory_format=torch.channels_last)
    scores = np.lib.format.open_memmap(output / "validation_scores.npy", mode="w+", dtype="float32",
                                      shape=(len(ds), 3, 256, 256))
    reloaded = run_epoch(model, loader(ds, config), score_maps=scores)
    scores.flush()
    save_json(output / "validation_reproduction.json", dict(
        expected_confusion=original["validation"]["confusion_matrix"],
        actual_confusion=reloaded["confusion_matrix"]))
    reproduction = verify_reproduction(original["validation"]["confusion_matrix"], reloaded["confusion_matrix"])
    target = np.load(ds.folder / "target.npy", mmap_mode="r")
    eligible = np.load(ds.folder / "evaluation.npy", mmap_mode="r").astype(bool)
    eligible &= target != 255
    labels = target[eligible]
    ranked = {}
    for channel, name in enumerate(SCORE_NAMES):
        reference = np.isin(labels, [1, 2]) if channel == 2 else labels == channel + 1
        ranked[name] = ranking_metrics(reference, scores[:, channel][eligible], budgets)
        print(json.dumps({"ranking": name, "average_precision": ranked[name]["average_precision"]}), flush=True)
    result = dict(schema_version=1, validation=reloaded, ranking=ranked, numerical_reproduction=reproduction,
        independently_reviewed_precision_floor=dict(status="not_available",
            reason="No independent reviewed evaluation labels supplied; weak-reference precision is not ground-truth precision"),
        site_detection=dict(status="not_available", reason="Pixel references are not independently annotated wetland objects"),
        provenance=provenance)
    if metadata_hashes(config["dataset"]) != completed["dataset_hashes"]:
        raise ValueError("Dataset metadata changed during evaluation")
    save_json(output / "results.json", result)
    plot_ranking(result, output)
    save_json(output / "complete.json", dict(finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        results_sha256=sha256(output / "results.json"), scores_sha256=sha256(output / "validation_scores.npy"),
        checkpoint_sha256=sha256(run / "best.pt"), original_results_sha256=sha256(run / "results.json")))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output")
    parser.add_argument("--budgets", type=float, nargs="+", default=DEFAULT_BUDGETS)
    args = parser.parse_args()
    print(evaluate_run(args.run, args.output, args.budgets))


if __name__ == "__main__":
    main()
