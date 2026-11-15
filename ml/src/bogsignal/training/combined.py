"""Evaluate five-class inference after summing Swamp/Marsh probabilities."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from .core import (CachedPatches, build_cache, make_model, metadata_hashes, sha256,
                   COMBINED_CLASS_NAMES, verify_reproduction)
from .train import loader, run_epoch, save_json, seed_everything


def plot_confusion(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.asarray(result["combined_class"]["confusion_matrix"])
    fractions = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 7))
    display = ax.imshow(fractions, cmap="Blues", vmin=0, vmax=1)
    ax.set(xticks=range(5), yticks=range(5), xticklabels=COMBINED_CLASS_NAMES,
           yticklabels=COMBINED_CLASS_NAMES, xlabel="Predicted", ylabel="Weak reference",
           title="Combined-class inference (row fractions)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for row in range(5):
        for col in range(5):
            ax.text(col, row, f"{fractions[row, col]:.2f}", ha="center", va="center",
                    color="white" if fractions[row, col] > .5 else "black")
    fig.colorbar(display, ax=ax)
    fig.tight_layout()
    fig.savefig(output / "combined_confusion.png", dpi=150)
    plt.close(fig)


def evaluate_combined_run(run, output=None):
    run = Path(run).resolve()
    output = Path(output).resolve() if output else run / "combined_evaluation_v1"
    complete = json.loads((run / "complete.json").read_text())
    original = json.loads((run / "results.json").read_text())
    config = json.loads((run / "manifest.json").read_text())["config"]
    if (sha256(run / "best.pt") != complete["checkpoint_sha256"]
            or sha256(run / "results.json") != complete["results_sha256"]
            or metadata_hashes(config["dataset"]) != complete["dataset_hashes"]):
        raise ValueError("Frozen experiment provenance mismatch")
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
    sources = [Path(__file__), Path(__file__).with_name("core.py"), Path(__file__).with_name("train.py")]
    (output / "source").mkdir()
    for source in sources:
        shutil.copy2(source, output / "source" / source.name)
    provenance = dict(checkpoint_sha256=complete["checkpoint_sha256"],
        original_results_sha256=complete["results_sha256"], dataset_hashes=complete["dataset_hashes"],
        source_hashes={p.name: sha256(p) for p in sources}, checkpoint_epoch=original["best_epoch"],
        class_names=COMBINED_CLASS_NAMES, reference_mapping=[0, 1, 1, 2, 3, 4], ignore_index=255,
        decision="argmax([pOther, pSwamp+pMarsh, pWater, pPeatCropland, pPeatNonCropland])",
        ties="First class in the declared order wins exact ties (torch.argmax)",
        evaluation="Original common label-eligible all-14-channel valid validation mask",
        retrained=False, test_evaluated=False)
    save_json(output / "manifest.json", provenance)
    seed_everything(config["seed"])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Frozen inference requires CUDA with bfloat16 support")
    ds = CachedPatches(build_cache(config["dataset"], config["channels"]), "validation")
    model = make_model(len(config["channels"]), config["decoder_channels"], pretrained=False)
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=False)["model"])
    model = model.cuda().to(memory_format=torch.channels_last)
    predictions = np.lib.format.open_memmap(output / "validation_predictions.npy", mode="w+",
        dtype="uint8", shape=(len(ds), 256, 256))
    reloaded = run_epoch(model, loader(ds, config), evaluate_combined=True, combined_predictions=predictions)
    predictions.flush()
    reproduction = verify_reproduction(original["validation"]["confusion_matrix"], reloaded["confusion_matrix"])
    if reloaded["combined_class"]["scored_pixels"] != original["validation"]["scored_pixels"]:
        raise ValueError("Combined evaluation changed the scoring area")
    result = dict(schema_version=1, combined_class=reloaded["combined_class"],
                  numerical_reproduction=reproduction,
                  original_six_class_confusion=reloaded["confusion_matrix"], provenance=provenance,
                  limitation="Weak-reference agreement; inference-only class merging, not combined-class training")
    if metadata_hashes(config["dataset"]) != complete["dataset_hashes"]:
        raise ValueError("Dataset changed during evaluation")
    save_json(output / "results.json", result)
    plot_confusion(result, output)
    save_json(output / "complete.json", dict(finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        results_sha256=sha256(output / "results.json"), predictions_sha256=sha256(output / "validation_predictions.npy"),
        checkpoint_sha256=sha256(run / "best.pt"), original_results_sha256=sha256(run / "results.json")))
    print(json.dumps(result["combined_class"]), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    print(evaluate_combined_run(args.run, args.output))


if __name__ == "__main__":
    main()
