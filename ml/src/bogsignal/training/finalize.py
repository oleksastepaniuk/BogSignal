"""Complete deferred checkpoint validation without rewriting training artifacts."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from .core import CachedPatches, build_cache, make_model, metadata_hashes, metrics, sha256, verify_reproduction
from .train import loader, plot_results, run_epoch, save_json, seed_everything


def finalize_run(run, output=None):
    run = Path(run).resolve()
    complete = json.loads((run / "complete.json").read_text())
    original = json.loads((run / "results.json").read_text())
    config = json.loads((run / "manifest.json").read_text())["config"]
    if (sha256(run / "best.pt") != complete["checkpoint_sha256"]
            or sha256(run / "results.json") != complete["results_sha256"]
            or metadata_hashes(config["dataset"]) != complete["dataset_hashes"]):
        raise ValueError("Frozen training artifacts changed")
    output = Path(output).resolve() if output else run / "validation_evaluation_v1"
    output.mkdir(exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
    (output / "source").mkdir()
    sources = [Path(__file__), Path(__file__).with_name("core.py"), Path(__file__).with_name("train.py")]
    for source in sources:
        shutil.copy2(source, output / "source" / source.name)
    seed_everything(config["seed"])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Frozen validation requires CUDA with bfloat16 support")
    ds = CachedPatches(build_cache(config["dataset"], config["channels"]), "validation")
    model = make_model(len(config["channels"]), config["decoder_channels"], pretrained=False)
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=False)["model"])
    model = model.cuda().to(memory_format=torch.channels_last)
    predictions = np.lib.format.open_memmap(output / "validation_predictions.npy", mode="w+",
                                            dtype="uint8", shape=(len(ds), 256, 256))
    validation = run_epoch(model, loader(ds, config), predictions=predictions)
    predictions.flush()
    save_json(output / "validation_reproduction.json", dict(
        expected_confusion=original["validation"]["confusion_matrix"],
        actual_confusion=validation["confusion_matrix"]))
    reproduction = verify_reproduction(original["validation"]["confusion_matrix"], validation["confusion_matrix"])
    background = np.zeros((6, 6), dtype="int64")
    background[:, 0] = np.asarray(validation["confusion_matrix"]).sum(1)
    result = dict(original, validation=validation, majority_other_baseline=metrics(background),
                  validation_source="Frozen checkpoint reloaded and evaluated on validation",
                  post_training_evaluation_deferred=False, checkpoint_predictions_reverified=True,
                  numerical_reproduction=reproduction,
                  limitation="One-seed weak-reference validation; not independent ground truth")
    save_json(output / "results.json", result)
    history = json.loads((run / "history.json").read_text())
    best = next(row for row in history if row["epoch"] == original["best_epoch"])
    plot_results(history, best, ds, predictions, output, config)
    if metadata_hashes(config["dataset"]) != complete["dataset_hashes"]:
        raise ValueError("Dataset metadata changed during validation")
    save_json(output / "complete.json", dict(finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        results_sha256=sha256(output / "results.json"), predictions_sha256=sha256(output / "validation_predictions.npy"),
        checkpoint_sha256=sha256(run / "best.pt"), original_results_sha256=sha256(run / "results.json"),
        dataset_hashes=complete["dataset_hashes"], source_hashes={p.name: sha256(p) for p in sources},
        test_evaluated=False))
    print(json.dumps(dict(run=run.name, validation_macro_iou=validation["macro_iou"],
                         checkpoint_predictions_reverified=True)), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    print(finalize_run(args.run, args.output))


if __name__ == "__main__":
    main()
