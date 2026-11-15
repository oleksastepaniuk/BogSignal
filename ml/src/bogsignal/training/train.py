"""Train a frozen sensor-ablation run; choose checkpoints on validation only."""
import argparse
import contextlib
import datetime as dt
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from .core import (CLASS_NAMES, CachedPatches, build_cache, confusion_counts,
                   make_model, metadata_hashes, metrics, sha256, combined_prediction,
                   combined_target, combined_metrics)


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loader(dataset, config, train=False):
    generator = torch.Generator().manual_seed(config["seed"])
    sampler = (WeightedRandomSampler(dataset.weights, len(dataset), replacement=True,
                                      generator=generator) if train else None)
    return DataLoader(dataset, batch_size=config["batch_size"], sampler=sampler,
                      shuffle=False, num_workers=config["workers"], pin_memory=True,
                      persistent_workers=config["workers"] > 0, generator=generator)


def optimizer_for(model, config):
    return torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": config["encoder_lr"]},
        {"params": list(model.decoder.parameters()) + list(model.segmentation_head.parameters()),
         "lr": config["decoder_lr"]}], weight_decay=config["weight_decay"])


def amp():
    return torch.autocast("cuda", dtype=torch.bfloat16)


def run_epoch(model, batches, optimizer=None, predictions=None, score_maps=None,
              evaluate_combined=False, combined_predictions=None):
    training = optimizer is not None
    if training and (evaluate_combined or score_maps is not None):
        raise ValueError("Supplemental inference decisions are evaluation-only")
    model.train(training)
    confusion = torch.zeros((6, 6), dtype=torch.int64, device="cuda")
    merged_confusion = torch.zeros((5, 5), dtype=torch.int64, device="cuda") if evaluate_combined else None
    loss_total = torch.zeros((), device="cuda")
    pixel_total = torch.zeros((), device="cuda")
    started = time.perf_counter()
    context = contextlib.nullcontext() if training else torch.inference_mode()
    with context:
        for step, (image, target, eligible, indices) in enumerate(batches):
            image = image.cuda(non_blocking=True).to(memory_format=torch.channels_last)
            target = target.cuda(non_blocking=True).long()
            eligible = eligible.cuda(non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with amp():
                logits = model(image)
                loss_sum = F.cross_entropy(logits, target, ignore_index=255, reduction="sum")
                count = (target != 255).sum()
                loss = loss_sum / count.clamp_min(1)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            predicted = logits.detach().argmax(dim=1)
            confusion += confusion_counts(target, predicted, eligible)
            loss_total += loss_sum.detach()
            pixel_total += count
            if predictions is not None:
                predictions[indices.numpy()] = predicted.cpu().numpy().astype("uint8")
            if score_maps is not None or evaluate_combined:
                probabilities = logits.detach().float().softmax(dim=1)
            if evaluate_combined:
                merged = combined_prediction(probabilities)
                merged_confusion += confusion_counts(combined_target(target), merged, eligible, classes=5)
                if combined_predictions is not None:
                    combined_predictions[indices.numpy()] = merged.cpu().numpy().astype("uint8")
            if score_maps is not None:
                selected = torch.stack([probabilities[:, 1], probabilities[:, 2],
                                        probabilities[:, 1] + probabilities[:, 2]], dim=1)
                score_maps[indices.numpy()] = selected.cpu().numpy()
            if training and (step + 1) % 50 == 0:
                print(json.dumps({"step": step + 1, "steps": len(batches),
                                  "loss_so_far": float(loss_total / pixel_total)}), flush=True)
    torch.cuda.synchronize()
    result = metrics(confusion.cpu().numpy())
    if evaluate_combined:
        result["combined_class"] = combined_metrics(merged_confusion.cpu().numpy())
    result.update(loss=float(loss_total / pixel_total),
                  seconds=time.perf_counter() - started)
    if not math.isfinite(result["loss"]):
        raise ValueError("Nonfinite loss")
    return result


def smoke(model, dataset, config, output):
    """Overfit six training examples only; benchmark the configured batch size."""
    root = Path(config["dataset"])
    records = [json.loads(line) for line in (root / "index.jsonl").read_text().splitlines()
               if json.loads(line)["split"] == "train"]
    selected = list(dict.fromkeys(int(np.argmax([r["counts"][c] for r in records]))
                                 for c in range(6)))
    selected = (selected * config["batch_size"])[:config["batch_size"]]
    batch = torch.utils.data.default_collate([dataset[i] for i in selected])
    image, target, eligible, _ = batch
    image = image.cuda().to(memory_format=torch.channels_last)
    target, eligible = target.cuda().long(), eligible.cuda()
    tiny_config = dict(config, encoder_lr=.001, decoder_lr=.003)
    optimizer = optimizer_for(model, tiny_config)
    losses, durations = [], []
    model.train()
    for step in range(120):
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with amp():
            logits = model(image)
            loss = F.cross_entropy(logits, target, ignore_index=255)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        losses.append(float(loss.detach()))
        durations.append(time.perf_counter() - tick)
        if (step + 1) % 20 == 0:
            print(json.dumps({"smoke_step": step + 1, "loss": losses[-1]}), flush=True)
    result = metrics(confusion_counts(target, logits.argmax(1), eligible).cpu().numpy())
    result.update(initial_loss=float(np.mean(losses[:5])), final_loss=float(np.mean(losses[-5:])),
                  seconds_per_step=float(np.median(durations[20:])),
                  peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30,
                  source_training_indices=sorted(set(selected)),
                  batch_size=len(selected), steps=120,
                  checkpoint="discarded; full training restarts from pretrained initialization")
    result["passed"] = result["final_loss"] < .5 * result["initial_loss"] and result["accuracy"] > .9
    save_json(output / "smoke.json", result)
    if not result["passed"]:
        raise RuntimeError("Tiny training-set overfit check failed; inspect smoke.json")
    return result


def plot_results(history, best, dataset, predictions, output, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for split in ["train", "validation"]:
        axes[0].plot(epochs, [h[split]["loss"] for h in history], label=split)
        axes[1].plot(epochs, [h[split]["macro_iou"] for h in history], label=split)
    for ax, title in zip(axes, ["Cross-entropy", "Macro IoU on common valid pixels"]):
        ax.set(xlabel="Epoch", title=title)
        ax.axvline(best["epoch"], color="gray", linestyle=":")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=150)
    plt.close(fig)

    cm = np.asarray(best["validation"]["confusion_matrix"])
    normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 7))
    drawing = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    ax.set(xticks=range(6), yticks=range(6), xticklabels=CLASS_NAMES,
           yticklabels=CLASS_NAMES, xlabel="Predicted", ylabel="Reference",
           title="Validation confusion matrix (row fractions)")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    for row in range(6):
        for col in range(6):
            ax.text(col, row, f"{normalized[row, col]:.2f}", ha="center", va="center",
                    color="white" if normalized[row, col] > .5 else "black")
    fig.colorbar(drawing, ax=ax)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    targets = np.load(dataset.folder / "target.npy", mmap_mode="r")
    # Selection uses reference composition, never prediction quality: a rich
    # example for each minority class and one typical background patch.
    counts = np.array([[np.count_nonzero(t == c) for c in range(6)] for t in targets])
    selected = [int(np.argmax(counts[:, c])) for c in range(1, 6)]
    selected.append(int(np.argmax(counts[:, 0])))
    stats = json.loads((Path(config["dataset"]) / "normalization.json").read_text())
    mean = np.asarray(stats["mean"])[range(6), None, None]
    std = np.asarray(stats["std"])[range(6), None, None]
    palette = ListedColormap(["#b9b9aa", "#156b45", "#78b95b", "#2879b9", "#cf9a4b", "#8b668b"])
    palette.set_bad("white")
    fig, axes = plt.subplots(len(selected), 3, figsize=(10, len(selected) * 2.8))
    for row, index in enumerate(selected):
        image, target, eligible, _ = dataset[index]
        physical = image[:6].numpy() * std + mean
        rgb = np.clip(physical[[2, 1, 0]].transpose(1, 2, 0) / .3, 0, 1) ** .7
        axes[row, 0].imshow(rgb)
        axes[row, 1].imshow(np.ma.masked_where(~eligible.numpy().astype(bool), target), cmap=palette, vmin=-.5, vmax=5.5)
        axes[row, 2].imshow(np.ma.masked_where(~eligible.numpy().astype(bool), predictions[index]), cmap=palette, vmin=-.5, vmax=5.5)
        axes[row, 0].set_ylabel(dataset.ids[index], fontsize=8)
        for col in range(3):
            axes[row, col].set(xticks=[], yticks=[])
        if row == 0:
            for col, title in enumerate(["Sentinel-2 RGB", "Weak reference", "Prediction"]):
                axes[row, col].set_title(title)
    fig.tight_layout()
    fig.savefig(output / "validation_examples.png", dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train-only", action="store_true",
                        help="Save training/checkpoint-selection results; defer post-training inference and plots")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    for name in ["dataset", "output"]:
        config[name] = str(Path(config[name]).resolve())
    if config["loss"] != "cross_entropy" or config["ignore_index"] != 255:
        raise ValueError("Unsupported loss policy")
    if config["evaluate_test"]:
        raise ValueError("Test remains locked until the sensor/temporal/model choices are frozen")
    output = Path(config["output"] + ("_smoke" if args.smoke else ""))
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
    torch.hub.set_dir(str(Path(config["dataset"]).parent.parent / "training_cache" / "torch"))
    seed_everything(config["seed"])
    torch.set_num_threads(4)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This configuration requires CUDA with bfloat16 support")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    repo = Path(__file__).resolve().parents[4]
    snapshot = output / "source"
    snapshot.mkdir()
    for source in [Path(__file__), Path(__file__).with_name("core.py"),
                   Path(__file__).parents[1] / "dataset" / "reader.py",
                   repo / "ml" / "pyproject.toml", repo / "ml" / "uv.lock", config_path]:
        shutil.copy2(source, snapshot / source.name)
    manifest = dict(config=config, train_only=args.train_only, started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    dataset_hashes=metadata_hashes(config["dataset"]),
                    source_hashes={p.name: sha256(p) for p in snapshot.iterdir()},
                    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                    git_status=subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True),
                    versions={p: importlib.metadata.version(p) for p in
                              ["torch", "torchvision", "segmentation-models-pytorch", "numpy", "rasterio"]},
                    gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda,
                    input_channels=len(config["channels"]) * 2,
                    initialization="RGB kernels repeated across imagery and scaled 3/C; mask kernels zero; rest ImageNet",
                    reproducibility="Fixed seed and sampler; CUDA/cuDNN benchmark kernels are not bitwise deterministic",
                    evaluation="Common label-eligible pixels valid across all 14 v4 imagery channels; validation only")
    save_json(output / "manifest.json", manifest)
    cache = build_cache(config["dataset"], config["channels"])
    train_ds = CachedPatches(cache, "train", augment=not args.smoke)
    val_ds = CachedPatches(cache, "validation")
    model = make_model(len(config["channels"]), config["decoder_channels"]).cuda().to(memory_format=torch.channels_last)
    manifest["parameters"] = sum(p.numel() for p in model.parameters())
    weight_file = Path(torch.hub.get_dir()) / "checkpoints" / "resnet34-b627a593.pth"
    manifest["pretrained_sha256"] = sha256(weight_file)
    save_json(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), "parameters": manifest["parameters"],
                      "gpu": manifest["gpu"], "train": len(train_ds), "validation": len(val_ds)}), flush=True)
    if args.smoke:
        result = smoke(model, train_ds, config, output)
        print(json.dumps(result), flush=True)
        return
    train_loader, val_loader = loader(train_ds, config, True), loader(val_ds, config)
    optimizer = optimizer_for(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-6)
    history, best, best_score = [], None, -1.0
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        row = dict(epoch=epoch, lr=[g["lr"] for g in optimizer.param_groups])
        row["train"] = run_epoch(model, train_loader, optimizer)
        row["validation"] = run_epoch(model, val_loader)
        history.append(row)
        score = row["validation"]["macro_iou"]
        if score > best_score + 1e-5:
            best_score, best = score, row
            torch.save(dict(model=model.state_dict(), epoch=epoch, config=config,
                            validation=row["validation"], dataset_hashes=manifest["dataset_hashes"]),
                       output / "best.pt")
        scheduler.step()
        save_json(output / "history.json", history)
        save_json(output / "progress.json", dict(epoch=epoch, best_epoch=best["epoch"],
                  best_macro_iou=best_score, elapsed_seconds=time.perf_counter() - started))
        print(json.dumps({"epoch": epoch, "train_loss": row["train"]["loss"],
                          "train_iou": row["train"]["macro_iou"], "val_loss": row["validation"]["loss"],
                          "val_iou": score, "best_epoch": best["epoch"],
                          "seconds": row["train"]["seconds"] + row["validation"]["seconds"]}), flush=True)
        if epoch >= config["min_epochs"] and epoch - best["epoch"] >= config["patience"]:
            break
    if args.train_only:
        if metadata_hashes(config["dataset"]) != manifest["dataset_hashes"]:
            raise RuntimeError("Dataset metadata changed during training")
        checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
        if checkpoint["epoch"] != best["epoch"] or checkpoint["dataset_hashes"] != manifest["dataset_hashes"]:
            raise RuntimeError("Saved checkpoint metadata differs from selected epoch")
        result = dict(experiment=config["experiment"], seed=config["seed"],
            best_epoch=best["epoch"], epochs_completed=len(history),
            stopped="early_stopping" if len(history) < config["epochs"] else "epoch_budget",
            validation=best["validation"], validation_source="Recorded during training for checkpoint selection",
            training_seconds=time.perf_counter() - started,
            peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30,
            post_training_evaluation_deferred=True, checkpoint_predictions_reverified=False,
            test_evaluated=False, limitation="One-seed training result; post-training evaluation and reports deferred")
        save_json(output / "results.json", result)
        save_json(output / "complete.json", dict(status="training_complete_evaluation_deferred",
            finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            checkpoint_sha256=sha256(output / "best.pt"), results_sha256=sha256(output / "results.json"),
            dataset_hashes=manifest["dataset_hashes"]))
        print(json.dumps({k: v for k, v in result.items() if k != "validation"}), flush=True)
        return
    model.load_state_dict(torch.load(output / "best.pt", map_location="cuda", weights_only=False)["model"])
    predictions = np.lib.format.open_memmap(output / "validation_predictions.npy", mode="w+", dtype="uint8",
                                            shape=(len(val_ds), 256, 256))
    reloaded = run_epoch(model, val_loader, predictions=predictions)
    predictions.flush()
    if reloaded["confusion_matrix"] != best["validation"]["confusion_matrix"]:
        raise RuntimeError("Reloaded checkpoint predictions differ from selected checkpoint")
    if metadata_hashes(config["dataset"]) != manifest["dataset_hashes"]:
        raise RuntimeError("Dataset metadata changed during training")
    background = np.zeros((6, 6), dtype="int64")
    background[:, 0] = np.asarray(reloaded["confusion_matrix"]).sum(axis=1)
    result = dict(experiment=config["experiment"], seed=config["seed"],
                  best_epoch=best["epoch"], epochs_completed=len(history),
                  stopped="early_stopping" if len(history) < config["epochs"] else "epoch_budget",
                  validation=reloaded, majority_other_baseline=metrics(background),
                  training_seconds=time.perf_counter() - started,
                  peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30,
                  test_evaluated=False, limitation="One-seed weak-reference validation result; not independent ground truth")
    save_json(output / "results.json", result)
    plot_results(history, best, val_ds, predictions, output, config)
    save_json(output / "complete.json", dict(finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
              checkpoint_sha256=sha256(output / "best.pt"), results_sha256=sha256(output / "results.json"),
              dataset_hashes=manifest["dataset_hashes"]))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
