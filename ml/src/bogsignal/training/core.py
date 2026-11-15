"""Dataset channel isolation, pretrained initialization and pooled metrics."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

CLASS_NAMES = ["Other", "Swamp", "Marsh", "Permanent water",
               "Peatland cropland", "Peatland non-cropland"]
COMBINED_CLASS_NAMES = ["Other", "Wetland (Swamp + Marsh)", "Permanent water",
                        "Peatland cropland", "Peatland non-cropland"]
METADATA = ["plan.json", "index.jsonl", "normalization.json", "training_sampling.json",
            "complete.json", "audit.json"]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metadata_hashes(root):
    return {name: sha256(Path(root) / name) for name in METADATA}


def select_channels(plan, names):
    available = [c["name"] for c in plan["channels"]]
    if not names or len(set(names)) != len(names):
        raise ValueError("Select unique predictor channels")
    return [available.index(name) for name in names]


def selected_sample(sample, indices):
    """Radar validity can affect scoring, but never S2-only predictors."""
    return (np.asarray(sample["image"])[indices],
            np.asarray(sample["input_valid"])[indices].astype("uint8"),
            np.asarray(sample["mask"]).astype("uint8"),
            np.asarray(sample["evaluation_valid"]).astype("uint8"))


def build_cache(root, names):
    """Cache selected normalized channels and full-stack scoring validity."""
    from bogsignal.dataset.reader import PatchDataset
    root = Path(root)
    hashes = metadata_hashes(root)
    plan = json.loads((root / "plan.json").read_text())
    indices = select_channels(plan, names)
    key = hashlib.sha256(json.dumps([hashes, names], sort_keys=True).encode()).hexdigest()[:16]
    cache = root.parent.parent / "training_cache" / key
    marker = cache / "complete.json"
    if marker.exists():
        manifest = json.loads(marker.read_text())
        if manifest["hashes"] != hashes or manifest["channels"] != names:
            raise ValueError("Cache provenance mismatch")
        return cache
    cache.mkdir(parents=True, exist_ok=True)
    splits = {}
    for split in ["train", "validation", "test"]:
        ds = PatchDataset(root, split, append_validity=False)
        n, c = len(ds), len(names)
        shape = (n, c, 256, 256)
        folder = cache / split
        folder.mkdir(exist_ok=True)
        arrays = [np.lib.format.open_memmap(folder / name, mode="w+", dtype=dtype, shape=s)
                  for name, dtype, s in [
                      ("image.npy", "float32", shape), ("validity.npy", "uint8", shape),
                      ("target.npy", "uint8", (n, 256, 256)),
                      ("evaluation.npy", "uint8", (n, 256, 256))]]
        for i in range(n):
            for destination, value in zip(arrays, selected_sample(ds[i], indices)):
                destination[i] = value
            if i % 200 == 0:
                print(json.dumps({"cache_split": split, "patch": i, "total": n}), flush=True)
        for array in arrays:
            array.flush()
        np.save(folder / "weights.npy", ds.sampling_weights)
        (folder / "ids.json").write_text(json.dumps([r["id"] for r in ds.records]))
        splits[split] = n
        del arrays
    if metadata_hashes(root) != hashes:
        raise ValueError("Dataset metadata changed while caching")
    marker.write_text(json.dumps(dict(hashes=hashes, channels=names, indices=indices,
                                     splits=splits), indent=2))
    return cache


class CachedPatches(Dataset):
    def __init__(self, cache, split, augment=False):
        self.folder = Path(cache) / split
        self.augment = augment
        self.ids = json.loads((self.folder / "ids.json").read_text())
        self.weights = np.load(self.folder / "weights.npy")
        self.arrays = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        if self.arrays is None:
            self.arrays = [np.load(self.folder / name, mmap_mode="r") for name in
                           ["image.npy", "validity.npy", "target.npy", "evaluation.npy"]]
        image, validity, target, evaluation = [a[i] for a in self.arrays]
        image = np.concatenate([image, validity.astype("float32")])
        if self.augment:
            k = int(torch.randint(4, ()).item())
            image, target, evaluation = [np.rot90(a, k, axes=(-2, -1))
                                          for a in [image, target, evaluation]]
            for axis in [-2, -1]:
                if torch.rand(()) < .5:
                    image, target, evaluation = [np.flip(a, axis=axis)
                                                  for a in [image, target, evaluation]]
        return (torch.from_numpy(image.copy()), torch.from_numpy(target.copy()),
                torch.from_numpy(evaluation.copy()), i)


def make_model(channels, decoder_channels=(256, 128, 64, 32, 16), pretrained=True):
    import segmentation_models_pytorch as smp
    from torchvision.models import ResNet34_Weights
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=channels * 2, classes=6, activation=None,
                     decoder_channels=tuple(decoder_channels))
    if pretrained:
        weights = ResNet34_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)
        rgb = weights["conv1.weight"]
        expanded = rgb.new_zeros((rgb.shape[0], channels * 2, *rgb.shape[2:]))
        expanded[:, :channels] = rgb[:, torch.arange(channels) % 3] * (3 / channels)
        weights["conv1.weight"] = expanded
        model.encoder.load_state_dict(weights, strict=True)
    return model


def confusion_counts(target, prediction, eligible, classes=6):
    keep = eligible.bool() & (target != 255)
    return torch.bincount((target[keep].long() * classes + prediction[keep].long()),
                          minlength=classes * classes).reshape(classes, classes)


def combined_prediction(probabilities):
    """Merge probabilities before argmax, with no thresholds or pixel ranking."""
    return torch.stack([probabilities[:, 0], probabilities[:, 1] + probabilities[:, 2],
                        probabilities[:, 3], probabilities[:, 4], probabilities[:, 5]], dim=1).argmax(dim=1)


def combined_target(target):
    result = target.clone()
    keep = target != 255
    mapping = target.new_tensor([0, 1, 1, 2, 3, 4])
    result[keep] = mapping[target[keep].long()]
    return result


def combined_metrics(confusion):
    cm = np.asarray(confusion, dtype="float64")
    if cm.shape != (5, 5) or not np.isfinite(cm).all() or (cm < 0).any():
        raise ValueError("Expected a nonnegative 5-class confusion matrix")
    tp, support, predicted = np.diag(cm), cm.sum(1), cm.sum(0)
    def ratio(a, b):
        return float(a / b) if b else None
    rows = {name: dict(iou=ratio(tp[i], support[i] + predicted[i] - tp[i]),
                       dice=ratio(2 * tp[i], support[i] + predicted[i]),
                       precision=ratio(tp[i], predicted[i]), recall=ratio(tp[i], support[i]),
                       f2=ratio(5 * tp[i], 4 * support[i] + predicted[i]),
                       support=int(support[i]), predicted_pixels=int(predicted[i]))
            for i, name in enumerate(COMBINED_CLASS_NAMES)}
    def average(key):
        values = [r[key] for r in rows.values() if r[key] is not None]
        return float(np.mean(values)) if values else None
    return dict(class_names=COMBINED_CLASS_NAMES, classes=rows,
                confusion_matrix=cm.astype("int64").tolist(), scored_pixels=int(cm.sum()),
                macro_iou=average("iou"), macro_dice=average("dice"), macro_recall=average("recall"),
                accuracy=ratio(tp.sum(), cm.sum()),
                wetland_area_fraction=ratio(predicted[1], cm.sum()))


def verify_reproduction(expected, actual, tolerance=.001):
    """Bound numerical drift to 0.1 percentage points on every defined metric."""
    expected, actual = np.asarray(expected, dtype="int64"), np.asarray(actual, dtype="int64")
    if expected.shape != (6, 6) or actual.shape != (6, 6) or not np.array_equal(expected.sum(1), actual.sum(1)):
        raise ValueError("Checkpoint validation changed reference support/scoring area")
    before, after = metrics(expected), metrics(actual)
    differences = [abs(before["macro_iou"] - after["macro_iou"])]
    for name in CLASS_NAMES:
        for metric in ["iou", "dice", "precision", "recall", "f2"]:
            a, b = before["classes"][name][metric], after["classes"][name][metric]
            if a is not None and b is not None:
                differences.append(abs(a - b))
            elif a != b:
                raise ValueError("Checkpoint validation changed a metric's definedness")
    delta = max(differences)
    if delta > tolerance:
        raise ValueError(f"Checkpoint metric difference {delta} exceeds numerical tolerance {tolerance}")
    return dict(exact_confusion_match=bool(np.array_equal(expected, actual)),
                max_absolute_metric_difference=float(delta), tolerance=tolerance,
                confusion_count_l1_difference=int(np.abs(expected - actual).sum()),
                expected_confusion=expected.tolist(), actual_confusion=actual.tolist())


def metrics(confusion):
    cm = np.asarray(confusion, dtype="float64")
    tp = np.diag(cm)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    def divide(a, b):
        return np.divide(a, b, out=np.full_like(a, np.nan), where=b > 0)
    iou = divide(tp, support + predicted - tp)
    dice = divide(2 * tp, support + predicted)
    precision, recall = divide(tp, predicted), divide(tp, support)
    f2 = divide(5 * tp, 4 * support + predicted)
    def finite(value):
        return float(value) if np.isfinite(value) else None
    def average(values):
        valid = np.asarray(values)[np.isfinite(values)]
        return float(valid.mean()) if valid.size else None
    # Wetland presence accepts either swamp or marsh. Water and peat classes
    # remain separate; "Other" is reference non-target, not verified dry land.
    wetland_tp = cm[1:3, 1:3].sum()
    wetland_support = support[1:3].sum()
    wetland_predicted = predicted[1:3].sum()
    def ratio(a, b):
        return float(a / b) if b else None
    wetland = dict(
        positive_classes=["Swamp", "Marsh"], decision="six-class argmax in {Swamp, Marsh}",
        tp=int(wetland_tp), fn=int(wetland_support - wetland_tp),
        fp=int(wetland_predicted - wetland_tp),
        tn=int(cm.sum() - wetland_support - wetland_predicted + wetland_tp),
        recall=ratio(wetland_tp, wetland_support),
        precision=ratio(wetland_tp, wetland_predicted),
        f2=ratio(5 * wetland_tp, 4 * wetland_support + wetland_predicted),
        dice=ratio(2 * wetland_tp, wetland_support + wetland_predicted),
        iou=ratio(wetland_tp, wetland_support + wetland_predicted - wetland_tp),
        reference_positive_pixels=int(wetland_support), predicted_pixels=int(wetland_predicted),
        flagged_area_fraction=ratio(wetland_predicted, cm.sum()))
    return dict(macro_iou=average(iou), macro_dice=average(dice),
                macro_recall=average(recall), macro_f2=average(f2),
                swamp_marsh_macro_recall=average(recall[1:3]),
                swamp_marsh_macro_f2=average(f2[1:3]),
                swamp_marsh_mean_iou=average(iou[1:3]), wetland_detection=wetland,
                accuracy=finite(tp.sum() / max(cm.sum(), 1)), scored_pixels=int(cm.sum()),
                confusion_matrix=cm.astype("int64").tolist(),
                classes={name: dict(iou=finite(iou[i]), dice=finite(dice[i]),
                                    precision=finite(precision[i]), recall=finite(recall[i]), f2=finite(f2[i]),
                                    support=int(support[i])) for i, name in enumerate(CLASS_NAMES)})
