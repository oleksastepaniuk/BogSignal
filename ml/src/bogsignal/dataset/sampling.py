"""Training-only 75/25 minority-rich/background sampling policy."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def mixture_weights(counts, patch_pixels=256 * 256):
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 2 or counts.shape[1] != 6 or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError('Expected finite nonnegative six-class patch counts')
    if (counts.sum(axis=1) <= 0).any() or (counts.sum(axis=1) > patch_pixels).any():
        raise ValueError('Invalid trainable pixel count')
    substantial = counts[:, 1:] >= np.ceil(.05 * patch_pixels)
    positive = substantial.any(axis=1)
    if not positive.any() or positive.all():
        raise ValueError('75/25 sampling requires both minority-rich and background pools')
    fractions = counts[:, 1:] / counts.sum(axis=1, keepdims=True)
    fractions = np.where(substantial, fractions, 0)
    totals = fractions.sum(axis=0)
    if (totals == 0).any():
        raise ValueError('Every minority class needs a substantial training patch')
    # Each class supplies one fifth of the positive-pool probability mass.
    positive_weights = (fractions / totals).sum(axis=1) / 5
    result = .75 * positive_weights
    result[~positive] = .25 / (~positive).sum()
    return result


def write_policy(root):
    root = Path(root)
    index_path = root / 'index.jsonl'
    records = [json.loads(line) for line in index_path.read_text().splitlines()]
    train = [r for r in records if r['split'] == 'train']
    counts = np.array([r['counts'] for r in train], dtype=float)
    weights = mixture_weights(counts)
    plan_path = root / 'plan.json'
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else None
    if plan is not None:
        expected = {p['id']: (p['split'], p['counts']) for p in plan['patches']}
        actual = {p['id']: (p['split'], p['counts']) for p in records}
        if len(actual) != len(records) or expected != actual:
            raise ValueError('Plan and index patch membership/counts differ')
    backup = root / 'metadata_history' / 'before_sampling_sync'
    backup.mkdir(parents=True, exist_ok=True)
    for name in ['index.jsonl', 'plan.json', 'complete.json', 'training_sampling.json', 'README.md']:
        source = root / name
        if source.exists() and not (backup / name).exists():
            shutil.copy2(source, backup / name)
    weight_by_id = dict(zip([r['id'] for r in train], map(float, weights)))
    for r in train:
        r['sampling_weight'] = weight_by_id[r['id']]
    temporary = index_path.with_suffix('.partial.jsonl')
    temporary.write_text(''.join(json.dumps(r)+'\n' for r in records))
    temporary.replace(index_path)
    positive = (counts[:, 1:] >= 3277).any(axis=1)
    fractions = counts / counts.sum(axis=1, keepdims=True)
    report = dict(version=1, scope='training only', minority_pool_probability=.75,
                  background_pool_probability=.25, class_presence_threshold=.05,
                  patch_pixels=65536, minority_patches=int(positive.sum()),
                  background_patches=int((~positive).sum()),
                  within_minority_pool='Equal mixture of five classes; probability within each class proportional to its qualifying pixel fraction',
                  within_background_pool='Uniform', replacement=True,
                  patch_ids=[r['id'] for r in train], weights=weights.tolist(),
                  index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
                  expected_class_fractions=(fractions * weights[:, None]).sum(axis=0).tolist(),
                  baseline_class_fractions=(counts.sum(axis=0) / counts.sum()).tolist(),
                  validation_test='unchanged; uniform evaluation',
                  note='Probabilities apply over draws, not an exact per-batch quota. TIFFs, splits, labels and normalization are unchanged. Weights are synchronized with plan/index.')
    path = root / 'training_sampling.json'
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(path)
    if plan is not None:
        for p in plan['patches']:
            if p['split'] == 'train':
                p['sampling_weight'] = weight_by_id[p['id']]
        plan.setdefault('summary', {}).setdefault('train', {})['weighted_expected_class_fractions'] = report['expected_class_fractions']
        plan['training_sampling'] = {k: report[k] for k in ['version', 'scope', 'minority_pool_probability', 'background_pool_probability', 'class_presence_threshold', 'replacement']}
        temporary = plan_path.with_suffix('.partial.json')
        temporary.write_text(json.dumps(plan, indent=2)+'\n')
        temporary.replace(plan_path)
    complete_path = root / 'complete.json'
    if complete_path.exists():
        complete = json.loads(complete_path.read_text())
        if plan is not None:
            complete['plan_sha256'] = hashlib.sha256(plan_path.read_bytes()).hexdigest()
            complete['summary'] = plan['summary']
        complete['training_sampling_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        temporary = complete_path.with_suffix('.partial.json')
        temporary.write_text(json.dumps(complete, indent=2)+'\n')
        temporary.replace(complete_path)
    print(json.dumps({k: v for k, v in report.items() if k not in ['patch_ids', 'weights']}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    write_policy(parser.parse_args().dataset)


if __name__ == '__main__':
    main()
