"""Audit exported spatial splits and write colleague-facing dataset reports."""
import argparse
import csv
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import box
from .reader import PatchDataset

NAMES = ['Other', 'Swamp', 'Marsh', 'Permanent water', 'Peatland cropland', 'Peatland non-cropland']


def sampling_section(root):
    root = Path(root)
    path = root / 'training_sampling.json'
    if not path.exists():
        return ''
    policy = json.loads(path.read_text())
    text = '\n## Training sampling distribution\n\n'
    text += f"75% of training draws come from {policy['minority_patches']:,} minority-rich patches; 25% come from {policy['background_patches']:,} background-heavy patches. A minority class must occupy at least 5% of a patch to qualify.\n\n"
    text += 'These are sampling probabilities with replacement, not physical removal of files or exact quotas per batch. Validation/test, patch membership, TIFF values and normalization remain unchanged. The stored pixel-share table above therefore remains correct.\n\n'
    text += '| Class | Stored training pixels | Expected sampled training share |\n| --- | ---: | ---: |\n'
    with (root / 'sampling_distribution.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class_id', 'class_name', 'stored_training_pixel_share', 'expected_mean_sampled_patch_trainable_share'])
        for i, name in enumerate(NAMES):
            before = policy['baseline_class_fractions'][i]
            after = policy['expected_class_fractions'][i]
            text += f'| {name} | {100*before:.2f}% | {100*after:.2f}% |\n'
            writer.writerow([i, name, before, after])
    text += '\nExpected shares average the trainable-pixel class fractions of sampled patches; actual batches fluctuate. Ignore pixels (255) are excluded. Mixed patches prevent equal per-pixel class balance. Use PatchDataset.sampling_weights with WeightedRandomSampler and replacement=True; uniform iteration does not apply the policy.\n'
    return text


def audit(root):
    root = Path(root)
    complete = json.loads((root / 'complete.json').read_text())
    plan = json.loads((root / 'plan.json').read_text())
    records = [json.loads(line) for line in (root / 'index.jsonl').read_text().splitlines()]
    assert len(records) == complete['patches']
    assert len({r['id'] for r in records}) == len(records)
    shapes = []
    hist = {s: np.zeros(6, dtype='int64') for s in ['train', 'validation', 'test']}
    present = {s: np.zeros(6, dtype='int64') for s in hist}
    for r in records:
        with rasterio.open(root / r['images']) as image, rasterio.open(root / r['labels']) as labels:
            assert image.count == len(plan['channels']) and image.shape == (256, 256)
            assert labels.count == 1 and labels.shape == image.shape
            assert image.crs == labels.crs and image.transform == labels.transform
            assert image.descriptions == tuple(c['name'] for c in plan['channels'])
            y = labels.read(1)
            assert np.isin(y, [0, 1, 2, 3, 4, 5, 255]).all()
            counts = np.bincount(y[y != 255], minlength=6)
            assert counts.tolist() == r['counts']
            hist[r['split']] += counts
            present[r['split']] += counts >= 3277
            shapes.append(box(*image.bounds))
    patches = gpd.GeoDataFrame(records, geometry=shapes, crs=plan['grid']['crs'])
    index = patches.sindex
    for i, geom in enumerate(patches.geometry):
        # Minimum 2.56 km separation between any differently assigned patches.
        hits = index.query(geom.buffer(2559.99), predicate='intersects')
        assert all(patches.iloc[int(j)]['split'] == records[i]['split'] for j in hits)
    blocks = gpd.read_file(root / 'splits.geojson')
    assert (blocks.groupby('group')['split'].nunique() == 1).all()
    for s in hist:
        assert hist[s].tolist() == plan['summary'][s]['class_pixels']
        sample = PatchDataset(root, s)[0]
        assert np.isfinite(sample['image']).all()
    stats = json.loads((root / 'normalization.json').read_text())
    assert stats['pixels'] == sum(r['split'] == 'train' for r in records) * 256 * 256
    with (root / 'class_distribution.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['split', 'class_id', 'class_name', 'pixels', 'pixel_share', 'substantial_patches', 'class_presence_split_share'])
        totals = sum(present.values())
        for s in hist:
            for c, name in enumerate(NAMES):
                writer.writerow([s, c, name, int(hist[s][c]), float(hist[s][c] / hist[s].sum()), int(present[s][c]), float(present[s][c] / totals[c])])
    report = dict(patches=len(records),channels=len(plan['channels']),
                  split_counts={s: sum(r['split']==s for r in records) for s in hist},
                  checks='all patch label histograms, grids and channel order; no duplicate IDs; cross-split 2.56 km separation; group isolation; reader smoke checks; training-only normalization pixel count')
    (root / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    rows = ['| Class | Train | Validation | Test |', '| --- | ---: | ---: | ---: |']
    for c, name in enumerate(NAMES):
        rows.append('| '+name+' | '+' | '.join(f'{100*hist[s][c]/hist[s].sum():.2f}%' for s in hist)+' |')
    readme = '''# Polissia 14-channel segmentation patches

Current class-stratified dataset. Inputs: six Sentinel-2 reflectance bands,
Sentinel-1 ascending/descending VV/VH, NISAR ascending/descending HH/HV.
All input pixels pass source and effective sensor-quality validity masks.
Patches are 256 × 256 at 20 m in EPSG:3035 (5.12 km square).

`train/`, `validation/`, `test/` each contain matching `images/` and `labels/`
GeoTIFFs. Images are float32 in physical units: S2 reflectance (scale applied),
S1 linear sigma-0, NISAR linear gamma-0. No dB conversion or clipping.
Labels are UInt8: 0 Other, 1 Swamp, 2 Marsh, 3 Permanent water,
4 Peatland cropland, 5 Peatland non-cropland, 255 ignore. At least 95% of a
patch's labels are trainable. Retain valid class 0 in the segmentation loss.

Spatial allocation targets 70/15/15 per class. A class represents a patch when
it occupies at least 5% of its pixels. Geographic blocks are grouped by mapped
OSM wetlands and buffered between different folds. Optimization uses class presence
and pixel counts after input filtering and buffering; exact shares are approximate.
Classes are not equally common within a split. No patch is duplicated across folds.

`index.jsonl` contains patch paths, offsets, class counts and training sampling
weights. training_sampling.json records the 75% minority-rich / 25% background-heavy
policy, synchronized with plan/index weights. The reader uses these training weights
only. Use weights with replacement; validation/test remain
unweighted. `normalization.json` contains training-only means/stds in physical units.
`plan.json` freezes input hashes, selected channel order, split settings and counts.
`splits.geojson` records the geographic assignment. `complete.json` marks export
completion; `audit.json` records independent checks. `class_distribution.csv`
reports pixel shares and substantial-class patch representation.

For PyTorch-compatible NumPy samples use `bogsignal.dataset.reader.PatchDataset`.
Default normalization uses the training statistics, including on validation/test.
Loss ignore index is 255; six output channels correspond to classes 0–5.
See BogSignal/ml/DATASET.md for commands and sampling details.

These are automatically generated reference labels with mixed source dates,
not independently verified 2026 ground truth. NISAR quality masks are operational
exclusions, not proof of artifact-free measurements. GPM peat evidence remains
coarse. Default Other and other source-map omissions remain uncertain.

## Trainable-pixel distribution

'''
    (root / 'README.md').write_text(readme + '\n'.join(rows) + '\n' + sampling_section(root))
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    audit(parser.parse_args().dataset)


if __name__ == '__main__':
    main()
