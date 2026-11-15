# Dataset preparation

**Current dataset:** `data/dataset/s1s2nisar_256_v4` (completed and audited 2026-11-14).

- 14 saved image channels; partial-data loader appends 14 validity indicators by
  default, giving 28 model inputs. Labels are a separate single-channel raster.
- Fresh geographic splits: 2,609 train, 585 validation,
  559 test patches; 256 x 256 pixels at 20 m EPSG:3035.
- Corrected dB radar inputs, conservative hole filling, one-pixel outer Other
  ignore fringe, minimum 80% trainable labels and 95% valid imagery per channel.
- Normalization fitted only on valid training imagery, separately per channel.
  Training sampling retains the established 75% minority-rich / 25% background
  draw policy; validation and test are evaluated uniformly.
- V3 remains preserved. Raw v3/v4 model scores do not isolate preprocessing
  effects because geographic splits and labels changed.

See [dataset creation](../docs/dataset_creation.md) for reproduction and validation,
and [channels and labels](../docs/dataset_creation.md#channel-and-label-reference)
for the exact channel order, units, validity masks and label codes. The dataset's
`plan.json`, `normalization.json`, `training_sampling.json`, `audit.json`,
`label_coverage_comparison.json`, and generated `README.md` contain final details.
`data/dataset/CURRENT.json` selects v4.

## Previous v3 dataset: retained reference

The following describes the preserved v3 release. Its commands select old linear
power inputs and must not be used to recreate v4.

**Previous dataset:** `data/dataset/s1s2nisar_256_v3` (from repository root).

**Previous v3 training sampler:** `training_sampling.json`, `plan.json` and
`index.jsonl` contain the same updated weights. It draws 75% probability from 1,249 patches with at
least one minority class occupying 5% of the whole patch, and 25% from the remaining
1,174 background-heavy patches. The positive pool is an equal mixture of five
minority classes, weighting qualifying patches by their within-patch class fraction.
The background pool is uniform. Probabilities are over draws, not exact per-batch
quotas. Validation/test remain uniform. Expected Other pixel exposure is 60.68%,
compared with 83.73% under uniform patch sampling; mixed patches prevent exact pixel
balance. Existing TIFFs, split assignments and training normalization are unchanged.

Apply/rebuild metadata in place after export; old metadata is retained in
`metadata_history/before_sampling_sync`, without duplicating TIFFs:
`PYTHONPATH=src .venv/bin/python -m bogsignal.dataset.sampling ../data/dataset/s1s2nisar_256_v3`.
`PatchDataset.sampling_weights` automatically loads this policy and checks its
patch IDs and index hash. Pass these weights to WeightedRandomSampler with replacement.
No sampler is automatically activated by iterating PatchDataset directly.
This supersedes the initial v1 export and v2 draft plan; those remain intact.
V3 assigns geographic groups using all 14 channels' valid candidates, with class
presence defined by at least 5% of patch pixels. A seeded search and group-move
refinement optimize 70/15/15 per-class presence and pixel allocation after buffers.
Do not confuse the initial v1 split statistics below with the current v3 plan.

Current commands from `ml/`, using a new output version when replanning:

```bash
PYTHONPATH=src .venv/bin/python -m bogsignal.dataset.pipeline plan --include-nisar \
  --data-root ../data --output ../data/dataset/s1s2nisar_256_v3
PYTHONPATH=src .venv/bin/python -m bogsignal.dataset.pipeline build \
  --data-root ../data --output ../data/dataset/s1s2nisar_256_v3
PYTHONPATH=src .venv/bin/python -m bogsignal.dataset.audit ../data/dataset/s1s2nisar_256_v3
```

V3 has 2,423 train, 524 validation and 541 test patches. Approximate substantial
class allocation train/validation/test: Swamp 72.9/13.5/13.5%; Marsh
72.3/13.5/14.2%; Water 69.1/14.8/16.1%; peat cropland 69.3/15.7/15.0%; peat
non-cropland 72.1/13.9/14.0%. These are shares of patches representing each class,
not class fractions within each fold. Mixed-patch sampling cannot guarantee equal
per-pixel classes. The detailed pixel shares are in the dataset report.

## Initial workflow (retained for reference)

Permanent code lives in `src/bogsignal/dataset/`. No model or Torch dependency
is required to prepare data. Run commands from `ml/` with the existing environment
(`uv run --locked python` or `.venv/bin/python`, with the package installed).

```bash
PYTHONPATH=src .venv/bin/python -m bogsignal.dataset.pipeline plan \
  --data-root ../data --output ../data/dataset/s1s2_256_v1
PYTHONPATH=src .venv/bin/python -m bogsignal.dataset.pipeline add-nisar \
  --data-root ../data --output ../data/dataset/s1s2nisar_256_v1 \
  --base-plan ../data/dataset/s1s2_256_v1/plan.json
PYTHONPATH=src .venv/bin/python -m bogsignal.dataset.pipeline build \
  --data-root ../data --output ../data/dataset/s1s2nisar_256_v1
```

`bogsignal-dataset` provides the same CLI after refreshing the editable package
installation. Geographic selection is frozen; use a new output directory to change
selection. Sampling weights can be updated in place with metadata backups.
Build can resume an interrupted export: existing patches must exactly
match the planned source data. Completed datasets cannot be overwritten. Input
hashes must match before and after export. The source data are never modified.

## Current agreed settings

- 256 × 256 pixels, 20 m grid, 5.12 km square, non-overlapping candidates.
- All selected input pixels must be valid, finite, and pass sensor validity masks.
- At least 95% trainable label coverage; other label pixels are exported as 255.
- Six target classes 0–5; Other is valid 0, not NoData. Loss ignore index is 255.
- Spatial blocks of 2048 pixels (40.96 km), grouped when touched by the same
  collected OSM wetland polygon. Entire groups have one split.
- Target 70/15/15 allocation considers patch and class counts. The approved
  Sentinel grouping actually retains 3642/567/616 patches after buffering.
- A patch expanded by 64 pixels (1.28 km) may not intersect another active split's
  block. Whole-patch grid rejection makes the actual exclusion larger in some
  places. This is not a claim of an exact-width buffer.
- NISAR filtering keeps that approved geography, selecting 2488/409/560 patches
  with all 14 channels valid. Its effective huge-blob masks are respected.

## Exported files

Each split has `images/<patch_id>.tif` and `labels/<patch_id>.tif`. Image arrays
are float32, C×256×256. S2 codes are scaled by 0.0001 to reflectance; S1 is linear
sigma-0 and NISAR is linear gamma-0. No dB conversion, clipping or per-patch
normalization is applied. Each band has a name and units. Labels are UInt8.
Both files retain EPSG:3035 and the actual patch affine transform.

Channel order: S2 B2, B3, B4, B8, B11, B12; S1 ascending VV, VH;
then descending VV, VH; NISAR ascending HH, HV;
then descending HH, HV. `plan.json` is the authoritative ordered channel list.

`index.jsonl` records paths, split, source-grid offsets, class counts and training
sampling weights. `splits.geojson` records spatial blocks and group IDs.
`normalization.json` contains per-channel means/stds computed from training images
only, in physical units. `complete.json` marks a fully readback-verified export.
`plan.json` records input hashes, selection rules, exclusions and class counts.

## Reading and training

```python
from bogsignal.dataset.reader import PatchDataset
dataset = PatchDataset('../data/dataset/s1s2nisar_256_v1', split='train')
sample = dataset[0]  # image float32 C,H,W; mask int64 H,W; valid bool H,W
```

Normalization uses training-only channel statistics and can be disabled with
`normalize=False`. PyTorch DataLoader's default collation converts these NumPy
arrays to tensors. For training, use WeightedRandomSampler with
`dataset.sampling_weights`, replacement=True and num_samples=len(dataset).
Uniform iteration does **not** apply balancing automatically. Do not use those
weights for natural validation/test evaluation. There are no duplicated physical
files to implement oversampling.

Training weights give each available class equal contribution to the sampling
mixture, using within-patch class fractions. They improve minority exposure but do
not promise equal per-pixel classes in mixed patches. Actual and expected weighted
class fractions are in the plan. Fixed class-stratified evaluation views are a
separate selection from the geographic holdouts, not a cross-split reshuffle.

The existing sources and automatically generated labels are provisional. Spatial
holdouts measure agreement with those labels, not independent real-world accuracy.
NISAR remains subject to residual artifacts after the operational quality mask.

Tests: `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v`.
