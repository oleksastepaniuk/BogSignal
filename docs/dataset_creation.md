# Versioned dataset creation

Updated 2026-11-14. **V4 is now completed, audited, and selected by CURRENT.json.**
It has 3,753 patches: 2,609 train, 585 validation,
and 559 test. V3 is preserved. The sections below document the
separate input, label and final dataset stages; historical diagnostic runs are
explicitly identified. Independent feature/temporal experiments remain separate.

## Channel and label reference

### Image channels and model inputs

The saved image patch has shape **14 x 256 x 256**, Float32, on the 20 m grid.
Its ordered channels are:

| Position (1-based) | Channel | Saved physical units |
| --- | --- | --- |
| 1 | Sentinel-2 B2, blue | Surface reflectance, source codes multiplied by 0.0001 |
| 2 | Sentinel-2 B3, green | Surface reflectance |
| 3 | Sentinel-2 B4, red | Surface reflectance |
| 4 | Sentinel-2 B8, near infrared | Surface reflectance; this is B8, not B8A |
| 5 | Sentinel-2 B11, SWIR 1 | Surface reflectance |
| 6 | Sentinel-2 B12, SWIR 2 | Surface reflectance |
| 7 | Sentinel-1 ascending VV | Clipped sigma-0 dB |
| 8 | Sentinel-1 ascending VH | Clipped sigma-0 dB |
| 9 | Sentinel-1 descending VV | Clipped sigma-0 dB |
| 10 | Sentinel-1 descending VH | Clipped sigma-0 dB |
| 11 | NISAR ascending HH | Clipped gamma-0 dB |
| 12 | NISAR ascending HV | Clipped gamma-0 dB |
| 13 | NISAR descending HH | Clipped gamma-0 dB |
| 14 | NISAR descending HV | Clipped gamma-0 dB |

These units describe the revised inputs; existing v3 radar patches remain in
linear power. The plan's ordered channel list is authoritative. No polarization
ratios or multitemporal summary channels are included in this baseline revision.

Future exported image patches store missing predictors as NoData -9999. Their
separate `input_validity` TIFF has the same 14 x 256 x 256 shape: 1 valid, 0 missing.
The reader normalizes valid predictors using training-only statistics, replaces
missing normalized values with zero, then by default appends the 14 binary masks.
Thus **saved imagery has 14 channels; the default partial-data model input has
28 channels**. `append_validity=False` keeps a 14-channel image tensor and returns
the masks separately for architectures that accept a separate mask input.

### Segmentation labels

A label patch is **one 256 x 256 UInt8 raster**, not 14 label channels. It stores
one class ID per pixel. A six-class segmentation model normally predicts six
logits at each pixel; the target IDs correspond to:

| Training target ID | Meaning |
| --- | --- |
| 0 | Other; supervised reference class, not guaranteed confirmed dry land |
| 1 | Swamp |
| 2 | Marsh |
| 3 | Permanent water |
| 4 | Peatland cropland |
| 5 | Peatland non-cropland |
| 255 | Ignore; contributes to neither supervised loss nor scored metrics |

Full source-label rasters additionally preserve exclusion reasons as raw codes:

| Raw source code | Meaning | Exported training target |
| --- | --- | --- |
| 6 | OSM wetland without an assignable GWL subtype | 255 |
| 7 | Flooded flat excluded from the six target classes | 255 |
| 8 | Ambiguous/conflicting OSM fill | 255 |
| 9 | Peat evidence with missing/unhandled land-cover subtype | 255 |
| 10 | New one-pixel uncertain Other fringe | 255 |
| 255 | Outside AOI / source NoData | 255 |

Training eligibility controls the export: wherever `training_mask != 1`, the
label becomes 255. Codes 6-10 are diagnostic reasons, not extra model classes.
Hole filling changes only eligible Other pixels to a supported class 1-5 and
records each inference; the later ignore fringe changes selected remaining Other
pixels to raw code 10. Positive labels are not eroded.

Input validity and label eligibility are independent. A missing S1 pixel does
not automatically erase an otherwise usable label. Conversely, a fully observed
pixel with an uncertain label remains ignored. Loss must exclude target 255;
`evaluation_valid` additionally restricts scoring to jointly valid imagery for
matched sensor comparisons. The agreed patch thresholds are at least 80%
trainable labels and at least 95% valid imagery in every required channel.
## Stage 1: corrected input revision

Run from `BogSignal/ml/` with the existing environment:

```bash
PYTHONPATH=src .venv/bin/python -m bogsignal.dataset.preprocessing \
  --data-root ../data \
  --output ../data/inputs/preprocessing_v4 \
  --nisar-archive-root /home/krk/projects/Argos
```

The output directory must be new and under `data/inputs`. Existing directories
are refused, including failed/incomplete builds; retain them for diagnosis and
choose a different revision name when retrying. This command is entirely local;
it does not authenticate, download scenes, modify source rasters, change labels,
select patches, split geography, compute normalization, or update CURRENT.json.
Native NISAR HDF5 archives are required to verify identification headers. Finished
input revisions remain usable with their recorded provenance files after the
archives are moved, provided the consumed inputs still match their hashes.

The 14 channels remain six S2 bands B2/B3/B4/B8/B11/B12, ascending/descending S1
VV/VH, and ascending/descending NISAR HH/HV, on the original 20 m EPSG:3035 grid.
No ratios, temporal summaries, or additional predictor channels are added here.

### Power, validity and clipping

The retained Sentinel-1 inputs originate from `COPERNICUS/S1_GRD_FLOAT` and store
linear sigma-0 power. NISAR HH/HV store linear gamma-0 power. Both radar sensors
therefore use `10 * log10(power)` in the new derived files. Do not apply this
conversion to dB files or confuse GCOV power with complex amplitudes.

Before conversion, intersect the existing effective pass mask, both polarization
source masks, finite strictly positive powers, and valid interpretation companions.
For S1, the ellipsoidal angle must remain between 0 and 90 degrees. For NISAR,
the GCOV subswath code must be 1–254 and the existing reviewed huge-blob exclusions
remain in effect. These masks do not certify absence of residual RFI or other
radar artifacts.

Zero, negative, masked and nonfinite radar samples are excluded. A positive
numerical floor `epsilon=1e-12` applies only to retained positive values. It never
turns an invalid zero into a usable observation. S2 zero reflectance remains valid.
The mask is common to both polarizations within a pass.

Fixed conservative clipping limits are:

| Polarization | dB limits |
| --- | --- |
| S1 VV / NISAR HH | -40 to +5 |
| S1 VH / NISAR HV | -45 to 0 |

These are explicit engineering limits, not fitted full-dataset percentiles or
validated optimal thresholds. They preserve a wider dark-water range than the
initial -25/-30 suggestions. `--co-clip LOW HIGH`, `--cross-clip LOW HIGH`, and
`--epsilon` define a separately named revision if changed. Full-resolution counts
of clipping, positive values below epsilon, and nonpositive exclusions are saved.
Clipping limits must not be tuned using held-out model results.

Radar outputs are Float32 dB TIFFs, NoData -9999, with channel descriptions and
units. Corrected pass masks are UInt8: 1 retained, 0 excluded inside AOI,
255 outside AOI. S2 reflectance codes, scale 0.0001, remain referenced unchanged.
No per-scene or per-patch standardization is performed.

### Acquisition geometry and processing integrity

Ascending and descending stay separate. Each retained pixel's source index is
checked against the correct pass, observation date, signed time offset and track.
All NISAR catalogue records are checked against native HDF5 granule ID, pass,
track, frame and acquisition start. Composite release ID, product version and
specification version must agree across both passes. A mismatch fails the build;
it is not hidden with a replacement date, a guessed orbit, or a masked discrepancy.
A failed consistency check requires a separately reviewed source mosaic rebuild.

Multiple relative tracks across Polissia are retained explicitly in per-pixel
orbit maps; this spatial mosaic is not a same-track temporal stack. No ascending/
descending averaging or cross-track temporal aggregation is introduced.

The retained S1 inventory identifies its collection, mode, platform and orbit,
but does not expose the original processor baseline. That baseline is recorded
as unverified; the package does not claim S1 processing-version homogenization.

### Per-pixel provenance and verification

`manifest.json` records ordered channel paths/units, masks, policy, grid, original
input SHA-256 hashes, generated file hashes, and selected-source counts.
`<sensor_pass>_source_lookup.json` records exact acquisition UTC timestamps and
scene IDs for retained sources, plus verified NISAR native identification fields.
The original per-pixel `source_index`, `observation_date`, `relative_orbit`, and
`time_offset_days` rasters are referenced and hashed instead of duplicated.
**Intersect all companion rasters with the new revision mask.** A source index
then recovers the exact UTC time; the date raster alone has only day precision.

For any pair of sensors, use their selected source indices to retrieve exact UTC
acquisitions and subtract the timestamps. The nominal target date 2026-10-25 is
not a claim that all sensors observed each pixel on that date.

Every exported radar pixel is read back and compared against the conversion of
its original source pixel. Every retained pixel's provenance is checked, all
consumed source hashes are checked again after processing, and exact grid
alignment is required. `complete.json` is written only after those checks and
identifies this as an input package, not a complete training dataset. During a
build, `progress.json` is an intermediate checkpoint, not an approval marker.

## Final dataset planning and export

1. Incorporate the remaining imagery and label revisions; freeze their manifests.
2. Reconstruct eligible patches and geographic train/validation/test groups from
   scratch. Keep repeated dates of the same location and connected wetland groups
   together, with cross-fold spatial buffers.
3. Fit per-channel mean/std on valid **new training pixels only**, after dB
   conversion and clipping. Apply those same statistics to validation/test. Do not
   reuse v3 normalization or compute statistics across all Polissia now.
4. Recompute class counts and agree the sampling policy for the revised labels.
5. Export, audit, and freeze the training version before selecting it as current.

### Agreed missing-input policy for the later patch rebuild

Implemented and applied to the completed v4 dataset on 2026-11-14. Reject a patch if **any required channel** has more than 5% invalid
pixels. For 256 x 256 patches, 3276 missing pixels pass; 3277 fail. Coverage is
checked independently for every channel, never averaged across channels. Radar
polarization pairs share a common mask; the six S2 bands also share observation
validity. Candidate patches must lie entirely inside the AOI. The planner and
exporter both enforce the threshold.

When `--input-manifest` is supplied, `plan` defaults to the agreed 5% rule.
Legacy planning without a manifest defaults to zero missing pixels, preserving
the previous workflow. `--max-missing-fraction 0.05` can make the setting explicit.
The frozen plan records `max_missing_fraction` and ordered `channel_valid_masks`.
The legacy `add-nisar` operation refuses partial-input plans; plan the complete
channel set together from its manifest instead.

Exported image TIFFs retain NoData -9999 for missing channels. Matching
`input_validity/<patch_id>.tif` files hold a C x 256 x 256 UInt8 mask, 1 valid /
0 missing, in exactly the same grid and channel order. Each index record stores
the validity path and `valid_input_pixels` counts. No neighbor interpolation is
performed. There is no invented radar measurement across a swath gap or boundary.

Normalization is computed only during the later training-patch build, using
**valid training values separately for each channel**. `normalization.json`
records `valid_pixels_per_channel`, in addition to the training patch pixel count.
NoData and validation/test pixels contribute to neither mean nor standard
deviation. No full-AOI or v3 statistics are reused.

At read time, `PatchDataset` fills missing entries with zero after normalization
and appends binary channel-validity indicators by default for partially observed
datasets. The physical stack remains 14 predictors; its default model tensor has
**28 channels: 14 normalized predictors followed by 14 validity masks**. The model
must use `dataset.model_channels` as its input count. Binary validity masks are
not standardized. Original v3 datasets retain their previous model input count.
With `normalize=False`, missing values are zero-filled in physical units but
remain explicitly marked missing.

```python
from bogsignal.dataset.reader import PatchDataset

dataset = PatchDataset('path/to/future/completed/v4')
sample = dataset[0]
# sample['image']: 28 x 256 x 256 with normalized predictors + binary masks
# sample['input_valid']: 14 x 256 x 256 boolean validity
# sample['valid']: H x W label eligibility, independent of missing predictors
# sample['evaluation_valid']: label eligibility AND validity of all 14 predictors
# dataset.model_channels: 28
```

For architectures that accept masks as a separate input, request
`append_validity=False` and explicitly supply `sample['input_valid']`; the image
then contains 14 zero-filled predictors. Silently dropping validity information
is not the recommended training interface.

Missing predictors do not invalidate known labels. Keep loss-ignore handling
separate from input validity. Primary sensor ablations must use the same jointly
valid evaluation pixels (`evaluation_valid`), including when comparing a subset
of sensors. Report partially observed coverage separately. No additional global
joint-coverage threshold is imposed at patch selection in this agreed revision.

The planner supports a completed input revision through `--input-manifest`.
The completed v4 plan was created with these settings:

```bash
PYTHONPATH=src .venv/bin/python -m bogsignal.dataset.pipeline plan \
  --data-root ../data --output ../data/dataset/s1s2nisar_256_v4 \
  --input-manifest ../data/inputs/preprocessing_v4/manifest.json \
  --label-manifest ../data/labels/ignore_fringe_v4/manifest.json \
  --min-label-fraction 0.80
```

The manifest supplies the channels and corrected masks and verifies all original
and derived hashes. The plan preserves acquisition companions and preprocessing
policy. The later existing `build` action computes training-only normalization
on exported physical values; dB inputs are already transformed and must not be
logged again. Labels and eligibility are taken from their then-approved sources.
Do not run the legacy `plan --include-nisar` command for v4: it selects the old
linear-power inputs.

New splits make raw v3/v4 model scores incomparable as a preprocessing ablation.
Use matching new splits and evaluation pixels for controlled model comparisons.
V3 TIFFs, metadata, sampling policy and normalization remain intact.

## Code validation

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Tests include conversion/floor/clipping behavior, pass/date/orbit mismatch
rejection, native identity mismatches, an end-to-end small input revision,
source preservation, no premature split/normalization files, existing-output
protection, and provenance-file tampering detection, alongside the v3 tests.

## First-stage local validation result (2026-11-14)

`data/inputs/preprocessing_v4` completed full-resolution readback and the final
67-file original/derived hash verification. No training split, training patch or
normalization file was created. All 38 native NISAR source headers passed checks:
composite release P05036, product version 1.0.20, specification version 1.6.0.
There was no confirmed NISAR processing-release mismatch requiring a mosaic rebuild.
Original S1 processor baselines remain unverified as documented above.

| Pass | Previously mask-valid pixels | Newly excluded nonpositive-power pixels | Retained pixels |
| --- | ---: | ---: | ---: |
| S1 ascending | 451,130,482 | 379,998 | 450,750,484 |
| S1 descending | 451,130,480 | 288,948 | 450,841,532 |
| NISAR ascending | 406,106,642 | 0 | 406,106,642 |
| NISAR descending | 372,767,617 | 0 | 372,767,617 |

The new partial-input patch policy is implemented and tested against small
synthetic fixtures only. V4 patch selection, geographic splits, normalization and
training remain pending the remaining agreed dataset changes. These counts are
input coverage, not a count of training patches or a model-quality result.
## Label revision: fill small holes before deciding on new ignore masks

Agreed 2026-11-14. The remaining original suggestions are separate experiments:
within-sensor polarization ratios (#4), multitemporal averaging (#6), and temporal
statistics (#7). They are not included in the corrected baseline input package.
The next baseline label stage fills tightly constrained holes first; a new
uncertainty-to-ignore rule remains pending inspection and agreement.

From `ml/`:

```bash
PYTHONPATH=src .venv/bin/python -m bogsignal.dataset.label_holes \
  --data-root ../data --output ../data/labels/hole_fill_v4_reviewed \
  --poland-boundary /home/krk/projects/Argos/data/targets/polissia_clip_provenance/PL_boundary.geojson.gz
```

Only training-eligible class-0 pixels with evidence code 0 (default Other) may
change. All other class assignments and exclusion codes are preserved. An optional
`--protected-negatives` aligned raster protects reviewed negatives: only explicit
zero values are editable; nonzero and NoData values block filling. No reviewed
negative raster was available for this local run, so protection relies on the
existing evidence codes and the source/conflict rules described here.

A hole is an 8-connected component of at most four Other pixels. Its entire
surrounding ring must belong to exactly one supported class (1-5), and every
hole pixel must have at least five original neighbors of that class. An opening
to a larger Other region, an image edge, an ignore pixel, or a different ring
class prevents filling. This is one pass on original labels: no cascading growth.
A five-pixel halo ensures results are independent of processing-tile boundaries.

For peatland classes 4/5, each proposed replacement additionally requires the
existing country-specific evidence: EWM class 10 inside Poland, GPM class 1
outside Poland, and matching WorldCover cropland (40) or non-cropland
(10/20/30/60). The original OSM water/wetland exclusions still block peat.
Known GWL wetland/flooded-flat evidence cannot be overwritten through this rule.
Consequently, unsupported peat pixels stay Other even if surrounding labels vote
for peat. Neighbor votes alone cannot establish peat or its land-cover subtype.

The revision contains `labels.tif`, `hole_fill_class.tif`, and `manifest.json`.
The change raster records 0 unchanged, 1-5 inferred replacement class, and 255
outside the source raster. The manifest records source/output hashes, per-class
replacement counts, and the exact rule. The original training-eligibility raster
is referenced unchanged; no new ignore pixels are created. Every changed pixel
is checked against its original Other label and recorded replacement class.

`hole_fill_v4` is an earlier diagnostic run that did not yet preserve all original
OSM peat exclusions; it is superseded by `hole_fill_v4_reviewed` and must not be
used for the dataset. The word reviewed in the revision name denotes code/source
rule checks, not independent field validation or human review of every pixel.
Neither revision changes the labels used by existing datasets or automatically
selects a label source for a future plan.

Review the hole-filled revision before choosing a boundary/fringe ignore rule.
Any later ignore mask must retain its own reason codes and be derived from the
approved hole-filled labels. Only after that label decision should the final
label sources be wired into patch planning, new geographic splits constructed,
and training-only normalization fitted. The final planner uses the revised label source through `--label-manifest`; default legacy paths remain available for reproducing older releases.
The completed reviewed run filled 26,468 Other pixels: 6,972 Swamp,
18,162 Marsh, and 1,334 Permanent water. No peat replacements passed the
source-evidence restrictions. This is 10.5872 km2 on the 20 m grid and remains
inferred weak supervision, not additional ground truth. The current 29-test
suite passes, including hole topology, source support and tile-boundary checks.

## Completed one-pixel ignore revision and relaxed label coverage

The one-pass 8-neighbor outer fringe marks only trainable Other pixels adjacent
to any eligible class 1-5. Positive labels are not eroded. Existing raw exclusion
codes 6-9 and outside-AOI 255 are unchanged. New fringe pixels use raw class 10,
with training eligibility 0; loss/export targets become 255. The separate reason
raster encodes 1 new fringe, 2 pre-existing ignore, 0 retained, 255 outside AOI.
There is no repeated dilation or automatic fringe around unclassified OSM code 6.
A protected-negative mask can be supplied, with nonzero/NoData values protected;
none was available for this run. Pixels beside confirmed water and peat classes
are included because the agreed rule covers all minority classes 1-5.

```bash
PYTHONPATH=src .venv/bin/python -m bogsignal.dataset.label_ignore \
  --data-root ../data --output ../data/labels/ignore_fringe_v4 \
  --label-revision ../data/labels/hole_fill_v4_reviewed
```

The completed run adds 6,484,531 ignored pixels (2,593.8124 km2). Full-resolution
readback independently recomputes the rule with a one-pixel halo; source hashes
and prior hole-filled output hashes are checked. The new label/eligibility rasters,
reason raster and their hashes are in `ignore_fringe_v4/manifest.json`. Existing
source labels and v3 remain unchanged.

The planner now accepts `--label-manifest` to freeze and use this label revision;
export uses the exact label and eligibility paths recorded by the plan. Revised
input or label manifests default to minimum label coverage 0.80; legacy planning
retains 0.95. Override with `--min-label-fraction`. Planning, export and audit all
check the threshold. For 256 x 256 patches, at least 52,429 trainable labels are
required at 80%. There is no relaxation of the separate per-channel 95% valid
imagery rule. Neither normalization nor label statistics include synthetic labels
at ignored pixels; image normalization still uses valid training imagery.

Compare retained patch counts and class distributions at 0.80, 0.90 and 0.95
before freezing the final new splits. Final planning retained 3,753 patches after geographic split buffers; the coverage comparison counts candidates before buffering.

## Completed v4 release

The release is `data/dataset/s1s2nisar_256_v4`. Exported images, labels and per-channel
validity masks were read back; the final audit checked spatial separation, group
isolation, channel order, per-patch class counts, imagery/label coverage and reader
behavior. All consumed file hashes were checked. The completed training-only
normalization has separate valid-pixel counts for all 14 channels. Sampling
metadata is synchronized across plan, index and completion marker. CURRENT.json
was changed only after audit success; its prior contents are saved in the v4 folder.

| Minimum trainable label coverage | Input-valid candidate patches | Swamp-rich | Marsh-rich |
| --- | ---: | ---: | ---: |
| 80% (selected) | 4,730 | 269 | 214 |
| 90% | 4,612 | 224 | 170 |
| 95% | 4,219 | 144 | 82 |

These candidate counts precede geographic buffering. Rich means the class occupies
at least 5% of the whole patch. Final train/validation/test counts are 2,609/585/559.
The audit measures consistency with weak reference labels, not independent field
accuracy. No model has been trained as part of this dataset creation.
