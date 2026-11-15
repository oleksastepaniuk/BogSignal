# Provisional monthly temporal pilot

Started 2026-11-15. User approved provisional data and a smaller initial pilot
instead of acquiring every available date. Original v4 and A-D models are kept.

Completed 2026-11-15: all 120 source products and 24 coverage groups acquired;
3,646 matched patches exported (2,578 train, 543 validation, 525 locked test),
107 candidates excluded by the declared validity rules. All four seed-17 models
trained and their frozen checkpoints evaluated. The 52 repository tests pass.
See the [comparison and individual reports](experiments/temporal_provisional_seed17.md).

Temporal averaging raised combined swamp/marsh validation recall from 40.76%
to 46.87% (+6.11 percentage points), with reference precision 44.26% to 42.53%,
F2 41.41% to 45.93%, and IoU 26.93% to 28.70%. Adding variability features
instead produced 33.72% recall and 57.16% reference precision. Retain the mean
representation for recall-focused confirmation across seeds. These are one-seed,
weak-reference validation findings; the test set remains unscored.

## Question and fixed design

### All-sensor follow-up against D

Completed 2026-11-15: TD trained for 41 epochs, selecting epoch 31. On the same
543 validation patches / 34,690,287 pixels, D versus TD combined wetland recall
was 39.75% versus 35.00%, reference precision 49.40% versus 56.06%, F2 41.36%
versus 37.85%, and IoU 28.25% versus 27.47%. Six-class macro IoU was essentially
unchanged (35.48% versus 35.50%). This pilot does not support replacing D for
recall-focused use. The earlier C-style mean gain did not carry over to this
all-sensor comparison. These are one-seed, weak-reference results with the
processing/training confounds below; they do not establish that averaging is
intrinsically harmful. [Full comparison](experiments/temporal_d_comparison_seed17/report.md)
and [standalone HTML](experiments/temporal_d_comparison_seed17/report.html).
The 56 repository tests pass, including the common-pixel alignment checks.

`ml/configs/temporal_d.json` trains TD with all 14 physical channels: static S2,
static S1 and temporal-mean NISAR, using the T1 dataset and D's seed-17 recipe.
This adds S1 to the previous C-style temporal model. The original D checkpoint
is retained. `bogsignal.training.temporal_d_compare` aligns validation patch IDs,
requires unchanged labels and baseline-valid pixels, and rescores saved D and TD
predictions on the same temporal validation area. The original 585-patch D score
is contextual only; the primary D-versus-TD table uses 543 patches for both.

This practical baseline comparison changes training eligibility (2,609 versus
2,578 patches) and NISAR processing/date regime as well as averaging. It does not
isolate averaging causally; that would require an all-sensor latest-date run on
the same provisional series. The locked test remains unscored.

From `ml`, train with `uv run --locked --extra training python -m
bogsignal.training.train --config configs/temporal_d.json --train-only`, then run
`uv run --locked --extra training python analyze_temporal_d.py` once to generate
the completed checkpoint evaluation and matched comparison artifacts.

### Initial C-style temporal variants

Does temporal NISAR information improve the S2 + NISAR U-Net when optical data,
geography and supervision are fixed?

| Run | NISAR representation | Selected predictors / model channels |
| --- | --- | ---: |
| T0 | Latest valid provisional observation per anchor pixel | 10 / 20 |
| T1 | Mean valid linear power, then dB and clipping | 10 / 20 |
| T2 | T1 plus per-polarization dB std, p10 and p90 | 22 / 44 |
| TB | Matched static S2 + S1 control; no NISAR predictor | 10 / 20 |

All runs use ResNet-34 U-Net, six output classes, seed 17, the A-D optimizer,
sampling formula, augmentation, epoch budget and validation checkpoint-selection
rule. No model is trained on the test fold. Combined-class and six-class
validation metrics are reported separately, as in A-D.

T0 is rebuilt from the same provisional series as T1/T2. The original validated
C checkpoint is not a direct temporal baseline: product maturity, imagery dates
and potentially eligible locations differ.

## Source selection

Argos `data/nisar_coverage_polissia.geojson` contains 323 provisional records.
Of these, 220 compatible HH/HV products cover the v4 anchor geometry, spanning
24 pass/track/frame/mode/polarization groups with 8–10 dates each. This is a file
count across the region, not 220 observations of every pixel.

For the initial pilot, select one date per month, June–October: closest to the
middle of each month, preferring a >=30-second frame acquisition when available.
This rule uses catalogue timing/geometry, never labels or model results. It
selects **120 products**, five per group. Already acquired extra dates are kept
but excluded from the pilot. The frozen source manifest records exact dates.

Use only collection `NISAR_L2_GCOV_PROVISIONAL_V1`, CRID P05023, native product
version 1.0.11 and specification 1.5.0. Native identity, pass, track, frame,
timestamp, L-band GCOV type and 20 m co-located HH/HV/mask grids are checked.
Validated P05036 products are not mixed into this time series.

## Pixels, labels and features

Each pixel retains its original v4 pass/track/frame anchor. Never average across
different tracks or ascending/descending geometries. Require at least **three
valid joint HH/HV dates** in each pass. Invalid, nonpositive and nonfinite power
is excluded per observation. Save actual observation counts; nominal five dates
does not imply every pixel has five usable observations.

Retain v4's reviewed spatial exclusions. These are not a comprehensive
multitemporal interference detector; residual provisional radar artifacts remain
a limitation, and source clip diagnostics are recorded.

Intersect temporal eligibility with v4's common input validity. Retain patches
with >=95% common imagery coverage and >=80% trainable labels; outside the shared
scoring mask labels are ignored. Predictors retain their own validity masks:
S2/S1 values and masks do not encode NISAR temporal availability. T0/T1/T2 use identical
patch IDs, labels, masks, original geographic folds/buffers and recomputed matched
75/25 training sampling weights. Optical/S1 raw values and their training-only
normalization stay identical. New feature normalization uses valid matched
training pixels only.

Compute the mean in linear gamma-0 power before converting to dB. Use the same
HH/HV clipping bounds as v4. Standard deviation (population, ddof=0) and linearly
interpolated p10/p90 use per-date clipped dB values, ignoring invalid dates.
With 3–5 dates these are coarse temporal summaries, not reliable estimates of
annual hydroperiod or hydrological extremes.

## Reproduction

From Argos:

```bash
uv run --locked python -m argos.temporal_nisar --monthly-five --acquire --workers 3
```

The acquisition uses authenticated HTTP ranges over only the required compressed
HH/HV/mask chunks. Source caches are Float32 patch arrays with native identity,
byte counts and SHA-256 hashes. Failed transfers retry; completed caches resume.
No credentials are copied into manifests or reports. Processing manifest:
`Argos/data/temporal_sources/provisional_v1/series_plan.json`.

From BogSignal/ml after acquisition completes:

```bash
uv run --locked --extra training python -m bogsignal.dataset.temporal_provisional
uv run --locked --extra training python -m bogsignal.training.train --config configs/temporal_t0.json --train-only
uv run --locked --extra training python -m bogsignal.training.train --config configs/temporal_t1.json --train-only
uv run --locked --extra training python -m bogsignal.training.train --config configs/temporal_t2.json --train-only
uv run --locked --extra training python -m bogsignal.training.train --config configs/temporal_control_b.json --train-only
```

Finalize checkpoint validation, combined-class inference and ranking using the
commands in [TRAINING.md](../ml/TRAINING.md), then compare matched temporal runs.
Primary contrasts: T1−T0 (averaging), T2−T1 (additional variability features).
Report shared coverage, class support, observation counts, recall/precision/F2,
IoU/Dice and weak-reference AP. Keep the optional area-budget ranking distinct
from ordinary per-pixel classification. No single-seed validation difference is
an independent-accuracy or significance claim.
