# Sensor-ablation training

For a training-only stage, append `--train-only` to `bogsignal-train`. Validation
during training still selects the checkpoint, but final inference, prediction
exports, plots and reports are deferred. `complete.json` then explicitly records
`training_complete_evaluation_deferred`; `results.json` retains the selected
epoch's training-time validation record. It does not claim checkpoint prediction
reverification. The report command refuses these runs until final evaluation is
completed. B/C/D use the same seed-17 recipe as A with their selected channels.

When evaluation is requested, finish a deferred run with:

```bash
uv run --locked --extra training python -m bogsignal.training.finalize --run RUN_DIRECTORY
uv run --locked --extra training python -m bogsignal.training.combined --run RUN_DIRECTORY
uv run --locked --extra training python -m bogsignal.training.evaluate --run RUN_DIRECTORY
```

`finalize` writes verified predictions, results and figures into a separate
`validation_evaluation_v1/` directory. Original training results and completion
hashes are preserved, including their historical deferred status. Reports verify
and use the new completed evaluation artifacts. Evaluators accept `--output` for
a separately versioned retry; reports select the latest numerically versioned
folder with a completion marker and verify its provenance. Failed/incomplete
attempts are retained and excluded when a completed version exists.

Generate the individual A-D reports and matched comparison with
`python -m bogsignal.training.compare --runs A_RUN B_RUN C_RUN D_RUN --output
../docs/experiments/sensor_comparison_seed17.md`. The comparison refuses mismatched
dataset hashes, reference support, scored-pixel counts, seeds or area budgets.

Reproduction checks require identical checkpoint hashes and reference support.
The initial evaluator required bitwise confusion-matrix identity; separate
BF16/cuDNN benchmarked inference passes showed small numerical differences.
The updated guard records both matrices and exact-match status and bounds every
defined per-class IoU/Dice/precision/recall/F2 and macro IoU difference to 0.001
absolute score units (0.1 percentage points). Reports disclose the actual maximum
drift. Larger discrepancies fail. This engineering tolerance is not statistical
uncertainty or an accuracy improvement, and does not change training, checkpoint
selection, labels or thresholds. Incomplete earlier attempts remain preserved.

Run from `BogSignal/ml` inside Linux/WSL with an NVIDIA GPU supporting bfloat16.
Install the pinned optional training dependencies (PyTorch 2.9.1 CUDA 12.8,
torchvision 0.24.1 and segmentation-models-pytorch 0.5.0):

```bash
uv sync --locked --extra training
uv run --locked --extra training python -m unittest discover -s tests -v
uv run --locked --extra training bogsignal-train --config configs/experiment_a_s2.json --smoke
uv run --locked --extra training bogsignal-train --config configs/experiment_a_s2.json
uv run --locked --extra training python -m bogsignal.training.evaluate \
  --run ../data/experiments/A_s2_resnet34_seed17
uv run --locked --extra training python -m bogsignal.training.combined \
  --run ../data/experiments/A_s2_resnet34_seed17
uv run --locked --extra training python -m bogsignal.training.report \
  --run ../data/experiments/A_s2_resnet34_seed17 \
  --output ../docs/experiments/A_s2_resnet34_seed17.md
```

The first run downloads the official torchvision ImageNet ResNet-34 weights into
`data/training_cache/torch`, checking their published hash. It builds an immutable
derived cache of normalized selected predictors, their validity masks, labels,
and the common full-stack evaluation mask. Dataset metadata hashes identify the
cache. Source datasets and their splits are not changed. The cache uses several
GB on disk; it avoids repeatedly decompressing all 14 TIFF bands each epoch.

`experiment_a_s2.json` selects only B2/B3/B4/B8/B11/B12. Its model receives 12
channels: six normalized bands followed by six binary validity masks. No radar
values or radar masks enter the model. The model predicts six full-resolution
logit maps. Target 255 is excluded from cross-entropy and metrics. Imagery
normalization comes exclusively from v4's frozen training-fold statistics.

The U-Net uses a ResNet-34 ImageNet-initialized encoder and decoder widths
256/128/64/32/16. Original RGB kernels are repeated over selected imagery bands
and scaled by 3/C; validity-mask kernels start at zero. The remaining encoder
weights are loaded unchanged. All layers are trainable.

The configuration freezes AdamW (encoder LR 1e-4, decoder LR 1e-3, weight decay
1e-4), cosine LR decay, unweighted cross-entropy, global gradient clipping at
1.0, batch size 16, bfloat16 autocast and flips/quarter-turn augmentation.
The frozen training-only patch sampling weights are used with replacement for
2,609 draws per epoch. Validation is uniform, unaugmented and in evaluation mode.
Maximum duration is 50 epochs, with a 15-epoch minimum and patience of 10 epochs
on validation six-class macro IoU. CUDA/cuDNN benchmarking means runs are seeded
but not guaranteed bitwise identical across systems/kernels.

The smoke check uses selected minority-rich training patches only, for 120
optimizer steps at higher learning rates; it must substantially reduce loss and
reach 90% training-pixel accuracy. It reports measured step duration and memory.
Those weights are discarded. Full training starts in a separate process from
the specified pretrained initialization with the original learning rates.

Run outputs are under `data/experiments/A_s2_resnet34_seed17`, with a separate
`_smoke` directory. Existing run directories are refused, preserving prior runs;
choose a new output path in a copied config to rerun. These local artifacts are
ignored by Git:

- `manifest.json`: frozen resolved config, source and dataset metadata hashes,
  package/GPU versions, actual parameter count and pretrained-file hash.
- `source/`: exact training code, reader, config and dependency-lock snapshots.
- `history.json` / `progress.json`: learning curves and live progress.
- `best.pt`: checkpoint selected solely by validation macro IoU.
- `validation_predictions.npy`: predictions in cached validation `ids.json` order.
- `results.json`: pooled confusion matrix, per-class IoU/Dice/precision/recall,
  majority-Other baseline, runtime and peak allocated VRAM.
- `learning_curves.png`, `confusion_matrix.png`, `validation_examples.png`.
- `complete.json`: written after reloading and checking the selected checkpoint,
  confirming dataset metadata is unchanged, and generating plots.

Scores use label-eligible pixels jointly valid in **all 14 v4 predictors**, the
same area intended for subsequent sensor comparisons. Loss retains eligible
labels at partly observed pixels, using explicit predictor masks. Weak-reference
scores measure agreement with the historical/inferred label product; they are
not independent contemporary wetland accuracy. The test split stays locked
until sensor, temporal and architecture choices are frozen. One seed is the
initial screening run; headline sensor comparisons require repeated seeds.

## Recall-focused validation

### Combined wetland class inference

`python -m bogsignal.training.combined --run RUN_DIRECTORY` evaluates the existing
checkpoint as five classes by summing Swamp and Marsh softmax scores **before**
argmax. Every pixel still receives a class. The other scores (Other, water and
the two peat classes) remain separate; reference labels use the matching mapping
`[0,1,1,2,3,4]`, preserving ignore 255. This is not a threshold/ranking policy or
five-class training. It uses the original common validation mask and verifies
the original six-class confusion matrix exactly.

Outputs live in `combined_evaluation_v1/`: five-class predictions in cached
validation ID order, metrics, confusion figure, code snapshots and integrity
hashes. Existing outputs are refused; `--output` permits a new version. The report
automatically includes the default combined evaluation folder. Five-class macro
averages are not directly comparable with six-class macro averages; compare the
same merged wetland target instead. Test remains unscored.

### Optional candidate-ranking analysis

The standard confusion-matrix metrics now also report per-class F2, macro recall,
swamp/marsh macro subtype recall, and grouped Swamp-or-Marsh detection. Grouped
detection accepts subtype confusion; water and peat classes remain separate.
These additions do not change the macro-IoU checkpoint-selection rule.

Run `python -m bogsignal.training.evaluate --run RUN_DIRECTORY` after training to
compute ranking metrics from the frozen checkpoint, without retraining. It
reproduces and checks the original confusion counts, then writes a separate
`recall_evaluation_v1/` folder. Existing folders are refused; `--output` selects
a new version. It preserves the original checkpoint, results, manifest, history
and completion hashes. Test is never scored by this command.

Scores are float32 softmax(Swamp), softmax(Marsh), and their sum. Exact AP uses all
distinct score thresholds with no interpolation. Recall is also reported at
1%, 5%, 10% and 20% flagged-area budgets, on the same label-eligible, jointly valid
validation mask. Equal-score pixels remain together: the conservative threshold
includes whole tied groups only when they fit the budget. There is no label-based
tie breaking. Use identical masks and budgets for subsequent sensor comparisons.
The `--budgets` argument is available for a separately declared evaluation policy;
do not select a budget after inspecting results and treat it as predeclared.

The local folder contains `validation_scores.npy` (three score maps per patch in
cached validation ID order), `results.json` (including a downsampled plot curve,
exact AP and exact budget counts), `recall_evaluation.png`, source/provenance
snapshots and completion hashes. The report generator detects this folder by
default; `--evaluation` selects another evaluation version. Reports copy its
figure and reviewable summaries into `docs/experiments`.

Recall measures recovery of mapped positives, not all real wetlands. Precision,
AP, F2 and IoU remain weak-reference agreement measures. There is no independent
reviewed precision floor or site/object recall without appropriate annotations;
the evaluator records these as unavailable rather than inventing them. High
recall alone can be obtained by flagging everything. A reviewed benchmark must
sample predicted negatives as well as detections and known wetlands, account for
its sampling design, and remain separate from threshold tuning.
