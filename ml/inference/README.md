# Inference: OSM wetlands and Polissia

This workflow predicts six-class segmentation from the frozen v4 imagery and
normalization. It never changes training data, labels, splits, normalization,
or CURRENT.json. The frozen model D adapter is now provided below; checkpoints and generated
regional rasters remain local ignored data. Code lives in `src/bogsignal/inference/pipeline.py`.

Run commands from `BogSignal/ml/` with the existing environment. Outputs are
separate named directories under `data/inference/`.

## Plan the unclassified OSM wetland experiment

```bash
PYTHONPATH=src .venv/bin/python -m bogsignal.inference.pipeline plan \
  --data-root ../data --dataset ../data/dataset/s1s2nisar_256_v4 \
  --output ../data/inference/osm_unclassified_v4 --region osm-unclassified
```

The planner selects OSM polygons whose existing polygon vote record has code 6
(no assignable GWL subtype) or code 8 (tied/ambiguous fill). This selects entire
polygons, including any supported pixels they contain; it is not a pixel-level
filter for raw label codes 6/8. Polygon votes are used only to select targets.
They are never passed to the predictor. No training labels or loss masks are
needed by the prediction runner.

The plan saves projected `targets.geojson`, the channel specification, frozen
normalization, source hashes and output tile list. Planning is not inference.
Verify coverage in the finished outputs before interpreting a polygon result.

## Plan Polissia-wide mapping

```bash
PYTHONPATH=src .venv/bin/python -m bogsignal.inference.pipeline plan \
  --data-root ../data --dataset ../data/dataset/s1s2nisar_256_v4 \
  --output ../data/inference/polissia_v4 --region polissia
```

Both modes use the full common grid. Outside requested geometry remains NoData.
The default model tile is 256 x 256 with a central 128 x 128 output core and a
64-pixel context halo. Core regions partition the grid without overlap; the model
sees neighboring context and its outer predictions are discarded to reduce tile
edge artifacts. This does not guarantee seam-free results. Raster-edge context
is padded with missing values/masks. `--core-size` can change the context amount
when planning a separately named run.

A contextual tile requires at least 95% valid pixels in every channel within the
AOI portion of the context. Outside-AOI context is masked. Predictions are emitted
only where all required channels are valid. Thus Polissia-wide means requesting
the whole AOI, not guaranteeing wall-to-wall predictions. Missing NISAR coverage
and rejected context remain explicit gaps. A separately trained sensor-subset
model would be needed to cover areas missing a required sensor extensively.

## Plug in the trained model

The adapter interface is `module:factory`. The factory receives the checkpoint
path and returns a callable taking a NumPy float32 array shaped [B,28,256,256].
The first 14 channels are v4-normalized predictors; the final 14 are binary
validity masks. Missing normalized values are zero. The callable returns NumPy
scores [B,6,256,256] in class order Other, Swamp, Marsh, Permanent water, Peatland
cropland, Peatland non-cropland. The runner currently sends B=1.

A Torch adapter must create the exact trained architecture with 28 input and
6 output channels, load its weights, call `eval()`, use `torch.inference_mode()`,
move inputs to the desired device, and return CPU NumPy outputs. The generic inference runner has no Torch requirement. The model D adapter
uses Torch and the trained architecture from the project ML environment.

Once that adapter and checkpoint exist:

```bash
PYTHONPATH=src .venv/bin/python -m bogsignal.inference.pipeline run \
  --data-root ../data --output ../data/inference/osm_unclassified_v4 \
  --predictor your_model_adapter:load_predictor --checkpoint /path/to/model.pt \
  --score-kind logits
```

The generic adapter name and checkpoint path above are placeholders; use the
concrete model D invocation below for this project. For a callable returning normalized probabilities instead, specify
`--score-kind probabilities`. Wrong dimensions, nonfinite scores or invalid
probabilities fail the run. No additional dB conversion or normalization fitting
is performed. The predictor must not apply v4 normalization a second time.

## Results and interpretation

- `classes.tif`: class IDs 0-5; NoData 255.
- `probabilities.tif`: six float32 bands; NoData -9999.
- `confidence.tif`: maximum class probability; NoData -9999. It is not a calibrated
  correctness estimate or a measure of independent ground truth.
- `coverage.tif`: 1 predicted, 0 requested AOI pixel without a prediction,
  255 outside the requested AOI/geometry.
- `polygon_summary.json`: target/predicted pixel counts, coverage fraction,
  per-class counts and fractions among predicted pixels for each target polygon.
  Polygons are not forced into one class; mixtures remain visible.
- `complete.json`: written only after output consistency readback, source hash
  checks and checkpoint hash checks. It records the model checkpoint digest and
  output hashes. An absent marker means an incomplete run.

Failed/completed output TIFFs are never overwritten. Use a new planned output
folder for a different model or retry; there is no resume mechanism yet. Processing
is single-tile inference, not a throughput-optimized GPU batch scheduler. Keep all
six probability bands in capacity planning for whole-Polissia runs.

Predictions on unclassified OSM wetlands are useful for inspection and hypothesis
generation. They are not automatically accepted as labels or independent test
truth, and may overlap training geography. Keep sensor comparisons on the same
joint-valid held-out sites. Do not use predicted labels to silently fill the
training dataset.

## Tests

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -q
```

Synthetic tests cover normalization/masks, score validation, label-independent
end-to-end prediction, boundary context, explicit NoData outside targets and at
missing pixels, sparse-output NoData, georeferenced probability outputs, polygon
coverage summaries, source-change rejection and overwrite protection.

Local v4 planning check (2026-11-14): selected 123 unclassified/ambiguous OSM
wetland target features spanning 195 contextual tiles. The saved plan is

data/inference/osm_unclassified_v4/inference_plan.json.

This is a planned experiment only: no real model predictions have run. Synthetic
end-to-end tests passed as part of the 36-test suite. The v4 dataset folder was
not changed while this inference code was added.


## Frozen model D regional run (website)

The completed adapter is `bogsignal.inference.torch_adapter:load_predictor`.
It loads the exact saved 14-channel / 28-input ResNet-34 U-Net checkpoint,
checks its frozen digest, and returns six logits using CUDA BF16 when available.
Inputs must already be normalized by the pipeline using frozen training statistics.

Run in WSL from BogSignal/ml with the project .venv:

    PYTHONPATH=src .venv/bin/python -m bogsignal.inference.pipeline plan --data-root ../data --dataset ../data/dataset/s1s2nisar_256_v4 --output ../data/inference/polissia_D_v4_web --region polissia
    PYTHONPATH=src .venv/bin/python -m bogsignal.inference.pipeline run --data-root ../data --output ../data/inference/polissia_D_v4_web --predictor bogsignal.inference.torch_adapter:load_predictor --checkpoint ../data/experiments/D_s2s1nisar_resnet34_seed17/best.pt --score-kind logits

Existing output folders are never overwritten. Use a new output name for a new run.
Only a successful complete.json certifies completion. Native rasters remain 20 m.
Regional inference includes training and held-out geography and does not compute
new test-set evaluation scores. It does not change dataset labels or splits.

The website exporter reads completed probabilities and sums Swamp+Marsh before
five-class argmax, matching the presentation's inference decision. It exports
one nearest-neighbour overview PNG per layer, not one file per validation patch.
Source probabilities, classes, confidence, coverage and polygon_summary remain
in data/inference; missing all-sensor coverage stays explicit.

Completed local run: `polissia_D_v4_web`, 28,337 planned contextual tiles;
332,349,087 / 451,130,482 target pixels predicted (73.67%, 132,939.63 km2).
7,379 contexts were rejected by the unchanged validity policy. Output readback,
probability/class consistency, input digests and checkpoint digest passed.
The 20 m classes, six probability bands, confidence, coverage and polygon summary
are retained under data/inference/polissia_D_v4_web/.
