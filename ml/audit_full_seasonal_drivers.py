"""Diagnostic channel swaps for the three full seasonal patches."""
import json
from pathlib import Path

import numpy as np
import rasterio

from bogsignal.inference.pipeline import model_input, probabilities
from bogsignal.inference.torch_adapter import load_predictor
from acquire_full_seasonal_patches import DATASET, OUTPUT, SELECTED

ROOT = Path(__file__).resolve().parents[1]
stats = json.loads((DATASET / "normalization.json").read_text())
predictor = load_predictor(ROOT / "data/experiments/D_s2s1nisar_resnet34_seed17/best.pt")
results = {}
for site in SELECTED:
    months = {}
    for name in ["june", "october"]:
        folder = OUTPUT / site / name
        with rasterio.open(folder / "model_inputs.tif") as ds:
            values = ds.read()
        with rasterio.open(folder / "input_validity.tif") as ds:
            valid = ds.read() == 1
        months[name] = (values, valid)
    june, j_valid = months["june"]
    october, o_valid = months["october"]
    rows = {}
    for variant, groups in [("june", []), ("S2_only", [slice(0, 6)]),
                            ("S1_only", [slice(6, 10)]), ("NISAR_only", [slice(10, 14)]),
                            ("all_october", [slice(0, 14)])]:
        values, valid = june.copy(), j_valid.copy()
        for group in groups:
            values[group] = october[group]
            valid[group] = o_valid[group]
        p = probabilities(predictor(model_input(values, valid, stats)[None]), 256, "logits")
        combined = np.concatenate((p[:1], p[1:3].sum(0, keepdims=True), p[3:]), axis=0)
        prediction = combined.argmax(0)
        common = j_valid.all(0) & o_valid.all(0)
        rows[variant] = {"wetland_pixels_on_final_common_mask": int(((prediction == 1) & common).sum()),
                         "common_pixels": int(common.sum())}
    results[site] = rows
    print(json.dumps({"site": site, "variants": rows}), flush=True)
(OUTPUT / "channel_swap_diagnostic.json").write_text(json.dumps(results, indent=2) + "\n")
