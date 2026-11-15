"""Protect physical averaging, missing dates and latest-valid temporal baselines."""
import unittest
import contextlib
import io
import json
import tempfile
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin

from bogsignal.dataset.temporal_provisional import summarize, build
from bogsignal.dataset.temporal import repeat_features
from bogsignal.dataset.pipeline import sha


class TemporalTests(unittest.TestCase):
    def test_vectorized_quantiles_match_numpy_with_varying_missing_dates(self):
        rng = np.random.default_rng(17)
        power = rng.uniform(.0001, .9, (5, 2, 1, 4))
        power[0, :, 0, 0] = np.nan
        power[:2, :, 0, 1] = -9999
        power[2, :, 0, 2] = 0
        _, _, extras, _, keep = summarize(power, 3)
        valid = np.isfinite(power).all(1) & (power > 0).all(1)
        masked = power.copy()
        masked[~np.broadcast_to(valid[:, None], power.shape)] = np.nan
        db = np.clip(10 * np.log10(masked), np.array([-40., -45.])[None, :, None, None],
                     np.array([5., 0.])[None, :, None, None])
        self.assertTrue(keep.all())
        np.testing.assert_allclose(extras[1], np.nanpercentile(db, 10, axis=0), rtol=1e-6)
        np.testing.assert_allclose(extras[2], np.nanpercentile(db, 90, axis=0), rtol=1e-6)

    def test_matched_dataset_export_preserves_splits_labels_masks_and_optical_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, series, output = root / "data", root / "series", root / "export"
            original = data / "dataset/s1s2nisar_256_v4"
            original.mkdir(parents=True)
            series.mkdir()
            names = ["S2_" + b for b in ["B2", "B3", "B4", "B8", "B11", "B12"]]
            names += [f"S1_{p}_{pol}" for p in ["ASCENDING", "DESCENDING"] for pol in ["VV", "VH"]]
            names += [f"NISAR_{p}_{pol}" for p in ["ASCENDING", "DESCENDING"] for pol in ["HH", "HV"]]
            records = []
            for i in range(9):
                split = "train" if i < 6 else "validation" if i < 8 else "test"
                record = dict(id=f"patch{i}", row=i * 256, col=0, split=split, counts=[0] * 6,
                    images=f"{split}/images/patch{i}.tif", labels=f"{split}/labels/patch{i}.tif",
                    input_validity=f"{split}/input_validity/patch{i}.tif")
                record["counts"][i % 6] = 65536
                profile = dict(driver="GTiff", width=256, height=256, crs="EPSG:3035",
                    transform=from_origin(5000000, 3500000 - i * 5120, 20, 20), compress="DEFLATE")
                image = np.full((14, 256, 256), .1, "float32")
                image[6:10] = -10
                image[10:14] = -8
                for key, array, nodata in [("images", image, -9999),
                    ("labels", np.full((1, 256, 256), i % 6, "uint8"), 255),
                    ("input_validity", np.ones((14, 256, 256), "uint8"), 255)]:
                    path = original / record[key]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with rasterio.open(path, "w", **profile, count=len(array), dtype=array.dtype, nodata=nodata) as ds:
                        ds.write(array)
                records.append(record)
            (original / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
            for name in ["normalization.json", "training_sampling.json", "complete.json", "audit.json"]:
                (original / name).write_text("{}")
            (original / "plan.json").write_text(json.dumps(dict(channels=[dict(name=n) for n in names])))
            groups = []
            for p in ["ASCENDING", "DESCENDING"]:
                folder = series / p
                folder.mkdir()
                sources = []
                for date in range(6):
                    path = folder / f"source{date}.npy"
                    power = np.full((9, 2, 256, 256), .1 + .01 * date, "float32")
                    if date < 2:
                        power[:, :, 0, 0] = -9999  # four remaining dates fail minimum five
                    np.save(path, power)
                    sources.append(dict(id=f"{p}{date}", cache=str(path), cache_sha256=sha(path)))
                (folder / "complete.json").write_text(json.dumps(dict(sources=sources)))
                groups.append(dict(key=[p, 1, 1, "2005", "DHDH"], folder=str(folder), records=records))
            (series / "series_plan.json").write_text(json.dumps(dict(groups=groups,
                native_processing_signature=["P05023", "1.0.11", "1.5.0"], window_utc=["2026-06-17", "2026-10-21"])))
            (series / "acquisition_complete.json").write_text("{}")
            with contextlib.redirect_stdout(io.StringIO()):
                result = build(data, series, output, minimum_observations=5)
            self.assertEqual(result["summary"]["train"]["patches"], 6)
            arrays = []
            for variant in ["T0", "T1", "T2"]:
                folder = output / variant
                exported = [json.loads(line) for line in (folder / "index.jsonl").read_text().splitlines()]
                self.assertEqual([(r["id"], r["split"]) for r in exported], [(r["id"], r["split"]) for r in records])
                with rasterio.open(folder / records[0]["images"]) as ds:
                    arrays.append(ds.read())
                with rasterio.open(folder / records[0]["labels"]) as ds:
                    self.assertEqual(ds.read(1)[0, 0], 255)
                with rasterio.open(folder / records[0]["input_validity"]) as ds:
                    validity = ds.read()
                    self.assertEqual(validity[0, 0, 0], 1)  # never leak NISAR availability into S2 masks
                    self.assertEqual(validity[10, 0, 0], 0)
                self.assertEqual(exported[0]["counts"][0], 65535)
            self.assertEqual([len(a) for a in arrays], [14, 14, 26])
            np.testing.assert_array_equal(arrays[0][:6], arrays[1][:6])
            np.testing.assert_array_equal(arrays[1][:6], arrays[2][:6])
            self.assertAlmostEqual(float(arrays[0][0, 0, 0]), .1)
            self.assertEqual(arrays[0][10, 0, 0], -9999)
            np.testing.assert_allclose(arrays[0][10:14, 10, 10], 10 * np.log10(.15), rtol=1e-6)
            np.testing.assert_allclose(arrays[1][10:14, 10, 10], 10 * np.log10(.125), rtol=1e-6)

    def test_mean_is_in_power_and_latest_is_not_the_mean(self):
        power = np.array([.01, .1, 1.])[:, None, None, None] * np.ones((1, 2, 1, 1))
        latest, mean, extras, count, keep = summarize(power, 3)
        self.assertTrue(keep[0, 0])
        self.assertEqual(count[0, 0], 3)
        np.testing.assert_allclose(latest[:, 0, 0], [0., 0.])
        np.testing.assert_allclose(mean[:, 0, 0], 10 * np.log10(.37), rtol=1e-6)
        self.assertNotAlmostEqual(float(mean[0, 0, 0]), -10.)
        np.testing.assert_allclose(extras[1, :, 0, 0], [-18., -18.])
        np.testing.assert_allclose(extras[2, :, 0, 0], [-2., -2.])

    def test_invalid_polarization_excludes_whole_date_and_latest_falls_back(self):
        power = np.ones((4, 2, 1, 2), dtype="float32") * .1
        power[3, 1, 0, 0] = 0
        power[2, 0, 0, 1] = np.nan
        power[3, 0, 0, 1] = -9999
        latest, mean, extras, count, keep = summarize(power, 3)
        self.assertEqual(count.tolist(), [[3, 2]])
        self.assertEqual(keep.tolist(), [[True, False]])
        np.testing.assert_allclose(latest[:, 0, 0], [-10., -10.], atol=1e-6)
        self.assertTrue((mean[:, 0, 1] == -9999).all())
        self.assertTrue((extras[:, :, 0, 1] == -9999).all())

    def test_two_date_helper_rejects_nonpositive_power(self):
        with self.assertRaises(ValueError):
            repeat_features(np.zeros((2, 2, 3)))


if __name__ == "__main__":
    unittest.main()
