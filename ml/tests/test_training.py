"""Guard against sensor leakage, ignored-label scoring and broken mask inputs."""
import unittest
import copy
import tempfile
from pathlib import Path
import numpy as np
try:
    import torch
    import segmentation_models_pytorch
except ImportError as exc:
    raise unittest.SkipTest("Install the training extra to run training tests") from exc

from bogsignal.training.core import (confusion_counts, make_model, metrics,
                                    select_channels, selected_sample)
from bogsignal.training.evaluate import ranking_metrics
from bogsignal.training.core import combined_prediction, combined_target, combined_metrics
from bogsignal.training.report import completed_version
from bogsignal.training.compare import validate_comparison
from bogsignal.training.core import verify_reproduction


class TrainingTests(unittest.TestCase):
    def test_numerical_reproduction_records_small_drift_and_rejects_large_errors(self):
        expected = np.eye(6, dtype="int64") * 100000
        actual = expected.copy()
        actual[1, 1] -= 1
        actual[1, 0] += 1
        check = verify_reproduction(expected, actual)
        self.assertFalse(check["exact_confusion_match"])
        self.assertEqual(check["confusion_count_l1_difference"], 2)
        self.assertLess(check["max_absolute_metric_difference"], .001)
        actual[1, 1] -= 10000
        actual[1, 0] += 10000
        with self.assertRaises(ValueError):
            verify_reproduction(expected, actual)
        with self.assertRaises(ValueError):
            verify_reproduction(expected, expected + np.eye(6, dtype="int64"))
    def test_report_uses_latest_completed_version_and_preserves_failed_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for version in [1, 2, 10, 11]:
                folder = root / f"recall_evaluation_v{version}"
                folder.mkdir()
                if version in [2, 10]:
                    (folder / "complete.json").write_text("{}")
            self.assertEqual(completed_version(root, "recall_evaluation").name, "recall_evaluation_v10")
            self.assertTrue((root / "recall_evaluation_v11").is_dir())

    def test_comparison_rejects_different_data_and_area_budgets(self):
        base = dict(seed=17, dataset_hashes={"plan": "same"},
            validation=dict(classes={"Swamp": {"support": 2}}, scored_pixels=3),
            combined_class_evaluation=dict(combined_class=dict(scored_pixels=3)),
            recall_evaluation=dict(ranking={"Swamp or marsh": dict(scored_pixels=3,
                positive_pixels=2, area_budgets=[dict(budget_fraction=.1)])}))
        rows = [dict(copy.deepcopy(base), experiment=name) for name in "ABCD"]
        validate_comparison(rows)
        rows[3]["dataset_hashes"]["plan"] = "different"
        with self.assertRaises(ValueError):
            validate_comparison(rows)
        rows[3]["dataset_hashes"]["plan"] = "same"
        rows[3]["recall_evaluation"]["ranking"]["Swamp or marsh"]["area_budgets"][0]["budget_fraction"] = .2
        with self.assertRaises(ValueError):
            validate_comparison(rows)

    def test_combining_before_argmax_changes_winner_and_keeps_water_separate(self):
        p = torch.tensor([[[[.40, .05]], [[.30, .10]], [[.25, .10]],
                           [[.03, .70]], [[.01, .03]], [[.01, .02]]]])
        self.assertEqual(p.argmax(1).tolist(), [[[0, 3]]])
        self.assertEqual(combined_prediction(p).tolist(), [[[1, 2]]])
        target = torch.tensor([0, 1, 2, 3, 4, 5, 255])
        self.assertEqual(combined_target(target).tolist(), [0, 1, 1, 2, 3, 4, 255])
        self.assertEqual(target[-1].item(), 255)

    def test_five_class_metrics_ignore_unknown_and_use_correct_wetland_support(self):
        target = combined_target(torch.tensor([1, 2, 0, 3, 255]))
        pred = torch.tensor([1, 0, 1, 2, 1])
        cm = confusion_counts(target, pred, torch.ones(5, dtype=torch.bool), classes=5)
        result = combined_metrics(cm.numpy())
        self.assertEqual(result["scored_pixels"], 4)
        wetland = result["classes"]["Wetland (Swamp + Marsh)"]
        self.assertEqual(wetland["support"], 2)
        self.assertEqual(wetland["recall"], .5)
        self.assertEqual(wetland["precision"], .5)
        self.assertAlmostEqual(wetland["iou"], 1 / 3)

    def test_sensor_selection_excludes_radar_and_masks(self):
        sample = dict(image=np.arange(14, dtype="float32")[:, None, None],
                      input_valid=np.ones((14, 1, 1), dtype=bool),
                      mask=np.array([[1]]), evaluation_valid=np.array([[False]]))
        sample["input_valid"][6:] = False
        image, validity, _, evaluation = selected_sample(sample, list(range(6)))
        np.testing.assert_array_equal(image[:, 0, 0], np.arange(6))
        self.assertTrue(validity.all())
        self.assertFalse(evaluation.any())
        with self.assertRaises(ValueError):
            select_channels({"channels": [{"name": "S2_B2"}]}, ["S2_B2", "S2_B2"])

    def test_ignore_and_common_validity_excluded_from_confusion(self):
        target = torch.tensor([0, 1, 255, 2, 1])
        prediction = torch.tensor([0, 2, 0, 2, 1])
        keep = torch.tensor([True, True, True, False, True])
        cm = confusion_counts(target, prediction, keep)
        self.assertEqual(int(cm.sum()), 3)
        self.assertEqual(int(cm[1, 2]), 1)
        result = metrics(cm.numpy())
        self.assertAlmostEqual(result["classes"]["Swamp"]["iou"], .5)
        self.assertAlmostEqual(result["classes"]["Swamp"]["precision"], 1.)
        self.assertAlmostEqual(result["classes"]["Swamp"]["recall"], .5)
        self.assertAlmostEqual(result["classes"]["Swamp"]["f2"], 5 / 9)

    def test_wetland_presence_accepts_subtype_confusion(self):
        cm = np.zeros((6, 6), dtype="int64")
        cm[1, 2], cm[2, 1], cm[0, 1], cm[1, 0] = 3, 2, 1, 1
        result = metrics(cm)
        self.assertEqual(result["swamp_marsh_macro_recall"], 0.)
        grouped = result["wetland_detection"]
        self.assertEqual(grouped["tp"], 5)
        self.assertEqual(grouped["fn"], 1)
        self.assertEqual(grouped["fp"], 1)
        self.assertAlmostEqual(grouped["recall"], 5 / 6)
        self.assertAlmostEqual(grouped["f2"], 5 / 6)

    def test_average_precision_and_budget_do_not_split_score_ties(self):
        ranked = ranking_metrics(np.array([1, 0, 1, 0]), np.array([.9, .8, .8, .2]), [.1, .5, .75, 1.])
        self.assertAlmostEqual(ranked["average_precision"], 5 / 6)
        points = ranked["area_budgets"]
        self.assertEqual(points[0]["decision"], "select_none")
        self.assertIsNone(points[0]["threshold"])
        self.assertEqual(points[1]["flagged_pixels"], 1)  # two .8s do not fit
        self.assertEqual(points[1]["recall"], .5)
        self.assertEqual(points[2]["flagged_pixels"], 3)
        self.assertEqual(points[2]["recall"], 1.)
        self.assertEqual(points[2]["precision"], 2 / 3)
        for p in points:
            self.assertLessEqual(p["flagged_pixels"], p["max_pixels"])
        # Tie ordering must not alter AP or recall at a threshold.
        swapped = ranking_metrics(np.array([1, 1, 0, 0]), np.array([.9, .8, .8, .2]), [.5])
        self.assertEqual(swapped["average_precision"], ranked["average_precision"])

    def test_ranking_undefined_recall_and_invalid_scores(self):
        no_positives = ranking_metrics(np.zeros(3), np.array([.8, .3, .1]), [1.])
        self.assertIsNone(no_positives["average_precision"])
        self.assertIsNone(no_positives["area_budgets"][0]["recall"])
        self.assertEqual(ranking_metrics(np.array([]), np.array([]))["status"], "no_scored_pixels")
        with self.assertRaises(ValueError):
            ranking_metrics(np.array([1]), np.array([np.nan]))
        with self.assertRaises(ValueError):
            ranking_metrics(np.array([1]), np.array([.5]), [1.1])

    def test_model_full_resolution_six_logits(self):
        torch.set_num_threads(2)
        model = make_model(6, pretrained=False).eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 12, 64, 64))
        self.assertEqual(tuple(output.shape), (1, 6, 64, 64))


if __name__ == "__main__":
    unittest.main()
