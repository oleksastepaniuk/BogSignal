"""Offline tests for the InSAR pair rules and phase measures (no downloads)."""
import unittest

import numpy as np
import pandas as pd
from shapely.geometry import box

from bogsignal import insar_pairs
from bogsignal.insar_flooded import flood_regime, paper_iteration, phase_measures, probability


def scene(day, **over):
    site = "Cheremske"
    base = {"site": site, "granule": f"g{day}", "start_utc": f"2026-07-{day:02d}T03:00:00Z", "track": 100, "frame": 28, "direction": "ASCENDING",
            "radar_band": "L", "polarizations": "HH,HV", "range_bandwidth": '["20+5"]', "beam_mode": "x", "collection": "C1", "url": f"u{day}",
            "footprint_wkt": box(20, 50, 30, 53).wkt}
    return {**base, **over}


class PairRules(unittest.TestCase):
    def pairs(self, rows):
        return insar_pairs.make_pairs(pd.DataFrame(rows), {}, tolerance=2, min_overlap=0.95)

    def test_temporal_tiers_and_tolerance(self):
        ok, rejected = self.pairs([scene(1), scene(13), scene(25), scene(8)])
        tiers = {(r.reference, r.secondary): r.temporal_tier for r in ok.itertuples()}
        self.assertEqual(tiers[("g1", "g13")], "preferred")
        self.assertEqual(tiers[("g1", "g25")], "secondary")
        self.assertIn("temporal separation", rejected[rejected.secondary == "g8"].reason.iloc[0])

    def test_missing_metadata_is_not_a_match(self):
        ok, _ = self.pairs([scene(1, beam_mode=None), scene(13, beam_mode=None)])
        self.assertEqual(ok.status.iloc[0], "needs_verification")
        self.assertIn("beam_mode", ok.unverified_fields.iloc[0])

    def test_incompatible_processing_and_polarization_rejected(self):
        ok, rejected = self.pairs([scene(1), scene(13, collection="C2")])
        self.assertEqual(len(ok), 0)
        self.assertIn("collection differs", rejected.reason.iloc[0])
        ok, rejected = self.pairs([scene(1), scene(13, polarizations="VV")])
        self.assertIn("no HH", rejected.reason.iloc[0])

    def test_partial_coverage_rejected(self):
        ok, rejected = self.pairs([scene(1), scene(13, footprint_wkt=box(25.55, 50, 30, 53).wkt)])
        self.assertEqual(len(ok), 0)
        self.assertIn("common AOI coverage", rejected.reason.iloc[0])


class PhaseMeasures(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.valid = np.ones((120, 120), bool)

    def test_constant_phase_is_homogeneous_without_fringe(self):
        ifg = np.full((120, 120), np.exp(1j * 3.1), np.complex64)  # next to the -pi/+pi boundary
        h, f = phase_measures(ifg, self.valid, 5, 41)
        self.assertGreater(h.min(), 0.999)
        self.assertLess(f.max(), 1e-3)

    def test_random_phase_is_not_homogeneous(self):
        ifg = np.exp(1j * self.rng.uniform(-np.pi, np.pi, (120, 120))).astype(np.complex64)
        h, _ = phase_measures(ifg, self.valid, 5, 41)
        self.assertLess(h.mean(), 0.35)  # about 1/sqrt(25) for independent samples

    def test_local_phase_offset_is_detected_across_the_wrap(self):
        phase = np.full((120, 120), 3.0)
        phase[50:70, 50:70] = -2.5  # 0.78 rad away through the wrap, not 5.5 rad
        ifg = np.exp(1j * phase).astype(np.complex64)
        _, f = phase_measures(ifg, self.valid, 5, 81)
        self.assertAlmostEqual(float(f[60, 60]), 2 * np.pi - 5.5, delta=0.15)
        self.assertLess(float(f[10, 10]), 0.1)

    def test_invalid_pixels_stay_invalid(self):
        ifg = np.exp(1j * self.rng.uniform(-1, 1, (120, 120))).astype(np.complex64)
        valid = self.valid.copy()
        valid[:, :60] = False
        prob, _ = probability(ifg, valid, 3, self.rng, 80.0)
        self.assertTrue(np.isnan(prob[:, :60]).all())
        self.assertTrue(np.isfinite(prob[:, 60:]).all())


class PaperWorkflow(unittest.TestCase):
    """Hübinger et al. (2026): fringe turnovers joined to connected homogeneous-phase groups."""

    def test_only_homogeneous_groups_touching_a_fringe_are_flagged(self):
        rng = np.random.default_rng(0)
        phase = rng.uniform(-np.pi, np.pi, (120, 300)).astype(np.float32)  # decorrelated background
        ramp = np.angle(np.exp(1j * np.linspace(1.5, 2 * np.pi + 1.5, 80)))  # smooth water-level gradient with one -pi/pi turnover
        phase[20:100, 20:100] = ramp[None, :]
        phase[20:100, 200:280] = 0.7  # coherent but no water-level change: no fringe
        valid = np.ones(phase.shape, bool)
        flagged = paper_iteration(phase, valid, sigma=1.5, low=3.0, high=6.7, window=5, var_thr=0.2, kernel=3)
        self.assertGreater(flagged[30:90, 30:90].mean(), 0.9)
        self.assertLess(flagged[30:90, 210:270].mean(), 0.05)
        self.assertLess(flagged[30:90, 120:180].mean(), 0.05)

    def test_regime_from_consecutive_pairs(self):
        m = lambda *v: np.array(v, bool)[None]  # noqa: E731
        dates = [("d0", "d1"), ("d1", "d2"), ("d2", "d3")]
        by_date, regime = flood_regime([m(1, 1, 0, 0), m(1, 0, 0, 0), m(1, 0, 0, 1)], dates, m(1, 1, 1, 0))
        self.assertEqual(list(by_date), ["d1", "d2"])
        self.assertEqual(regime.tolist(), [[2, 1, 0, 255]])


if __name__ == "__main__":
    unittest.main()
