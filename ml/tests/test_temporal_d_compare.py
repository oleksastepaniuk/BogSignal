import unittest
import numpy as np
from bogsignal.training.temporal_d_compare import score_aligned

class TemporalDComparisonTests(unittest.TestCase):
    def fixture(self):
        old = np.array([[[0,1],[2,3]], [[4,5],[1,2]]], dtype='uint8')
        new = old[[1,0]].copy()
        new[0,0,0] = 255
        ov = np.ones_like(old, dtype='uint8')
        nv = np.ones_like(new, dtype='uint8'); nv[0,0,0] = 0
        mapping = np.array([0,1,1,2,3,4],dtype='uint8')
        op = [old.copy(), mapping[old]]
        npred = [old[[1,0]].copy(), mapping[old[[1,0]]]]
        return [['a','b'],['b','a'],old,new,ov,nv,op,npred]

    def test_aligns_ids_and_ignores_only_declared_pixels(self):
        scores = score_aligned(*self.fixture())
        self.assertEqual(scores[0],scores[1])
        self.assertEqual(scores[0]['validation']['scored_pixels'],7)
        self.assertEqual(scores[0]['combined_class']['accuracy'],1.)

    def test_rejects_reference_changes(self):
        args = self.fixture(); args[3][0,1,0] = 0
        with self.assertRaisesRegex(ValueError,'changed labels'):
            score_aligned(*args)

    def test_rejects_pixels_invalid_in_baseline(self):
        args = self.fixture(); args[4][1,1,0] = 0
        with self.assertRaisesRegex(ValueError,'invalid in D'):
            score_aligned(*args)

    def test_rejects_missing_patch(self):
        args = self.fixture(); args[1][0] = 'missing'
        with self.assertRaisesRegex(ValueError,'missing from D'):
            score_aligned(*args)
