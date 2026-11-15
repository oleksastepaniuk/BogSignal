"""Partial-channel coverage, export, normalization and reader regression tests."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import rasterio

import test_dataset
from bogsignal.dataset.pipeline import (patch_coverage, channel_masks, channels,
                                       export, MASKS, sha)
from bogsignal.dataset.reader import PatchDataset


class MissingInputTests(unittest.TestCase):
    def fixture(self, root, missing):
        out = test_dataset.DatasetTests().fixture(root)
        path = root / MASKS[1]
        with rasterio.open(path, 'r+') as d:
            a = d.read(1)
            a.flat[:missing] = 0
            d.write(a, 1)
        plan = json.loads((out/'plan.json').read_text())
        plan.update(max_missing_fraction=.05, valid_masks=MASKS,
                    channel_valid_masks=channel_masks(channels(), MASKS))
        plan['input_sha256'][MASKS[1]] = sha(path)
        # Put a bright value exclusively inside the invalid area. It must not
        # contaminate the per-channel normalization, despite being numerically finite.
        band = root / channels()[6]['path']
        with rasterio.open(band,'r+') as d:
            a=d.read(1); a.flat[:missing]=60000; d.write(a,1)
        plan['input_sha256'][channels()[6]['path']] = sha(band)
        (out/'plan.json').write_text(json.dumps(plan))
        return out

    def test_exact_five_percent_boundary_per_channel(self):
        valid = np.ones((14,256,256),bool)
        valid[13].flat[:3276]=False
        self.assertTrue(patch_coverage(valid,.05))
        valid[13].flat[3276]=False
        self.assertFalse(patch_coverage(valid,.05))
        # An average over channels would incorrectly accept this patch.
        self.assertLess((~valid).mean(),.05)

    def test_partial_export_masks_statistics_and_model_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); out=self.fixture(root,3276)
            export(root,out)
            r=json.loads((out/'index.jsonl').read_text())
            stats=json.loads((out/'normalization.json').read_text())
            self.assertEqual(stats['valid_pixels_per_channel'][6:8],[65536-3276]*2)
            self.assertEqual(stats['mean'][6:8],[1.,1.])
            self.assertEqual(stats['std'][6:8],[0.,0.])
            with rasterio.open(out/r['images']) as d:
                self.assertEqual(d.nodata,-9999)
                self.assertTrue(d.read(7,masked=True).mask.flat[:3276].all())
            dataset=PatchDataset(out)
            sample=dataset[0]
            self.assertEqual(dataset.model_channels,20)
            self.assertEqual(sample['image'].shape,(20,256,256))
            np.testing.assert_array_equal(sample['input_valid'][6], sample['input_valid'][7])
            self.assertFalse(sample['input_valid'][6,1,1])
            self.assertEqual(sample['image'][6,1,1],0)
            self.assertEqual(sample['image'][16,1,1],0)
            self.assertTrue(sample['valid'][1,1])  # Known label remains trainable.
            self.assertFalse(sample['evaluation_valid'][1,1])
            self.assertTrue(sample['evaluation_valid'][100,100])
            self.assertEqual(PatchDataset(out,append_validity=False)[0]['image'].shape[0],10)

    def test_over_limit_export_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); out=self.fixture(root,3277)
            with self.assertRaisesRegex(ValueError,'missing-channel limit'):
                export(root,out)
            self.assertFalse((out/'complete.json').exists())

    def test_validation_values_excluded_from_normalization(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); out=self.fixture(root,1)
            plan=json.loads((out/'plan.json').read_text())
            row=dict(plan['patches'][0],id='v',split='validation')
            plan['patches'].append(row)
            (out/'plan.json').write_text(json.dumps(plan))
            export(root,out)
            stats=json.loads((out/'normalization.json').read_text())
            self.assertEqual(stats['pixels'],65536)
            self.assertEqual(stats['valid_pixels_per_channel'][6],65535)
