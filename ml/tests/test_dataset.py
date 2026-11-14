import unittest
import numpy as np
import json
import tempfile
from pathlib import Path
import rasterio
from rasterio.transform import from_origin
from bogsignal.dataset.pipeline import balancing_weights, allowed_buffer, assign_groups
from bogsignal.dataset.pipeline import export, channels, MASKS, LABEL, ELIGIBLE, sha
from bogsignal.dataset.reader import PatchDataset
from bogsignal.dataset.pipeline import split_features
from bogsignal.dataset.sampling import mixture_weights, write_policy

class DatasetTests(unittest.TestCase):

    def test_sampling_pool_probabilities(self):
        # Use six-class rows with nonzero totals and each substantial minority class.
        counts = [[65536, 0, 0, 0, 0, 0], [65000, 536, 0, 0, 0, 0]] + [[60000] + [5536 if j == c else 0 for j in range(5)] for c in range(5)]
        w = mixture_weights(counts)
        self.assertAlmostEqual(w[:2].sum(), .25)
        self.assertAlmostEqual(w[2:].sum(), .75)
        self.assertTrue(np.allclose(w[2:], .15))

    def test_sampling_missing_pool_rejected(self):
        with self.assertRaisesRegex(ValueError, 'both'):
            mixture_weights([[65536, 0, 0, 0, 0, 0]])

    def test_reader_applies_policy_only_to_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            counts = [[65536, 0, 0, 0, 0, 0]] + [[60000] + [5536 if j == c else 0 for j in range(5)] for c in range(5)]
            records = [dict(id=str(i), split='train', counts=c) for i, c in enumerate(counts)]
            records.append(dict(id='validation', split='validation', counts=counts[0]))
            (root / 'index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            (root / 'complete.json').write_text('{}')
            (root / 'plan.json').write_text(json.dumps(dict(patches=records, summary={})))
            (root / 'normalization.json').write_text(json.dumps(dict(mean=[0], std=[1])))
            write_policy(root)
            policy = json.loads((root / 'training_sampling.json').read_text())
            updated_plan = json.loads((root / 'plan.json').read_text())
            updated_index = [json.loads(line) for line in (root / 'index.jsonl').read_text().splitlines()]
            self.assertEqual(policy['weights'], [r['sampling_weight'] for r in updated_plan['patches'] if r['split'] == 'train'])
            self.assertEqual(policy['weights'], [r['sampling_weight'] for r in updated_index if r['split'] == 'train'])
            self.assertEqual(json.loads((root / 'complete.json').read_text())['plan_sha256'], sha(root / 'plan.json'))
            self.assertTrue((root / 'metadata_history/before_sampling_sync/plan.json').exists())
            self.assertAlmostEqual(PatchDataset(root).sampling_weights[0], .25)
            self.assertTrue(np.array_equal(PatchDataset(root, 'validation').sampling_weights, [1.]))
            records[0]['note'] = 'changed metadata'
            (root / 'index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            with self.assertRaisesRegex(ValueError, 'index mismatch'):
                PatchDataset(root)

    def test_class_presence_threshold(self):
        features = split_features([dict(counts=[3276, 3277, 0, 0, 0, 0])])
        self.assertEqual(features[1:7].tolist(), [0, 1, 0, 0, 0, 0])

    def test_nisar_channel_definition(self):
        c = channels(True)
        self.assertEqual(len(c), 14)
        self.assertEqual([x['name'] for x in c[-4:]], ['NISAR_ASCENDING_HH', 'NISAR_ASCENDING_HV', 'NISAR_DESCENDING_HH', 'NISAR_DESCENDING_HV'])
        self.assertTrue(all((x['units'] == 'linear_gamma0' and x['scale'] == 1 for x in c[-4:])))

    def test_balance(self):
        weights = balancing_weights([[100, 0]] * 9 + [[0, 100]])
        self.assertAlmostEqual(weights[-1], 0.5)
        self.assertAlmostEqual(weights.sum(), 1)

    def test_buffer(self):
        b = {(0, 0): 'train', (0, 1): 'test'}
        self.assertFalse(allowed_buffer(512, 1792, b))
        self.assertFalse(allowed_buffer(512, 2048, b))
        self.assertTrue(allowed_buffer(512, 1536, b))

    def test_same_split(self):
        self.assertTrue(allowed_buffer(512, 1792, {(0, 0): 'train', (0, 1): 'train'}))

    def test_groups(self):
        groups = {i: [dict(counts=[100] * 6)] for i in range(30)}
        a = assign_groups(groups)
        self.assertEqual(a, assign_groups(groups))
        self.assertEqual(set(a.values()), {'train', 'validation', 'test'})

    def fixture(self, root, invalid=False):
        paths = {c['path'] for c in channels()} | set(MASKS) | {LABEL, ELIGIBLE}
        for path in paths:
            p = root / path
            p.parent.mkdir(parents=True, exist_ok=True)
            value = 10000 if 'sentinel-2/' in path and path not in MASKS else 1
            a = np.full((256, 256), value, dtype='uint16')
            if path == LABEL:
                a[:] = 0
                a[0, 0] = 6
            if path == ELIGIBLE:
                a[0, 0] = 0
            if invalid and path == MASKS[0]:
                a[10, 10] = 0
            with rasterio.open(p, 'w', driver='GTiff', width=256, height=256, count=1, dtype='uint16', crs='EPSG:3035', transform=from_origin(0, 5120, 20, 20), nodata=65535) as d:
                d.write(a, 1)
        out = root / 'dataset/test'
        out.mkdir(parents=True)
        plan = dict(input_sha256={p: sha(root / p) for p in paths}, channels=channels(), patches=[dict(id='p', row=0, col=0, split='train', counts=[65535, 0, 0, 0, 0, 0], sampling_weight=1)], summary={})
        (out / 'plan.json').write_text(json.dumps(plan))
        return out

    def test_physical_export_ignore_and_reader(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out = self.fixture(root)
            export(root, out)
            sample = PatchDataset(out, normalize=False)[0]
            self.assertEqual(sample['image'].shape, (10, 256, 256))
            self.assertTrue(np.all(sample['image'] == 1))
            self.assertEqual(sample['mask'][0, 0], 255)
            self.assertEqual(sample['mask'][1, 1], 0)
            self.assertEqual(PatchDataset(out).stats['pixels'], 65536)

    def test_invalid_mask_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out = self.fixture(root, True)
            with self.assertRaisesRegex(ValueError, 'sensor mask'):
                export(root, out)

    def test_valid_radar_zero_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out = self.fixture(root)
            channel = channels()[6]
            with rasterio.open(root / channel['path'], 'r+') as dataset:
                dataset.write(np.zeros((256, 256), dtype='uint16'), 1)
            plan = json.loads((out / 'plan.json').read_text())
            plan['input_sha256'][channel['path']] = sha(root / channel['path'])
            (out / 'plan.json').write_text(json.dumps(plan))
            export(root, out)
            sample = PatchDataset(out, normalize=False)[0]
            self.assertTrue((sample['image'][6] == 0).all())
            self.assertTrue(sample['valid'][1, 1])

    def test_changed_source_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out = self.fixture(root)
            with rasterio.open(root / channels()[0]['path'], 'r+') as dataset:
                dataset.write(np.zeros((256, 256), dtype='uint16'), 1)
            with self.assertRaisesRegex(ValueError, 'Changed input'):
                export(root, out)
if __name__ == '__main__':
    unittest.main()
