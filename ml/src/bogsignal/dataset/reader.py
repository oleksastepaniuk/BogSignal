"""NumPy samples usable with PyTorch default collation; no model dependency."""
import json
import hashlib
from pathlib import Path
import numpy as np
import rasterio

class PatchDataset:

    def __init__(self, root, split='train', normalize=True):
        self.root = Path(root)
        if not (self.root / 'complete.json').exists():
            raise ValueError('Dataset incomplete')
        if split not in ['train', 'validation', 'test']:
            raise ValueError('Invalid split')
        self.records = [json.loads(r) for r in (self.root / 'index.jsonl').read_text().splitlines() if json.loads(r)['split'] == split]
        self.stats = json.loads((self.root / 'normalization.json').read_text())
        self.normalize = normalize
        self.split = split
        self._sampling_weights = np.ones(len(self.records), dtype=float)
        policy_path = self.root / 'training_sampling.json'
        if split == 'train':
            if policy_path.exists():
                policy = json.loads(policy_path.read_text())
                if policy['index_sha256'] != hashlib.sha256((self.root / 'index.jsonl').read_bytes()).hexdigest():
                    raise ValueError('Sampling policy index mismatch')
                if policy['patch_ids'] != [r['id'] for r in self.records]:
                    raise ValueError('Sampling policy patch order mismatch')
                weights = np.asarray(policy['weights'], dtype=float)
                if weights.shape != (len(self.records),) or not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1):
                    raise ValueError('Invalid sampling weights')
                self._sampling_weights = weights
            else:
                self._sampling_weights = np.array([r.get('sampling_weight', 1.) for r in self.records])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        with rasterio.open(self.root / r['images']) as d:
            image = d.read()
        with rasterio.open(self.root / r['labels']) as d:
            label = d.read(1).astype('int64')
        if self.normalize:
            mean = np.asarray(self.stats['mean'], dtype='float32')[:, None, None]
            std = np.asarray(self.stats['std'], dtype='float32')[:, None, None]
            image = (image - mean) / np.maximum(std, 1e-08)
        return dict(image=image, mask=label, valid=label != 255, id=r['id'])

    @property
    def sampling_weights(self):
        return self._sampling_weights.copy()
