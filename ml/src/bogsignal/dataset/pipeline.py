"""Plan and export fully input-valid, spatially separated segmentation patches."""
import argparse
import hashlib
import json
import shutil
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window, bounds, transform
from shapely.geometry import box, mapping
SIZE, BLOCK, BUFFER = (256, 2048, 64)
SPLITS = ['train', 'validation', 'test']
FRACTIONS = np.array([0.7, 0.15, 0.15])
MASKS = [f'inputs/{folder}/{prefix}_polissia_target-20261025_valid_mask.tif' for folder, prefix in [('sentinel-2', 'S2'), ('sentinel-1', 'S1_ASCENDING'), ('sentinel-1', 'S1_DESCENDING')]]
LABEL = 'labels/polissia_labels_20m_3035.tif'
ELIGIBLE = 'labels/polissia_training_mask_20m_3035.tif'
AOI = 'aoi/polissia_aoi_mask_20m_3035.tif'
OSM = 'references/osm/OSM_polissia_water_wetlands.geojson'

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def save(p, obj):
    t = p.with_suffix(p.suffix + '.partial')
    t.write_text(json.dumps(obj, indent=2) + '\n')
    t.replace(p)

def channels(include_nisar=False):
    result = [dict(name='S2_' + b, path=f'inputs/sentinel-2/S2_polissia_target-20261025_{b}.tif', scale=0.0001, units='reflectance') for b in ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']] + [dict(name=f'S1_{p}_{b}', path=f'inputs/sentinel-1/S1_{p}_polissia_target-20261025_{b}.tif', scale=1.0, units='linear_sigma0') for p in ['ASCENDING', 'DESCENDING'] for b in ['VV', 'VH']]
    if include_nisar:
        result += [dict(name=f'NISAR_{p}_{b}', path=f'inputs/nisar/NISAR_{p}_polissia_target-20261025_{b}.tif', scale=1.0, units='linear_gamma0') for p in ['ASCENDING', 'DESCENDING'] for b in ['HH', 'HV']]
    return result

def balancing_weights(counts):
    a = np.asarray(counts, float)
    f = a / np.maximum(a.sum(axis=1, keepdims=True), 1)
    totals = f.sum(axis=0)
    w = (f / np.where(totals > 0, totals, 1)).sum(axis=1)
    return w / w.sum()

def split_features(patches):
    counts = np.asarray([p['counts'] for p in patches], float)
    return np.r_[len(patches), (counts >= np.ceil(0.05 * SIZE * SIZE)).sum(axis=0), counts.sum(axis=0)]

def assign_groups(groups, seed=2026):
    order = sorted(groups)
    np.random.default_rng(seed).shuffle(order)
    stats = {g: split_features(groups[g]) for g in groups}
    totals = np.sum(list(stats.values()), axis=0)
    order.sort(key=lambda k: -np.sum((stats[k] / np.maximum(totals, 1)) ** 2))
    target = FRACTIONS[:, None] * totals
    den = np.maximum(totals, 1)
    used = np.zeros((3, 13))
    result = {}
    for g in order:
        v = stats[g]
        delta = (((used + v - target) / den) ** 2 - ((used - target) / den) ** 2).sum(axis=1)
        s = int(np.argmin(delta))
        used[s] += v
        result[g] = SPLITS[s]
    return result

def allowed_buffer(row, col, blocks):
    own = blocks[row // BLOCK, col // BLOCK]
    return all((blocks.get((r, c), own) == own for r in range(max(0, (row - BUFFER) // BLOCK), (row + SIZE + BUFFER - 1) // BLOCK + 1) for c in range(max(0, (col - BUFFER) // BLOCK), (col + SIZE + BUFFER - 1) // BLOCK + 1)))

def plan(root, out, include_nisar=False):
    if (out / 'plan.json').exists():
        raise FileExistsError('Plan exists: choose a new output directory')
    channel_list = channels(include_nisar)
    masks = MASKS + ([f'inputs/nisar/NISAR_{p}_polissia_target-20261025_valid_mask.tif' for p in ['ASCENDING', 'DESCENDING']] if include_nisar else [])
    paths = sorted({c['path'] for c in channel_list} | set(masks) | {LABEL, ELIGIBLE, AOI, OSM})
    hashes = {p: sha(root / p) for p in paths}
    candidates = []
    rejected = Counter()
    with ExitStack() as stack:
        datasets = {p: stack.enter_context(rasterio.open(root / p)) for p in paths if p != OSM}
        ref = datasets[AOI]
        affine = ref.transform
        for d in datasets.values():
            if (d.crs, d.transform, d.shape) != (ref.crs, ref.transform, ref.shape):
                raise ValueError('Grid mismatch')
        grid = dict(crs=str(ref.crs), transform=list(affine), width=ref.width, height=ref.height)
        for row in range(0, ref.height, 1024):
            for col in range(0, ref.width, 1024):
                w = Window(col, row, min(1024, ref.width - col), min(1024, ref.height - row))
                valid = datasets[AOI].read(1, window=w) == 1
                for p in masks:
                    valid &= datasets[p].read(1, window=w) == 1
                if not valid.any():
                    continue
                for c in channel_list:
                    a = datasets[c['path']].read(1, window=w, masked=True)
                    valid &= ~np.ma.getmaskarray(a) & np.isfinite(a.data)
                y = datasets[LABEL].read(1, window=w)
                eligible = datasets[ELIGIBLE].read(1, window=w) == 1
                if np.any(eligible & ~np.isin(y, range(6))):
                    raise ValueError('Invalid eligible class')
                for r in range(0, int(w.height) - SIZE + 1, SIZE):
                    for c in range(0, int(w.width) - SIZE + 1, SIZE):
                        sl = np.s_[r:r + SIZE, c:c + SIZE]
                        if not valid[sl].all():
                            rejected['invalid_input'] += 1
                            continue
                        if eligible[sl].mean() < 0.95:
                            rejected['label_fraction_below_95pct'] += 1
                            continue
                        candidates.append(dict(row=row + r, col=col + c, block=[(row + r) // BLOCK, (col + c) // BLOCK], counts=np.bincount(y[sl][eligible[sl]], minlength=6).tolist()))
    print(f'{len(candidates)} input-valid candidates scanned', flush=True)
    active = {tuple(p['block']) for p in candidates}
    parent = {b: b for b in active}

    def find(b):
        while parent[b] != b:
            parent[b] = parent[parent[b]]
            b = parent[b]
        return b
    osm = gpd.read_file(root / OSM).to_crs(grid['crs'])
    for geom in osm[osm.osm_category.isin(['wetland', 'water_and_wetland'])].geometry:
        hits = [b for b in sorted(active) if geom.intersects(box(*bounds(Window(b[1] * BLOCK, b[0] * BLOCK, BLOCK, BLOCK), affine)))]
        for b in hits[1:]:
            parent[find(b)] = find(hits[0])
    groups = {}
    for p in candidates:
        groups.setdefault(find(tuple(p['block'])), []).append(p)
    best = None
    group_keys = sorted(groups)
    group_index = {g: i for i, g in enumerate(group_keys)}
    owners = np.array([group_index[find(tuple(p['block']))] for p in candidates])
    neighbors = []
    for p, owner in zip(candidates, owners):
        row, col = (p['row'], p['col'])
        values = []
        for br in range(max(0, (row - BUFFER) // BLOCK), (row + SIZE + BUFFER - 1) // BLOCK + 1):
            for bc in range(max(0, (col - BUFFER) // BLOCK), (col + SIZE + BUFFER - 1) // BLOCK + 1):
                values.append(group_index[find((br, bc))] if (br, bc) in active else owner)
        neighbors.append(values + [owner] * (4 - len(values)))
    neighbors = np.asarray(neighbors)
    counts = np.array([p['counts'] for p in candidates], float)
    patch_features = np.column_stack([np.ones(len(candidates)), counts >= np.ceil(0.05 * SIZE * SIZE), counts])

    def evaluate(assignment):
        folds = assignment[owners]
        retained = (assignment[neighbors] == folds[:, None]).all(axis=1)
        features = np.array([[patch_features[retained & (folds == s), c].sum() for c in range(13)] for s in range(3)])
        if np.any(features[:, 1:7] == 0):
            return float('inf')
        shares = features / np.maximum(features.sum(axis=0), 1)
        return float(((shares - FRACTIONS[:, None]) ** 2 * np.r_[1, np.full(6, 2), np.ones(6)]).mean() + 0.0001 * (1 - retained.mean()))
    for seed in range(2026, 2154):
        assignment = assign_groups(groups, seed)
        vector = np.array([SPLITS.index(assignment[g]) for g in group_keys])
        score = evaluate(vector)
        if np.isfinite(score) and (best is None or score < best[0]):
            best = (score, seed, vector)
    if best is None:
        raise ValueError('No split with substantial presence of every class in all folds')
    score, chosen_seed, vector = best
    for iteration in range(30):
        improved = False
        for group in np.random.default_rng(chosen_seed + iteration).permutation(len(group_keys)):
            original = int(vector[group])
            choice = original
            local_score = score
            for fold in range(3):
                if fold == original:
                    continue
                vector[group] = fold
                trial_score = evaluate(vector)
                if trial_score < local_score - 1e-10:
                    local_score = trial_score
                    choice = fold
            vector[group] = choice
            if choice != original:
                score = local_score
                improved = True
        if not improved:
            break
    blocks = {b: SPLITS[vector[group_index[find(b)]]] for b in active}
    selected = []
    for p in candidates:
        if not allowed_buffer(p['row'], p['col'], blocks):
            rejected['cross_split_buffer'] += 1
            continue
        p.update(split=blocks[tuple(p['block'])], id=f"r{p['row']:06d}_c{p['col']:06d}")
        selected.append(p)
    if any((not any((p['split'] == s for p in selected)) for s in SPLITS)):
        raise ValueError('Empty spatial split')
    train = [p for p in selected if p['split'] == 'train']
    if np.any(np.sum([p['counts'] for p in train], axis=0) == 0):
        raise ValueError('Missing training class')
    for p, w in zip(train, balancing_weights([p['counts'] for p in train])):
        p['sampling_weight'] = float(w)
    summary = {s: dict(patches=sum((p['split'] == s for p in selected)), class_pixels=np.sum([p['counts'] for p in selected if p['split'] == s], axis=0).tolist()) for s in SPLITS}
    f = np.array([p['counts'] for p in train], float)
    f /= f.sum(axis=1, keepdims=True)
    summary['train']['weighted_expected_class_fractions'] = (f * np.array([p['sampling_weight'] for p in train])[:, None]).sum(axis=0).tolist()
    for s in SPLITS:
        summary[s]['substantial_class_patches'] = (np.array([p['counts'] for p in selected if p['split'] == s]) >= np.ceil(0.05 * SIZE * SIZE)).sum(axis=0).tolist()
    totals = np.sum([summary[s]['substantial_class_patches'] for s in SPLITS], axis=0)
    for s in SPLITS:
        summary[s]['per_class_presence_split_shares'] = (np.array(summary[s]['substantial_class_patches']) / totals).tolist()
    result = dict(version=2, channels=channel_list, valid_masks=masks, grid=grid, input_sha256=hashes, patch_size=SIZE, block_size=BLOCK, buffer=BUFFER, seed=chosen_seed, seed_search_count=128, split_score=score, class_presence_threshold=0.05, split_targets=FRACTIONS.tolist(), min_label_fraction=0.95, patches=selected, summary=summary, rejections=dict(rejected))
    out.mkdir(parents=True, exist_ok=True)
    save(out / 'plan.json', result)
    save(out / 'candidates.json', candidates)
    save(out / 'splits.geojson', dict(type='FeatureCollection', crs=dict(type='name', properties=dict(name=grid['crs'])), features=[dict(type='Feature', properties=dict(split=blocks[b], group=str(find(b)), block=list(b)), geometry=mapping(box(*bounds(Window(b[1] * BLOCK, b[0] * BLOCK, BLOCK, BLOCK), affine)))) for b in sorted(active)]))
    print(json.dumps(summary, indent=2), flush=True)

def add_nisar(root, out, base):
    if (out / 'plan.json').exists():
        raise FileExistsError('Plan exists')
    plan = json.loads(base.read_text())
    for p, digest in plan['input_sha256'].items():
        if sha(root / p) != digest:
            raise ValueError(f'Base input changed: {p}')
    extra = [dict(name=f'NISAR_{p}_{b}', path=f'inputs/nisar/NISAR_{p}_polissia_target-20261025_{b}.tif', scale=1.0, units='linear_gamma0') for p in ['ASCENDING', 'DESCENDING'] for b in ['HH', 'HV']]
    masks = [f'inputs/nisar/NISAR_{p}_polissia_target-20261025_valid_mask.tif' for p in ['ASCENDING', 'DESCENDING']]
    for path in [c['path'] for c in extra] + masks:
        plan['input_sha256'][path] = sha(root / path)
    kept = []
    with ExitStack() as stack:
        bands = [stack.enter_context(rasterio.open(root / c['path'])) for c in extra]
        valid_masks = [stack.enter_context(rasterio.open(root / p)) for p in masks]
        for d in bands + valid_masks:
            if str(d.crs) != plan['grid']['crs'] or list(d.transform) != plan['grid']['transform'] or [d.height, d.width] != [plan['grid']['height'], plan['grid']['width']]:
                raise ValueError('NISAR grid mismatch')
        for p in plan['patches']:
            w = Window(p['col'], p['row'], SIZE, SIZE)
            if not all(((d.read(1, window=w) == 1).all() for d in valid_masks)):
                continue
            valid = True
            for d in bands:
                a = d.read(1, window=w, masked=True)
                if np.ma.getmaskarray(a).any() or not np.isfinite(a.data).all():
                    valid = False
                    break
            if valid:
                kept.append(p)
    if any((not any((p['split'] == s for p in kept)) for s in SPLITS)):
        raise ValueError('NISAR filtering emptied a split')
    train = [p for p in kept if p['split'] == 'train']
    if np.any(np.sum([p['counts'] for p in train], axis=0) == 0):
        raise ValueError('Missing training class after NISAR filter')
    for p, w in zip(train, balancing_weights([p['counts'] for p in train])):
        p['sampling_weight'] = float(w)
    plan['rejections']['nisar_invalid_patch'] = len(plan['patches']) - len(kept)
    plan.update(patches=kept, channels=plan['channels'] + extra, valid_masks=MASKS + masks, base_plan_sha256=sha(base), split_geography='unchanged from approved Sentinel-only plan')
    plan['summary'] = {s: dict(patches=sum((p['split'] == s for p in kept)), class_pixels=np.sum([p['counts'] for p in kept if p['split'] == s], axis=0).tolist()) for s in SPLITS}
    f = np.array([p['counts'] for p in train], float)
    f /= f.sum(axis=1, keepdims=True)
    plan['summary']['train']['weighted_expected_class_fractions'] = (f * np.array([p['sampling_weight'] for p in train])[:, None]).sum(axis=0).tolist()
    out.mkdir(parents=True, exist_ok=True)
    save(out / 'plan.json', plan)
    shutil.copy2(base.parent / 'splits.geojson', out / 'splits.geojson')
    print(json.dumps(plan['summary'], indent=2), flush=True)

def export(root, out):
    plan = json.loads((out / 'plan.json').read_text())
    for p, digest in plan['input_sha256'].items():
        if sha(root / p) != digest:
            raise ValueError(f'Changed input: {p}')
    if (out / 'complete.json').exists():
        raise FileExistsError('Completed dataset exists')
    records = []
    nc = len(plan['channels'])
    sums = np.zeros(nc)
    squares = np.zeros(nc)
    n = 0
    with ExitStack() as stack:
        bands = [stack.enter_context(rasterio.open(root / c['path'])) for c in plan['channels']]
        label = stack.enter_context(rasterio.open(root / LABEL))
        eligible = stack.enter_context(rasterio.open(root / ELIGIBLE))
        masks = [stack.enter_context(rasterio.open(root / p)) for p in plan.get('valid_masks', MASKS)]
        for i, p in enumerate(plan['patches']):
            w = Window(p['col'], p['row'], SIZE, SIZE)
            layers = []
            for d, c in zip(bands, plan['channels']):
                a = d.read(1, window=w, masked=True)
                if np.ma.getmaskarray(a).any() or not np.isfinite(a.data).all():
                    raise ValueError('Invalid input patch')
                layers.append(a.data.astype('float32') * np.float32(c['scale']))
            if not all(((d.read(1, window=w) == 1).all() for d in masks)):
                raise ValueError('Invalid sensor mask')
            x = np.stack(layers)
            y = np.where(eligible.read(1, window=w) == 1, label.read(1, window=w), 255).astype('uint8')
            assert np.isin(y, [0, 1, 2, 3, 4, 5, 255]).all()
            profile = dict(driver='GTiff', width=SIZE, height=SIZE, crs=label.crs, transform=transform(w, label.transform), compress='deflate', tiled=True, blockxsize=128, blockysize=128)
            paths = {}
            for kind, array in [('images', x), ('labels', y[None])]:
                dest = out / p['split'] / kind / (p['id'] + '.tif')
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    tmp = dest.with_suffix('.partial.tif')
                    with rasterio.open(tmp, 'w', **profile, count=array.shape[0], dtype=str(array.dtype), nodata=255 if kind == 'labels' else None) as d:
                        d.write(array)
                        if kind == 'images':
                            d.descriptions = tuple((c['name'] for c in plan['channels']))
                            for b, c in enumerate(plan['channels'], 1):
                                d.update_tags(b, units=c['units'], source_scale_applied=c['scale'])
                    tmp.rename(dest)
                with rasterio.open(dest) as d:
                    if not np.array_equal(d.read(), array) or d.transform != profile['transform']:
                        raise ValueError(f'Patch mismatch: {dest}')
                paths[kind] = str(dest.relative_to(out))
            if p['split'] == 'train':
                values = x.astype('float64').reshape(nc, -1)
                sums += values.sum(axis=1)
                squares += (values * values).sum(axis=1)
                n += values.shape[1]
            records.append(dict(p, **paths))
            if i % 250 == 0:
                print(f"Exported {i + 1}/{len(plan['patches'])}", flush=True)
    mean = sums / n
    std = np.sqrt(np.maximum(squares / n - mean * mean, 0))
    save(out / 'normalization.json', dict(channels=[c['name'] for c in plan['channels']], mean=mean.tolist(), std=std.tolist(), pixels=n, scope='training patches only; physical units'))
    with (out / 'index.jsonl').open('w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    for p, digest in plan['input_sha256'].items():
        if sha(root / p) != digest:
            raise ValueError(f'Input changed during build: {p}')
    save(out / 'complete.json', dict(patches=len(records), summary=plan['summary'], plan_sha256=sha(out / 'plan.json'), validation='every exported image and label pixel read back; every selected sensor pixel valid; source hashes unchanged'))
    print('Dataset complete', flush=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['plan', 'add-nisar', 'build'])
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base-plan', type=Path)
    p.add_argument('--include-nisar', action='store_true')
    a = p.parse_args()
    root = a.data_root.resolve()
    out = a.output.resolve()
    if not out.is_relative_to(root / 'dataset') or out == root / 'dataset':
        p.error('Output must be a named directory under data/dataset')
    if a.action == 'add-nisar':
        if a.base_plan is None:
            p.error('--base-plan is required for add-nisar')
        add_nisar(root, out, a.base_plan.resolve())
    elif a.action == 'plan':
        plan(root, out, a.include_nisar)
    else:
        export(root, out)
if __name__ == '__main__':
    main()
