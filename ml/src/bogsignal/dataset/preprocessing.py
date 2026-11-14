"""Prepare versioned radar inputs without selecting patches or fitting normalization.

Run with --help. Original inputs and completed dataset versions are never changed.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path

import h5py
import numpy as np
import rasterio
from rasterio.windows import Window

from .pipeline import channels, sha, save

TARGET = datetime(2026, 10, 25, 17, 34, 25, tzinfo=timezone.utc)
NODATA = -9999.0
DEFAULT_CLIP = {'co': [-40.0, 5.0], 'cross': [-45.0, 0.0]}
IDENTITY = ['granuleId', 'orbitPassDirection', 'trackNumber', 'frameNumber',
            'zeroDopplerStartTime', 'zeroDopplerEndTime', 'compositeReleaseId',
            'productVersion', 'productSpecificationVersion', 'productType', 'radarBand']


def text_scalar(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return text_scalar(value.item())
    return value


def timestamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def power_to_db(power, source_valid, clip, epsilon=1e-12):
    """Zero/negative/nonfinite power is invalid, never converted into a dark pixel."""
    if not (0 < epsilon < np.inf) or len(clip) != 2 or not np.isfinite(clip).all() or clip[0] >= clip[1]:
        raise ValueError('Invalid epsilon or clipping limits')
    valid = source_valid & np.isfinite(power) & (power > 0)
    result = np.full(power.shape, NODATA, dtype='float32')
    db = 10.0 * np.log10(np.maximum(power[valid].astype('float64'), epsilon))
    result[valid] = np.clip(db, *clip).astype('float32')
    return result, valid, dict(valid=int(valid.sum()),
                               clipped_low=int((db < clip[0]).sum()),
                               clipped_high=int((db > clip[1]).sum()),
                               floored_positive=int((power[valid] < epsilon).sum()))


def canonical_source(source):
    """Publish acquisition timestamps as UTC, including inventories with local offsets."""
    result = dict(source)
    value = source.get('acquired_utc') or source.get('acquired_start')
    if value:
        result['acquired_utc'] = timestamp(value).isoformat()
    return result

def validate_nisar_identity(source, identity):
    expected = dict(granuleId=source['granule_id'], orbitPassDirection=source['orbit_pass'],
                    trackNumber=source['relative_orbit'], frameNumber=source['frame'])
    for key, value in expected.items():
        if str(identity[key]).upper() != str(value).upper():
            raise ValueError(f"NISAR {source['id']}: {key} mismatch")
    if abs((timestamp(identity['zeroDopplerStartTime']) - timestamp(source['acquired_start'])).total_seconds()) >= 1:
        raise ValueError(f"NISAR {source['id']}: acquisition mismatch")
    if identity['productType'] != 'GCOV' or identity['radarBand'] != 'L':
        raise ValueError('Expected L-band GCOV')
    if source.get('maturity') != 'VALIDATED':
        raise ValueError('This revision requires validated NISAR products')


def nisar_sources(sources, archive_root):
    result = []
    for source in sources:
        paths = [archive_root / 'data/h5' / (source['id'] + '.h5'),
                 archive_root / 'data/dataset/raw/NISAR/source_h5' / (source['id'] + '.h5')]
        path = next((p for p in paths if p.exists()), None)
        if path is None:
            raise FileNotFoundError(f"Native archive required for identity check: {source['id']}")
        with h5py.File(path, 'r') as h:
            group = h['science/LSAR/identification']
            identity = {key: text_scalar(group[key][()]) for key in IDENTITY}
        validate_nisar_identity(source, identity)
        result.append(dict(source, native_identity=identity))
    signatures = {tuple(s['native_identity'][k] for k in
                        ['compositeReleaseId', 'productVersion', 'productSpecificationVersion']) for s in result}
    if len(signatures) != 1:
        raise ValueError(f'Mixed NISAR processing versions: {sorted(signatures)}; rebuild from consistent sources')
    return result


def source_tables(sources, sensor, direction):
    indices = [int(s['source_index']) for s in sources]
    if len(set(indices)) != len(indices) or not indices or min(indices) < 1 or max(indices) >= 65535:
        raise ValueError('Invalid or duplicate source indices')
    n = max(indices) + 1
    tables = dict(allowed=np.zeros(n, bool), day=np.zeros(n, 'int64'),
                  orbit=np.zeros(n, 'int64'), offset=np.zeros(n, 'float64'))
    for s in sources:
        key = int(s['source_index'])
        acquisition = s.get('acquired_utc') or s.get('acquired_start') or s.get('acquired')
        if acquisition is None and 'time_ms' in s:
            acquisition = datetime.fromtimestamp(s['time_ms'] / 1000, timezone.utc).isoformat()
        t = timestamp(acquisition)
        passed = s.get('orbit_pass', s.get('pass'))
        tables['allowed'][key] = sensor == 'S2' or passed == direction
        tables['day'][key] = int(t.timestamp() // 86400)
        tables['offset'][key] = (t - TARGET).total_seconds() / 86400
        tables['orbit'][key] = int(s.get('relative_orbit', 0))
        if sensor == 'S1' and (s.get('mode') != 'IW' or set(s['polarizations']) != {'VV', 'VH'}):
            raise ValueError('Unexpected Sentinel-1 mode/polarizations')
    return tables


def validate_provenance(ids, days, offsets, orbits, valid, tables):
    selected = ids[valid].astype('int64')
    if not len(selected):
        return
    if np.any(selected < 1) or np.any(selected >= len(tables['allowed'])) or not tables['allowed'][selected].all():
        raise ValueError('Unknown source index or wrong orbit pass')
    if not np.array_equal(days[valid], tables['day'][selected]):
        raise ValueError('Per-pixel acquisition date disagrees with selected source')
    if not np.allclose(offsets[valid], tables['offset'][selected], atol=2e-6, rtol=0):
        raise ValueError('Per-pixel acquisition offset disagrees with selected source')
    if orbits is not None and not np.array_equal(orbits[valid], tables['orbit'][selected]):
        raise ValueError('Per-pixel orbit disagrees with selected source')


def windows(ds, size=1024):
    for row in range(0, ds.height, size):
        for col in range(0, ds.width, size):
            yield Window(col, row, min(size, ds.width-col), min(size, ds.height-row))


def check_grid(reference, datasets):
    for ds in datasets:
        if (ds.crs, ds.transform, ds.shape) != (reference.crs, reference.transform, reference.shape):
            raise ValueError(f'Grid mismatch: {ds.name}')


def prepare(root, out, archive_root, clips=None, epsilon=1e-12):
    root, out = root.resolve(), out.resolve()
    if not out.is_relative_to(root / 'inputs') or out == root / 'inputs':
        raise ValueError('Output must be a new named directory under data/inputs')
    if out.exists():
        raise FileExistsError('Choose a new input revision; existing output is never overwritten')
    clips = clips or DEFAULT_CLIP
    # Validate configuration before creating any output.
    for limits in clips.values():
        power_to_db(np.array([1.0]), np.array([True]), limits, epsilon)
    folders = {'S1': 'sentinel-1', 'NISAR': 'nisar', 'S2': 'sentinel-2'}
    lookups = {s: json.loads((root / 'inputs' / f / 'source_lookup.json').read_text())['sources'] for s, f in folders.items()}
    lookups['NISAR'] = nisar_sources(lookups['NISAR'], archive_root)
    out.mkdir(parents=True)
    manifest = dict(schema_version=1, revision=out.name, status='building',
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    channels=channels(True), valid_masks=[], auxiliary={}, input_sha256={},
                    policy=dict(radar_transform='10*log10(max(valid positive power, epsilon))',
                                epsilon=epsilon, clip_db=clips,
                                clipping_selection='Fixed conservative engineering limits; no fitted quantiles',
                                radar_validity='existing effective sensor mask AND both source band masks AND finite strictly positive power AND valid provenance',
                                normalization='pending new geographic split; fit training pixels only',
                                split='not constructed', labels='unchanged',
                                geometry='passes separate; retain per-pixel relative track; tracks may differ spatially, never averaged',
                                s1_processing_version='not exposed by retained Earth Engine inventory; not independently verified',
                                nisar_processing_version='all native identification headers must agree'))
    # Freeze every consumed input, including the provenance rasters and mask policy.
    def freeze(path):
        key = str(path.relative_to(root))
        if key not in manifest['input_sha256']:
            manifest['input_sha256'][key] = sha(path)
        return key
    aoi_path = root / 'aoi/polissia_aoi_mask_20m_3035.tif'
    freeze(aoi_path)
    for sensor, folder in folders.items():
        freeze(root / 'inputs' / folder / 'source_lookup.json')
    for name in ['mask_policy.json', 'huge_blob_exclusions.geojson']:
        freeze(root / 'inputs/nisar' / name)
    output_hashes = {}
    for sensor, directions in [('S1', ['ASCENDING', 'DESCENDING']),
                               ('NISAR', ['ASCENDING', 'DESCENDING']), ('S2', [None])]:
        for direction in directions:
            folder = root / 'inputs' / folders[sensor]
            prefix = f'{sensor}_{direction}_polissia_target-20261025' if direction else 'S2_polissia_target-20261025'
            key = f'{sensor}_{direction}' if direction else 'S2'
            pols = ['VV', 'VH'] if sensor == 'S1' else ['HH', 'HV'] if sensor == 'NISAR' else ['B2','B3','B4','B8','B11','B12']
            fields = pols + ['valid_mask', 'source_index', 'observation_date', 'time_offset_days']
            if direction:
                fields += ['relative_orbit', 'angle' if sensor == 'S1' else 'gcov_mask']
            paths = {f: folder / f'{prefix}_{f}.tif' for f in fields}
            for path in paths.values():
                freeze(path)
            tables = source_tables(lookups[sensor], sensor, direction)
            stats = Counter()
            hist = Counter()
            band_stats = {p: Counter() for p in pols}
            with ExitStack() as stack:
                ds = {f: stack.enter_context(rasterio.open(path)) for f, path in paths.items()}
                aoi = stack.enter_context(rasterio.open(aoi_path))
                check_grid(aoi, ds.values())
                profile = ds[pols[0]].profile.copy()
                profile.update(driver='GTiff', count=1, tiled=True, blockxsize=512, blockysize=512,
                               compress='deflate', BIGTIFF='IF_SAFER', SPARSE_OK='TRUE', NUM_THREADS='2')
                manifest['grid'] = dict(crs=str(aoi.crs), transform=list(aoi.transform), width=aoi.width, height=aoi.height)
                writers = {}
                if direction:
                    for pol in pols:
                        dest = out / f'{key}_{pol}_db.tif'
                        writers[pol] = stack.enter_context(rasterio.open(dest, 'w', **dict(profile, dtype='float32', nodata=NODATA, predictor=3)))
                        writers[pol].set_band_description(1, f'{key}_{pol}')
                        writers[pol].units = ('dB',)
                        writers[pol].update_tags(UNITS='dB', SOURCE_UNITS='linear_sigma0' if sensor=='S1' else 'linear_gamma0', ORBIT_PASS=direction,
                                                TRANSFORM='10*log10(max(valid positive power, epsilon))', EPSILON=epsilon,
                                                CLIP_DB=json.dumps(clips['co' if pol in ['VV','HH'] else 'cross']), MANIFEST='manifest.json')
                        ch = next(c for c in manifest['channels'] if c['name'] == f'{key}_{pol}')
                        ch.update(path=str(dest.relative_to(root)), scale=1.0, units='db_sigma0' if sensor=='S1' else 'db_gamma0',
                                  source_units=ch['units'], preprocessing=dict(epsilon=epsilon, clip_db=clips['co' if pol in ['VV','HH'] else 'cross']))
                    mask_dest = out / f'{key}_valid_mask.tif'
                    writers['valid_mask'] = stack.enter_context(rasterio.open(mask_dest, 'w', **dict(profile, dtype='uint8', nodata=255, predictor=2)))
                    mask_path = str(mask_dest.relative_to(root))
                else:
                    mask_path = str(paths['valid_mask'].relative_to(root))
                manifest['valid_masks'].append(mask_path)
                for i, win in enumerate(windows(aoi)):
                    inside = aoi.read(1, window=win) == 1
                    if not inside.any():
                        # Explicitly fill outside-AOI pixels, including entirely empty blocks.
                        for field, writer in writers.items():
                            fill = 255 if field == 'valid_mask' else NODATA
                            writer.write(np.full((int(win.height),int(win.width)), fill, dtype=writer.dtypes[0]), 1, window=win)
                        continue
                    base = inside & (ds['valid_mask'].read(1, window=win) == 1)
                    valid = base.copy()
                    arrays = {p: ds[p].read(1, window=win, masked=True) for p in pols}
                    for p, a in arrays.items():
                        good = ~np.ma.getmaskarray(a) & np.isfinite(a.data)
                        if direction:
                            band_stats[p]['source_valid_nonpositive'] += int((base & good & (a.data <= 0)).sum())
                            good &= a.data > 0
                        valid &= good
                    extras = {f: ds[f].read(1, window=win, masked=True) for f in fields if f not in pols + ['valid_mask']}
                    for a in extras.values():
                        valid &= ~np.ma.getmaskarray(a) & np.isfinite(a.data)
                    if sensor == 'S1':
                        valid &= (extras['angle'].data >= 0) & (extras['angle'].data <= 90)
                    if sensor == 'NISAR':
                        valid &= (extras['gcov_mask'].data > 0) & (extras['gcov_mask'].data < 255)
                    validate_provenance(extras['source_index'].data, extras['observation_date'].data,
                                        extras['time_offset_days'].data,
                                        extras['relative_orbit'].data if direction else None, valid, tables)
                    # S2 is unchanged: failures must be repaired, never silently ignored.
                    if not direction and not np.array_equal(valid, base):
                        raise ValueError('S2 validity inconsistent with data/provenance')
                    ids, nums = np.unique(extras['source_index'].data[valid], return_counts=True)
                    hist.update({int(k):int(v) for k,v in zip(ids,nums)})
                    stats.update(inside=int(inside.sum()), source_valid=int(base.sum()), retained=int(valid.sum()), excluded_new=int((base & ~valid).sum()))
                    if direction:
                        mask = np.where(inside, valid.astype('uint8'), 255).astype('uint8')
                        writers['valid_mask'].write(mask, 1, window=win)
                        for p, a in arrays.items():
                            db, _, counts = power_to_db(a.data, valid, clips['co' if p in ['VV','HH'] else 'cross'], epsilon)
                            writers[p].write(db, 1, window=win)
                            band_stats[p].update(counts)
                    if i % 100 == 0:
                        print(key, i, 'blocks', stats['retained'], 'retained', flush=True)
            if not stats['retained']:
                raise ValueError(f'No retained pixels: {key}')
            selected = [canonical_source(dict(s, selected_pixels=hist[int(s['source_index'])])) for s in lookups[sensor] if hist[int(s['source_index'])]]
            manifest['auxiliary'][key] = dict(valid_mask=mask_path, source_lookup=f'{key}_source_lookup.json',
                rasters={f:str(paths[f].relative_to(root)) for f in fields if f not in pols + ['valid_mask']},
                interpretation='Intersect every auxiliary raster with this revision valid mask; source indices recover exact UTC acquisition times',
                counts=dict(stats), bands={p:dict(s) for p,s in band_stats.items()},
                acquisition_range_utc=[min(s.get('acquired_utc',s.get('acquired_start','')) for s in selected),
                                       max(s.get('acquired_utc',s.get('acquired_start','')) for s in selected)])
            save(out / f'{key}_source_lookup.json', dict(sources=selected))
            output_hashes[f'{key}_source_lookup.json'] = sha(out / f'{key}_source_lookup.json')
            if direction:
                # Verify every exported full-resolution pixel against the original data.
                with ExitStack() as verify:
                    bands = {p:verify.enter_context(rasterio.open(paths[p])) for p in pols}
                    output = {p:verify.enter_context(rasterio.open(out / f'{key}_{p}_db.tif')) for p in pols}
                    mask = verify.enter_context(rasterio.open(root / mask_path))
                    for win in windows(mask):
                        good = mask.read(1, window=win) == 1
                        for p in pols:
                            expected, _, _ = power_to_db(bands[p].read(1, window=win), good, clips['co' if p in ['VV','HH'] else 'cross'], epsilon)
                            if not np.array_equal(expected, output[p].read(1, window=win)):
                                raise ValueError(f'Export readback mismatch: {key} {p}')
                for path in [out / f'{key}_{p}_db.tif' for p in pols] + [root / mask_path]:
                    output_hashes[path.name] = sha(path)
            save(out / 'progress.json', manifest)
            print(key, dict(stats), 'verified', flush=True)
    for path, digest in manifest['input_sha256'].items():
        if sha(root / path) != digest:
            raise ValueError(f'Source changed during preprocessing: {path}')
    manifest.update(status='complete', output_sha256=output_hashes,
                    validation='all pixels read back; provenance pass/date/orbit/offset checked; source hashes unchanged')
    save(out / 'manifest.json', manifest)
    save(out / 'complete.json', dict(manifest_sha256=sha(out/'manifest.json'),
                                    scope='preprocessed inputs only; no splits, labels or normalization fitted'))
    print('Input revision complete:', out, flush=True)


def load_manifest(root, path):
    """Verify a completed input revision before the later patch-planning stage."""
    root, path = root.resolve(), path.resolve()
    if not path.is_relative_to(root / 'inputs'):
        raise ValueError('Input manifest must be under data/inputs')
    m = json.loads(path.read_text())
    complete = json.loads((path.parent / 'complete.json').read_text())
    if m['status'] != 'complete' or complete['manifest_sha256'] != sha(path):
        raise ValueError('Incomplete or changed input manifest')
    hashes = dict(m['input_sha256'])
    hashes[str(path.relative_to(root))] = sha(path)
    hashes[str((path.parent/'complete.json').relative_to(root))] = sha(path.parent/'complete.json')
    for name, digest in m['output_sha256'].items():
        hashes[str((path.parent / name).relative_to(root))] = digest
    for name, digest in hashes.items():
        candidate = (root / name).resolve()
        if not candidate.is_relative_to(root) or sha(candidate) != digest:
            raise ValueError(f'Changed preprocessing input/output: {name}')
    return m, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--nisar-archive-root', required=True, type=Path)
    parser.add_argument('--co-clip', type=float, nargs=2, default=DEFAULT_CLIP['co'])
    parser.add_argument('--cross-clip', type=float, nargs=2, default=DEFAULT_CLIP['cross'])
    parser.add_argument('--epsilon', type=float, default=1e-12)
    args = parser.parse_args()
    prepare(args.data_root, args.output, args.nisar_archive_root,
            {'co':args.co_clip, 'cross':args.cross_clip}, args.epsilon)


if __name__ == '__main__':
    main()
