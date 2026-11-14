import json
from pathlib import Path
import tempfile
from datetime import datetime, timezone
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import rasterio
from rasterio.transform import from_origin

from bogsignal.dataset.preprocessing import (power_to_db, source_tables,
    validate_provenance, validate_nisar_identity, prepare, load_manifest, canonical_source, nisar_sources)
from bogsignal.dataset.pipeline import channels, sha


class PreprocessingTests(unittest.TestCase):
    def test_local_offset_is_published_as_utc(self):
        result = canonical_source(dict(acquired_utc='2026-10-25T18:00:00+03:00'))
        self.assertEqual(result['acquired_utc'], '2026-10-25T15:00:00+00:00')

    def test_db_mask_and_floor(self):
        x = np.array([0, -1, np.nan, np.inf, 1e-20, 1e-2, 1, 100], dtype='float32')
        y, valid, counts = power_to_db(x, np.ones(8, bool), [-40, 5])
        np.testing.assert_array_equal(valid, [0,0,0,0,1,1,1,1])
        np.testing.assert_allclose(y[valid], [-40,-20,0,5], atol=1e-5)
        self.assertTrue((y[~valid] == -9999).all())
        self.assertEqual(counts['floored_positive'], 1)
        with self.assertRaises(ValueError):
            power_to_db(x, valid, [0, -10])

    def test_provenance_rejects_wrong_pass_date_orbit(self):
        sources = [dict(source_index=1, acquired_utc='2026-10-25T17:34:25Z',
                        relative_orbit=7, mode='IW', pass_='ASCENDING', polarizations=['VV','VH'])]
        sources[0]['pass'] = sources[0].pop('pass_')
        tables = source_tables(sources, 'S1', 'ASCENDING')
        ids = np.array([1]); days = np.array([20751]); offsets = np.array([0.]); orbit=np.array([7]); valid=np.array([True])
        days[:] = tables['day'][1]
        validate_provenance(ids, days, offsets, orbit, valid, tables)
        for changed in [('date', days+1, orbit), ('orbit', days, orbit+1)]:
            with self.assertRaises(ValueError):
                validate_provenance(ids, changed[1], offsets, changed[2], valid, tables)
        with self.assertRaises(ValueError):
            validate_provenance(ids, days, offsets, orbit, valid, source_tables(sources, 'S1', 'DESCENDING'))

    def test_native_identity_mismatch(self):
        source=dict(id='a', granule_id='g', orbit_pass='ASCENDING', relative_orbit=1,
                    frame=2, acquired_start='2026-10-25T00:00:00Z', maturity='VALIDATED')
        identity=dict(granuleId='g',orbitPassDirection='Ascending',trackNumber=1,frameNumber=2,
                      zeroDopplerStartTime='2026-10-25T00:00:00.000000000', productType='GCOV',radarBand='L')
        validate_nisar_identity(source,identity)
        identity['orbitPassDirection']='Descending'
        with self.assertRaises(ValueError):
            validate_nisar_identity(source,identity)

    def test_mixed_native_versions_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'data/h5').mkdir(parents=True)
            sources = []
            for index, release in enumerate(['P05036','P05037']):
                source = dict(id=str(index),granule_id=f'granule{index}',orbit_pass='ASCENDING',
                              relative_orbit=7,frame=2,acquired_start='2026-10-25T17:34:25Z',maturity='VALIDATED')
                sources.append(source)
                with h5py.File(root/'data/h5'/f'{index}.h5','w') as h:
                    g=h.create_group('science/LSAR/identification')
                    values = dict(granuleId=source['granule_id'],orbitPassDirection='Ascending',
                        trackNumber=7,frameNumber=2,zeroDopplerStartTime='2026-10-25T17:34:25Z',
                        zeroDopplerEndTime='2026-10-25T17:35:00Z',compositeReleaseId=release,
                        productVersion='1.0.20',productSpecificationVersion='1.6.0',productType='GCOV',radarBand='L')
                    for key,value in values.items():
                        g[key]=value
            with self.assertRaisesRegex(ValueError,'Mixed NISAR processing versions'):
                nisar_sources(sources,root)

    def test_prepare_full_revision_without_split_or_normalization(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            def write(path, array, nodata):
                path.parent.mkdir(parents=True,exist_ok=True)
                with rasterio.open(path,'w',driver='GTiff',height=4,width=4,count=1,
                                   dtype=array.dtype,crs='EPSG:3035',transform=from_origin(0,80,20,20),nodata=nodata) as d:
                    d.write(array,1)
            write(root/'aoi/polissia_aoi_mask_20m_3035.tif',np.ones((4,4),'uint8'),255)
            for sensor,folder,pols in [('S1','sentinel-1',['VV','VH']),('NISAR','nisar',['HH','HV']),('S2','sentinel-2',['B2','B3','B4','B8','B11','B12'])]:
                sources=[]
                for index,direction in enumerate(['ASCENDING','DESCENDING'] if sensor!='S2' else [None],1):
                    sources.append(dict(source_index=index,acquired_utc='2026-10-25T17:34:25Z',relative_orbit=7,
                                        mode='IW',polarizations=['VV','VH'],**({'pass':direction} if sensor=='S1' else {'orbit_pass':direction})))
                    prefix=f'{sensor}_{direction}_polissia_target-20261025' if direction else 'S2_polissia_target-20261025'
                    arrays={p:np.full((4,4),.01,'float32') for p in pols}
                    if direction:
                        arrays[pols[0]][0,0]=0
                    arrays.update(valid_mask=np.ones((4,4),'uint8'),source_index=np.full((4,4),index,'uint16'),
                                  observation_date=np.full((4,4),int(datetime(2026,10,25,tzinfo=timezone.utc).timestamp()//86400),'uint16'),
                                  time_offset_days=np.zeros((4,4),'float32'))
                    if direction:
                        arrays['relative_orbit']=np.full((4,4),7,'uint16')
                        arrays['angle' if sensor=='S1' else 'gcov_mask']=np.full((4,4),30 if sensor=='S1' else 1,'float32')
                    for field,a in arrays.items():
                        write(root/'inputs'/folder/f'{prefix}_{field}.tif',a,255 if a.dtype==np.uint8 else 65535 if a.dtype==np.uint16 else -9999)
                (root/'inputs'/folder/'source_lookup.json').write_text(json.dumps(dict(sources=sources)))
            for name in ['mask_policy.json','huge_blob_exclusions.geojson']:
                (root/'inputs/nisar'/name).write_text('{}')
            original= root / channels(True)[6]['path']
            before=sha(original)
            out=root/'inputs/preprocessing_v4'
            with patch('bogsignal.dataset.preprocessing.nisar_sources',side_effect=lambda s,r:s):
                prepare(root,out,root)
            manifest,_=load_manifest(root,out/'manifest.json')
            self.assertEqual(manifest['status'],'complete')
            self.assertEqual(len(manifest['channels']),14)
            self.assertEqual(manifest['auxiliary']['S1_ASCENDING']['counts']['excluded_new'],1)
            self.assertEqual(sha(original),before)
            self.assertFalse((out/'normalization.json').exists())
            self.assertFalse((out/'plan.json').exists())
            with rasterio.open(root/manifest['channels'][6]['path']) as ds:
                self.assertTrue(ds.read(1,masked=True).mask[0,0])
            with self.assertRaises(FileExistsError):
                prepare(root,out,root)
            (out/'S1_ASCENDING_source_lookup.json').write_text('{}')
            with self.assertRaises(ValueError):
                load_manifest(root,out/'manifest.json')
