import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin

from bogsignal.dataset.pipeline import sha
from bogsignal.inference.pipeline import plan,run,model_input,probabilities

class InferenceTests(unittest.TestCase):
    def test_normalization_masks_and_scores(self):
        x=np.array([[[4.,99.]]]);valid=np.array([[[True,False]]])
        result=model_input(x,valid,dict(mean=[2],std=[2]))
        np.testing.assert_array_equal(result,[[[1,0]],[[1,0]]])
        logits=np.zeros((1,6,2,2));logits[:,2]=1000
        self.assertTrue((probabilities(logits,2,'logits').argmax(axis=0)==2).all())
        with self.assertRaises(ValueError):
            probabilities(np.ones((1,6,2,2)),2,'probabilities')

    def test_label_free_end_to_end_with_context_and_missing_pixel(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);dataset=root/'dataset/v4';dataset.mkdir(parents=True)
            def write(name,a,nodata):
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
                with rasterio.open(path,'w',driver='GTiff',height=8,width=8,count=1,
                     dtype=a.dtype,crs='EPSG:4326',transform=from_origin(0,8,1,1),nodata=nodata) as d:
                    d.write(a,1)
            write('aoi/polissia_aoi_mask_20m_3035.tif',np.ones((8,8),'uint8'),255)
            for band in ['a','b']:
                write('inputs/'+band+'.tif',np.ones((8,8),'float32'),-9999)
            mask=np.ones((8,8),'uint8');mask[2,2]=0
            write('inputs/mask.tif',mask,255)
            boundary=dict(type='FeatureCollection',features=[dict(type='Feature',properties={},geometry=dict(
                type='Polygon',coordinates=[[[0,0],[4,0],[4,8],[0,8],[0,0]]]))])
            (root/'aoi/polissia_combined_wgs84.geojson').write_text(json.dumps(boundary))
            spec=dict(channels=[dict(name='S2_'+b,path='inputs/'+b+'.tif',scale=1) for b in ['a','b']],
                patch_size=8,grid=dict(height=8,width=8,crs='EPSG:4326',transform=list(from_origin(0,8,1,1))),
                valid_masks=['inputs/mask.tif'],channel_valid_masks=['inputs/mask.tif']*2,max_missing_fraction=.05)
            (dataset/'plan.json').write_text(json.dumps(spec))
            (dataset/'complete.json').write_text(json.dumps(dict(plan_sha256=sha(dataset/'plan.json'))))
            (dataset/'normalization.json').write_text(json.dumps(dict(channels=['S2_a','S2_b'],mean=[0,0],std=[1,1])))
            out=root/'inference/test'
            plan(root,dataset,out,'polissia',4)
            checkpoint=root/'checkpoint';checkpoint.write_text('synthetic test model only')
            calls=[]
            def predict(x):
                calls.append(x.copy());self.assertEqual(x.shape,(1,4,8,8))
                p=np.zeros((1,6,8,8),'float32');p[:,2]=.9;p[:,0]=.1
                return p
            run(root,out,predict,checkpoint,'probabilities')
            self.assertTrue(calls)
            with rasterio.open(out/'classes.tif') as d:
                y=d.read(1)
                self.assertTrue((y[:,4:]==255).all())
                self.assertEqual(y[2,2],255)
                self.assertEqual((y==2).sum(),31)
            with rasterio.open(out/'probabilities.tif') as d:
                self.assertEqual(d.count,6)
                self.assertAlmostEqual(float(d.read(3)[0,0]),.9,places=6)
            report=json.loads((out/'polygon_summary.json').read_text())['polygons'][0]
            self.assertEqual(report['target_pixels'],32)
            self.assertEqual(report['predicted_pixels'],31)
            self.assertTrue((out/'complete.json').exists())
            # OSM selection uses polygon votes only, not training-label rasters.
            (root/'references/osm').mkdir(parents=True)
            osm=json.loads(json.dumps(boundary));osm['features'][0]['properties']={'record_id':'wetland-unknown','name':'Unknown wetland'}
            other=json.loads(json.dumps(osm['features'][0]));other['properties']['record_id']='wetland-classified'
            osm['features'].append(other)
            (root/'references/osm/OSM_polissia_water_wetlands.geojson').write_text(json.dumps(osm))
            (root/'labels').mkdir()
            (root/'labels/osm_polygon_votes.csv').write_text('record_id,fill_class\nwetland-unknown,6\nwetland-classified,1\n')
            osm_out=root/'inference/osm'
            plan(root,dataset,osm_out,'osm-unclassified',4)
            selected=json.loads((osm_out/'targets.geojson').read_text())['features']
            self.assertEqual([f['properties']['record_id'] for f in selected],['wetland-unknown'])
            with self.assertRaises(FileExistsError):
                run(root,out,predict,checkpoint,'probabilities')

    def test_changed_input_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out=root/'inference/x';out.mkdir(parents=True)
            (out/'inference_plan.json').write_text(json.dumps(dict(input_sha256={'missing':'bad'})))
            with self.assertRaises(FileNotFoundError):
                run(root,out,lambda x:x,root/'missing')
