import unittest
import numpy as np
from bogsignal.dataset.label_ignore import ignore_fringe

class IgnoreTests(unittest.TestCase):
    def test_outer_ring_only_no_cascade_and_no_positive_erosion(self):
        y=np.zeros((7,7),'uint8');y[3,3]=1
        labels,eligible,changed=ignore_fringe(y,np.ones_like(y))
        self.assertEqual(changed.sum(),8)
        self.assertEqual(labels[3,3],1)
        self.assertEqual(eligible[3,3],1)
        self.assertEqual(labels[1,3],0)
        self.assertEqual(labels[2,2],10)
        self.assertEqual(eligible[2,2],0)

    def test_existing_ignores_peat_water_and_protection(self):
        for cls in range(1,6):
            y=np.zeros((5,5),'uint8');y[2,2]=cls;y[1,1]=6;y[1,2]=255
            t=np.ones_like(y);t[1,1]=0;t[1,2]=255
            protected=np.zeros(y.shape,bool);protected[2,1]=True
            out,elig,changed=ignore_fringe(y,t,protected)
            self.assertEqual(out[1,1],6);self.assertEqual(out[1,2],255)
            self.assertEqual(out[2,1],0);self.assertEqual(out[2,2],cls)
            self.assertEqual(changed.sum(),5)

    def test_halo_tiling_is_identical(self):
        y=np.zeros((8,8),'uint8');y[3:5,3:5]=2;t=np.ones_like(y)
        expected,_,_=ignore_fringe(y,t);actual=y.copy()
        for r in [0,4]:
            for c in [0,4]:
                top=max(0,r-1);left=max(0,c-1);bottom=min(8,r+5);right=min(8,c+5)
                tile,_,_=ignore_fringe(y[top:bottom,left:right],t[top:bottom,left:right])
                actual[r:r+4,c:c+4]=tile[r-top:r-top+4,c-left:c-left+4]
        np.testing.assert_array_equal(expected,actual)

class LabelCoverageTests(unittest.TestCase):
    def test_export_eighty_percent_boundary_and_configurable_threshold(self):
        import json
        from pathlib import Path
        import tempfile
        import rasterio
        import test_dataset
        from bogsignal.dataset.pipeline import export, sha, ELIGIBLE
        for valid_count,threshold,accepted in [(52429,.8,True),(52428,.8,False),(52429,.95,False)]:
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);out=test_dataset.DatasetTests().fixture(root)
                path=root/ELIGIBLE
                with rasterio.open(path,'r+') as d:
                    a=np.zeros((256,256),'uint16');a.flat[-valid_count:]=1;d.write(a,1)
                plan=json.loads((out/'plan.json').read_text())
                plan['input_sha256'][ELIGIBLE]=sha(path)
                plan['min_label_fraction']=threshold
                plan['patches'][0]['counts']=[valid_count,0,0,0,0,0]
                (out/'plan.json').write_text(json.dumps(plan))
                if accepted:
                    export(root,out)
                    with rasterio.open(out/'train/labels/p.tif') as d:
                        self.assertEqual((d.read(1)!=255).sum(),valid_count)
                else:
                    with self.assertRaisesRegex(ValueError,'label coverage'):
                        export(root,out)