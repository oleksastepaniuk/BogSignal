import unittest
import numpy as np
from bogsignal.dataset.label_holes import fill_holes

class HoleTests(unittest.TestCase):
    def run_rule(self,y,editable=None,support=None):
        return fill_holes(y,np.ones(y.shape,bool) if editable is None else editable,
                          {c:np.ones(y.shape,bool) for c in range(1,6)} if support is None else support)

    def test_one_and_four_pixel_enclosed_holes(self):
        for sl in [np.s_[3:4,3:4],np.s_[3:5,3:5]]:
            y=np.ones((9,9),'uint8');y[sl]=0
            out,changes=self.run_rule(y)
            self.assertTrue((out==1).all())
            self.assertEqual((changes>0).sum(),(y==0).sum())

    def test_large_holes_and_open_boundaries_preserved(self):
        y=np.ones((9,9),'uint8');y[3:6,3:6]=0
        out,_=self.run_rule(y);np.testing.assert_array_equal(out,y)
        y=np.ones((9,9),'uint8');y[:5,4]=0
        out,_=self.run_rule(y);np.testing.assert_array_equal(out,y)

    def test_conflicting_ring_or_protected_pixel_preserved(self):
        y=np.ones((9,9),'uint8');y[4,4]=0;y[3,3]=2
        out,_=self.run_rule(y);np.testing.assert_array_equal(out,y)
        y[3,3]=1;editable=np.ones(y.shape,bool);editable[4,4]=False
        out,_=self.run_rule(y,editable);np.testing.assert_array_equal(out,y)

    def test_peat_requires_support_and_cannot_use_ignore(self):
        y=np.full((9,9),4,'uint8');y[4,4]=0
        support={c:np.ones(y.shape,bool) for c in range(1,6)};support[4][4,4]=False
        out,_=self.run_rule(y,support=support);np.testing.assert_array_equal(out,y)
        y[4,4]=255
        out,_=self.run_rule(y);np.testing.assert_array_equal(out,y)

    def test_three_neighbor_edge_is_not_filled(self):
        y=np.zeros((9,9),'uint8');y[3:6,3]=1
        out,_=self.run_rule(y);np.testing.assert_array_equal(out,y)

    def test_tiled_halo_equals_whole_image(self):
        y=np.ones((32,32),'uint8');y[15:17,15:17]=0;y[5,16]=0
        expected,_=self.run_rule(y);actual=y.copy()
        for r in [0,16]:
            for c in [0,16]:
                top=max(0,r-5);left=max(0,c-5);bottom=min(32,r+21);right=min(32,c+21)
                tile,_=self.run_rule(y[top:bottom,left:right])
                actual[r:r+16,c:c+16]=tile[r-top:r-top+16,c-left:c-left+16]
        np.testing.assert_array_equal(actual,expected)
