"""Conservative one-pass repair of small enclosed Other holes; no new ignore rule."""
import argparse
from collections import Counter
from contextlib import ExitStack
import gzip
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import shape
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window, transform

from .pipeline import LABEL, ELIGIBLE, sha, save
from .preprocessing import check_grid, windows

OFFSETS = [(r,c) for r in [-1,0,1] for c in [-1,0,1] if r or c]


def fill_holes(labels, editable, support, max_size=4, min_neighbors=5):
    """8-connected holes, single surrounding class, original-label votes only."""
    if not 1 <= max_size <= 4 or not 5 <= min_neighbors <= 8:
        raise ValueError('This conservative revision permits size 1-4 and votes 5-8')
    padded = np.pad(labels, 1, constant_values=255)
    candidates = np.zeros(labels.shape, bool)
    h,w = labels.shape
    for cls in range(1,6):
        votes = np.zeros(labels.shape, 'uint8')
        for dr,dc in OFFSETS:
            votes += padded[1+dr:1+dr+h,1+dc:1+dc+w] == cls
        candidates |= (labels == 0) & editable & support[cls] & (votes >= min_neighbors)
    output = labels.copy()
    changed = np.zeros(labels.shape, 'uint8')
    seen = set()
    for row,col in np.argwhere(candidates):
        seed=(int(row),int(col))
        if seed in seen:
            continue
        component={seed}; queue=[seed]; ring=set(); invalid=False
        while queue and len(component) <= max_size:
            r,c=queue.pop()
            for dr,dc in OFFSETS:
                nr,nc=r+dr,c+dc
                if nr<0 or nc<0 or nr>=h or nc>=w:
                    invalid=True
                elif labels[nr,nc] == 0:
                    if (nr,nc) not in component:
                        component.add((nr,nc));queue.append((nr,nc))
                else:
                    ring.add(int(labels[nr,nc]))
        seen.update(component)
        if invalid or queue or len(component)>max_size or len(ring)!=1:
            continue
        target=next(iter(ring))
        if target not in range(1,6):
            continue
        for r,c in component:
            count=sum(labels[r+dr,c+dc]==target for dr,dc in OFFSETS)
            if not editable[r,c] or not support[target][r,c] or count<min_neighbors:
                break
        else:
            for r,c in component:
                output[r,c]=target;changed[r,c]=target
    return output, changed


def prepare(root, out, poland_boundary, protected=None):
    root,out=root.resolve(),out.resolve()
    if not out.is_relative_to(root/'labels') or out==root/'labels' or out.exists():
        raise ValueError('Output must be a new named revision under data/labels')
    paths=dict(labels=root/LABEL, eligibility=root/ELIGIBLE,
        evidence=root/'labels/polissia_evidence_20m_3035.tif',
        gwl=root/'references/gwl/GWL_2022_polissia_aligned_20m_3035.tif',
        wc=root/'references/worldcover/WorldCover_2021_v200_polissia_aligned_20m_3035.tif',
        gpm=root/'references/peatland/GPM_2_0_polissia_aligned_20m_3035.tif',
        ewm=root/'references/peatland/EWM_v5_poland_polissia_aligned_20m_3035.tif',
        osm=root/'references/osm/OSM_presence_aligned_20m_3035.tif')
    if protected:
        paths['protected']=protected.resolve()
    hashes={str(p):sha(p) for p in [*paths.values(),poland_boundary.resolve()]}
    raw=poland_boundary.read_bytes()
    if poland_boundary.suffix=='.gz':
        raw=gzip.decompress(raw)
    obj=json.loads(raw)
    country=gpd.GeoSeries([shape(f.get('geometry',f)) for f in obj.get('features',[obj])],crs=4326)
    counts=Counter()
    out.mkdir(parents=True)
    with ExitStack() as stack:
        ds={key:stack.enter_context(rasterio.open(path)) for key,path in paths.items()}
        ref=ds['labels'];check_grid(ref,ds.values())
        geom=country.to_crs(ref.crs).union_all().__geo_interface__
        profile=ref.profile.copy();profile.update(compress='deflate',tiled=True,blockxsize=512,blockysize=512)
        dest=stack.enter_context(rasterio.open(out/'labels.tif','w',**profile))
        reasons=stack.enter_context(rasterio.open(out/'hole_fill_class.tif','w',**profile))
        dest.update_tags(RULE='one-pass enclosed 8-connected Other holes <=4 pixels, >=5 original neighbors of one class',
                         STATUS='inferred provisional labels; no new ignore rule applied')
        reasons.update_tags(LEGEND='0 unchanged; 1-5 inferred replacement class; 255 outside source raster')
        for i,win in enumerate(windows(ref)):
            # A component of <=4 pixels plus its boundary fits inside this halo.
            left=max(0,int(win.col_off)-5);top=max(0,int(win.row_off)-5)
            right=min(ref.width,int(win.col_off+win.width)+5);bottom=min(ref.height,int(win.row_off+win.height)+5)
            halo=Window(left,top,right-left,bottom-top)
            a={key:d.read(1,window=halo,masked=True) for key,d in ds.items()}
            y=a['labels'].data
            editable=(y==0)&(a['eligibility'].data==1)&(a['evidence'].data==0)
            editable &= ~np.isin(a['gwl'].data,[180,181,182,183])
            if protected:
                editable &= ~np.ma.getmaskarray(a['protected']) & (a['protected'].data==0)
            poland=geometry_mask([geom],out_shape=y.shape,transform=transform(halo,ref.transform),invert=True)
            peat=np.where(poland,(a['ewm'].data==10)&~np.ma.getmaskarray(a['ewm']),
                          (a['gpm'].data==1)&~np.ma.getmaskarray(a['gpm']))
            # Preserve the baseline rule: any mapped OSM water/wetland blocks peat assignment.
            peat &= (a['osm'].data != 1) & (ds['osm'].read(2,window=halo) != 1)
            wc_valid=~np.ma.getmaskarray(a['wc'])
            support={k:np.ones(y.shape,bool) for k in [1,2,3]}
            support[4]=peat & wc_valid & (a['wc'].data==40)
            support[5]=peat & wc_valid & np.isin(a['wc'].data,[10,20,30,60])
            new,changed=fill_holes(y,editable,support)
            rs=int(win.row_off)-top;cs=int(win.col_off)-left
            core=np.s_[rs:rs+int(win.height),cs:cs+int(win.width)]
            dest.write(new[core],1,window=win)
            codes=np.where(y==255,255,changed).astype('uint8')[core]
            reasons.write(codes,1,window=win)
            keys,nums=np.unique(changed[core],return_counts=True)
            counts.update({int(k):int(v) for k,v in zip(keys,nums) if k})
            if i%100==0:
                print(i,'blocks; filled pixels',dict(counts),flush=True)
    # Independent invariants over every exported pixel, with original training mask unchanged.
    observed=Counter()
    with rasterio.open(out/'labels.tif') as result, rasterio.open(out/'hole_fill_class.tif') as changes, rasterio.open(paths['labels']) as original:
        check_grid(original,[result,changes])
        for win in windows(original):
            a=original.read(1,window=win);b=result.read(1,window=win);c=changes.read(1,window=win)
            filled=(c>0)&(c<255)
            if not np.array_equal(a[~filled],b[~filled]) or not (a[filled]==0).all() or not np.array_equal(b[filled],c[filled]):
                raise ValueError('Hole-fill readback invariant failed')
            keys,nums=np.unique(c[filled],return_counts=True)
            observed.update({int(k):int(v) for k,v in zip(keys,nums)})
    if counts!=observed:
        raise ValueError('Hole-fill counts disagree')
    for path,digest in hashes.items():
        if sha(Path(path))!=digest:
            raise ValueError(f'Source changed: {path}')
    save(out/'manifest.json',dict(status='complete',stage='hole filling only; ignore policy pending review',
         rule=dict(connectivity=8,max_hole_pixels=4,min_matching_neighbors=5,passes=1,
                   boundary='all surrounding non-Other pixels must have the same class',
                   editable='training-eligible default Other only; protected negatives preserved',
                   peat='existing country-specific peat and WorldCover evidence required'),
         input_sha256=hashes,filled_pixels_by_class=dict(counts),
         labels=str((out/'labels.tif').relative_to(root)),eligibility=ELIGIBLE,
         outputs={name:sha(out/name) for name in ['labels.tif','hole_fill_class.tif']},
         protected_negatives='explicit mask supplied' if protected else 'no reviewed-negative mask available; only default-evidence Other is eligible'))
    print('Hole-fill revision complete',dict(counts),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--poland-boundary',type=Path,required=True)
    p.add_argument('--protected-negatives',type=Path)
    a=p.parse_args();prepare(a.data_root,a.output,a.poland_boundary,a.protected_negatives)


if __name__=='__main__':
    main()
