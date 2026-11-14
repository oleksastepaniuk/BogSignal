"""One-pixel outer uncertainty fringe on Other; preserves positive classes."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from .pipeline import ELIGIBLE, sha, save
from .preprocessing import check_grid, windows


def ignore_fringe(labels, eligible, protected=None):
    """8-neighbor outer ring only; never erode a positive label or cascade."""
    positive = np.isin(labels, [1,2,3,4,5]) & (eligible == 1)
    padded = np.pad(positive,1)
    adjacent = np.zeros(labels.shape,bool)
    h,w = labels.shape
    for dr in [-1,0,1]:
        for dc in [-1,0,1]:
            if dr or dc:
                adjacent |= padded[1+dr:1+dr+h,1+dc:1+dc+w]
    changed=(labels==0)&(eligible==1)&adjacent
    if protected is not None:
        changed &= ~protected
    result=labels.copy();training=eligible.copy()
    result[changed]=10
    training[changed]=0
    return result,training,changed


def prepare(root,out,label_revision,protected=None):
    root,out=root.resolve(),out.resolve()
    if not out.is_relative_to(root/'labels') or out==root/'labels' or out.exists():
        raise ValueError('Output must be a new label revision directory')
    previous=json.loads((label_revision/'manifest.json').read_text())
    if previous['status']!='complete':
        raise ValueError('Hole-fill revision is incomplete')
    for name,digest in previous['outputs'].items():
        if sha(label_revision/name)!=digest:
            raise ValueError(f'Changed hole-fill output: {name}')
    paths=dict(labels=label_revision/'labels.tif',eligible=root/ELIGIBLE)
    if protected:
        paths['protected']=protected.resolve()
    hashes={str(p):sha(p) for p in [*paths.values(),label_revision/'manifest.json']}
    out.mkdir(parents=True)
    total=0;existing=0
    with ExitStack() as stack:
        ds={k:stack.enter_context(rasterio.open(p)) for k,p in paths.items()}
        ref=ds['labels'];check_grid(ref,ds.values())
        profile=ref.profile.copy();profile.update(compress='deflate',tiled=True,blockxsize=512,blockysize=512)
        writers={k:stack.enter_context(rasterio.open(out/name,'w',**profile)) for k,name in
                 [('labels','labels.tif'),('eligible','training_mask.tif'),('reason','ignore_reason.tif')]}
        for d in writers.values():
            d.update_tags(POLICY='one-pass 8-neighbor Other fringe around classes 1-5; no positive-label erosion',
                          RAW_IGNORE_CLASS='10=new uncertain Other fringe; training eligibility=0; export loss index=255')
        for i,win in enumerate(windows(ref)):
            left=max(0,int(win.col_off)-1);top=max(0,int(win.row_off)-1)
            right=min(ref.width,int(win.col_off+win.width)+1);bottom=min(ref.height,int(win.row_off+win.height)+1)
            halo=Window(left,top,right-left,bottom-top)
            y=ref.read(1,window=halo);t=ds['eligible'].read(1,window=halo)
            protection=None
            if protected:
                p=ds['protected'].read(1,window=halo,masked=True)
                protection=np.ma.getmaskarray(p)|(p.data!=0)
            a,b,changed=ignore_fringe(y,t,protection)
            rs=int(win.row_off)-top;cs=int(win.col_off)-left
            core=np.s_[rs:rs+int(win.height),cs:cs+int(win.width)]
            reason=np.where(y==255,255,np.where(changed,1,np.where(t==0,2,0))).astype('uint8')
            for key,array in [('labels',a),('eligible',b),('reason',reason)]:
                writers[key].write(array[core],1,window=win)
            total+=int(changed[core].sum());existing+=int((t[core]==0).sum())
            if i%200==0:
                print(i,'blocks;',total,'new ignore pixels',flush=True)
    # Read back and independently recompute the complete rule with the same one-pixel halo.
    with ExitStack() as stack:
        readers={k:stack.enter_context(rasterio.open(out/p)) for k,p in
                 [('labels','labels.tif'),('eligible','training_mask.tif'),('reason','ignore_reason.tif')]}
        original=stack.enter_context(rasterio.open(paths['labels']))
        eligibility=stack.enter_context(rasterio.open(paths['eligible']))
        protection_ds=stack.enter_context(rasterio.open(paths['protected'])) if protected else None
        check_grid(original,readers.values())
        for win in windows(original):
            l=max(0,int(win.col_off)-1);t=max(0,int(win.row_off)-1)
            r=min(original.width,int(win.col_off+win.width)+1);b=min(original.height,int(win.row_off+win.height)+1)
            halo=Window(l,t,r-l,b-t)
            a=original.read(1,window=halo);e=eligibility.read(1,window=halo)
            p=protection_ds.read(1,window=halo,masked=True) if protected else None
            protection=(np.ma.getmaskarray(p)|(p.data!=0)) if protected else None
            expected,expected_e,changed=ignore_fringe(a,e,protection)
            rs=int(win.row_off)-t;cs=int(win.col_off)-l;core=np.s_[rs:rs+int(win.height),cs:cs+int(win.width)]
            reason=np.where(a==255,255,np.where(changed,1,np.where(e==0,2,0))).astype('uint8')
            for key,array in [('labels',expected),('eligible',expected_e),('reason',reason)]:
                if not np.array_equal(readers[key].read(1,window=win),array[core]):
                    raise ValueError('Full-resolution readback failed')
    for path,digest in hashes.items():
        if sha(Path(path))!=digest:
            raise ValueError('Label source changed')
    save(out/'manifest.json',dict(status='complete',stage='hole filled labels plus one-pixel outer Other ignore fringe',
         labels=str((out/'labels.tif').relative_to(root)),eligibility=str((out/'training_mask.tif').relative_to(root)),
         input_sha256=hashes,new_ignore_pixels=total,existing_ignore_pixels=existing,
         new_ignore_area_km2=total*.0004,
         policy=dict(target_classes=[1,2,3,4,5],connectivity=8,radius_pixels=1,passes=1,
                     editable='Other with training eligibility 1 only',raw_ignore_code=10,loss_ignore_index=255,
                     protected_negatives='explicit mask' if protected else 'no reviewed-negative raster available'),
         outputs={name:sha(out/name) for name in ['labels.tif','training_mask.tif','ignore_reason.tif']}))
    print('Ignore revision complete:',total,'new ignore pixels;',total*.0004,'km2',flush=True)


def load_label_manifest(root, path):
    root,path=root.resolve(),path.resolve()
    if not path.is_relative_to(root/'labels'):
        raise ValueError('Label revision must be under data/labels')
    m=json.loads(path.read_text())
    if m['status']!='complete':
        raise ValueError('Label revision incomplete')
    hashes=dict(m['input_sha256'])
    hashes[str(path.relative_to(root))]=sha(path)
    for name,digest in m['outputs'].items():
        candidate=(path.parent/name).resolve()
        if not candidate.is_relative_to(path.parent):
            raise ValueError('Invalid label output path')
        hashes[str(candidate.relative_to(root))]=digest
    for name,digest in hashes.items():
        if sha(root/name)!=digest:
            raise ValueError(f'Label revision input/output changed: {name}')
    for key in ['labels','eligibility']:
        if not (root/m[key]).resolve().is_relative_to(root):
            raise ValueError('Label source must be within data root')
    return m,hashes

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--label-revision',required=True,type=Path)
    p.add_argument('--protected-negatives',type=Path)
    a=p.parse_args();prepare(a.data_root,a.output,a.label_revision.resolve(),a.protected_negatives)


if __name__=='__main__':
    main()
