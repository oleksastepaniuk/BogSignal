"""Label-independent tiled inference using a frozen dataset specification."""
import argparse
from collections import Counter
from contextlib import ExitStack
import csv
import importlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window, transform
from shapely.geometry import box, mapping
from shapely.prepared import prep

from bogsignal.dataset.pipeline import AOI, sha, save, read_input_validity
from bogsignal.dataset.preprocessing import check_grid

NAMES=['Other','Swamp','Marsh','Permanent water','Peatland cropland','Peatland non-cropland']


def model_input(values,valid,stats):
    mean=np.asarray(stats['mean'],dtype='float32')[:,None,None]
    std=np.asarray(stats['std'],dtype='float32')[:,None,None]
    x=np.where(valid,(values-mean)/np.maximum(std,1e-8),0).astype('float32')
    return np.concatenate([x,valid.astype('float32')],axis=0)


def probabilities(output,size,kind):
    if kind not in ['logits','probabilities']:
        raise ValueError('Score kind must be logits or probabilities')
    p=np.asarray(output)
    if p.shape!=(1,6,size,size) or not np.isfinite(p).all():
        raise ValueError('Predictor must return finite [1,6,H,W] scores')
    p=p[0].astype('float32')
    if kind=='logits':
        p=np.exp(p-p.max(axis=0,keepdims=True));p/=p.sum(axis=0,keepdims=True)
    elif (p<0).any() or (p>1).any() or not np.allclose(p.sum(axis=0),1,atol=1e-4):
        raise ValueError('Invalid class probabilities')
    return p


def plan(root,dataset,out,region='osm-unclassified',core=128):
    root,dataset,out=root.resolve(),dataset.resolve(),out.resolve()
    if not out.is_relative_to(root/'inference') or out==root/'inference' or out.exists():
        raise ValueError('Choose a new output directory under data/inference')
    spec=json.loads((dataset/'plan.json').read_text())
    complete=json.loads((dataset/'complete.json').read_text())
    if complete['plan_sha256']!=sha(dataset/'plan.json'):
        raise ValueError('Dataset plan does not match completion marker')
    stats=json.loads((dataset/'normalization.json').read_text())
    if stats['channels']!=[c['name'] for c in spec['channels']]:
        raise ValueError('Normalization channel order mismatch')
    size=spec['patch_size']
    if core<=0 or core>size or (size-core)%2:
        raise ValueError('Core size must leave symmetric integer context')
    paths={AOI,dataset/'plan.json',dataset/'normalization.json',dataset/'complete.json'}
    paths.update(c['path'] for c in spec['channels']);paths.update(spec['valid_masks'])
    osm_path=root/'references/osm/OSM_polissia_water_wetlands.geojson'
    if region=='osm-unclassified':
        votes=root/'labels/osm_polygon_votes.csv'
        with votes.open(encoding='utf-8-sig') as f:
            ids={r['record_id'] for r in csv.DictReader(f) if int(r['fill_class']) in [6,8]}
        targets=gpd.read_file(osm_path)
        targets=targets[targets.record_id.isin(ids)].to_crs(spec['grid']['crs'])
        paths.update([osm_path,votes])
        if targets.empty:
            raise ValueError('No unclassified OSM wetland polygons found')
    else:
        boundary=root/'aoi/polissia_combined_wgs84.geojson'
        targets=gpd.read_file(boundary).to_crs(spec['grid']['crs'])
        paths.add(boundary)
    geometry=targets.geometry.union_all();prepared=prep(geometry)
    affine=rasterio.Affine(*spec['grid']['transform'])
    tiles=[]
    for row in range(0,spec['grid']['height'],core):
        for col in range(0,spec['grid']['width'],core):
            win=Window(col,row,min(core,spec['grid']['width']-col),min(core,spec['grid']['height']-row))
            if prepared.intersects(box(*rasterio.windows.bounds(win,affine))):
                tiles.append([row,col,int(win.height),int(win.width)])
    out.mkdir(parents=True)
    targets.to_file(out/'targets.geojson',driver='GeoJSON')
    hashes={str(p):sha(root/p) for p in paths}
    hashes[str(out/'targets.geojson')]=sha(out/'targets.geojson')
    save(out/'inference_plan.json',dict(version=1,region=region,grid=spec['grid'],channels=spec['channels'],
         normalization=stats,valid_masks=spec['valid_masks'],channel_valid_masks=spec['channel_valid_masks'],
         patch_size=size,core_size=core,halo=(size-core)//2,max_missing_fraction=spec['max_missing_fraction'],
         model_channels=2*len(spec['channels']),tiles=tiles,input_sha256=hashes,
         targets='targets.geojson',class_names=NAMES,prediction_policy='joint-valid output pixels only; >=95% per-channel context coverage within AOI',
         status='planned; no model inference has run'))
    print('Planned',region,':',len(targets),'target features,',len(tiles),'tiles;',2*len(spec['channels']),'model inputs',flush=True)


def run(root,out,predictor,checkpoint,score_kind='logits'):
    root,out=root.resolve(),out.resolve()
    if not out.is_relative_to(root/'inference') or out==root/'inference':
        raise ValueError('Inference runs must stay under data/inference')
    spec=json.loads((out/'inference_plan.json').read_text())
    if (out/'complete.json').exists() or any((out/name).exists() for name in ['classes.tif','confidence.tif','probabilities.tif','coverage.tif']):
        raise FileExistsError('Existing inference outputs; use a new run directory')
    for path,digest in spec['input_sha256'].items():
        if sha(root/path)!=digest:
            raise ValueError(f'Changed inference input: {path}')
    checkpoint_hash=sha(checkpoint)
    size,halo=spec['patch_size'],spec['halo']
    nc=len(spec['channels'])
    targets=gpd.read_file(out/spec['targets']);geometry=targets.geometry.union_all()
    geojson_geometry=mapping(geometry)
    grid=spec['grid'];affine=rasterio.Affine(*grid['transform'])
    profile=dict(driver='GTiff',height=grid['height'],width=grid['width'],crs=grid['crs'],transform=affine,
                 tiled=True,blockxsize=128,blockysize=128,compress='deflate',BIGTIFF='IF_SAFER',SPARSE_OK='TRUE')
    counters=Counter()
    with ExitStack() as stack:
        bands=[stack.enter_context(rasterio.open(root/c['path'])) for c in spec['channels']]
        masks={p:stack.enter_context(rasterio.open(root/p)) for p in spec['valid_masks']}
        aoi=stack.enter_context(rasterio.open(root/AOI));check_grid(aoi,[*bands,*masks.values()])
        writers={}
        for name,dtype,nodata,count in [('classes','uint8',255,1),('coverage','uint8',255,1),
                                        ('confidence','float32',-9999,1),('probabilities','float32',-9999,6)]:
            writers[name]=stack.enter_context(rasterio.open(out/f'{name}.tif','w',**profile,dtype=dtype,nodata=nodata,count=count))
        writers['probabilities'].descriptions=tuple(NAMES)
        for i,(row,col,height,width) in enumerate(spec['tiles']):
            core_win=Window(col,row,width,height)
            roi=geometry_mask([geojson_geometry],out_shape=(height,width),transform=transform(core_win,affine),invert=True)
            roi &= aoi.read(1,window=core_win)==1
            if not roi.any():
                continue
            # Full-size contextual tile. Missing raster-edge context is padded explicitly.
            left,top=col-halo,row-halo
            x0,y0=max(0,left),max(0,top)
            x1,y1=min(grid['width'],left+size),min(grid['height'],top+size)
            win=Window(x0,y0,x1-x0,y1-y0)
            arrays,valid=read_input_validity(bands,masks,spec['channel_valid_masks'],win)
            inside=aoi.read(1,window=win)==1
            valid &= inside[None]
            values=np.zeros((nc,size,size),'float32');full_valid=np.zeros_like(values,bool)
            sy,sx=y0-top,x0-left
            sl=np.s_[sy:sy+int(win.height),sx:sx+int(win.width)]
            for b,(a,c) in enumerate(zip(arrays,spec['channels'])):
                values[b][sl]=a.data.astype('float32')*c['scale']
            full_valid[(slice(None),)+sl]=valid
            coverage=valid.reshape(nc,-1).sum(axis=1)/max(1,int(inside.sum()))
            usable=bool(np.all(coverage>=1-spec['max_missing_fraction']))
            cls=np.full((height,width),255,'uint8');conf=np.full((height,width),-9999,'float32')
            probs=np.full((6,height,width),-9999,'float32');covered=np.where(roi,0,255).astype('uint8')
            good=roi & full_valid[:,halo:halo+height,halo:halo+width].all(axis=0)
            if usable and good.any():
                p=probabilities(predictor(model_input(values,full_valid,spec['normalization'])[None]),size,score_kind)
                cropped=p[:,halo:halo+height,halo:halo+width]
                cls[good]=cropped.argmax(axis=0)[good];conf[good]=cropped.max(axis=0)[good]
                probs[:,good]=cropped[:,good];covered[good]=1
                counters['predicted_pixels']+=int(good.sum())
            else:
                counters['rejected_context_tiles']+=1
            counters['target_pixels']+=int(roi.sum())
            for key,array in [('classes',cls[None]),('confidence',conf[None]),('probabilities',probs),('coverage',covered[None])]:
                writers[key].write(array,window=core_win)
            if i%100==0:
                print(i,'/',len(spec['tiles']),dict(counters),flush=True)
    # Read back every planned output tile before marking the run complete.
    with ExitStack() as verification:
        results={name:verification.enter_context(rasterio.open(out/f'{name}.tif')) for name in writers}
        for row,col,height,width in spec['tiles']:
            win=Window(col,row,width,height)
            cls=results['classes'].read(1,window=win)
            cov=results['coverage'].read(1,window=win)
            conf=results['confidence'].read(1,window=win)
            p=results['probabilities'].read(window=win)
            good=cov==1
            if not np.isin(cov,[0,1,255]).all() or not (cls[~good]==255).all():
                raise ValueError('Invalid output coverage or class NoData')
            if not np.isin(cls[good],range(6)).all() or not np.isfinite(p[:,good]).all():
                raise ValueError('Invalid predicted classes/probabilities')
            if not np.allclose(p[:,good].sum(axis=0),1,atol=1e-4) or not np.array_equal(cls[good],p[:,good].argmax(axis=0)):
                raise ValueError('Class output disagrees with probabilities')
            if not np.allclose(conf[good],p[:,good].max(axis=0)):
                raise ValueError('Confidence output disagrees with probabilities')
    # Polygon summaries use predicted pixels only; missing coverage stays explicit.
    from rasterio.features import geometry_window
    summaries=[]
    with rasterio.open(out/'classes.tif') as classes,rasterio.open(out/'coverage.tif') as coverage:
        for idx,target in targets.iterrows():
            try:
                win=geometry_window(classes,[mapping(target.geometry)])
            except rasterio.errors.WindowError:
                continue
            total=0;predicted=0;counts=np.zeros(6,dtype='int64')
            prepared=prep(target.geometry)
            for row in range(int(win.row_off),int(win.row_off+win.height),1024):
                for col in range(int(win.col_off),int(win.col_off+win.width),1024):
                    block=Window(col,row,min(1024,int(win.col_off+win.width)-col),min(1024,int(win.row_off+win.height)-row))
                    rect=box(*rasterio.windows.bounds(block,classes.transform))
                    if not prepared.intersects(rect):
                        continue
                    local=np.ones((int(block.height),int(block.width)),bool) if prepared.covers(rect) else geometry_mask(
                        [mapping(target.geometry)],out_shape=(int(block.height),int(block.width)),
                        transform=transform(block,classes.transform),invert=True)
                    y=classes.read(1,window=block);cov=coverage.read(1,window=block)
                    good=local & (cov==1)
                    total+=int((local & (cov!=255)).sum());predicted+=int(good.sum())
                    counts+=np.bincount(y[good],minlength=6)
            summaries.append(dict(record_id=str(target.get('record_id',idx)),name=str(target.get('name','')),
                 target_pixels=total,predicted_pixels=predicted,coverage_fraction=float(predicted/total) if total else 0,
                 class_pixels=counts.tolist(),class_fractions=(counts/max(1,predicted)).tolist()))
    save(out/'polygon_summary.json',dict(class_names=NAMES,polygons=summaries))
    for path,digest in spec['input_sha256'].items():
        if sha(root/path)!=digest:
            raise ValueError('Inference input changed during run')
    if sha(checkpoint)!=checkpoint_hash:
        raise ValueError('Checkpoint changed during run')
    save(out/'complete.json',dict(status='complete',counts=dict(counters),checkpoint_sha256=checkpoint_hash,
         plan_sha256=sha(out/'inference_plan.json'),score_kind=score_kind,
         outputs={p.name:sha(p) for p in out.glob('*.tif')},
         interpretation='model predictions, not new training labels; confidence is not independently calibrated'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','run'])
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dataset',type=Path)
    parser.add_argument('--region',choices=['polissia','osm-unclassified'],default='osm-unclassified')
    parser.add_argument('--core-size',type=int,default=128)
    parser.add_argument('--predictor',help='module:factory; factory(checkpoint_path) returns a NumPy-batch predictor')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--score-kind',choices=['logits','probabilities'],default='logits')
    a=parser.parse_args()
    if a.action=='plan':
        if not a.dataset:
            parser.error('--dataset is required for planning')
        plan(a.data_root,a.dataset,a.output,a.region,a.core_size)
    else:
        if not a.predictor or not a.checkpoint:
            parser.error('--predictor and --checkpoint are required for run')
        module,name=a.predictor.split(':',1)
        predictor=getattr(importlib.import_module(module),name)(a.checkpoint)
        run(a.data_root,a.output,predictor,a.checkpoint,a.score_kind)


if __name__=='__main__':
    main()
