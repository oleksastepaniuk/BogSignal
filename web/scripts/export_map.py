"""Export a single overview image per regional map layer; retain native TIFFs."""
from pathlib import Path
import json,gzip,sys,re,hashlib
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform,transform_bounds,Resampling
from rasterio.transform import array_bounds
from rasterio.windows import Window
from affine import Affine
from PIL import Image
from shapely.geometry import shape,mapping,box
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'web/public/map'
OUT.mkdir(parents=True,exist_ok=True)
assert OUT.resolve()==(ROOT/'web/public/map').resolve()
for old in OUT.iterdir():
    if re.fullmatch(r'r\d+_c\d+-(labels|prediction)\.png',old.name): old.unlink()
targets=Path('/home/krk/projects/Argos/data/targets')
geo=json.loads((targets/'polissia_combined_wgs84.geojson').read_text())
polissia=[dict(type='Feature',properties={'name':'Polissia study area','caveat':'Approximate ecological outline adapted from Save Polesia regional map and local polygons','outline_reference':'https://savepolesia.org/'},geometry=mapping(shape(f['geometry']).simplify(.002,preserve_topology=True))) for f in geo['features']]
(OUT/'polissia.geojson').write_text(json.dumps(dict(type='FeatureCollection',features=polissia),separators=(',',':')))
countries=[]
for code,name in [('UA','Ukraine'),('BY','Belarus'),('PL','Poland'),('RU','Russia')]:
    raw=json.loads(gzip.decompress((targets/'polissia_clip_provenance'/f'{code}_boundary.geojson.gz').read_bytes()))
    for feature in raw.get('features',[dict(geometry=raw)]):
        g=shape(feature['geometry']).intersection(box(14,46,44,58)).simplify(.008,preserve_topology=True)
        if not g.is_empty: countries.append(dict(type='Feature',properties=dict(name=name),geometry=mapping(g)))
(OUT/'countries.geojson').write_text(json.dumps(dict(type='FeatureCollection',features=countries),separators=(',',':')))
if '--geography-only' in sys.argv: print('Local boundaries exported; old per-patch PNGs removed');sys.exit(0)
run=ROOT/'data/inference/polissia_D_v4_web'
complete=json.loads((run/'complete.json').read_text())
assert complete['status']=='complete'
plan=json.loads((ROOT/'data/dataset/s1s2nisar_256_v4/plan.json').read_text())
palette=np.array([[0,0,0,0],[255,220,0,235],[40,121,185,235],[207,154,75,235],[139,102,139,235],[0,0,0,0]],dtype='uint8')
mapping_codes=np.array([0,1,1,2,3,4],dtype='uint8')
with rasterio.open(run/'coverage.tif') as coverage:
    tr,w,h=calculate_default_transform(coverage.crs,'EPSG:3857',coverage.width,coverage.height,*coverage.bounds)
    scale=max(w,h)/4096; width=round(w/scale);height=round(h/scale); tr=tr*Affine.scale(w/width,h/height)
    bbox=transform_bounds('EPSG:3857','EPSG:4326',*array_bounds(height,width,tr),densify_pts=21)
params=dict(crs='EPSG:3857',transform=tr,width=width,height=height,resampling=Resampling.nearest)
pred=np.full((height,width),5,dtype='uint8'); labels=np.full_like(pred,5); coverage_map=np.full_like(pred,255)
with rasterio.open(run/'probabilities.tif') as p, rasterio.open(run/'coverage.tif') as c, rasterio.open(ROOT/'data'/plan['label_path']) as l, rasterio.open(ROOT/'data'/plan['eligibility_path']) as e:
    with WarpedVRT(p,**params) as pv,WarpedVRT(c,**params) as cv,WarpedVRT(l,**params) as lv,WarpedVRT(e,**params) as ev:
        for row in range(0,height,256):
            win=Window(0,row,width,min(256,height-row));sl=np.s_[row:row+int(win.height),:]
            cov=cv.read(1,window=win); probs=pv.read(window=win)
            combined=np.stack([probs[0],probs[1]+probs[2],probs[3],probs[4],probs[5]])
            pred[sl]=np.where(cov==1,combined.argmax(axis=0),5)
            raw=lv.read(1,window=win); eligible=ev.read(1,window=win)==1
            good=eligible&(raw<6); chunk=np.full(raw.shape,5,dtype='uint8');chunk[good]=mapping_codes[raw[good]]
            labels[sl]=chunk;coverage_map[sl]=cov
            if row%1024==0:print('Regional overview row',row,'/',height,flush=True)
Image.fromarray(palette[pred]).save(OUT/'prediction.png')
Image.fromarray(palette[labels]).save(OUT/'labels.png')
rgba=np.zeros((height,width,4),dtype='uint8');rgba[coverage_map==1]=[46,116,128,65];rgba[coverage_map==0]=[80,80,80,80]
Image.fromarray(rgba).save(OUT/'coverage.png')
counts=complete['counts']
from pyproj import Transformer
center=Transformer.from_crs('EPSG:3035','EPSG:4326',always_xy=True).transform(5187420+(19456+128)*20,3609020-(11008+128)*20)
payload=dict(example_center=[center[1],center[0]],bounds=[[bbox[1],bbox[0]],[bbox[3],bbox[2]]],width=width,height=height,display_pixel_size_meters=tr.a,native_pixel_size_meters=20,resampling='nearest',counts=counts,coverage_fraction=counts['predicted_pixels']/counts['target_pixels'],checkpoint_sha256=complete['checkpoint_sha256'],inference_plan_sha256=complete['plan_sha256'],source='data/inference/polissia_D_v4_web',outputs=complete['outputs'],scope='Regional inference wherever all-sensor validity rules permit; includes training and held-out geography; no new evaluation or test-set scores.',decision='argmax([pOther,pSwamp+pMarsh,pWater,pPeatCrop,pPeatNonCrop])',label_source=plan['label_path'],eligibility_source=plan['eligibility_path'],country_attribution='OpenStreetMap contributors, ODbL 1.0; saved boundaries simplified only for display',outline_reference='https://savepolesia.org/',polissia_caveat='Approximate ecological boundary adapted from Save Polesia regional map and local polygons; not a legal or habitat boundary')
payload['implementation_provenance']=json.loads((run/'implementation_provenance.json').read_text())
payload['display_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.suffix in ['.png','.geojson']}
(OUT/'manifest.json').write_text(json.dumps(payload,indent=2)+'\n')
print('Regional map exported:',payload['coverage_fraction'],width,height,flush=True)