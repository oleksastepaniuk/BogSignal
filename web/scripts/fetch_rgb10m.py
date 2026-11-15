"""Fetch true 10 m RGB using exactly the source scenes selected in saved 20 m mosaics."""
from pathlib import Path
import json,hashlib,io
import ee,numpy as np,rasterio,requests
from rasterio.windows import Window
from affine import Affine
ROOT=Path('/home/krk/projects/BogSignal'); OUT=ROOT/'data/presentation/sentinel2_rgb10m';OUT.mkdir(parents=True,exist_ok=True)
dsroot=ROOT/'data/dataset/s1s2nisar_256_v4'
records={r['id']:r for r in map(json.loads,(dsroot/'index.jsonl').read_text().splitlines())}
demo=json.loads((ROOT/'web/public/experiment/demo.json').read_text())
lookup={s['source_index']:s for s in json.loads((dsroot/'provenance/S2_source_lookup.json').read_text())['sources']}
ee.Initialize(project='pysheets-384513');ee.data.setDeadline(180000)
manifest=[]
with rasterio.open(ROOT/'data/inputs/sentinel-2/S2_polissia_target-20261025_source_index.tif') as source:
    for row in demo['examples']:
        record=records[row['id']];w=Window(record['col'],record['row'],256,256)
        indices=source.read(1,window=w);index10=np.repeat(np.repeat(indices,2,axis=0),2,axis=1)
        tr=source.window_transform(w)*Affine.scale(.5,.5)
        dest=OUT/(row['id']+'.tif'); provenance=OUT/(row['id']+'.json')
        if dest.exists():
            assert provenance.exists();manifest.append(json.loads(provenance.read_text()));print('Cached',row['id'],flush=True);continue
        rgb=np.full((3,512,512),65535,dtype='uint16');sources=[]
        for value in np.unique(indices):
            if int(value) not in lookup:continue
            s=lookup[int(value)];asset=s['asset_id']
            url=ee.Image(asset).select(['B4','B3','B2']).unmask(65535).toUint16().getDownloadURL(dict(crs='EPSG:3035',crs_transform=list(tr)[:6],dimensions=[512,512],format='GEO_TIFF'))
            response=requests.get(url,timeout=(30,180))
            if not response.ok:raise RuntimeError('RGB download HTTP '+str(response.status_code))
            with rasterio.MemoryFile(response.content) as mem:
                with mem.open() as chip:
                    assert chip.width==512 and chip.height==512 and chip.count==3 and chip.transform.almost_equals(tr)
                    a=chip.read();keep=index10==value;rgb[:,keep]=a[:,keep]
            sources.append(dict(source_index=int(value),asset_id=asset,acquired_utc=s['acquired_utc']))
            print(row['id'],'source',int(value),'retrieved at 10 m',flush=True)
        with rasterio.open(dest,'w',driver='GTiff',width=512,height=512,count=3,dtype='uint16',crs='EPSG:3035',transform=tr,nodata=65535,compress='deflate') as f:f.write(rgb)
        record=dict(patch_id=row['id'],path=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),pixel_size_m=10,sources=sources,selection='Original 20 m mosaic source-index selection repeated to 10 m; genuine 10 m B4/B3/B2 source pixels on the exact patch extent; context display only')
        provenance.write_text(json.dumps(record,indent=2)+'\n');manifest.append(record)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Downloaded all three real 10 m RGB backgrounds',flush=True)