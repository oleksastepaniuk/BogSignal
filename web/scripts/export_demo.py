"""Export retained validation outputs only; no inference or test-set access."""
from pathlib import Path
import json, hashlib, shutil, sys
import numpy as np
import rasterio
from PIL import Image
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ml/src'))
from bogsignal.training.temporal_d_compare import load_run
OUT = ROOT / 'web/public/experiment'
OUT.mkdir(parents=True, exist_ok=True)
b, d = [load_run(ROOT / 'data/experiments' / name) for name in
        ['B_s2s1_resnet34_seed17', 'D_s2s1nisar_resnet34_seed17']]
assert b['ids'] == d['ids']
assert np.array_equal(b['target'], d['target'])
assert np.array_equal(b['valid'], d['valid'])
mapping = np.array([0,1,1,2,3,4], dtype='uint8')
rows = []
for i, name in enumerate(d['ids']):
    keep = d['valid'][i].astype(bool) & (d['target'][i] != 255)
    truth = np.isin(d['target'][i], [1,2]) & keep
    metrics = []
    for run in [b,d]:
        pred = (run['predictions'][1][i] == 1) & keep
        tp = int((pred & truth).sum())
        union = int((pred | truth).sum())
        metrics.append(dict(iou=tp/union if union else 0, tp=tp,
                            fp=int((pred & ~truth).sum()), fn=int((truth & ~pred).sum())))
    if truth.sum() >= 500 and min(x['iou'] for x in metrics) > .05:
        rows.append(dict(index=i, id=name, reference_pixels=int(truth.sum()),
                         b=metrics[0], d=metrics[1], delta=metrics[1]['iou']-metrics[0]['iou']))
selected = [max(rows,key=lambda r:r['delta']),
            min((r for r in rows if min(r['b']['iou'],r['d']['iou']) >= .2),key=lambda r:abs(r['delta'])),
            min(rows,key=lambda r:r['delta'])]
assert selected[0]['delta'] > .05 and abs(selected[1]['delta']) < .01 and selected[2]['delta'] < -.05
colors = np.array([[185,185,170],[25,123,73],[40,121,185],[207,154,75],[139,102,139],[255,255,255]],dtype='uint8')
stats = json.loads((Path(d['config']['dataset'])/'normalization.json').read_text())
images = np.load(d['cache']/'image.npy', mmap_mode='r')
backgrounds=[]
for row, label in zip(selected,['helps','little','worsens']):
    i=row.pop('index'); row['key']=label
    keep = d['valid'][i].astype(bool) & (d['target'][i] != 255)
    target=np.full((256,256),5,dtype='uint8'); target[keep]=mapping[d['target'][i][keep]]
    physical=images[i,:6]*np.array(stats['std'])[:6,None,None]+np.array(stats['mean'])[:6,None,None]
    rgb=(np.clip(physical[[2,1,0]].transpose(1,2,0)/.3,0,1)**.7*255).astype('uint8')
    original=ROOT/'data/presentation/sentinel2_rgb10m'/(row['id']+'.tif')
    if original.exists():
        metadata=json.loads(original.with_suffix('.json').read_text())
        assert hashlib.sha256(original.read_bytes()).hexdigest()==metadata['sha256']
        with rasterio.open(original) as source:
            assert source.width==512 and source.height==512 and source.res==(10,10)
            raw=source.read(); invalid=np.any(raw==65535,axis=0)
            rgb=(np.clip(raw.transpose(1,2,0)/3000,0,1)**.7*255).astype('uint8')
            rgb[invalid]=[243,246,240]
            # Tiny source-mosaic gaps: nearest valid RGB within four pixels, display only.
            filled=0
            for y,x in np.argwhere(invalid):
                y0,y1=max(0,y-4),min(rgb.shape[0],y+5)
                x0,x1=max(0,x-4),min(rgb.shape[1],x+5)
                yy,xx=np.nonzero(~invalid[y0:y1,x0:x1]); yy+=y0;xx+=x0
                if len(yy):
                    distances=(yy-y)**2+(xx-x)**2; nearest=int(distances.argmin())
                    if distances[nearest]<=16:
                        rgb[y,x]=rgb[yy[nearest],xx[nearest]];filled+=1
            metadata['display_gap_fill_pixels']=filled
            metadata['display_gap_fill_policy']='Nearest original valid RGB within 4 pixels; visualization only, no source/prediction/label changes'

        backgrounds.append(metadata)
    else:
        backgrounds.append(dict(patch_id=row['id'],pixel_size_m=20,selection='Retained normalized training RGB, reversed normalization'))
    Image.fromarray(rgb).save(OUT/f'{label}-rgb.png')
    for name,arr in [('reference',target),('b',b['predictions'][1][i]),('d',d['predictions'][1][i])]:
        arr=np.where(keep,arr,5)
        Image.fromarray(colors[arr]).save(OUT/f'{label}-{name}.png')
        if name in ['b','d']:
            # Highlight combined swamp/marsh only; retain full classes in the top panels.
            rgba=np.zeros((256,256,4),dtype='uint8')
            rgba[:,:,:3]=[255,220,0]
            rgba[:,:,3]=np.where(arr==1,195,0).astype('uint8')
            Image.fromarray(rgba).save(OUT/f'{label}-{name}-overlay.png')

    disagreement=np.full((256,256,3),[230,235,225],dtype='uint8')
    bp=b['predictions'][1][i]==1; dp=d['predictions'][1][i]==1
    disagreement[keep & bp & ~dp]=[201,111,52]
    disagreement[keep & dp & ~bp]=[66,96,183]
    disagreement[~keep]=255
    Image.fromarray(disagreement).save(OUT/f'{label}-disagreement.png')

excluded={row['id'] for row in selected}
validity=np.load(d['cache']/'validity.npy',mmap_mode='r')
candidates=[i for i,name in enumerate(d['ids']) if name not in excluded and np.asarray(validity[i],bool).mean()>.98]
sensor_index=max(candidates,key=lambda i:int((np.isin(d['target'][i],[1,2]) & d['valid'][i].astype(bool)).sum()))
sensor_id=d['ids'][sensor_index]
physical=images[sensor_index]*np.array(stats['std'])[:,None,None]+np.array(stats['mean'])[:,None,None]
sensor_cards=[]
for name,band,title,lookup in [('s2',None,'Sentinel-2 RGB','S2'),('s1',7,'Sentinel-1 ascending VH','S1_ASCENDING'),('nisar',11,'NISAR ascending HV','NISAR_ASCENDING')]:
    if band is None:
        display=(np.clip(physical[[2,1,0]].transpose(1,2,0)/.3,0,1)**.7*255).astype('uint8')
        keep=validity[sensor_index,:3].all(axis=0)
    else:
        gray=(np.clip((physical[band]+30)/25,0,1)*255).astype('uint8')
        display=np.repeat(gray[:,:,None],3,axis=2)
        keep=validity[sensor_index,band].astype(bool)
    display[~keep]=255
    Image.fromarray(display).save(OUT/f'sensor-{name}.png')
    sources=json.loads((Path(d['config']['dataset'])/'provenance'/f'{lookup}_source_lookup.json').read_text())['sources']
    dates=sorted(source['acquired_utc'][:10] for source in sources)
    sensor_cards.append(dict(sensor=name,patch_id=sensor_id,title=title,source_window=[dates[0],dates[-1]],display='RGB 0-0.3 reflectance, gamma 0.7' if band is None else 'Grayscale -30 to -5 dB; brightness is backscatter, not confirmed wetland',filename=f'sensor-{name}.png'))
report=Path('/home/krk/projects/Argos/docs')
summary=json.loads((report/'final_experiment_report_data.json').read_text())
assert all(r['validation']['scored_pixels']==37411425 for r in summary['baseline'])
# Verify retained arrays reproduce the report's complete B/D combined matrices.
for run, row_index in [(b,1),(d,3)]:
    cm=np.zeros((5,5),dtype='int64')
    for i in range(len(run['ids'])):
        keep=run['valid'][i].astype(bool) & (run['target'][i]!=255)
        truth=mapping[run['target'][i][keep]].astype('int64')
        pred=run['predictions'][1][i][keep].astype('int64')
        cm+=np.bincount(truth*5+pred,minlength=25).reshape(5,5)
    assert np.array_equal(cm,summary['baseline'][row_index]['combined_class']['confusion_matrix'])

payload=dict(examples=selected, baseline=summary['baseline'], sensor_cards=sensor_cards, temporal=summary['all_sensor_common_area'],
             budgets=summary['candidate_ranking']['baseline'][3]['area_budgets'],
             selection='Among validation patches with >=500 reference swamp/marsh pixels and B/D IoU >0.05: largest and smallest D-B binary wetland IoU change; little-effect minimizes absolute change with both IoUs >=0.20. Deliberate performance-selected illustrations, not representative samples.',
             provenance=[{k:r[k] for k in ['run','checkpoint_sha256','dataset_hashes','prediction_sources']} for r in [b,d]],
             decision='Five-class argmax after summing Swamp+Marsh probabilities; wetland is combined class 1.',
             swipe_display='Yellow #ffdc00 at alpha 195 only for combined Swamp+Marsh predictions; all other/excluded pixels transparent. Top panels retain original class palette.',
             rgb_backgrounds=backgrounds,rgb_display='RGB B4/B3/B2, fixed 0-0.3 reflectance stretch, gamma 0.7. Swipe uses original 10 m source pixels where present; predictions remain 20 m. Sensor-card RGB uses the retained 20 m training mosaic.',
             report_sha256=hashlib.sha256((report/'final_experiment_report.html').read_bytes()).hexdigest())
(OUT/'demo.json').write_text(json.dumps(payload,indent=2)+'\n')
shutil.copy2(report/'final_experiment_report.html',OUT/'final_experiment_report.html')
shutil.copy2(report/'final_experiment_report_data.json',OUT/'final_experiment_report_data.json')
print(json.dumps(selected,indent=2))
