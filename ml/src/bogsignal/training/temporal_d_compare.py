"""Compare frozen D and all-sensor temporal-mean checkpoints on common pixels."""
import argparse
import base64
import html
import json
from pathlib import Path
import numpy as np
from .core import build_cache, combined_metrics, metadata_hashes, metrics, sha256
from .report import completed_version, percent, write_report


def score_aligned(old_ids, new_ids, old_target, new_target, old_valid, new_valid, old_predictions, new_predictions):
    if len(set(old_ids)) != len(old_ids) or len(set(new_ids)) != len(new_ids):
        raise ValueError('Duplicate patch IDs')
    lookup = {name: i for i, name in enumerate(old_ids)}
    cms = [[np.zeros((6, 6), dtype='int64'), np.zeros((5, 5), dtype='int64')] for _ in range(2)]
    mapping = np.array([0, 1, 1, 2, 3, 4], dtype='uint8')
    for j, name in enumerate(new_ids):
        if name not in lookup:
            raise ValueError('Temporal validation patch missing from D')
        i = lookup[name]
        keep = np.asarray(new_valid[j], bool) & (new_target[j] != 255)
        if np.any(keep & ~np.asarray(old_valid[i], bool)) or not np.array_equal(old_target[i][keep], new_target[j][keep]):
            raise ValueError('Common pixels have changed labels or are invalid in D')
        target = new_target[j][keep].astype('int64')
        for model, index, predictions in [(0, i, old_predictions), (1, j, new_predictions)]:
            for kind, classes in [(0, 6), (1, 5)]:
                truth = target if kind == 0 else mapping[target]
                predicted = predictions[kind][index][keep].astype('int64')
                if np.any((predicted < 0) | (predicted >= classes)):
                    raise ValueError('Invalid predicted class on scoring mask')
                cms[model][kind] += np.bincount(truth * classes + predicted, minlength=classes**2).reshape(classes, classes)
    return [dict(validation=metrics(c[0]), combined_class=combined_metrics(c[1])) for c in cms]


def load_run(run):
    run = Path(run).resolve()
    frozen = json.loads((run / 'complete.json').read_text())
    config = json.loads((run / 'manifest.json').read_text())['config']
    if sha256(run / 'best.pt') != frozen['checkpoint_sha256'] or sha256(run / 'results.json') != frozen['results_sha256'] or metadata_hashes(config['dataset']) != frozen['dataset_hashes']:
        raise ValueError('Frozen training provenance changed')
    cache = build_cache(config['dataset'], config['channels']) / 'validation'
    prediction_paths = [completed_version(run, prefix) / 'validation_predictions.npy' for prefix in ['validation_evaluation', 'combined_evaluation']]
    provenance = []
    for path in prediction_paths:
        marker = json.loads((path.parent / 'complete.json').read_text())
        if marker['checkpoint_sha256'] != frozen['checkpoint_sha256'] or marker['original_results_sha256'] != frozen['results_sha256'] or sha256(path) != marker['predictions_sha256'] or sha256(path.parent / 'results.json') != marker['results_sha256']:
            raise ValueError('Saved prediction provenance changed')
        provenance.append(dict(path=str(path), sha256=marker['predictions_sha256']))
    return dict(run=str(run), config=config, checkpoint_sha256=frozen['checkpoint_sha256'], dataset_hashes=frozen['dataset_hashes'],
        cache=cache, ids=json.loads((cache / 'ids.json').read_text()), target=np.load(cache / 'target.npy', mmap_mode='r'),
        valid=np.load(cache / 'evaluation.npy', mmap_mode='r'), predictions=[np.load(p, mmap_mode='r') for p in prediction_paths],
        prediction_sources=provenance)


def compare(baseline, temporal, output):
    output = Path(output).resolve()
    output.mkdir(exist_ok=False, parents=True)
    old, new = load_run(baseline), load_run(temporal)
    recipes = [{k:v for k,v in x['config'].items() if k not in ['experiment', 'dataset', 'output']} for x in [old, new]]
    if recipes[0] != recipes[1]:
        raise ValueError('D and temporal D must use the same channels and training recipe')
    scored = score_aligned(old['ids'], new['ids'], old['target'], new['target'], old['valid'], new['valid'], old['predictions'], new['predictions'])
    wetland = [s['combined_class']['classes']['Wetland (Swamp + Marsh)'] for s in scored]
    keys = ['recall', 'precision', 'dice', 'f2', 'iou']
    rows = ['| ' + ' | '.join([label, percent(s['validation']['macro_iou'])] + [percent(w[k]) for k in keys]) + ' |' for label,s,w in zip(['D: original static NISAR', 'TD: averaged NISAR'], scored, wetland)]
    delta = {k:wetland[1][k] - wetland[0][k] for k in keys}
    delta['macro_iou'] = scored[1]['validation']['macro_iou'] - scored[0]['validation']['macro_iou']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(keys))
    for offset,w,label in [(-.18,wetland[0],'D: static NISAR'),(.18,wetland[1],'TD: averaged NISAR')]:
        ax.bar(x+offset,[100*w[k] for k in keys],width=.36,label=label)
    ax.set(xticks=x,xticklabels=['Recall','Reference precision','F1','F2','IoU'],ylim=(0,100),ylabel='Weak-reference score (%)',title='Combined swamp/marsh on identical validation pixels')
    ax.legend(); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(output / 'comparison.png',dpi=150); plt.close(fig)
    payload = dict(baseline=scored[0],temporal=scored[1],delta=delta,validation_patches=len(new['ids']),
        validation_ids=new['ids'],scored_pixels=scored[0]['validation']['scored_pixels'],test_evaluated=False,
        provenance=[{k:x[k] for k in ['run','config','checkpoint_sha256','dataset_hashes','prediction_sources']} for x in [old,new]],
        source_sha256=sha256(__file__))
    (output/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
    direction = 'increased' if delta['recall'] >= 0 else 'decreased'
    text = f"""# All-sensor temporal mean versus experiment D — seed 17

TD uses Sentinel-2 + Sentinel-1 + averaged NISAR (14 physical channels plus 14
validity channels). Only NISAR is temporally averaged; Sentinel-1 and Sentinel-2
remain static. The original D checkpoint is reused, not retrained or reselected.

## Ordinary per-pixel comparison

| Model | Six-class macro IoU | Wetland recall | Reference precision | F1 | F2 | Wetland IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

TD minus D, percentage points: recall {100*delta['recall']:+.2f}, precision
{100*delta['precision']:+.2f}, F1 {100*delta['dice']:+.2f}, F2
{100*delta['f2']:+.2f}, wetland IoU {100*delta['iou']:+.2f}, six-class macro IoU
{100*delta['macro_iou']:+.2f}. Reference wetland recall {direction}.

Both maps use probability-sum combined-class inference:
`argmax([pOther, pSwamp+pMarsh, pWater, pPeatCrop, pPeatNonCrop])`.
Every eligible pixel receives a class; no area-budget ranking or threshold tuning
is used for this comparison. Precision is agreement with incomplete weak labels,
not independently verified real-world precision.

![Comparison](comparison.png)

## Matched scoring and limits

Both checkpoints are scored on the same {len(new['ids']):,} validation patches and
{payload['scored_pixels']:,} valid pixels. Temporal validation IDs are matched to
D by patch ID, not array position. Every scored label is checked equal and every
pixel is required to be valid for both models. The original D full-validation
scores use 585 patches, so they must not be directly subtracted from TD's scores.
The six-class and combined confusion matrices are in results.json, alongside
checkpoint, source-prediction and dataset hashes. Source artifacts remain intact.

TD was trained on 2,578 matched temporal patches; original D was trained on 2,609.
Normalization and sampling are fitted to the respective training sets. D uses
validated P05036 / 1.0.20 / 1.6.0 NISAR; TD uses provisional
P05023 / 1.0.11 / 1.5.0 observations from June–October, requiring at least three
jointly valid HH/HV dates per pixel/pass and averaging linear power before dB.
Thus this is the requested practical comparison against D, not an isolated causal
test of averaging. A same-footprint, same-processing all-sensor latest-date model
would be needed to isolate averaging. Original spatial exclusions do not certify
all provisional dates free of RFI. Both use the same seed-17 ResNet-34 U-Net recipe
and validation six-class macro-IoU checkpoint selection. Single-seed weak-reference
results do not establish statistical significance. The test set remains unscored.
"""
    (output/'report.md').write_text(text)
    chart = base64.b64encode((output/'comparison.png').read_bytes()).decode()
    table = '<table><tr><th>Model</th><th>Recall</th><th>Reference precision</th><th>F1</th><th>F2</th><th>IoU</th></tr>' + ''.join('<tr><td>'+label+'</td>'+''.join('<td>'+percent(w[k])+'</td>' for k in keys)+'</tr>' for label,w in zip(['D: static NISAR','TD: averaged NISAR'],wetland))+'</table>'
    (output/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Temporal D comparison</title><style>body{font:17px/1.6 system-ui;max-width:1050px;margin:40px auto;padding:0 20px}table{border-collapse:collapse;width:100%}th,td{padding:10px;text-align:left;border-bottom:1px solid #ddd}img{max-width:100%}pre{white-space:pre-wrap;font:14px/1.6 monospace}</style><h1>Averaged NISAR versus experiment D</h1>'+table+'<img alt="Matched validation comparison" src="data:image/png;base64,'+chart+'"><h2>Full analysis</h2><pre>'+html.escape(text)+'</pre>')
    (output/'complete.json').write_text(json.dumps(dict(results_sha256=sha256(output/'results.json'),report_sha256=sha256(output/'report.md'),test_evaluated=False),indent=2)+'\n')
    print(json.dumps(dict(output=str(output),delta=delta)),flush=True)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',required=True); p.add_argument('--temporal',required=True); p.add_argument('--output',required=True)
    args=p.parse_args(); compare(args.baseline,args.temporal,args.output)

if __name__ == '__main__':
    main()
