"""Build a self-contained final report from completed, frozen evaluations."""
import base64
import datetime as dt
import html
import json
from pathlib import Path
import numpy as np
from bogsignal.training.core import sha256
from bogsignal.training.temporal_d_compare import load_run

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / 'docs'
EX = DOCS / 'experiments'
SOURCES = []

def read(path):
    SOURCES.append(dict(path=str(path.relative_to(ROOT)), sha256=sha256(path)))
    return json.loads(path.read_text())

def image(path, caption):
    data = base64.b64encode(path.read_bytes()).decode()
    return '<figure><img alt="'+html.escape(caption,quote=True)+'" src="data:image/png;base64,'+data+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>'

def pct(x):
    return f'{100*x:.2f}%' if x is not None else '—'

def ranking_section(baseline, temporal):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def ranked(records):
        result = [r['recall_evaluation']['ranking']['Swamp or marsh'] for r in records]
        if len({(r['scored_pixels'],r['positive_pixels']) for r in result}) != 1:
            raise ValueError('Ranking comparison has unmatched scoring support')
        if any([b['budget_fraction'] for b in r['area_budgets']] != [.01,.05,.1,.2] for r in result):
            raise ValueError('Ranking budgets differ')
        return result
    base, temp = ranked(baseline), ranked(temporal)
    def ranking_table(records, labels):
        head = ['Model','Average precision','Recall at 1% area','Recall at 5% area','Recall at 10% area','Recall at 20% area','Reference precision at 5%']
        rows = []
        for r,label in zip(records,labels):
            vals = [label,pct(r['average_precision'])] + [pct(b['recall']) for b in r['area_budgets']] + [pct(r['area_budgets'][1]['precision'])]
            rows.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in vals)+'</tr>')
        return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+v+'</th>' for v in head)+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'
    fig,axes = plt.subplots(1,2,figsize=(12,4.7),sharey=True)
    for ax,records,labels,title in [(axes[0],base,['A: S2','B: S2 + S1','C: S2 + NISAR','D: all sensors'],'Original A–D validation area'),(axes[1],temp,['T0: latest','T1: mean','T2: mean + variability','TB: S2 + S1'],'Matched temporal validation area')]:
        for r,label in zip(records,labels):
            ax.plot([100*b['flagged_area_fraction'] for b in r['area_budgets']],[100*b['recall'] for b in r['area_budgets']],marker='o',label=label)
        ax.plot([0,20],[0,20],':',color='gray',label='Uniform random selection (expected)')
        ax.set(title=title,xlabel='Selected fraction of scored area (%)',xlim=(0,20.5),ylim=(0,100),xticks=[1,5,10,20])
        ax.grid(alpha=.2); ax.legend(fontsize=8,loc='lower right')
    axes[0].set_ylabel('Reference swamp/marsh recall (%)')
    fig.tight_layout(); path=EX/'final_candidate_ranking.png'; fig.savefig(path,dpi=150); plt.close(fig)
    d = base[3]; five = d['area_budgets'][1]
    text = '<section id="ranking"><h2>5. Candidate ranking for targeted review</h2><p>Ranking answers a different question: <strong>if we can inspect only a small fraction of the scored area, where should we look first?</strong> Use each pixel’s score <code>pSwamp + pMarsh</code>, order pixels from highest to lowest, and take the highest-scoring pixels within a fixed area budget. This produces a review-priority layer; it does not change the ordinary five-class inference map.</p><div class="summary"><h3>How to read D at a 5% area budget</h3><p>Inspecting the highest-scoring <strong>5% of the common validation area</strong> would recover <strong>'+pct(five['recall'])+'</strong> of its reference swamp/marsh pixels—about 73 of every 100 labelled wetland pixels. Of the selected pixels, <strong>'+pct(five['precision'])+'</strong> overlap the weak wetland labels, compared with '+pct(d['reference_prevalence'])+' prevalence over the whole scored area. The remaining selected pixels are not all verified errors: the reference labels are incomplete.</p><p>At 1% of area, D recovers '+pct(d['area_budgets'][0]['recall'])+' of reference wetland pixels; expanding review to 10% recovers '+pct(d['area_budgets'][2]['recall'])+'. More coverage costs more review effort and reduces reference precision.</p></div><h3>A–D: same original validation area</h3>'
    text += ranking_table(base,['A · S2','B · S2 + S1','C · S2 + NISAR','D · all sensors'])
    text += '<p><strong>D leads all four reported area budgets and average precision among A–D.</strong> Thus C’s higher ordinary-map recall does not make it the best model for limited-area candidate review. Average precision summarizes precision–recall across score thresholds; it is not pixel accuracy, and it is not precision at the chosen 5% budget.</p><h3>Temporal pilot: same matched temporal validation area</h3>'
    text += ranking_table(temp,['T0 · latest NISAR','T1 · mean NISAR','T2 · mean + variability','TB · static S2 + S1'])
    text += '<p>T2 has the highest average precision in this temporal panel, even though T1 has the higher ordinary-map recall. Ordering pixels well and assigning a wetland class at the fixed argmax decision are different uses of the scores. Compare ranking rows within each panel; the original and temporal panels have different eligible footprints and processing regimes.</p>'
    text += image(path,'Reference wetland recovery as the review-area budget grows. The dotted line is expected recall for uniformly random pixel selection, not a trained model.')
    text += '<p class="note">Budgets apply only to the fixed scored validation area, not the full AOI. Whole equal-score groups are retained only when they fit, so selected area never exceeds the declared budget. Average precision is exact, non-interpolated, and computed over distinct score thresholds. The 1%, 5%, 10% and 20% budgets were fixed for evaluation; no threshold was optimized for these tables. Scores are not established calibrated wetland probabilities. These pixel selections do not establish object-level site detection, independent discovery accuracy or restoration suitability.</p><p>TD’s completed analysis evaluated ordinary classification only; it has no saved candidate-ranking evaluation and is deliberately omitted here. No new inference or training was performed to add this section. The test set remains unscored.</p></section>'
    return text

def table(rows, labels):
    head = ['Model','Six-class macro IoU','Wetland recall','Reference precision','F1','F2','Wetland IoU']
    body = []
    for r, label in zip(rows,labels):
        w = r['combined_class_evaluation']['combined_class']['classes']['Wetland (Swamp + Marsh)'] if 'combined_class_evaluation' in r else r['combined_class']['classes']['Wetland (Swamp + Marsh)']
        vals = [label,pct(r['validation']['macro_iou'])]+[pct(w[k]) for k in ['recall','precision','dice','f2','iou']]
        body.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in vals)+'</tr>')
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+v+'</th>' for v in head)+'</tr></thead><tbody>'+''.join(body)+'</tbody></table></div>'

def examples():
    # Plot identical common validation pixels. Selection uses reference abundance,
    # never prediction quality, and no checkpoint inference is rerun here.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    old = load_run(ROOT/'data/experiments/D_s2s1nisar_resnet34_seed17')
    new = load_run(ROOT/'data/experiments/TD_s2s1_mean_nisar_resnet34_seed17')
    lookup = {name:i for i,name in enumerate(old['ids'])}
    counts = np.array([[np.count_nonzero((t==c)&v.astype(bool)) for c in [1,2,0]] for t,v in zip(new['target'],new['valid'])])
    selected = [int(np.argmax(counts[:,c])) for c in range(3)]
    stats = json.loads((Path(new['config']['dataset'])/'normalization.json').read_text())
    mean=np.asarray(stats['mean'])[:6,None,None]; std=np.asarray(stats['std'])[:6,None,None]
    images=np.load(new['cache']/'image.npy',mmap_mode='r')
    colors=['#b9b9aa','#197b49','#2879b9','#cf9a4b','#8b668b']
    names=['Other','Wetland (Swamp + Marsh)','Permanent water','Peatland cropland','Peatland non-cropland']
    palette=ListedColormap(colors); palette.set_bad('white')
    mapping=np.array([0,1,1,2,3,4],dtype='uint8')
    fig,axes=plt.subplots(3,4,figsize=(13,10))
    patch_ids=[]
    for row,j in enumerate(selected):
        patch_ids.append(new['ids'][j]); i=lookup[new['ids'][j]]
        keep=new['valid'][j].astype(bool)&(new['target'][j]!=255)
        if not np.array_equal(new['target'][j][keep],old['target'][i][keep]):
            raise ValueError('Example labels differ')
        physical=images[j,:6]*std+mean
        rgb=np.clip(physical[[2,1,0]].transpose(1,2,0)/.3,0,1)**.7
        axes[row,0].imshow(rgb)
        target=np.full((256,256),255,dtype='uint8'); target[keep]=mapping[new['target'][j][keep]]
        maps=[target,old['predictions'][1][i],new['predictions'][1][j]]
        for col,m in enumerate(maps,1):
            axes[row,col].imshow(np.ma.masked_where(~keep,m),cmap=palette,vmin=-.5,vmax=4.5,interpolation='nearest')
        axes[row,0].set_ylabel(['Swamp-rich reference','Marsh-rich reference','Other-rich reference'][row]+'\n'+new['ids'][j],fontsize=8)
        for col in range(4):
            axes[row,col].set(xticks=[],yticks=[])
            if row==0: axes[row,col].set_title(['Sentinel-2 RGB','Combined weak reference','D: static NISAR','TD: averaged NISAR'][col],fontsize=11)
    fig.legend(handles=[Patch(facecolor=c,label=n) for c,n in zip(colors,names)],loc='lower center',ncol=3,fontsize=9,frameon=False)
    fig.tight_layout(rect=(0,.06,1,1))
    dest=EX/'final_prediction_examples.png'; fig.savefig(dest,dpi=140); plt.close(fig)
    return dest,patch_ids

def main():
    basenames=['A_s2_resnet34_seed17','B_s2s1_resnet34_seed17','C_s2nisar_resnet34_seed17','D_s2s1nisar_resnet34_seed17']
    baseline=[read(EX/(n+'_metrics.json')) for n in basenames]
    sensor=read(EX/'sensor_comparison_seed17.json')
    temporal=read(EX/'temporal_provisional_seed17.json')
    paired=read(EX/'temporal_d_comparison_seed17/results.json')
    td=read(EX/'TD_s2s1_mean_nisar_resnet34_seed17_metrics.json')
    figure,patch_ids=examples()
    trained=baseline+temporal['records']+[td]
    runlabels=['A','B','C','D','T0','T1','T2','TB','TD']
    runtime_rows=''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in [name,r['epochs_completed'],r['best_epoch'],f"{r['training_seconds']/60:.2f}"] )+'</tr>' for name,r in zip(runlabels,trained))
    source_rows=''.join('<tr><td>'+html.escape(s['path'])+'</td><td><code>'+s['sha256']+'</code></td></tr>' for s in SOURCES)
    body = '''<header><p class="eyebrow">BogSignal / Argos · 15 November 2026</p><h1>Wetland segmentation:<br>sensor and temporal experiments</h1><p class="subtitle">Final report for the completed seed-17 experiments</p></header>
<section class="summary"><h2>Executive findings</h2><p><strong>NISAR added useful weak-reference signal in the controlled sensor comparison.</strong> Adding it to Sentinel-2 + Sentinel-1 increased combined swamp/marsh recall by 7.07 percentage points and IoU by 7.81 points. All-sensor model D had the strongest overlap scores among the original four models; model C had the highest recall.</p><p><strong>Temporal averaging did not provide a consistent improvement across configurations.</strong> It improved recall for Sentinel-2 + NISAR, but the all-sensor averaged-NISAR follow-up missed more reference wetland pixels than D.</p><p><strong>Recommendation:</strong> retain D as the balanced all-sensor baseline and C as the original recall-oriented alternative. Do not replace D with TD for a recall-focused use case on this evidence. Conclude the current experiment phase; further seeds and independent review belong to follow-up work.</p><p class="note">These are validation results against historical/inferred weak labels, not independent contemporary wetland accuracy. Every model uses one training seed. The test set remains unscored.</p></section>
<nav aria-label="Report contents"><a href="#design">Design</a><a href="#sensors">Sensor results</a><a href="#temporal">Temporal results</a><a href="#all-sensor">D vs averaged NISAR</a><a href="#ranking">Candidate ranking</a><a href="#examples">Examples</a><a href="#limits">Limits</a><a href="#artifacts">Artifacts</a></nav>
<section id="design"><h2>1. Study question and fixed design</h2><p>The objective was to classify each pixel in a patch and assess the contribution of Sentinel-1 and NISAR beyond Sentinel-2, followed by a temporal NISAR pilot. The model predicts six classes: Other, Swamp, Marsh, Permanent water, Peatland cropland and Peatland non-cropland. Peatland classes and permanent water are kept separate from the combined swamp/marsh target.</p><div class="table-wrap"><table><tr><th>Item</th><th>Specification</th></tr><tr><td>Base dataset</td><td>V4: 3,753 patches; 2,609 train / 585 validation / 559 locked test</td></tr><tr><td>Patch grid</td><td>256 × 256 pixels; 20 m; EPSG:3035; geographic folds and buffers</td></tr><tr><td>Optical channels</td><td>Sentinel-2 B2, B3, B4, B8, B11, B12</td></tr><tr><td>Radar channels</td><td>S1 ascending/descending VV, VH; NISAR ascending/descending HH, HV</td></tr><tr><td>Model</td><td>U-Net, ImageNet-pretrained ResNet-34 encoder; six logits; approximately 24.5M parameters</td></tr><tr><td>Inputs</td><td>Selected physical channels plus matching validity masks; D and TD: 14 physical + 14 validity = 28 model channels</td></tr><tr><td>Optimization</td><td>Seed 17; batch 16; AdamW; encoder LR 10⁻⁴, decoder LR 10⁻³; weight decay 10⁻⁴; cosine decay; BF16</td></tr><tr><td>Training and selection</td><td>Cross-entropy, ignore 255; 75/25 minority/background weighted sampling; flips/quarter turns; maximum 50 epochs, minimum 15, patience 10; best validation six-class macro IoU</td></tr></table></div><p>Normalization is fitted on training data only. Missing normalized predictors are zero-filled, with explicit validity channels. Validation is uniform and unaugmented. Sensor experiments A–D share identical patch IDs, labels and evaluation masks; their configurations differ only by sensor selection, name and output location.</p><h3>How to read the metrics</h3><p>For combined wetland inference, sum the Swamp and Marsh probabilities first, then choose among Other, combined Wetland, Permanent water, Peatland cropland and Peatland non-cropland. This is an inference-only merge of a six-class model; no five-class retraining was performed. Every eligible pixel receives a class. There is no ranking, area budget or tuned threshold in the main tables.</p><ul><li><strong>Recall:</strong> fraction of reference swamp/marsh pixels recovered. A Swamp–Marsh subtype swap is accepted in the combined target.</li><li><strong>Reference precision:</strong> fraction of predicted swamp/marsh pixels overlapping the weak positive labels. Missing wetland labels can depress this value.</li><li><strong>F2:</strong> combines precision and recall, weighting recall more heavily. Higher recall alone is insufficient.</li><li><strong>F1 and IoU:</strong> measure combined-class agreement/overlap. Six-class macro IoU separately averages the original six classes.</li></ul></section>'''
    body += '<section id="sensors"><h2>2. Controlled sensor comparison: A–D</h2><p>Common evaluation: <strong>585 validation patches and 37,411,425 pixels</strong>. These four rows are directly comparable within this panel.</p>'
    body += table(baseline,['A · Sentinel-2','B · Sentinel-2 + Sentinel-1','C · Sentinel-2 + NISAR','D · Sentinel-2 + Sentinel-1 + NISAR'])
    body += '''<p><strong>Primary NISAR contrast, D − B:</strong> recall +7.07 points, reference precision +14.11, F2 +8.24 and wetland IoU +7.81; six-class macro IoU +1.10. These results support a useful NISAR contribution in this run, without establishing statistical significance.</p><p><strong>Objective matters:</strong> C reaches 45.09% recall, while D reaches 39.62% recall with much higher reference precision (48.23% versus 33.90%) and higher IoU (27.80% versus 24.00%). D is the balanced baseline; C remains the original recall-oriented alternative. Adding S1 to C did not improve recall.</p>'''
    body += image(EX/'sensor_comparison_seed17.png','A–D: controlled sensor comparison on the full common V4 validation area.')+'</section>'
    body += '<section id="temporal"><h2>3. Temporal NISAR pilot</h2><p>The pilot acquired <strong>120 provisional GCOV products across 24 coverage groups</strong>: one selected observation per month from June through October. The selected window is 18 June–20 October 2026. Ascending/descending passes and anchor tracks/frames remain separate. At least three jointly valid HH/HV observations are required per pixel and pass; valid pixels use three, four or five dates.</p><p>T0 uses the latest valid date; T1 averages <strong>linear radar power before conversion to dB</strong>; T2 adds clipped-dB population standard deviation and p10/p90 from the same dates. S2 and S1 stay static. T0/T1/T2 select S2 + NISAR channels; TB is a matched static S2 + S1 reference. TB is not a pure test of incremental NISAR because the sensor combinations differ.</p><p>Matched dataset: <strong>2,578 train / 543 validation / 525 locked test</strong>; 107 original candidate patches were excluded by declared validity rules. All four rows below share identical validation IDs, labels and masks: <strong>34,690,287 scored pixels</strong>. Do not directly subtract these rows from the original A–D table: the eligible footprint and processing regime changed.</p>'
    body += table(temporal['records'],['T0 · S2 + latest NISAR','T1 · S2 + mean NISAR','T2 · S2 + mean and variability NISAR','TB · matched static S2 + S1'])
    body += '<p><strong>Averaging, T1 − T0:</strong> recall +6.11 points, precision −1.72, F2 +4.52 and IoU +1.76. T1 is the best recall-oriented representation within this matched temporal pilot. Adding variability features raised precision but reduced recall and F2. Coarse percentiles from three to five dates do not establish annual hydrological extremes.</p>'
    body += image(EX/'temporal_provisional_seed17.png','Temporal variants: only compare rows within the matched temporal evaluation area.')+'</section>'
    body += '<section id="all-sensor"><h2>4. All-sensor follow-up: averaged NISAR versus D</h2><p>TD uses <strong>static Sentinel-2 + static Sentinel-1 + averaged NISAR</strong>. It trained for 41 epochs, selecting epoch 31. The existing D checkpoint was reused and rescored on the same 543 validation patches and 34,690,287 valid pixels as TD. Patch IDs, common labels and baseline validity were checked; no D retraining or checkpoint reselection was performed.</p>'
    body += table([paired['baseline'],paired['temporal']],['D · original NISAR, common area','TD · averaged NISAR, common area'])
    body += '<p><strong>TD − D:</strong> recall −4.75 points, precision +6.66, F2 −3.52 and wetland IoU −0.78. Six-class macro IoU was effectively unchanged (+0.02 points). TD is more conservative: it has higher reference precision but misses more labelled wetland pixels. This run does not support replacing D for recall-focused use.</p><p class="note">This is a practical comparison against the existing baseline, not an isolated test of averaging. D trained on 2,609 patches with validated NISAR P05036 / 1.0.20 / 1.6.0; TD trained on 2,578 patches with provisional P05023 / 1.0.11 / 1.5.0. Dates, processing maturity, training normalization and sampling populations differ. A same-footprint, same-processing all-sensor latest-date run would be needed to isolate averaging.</p><p>D’s numbers here differ slightly from section 2 because it is rescored on a smaller shared validation area, not because D was changed.</p>'
    body += image(EX/'temporal_d_comparison_seed17/comparison.png','Original D versus TD, evaluated on identical pixels.')+'</section>'
    body += ranking_section(baseline, temporal['records'])
    body += '<section id="examples"><h2>6. Matched prediction examples</h2><p>The following patches are selected by the largest reference Swamp, Marsh and Other pixel counts on the common validation mask—not by model performance. They are illustrations, not a representative accuracy sample. Both predictions use combined-class inference. White pixels are excluded from scoring. RGB uses the same fixed display stretch.</p>'
    body += image(figure,'Same three locations, same reference labels, same valid pixels: original D and averaged-NISAR TD.')
    body += '<p class="small">Patch IDs: '+', '.join('<code>'+p+'</code>' for p in patch_ids)+'.</p><p>The examples show where classes extend beyond or fail to reproduce the weak labels. Image appearance alone cannot determine which disagreement is a model error: independent contemporary annotation is needed.</p></section>'
    body += '''<section id="limits"><h2>7. Interpretation, limitations and stopping point</h2><ul><li><strong>Weak references:</strong> labels are historical/inferred and can be incomplete, wrong or temporally mismatched. Other is not verified dry land. Precision and recall both describe agreement with these references.</li><li><strong>One seed:</strong> no repeated-seed uncertainty or spatial confidence intervals were estimated. Millions of correlated pixels do not constitute millions of independent samples. Small differences should remain inconclusive.</li><li><strong>Validation reuse:</strong> checkpoints and subsequent representations were considered using validation performance. Results are exploratory model-development findings, not a final generalization estimate. The locked test has not been scored.</li><li><strong>Temporal quality:</strong> provisional NISAR maturity, residual RFI, irregular timing and only three to five valid dates limit interpretation. Existing spatial exclusions do not certify every provisional date as artifact-free.</li><li><strong>Task scope:</strong> combined swamp/marsh detection is easier than correct subtype classification. These experiments do not establish restoration suitability, object-level site detection or independent discovery accuracy.</li><li><strong>Numerical checks:</strong> frozen-checkpoint inference preserved hashes and scoring support and passed the 0.1-percentage-point engineering tolerance. BF16/cuDNN benchmarking is not bitwise deterministic; that tolerance is not statistical uncertainty.</li></ul><p><strong>Final conclusion:</strong> the controlled baseline experiments indicate useful additional NISAR information. Simple temporal averaging helped the S2 + NISAR configuration, but the improvement did not carry over to the all-sensor comparison against D. The defensible outcome is a working baseline and a documented precision–recall tradeoff, not a universal temporal-model improvement.</p><p><strong>Stop new experiments for this submission.</strong> Preserve D, C and the temporal checkpoints and report the observed results and limits. After the deadline, prioritize repeated seeds, independent spatial review and a matched-processing all-sensor temporal control before more complex architectures. No new training or test evaluation was performed to build this report.</p></section>'''
    body += '<section id="artifacts"><h2>8. Reproducibility and retained artifacts</h2><p>All nine trained checkpoints are retained with frozen configurations, dataset hashes, learning curves and evaluation artifacts. The new comparison checks align patch IDs and reject changed labels or pixels invalid for D. The repository test suite passed <strong>56 tests</strong>. Retention is confirmed; this report does not claim that an off-machine backup was created.</p><div class="table-wrap"><table><tr><th>Run</th><th>Epochs completed</th><th>Best epoch</th><th>Recorded training minutes</th></tr>'+runtime_rows+'</table></div><p class="small">Times exclude dataset acquisition, preprocessing, cache creation and deferred analysis; the original A timing includes its immediate checkpoint verification. Timing is operational context, not a controlled speed benchmark.</p><p>Detailed source reports are in <code>docs/experiments/</code>; methods are in <code>docs/temporal_experiment.md</code>, <code>docs/dataset_creation.md</code> and <code>ml/TRAINING.md</code>. Rebuild this HTML from <code>ml</code> with <code>uv run --locked --extra training python build_final_report.py</code>. A machine-readable summary accompanies the HTML as <code>final_experiment_report_data.json</code>.</p><details><summary>Source evaluation files and SHA-256 hashes</summary><div class="table-wrap"><table><tr><th>Source</th><th>SHA-256</th></tr>'+source_rows+'</table></div></details></section>'
    css='''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f1f4f5;color:#1a2b33;font:16px/1.65 system-ui,-apple-system,Segoe UI,sans-serif}main{max-width:1160px;margin:32px auto;background:white;padding:44px 48px;border-radius:12px}h1{font-size:38px;line-height:1.18;letter-spacing:-1px}h2{font-size:25px;line-height:1.35;margin-top:0}h3{font-size:19px}header{border-bottom:3px solid #20836a;margin-bottom:28px;padding-bottom:16px}.eyebrow{color:#366d64;letter-spacing:.1em;text-transform:uppercase;font-size:12px;font-weight:700}.subtitle{font-size:20px;color:#53646c}.summary{background:#ecf6f1;border-left:5px solid #20836a;padding:24px!important;border-radius:6px}.note{border-left:3px solid #ba913e;background:#fff9e9;padding:12px 16px}nav{display:flex;flex-wrap:wrap;gap:10px 22px;margin:28px 0}a{color:#176b58}section{padding-top:20px;margin:25px 0 38px}table{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}td,th{padding:11px 10px;text-align:left;border-bottom:1px solid #dce5e5;vertical-align:top}th{background:#eaf0f0;font-weight:650}tbody tr:nth-child(even){background:#f7fafa}.table-wrap{overflow-x:auto;margin:16px 0}figure{margin:24px 0}img{display:block;width:100%;height:auto}figcaption,.small{font-size:13px;color:#52646c}figcaption{margin-top:9px}li{margin-bottom:10px}code{font:12px/1.6 ui-monospace,Consolas,monospace;overflow-wrap:anywhere}details{margin:20px 0}summary{cursor:pointer;font-weight:600}@media(max-width:700px){main{margin:0;padding:24px 16px;border-radius:0}h1{font-size:30px}h2{font-size:22px}}@media print{body{background:white;font-size:10pt}main{margin:0;padding:0;max-width:none}nav{display:none}section{break-before:auto}tr,img{break-inside:avoid}h2,h3{break-after:avoid}.table-wrap{overflow:visible}details{display:none}a{color:inherit}}'''
    output=DOCS/'final_experiment_report.html'
    output.write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Wetland segmentation — final experiment report</title><style>'+css+'</style></head><body><main>'+body+'</main></body></html>')
    summary=dict(generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),seed=17,test_evaluated=False,
        source_evaluations=SOURCES,baseline=[dict(experiment=r['experiment'],validation=r['validation'],combined_class=r['combined_class_evaluation']['combined_class']) for r in baseline],
        temporal=[dict(experiment=r['experiment'],validation=r['validation'],combined_class=r['combined_class_evaluation']['combined_class']) for r in temporal['records']],
        candidate_ranking=dict(baseline=[r['recall_evaluation']['ranking']['Swamp or marsh'] for r in baseline], temporal=[r['recall_evaluation']['ranking']['Swamp or marsh'] for r in temporal['records']], TD_status='not_evaluated'),
        all_sensor_common_area=paired,example_patch_ids=patch_ids,html_sha256=sha256(output),generator_sha256=sha256(__file__))
    (DOCS/'final_experiment_report_data.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(report=str(output),bytes=output.stat().st_size,embedded_figures=5,example_patch_ids=patch_ids)),flush=True)

if __name__=='__main__':
    main()


