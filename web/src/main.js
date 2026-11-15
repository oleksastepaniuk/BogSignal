import { setupLandscape } from './landscape.js';
import './style.css';

// Highlight the nav link for the section currently in view.
const links = document.querySelectorAll('.nav a');
const observer = new IntersectionObserver(
  (entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      for (const link of links) {
        link.classList.toggle('active', link.hash === `#${entry.target.id}`);
      }
    }
  },
  { rootMargin: '-50% 0px -50% 0px' },
);
document.querySelectorAll('main section').forEach((s) => observer.observe(s));

const pct = x => (100*x).toFixed(1)+'%';
const $ = id => document.getElementById(id);
async function setupDemo() {
  const response = await fetch('/experiment/demo.json');
  if (!response.ok) throw new Error('Saved data unavailable');
  const data = await response.json();
  for(const card of data.sensor_cards) {
    $('sensor-'+card.sensor).src='/experiment/'+card.filename;
    $('sensor-'+card.sensor).alt=card.title+', validation patch '+card.patch_id;
    $('sensor-'+card.sensor+'-caption').textContent=card.title+' / '+card.patch_id+'. Mosaic source window: '+card.source_window.join(' to ')+'. '+card.display+'.';
  }

  $('demo').innerHTML = `
    <fieldset><legend>Choose a validation example</legend><div class="choices" id="examples">
    <button data-key="helps" aria-pressed="true">Where NISAR helped</button>
    <button data-key="little" aria-pressed="false">Made little difference</button>
    <button data-key="worsens" aria-pressed="false">Worked less well</button></div></fieldset>
    <p id="patch-caption" class="note" aria-live="polite"></p>
    <div class="context-grid" aria-label="Weak labels, B prediction, D prediction">
    <figure><img id="reference" width="256" height="256" alt=""><figcaption>Weak labels<br>Historical / inferred</figcaption></figure>
    <figure><img id="panel-b" width="256" height="256" alt=""><figcaption>B: Sentinel-2 + Sentinel-1</figcaption></figure>
    <figure><img id="panel-d" width="256" height="256" alt=""><figcaption>D: all sensors + NISAR</figcaption></figure></div>
    <p id="class-legend" class="legend"></p>
    <h3>Swipe to compare B and D</h3>
    <p class="note swipe-key"><span aria-hidden="true"></span>Yellow = predicted Swamp + Marsh. Other classes are hidden in this overlay.</p>
    <figure><div class="comparison" id="comparison">
    <img id="satellite" width="512" height="512" alt="" draggable="false">
    <img id="map-b" width="256" height="256" alt="" draggable="false"><img id="map-d" width="256" height="256" alt="" draggable="false">
    <div id="swipe-line" role="slider" tabindex="0" aria-label="Swipe boundary: B left, D right" aria-valuemin="0" aria-valuemax="100" aria-valuenow="50" aria-orientation="horizontal"><span aria-hidden="true">&#8596;</span></div>
    </div><figcaption>B: Sentinel-2 + Sentinel-1 (left) / D: all sensors, including NISAR (right). 20 m predictions over matching-source 10 m Sentinel-2 imagery. Drag the boundary or use arrow keys.</figcaption></figure>
    <p class="note">256 x 256 pixels / 20 m grid / identical scoring mask. Top panels: white = excluded. Swipe: yellow 20 m swamp/marsh predictions over original 10 m Sentinel-2 RGB; all other and excluded pixels are transparent. Selected illustrations of weak-reference agreement, not independently verified outcomes.</p>`;
  let example=data.examples[0];
  let position=50;
  const boundary=$('swipe-line');
  function renderSwipe() {
    const value=position;
    $('map-b').style.clipPath=`inset(0 ${100-value}% 0 0)`;
    $('map-d').style.clipPath=`inset(0 0 0 ${value}%)`;
    boundary.style.left=value+'%';
    boundary.setAttribute('aria-valuenow',String(value));
    boundary.setAttribute('aria-valuetext',`B ${value}%, D ${100-value}%`);
  }
  $('class-legend').innerHTML='<span style="--swatch:#b9b9aa">Other</span><span style="--swatch:#197b49">Swamp + Marsh</span><span style="--swatch:#2879b9">Water</span><span style="--swatch:#cf9a4b">Peatland cropland</span><span style="--swatch:#8b668b">Peatland non-cropland</span>';
  let dragId=null;
  function moveBoundary(event) {
    const rect=$('comparison').getBoundingClientRect();
    position=Math.round(Math.max(0,Math.min(100,(event.clientX-rect.left)/rect.width*100)));
    renderSwipe();
  }
  boundary.addEventListener('pointerdown',event=>{
    if (!event.isPrimary || event.button!==0) return;
    event.preventDefault();
    dragId=event.pointerId;
    boundary.setPointerCapture(dragId);
    boundary.focus({preventScroll:true});
    moveBoundary(event);
  });
  boundary.addEventListener('pointermove',event=>{
    if(event.pointerId===dragId) moveBoundary(event);
  });
  for(const type of ['pointerup','pointercancel','lostpointercapture']) {
    boundary.addEventListener(type,event=>{
      if(event.pointerId!==dragId) return;
      dragId=null;
      if(boundary.hasPointerCapture(event.pointerId)) boundary.releasePointerCapture(event.pointerId);
    });
  }
  boundary.addEventListener('keydown',event=>{
    const value=position;
    const next={ArrowLeft:value-1,ArrowDown:value-1,ArrowRight:value+1,ArrowUp:value+1,PageDown:value-10,PageUp:value+10,Home:0,End:100}[event.key];
    if(next===undefined) return;
    event.preventDefault();
    position=Math.max(0,Math.min(100,next));
    renderSwipe();
  });
  function renderExample() {
    for (const [id,suffix,label] of [['panel-b','b','B prediction'],['panel-d','d','D prediction'],['reference','reference','Weak reference'],['satellite','rgb','Matching Sentinel-2 satellite background'],['map-b','b-overlay','B prediction overlay'],['map-d','d-overlay','D prediction overlay']]) {
      $(id).src=`/experiment/${example.key}-${suffix}.png`;
      $(id).alt=`${label}, validation patch ${example.id}`;
    }
    const delta=100*example.delta;
    $('patch-caption').textContent=`${example.id} / Wetland IoU: B ${pct(example.b.iou)} to D ${pct(example.d.iou)} (${delta>=0?'+':''}${delta.toFixed(1)} percentage points).`;
    renderSwipe();
  }
  document.querySelectorAll('#examples button').forEach(button=>button.addEventListener('click',()=>{
    example=data.examples.find(row=>row.key===button.dataset.key);
    document.querySelectorAll('#examples button').forEach(item=>item.setAttribute('aria-pressed',String(item===button)));
    renderExample();
  }));
  renderExample();
  const wetlandB=data.baseline[1].combined_class.classes['Wetland (Swamp + Marsh)'];
  const wetlandD=data.baseline[3].combined_class.classes['Wetland (Swamp + Marsh)'];
  $('sensor-finding').innerHTML=`<div class="finding-card">
    <h3>01 &mdash; NISAR improved wetland overlap</h3>
    <p class="finding-number">+${(100*(wetlandD.iou-wetlandB.iou)).toFixed(1)}<span>percentage points in wetland overlap</span></p>
    <p class="finding-comparison"><span>Without NISAR <strong>${pct(wetlandB.iou)}</strong></span><span aria-hidden="true">&rarr;</span><span>With NISAR <strong>${pct(wetlandD.iou)}</strong></span></p>
  </div>`;
  $('ranking').innerHTML=`<figure class="budget-chart" aria-labelledby="budget-chart-title">
    <figcaption id="budget-chart-title"><strong>How much of the reference wetland can we find?</strong><span>Combined model with NISAR / highest-scoring areas first</span></figcaption>
    <div class="budget-plot"><div class="budget-axis" aria-hidden="true"><span>100%</span><span>50%</span><span>0%</span></div>
    <div class="budget-bars">${data.budgets.map(r=>`<div class="budget-column ${r.budget_fraction===.05?'highlight':''}" data-budget="${r.budget_fraction}">
      <div class="budget-track"><div class="budget-bar" style="height:${100*r.recall}%"><strong>${pct(r.recall)}</strong></div></div>
      <span class="budget-label">${100*r.budget_fraction}%</span>
    </div>`).join('')}</div>
    </div><p class="budget-x-label">Share of scored area reviewed, highest scores first</p>
    <p class="note">Share of historical and inferred reference swamp/marsh pixels captured at each saved review budget. Area means pixels, not a number of sites; values are not interpolated.</p>
  </figure>`;
  const sensors=['S2','S2 + S1','S2 + NISAR','S2 + S1 + NISAR'];
  $('results').innerHTML=`<table><caption>Combined swamp/marsh inference / validation agreement (%)</caption>
    <thead><tr><th scope="col">Inputs</th><th scope="col">IoU</th><th scope="col">Recall</th><th scope="col">Reference precision</th><th scope="col">F1</th></tr></thead>
    <tbody>${data.baseline.map((r,i)=>{const w=r.combined_class.classes['Wetland (Swamp + Marsh)']; return `<tr ${i===3?'class="best-row"':''}><th scope="row">${'ABCD'[i]} / ${sensors[i]}</th>${['iou','recall','precision','dice'].map(k=>`<td>${pct(w[k])}</td>`).join('')}</tr>`;}).join('')}</tbody>
    <tbody><tr class="temporal-heading"><th colspan="5" scope="colgroup">Temporal follow-up / 543 matched validation patches, 34,690,287 pixels. Each cell shows TD, with D rescored on the same area below.</th></tr>
    <tr><th scope="row">TD / all sensors + averaged NISAR<small>Separate matched-area comparison</small></th>
    ${['iou','recall','precision','dice'].map(key=> {
      const td=data.temporal.temporal.combined_class.classes['Wetland (Swamp + Marsh)'];
      const matched=data.temporal.baseline.combined_class.classes['Wetland (Swamp + Marsh)'];
      return `<td>${pct(td[key])}<small>D matched: ${pct(matched[key])}</small></td>`;
    }).join('')}</tr></tbody></table>`;
}
setupDemo().catch(error=>{
  $('demo').textContent='Saved comparison could not load. Reload or open the complete report in Methods & Sources.';
  console.error(error);
});

setupLandscape().catch(error=>{document.getElementById('map-status').textContent='The regional map could not load. Please reload the page.';console.error(error);});
