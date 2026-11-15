// "A season in motion": NISAR 12-day interferometric coherence, June to October 2026.
// Data and images are written by web/scripts/export_change.py.

const $ = id => document.getElementById(id);
const BASE = (import.meta.env?.BASE_URL ?? '/') + 'change/';
const fmtDate = iso => new Date(iso + 'T00:00:00Z').toLocaleDateString('en-GB', { day: 'numeric', month: 'short', timeZone: 'UTC' });
const SERIES = [
  { key: 'wetland', label: 'Swamp + marsh', color: '#197b49', dash: '', classes: ['Swamp', 'Marsh'] },
  { key: 'noncrop', label: 'Peatland, not farmed', color: '#8b668b', dash: '6 3', classes: ['Peatland non-cropland'] },
  { key: 'crop', label: 'Peatland cropland', color: '#cf9a4b', dash: '2 3', classes: ['Peatland cropland'] },
  { key: 'water', label: 'Open water', color: '#2879b9', dash: '8 3 2 3', classes: ['Permanent water'] },
];
const CONTEXT = {
  labels: { label: 'Land type', caption: 'Historical / inferred land types (weak reference labels), 80 m.' },
  s2: { label: 'Sentinel-2 colour', caption: 'Sentinel-2 true colour from the dataset mosaic. Some cloud remains.' },
  hh: { label: 'NISAR HH', caption: 'NISAR L-band backscatter, HH polarization (dB). Brighter = stronger return.' },
  hv: { label: 'NISAR HV', caption: 'NISAR L-band backscatter, HV polarization (dB). Brighter = more scattering inside vegetation.' },
  pol: { label: 'HH + HV', caption: 'Polarization composite: red = HH, green = HV, blue = HH minus HV.' },
};

// pixel-weighted mean coherence of a group of classes, over the whole radar frame
function groupMean(pair, classes, weights) {
  let sum = 0, n = 0;
  for (const c of classes) {
    const v = pair.frame_coherence[c];
    if (v == null) continue;
    sum += v * weights[c]; n += weights[c];
  }
  return n ? sum / n : null;
}

function chart(data, values) {
  const W = 720, H = 300, m = { l: 44, r: 150, t: 16, b: 46 };
  const all = values.flatMap(s => s.v);
  const lo = Math.floor(Math.min(...all) * 10) / 10, hi = Math.ceil(Math.max(...all) * 10) / 10;
  const x = i => m.l + (W - m.l - m.r) * i / (data.pairs.length - 1);
  const y = v => m.t + (H - m.t - m.b) * (1 - (v - lo) / (hi - lo));
  const ticks = [];
  for (let t = lo; t <= hi + 1e-9; t += 0.1) ticks.push(+t.toFixed(1));
  // keep end labels from colliding
  const ends = values.map(s => ({ s, y: y(s.v.at(-1)) })).sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 15) ends[i].y = ends[i - 1].y + 15;
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="season-chart-title" id="season-svg">
    ${ticks.map(t => `<line x1="${m.l}" x2="${W - m.r}" y1="${y(t)}" y2="${y(t)}" class="grid"/><text x="${m.l - 8}" y="${y(t) + 4}" text-anchor="end" class="tick">${t.toFixed(1)}</text>`).join('')}
    ${data.pairs.map((p, i) => `<text x="${x(i)}" y="${H - m.b + 18}" text-anchor="middle" class="tick">${fmtDate(p.start)}</text>${p.days !== 12 ? `<text x="${x(i)}" y="${H - m.b + 32}" text-anchor="middle" class="tick flag">${p.days} days</text>` : ''}`).join('')}
    <line id="season-cursor" class="cursor" y1="${m.t}" y2="${H - m.b}" x1="${x(0)}" x2="${x(0)}"/>
    ${values.map(s => `<polyline fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round" stroke-dasharray="${s.dash}" points="${s.v.map((v, i) => `${x(i)},${y(v)}`).join(' ')}"/>
      ${s.v.map((v, i) => `<circle cx="${x(i)}" cy="${y(v)}" r="4" fill="${s.color}" stroke="#f3f6f0" stroke-width="2"/>`).join('')}`).join('')}
    ${ends.map(e => `<text x="${W - m.r + 10}" y="${e.y + 4}" class="end"><tspan fill="${e.s.color}">&#9632;</tspan> ${e.s.label}</text>`).join('')}
    ${data.pairs.map((_, i) => `<rect class="hit" data-i="${i}" x="${x(i) - (W - m.l - m.r) / (data.pairs.length - 1) / 2}" y="${m.t}" width="${(W - m.l - m.r) / (data.pairs.length - 1)}" height="${H - m.t - m.b}"/>`).join('')}
  </svg>`;
}

export async function setupChange() {
  const response = await fetch(BASE + 'change.json');
  if (!response.ok) throw new Error('Season data unavailable');
  const data = await response.json();
  const weights = data.frame_labelled_pixels_80m;
  const values = SERIES.map(s => ({ ...s, v: data.pairs.map(p => groupMean(p, s.classes, weights)) }));
  const mean = key => { const v = values.find(s => s.key === key).v; return v.reduce((a, b) => a + b, 0) / v.length; };
  const n = data.pairs.length;
  const b = data.window_backscatter;

  $('change-app').innerHTML = `
    <div class="season-controls">
      <button id="season-play" type="button" aria-pressed="false">Play</button>
      <label for="season-slider">NISAR pass pair</label>
      <input id="season-slider" type="range" min="0" max="${n - 1}" step="1" value="0">
      <output id="season-date" for="season-slider"></output>
    </div>
    <div class="season-grid">
      <figure><img id="season-change" width="${data.size_px}" height="${data.size_px}" alt="">
        <figcaption><strong>Surface change between two NISAR passes</strong>
        <span class="ramp" aria-hidden="true"></span><span class="ramp-ends"><span>stayed the same</span><span>changed</span></span></figcaption></figure>
      <figure><img id="season-context" width="${data.size_px}" height="${data.size_px}" alt="">
        <figcaption><div class="choices" id="season-layers">${Object.entries(CONTEXT).map(([k, c], i) => `<button type="button" data-key="${k}" aria-pressed="${i === 0}">${c.label}</button>`).join('')}</div>
        <span id="season-context-caption"></span></figcaption></figure>
    </div>
    <p class="legend"><span style="--swatch:#197b49">Swamp + Marsh</span><span style="--swatch:#2879b9">Water</span><span style="--swatch:#cf9a4b">Peatland cropland</span><span style="--swatch:#8b668b">Peatland non-cropland</span><span style="--swatch:#b9b9aa">Other</span></p>
    <p class="note">${data.window_km.toFixed(1)} km square near ${data.center[0].toFixed(2)}&deg;N, ${data.center[1].toFixed(2)}&deg;E (<a href="https://www.openstreetmap.org/#map=11/${data.center[0].toFixed(3)}/${data.center[1].toFixed(3)}">open in OpenStreetMap</a>), ${data.pixel_m} m pixels, HH polarization. Each frame compares two passes; darker means the radar echo decorrelated more between them. The square was chosen automatically as the fully observed area with the most balanced mix of reference wetland and peatland cropland.</p>
    <figure class="season-chart">
      <figcaption id="season-chart-title"><strong>How steady did each land type stay?</strong><span>Mean coherence between consecutive NISAR passes, whole radar frame. 1 = unchanged, 0 = completely changed. Dates mark the first pass of each pair.</span></figcaption>
      <div class="season-plot">${chart(data, values)}<div id="season-tip" class="season-tip" hidden></div></div>
      <details><summary>Show as table</summary><div class="table-wrap"><table><thead><tr><th scope="col">Passes</th>${values.map(s => `<th scope="col">${s.label}</th>`).join('')}</tr></thead>
      <tbody>${data.pairs.map((p, i) => `<tr><th scope="row">${fmtDate(p.start)} &ndash; ${fmtDate(p.end)}${p.days !== 12 ? ` (${p.days} days)` : ''}</th>${values.map(s => `<td>${s.v[i].toFixed(2)}</td>`).join('')}</tr>`).join('')}</tbody></table></div></details>
      <p class="note">Land types are historical / inferred reference labels on the 80 m radar grid, not field observations. One pair spans ${Math.max(...data.pairs.map(p => p.days))} days instead of 12; longer gaps lower coherence by themselves.</p>
    </figure>`;

  $('season-wetland').textContent = mean('wetland').toFixed(2);
  $('season-crop').textContent = mean('crop').toFixed(2);
  $('season-water').textContent = mean('water').toFixed(2);
  $('season-crop-share').textContent = (100 * data.window_class_share['Peatland cropland']).toFixed(0) + '%';
  if (b.Swamp && b['Peatland cropland']) {
    $('season-hh-gap').textContent = (b.Swamp.hh_db - b['Peatland cropland'].hh_db).toFixed(1);
    $('season-hv-gap').textContent = (b.Swamp.hv_db - b['Peatland cropland'].hv_db).toFixed(1);
  }

  for (const p of data.pairs) new Image().src = BASE + p.image; // preload so the animation does not flicker
  const slider = $('season-slider'), play = $('season-play'), tip = $('season-tip'), svg = $('season-svg');
  const hits = [...svg.querySelectorAll('.hit')];
  function show(i) {
    const p = data.pairs[i];
    slider.value = i;
    const text = `${fmtDate(p.start)} to ${fmtDate(p.end)} 2026 (${p.days} days apart)`;
    $('season-date').textContent = text;
    slider.setAttribute('aria-valuetext', text);
    $('season-change').src = BASE + p.image;
    $('season-change').alt = `NISAR surface change map, ${text}`;
    const cx = hits[i].x.baseVal.value + hits[i].width.baseVal.value / 2;
    $('season-cursor').setAttribute('x1', cx); $('season-cursor').setAttribute('x2', cx);
  }
  let timer = null;
  function stop() { clearInterval(timer); timer = null; play.textContent = 'Play'; play.setAttribute('aria-pressed', 'false'); }
  play.addEventListener('click', () => {
    if (timer) return stop();
    play.textContent = 'Pause'; play.setAttribute('aria-pressed', 'true');
    timer = setInterval(() => show((+slider.value + 1) % n), 1100);
  });
  slider.addEventListener('input', () => { stop(); show(+slider.value); });
  for (const hit of hits) {
    const i = +hit.dataset.i;
    hit.addEventListener('click', () => { stop(); show(i); });
    hit.addEventListener('pointerenter', () => {
      const p = data.pairs[i];
      tip.innerHTML = `<strong>${fmtDate(p.start)} &ndash; ${fmtDate(p.end)}</strong>${values.map(s => `<span><i style="background:${s.color}"></i>${s.label}<b>${s.v[i].toFixed(2)}</b></span>`).join('')}`;
      tip.hidden = false;
      const left = (hit.x.baseVal.value + hit.width.baseVal.value / 2) / 720 * 100;
      tip.style.left = Math.min(left, 62) + '%';
    });
    hit.addEventListener('pointerleave', () => { tip.hidden = true; });
  }
  function context(key) {
    $('season-context').src = `${BASE}${key}.png`;
    $('season-context').alt = CONTEXT[key].caption;
    $('season-context-caption').textContent = CONTEXT[key].caption;
    document.querySelectorAll('#season-layers button').forEach(btn => btn.setAttribute('aria-pressed', String(btn.dataset.key === key)));
  }
  document.querySelectorAll('#season-layers button').forEach(btn => btn.addEventListener('click', () => context(btn.dataset.key)));
  context('labels');
  show(0);
}
