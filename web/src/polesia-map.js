// Polesia overview map, replicated from https://savepolesia.org/polesia/
// Style, tilesets and access token belong to the Frankfurt Zoological Society (FZS)
// Mapbox account; map by pg-webstudio.de. Credited in index.html.
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';

// Token is read from web/.env.local (gitignored); see web/.env.example.
// TODO: Replace FZS's token with our own Mapbox token (restricted by URL) before going
// beyond the demo. Their style and `mapbox://fzs.*` tilesets are private to the FZS
// account, so this also means switching to our own style and re-hosting the Polesia layers.
mapboxgl.accessToken = import.meta.env.VITE_MAPBOX_TOKEN;
if (!mapboxgl.accessToken) {
  console.warn('VITE_MAPBOX_TOKEN is not set; the Polesia map will not load. See web/.env.example.');
}

const vector = (id) => ({ type: 'vector', url: `mapbox://fzs.${id}` });
const points = (features) => ({
  type: 'geojson',
  data: {
    type: 'FeatureCollection',
    features: features.map(([title, coordinates, extra = {}]) => ({
      type: 'Feature',
      geometry: { type: 'Point', coordinates },
      properties: { title, ...extra },
    })),
  },
});
const halo = (width, color = '#000000') => ({
  'text-color': color,
  'text-halo-color': '#fff',
  'text-halo-width': width,
});

// Toggle buttons: each controls one or more layers.
const TOGGLES = [
  { label: 'International Protected Areas', color: '#008000', on: true,
    layers: ['polesia-emerald', 'polesia-emerald-label', 'almany-mire', 'almany-label'] },
  { label: 'Chernobyl Exclusion Zone', color: '#c30df6', on: false, layers: ['cez-zone'] },
  { label: 'Biodiversity Hotspots', color: '#0d4406', on: false, layers: ['biodiversity-hotspots'] },
];

const HOTSPOTS = [
  ['Almany mire', [27.331, 51.791]],
  ['Lower Vistula River Valley', [18.737, 53.489]],
  ['Dolna Wisła', [18.88, 53.98]],
  ['Solecka Dolina Wisły', [18.115, 53.127]],
  ['Dybowska Dolina Wisły', [18.398, 53.043]],
  ['Nieszawska Dolina Wisły', [18.782, 52.91]],
  ['Włocławska Dolina Wisły', [18.994, 52.728]],
  ['Kampinoska Dolina Wisły', [20.094, 52.364]],
  ['Kampinoska Dolina Wisły', [19.853, 52.431]],
  ['Dolina Środkowej Wisły', [20.921, 52.337]],
  ['Dolina Środkowej Wisły', [21.174, 52.138]],
  ['Dolina Środkowej Wisły', [21.366, 51.831]],
  ['Dolny Wieprz', [22.182, 51.581]],
  ['Dolina Tyśmienicy', [22.691, 51.7]],
  ['Pripyatsky National Park', [27.823, 52.055]],
  ['Prypiat-Stokhid National Nature Park', [25.231, 51.86]],
  ['Chornobylskyi Biosphere Reserve', [29.758, 51.347]],
];

function addLayers(map) {
  const visibility = (id) =>
    TOGGLES.find((t) => t.layers.includes(id))?.on === false ? 'none' : 'visible';

  const layers = [
    { id: 'polesia-polygon', type: 'fill', source: vector('0uu8x1fm'),
      'source-layer': 'Polesia_Borders-bp2ak3',
      paint: { 'fill-color': '#6B8E23', 'fill-opacity': 0.2 } },
    { id: 'polesia-line', type: 'line', source: vector('0uu8x1fm'),
      'source-layer': 'Polesia_Borders-bp2ak3',
      paint: { 'line-color': '#008000', 'line-width': 2, 'line-dasharray': [2, 1.8] } },
    { id: 'europe-overlay', type: 'fill', source: vector('1uvn7014'),
      'source-layer': 'europe_overlay_square-231204',
      paint: { 'fill-color': '#696969', 'fill-opacity': 0.2 } },
    { id: 'overlay-plbeuk-line', type: 'line', source: vector('1slmn6si'),
      'source-layer': 'PL_BE_UK_3_border',
      paint: { 'line-color': '#696969', 'line-width': 1.3 } },
    { id: 'oceans-label', type: 'symbol', source: vector('ab2mnyaq'),
      'source-layer': 'ocean-points-akwoel',
      layout: {
        'text-field': ['to-string', ['get', 'Name']],
        'text-offset': ['match', ['get', 'Name'],
          ['Black Sea'], ['literal', [1.4, 1]],
          ['Baltic Sea'], ['literal', [0, -1]],
          ['literal', [0, 0]]],
        'text-size': 16,
      },
      paint: halo(2) },
    { id: 'polesia-emerald', type: 'fill', source: vector('bjfgp1ix'),
      'source-layer': 'Emerald_Natura2000_Polesia_UA-d5e1t5',
      paint: { 'fill-color': '#008000', 'fill-opacity': 0.5 } },
    { id: 'polesia-emerald-label', type: 'symbol', source: vector('bjfgp1ix'),
      'source-layer': 'Emerald_Natura2000_Polesia_UA-d5e1t5', minzoom: 6.5,
      layout: {
        'text-field': ['get', 'SITE_NAME'],
        'text-size': 10,
        'text-padding': 5,
        'text-variable-anchor': ['top', 'bottom', 'left', 'right'],
      },
      paint: halo(2.5, '#556B2F') },
    { id: 'cez-zone', type: 'fill', source: vector('7cz3levr'),
      'source-layer': 'Chernobyl_CEZ_Border_30km-3hqv5x',
      paint: { 'fill-color': '#c30df6' } },
    { id: 'almany-mire', type: 'line', source: vector('ck4yr13w'),
      'source-layer': 'Pripyatsky_Almany_National_Pa-4d3n2o',
      paint: { 'line-color': '#006400', 'line-width': 2 } },
    { id: 'almany-label', type: 'symbol', minzoom: 5,
      source: points([
        ['Almany mire', [27.37495, 51.853588]],
        ['Pripyatsky National Park', [28.230056, 52.024056]],
      ]),
      layout: {
        'text-field': ['get', 'title'],
        'text-size': 12,
        'text-offset': [1, -1],
        'text-variable-anchor': ['left', 'right', 'top', 'bottom'],
        'text-allow-overlap': true,
      },
      paint: halo(3, '#006400') },
    { id: 'tschernobyl-label', type: 'symbol',
      source: points([['Chernobyl', [30.112583, 51.385964], { icon: 'nuclear-15-rw' }]]),
      layout: {
        'icon-image': ['get', 'icon'],
        'text-field': ['get', 'title'],
        'text-font': ['Open Sans Semibold', 'Arial Unicode MS Bold'],
        'text-size': 11,
        'text-offset': [1, -1],
        'text-variable-anchor': ['left', 'right', 'top', 'bottom'],
        'text-allow-overlap': true,
        'icon-allow-overlap': true,
      },
      paint: halo(3) },
    { id: 'label-for-states-polesia', type: 'symbol',
      source: points([['Polesia', [26.55535, 51.32773]]]),
      layout: { 'text-field': ['get', 'title'], 'text-offset': [2, 0], 'text-size': 28 },
      paint: halo(6, '#008000') },
    { id: 'cities-label', type: 'symbol', source: vector('571abfqh'),
      'source-layer': 'Hauptstaedte_points-bootnw',
      layout: {
        'text-field': ['match', ['get', 'Name_en'], 'Kiev', 'Kyiv', ['get', 'Name_en']],
        'text-size': 16,
        'text-offset': [0, -0.8],
        'icon-image': 'dot-11',
        'icon-size': 0.9,
      },
      paint: halo(2) },
    { id: 'biodiversity-hotspots', type: 'symbol', source: points(HOTSPOTS),
      layout: {
        'text-field': ['get', 'title'],
        'text-variable-anchor': ['right', 'left', 'top', 'bottom'],
        'text-size': 11,
        'text-offset': [1, 0],
        'icon-image': 'biodiv3_pin_FZG15',
        'icon-size': 2,
        'icon-allow-overlap': true,
        'icon-offset': [0, -2],
      },
      paint: halo(1) },
  ];

  for (const layer of layers) {
    layer.layout = { ...layer.layout, visibility: visibility(layer.id) };
    map.addLayer(layer);
  }
}

function addToggles(map, container) {
  const menu = document.createElement('div');
  menu.className = 'map-toggles';
  for (const toggle of TOGGLES) {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = toggle.label;
    button.style.setProperty('--active', toggle.color);
    button.setAttribute('aria-pressed', String(toggle.on));
    button.addEventListener('click', () => {
      toggle.on = !toggle.on;
      button.setAttribute('aria-pressed', String(toggle.on));
      for (const id of toggle.layers) {
        map.setLayoutProperty(id, 'visibility', toggle.on ? 'visible' : 'none');
      }
    });
    menu.append(button);
  }
  container.append(menu);
}

export function initPolesiaMap(container) {
  const map = new mapboxgl.Map({
    container,
    style: 'mapbox://styles/fzs/ck8afv7pa0c131iqlgx7njk1g',
    center: [28.394993, 51.552022],
    maxBounds: [
      [8.263322, 44.396022],
      [49.147726, 56.294789],
    ],
    attributionControl: false,
    // Require Ctrl/two fingers to zoom so the map doesn't trap page scrolling.
    cooperativeGestures: true,
  })
    .addControl(new mapboxgl.NavigationControl())
    .addControl(new mapboxgl.AttributionControl({ compact: true }))
    .addControl(new mapboxgl.ScaleControl());

  addToggles(map, container);
  map.on('load', () => addLayers(map));
  map.on('click', 'polesia-emerald', (e) => {
    const popup = document.createElement('div');
    popup.textContent = e.features[0].properties.SITE_NAME;
    new mapboxgl.Popup().setLngLat(e.lngLat).setDOMContent(popup).addTo(map);
  });
  return map;
}
