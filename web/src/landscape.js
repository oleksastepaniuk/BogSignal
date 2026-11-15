import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

export async function setupLandscape() {
  const host=document.getElementById('landscape-map');
  const status=document.getElementById('map-status');
  const load=async path=>{const response=await fetch(import.meta.env.BASE_URL+'map/'+path);if(!response.ok)throw new Error('Map asset unavailable: '+path);return response.json();};
  const [info,countries,polissia]=await Promise.all([load('manifest.json'),load('countries.geojson'),load('polissia.geojson')]);
  const map=L.map(host,{scrollWheelZoom:false,minZoom:5,maxZoom:13,maxBounds:[[46,14],[58,44]],maxBoundsViscosity:.8});
  map.attributionControl.setPrefix('<a href="https://leafletjs.com">Leaflet</a>');
  L.geoJSON(countries,{style:{color:'#89958e',weight:1.4,fillColor:'#edf0e9',fillOpacity:1},onEachFeature:(f,layer)=>{if(f.properties.name!=='Russia')layer.bindTooltip(f.properties.name,{sticky:true});}}).addTo(map);
  map.attributionControl.addAttribution('&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a> (saved boundaries, ODbL)');
  for(const [name,lat,lon] of [['UKRAINE',50.1,29.6],['BELARUS',53.4,28.5],['POLAND',52.1,21.1]]) {
    L.marker([lat,lon],{interactive:false,icon:L.divIcon({className:'country-name',html:name,iconSize:[100,20],iconAnchor:[50,10]})}).addTo(map);
  }
  map.createPane('coverage');map.getPane('coverage').style.zIndex=410;
  map.createPane('references');map.getPane('references').style.zIndex=420;
  map.createPane('predictions');map.getPane('predictions').style.zIndex=430;
  map.createPane('study');map.getPane('study').style.zIndex=450;
  const outline=L.geoJSON(polissia,{pane:'study',style:{color:'#26664d',weight:2,fillOpacity:.02},onEachFeature:(_,layer)=>layer.bindPopup('Polissia study area. Outline adapted from the regional map on savepolesia.org and local polygons. Approximate ecological outline; not a legal or habitat boundary.')});
  const overlay=(name,pane)=>L.imageOverlay(import.meta.env.BASE_URL+'map/'+name+'.png',info.bounds,{pane,opacity:.9,alt:name==='labels'?'Historical/inferred weak-reference classes':name==='prediction'?'Regional model D predicted classes':'Coverage: teal predicted, gray unavailable',className:'map-raster'});
  const layers={polissia:outline,labels:overlay('labels','references'),prediction:overlay('prediction','predictions'),coverage:overlay('coverage','coverage')};
  for(const [key,layer] of Object.entries(layers)) {
    const checkbox=document.getElementById('map-'+key);
    if(checkbox.checked) layer.addTo(map);
    checkbox.addEventListener('change',()=>checkbox.checked?layer.addTo(map):map.removeLayer(layer));
    if(layer instanceof L.ImageOverlay) layer.on('error',()=>{status.textContent='A map image could not load. Reload the page to retry.';});
  }
  const studyBounds=outline.getBounds();
  map.fitBounds(studyBounds,{padding:[18,18],animate:false});
  document.getElementById('map-reset').addEventListener('click',()=>map.fitBounds(studyBounds,{padding:[18,18],animate:false}));
  document.getElementById('map-detail').addEventListener('click',()=>map.setView(info.example_center,10,{animate:false}));
  new ResizeObserver(()=>map.invalidateSize({pan:false})).observe(host);
  status.textContent='Our model covers '+(info.counts.predicted_pixels*0.0004).toLocaleString(undefined,{maximumFractionDigits:0})+' km² — nearly three-quarters of the study area. Remaining gaps are primarily due to missing or unusable satellite observations.';
  host.dataset.ready='true';
}
