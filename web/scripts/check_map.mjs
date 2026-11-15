import assert from 'node:assert/strict';
import {readFile,readdir} from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const info=JSON.parse(await readFile(new URL('../public/map/manifest.json',import.meta.url)));
assert.equal((await readdir(new URL('../public/map/',import.meta.url))).filter(n=>n.endsWith('.png')).length,3);
assert.ok(info.coverage_fraction>0&&info.coverage_fraction<1);
const browser=await chromium.launch({headless:true});
try {
 for(const viewport of [{width:1280,height:900},{width:390,height:844}]) {
  const page=await browser.newPage({viewport});const errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  page.on('response',r=>{if(r.status()>=400)errors.push(r.url()+': '+r.status());});
  await page.goto(process.env.DEMO_URL||'http://localhost:5173/');
  await page.waitForFunction(()=>document.getElementById('landscape-map')?.dataset.ready==='true');
  await page.waitForFunction(()=>[...document.querySelectorAll('#landscape-map img.map-raster')].every(i=>i.complete&&i.naturalWidth>0));
  assert.ok((await page.locator('#map-status').textContent()).includes((info.counts.predicted_pixels*0.0004).toLocaleString(undefined,{maximumFractionDigits:0})));
  for(const id of ['prediction','labels','coverage']) {
   await page.locator('#map-'+id).check();
   const filename=id==='prediction'?'prediction':id;
   const image=page.locator('#landscape-map img[src="/map/'+filename+'.png"]');
   await image.waitFor();await page.waitForFunction(src=>{const i=document.querySelector('img[src="'+src+'"]');return i?.complete&&i.naturalWidth>0;},'/map/'+filename+'.png');
   await page.locator('#map-'+id).uncheck();assert.equal(await image.count(),0);
  }
  await page.locator('#map-polissia').uncheck();
  assert.equal(await page.locator('.leaflet-study-pane path').count(),0);
  await page.locator('#map-polissia').check();assert.ok(await page.locator('.leaflet-study-pane path').count()>0);
  await page.locator('#map-prediction').check();
  await page.locator('#map-detail').click();
  await page.waitForTimeout(400);
  await page.locator('#landscape-map').screenshot({path:'/tmp/bogsignal-map-detail-'+viewport.width+'.png'});
  await page.locator('.leaflet-control-zoom-out').click();
  await page.waitForTimeout(400);
  await page.locator('#map-reset').click();
  await page.waitForTimeout(400);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),true);
  assert.ok((await page.locator('.leaflet-control-attribution').textContent()).includes('OpenStreetMap'));
  assert.ok(!(await page.locator('.country-name').allTextContents()).includes('RUSSIA'));
  assert.equal(await page.locator('#landscape h2').textContent(),'Europe’s Amazon');
  assert.equal(await page.locator('#landscape #map-resolution').count(),0);
  assert.equal(await page.locator('#map-resolution').count(),0);
  await page.locator('#landscape').screenshot({path:'/tmp/bogsignal-map-'+viewport.width+'.png'});
  assert.deepEqual(errors,[]);
  console.log('PASS regional map layers, coverage, local assets, zoom/reset and layout at '+viewport.width+'px');
  await page.close();
 }
} finally {await browser.close();}