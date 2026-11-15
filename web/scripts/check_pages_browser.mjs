import assert from 'node:assert/strict';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const browser=await chromium.launch({headless:true,executablePath:process.env.PLAYWRIGHT_CHROMIUM_PATH});
const base=process.env.PAGES_PREVIEW_URL || 'http://127.0.0.1:4175/BogSignal/';
try {
for(const viewport of [{width:1280,height:900},{width:390,height:844}]) {
 const page=await browser.newPage({viewport});const errors=[];
 page.on('pageerror',error=>errors.push(error.message));
 page.on('response',response=>{if(response.status()>=400)errors.push(response.url()+': '+response.status());});
 await page.goto(base);
 await page.waitForFunction(()=>document.getElementById('landscape-map')?.dataset.ready==='true');
 await page.locator('#examples button').last().waitFor();
 for(const sensor of ['s2','s1','nisar']) {
  await page.locator('#sensor-'+sensor).scrollIntoViewIfNeeded();
  await page.waitForFunction(id=>{const i=document.getElementById(id);return i.complete&&i.naturalWidth>0;},'sensor-'+sensor);
 }
 for(const example of ['helps','little','worsens']) {
  await page.locator('[data-key="'+example+'"]').click();
  await page.waitForFunction(()=>[...document.querySelectorAll('#demo img')].every(i=>i.complete&&i.naturalWidth>0));
  assert.ok((await page.locator('#satellite').getAttribute('src')).startsWith('/BogSignal/experiment/'));
 }
 for(const layer of ['labels','coverage','prediction']) {
  await page.locator('#map-'+layer).check();
  await page.waitForFunction(()=>[...document.querySelectorAll('.map-raster')].every(i=>i.complete&&i.naturalWidth>0));
  if(layer!=='prediction')await page.locator('#map-'+layer).uncheck();
 }
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
 assert.ok((await page.request.get(new URL('ATTRIBUTIONS.md',base).href)).ok());
 await page.locator('#full-results > summary').click();
 assert.equal(await page.locator('#full-results').getAttribute('open'),'');
 assert.equal(await page.locator('#full-results .cred-card').count(),2);
 assert.equal(await page.locator('#map-provenance').count(),0);
 const report=page.locator('#full-results a[href="/BogSignal/experiment/final_experiment_report.html"]').first();
 await report.click();await page.locator('.report-return a').waitFor();
 assert.equal(await page.locator('.report-data a').count(),3);
 for(const href of await page.locator('.report-data a').evaluateAll(links=>links.map(a=>a.href)))assert.ok((await page.request.get(href)).ok());

 await page.locator('.report-return a').click();
 await page.waitForURL(base);
 await page.locator('#examples button').last().waitFor();
 assert.deepEqual(errors,[]);
 console.log('PASS Pages subpath: images, map layers, report round-trip and responsive layout at '+viewport.width+'px');
 await page.close();
}
} finally {await browser.close();}