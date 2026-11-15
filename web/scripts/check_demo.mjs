import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const data=JSON.parse(await readFile(new URL('../public/experiment/demo.json',import.meta.url)));
const browser=await chromium.launch({headless:true});
try {
 for (const viewport of [{width:1280,height:900},{width:390,height:844}]) {
  const page=await browser.newPage({viewport});
  const errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  page.on('response',r=>{if(r.status()>=400) errors.push(r.url()+': '+r.status());});
  await page.goto(process.env.DEMO_URL || 'http://localhost:5173/');
  await page.locator('#results tbody tr').last().waitFor({state:'attached'});
  assert.equal(await page.locator('#full-results').getAttribute('open'),null);
  assert.match(await page.locator('#sensor-finding').textContent(),/\+7\.8/);
  await page.locator('#full-results summary').click();
  await page.locator('#results').waitFor();
  assert.equal(await page.locator('main section').count(),5);
  assert.equal(await page.locator('.nav a').count(),5);

  for(const card of data.sensor_cards) {
   assert.ok(!data.examples.some(e=>e.id===card.patch_id));
   await page.locator('#sensor-'+card.sensor).scrollIntoViewIfNeeded();
   await page.waitForFunction(id=>{const i=document.getElementById(id);return i.complete&&i.naturalWidth===(i.id==='satellite'?512:256);},'sensor-'+card.sensor);
  }
  assert.equal(await page.locator('#results .best-row').count(),1);
  assert.match(await page.locator('#results .best-row th').textContent(),/^D/);
  const tdCells=page.locator('#results tbody').nth(1).locator('tr').last().locator('td');
  for(const [i,key] of ['iou','recall','precision','dice'].entries()) {
   const expected=data.temporal.temporal.combined_class.classes['Wetland (Swamp + Marsh)'][key];
   assert.ok((await tdCells.nth(i).textContent()).startsWith((100*expected).toFixed(1)+'%'));
  }
  for(const example of data.examples) {
   await page.locator('[data-key="'+example.key+'"]').click();
   assert.match(await page.locator('#patch-caption').textContent(),new RegExp(example.id));
   assert.equal(await page.locator('#examples [aria-pressed="true"]').count(),1);
   assert.deepEqual(await page.locator('.context-grid img').evaluateAll(imgs=>imgs.map(i=>i.id)),['reference','panel-b','panel-d']);
   for(const [id,suffix] of [['panel-b','b'],['reference','reference'],['panel-d','d']]) {
    assert.match(await page.locator('#'+id).getAttribute('src'),new RegExp(example.key+'-'+suffix+'.png$'));
   }

   assert.equal(await page.locator('#modes').count(),0);
   assert.equal(await page.locator('#compare-range').count(),0);
   await page.locator('#swipe-line').focus();
   for(const [key,value] of [['Home','0'],['End','100'],['Home','0']]) {
    await page.keyboard.press(key);
    assert.equal(await page.locator('#map-d').evaluate(el=>el.style.clipPath),'inset(0px 0px 0px '+value+'%)');
    assert.equal(await page.locator('#map-b').evaluate(el=>el.style.clipPath),'inset(0px '+(100-Number(value))+'% 0px 0px)');
   }
   for(let step=0;step<5;step++) await page.keyboard.press('PageUp');
   assert.match(await page.locator('#satellite').getAttribute('src'),new RegExp(example.key+'-rgb.png$'));
   assert.match(await page.locator('#map-d').getAttribute('src'),/-d-overlay.png$/);
   await page.locator('#swipe-line').scrollIntoViewIfNeeded();
   const rect=await page.locator('#comparison').boundingBox();
   const handle=await page.locator('#swipe-line').boundingBox();
   await page.mouse.move(handle.x+handle.width/2,handle.y+handle.height/2);
   await page.mouse.down();
   await page.mouse.move(rect.x+rect.width*.75,handle.y+handle.height/2,{steps:8});
   await page.mouse.up();
   assert.ok(Math.abs(Number(await page.locator('#swipe-line').getAttribute('aria-valuenow'))-75)<=1);
   await page.locator('#swipe-line').focus();
   await page.keyboard.press('Home');
   assert.equal(await page.locator('#swipe-line').getAttribute('aria-valuenow'),'0');
   await page.keyboard.press('ArrowRight');
   assert.equal(await page.locator('#swipe-line').getAttribute('aria-valuenow'),'1');
   await page.keyboard.press('End');
   assert.equal(await page.locator('#swipe-line').getAttribute('aria-valuenow'),'100');
   await page.waitForFunction(()=>[...document.querySelectorAll('#demo img')].every(i=>i.complete&&i.naturalWidth===(i.id==='satellite'?512:256)));
  }
  assert.equal(await page.locator('.budget-column').count(),4);
  assert.equal(await page.locator('#budget').count(),0);
  for(let i=0;i<data.budgets.length;i++) {
   const column=page.locator('.budget-column').nth(i);
   assert.equal(await column.locator('strong').textContent(),(100*data.budgets[i].recall).toFixed(1)+'%');
   const height=await column.locator('.budget-bar').evaluate(el=>parseFloat(el.style.height));
   assert.ok(Math.abs(height-100*data.budgets[i].recall)<.0001);
  }
  assert.equal(await page.locator('.budget-column.highlight').getAttribute('data-budget'),'0.05');
  for(let i=0;i<4;i++) {
   const expected=data.baseline[i].combined_class.classes['Wetland (Swamp + Marsh)'];
   assert.deepEqual(await page.locator('#results tbody tr').nth(i).locator('td').allTextContents(),
    ['iou','recall','precision','dice'].map(k=>(100*expected[k]).toFixed(1)+'%'));
  }
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),true);
  await page.locator('a[href="#findings"]').click();
  await page.waitForFunction(()=>document.querySelector('.nav a[href="#findings"]').classList.contains('active'));
  for(const path of ['/experiment/final_experiment_report.html','/experiment/final_experiment_report_data.json','/experiment/demo.json']) {
   assert.equal((await page.request.get(new URL(path,page.url()).href)).status(),200);
  }
  await page.locator('[data-key="helps"]').click();
  await page.locator('#swipe-line').focus();
  await page.keyboard.press('Home');
  for(let step=0;step<5;step++) await page.keyboard.press('PageUp');
  await page.locator('#experiment').screenshot({path:'/tmp/bogsignal-experiment-'+viewport.width+'.png'});
  assert.deepEqual(errors,[]);
  console.log('PASS interactions, assets, metrics, navigation and report links at '+viewport.width+'px');
  await page.close();
 }
} finally { await browser.close(); }
