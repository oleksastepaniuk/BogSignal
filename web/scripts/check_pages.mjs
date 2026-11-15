import assert from 'node:assert/strict';
import {readFile,stat} from 'node:fs/promises';
import path from 'node:path';
const root=path.resolve('dist');
const base=process.env.PAGES_BASE_PATH || '/BogSignal/';
assert.ok(base.startsWith('/')&&base.endsWith('/'));
const html=await readFile(path.join(root,'index.html'),'utf8');
assert.ok(!html.includes('%BASE_URL%'));
for(const match of html.matchAll(/(?:src|href)="(\/[^"]*)"/g)) {
 const url=match[1];assert.ok(url.startsWith(base),'Root-relative link bypasses repository base: '+url);
 const local=url.slice(base.length).split(/[?#]/)[0];if(local)assert.ok((await stat(path.join(root,local))).isFile(),url);
}
const manifest=JSON.parse(await readFile(path.join(root,'map/manifest.json'),'utf8'));
assert.ok(manifest.counts.predicted_pixels>0);
for(const name of ['labels','prediction','coverage'])assert.ok((await stat(path.join(root,'map',name+'.png'))).size>0);
const demo=JSON.parse(await readFile(path.join(root,'experiment/demo.json'),'utf8'));
for(const row of demo.examples)for(const suffix of ['rgb','reference','b','d','b-overlay','d-overlay'])assert.ok((await stat(path.join(root,'experiment',row.key+'-'+suffix+'.png'))).isFile());
const seasonal=JSON.parse(await readFile(path.join(root,'seasonal_full/manifest.json'),'utf8'));
assert.equal(seasonal.examples.length,3);
for(const site of ['cheremske','somyne','syra_pohonia'])assert.equal(seasonal.examples.filter(row=>row.site===site).length,1);
for(const row of seasonal.examples)for(const file of Object.values(row.files))assert.ok((await stat(path.join(root,'seasonal_full',file.filename))).size>0);
const report=await readFile(path.join(root,'experiment/final_experiment_report.html'),'utf8');
assert.ok(report.includes('href="../"'),'Report must return to the repository site root');
assert.ok((await stat(path.join(root,'ATTRIBUTIONS.md'))).isFile());
console.log('PASS repository-base URLs, saved images, report return link and attribution artifact');
