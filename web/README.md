# BogSignal presentation website

Existing Vite/plain-JS site, retaining its five sections, navigation and style.

## Launch in WSL Ubuntu-Work

From Windows:

```sh
wsl -d Ubuntu-Work -- bash -lc 'cd /home/krk/projects/BogSignal/web && npm run dev -- --host 0.0.0.0 --port 5173 --strictPort'
```

Open http://localhost:5173/. If already running, refresh the existing page.
Node 24 and installed frontend dependencies are required.

## Retained experiment assets

`public/experiment/` contains three real validation examples, exact saved
ranking budgets, A-D metrics, provenance and a self-contained copy of the
complete Argos final report. No training, inference or test-set scoring is
performed by the website.

To regenerate from retained local experiments using the project environment:

```sh
cd /home/krk/projects/BogSignal
ml/.venv/bin/python web/scripts/export_demo.py
```

The exporter checks frozen provenance, matches B/D IDs, labels and masks, and
checks their complete combined-class confusion matrices against the report.
Examples deliberately select extreme positive/negative wetland IoU changes
and a near-zero change among nontrivial reference-wetland patches. They are
not representative samples. The exact selection rule is in demo.json.

## Verification

`npm run build` builds the static site. `node --check src/main.js` checks JS syntax.

`scripts/check_demo.mjs` uses Playwright (optional verification dependency,
not required to run the site) against a running server. It checks all examples,
boundary dragging, keyboard controls and slider endpoints, all four saved budget values, table
metrics, images, report links, navigation and desktop/mobile overflow.

```sh
node scripts/check_demo.mjs
```

For an external Playwright installation set PLAYWRIGHT_MODULE to its
index.mjs file URL; DEMO_URL can override http://localhost:5173/.


## Regional landscape map

Leaflet renders locally saved, display-simplified country boundaries and the
Polissia study outline; no external tile service or map key is required.
Map toggles select weak labels, regional model D predictions, and coverage.
Other is transparent; use Coverage to separate predicted Other from unavailable
predictions. Weak labels have their own eligibility footprint.

After the regional inference run has a successful complete.json, export using:

    cd /home/krk/projects/BogSignal
    ml/.venv/bin/python web/scripts/export_map.py

This writes three overview PNGs in web/public/map, plus the outline, countries
and provenance manifest. It reprojects with nearest-neighbour sampling to Web
Mercator, then Leaflet positions the images by their matching geographic bounds.
The full-resolution 20 m source outputs stay in data/inference.

Map verification: with the site running, execute `node scripts/check_map.mjs`
using the same optional PLAYWRIGHT_MODULE setting as the other browser checks.


## Original-resolution swipe backgrounds

The three swipe RGB backgrounds now use real 10 m B4/B3/B2 source pixels.
The download uses each saved 20 m source-index raster to choose the same scenes
for each location; it does not select new dates or change the model inputs.
Native downloaded chips and scene-level provenance are retained under
data/presentation/sentinel2_rgb10m. Predictions and reference masks remain 20 m.

To fetch missing chips using the existing Argos Earth Engine connection:

    cd /home/krk/projects/Argos
    .venv/bin/python /home/krk/projects/BogSignal/web/scripts/fetch_rgb10m.py

Then regenerate web assets with the BogSignal project environment:

    cd /home/krk/projects/BogSignal
    ml/.venv/bin/python web/scripts/export_demo.py


## GitHub Pages

The existing repository publishes the website from `develop` using
`.github/workflows/pages.yml`. Only `web/dist` is uploaded; the ML environment,
raw imagery and native inference products are not required by GitHub Actions.
The checked-in presentation assets under web/public are the deployment inputs.

In repository Settings > Pages, select GitHub Actions as the source.
The expected URL is https://oleksastepaniuk.github.io/BogSignal/ .
GitHub's github-pages environment must allow deployments from develop.
Pushes changing web/ or the workflow on develop build and deploy automatically.
Pull requests to develop build and check paths without deploying.

Local development remains at `/`. To verify the repository-path build:

    PAGES_BASE_PATH=/BogSignal/ npm run build
    PAGES_BASE_PATH=/BogSignal/ node scripts/check_pages.mjs
    PAGES_BASE_PATH=/BogSignal/ npm run preview -- --host 0.0.0.0

Open http://localhost:4173/BogSignal/ for the production preview.
Public data/image credits are in public/ATTRIBUTIONS.md and deployed alongside
the site; source-specific terms still apply.
