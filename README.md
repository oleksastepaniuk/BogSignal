# BogSignal

An Argos team project exploring hidden and degraded wetlands in Polissia using NASA NISAR radar, satellite imagery, and machine learning.

We aim to combine radar and optical observations to study wetland extent and change, and explore how these observations could support ecological restoration. The project is experimental; these are research directions, not validated capabilities.

- `ml/` — Python tools for satellite data processing and machine learning.
- `web/` — future map-based visualization.
- `data/` — local datasets; large or restricted data must stay out of Git.

Dataset creation: [versioned workflow](docs/dataset_creation.md) and [existing v3 details](ml/DATASET.md). V4 is completed and audited, with 3,753 patches in fresh geographic splits; v3 remains preserved.

## Python setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first. It manages Python, dependencies, and the local virtual environment.

From the repository root:

```bash
cd ml
uv sync --locked
```

This creates `ml/.venv/` with the project's dependencies, using Python 3.11. uv can download Python if needed. The first setup requires internet access.

Run Python through the environment without activating it:

```bash
uv run --locked python
```

For a script, use `uv run --locked python path/to/script.py`. Run these commands from `ml/`. After pulling dependency changes, run `uv sync --locked` again.

If you prefer to activate the environment in Linux, macOS, or WSL:

```bash
source .venv/bin/activate
```

In Windows PowerShell, use `.venv\Scripts\Activate.ps1`. Run `deactivate` when finished. In your editor, select the Python interpreter inside `ml/.venv/`.

Commit `pyproject.toml` and `uv.lock` when dependencies change; keep `.venv/`, credentials, and downloaded datasets local.

## Web setup

The web app uses Vite and plain JavaScript. It currently shows a simple BogSignal landing page; the interactive map is planned.

### Install Node.js

Install [Node.js 24 LTS](https://nodejs.org/en/download), which includes npm. If you work in WSL, install Node.js inside your WSL distribution. Check that both tools are available:

```bash
node --version
npm --version
```

The web environment is separate from Python. Dependencies live in `web/node_modules/`; no Python environment activation is needed. The `.nvmrc` file records the Node.js version for people using nvm.

### Run locally

From the repository root:

```bash
cd web
npm ci
npm run dev
```

`npm ci` installs the versions recorded in `package-lock.json`. Run it on first setup and after pulling dependency changes.

Open the local address printed in the terminal (usually `http://localhost:5173`). The page updates as you edit files in `web/src/`. Press Ctrl+C to stop the server.

### Build and preview

From `web/`:

```bash
npm run build
npm run preview
```

The build is written to `web/dist/`. The preview command serves that build locally for checking; it does not publish the app.

Commit `package.json` and `package-lock.json` when dependencies change. Keep `node_modules/` and `dist/` local; both are ignored by Git.

Inference workflow: [OSM wetland inspection and Polissia mapping](ml/inference/README.md). Uses frozen v4 inputs; a trained model adapter/checkpoint is required.
