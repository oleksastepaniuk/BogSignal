# BogSignal

An Argos team project exploring hidden and degraded wetlands in Polissia using NASA NISAR radar, satellite imagery, and machine learning.

We aim to combine radar and optical observations to study wetland extent and change, and explore how these observations could support ecological restoration. The project is experimental; these are research directions, not validated capabilities.

- `ml/` — Python tools for satellite data processing and machine learning.
- `web/` — future map-based visualization.
- `data/` — local datasets; large or restricted data must stay out of Git.

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
