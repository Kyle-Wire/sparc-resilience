# Contributing to SPARC

## Development Setup

```bash
# Clone and install in editable mode with dev dependencies
git clone <repo-url>
cd GW3C_v2.1
pip install -e ".[dev,ui,dask]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Linting

```bash
ruff check sparc/
ruff format sparc/     # auto-format
```

## Project Structure

All source code lives under `sparc/`:

| Directory | Purpose |
|-----------|---------|
| `sparc/config/` | Configuration loading + JSON Schema validation |
| `sparc/models/` | GWEN, OLS, GWR, GWRF, GAM, Meta-Ensemble, Deep Kriging |
| `sparc/causal/` | DAG definition, causal discovery, counterfactuals |
| `sparc/data/` | Data loading, caching, temporal preparation |
| `sparc/evaluation/` | Spatial evaluators and causal diagnostics |
| `sparc/features/` | Laplacian eigenmaps and fold-aware variants |
| `sparc/interventions/` | Scenario simulation, extrapolation guard, physics priors |
| `sparc/run/` | Pipeline orchestration: stages, paths, configuration |
| `sparc/server/` | FastAPI backend and WebSocket streaming |

## Adding a New Model

1. Create `sparc/models/my_model.py` implementing `fit()` and `predict()`
2. Register it in `sparc/models/__init__.py`
3. Wire it into `sparc/run/enhanced_spatial_cv.py`
4. Add a test in `tests/`

## Adding a New Domain Template

1. Copy `templates/blank/` to `templates/<domain>/`
2. Fill in `project.yml` with domain-specific defaults
3. Optionally add `physics/priors.yml` and `causal/dag.yml`
