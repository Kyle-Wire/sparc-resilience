"""
suggest_gwen_config — derive a validated GWENConfig for Stage 1.

Replaces the raw-dict-returning ``auto_suggest_gwen_params()`` in
``gwen_variable_selection.py`` with a typed seam that always returns a usable
:class:`~sparc.models.gwen_config.GWENConfig` and logs a structured warning
(never silently swallows) when Stage 0 outputs are absent or incomplete.

Priority order (highest wins):
  1. Explicit values in ``config['gwen_params']`` (project.yml).
  2. Stage 0 hints: k_neighbors derived via BandwidthAdvisor from the
     CorrelogramReport; l1_ratios shifted toward L1-heavy when high
     multicollinearity is detected in the predictor matrix.
  3. Built-in safe defaults.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from sparc.models.gwen_config import GWENConfig

_logger = logging.getLogger(__name__)

_DEFAULT_K_NEIGHBORS: int = 50
_DEFAULT_CV_FOLDS: int = 5
_DEFAULT_N_ALPHAS: int = 100
_DEFAULT_L1_RATIOS = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99]
_DEFAULT_SELECTION_THRESHOLD: float = 0.1


def suggest_gwen_config(
    config: dict,
    store=None,
    correlogram_dir: Optional[str] = None,
) -> "GWENConfig":
    """Return a validated :class:`~sparc.models.gwen_config.GWENConfig` for Stage 1.

    Parameters
    ----------
    config:
        Loaded project config dict (from ``load_config()``).
    store:
        Optional active :class:`~sparc.registry.store.ArtifactStore`.  When
        supplied, Stage 0 artifacts are read from the store before disk
        fallbacks.
    correlogram_dir:
        Disk fallback directory for Stage 0 correlogram JSON / report files.
    """
    from sparc.models.gwen_config import GWENConfig

    gwen_params: dict = dict(config.get("gwen_params") or {})

    # Derive Stage 0 hints — never raises, produces warnings on failures.
    stage0_hints = _load_stage0_hints(config, store, correlogram_dir)

    # Stage 0 hints fill only what project.yml doesn't explicitly set.
    # A None value in gwen_params is treated as "not set".
    merged: dict = dict(stage0_hints)
    for k, v in gwen_params.items():
        if v is not None:
            merged[k] = v

    k_neighbors = int(merged.get("k_neighbors") or _DEFAULT_K_NEIGHBORS)

    return GWENConfig(
        k_neighbors=max(1, k_neighbors),
        cv_folds=int(merged.get("cv_folds") or _DEFAULT_CV_FOLDS),
        n_alphas=int(merged.get("n_alphas") or _DEFAULT_N_ALPHAS),
        l1_ratios=list(merged.get("l1_ratios") or _DEFAULT_L1_RATIOS),
        sample_size=merged.get("sample_size"),
        quick_mode=bool(merged.get("quick_mode", False)),
        n_jobs=int(merged.get("n_jobs") or 1),
        selection_threshold=float(
            merged.get("selection_threshold") or _DEFAULT_SELECTION_THRESHOLD
        ),
        local_cv=bool(merged.get("local_cv", False)),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_stage0_hints(
    config: dict,
    store,
    correlogram_dir: Optional[str],
) -> dict:
    """Derive GWEN hyperparameter hints from Stage 0 results.

    Returns a dict with zero or more of the keys accepted by GWENConfig.
    Never raises — every failure produces a warning log entry and an empty
    (or partial) hints dict.
    """
    hints: dict = {}

    # ── 1. Bandwidth → k_neighbors via CorrelogramReport.gwen_config() ──────
    report = _try_load_correlogram_report(store, correlogram_dir)
    if report is not None:
        try:
            coords = _load_coords(config)
            if coords is not None and len(coords) >= 2:
                k = report._k_neighbors(coords)
                hints["k_neighbors"] = k
                _logger.info(
                    "Stage 0: k_neighbors=%d derived from CorrelogramReport "
                    "(median_range=%.0fm, n=%d)",
                    k,
                    report._median_range(),
                    len(coords),
                )
            else:
                _logger.warning(
                    "Stage 0 CorrelogramReport loaded but coords unavailable; "
                    "k_neighbors will use project config or default."
                )
        except Exception as exc:
            _logger.warning(
                "CorrelogramReport → k_neighbors derivation failed: %s", exc
            )
    else:
        _logger.info(
            "Stage 0 CorrelogramReport not found; k_neighbors will use "
            "project config or default (%d).",
            _DEFAULT_K_NEIGHBORS,
        )

    # ── 2. VIF → l1_ratios (high multicollinearity → L1-heavy) ──────────────
    try:
        data_path = (config.get("data") or {}).get("file_path")
        predictors = (config.get("predictors") or {}).get("base_model") or []
        if data_path and os.path.exists(str(data_path)) and len(predictors) >= 2:
            import pandas as pd
            from numpy.linalg import cond

            df = pd.read_csv(data_path, nrows=5000)
            valid = [p for p in predictors if p in df.columns]
            if len(valid) >= 2:
                X_check = df[valid].dropna().values
                if len(X_check) > 50:
                    cn = float(cond(X_check))
                    if cn > 30:
                        hints["l1_ratios"] = [0.7, 0.8, 0.9, 0.95, 0.99]
                        _logger.warning(
                            "High multicollinearity detected (cond=%.0f); "
                            "shifting l1_ratios toward L1.",
                            cn,
                        )
    except Exception as exc:
        _logger.debug("VIF l1_ratio check failed (non-critical): %s", exc)

    return hints


def _try_load_correlogram_report(store, correlogram_dir: Optional[str]):
    """Try to load a CorrelogramReport from store or disk.  Returns None on failure."""
    from sparc.run.correlogram_report import CorrelogramReport

    # ── Store: prefer typed correlogram_report artifact ──────────────────────
    if store is not None:
        try:
            if store.has("0", "correlogram_report"):
                payload = store.read_any("0", "correlogram_report")
                if payload is not None:
                    return CorrelogramReport.from_artifact(payload)
        except Exception as exc:
            _logger.warning(
                "Failed to read correlogram_report from store: %s", exc
            )

        # ── Store: fallback — reconstruct from raw correlogram_results ────────
        try:
            if store.has("0", "correlogram_results"):
                raw = store.read_any("0", "correlogram_results")
                if raw is not None:
                    report = _correlogram_report_from_raw(raw)
                    if report is not None:
                        return report
        except Exception as exc:
            _logger.warning(
                "Failed to reconstruct CorrelogramReport from correlogram_results: %s",
                exc,
            )

    # ── Disk fallback ─────────────────────────────────────────────────────────
    if correlogram_dir:
        import json

        for fname in ("correlogram_report.json", "correlogram_analysis_results.json"):
            path = os.path.join(correlogram_dir, fname)
            if os.path.exists(path):
                try:
                    with open(path) as fh:
                        data = json.load(fh)
                    if "bandwidths" in data:
                        return CorrelogramReport.from_artifact(data)
                    report = _correlogram_report_from_raw(data)
                    if report is not None:
                        return report
                except Exception as exc:
                    _logger.warning(
                        "Failed to load CorrelogramReport from disk (%s): %s",
                        path,
                        exc,
                    )

    return None


def _correlogram_report_from_raw(raw: dict):
    """Reconstruct a minimal CorrelogramReport from a raw correlogram_results dict.

    The raw dict has the shape written by Stage 0::

        {
          'individual_results': {
            'var1': {'optimal_bandwidth': X, 'effective_range': Y, ...},
            ...
          },
          'spatial_cv_configuration': {'optimal_block_size': Z, ...},
        }
    """
    from sparc.run.correlogram_report import CorrelogramReport

    try:
        ind = raw.get("individual_results") or {}
        if not ind:
            return None

        bandwidths = {}
        effective_ranges = {}
        for var, res in ind.items():
            bw = res.get("optimal_bandwidth")
            er = res.get("effective_range") or bw
            if bw is not None and float(bw) > 0:
                bandwidths[var] = float(bw)
            if er is not None and float(er) > 0:
                effective_ranges[var] = float(er)

        if not bandwidths:
            return None

        cv_cfg = raw.get("spatial_cv_configuration") or {}
        block_size = cv_cfg.get("optimal_block_size")
        if block_size is None and bandwidths:
            block_size = float(np.median(list(bandwidths.values())))

        return CorrelogramReport(
            bandwidths=bandwidths,
            block_size=float(block_size),
            block_size_source="correlogram",
            stationarity_warnings=[],
            variable_effective_ranges=effective_ranges,
        )
    except Exception as exc:
        _logger.debug("_correlogram_report_from_raw failed: %s", exc)
        return None


def _load_coords(config: dict):
    """Load spatial coordinates from the data file in config.  Returns None on failure."""
    try:
        data_cfg = config.get("data") or {}
        data_path = data_cfg.get("file_path")
        coord_cols = data_cfg.get("coord_columns") or ["POINT_X", "POINT_Y"]
        if not data_path or not os.path.exists(str(data_path)):
            return None
        import pandas as pd

        available = pd.read_csv(data_path, nrows=0).columns.tolist()
        valid_cols = [c for c in coord_cols if c in available]
        if len(valid_cols) < 2:
            return None
        df = pd.read_csv(data_path, usecols=valid_cols, nrows=5000)
        return df[valid_cols].dropna().values
    except Exception:
        return None
