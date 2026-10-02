"""RunReader: lazily loaded, cached views of one run directory (SPEC §6.2).

A :class:`RunContext` per run, cached in a byte-capped LRU (768 MB across runs) and keyed by the run id
plus the stat of ``manifest.json``, ``run_state.json``, ``predictions.parquet``, ``launch.json`` and the
snapshot's data file - a finished run keeps one context, a running one gets a fresh context whenever a stage
writes.  Attributes:

* ``cfg`` - the run's config, resolved in the order of SPEC §4.3: ``launch.json`` (config_raw +
  config_dir, mode overrides from its args) → ``manifest.config`` + ``provenance.config_dir`` → the
  config path given at import.  A candidate whose data file is missing yields to the next one.
* ``manifest`` - the manifest **merged with the per-stage files** (``causal.json``, ``optimize.json``,
  ``cv_distance.json``, ``baselines.json``, ``climate.json``, ``response_curves.json``), with ``sections``
  tagging each ``{present, source, stale, older_code}``.  A file section is stale when the file is more
  than 60 s older than ``manifest.created_utc`` and the manifest lacks the section.  Older code is detected
  by feature presence (provenance, literature, interval_diagnostics, mean_delta_se, frac_sigmoid,
  inflection_dose, qa.flags).
* ``data`` - ``load_core_data`` with coarse/subsample from ``launch.json`` args, ``run_state.meta`` or
  ``manifest.qa`` (``input_frame.parquet`` for frame-input runs); ids are checked against
  ``predictions.parquet``.  When it cannot be rebuilt (``data_error``) predictor layers are hidden and the
  prediction layers still work.
* ``grid`` - :class:`~sparc.studio.runs.grid.RunGrid` from ``predictions.parquet`` (or the data mid-run),
  cached in ``<studio_dir>/cache/grid.npz``.
* ``folds`` - ``cv.make_spatial_folds`` from the manifest ``cv`` section or ``run_state.meta.cv``
  (deterministic), cross-checked against ``predictions.fold``.  The checkpoint is never unpickled.
* ``metrics`` / ``metrics_live`` - the manifest metrics, or R², RMSE, MAE, bias, interval coverage and
  per-model OOF metrics computed from ``predictions.parquet`` while the run continues.
"""

from __future__ import annotations

import copy
import logging
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from sparc.studio.runs.common import clean, file_stat, fnum, read_json_cached
from sparc.studio.workspace import parse_utc

log = logging.getLogger("sparc.studio.runs")

__all__ = ["RunContext", "RunReader", "FILE_SECTIONS", "STALE_S", "load_config_raw", "core_cfg"]

STALE_S = 60.0
CONTEXT_BYTES = 768 * 1024 ** 2
#: manifest section → stage file that carries it when the manifest does not (SPEC §6.2)
FILE_SECTIONS = {"causal": "causal.json", "optimize": "optimize.json", "cv_distance": "cv_distance.json",
                 "baselines": "baselines.json", "climate": "climate.json", "response": "response_curves.json"}
_FEATURES = {"provenance": "provenance", "literature": "literature"}


def _cfg_from_raw(raw: dict, base_dir):
    """``core_config_from_dict`` that tolerates a forcing file that moved (its values are in the raw config)."""
    from sparc.core.config import core_config_from_dict

    try:
        return core_config_from_dict(copy.deepcopy(raw), base_dir=base_dir)
    except (OSError, ValueError) as exc:
        phys = (raw.get("core", raw).get("physics") or {}) if isinstance(raw, dict) else {}
        if not phys.get("forcing"):
            raise
        raw2 = copy.deepcopy(raw)
        block = raw2.get("core", raw2)
        block["physics"] = {k: v for k, v in block["physics"].items() if k != "forcing"}
        log.info("config forcing file unavailable (%s); using the stored physics values", exc)
        return core_config_from_dict(raw2, base_dir=base_dir)


def load_config_raw(path) -> dict:
    """The raw ``core`` block of a YAML config file."""
    import yaml

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return raw.get("core", raw) if isinstance(raw, dict) else {}


def core_cfg(raw: dict, base_dir):
    return _cfg_from_raw(raw, base_dir)


class RunContext:
    """Cached, lazily computed views of one run (see the module docstring)."""

    def __init__(self, row: dict, key: tuple):
        self.row = row
        self.run_id: str = row["id"]
        self.run_dir = Path(row["run_dir"])
        self.studio_dir = Path(row["studio_dir"]) if row.get("studio_dir") else self.run_dir / "studio"
        self.key = key
        self._lock = threading.RLock()
        self._c: dict[str, Any] = {}
        self.launch: dict | None = read_json_cached(self.studio_dir / "launch.json")
        self.import_rec: dict | None = read_json_cached(self.studio_dir / "import.json")
        self.run_state: dict | None = read_json_cached(self.run_dir / "run_state.json")
        self.checkpoint_json: dict | None = read_json_cached(self.run_dir / "checkpoint.json")
        self.manifest_raw: dict | None = read_json_cached(self.run_dir / "manifest.json")
        self.layer_keys: set[str] = set()

    # ------------------------------------------------------------------ helpers

    def _get(self, name: str, fn):
        with self._lock:
            if name not in self._c:
                self._c[name] = fn()
            return self._c[name]

    def path(self, rel: str) -> Path:
        return self.run_dir / rel

    def exists(self, rel: str) -> bool:
        return (self.run_dir / rel).exists()

    def json(self, rel: str, default=None):
        return read_json_cached(self.run_dir / rel, default)

    def parquet(self, rel: str, columns: list[str] | None = None):
        """A parquet file of the run as a DataFrame (cached by file stat; ``columns`` projects)."""
        import pandas as pd

        p = self.run_dir / rel
        st = file_stat(p)
        if st is None:
            return None
        ck = f"pq:{rel}:{st}"
        with self._lock:
            df = self._c.get(ck)
        if df is None:
            try:
                df = pd.read_parquet(p)
            except Exception as exc:          # a file being written, or unreadable
                log.warning("cannot read %s: %s", p, exc)
                return None
            with self._lock:
                self._c[ck] = df
        if columns is not None:
            return df[[c for c in columns if c in df.columns]]
        return df

    @property
    def nbytes(self) -> int:
        total = 0
        with self._lock:
            items = list(self._c.items())
        for k, v in items:
            if k.startswith("pq:"):
                try:
                    total += int(v.memory_usage(deep=False).sum())
                except Exception:
                    pass
            elif hasattr(v, "nbytes"):
                total += int(getattr(v, "nbytes") or 0)
            elif k == "data" and v is not None:
                try:
                    total += int(v.frame.memory_usage(deep=False).sum()) * 2
                except Exception:
                    pass
        return total

    # ------------------------------------------------------------------ identity

    @property
    def status(self) -> str:
        return str(self.row.get("status") or "")

    @property
    def finished(self) -> bool:
        """Outputs no longer change: immutable caching applies (api.md §0.5).  Partial, interrupted, failed and
        cancelled runs can be resumed in place, which rewrites files under the same URLs, so they stay
        ``no-store``."""
        return self.status in ("complete", "imported")

    @property
    def owned(self) -> bool:
        """Studio wrote this run (launched it, or a study did): its folder changes only through Studio.  A run
        imported in place may be rewritten by another CLI run into the same folder."""
        return self.row.get("origin") in ("studio", "study_child", "reproduction")

    @property
    def name(self) -> str:
        m = self.manifest_raw or {}
        if m.get("name"):
            return str(m["name"])
        raw = (self.launch or {}).get("config_raw") or {}
        raw = raw.get("core", raw)
        return str(raw.get("name") or self.row.get("label") or self.run_id)

    @property
    def project_id(self) -> str | None:
        return self.row.get("project_id")

    # ------------------------------------------------------------------ config

    def _cfg_candidates(self) -> list[tuple[str, Any, str]]:
        """``(source, cfg, config_dir)`` candidates in SPEC §4.3 order (unusable ones skipped)."""
        out = []
        launch = self.launch or {}
        if launch.get("config_raw") is not None:
            try:
                from sparc.core.pipeline import apply_mode_overrides

                base = launch.get("config_dir") or str(self.run_dir)
                cfg = _cfg_from_raw(launch["config_raw"], base)
                a = launch.get("args") or {}
                apply_mode_overrides(cfg, fast=bool(a.get("fast")), coarse=a.get("coarse"), cv_curve=a.get("cv_curve"),
                                     frame_given=self.exists("input_frame.parquet"))
                out.append(("launch", cfg, str(base)))
            except Exception as exc:
                log.warning("%s: launch.json config unusable: %s", self.run_id, exc)
        m = self.manifest_raw or {}
        if isinstance(m.get("config"), dict):
            base = (m.get("provenance") or {}).get("config_dir")
            if base and Path(base).is_dir():
                try:
                    out.append(("manifest", _cfg_from_raw(m["config"], base), str(base)))
                except Exception as exc:
                    log.warning("%s: manifest config unusable: %s", self.run_id, exc)
        imp = (self.import_rec or {}).get("config_path")
        if imp and Path(imp).is_file():
            try:
                raw = load_config_raw(imp)
                cfg = _cfg_from_raw(raw, Path(imp).resolve().parent)
                self._apply_manifest_modes(cfg)
                out.append(("import", cfg, str(Path(imp).resolve().parent)))
            except Exception as exc:
                log.warning("%s: import config %s unusable: %s", self.run_id, imp, exc)
        if not out and isinstance(m.get("config"), dict):
            base = (m.get("provenance") or {}).get("config_dir") or str(self.run_dir)
            try:
                out.append(("manifest", _cfg_from_raw(m["config"], base), str(base)))
            except Exception as exc:
                log.warning("%s: manifest config unusable: %s", self.run_id, exc)
        return out

    def _apply_manifest_modes(self, cfg) -> None:
        from sparc.core.pipeline import apply_mode_overrides

        m = self.manifest_raw or {}
        co = (m.get("qa") or {}).get("coarse")
        apply_mode_overrides(cfg, fast=bool(m.get("fast_mode")), coarse=(co or {}).get("cell_m"),
                             frame_given=self.exists("input_frame.parquet"))

    def _cfg_info(self):
        cands = self._cfg_candidates()
        if not cands:
            return None, None, None
        frame = self.exists("input_frame.parquet")
        for src, cfg, base in cands:
            try:
                if frame or Path(cfg.data_path).is_file():
                    return cfg, src, base
            except ValueError:
                continue
        src, cfg, base = cands[0]
        return cfg, src, base

    @property
    def cfg(self):
        return self._get("cfg_info", self._cfg_info)[0]

    @property
    def cfg_source(self) -> str | None:
        return self._get("cfg_info", self._cfg_info)[1]

    @property
    def config_dir(self) -> str | None:
        return self._get("cfg_info", self._cfg_info)[2]

    @property
    def cfg_raw(self) -> dict:
        cfg = self.cfg
        if cfg is not None:
            return cfg.raw
        m = self.manifest_raw or {}
        if isinstance(m.get("config"), dict):
            return m["config"]
        raw = (self.launch or {}).get("config_raw") or {}
        return raw.get("core", raw)

    @property
    def args(self) -> dict:
        """Mode arguments: ``launch.json`` args, else derived from the manifest."""
        a = dict((self.launch or {}).get("args") or {})
        if a:
            return a
        m = self.manifest_raw or {}
        co = (m.get("qa") or {}).get("coarse")
        return {"stages": None, "fast": bool(m.get("fast_mode")), "coarse": (co or {}).get("cell_m"),
                "cv_curve": None, "threads": None}

    @property
    def target_units(self) -> str:
        return str(((self.cfg_raw.get("data") or {}).get("target_units")) or "degF")

    @property
    def units(self) -> dict:
        from sparc.core.catalog import units_for

        return units_for(self.cfg_raw)

    @property
    def mode(self) -> str:
        a = self.args
        if a.get("coarse"):
            return "coarse"
        if a.get("fast"):
            return "fast"
        st = a.get("stages")
        if st and set(st) != {"S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"}:
            return "custom"
        return "full"

    # ------------------------------------------------------------------ manifest

    def _merged(self) -> tuple[dict, dict]:
        m = copy.deepcopy(self.manifest_raw) if isinstance(self.manifest_raw, dict) else {}
        created = parse_utc(m.get("created_utc")) if m else None
        sections: dict[str, dict] = {}
        for k in m:
            sections[k] = {"present": True, "source": "manifest", "stale": False, "older_code": False}
        for key, rel in FILE_SECTIONS.items():
            if key in m:
                continue
            obj = self.json(rel)
            if obj is None:
                continue
            if key == "response" and isinstance(obj, dict):
                obj = {v: (d or {}).get("summary") for v, d in obj.items() if isinstance(d, dict)}
            m[key] = obj
            stale = False
            if created is not None:
                try:
                    stale = os.stat(self.run_dir / rel).st_mtime < created - STALE_S
                except OSError:
                    stale = False
            sections[key] = {"present": True, "source": "file", "stale": stale, "older_code": False}
        if self.manifest_raw:
            older = self._older_code(m)
            for sec, flag in older.items():
                if flag:
                    sections.setdefault(sec, {"present": sec in m, "source": "manifest" if sec in m else None,
                                              "stale": False, "older_code": True})["older_code"] = True
        return m, sections

    def _older_code(self, m: dict) -> dict:
        """Sections written by older core code, by feature presence (SPEC §6.2)."""
        out = {"provenance": "provenance" not in m, "literature": "literature" not in m}
        st = (m.get("metrics") or {}).get("stacker") or {}
        out["metrics"] = bool(st) and "interval_diagnostics" not in st
        sc = m.get("scenarios") or []
        out["scenarios"] = bool(sc) and isinstance(sc, list) and "mean_delta_se" not in (sc[0] or {})
        resp = m.get("response") or {}
        out["response"] = (bool(resp) and any(isinstance(v, dict) and "frac_sigmoid" not in v for v in resp.values())
                           or self._maps_lack("inflection_dose"))
        qa = m.get("qa") or {}
        out["qa"] = bool(qa) and "flags" not in qa
        return out

    def _maps_lack(self, column: str) -> bool:
        """Whether some ``response_<var>.parquet`` lacks ``column`` (read from the parquet schema only)."""
        import pyarrow.parquet as pq

        for p in self.run_dir.glob("response_*.parquet"):
            try:
                if column not in pq.read_schema(p).names:
                    return True
            except Exception:
                continue
        return False

    @property
    def manifest(self) -> dict:
        return self._get("merged", self._merged)[0]

    @property
    def sections(self) -> dict:
        return self._get("merged", self._merged)[1]

    @property
    def created_ts(self) -> float | None:
        return parse_utc((self.manifest_raw or {}).get("created_utc"))

    def stale_file(self, rel: str, manifest_key: str | None) -> bool:
        """SPEC §6.1: older than the manifest by > 60 s and the manifest lacks the section."""
        if not manifest_key or not self.manifest_raw:
            return False
        if manifest_key in self.manifest_raw:
            return False
        created = self.created_ts
        if created is None:
            return False
        try:
            return os.stat(self.run_dir / rel).st_mtime < created - STALE_S
        except OSError:
            return False

    # ------------------------------------------------------------------ predictions and metrics

    @property
    def predictions(self):
        return self.parquet("predictions.parquet")

    @property
    def meta(self) -> dict:
        return dict((self.run_state or {}).get("meta") or {})

    @property
    def cell_m(self) -> float | None:
        for v in (self.meta.get("cell_m"), (self.manifest_raw or {}).get("qa", {}).get("cell_m"),
                  (self.json("influence.json") or {}).get("cell_m")):
            if fnum(v):
                return float(v)
        d = self.data
        return float(d.grid.dx) if d is not None else None

    @property
    def cv_meta(self) -> dict | None:
        """``{n_folds, block_m, buffer_m, seed}`` from the manifest or ``run_state.meta.cv``."""
        cv = (self.manifest_raw or {}).get("cv") or self.meta.get("cv")
        if not cv or not cv.get("n_folds") or cv.get("block_m") is None:
            return None
        seed = cv.get("seed")
        if seed is None:
            seed = ((self.cfg_raw.get("cv") or {}).get("seed", 42))
        return {"n_folds": int(cv["n_folds"]), "block_m": float(cv["block_m"]),
                "buffer_m": float(cv.get("buffer_m") if cv.get("buffer_m") is not None else cv["block_m"] / 3.0),
                "seed": int(seed)}

    @property
    def metrics(self) -> dict | None:
        """Manifest metrics, else the live metrics from ``predictions.parquet``."""
        m = (self.manifest_raw or {}).get("metrics")
        if isinstance(m, dict) and m:
            return m
        return self.metrics_live

    @property
    def metrics_are_live(self) -> bool:
        m = (self.manifest_raw or {}).get("metrics")
        return not (isinstance(m, dict) and m) and self.metrics_live is not None

    @property
    def metrics_live(self) -> dict | None:
        return self._get("metrics_live", self._metrics_live)

    def _metrics_live(self) -> dict | None:
        pred = self.predictions
        if pred is None or "target" not in pred or "pred" not in pred:
            return None
        y = pred["target"].to_numpy(float)

        def met(p):
            ok = np.isfinite(p) & np.isfinite(y)
            if ok.sum() < 2:
                return None
            r = y[ok] - p[ok]
            sst = float(np.sum((y[ok] - y[ok].mean()) ** 2))
            return {"rmse": float(np.sqrt(np.mean(r ** 2))), "mae": float(np.mean(np.abs(r))),
                    "r2": float(1 - np.sum(r ** 2) / sst) if sst > 0 else None, "bias": float(-r.mean()),
                    "n": int(ok.sum())}

        out = {}
        for c in pred.columns:
            if c.startswith("oof_"):
                mm = met(pred[c].to_numpy(float))
                if mm:
                    out[c[4:]] = mm
        st = met(pred["pred"].to_numpy(float))
        if st is None:
            return None
        if "pi_lo" in pred and "pi_hi" in pred:
            lo, hi = pred["pi_lo"].to_numpy(float), pred["pi_hi"].to_numpy(float)
            ok = np.isfinite(lo) & np.isfinite(hi)
            st["interval_coverage"] = float(np.mean((y[ok] >= lo[ok]) & (y[ok] <= hi[ok])))
            st["interval_mean_halfwidth"] = float(np.mean((hi[ok] - lo[ok]) / 2.0))
            st["interval_target"] = float((self.cfg_raw.get("stacker") or {}).get("coverage", 0.9))
            if "pi_lo_adaptive" in pred and "pi_hi_adaptive" in pred:
                la, ha = pred["pi_lo_adaptive"].to_numpy(float), pred["pi_hi_adaptive"].to_numpy(float)
                oka = np.isfinite(la) & np.isfinite(ha)
                st["interval_coverage_adaptive"] = float(np.mean((y[oka] >= la[oka]) & (y[oka] <= ha[oka])))
        out["stacker"] = st
        return out

    @property
    def headline_metrics(self) -> dict:
        """``{r2, rmse, coverage}`` of the stack (manifest or live)."""
        st = (self.metrics or {}).get("stacker") or {}
        return {"r2": fnum(st.get("r2")), "rmse": fnum(st.get("rmse")), "coverage": fnum(st.get("interval_coverage"))}

    # ------------------------------------------------------------------ data

    @property
    def data(self):
        return self._get("data", self._load_data)

    @property
    def data_error(self) -> str | None:
        self.data
        return self._c.get("data_error")

    def _load_data(self):
        cfg = self.cfg
        prov = (self.manifest_raw or {}).get("provenance") or {}
        frame_file = prov.get("input_frame") or "input_frame.parquet"
        if cfg is None:
            self._c["data_error"] = "no usable config for this run"
            return None
        if prov.get("input_kind") == "frame" and not self.exists(frame_file):
            self._c["data_error"] = "inputs not reproducible: an in-memory frame run without input_frame.parquet"
            return None
        c = copy.deepcopy(cfg)
        qa = (self.manifest_raw or {}).get("qa") or {}
        meta = self.meta
        co = (qa.get("coarse") or {}).get("cell_m") or meta.get("coarse_m")
        if co:
            c.raw["data"]["coarse_m"] = float(co)
        win = qa.get("subsample_window_n") or meta.get("subsample_window_n")
        if win and not c.data.get("subsample"):
            c.raw["data"]["subsample"] = int(win)
        try:
            from threadpoolctl import threadpool_limits

            with threadpool_limits(1):
                if self.exists(frame_file):
                    import pandas as pd

                    from sparc.core.data import prepare_frame

                    data = prepare_frame(pd.read_parquet(self.run_dir / frame_file), c)
                else:
                    from sparc.core.data import load_core_data

                    data = load_core_data(c)
        except Exception as exc:
            self._c["data_error"] = f"the run's data cannot be rebuilt: {type(exc).__name__}: {exc}"[:500]
            return None
        pred = self.predictions
        if pred is not None and "id" in pred:
            pid = pred["id"].to_numpy()
            if pid.shape != np.asarray(data.ids).shape or not np.array_equal(pid.astype(str), np.asarray(data.ids).astype(str)):
                self._c["data_error"] = "the data rebuilt from the config does not match predictions.parquet (ids differ)"
                return None
        return data

    # ------------------------------------------------------------------ geometry

    def _raw_xy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any] | None:
        """``(x_m, y_m, ids, zones)`` in row order: predictions when present, else the data."""
        pred = self.predictions
        if pred is not None and {"x_m", "y_m", "id"} <= set(pred.columns):
            zones = pred["zone"].to_numpy() if "zone" in pred.columns else None
            return pred["x_m"].to_numpy(float), pred["y_m"].to_numpy(float), pred["id"].to_numpy(), zones
        data = self.data
        if data is None:
            return None
        return np.asarray(data.x, float), np.asarray(data.y_coord, float), np.asarray(data.ids), data.zones

    @property
    def xy(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Raw row-order coordinates (run metres) - the ones the folds were built from."""
        r = self._get("raw_xy", self._raw_xy)
        return None if r is None else (r[0], r[1])

    def _grid_key(self) -> str:
        m = self.manifest_raw or {}
        stamp = m.get("created_utc") or (self.run_state or {}).get("started_utc") or ""
        st = file_stat(self.run_dir / "predictions.parquet")
        return f"{stamp}|{st[0] if st else 'nopred'}"

    @property
    def grid(self):
        return self._get("grid", self._load_grid)

    def _load_grid(self):
        from sparc.studio.runs import grid as gridmod

        key = self._grid_key()
        g = gridmod.load_cached(self.studio_dir, key)
        if g is not None:
            return g
        raw = self._get("raw_xy", self._raw_xy)
        if raw is None:
            return None
        x, y, ids, zones = raw
        g = gridmod.build_grid(x, y, self.cell_m, ids, zones, self.cfg_raw, key=key)
        gridmod.save_cached(self.studio_dir, g)
        return g

    @property
    def n(self) -> int | None:
        g = self.grid
        return None if g is None else g.n

    # ------------------------------------------------------------------ folds

    @property
    def folds(self):
        return self._get("folds", self._load_folds)

    @property
    def folds_error(self) -> str | None:
        self.folds
        return self._c.get("folds_error")

    def _load_folds(self):
        cv = self.cv_meta
        xy = self.xy
        if cv is None or xy is None:
            self._c["folds_error"] = "the CV design is not known yet"
            return None
        from sparc.core.cv import make_spatial_folds

        try:
            folds = make_spatial_folds(np.column_stack(xy), n_folds=cv["n_folds"], block_m=cv["block_m"],
                                       buffer_m=cv["buffer_m"], seed=cv["seed"])
        except Exception as exc:
            self._c["folds_error"] = f"the folds cannot be rebuilt: {exc}"
            return None
        pred = self.predictions
        if pred is not None and "fold" in pred and not np.array_equal(folds.fold_id, pred["fold"].to_numpy()):
            self._c["folds_error"] = "rebuilt folds differ from predictions.fold (different cv.seed?)"
            return None
        return folds

    # ------------------------------------------------------------------ misc

    def done_set(self) -> list[str]:
        st = self.run_state or {}
        ck = self.checkpoint_json or {}
        return sorted(set(st.get("done") or []) | set(ck.get("done") or []))

    def response_vars(self) -> list[str]:
        """Levers with a ``response_<var>.parquet``, in config order."""
        levers = list((self.cfg_raw.get("actionable") or {}))
        out = [v for v in levers if self.exists(f"response_{v}.parquet")]
        for p in sorted(self.run_dir.glob("response_*.parquet")):
            v = p.stem[len("response_"):]
            if v not in out:
                out.append(v)
        return out

    def configured_scenarios(self) -> list[dict]:
        """``[{name, slug, index}]`` of the S5 scenarios (scenario_deltas columns), slugs unique."""
        from sparc.core.catalog import scenario_slug

        names: list[str] = []
        sc = self.manifest.get("scenarios")
        if isinstance(sc, list) and sc:
            names = [str(s.get("name")) for s in sc if isinstance(s, dict) and s.get("name")]
        if not names:
            df = self.parquet("scenario_deltas.parquet")
            if df is not None:
                names = [c for c in df.columns if c != "id"]
        taken: set[str] = set()
        out = []
        for i, nm in enumerate(names):
            slug = scenario_slug(nm, taken)
            taken.add(slug)
            out.append({"name": nm, "slug": slug, "index": i})
        return out

    def scenario_detail(self) -> dict | None:
        """``{names, folds: {name: K×n}, sd: {name: n}, ex: {name: n}}`` from ``scenario_detail.npz``."""
        return self._get("scenario_detail", self._load_detail)

    def _load_detail(self):
        import json as _json

        p = self.run_dir / "scenario_detail.npz"
        if not p.exists():
            return None
        try:
            with np.load(p, allow_pickle=False) as z:
                names = z["names"]
                names = _json.loads(bytes(names).decode("utf-8")) if names.dtype == np.uint8 else [str(x) for x in names]
                out = {"names": names, "ids": z["ids"], "folds": {}, "sd": {}, "ex": {}}
                for i, nm in enumerate(names):
                    if f"f{i}" in z.files:
                        out["folds"][nm] = np.asarray(z[f"f{i}"], np.float32)
                    if f"sd{i}" in z.files:
                        out["sd"][nm] = np.asarray(z[f"sd{i}"], np.float32)
                    if f"ex{i}" in z.files:
                        out["ex"][nm] = np.asarray(z[f"ex{i}"], np.float32)
                return out
        except Exception as exc:
            log.warning("%s: unreadable scenario_detail.npz: %s", self.run_id, exc)
            return None

    def to_json(self, obj):
        return clean(obj)


class RunReader:
    """The process-wide cache of :class:`RunContext` objects (byte-capped LRU)."""

    def __init__(self, db, *, max_bytes: int = CONTEXT_BYTES):
        self.db = db
        self.max_bytes = max_bytes
        self._ctx: "OrderedDict[str, RunContext]" = OrderedDict()
        self._lock = threading.Lock()

    def _key(self, row: dict) -> tuple:
        run_dir = Path(row["run_dir"])
        studio = Path(row["studio_dir"]) if row.get("studio_dir") else run_dir / "studio"
        launch = read_json_cached(studio / "launch.json") or {}
        raw = launch.get("config_raw") if isinstance(launch.get("config_raw"), dict) else {}
        data_path = (raw.get("data") or {}).get("path") if isinstance(raw.get("data"), dict) else None
        return (row["id"], row.get("status"), file_stat(run_dir / "manifest.json"),
                file_stat(run_dir / "run_state.json"), file_stat(run_dir / "predictions.parquet"),
                file_stat(run_dir / "checkpoint.json"), file_stat(studio / "launch.json"),
                file_stat(studio / "import.json"),
                # the snapshot's data file (absolute): a replaced file rebuilds the data and input layers
                file_stat(data_path) if isinstance(data_path, str) and os.path.isabs(data_path) else None)

    def row(self, run_id: str) -> dict:
        from sparc.studio.errors import ApiError

        row = self.db.fetchone("SELECT * FROM runs WHERE id = ?", (run_id,))
        if row is None:
            raise ApiError("not_found", f"no run {run_id!r}")
        return row

    def get(self, run_id: str) -> RunContext:
        """The current context of ``run_id`` (``404 not_found`` when the run is not indexed)."""
        return self.for_row(self.row(run_id))

    def for_row(self, row: dict) -> RunContext:
        key = self._key(row)
        with self._lock:
            ctx = self._ctx.get(row["id"])
            if ctx is not None and ctx.key == key:
                self._ctx.move_to_end(row["id"])
                ctx.row = row
                return ctx
        ctx = RunContext(row, key)
        with self._lock:
            self._ctx[row["id"]] = ctx
            self._ctx.move_to_end(row["id"])
            self._evict()
        return ctx

    def _evict(self) -> None:
        total = sum(c.nbytes for c in self._ctx.values())
        while total > self.max_bytes and len(self._ctx) > 1:
            _, old = self._ctx.popitem(last=False)
            total -= old.nbytes

    def forget(self, run_id: str) -> None:
        with self._lock:
            self._ctx.pop(run_id, None)
