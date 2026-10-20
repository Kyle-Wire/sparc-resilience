"""The QA registry of the LST pilot: verified faults of single campaigns that no generic filter catches.

A rain shower over one route, a sensor still warming up at the start of a run, a run driven through rain: the
review of the 31 built cities found these in the raw OSF files, and a second, skeptical reading confirmed each
one.  They are listed in ``exclusions.yml`` next to this module (one entry per fault, with its reason and a line
of evidence) and applied by the dataset build, so a rebuild repeats them and ``meta.json`` says what was dropped:

* ``kind: readings`` drops one sensor's readings of one run, at or after ``from_local`` and before
  ``until_local`` (local clock HH:MM; a missing bound is the run's start or end): :func:`reading_mask`.  The
  sensor is matched on its id as the reader parses it (:func:`sparc.core.identify.traverses.sensor_ids`:
  ``vehicle_id``, else the part of ``vehicle_key`` after its last ':'), never on a substring of a file name, and
  the run on its id ``"<date> <window>"`` (:func:`~sparc.core.identify.traverses.assign_runs`).
* ``kind: run`` excludes a whole run by default: :func:`run_exclusions` gives ``run_id → reason``, and the build
  writes :data:`PREFIX` + reason into the run's ``qa_exclude`` (``table.parquet`` rows and ``meta["runs"]``),
  which :mod:`~sparc.core.lstair.evaluate` drops unless asked to include flagged runs.

The registry is checked when it is read (:func:`load_registry`): an unknown key, kind or window, a date that is not
ISO, or a time that is not a quoted ``"HH:MM"`` (YAML reads an unquoted 15:05 as the number 905) raises
ValueError instead of silently matching nothing.
"""

from __future__ import annotations

import datetime as dt
import functools
import re
from pathlib import Path

import numpy as np
import pandas as pd

#: The registry shipped with the package.
REGISTRY = Path(__file__).with_name("exclusions.yml")
#: How the build prefixes a registry reason in ``qa_exclude`` (e.g. "qa registry: rain over 5 of 7 routes ...").
PREFIX = "qa registry: "
KINDS = ("readings", "run")
WINDOWS = ("morning", "midday", "evening", "night")
FIELDS = ("slug", "kind", "sensor", "date", "window", "from_local", "until_local", "reason", "evidence")
_REQUIRED = ("slug", "kind", "date", "window", "reason", "evidence")
_HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


def _check(entry: object, i: int, src: str) -> dict:
    """One registry entry, validated, with its date as an ISO string."""
    where = f"{src}, entry {i + 1}"
    if not isinstance(entry, dict):
        raise ValueError(f"{where}: an entry is a mapping of {', '.join(FIELDS)}")
    unknown = sorted(set(entry) - set(FIELDS))
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown} (known: {', '.join(FIELDS)})")
    missing = [k for k in _REQUIRED if entry.get(k) in (None, "")]
    if missing:
        raise ValueError(f"{where}: missing {missing}")
    e = dict(entry)
    if e["kind"] not in KINDS:
        raise ValueError(f"{where}: kind {e['kind']!r} is not one of {KINDS}")
    if e["window"] not in WINDOWS:
        raise ValueError(f"{where}: window {e['window']!r} is not one of {WINDOWS}")
    d = e["date"]
    try:
        e["date"] = (d if isinstance(d, dt.date) else dt.date.fromisoformat(str(d))).isoformat()
    except ValueError:
        raise ValueError(f"{where}: date {d!r} is not YYYY-MM-DD") from None
    for k in ("from_local", "until_local"):
        if k not in e or e[k] is None:
            e.pop(k, None)
            continue
        if e["kind"] == "run":
            raise ValueError(f"{where}: a run entry excludes the whole run; {k} applies to readings entries only")
        if not isinstance(e[k], str) or not _HHMM.match(e[k]):
            raise ValueError(f"{where}: {k} {e[k]!r} is not a quoted local clock time \"HH:MM\"")
    if e["kind"] == "readings":
        if not str(e.get("sensor") or "").strip():
            raise ValueError(f"{where}: a readings entry names its sensor (the id as parsed, e.g. CAPA2037)")
        e["sensor"] = str(e["sensor"]).strip()
        if "from_local" in e and "until_local" in e and _bound(e, "from_local") >= _bound(e, "until_local"):
            raise ValueError(f"{where}: from_local {e['from_local']} is not before until_local {e['until_local']}")
    return e


def _load(path: Path) -> tuple[dict, ...]:
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if not isinstance(raw, list):
        raise ValueError(f"{path}: the registry is a list of entries")
    return tuple(_check(e, i, path.name) for i, e in enumerate(raw))


@functools.lru_cache(maxsize=1)
def _shipped() -> tuple[dict, ...]:
    return _load(REGISTRY)


def load_registry(path: str | Path | None = None) -> list[dict]:
    """Every entry of the registry at ``path`` (default :data:`REGISTRY`), validated."""
    return [dict(e) for e in (_shipped() if path is None else _load(Path(path)))]


def entries(slug: str, registry: list[dict] | None = None) -> list[dict]:
    """The registry's entries for one city."""
    return [e for e in (load_registry() if registry is None else registry) if e["slug"] == slug]


def _bound(e: dict, key: str) -> pd.Timestamp | None:
    """A from/until bound as a local timestamp on the run's date (a night run's times before noon are the next
    day's: the run starts before midnight)."""
    if key not in e:
        return None
    ts = pd.Timestamp(f"{e['date']} {e[key]}")
    return ts + pd.Timedelta(days=1) if e["window"] == "night" and ts.hour < 12 else ts


def reading_mask(points: pd.DataFrame, slug: str, registry: list[dict] | None = None) -> tuple[np.ndarray, list[dict]]:
    """Which traverse points to keep: False for the readings a ``kind: readings`` entry of ``slug`` drops.

    ``points`` are the reader's (:func:`~sparc.core.identify.traverses.scan_traverses`: ``time`` on the local
    clock, ``vehicle_key`` / ``vehicle_id``, ``run``; without a ``run`` column the runs are assigned here).
    Returns (keep, records): one record per entry, its fields plus ``run_id``, ``n_readings`` dropped and the
    ``first`` / ``last`` time dropped, so an entry that matches nothing (a sensor id parsed differently) shows as
    ``n_readings`` 0 rather than vanishing."""
    from sparc.core.identify.traverses import assign_runs, sensor_ids

    keep = np.ones(len(points), dtype=bool)
    todo = [e for e in entries(slug, registry) if e["kind"] == "readings"]
    if not todo or not len(points):
        return keep, [{**e, "run_id": f"{e['date']} {e['window']}", "n_readings": 0} for e in todo]
    t = pd.to_datetime(points["time"], errors="coerce")
    run = (points["run"] if "run" in points else assign_runs(points.assign(time=t))["run"]).astype(str).to_numpy()
    sid = sensor_ids(points).to_numpy()
    records = []
    for e in todo:
        run_id = f"{e['date']} {e['window']}"
        sel = (sid == e["sensor"]) & (run == run_id)
        lo, hi = _bound(e, "from_local"), _bound(e, "until_local")
        if lo is not None:
            sel &= (t >= lo).to_numpy()
        if hi is not None:
            sel &= (t < hi).to_numpy()
        keep &= ~sel
        rec = {**e, "run_id": run_id, "n_readings": int(sel.sum())}
        if sel.any():
            rec.update(first=str(t[sel].min()), last=str(t[sel].max()))
        records.append(rec)
    return keep, records


def run_exclusions(slug: str, registry: list[dict] | None = None) -> dict[str, str]:
    """``run_id → reason`` of the runs a ``kind: run`` entry of ``slug`` excludes (run ids ``"<date> <window>"`` as
    in ``meta["runs"]``; several entries for one run are joined with "; ").  The build writes
    :data:`PREFIX` + reason into the run's ``qa_exclude``."""
    out: dict[str, str] = {}
    for e in entries(slug, registry):
        if e["kind"] == "run":
            rid = f"{e['date']} {e['window']}"
            out[rid] = f"{out[rid]}; {e['reason']}" if rid in out else e["reason"]
    return out


__all__ = ["REGISTRY", "PREFIX", "load_registry", "entries", "reading_mask", "run_exclusions"]
