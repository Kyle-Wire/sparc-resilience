"""Project files (api.md §5.1): layout, listing, inspect, column suggestions and path resolution.

Uploads land by kind::

    data, join → data/        layers, features, forcing, climate → inputs/<kind>/        other → other/

Tables are CSV (read with ``utf-8-sig``, like core) or parquet.  Inspect
reads at most ``rows`` rows (``pandas.read_csv(nrows=…)`` / the first
parquet batches) and counts the rest without parsing it.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from sparc.studio.errors import ApiError
from sparc.studio.security import safe_path
from sparc.studio.workspace import utc_iso

__all__ = ["KIND_DIRS", "FILE_KINDS", "TABLE_SUFFIXES", "kind_dir", "in_kind_dirs", "resolve_path",
           "read_table_head", "read_columns", "count_rows", "inspect_file", "suggest_columns", "list_files", "used_by",
           "jsonable", "config_file_refs", "is_csv_name"]

KIND_DIRS = {"data": "data", "join": "data", "layers": "inputs/layers", "features": "inputs/features",
             "forcing": "inputs/forcing", "climate": "inputs/climate", "other": "other"}
FILE_KINDS = tuple(KIND_DIRS)
TABLE_SUFFIXES = (".csv", ".parquet")
PREVIEW_ROWS = 20
SAMPLE_VALUES = 5
_COUNT_LIMIT = 512 * 1024 ** 2          # count CSV lines exactly up to this size; estimate above


def kind_dir(project_dir: str | os.PathLike, kind: str) -> Path:
    if kind not in KIND_DIRS:
        raise ApiError("validation", f"unknown file kind {kind!r}",
                       detail={"errors": [{"path": "kind", "message": f"one of {', '.join(FILE_KINDS)}",
                                           "code": "unknown_kind"}]})
    return Path(project_dir) / KIND_DIRS[kind]


def in_kind_dirs(project_dir: str | os.PathLike, path: str | os.PathLike) -> bool:
    """Whether ``path`` lies in one of the project's file folders (``data/``, ``inputs/<kind>/``, ``other/``),
    i.e. it is a file ``GET /files`` lists - never ``config.yml``, ``project.json`` or a run folder."""
    pdir = Path(project_dir).resolve()
    target = Path(path).resolve()
    return any((pdir / sub).resolve() in target.parents for sub in set(KIND_DIRS.values()))


def resolve_path(project_dir: str | os.PathLike, raw: str, roots: Iterable[str | os.PathLike] = ()) -> Path:
    """A path parameter as a file path: relative paths resolve under the project folder (``safe_path``);
    absolute ones must lie inside the project or one of ``roots`` (folders registered at import)."""
    if raw is None or not str(raw).strip():
        raise ApiError("validation", "path is required",
                       detail={"errors": [{"path": "path", "message": "required", "code": "missing"}]})
    pdir = Path(project_dir).resolve()
    s = str(raw)
    if os.path.isabs(s):
        target = Path(s).resolve()
        allowed = [pdir, *[Path(r).resolve() for r in roots]]
        if not any(target == r or r in target.parents for r in allowed):
            raise ApiError("validation", f"unsafe path {s!r}: outside the allowed folders",
                           detail={"errors": [{"path": "path", "message": "outside the allowed folders",
                                               "code": "unsafe_path"}]})
        return target
    return safe_path(s, pdir)


# ---------------------------------------------------------------------------
# reading tables
# ---------------------------------------------------------------------------

_CSV_COMPRESSED = {".gz": "gzip", ".bz2": "bz2", ".xz": "lzma"}


def is_csv_name(name: str) -> bool:
    """A CSV (``.csv``, or ``.csv.gz|.bz2|.xz``, which ``pandas.read_csv`` - and so core - reads too)."""
    n = str(name).lower()
    return n.endswith(".csv") or any(n.endswith(".csv" + c) for c in _CSV_COMPRESSED)


def _suffix(path: Path) -> str:
    if is_csv_name(path.name):
        return ".csv"
    s = path.suffix.lower()
    if s not in TABLE_SUFFIXES:
        raise ApiError("bad_suffix", f"{path.name}: only CSV and parquet tables can be read",
                       detail={"suffix": s, "allowed": list(TABLE_SUFFIXES)})
    return s


def _open_raw(path: Path):
    """The file's bytes as a stream (decompressed for ``.csv.gz`` and friends)."""
    comp = _CSV_COMPRESSED.get(path.suffix.lower())
    if comp is None:
        return open(path, "rb")
    import importlib

    return importlib.import_module(comp).open(path, "rb")


def read_table_head(path: str | os.PathLike, rows: int | None = 50000, columns: list[str] | None = None) -> pd.DataFrame:
    """The first ``rows`` rows (all when None) of a CSV or parquet table."""
    path = Path(path)
    if not path.is_file():
        raise ApiError("not_found", f"no such file: {path.name}", detail={"path": str(path)})
    if _suffix(path) == ".csv":
        try:
            return pd.read_csv(path, encoding="utf-8-sig", nrows=rows, usecols=columns, low_memory=False)
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            raise ApiError("validation", f"{path.name} is not a readable CSV: {exc}",
                           detail={"errors": [{"path": "path", "message": str(exc)[:300], "code": "unreadable"}]})
    import pyarrow.parquet as pq

    try:
        pf = pq.ParquetFile(path)
        if rows is None:
            return pf.read(columns=columns).to_pandas()
        batches = []
        got = 0
        for b in pf.iter_batches(batch_size=min(max(rows, 1), 65536), columns=columns):
            batches.append(b)
            got += b.num_rows
            if got >= rows:
                break
        if not batches:
            return pf.schema_arrow.empty_table().to_pandas()
        import pyarrow as pa

        return pa.Table.from_batches(batches).slice(0, rows).to_pandas()
    except (OSError, ValueError) as exc:
        raise ApiError("validation", f"{path.name} is not a readable parquet file: {exc}",
                       detail={"errors": [{"path": "path", "message": str(exc)[:300], "code": "unreadable"}]})


def read_columns(path: str | os.PathLike) -> list[str]:
    """Column names of a table without reading its rows."""
    path = Path(path)
    if _suffix(path) == ".parquet":
        import pyarrow.parquet as pq

        return list(pq.ParquetFile(path).schema_arrow.names)
    return list(read_table_head(path, rows=0).columns)


def count_rows(path: str | os.PathLike) -> tuple[int, bool]:
    """``(n_rows, exact)``: parquet metadata; CSV newlines (estimated from the size above 512 MB)."""
    path = Path(path)
    if _suffix(path) == ".parquet":
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows), True
    size = path.stat().st_size
    compressed = path.suffix.lower() in _CSV_COMPRESSED
    if size > _COUNT_LIMIT and not compressed:
        with open(path, "rb") as f:
            head = f.read(4 * 1024 ** 2)
        lines = max(head.count(b"\n"), 1)
        return max(int(size / len(head) * lines) - 1, 0), False
    n = 0
    last = b""
    with _open_raw(path) as f:
        while True:
            chunk = f.read(4 * 1024 ** 2)
            if not chunk:
                break
            n += chunk.count(b"\n")
            last = chunk[-1:]
    if last and last != b"\n":
        n += 1                                   # no trailing newline
    return max(n - 1, 0), True                   # minus the header


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------

def jsonable(v: Any) -> Any:
    """A JSON-safe scalar (numpy → Python, NaN/Inf/NaT → None, timestamps → ISO)."""
    if v is None:
        return None
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        try:
            return None if pd.isna(v) else pd.Timestamp(v).isoformat()
        except (ValueError, TypeError):
            return str(v)
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", "replace")
    if isinstance(v, (np.ndarray, list, tuple)):          # parquet list cells
        return [jsonable(x) for x in v]
    if isinstance(v, dict):                                # parquet struct cells
        return {str(k): jsonable(x) for k, x in v.items()}
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v if isinstance(v, (int, str)) else str(v)


def _column_info(name: str, s: pd.Series) -> dict:
    numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
    vals = pd.to_numeric(s, errors="coerce") if numeric else None
    try:
        sample = [jsonable(v) for v in s.dropna().unique()[:SAMPLE_VALUES]]
        n_unique: int | None = int(s.nunique(dropna=True))
    except TypeError:                              # unhashable cells (parquet list/struct columns)
        sample = [jsonable(v) for v in s.dropna().head(SAMPLE_VALUES)]
        n_unique = None
    finite = vals[np.isfinite(vals.to_numpy(float))] if numeric else None
    return {"name": str(name), "dtype": str(s.dtype), "n_null": int(s.isna().sum()),
            "min": jsonable(finite.min()) if numeric and len(finite) else None,
            "max": jsonable(finite.max()) if numeric and len(finite) else None,
            "n_unique": n_unique, "sample": sample}


def inspect_file(path: str | os.PathLike, rows: int = 50000) -> dict:
    """``FileInspect`` (api.md §5.1) of a CSV or parquet table: stats over the first ``rows`` rows."""
    path = Path(path)
    rows = max(1, min(int(rows), 2_000_000))
    df = read_table_head(path, rows=rows)
    if len(df) < rows:
        n, exact = len(df), True
    else:
        n, exact = count_rows(path)
    preview = [[jsonable(v) for v in r] for r in df.head(PREVIEW_ROWS).itertuples(index=False, name=None)]
    return {"n_rows": int(n), "n_rows_exact": bool(exact),
            "columns": [_column_info(c, df[c]) for c in df.columns], "preview": preview}


# ---------------------------------------------------------------------------
# column suggestions
# ---------------------------------------------------------------------------

_TARGET = re.compile(r"(^t$|^tair$|temp|^aat|air_?t|^lst|uhi|target|^t_?(air|max|mean|2m)\b|heat_?index)", re.I)
_ID = re.compile(r"^(id|objectid|object_id|fid|gid|oid|ogc_fid|pointid|point_id|cell_?id|uid|index|pid)$", re.I)
_X = re.compile(r"^(x|point_x|x_?coord|coord_?x|easting|east|lon|lng|long|longitude|x_m|xm|utm_?e|xcoord)$", re.I)
_Y = re.compile(r"^(y|point_y|y_?coord|coord_?y|northing|north|lat|latitude|y_m|ym|utm_?n|ycoord)$", re.I)
_ZONE = re.compile(r"(zone|location|neighbou?rhood|district|ward|tract|block_?group|region|area_?code|nbhd)", re.I)
_ROLES = {"canopy": re.compile(r"(canopy|tree)", re.I),
          "impervious": re.compile(r"(imperv|paved|sealed|built)", re.I),
          "ndvi": re.compile(r"ndvi", re.I),
          "albedo": re.compile(r"(albedo|reflectance)", re.I),
          "elevation": re.compile(r"(elev|^dem$|altitude|^height)", re.I),
          "water_distance": re.compile(r"(water.*dist|dist.*water|d_?water)", re.I)}
_ROUND_M = (1.0, 2.0, 5.0, 10.0, 20.0, 25.0, 30.0, 50.0, 100.0, 200.0, 250.0, 500.0, 1000.0)


def _nunique(s: pd.Series) -> int | None:
    """Distinct non-null values; None for unhashable cells (parquet list/struct columns)."""
    try:
        return int(s.nunique(dropna=True))
    except TypeError:
        return None


def _best(cols: list[str], pat: re.Pattern, *, exclude: set[str] = frozenset()) -> str | None:
    hits = [c for c in cols if c not in exclude and pat.search(c)]
    if not hits:
        return None
    exact = [c for c in hits if pat.fullmatch(c)]
    return (exact or hits)[0]


def _round_error(v: float) -> float:
    return min(abs(v - m) / m for m in _ROUND_M)


def _lattice_spacing(x: np.ndarray, y: np.ndarray) -> float | None:
    ok = np.isfinite(x) & np.isfinite(y)
    pts = np.column_stack([x[ok], y[ok]])
    if len(pts) < 3:
        return None
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(0)
    idx = rng.choice(len(pts), size=min(2000, len(pts)), replace=False)
    d, _ = cKDTree(pts).query(pts[idx], k=2)
    nn = d[:, 1]
    nn = nn[nn > 1e-9]
    return float(np.median(nn)) if nn.size else None


def suggest_columns(path: str | os.PathLike, rows: int = 20000) -> dict:
    """Mapping suggestions for a new data table (api.md ``columns/suggest``).

    Name patterns pick target, id, x/y, zone and the physics roles; the
    coordinate unit comes from the lattice spacing (a spacing that is a round
    number of metres only in feet → US survey feet) and lon/lat ranges give
    ``crs_guess: EPSG:4326``.  ``confidence`` is 0–1 per suggestion.
    """
    df = read_table_head(path, rows=rows)
    cols = [str(c) for c in df.columns]
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    conf: dict[str, float] = {}

    x = _best(numeric, _X)
    y = _best(numeric, _Y, exclude={x} if x else set())
    conf["x"] = 0.9 if x else 0.0
    conf["y"] = 0.9 if y else 0.0
    ident = _best(cols, _ID)
    if ident is None:                               # an integer column with unique values
        for c in numeric:
            s = df[c].dropna()
            if len(s) and pd.api.types.is_integer_dtype(df[c]) and s.is_unique and c not in (x, y):
                ident = c
                break
        conf["id"] = 0.5 if ident else 0.0
    else:
        conf["id"] = 0.9 if _nunique(df[ident]) == int(df[ident].notna().sum()) else 0.5
    used = {c for c in (x, y, ident) if c}
    target = _best([c for c in numeric if c not in used], _TARGET)
    conf["target"] = 0.8 if target else 0.0
    if target is None:
        rest = [c for c in numeric if c not in used and not any(p.search(c) for p in _ROLES.values())]
        if rest:
            target, conf["target"] = rest[0], 0.2
    zone = None
    for c in cols:
        if c in used or c == target or not _ZONE.search(c):
            continue
        nun = _nunique(df[c])
        if nun is not None and 1 < nun <= max(200, int(0.05 * len(df))):
            zone = c
            break
    conf["zone"] = 0.7 if zone else 0.0
    used |= {c for c in (target, zone) if c}

    coord_unit, crs_guess = "m", None
    conf["coord_unit"], conf["crs"] = 0.3, 0.0
    if x and y:
        xs = pd.to_numeric(df[x], errors="coerce").to_numpy(float)
        ys = pd.to_numeric(df[y], errors="coerce").to_numpy(float)
        fin = np.isfinite(xs) & np.isfinite(ys)
        if fin.any() and np.nanmax(np.abs(xs[fin])) <= 180.0 and np.nanmax(np.abs(ys[fin])) <= 90.0:
            crs_guess, conf["crs"] = "EPSG:4326", 0.8
            conf["coord_unit"] = 0.2
        else:
            s = _lattice_spacing(xs, ys)
            if s:
                e_m, e_ft = _round_error(s), _round_error(s * 1200.0 / 3937.0)
                if e_ft < e_m and e_ft < 0.02:
                    coord_unit, conf["coord_unit"] = "us_survey_foot", 0.8
                elif e_m < 0.02:
                    conf["coord_unit"] = 0.8
                else:
                    conf["coord_unit"] = 0.4
    excluded = used | {c for c in cols if _ID.search(c)}
    predictors = []
    for c in numeric:
        if c in excluded:
            continue
        s = df[c]
        if s.isna().mean() > 0.5 or s.nunique(dropna=True) <= 1:
            continue
        predictors.append(c)
    roles: dict[str, str] = {}
    for role, pat in _ROLES.items():
        c = _best([p for p in predictors if p not in roles.values()], pat)
        if c:
            roles[role] = c
            conf[f"roles.{role}"] = 0.8
    return {"target": target, "id": ident, "x": x, "y": y, "zone": zone, "coord_unit": coord_unit,
            "crs_guess": crs_guess, "predictors": predictors, "roles": roles,
            "confidence": {k: round(float(v), 2) for k, v in conf.items()}}


# ---------------------------------------------------------------------------
# listing
# ---------------------------------------------------------------------------

def config_file_refs(raw: dict, project_dir: str | os.PathLike) -> dict[str, Path]:
    """Dotted config key → resolved path of every file the config references."""
    from sparc.studio.projects.config_service import get_dotted, path_keys

    pdir = Path(project_dir)
    out = {}
    for key in path_keys(raw or {}):
        v = get_dotted(raw or {}, key)
        if isinstance(v, str) and v.strip():
            p = Path(v)
            out[key] = (p if p.is_absolute() else pdir / p).resolve()
    return out


def used_by(raw: dict, project_dir: str | os.PathLike, path: str | os.PathLike) -> list[str]:
    target = Path(path).resolve()
    return [k for k, p in config_file_refs(raw, project_dir).items() if p == target]


def list_files(project_dir: str | os.PathLike, raw: dict) -> list[dict]:
    """``GET /files``: every file under the kind folders, with the config keys that use it."""
    pdir = Path(project_dir)
    refs = config_file_refs(raw, pdir)
    out = []
    seen: set[Path] = set()
    for kind, sub in KIND_DIRS.items():
        d = pdir / sub
        if not d.is_dir() or d in seen:
            continue
        seen.add(d)
        for p in sorted(d.rglob("*")):
            if not p.is_file() or p.name.startswith("."):
                continue
            rp = p.resolve()
            keys = [k for k, ref in refs.items() if ref == rp]
            k = kind
            if sub == "data":
                k = "join" if any(key.startswith("data.join") for key in keys) else "data"
            st = p.stat()
            out.append({"path": p.relative_to(pdir).as_posix(), "kind": k, "bytes": int(st.st_size),
                        "mtime": utc_iso(st.st_mtime), "used_by": keys})
    return out
