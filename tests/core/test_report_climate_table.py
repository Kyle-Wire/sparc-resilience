"""report.md climate table: one header cell per column and a matching separator, with and
without adaptation variants (a stray empty header cell leaves the table as raw pipes)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sparc.core import climate as C
from sparc.core.report import render_report


def _report(adaptation: dict[str, np.ndarray]) -> str:
    obs = np.array([86.0, 88.0, 90.0, 94.0])
    factors = pd.DataFrame({"experiment": ["ssp245"] * 3, "period": ["2041-2060"] * 3, "model": list("abc"),
                            "delta_K": [1.0, 2.0, 3.0]})
    climate = C.summarize_projections(obs, factors, adaptation, thresholds=[95], to_units=1.8)
    climate.update({"months": [6, 7, 8], "variable": "tasmax", "baseline": "1995-2014", "n_models": 3,
                    "site": {"lat": 41.8, "lon": -71.4}})
    manifest = {"name": "t", "created_utc": "2026-01-01T00:00:00+00:00", "n_points": 4, "qa": {}, "climate": climate}
    return render_report(SimpleNamespace(manifest=manifest, data=SimpleNamespace(target_units="°F")))


def _climate_table(md: str) -> list[str]:
    lines = md.splitlines()
    start = lines.index("## Climate projections (CMIP6 delta method)")
    table = []
    for line in lines[start + 1:]:
        if line.startswith("|"):
            table.append(line)
        elif table:
            break
    return table


def _cells(row: str) -> list[str]:
    assert row.startswith("|") and row.endswith("|"), row
    return [c.strip() for c in row[1:-1].split("|")]


@pytest.mark.parametrize("adaptation, n_cols", [({}, 4), ({"trees": np.full(4, -1.0)}, 5)])
def test_climate_table_columns_match(adaptation, n_cols):
    table = _climate_table(_report(adaptation))
    header, sep, *rows = table
    assert len(_cells(header)) == n_cols and all(_cells(header)), header       # no empty header cell
    assert _cells(sep) == ["---"] * n_cols
    assert rows and all(len(_cells(r)) == n_cols for r in rows)
    assert _cells(header)[-1] == ("with trees" if adaptation else "share ≥ 95 °F: no adaptation")
