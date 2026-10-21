"""LST pilot QA registry (sparc/core/lstair/exclusions.yml): the shipped entries load and validate, a readings
entry drops exactly one sensor's readings of one run between its local clock bounds, a run entry gives the run id
the build flags, and malformed entries fail loudly.  Synthetic points only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sparc.core.lstair import exclusions as X


def _points(rows: list[tuple[str, str, str | None]]) -> pd.DataFrame:
    """Reader-style points from (local time, sensor id, vehicle_id or None: the sensor only in the key)."""
    from sparc.core.identify.traverses import assign_runs

    t = pd.to_datetime([r[0] for r in rows])
    return assign_runs(pd.DataFrame({
        "time": t, "lat": 35.6, "lon": -82.4, "temp_f": 80.0, "window": None,
        "source_file": "af_trav.shp", "vehicle_key": [f"af_trav.shp:{r[1]}" for r in rows],
        "vehicle_id": [r[2] for r in rows]}))


def test_the_shipped_registry_loads_and_names_its_evidence():
    reg = X.load_registry()
    assert {(e["slug"], e["kind"]) for e in reg} == {("asheville_nc", "readings"), ("los_angeles_ca", "readings"),
                                                     ("fort_lauderdale_fl", "run")}
    assert all(e["reason"] and e["evidence"] and "\n" not in e["evidence"] for e in reg)
    assert all(e["window"] in X.WINDOWS and len(e["date"]) == 10 for e in reg)
    assert X.run_exclusions("fort_lauderdale_fl") == {
        "2019-09-18 morning": "rain over 5 of 7 routes (51% of readings at RH >= 99%)"}
    assert X.run_exclusions("asheville_nc") == {} and X.run_exclusions("providence_ri") == {}
    assert X.reading_mask(_points([("2020-07-29 15:00:00", "CAPA1", "CAPA1")]), "providence_ri")[1] == []


def test_a_rain_cooled_route_drops_one_sensors_readings_from_the_bound_on():
    """Asheville 2023-07-24 af: CAPA2037 from 15:05 on (rain: RH 40 → 68%, −5.8 °C), nothing of the other sensors,
    nothing of CAPA2037's other runs, and no sensor whose id merely contains it."""
    rows = ([(f"2023-07-24 15:0{m}:30", "CAPA2037", "CAPA2037") for m in range(3, 8)]
            + [("2023-07-24 15:05:00", "CAPA2037", None)]                  # the sensor only in the vehicle key
            + [(f"2023-07-24 15:0{m}:30", "CAPA2033", "CAPA2033") for m in range(3, 8)]
            + [("2023-07-24 15:06:00", "CAPA20370", "CAPA20370")]
            + [("2023-07-24 06:30:00", "CAPA2037", "CAPA2037"), ("2023-07-24 19:30:00", "CAPA2037", "CAPA2037")])
    pts = _points(rows)
    keep, recs = X.reading_mask(pts, "asheville_nc")
    dropped = pts.loc[~keep]
    assert set(dropped["vehicle_key"].str.split(":").str[-1]) == {"CAPA2037"}
    assert set(dropped["run"]) == {"2023-07-24 midday"}
    assert (dropped["time"] >= pd.Timestamp("2023-07-24 15:05")).all() and len(dropped) == 4    # 15:05:00 included
    assert keep.sum() == len(pts) - 4 and keep.dtype == bool
    (rec,) = recs
    assert rec["run_id"] == "2023-07-24 midday" and rec["n_readings"] == 4 and rec["sensor"] == "CAPA2037"
    assert rec["first"] == "2023-07-24 15:05:00" and rec["last"] == "2023-07-24 15:07:30" and rec["evidence"]


def test_warm_up_bounds_drop_the_readings_before_until_local_in_each_run():
    """Los Angeles 2019-09-15: CAPA47 before 06:16 in am and before 19:11 in pm (warm-up); its later readings and
    its midday run are kept.  Points without a run column get their runs assigned."""
    rows = [("2019-09-15 06:04:31", "CAPA47", "CAPA47"), ("2019-09-15 06:15:59", "CAPA47", "CAPA47"),
            ("2019-09-15 06:16:00", "CAPA47", "CAPA47"), ("2019-09-15 06:40:00", "CAPA47", "CAPA47"),
            ("2019-09-15 06:05:00", "CAPA39", "CAPA39"), ("2019-09-15 15:05:00", "CAPA47", "CAPA47"),
            ("2019-09-15 19:08:16", "CAPA47", "CAPA47"), ("2019-09-15 19:11:00", "CAPA47", "CAPA47")]
    pts = _points(rows).drop(columns="run")
    keep, recs = X.reading_mask(pts, "los_angeles_ca")
    assert keep.tolist() == [False, False, True, True, True, True, False, True]
    assert [(r["run_id"], r["n_readings"]) for r in recs] == [("2019-09-15 morning", 2), ("2019-09-15 evening", 1)]
    # a sensor id parsed differently matches nothing, and the record says so
    keep, recs = X.reading_mask(pts.assign(vehicle_id="capa47", vehicle_key="x:capa47"), "los_angeles_ca")
    assert keep.all() and [r["n_readings"] for r in recs] == [0, 0] and "first" not in recs[0]


def test_night_bounds_after_midnight_belong_to_the_next_day(tmp_path):
    reg = tmp_path / "reg.yml"
    reg.write_text('- {slug: c, kind: readings, sensor: S1, date: "2020-07-29", window: night, from_local: "23:50",'
                   ' until_local: "00:10", reason: r, evidence: e}\n'
                   '- {slug: c, kind: run, date: "2020-07-29", window: night, reason: wet, evidence: e}\n'
                   '- {slug: c, kind: run, date: "2020-07-29", window: night, reason: windy, evidence: e}\n')
    entries = X.load_registry(reg)
    rows = [("2020-07-29 23:40:00", "S1", "S1"), ("2020-07-29 23:55:00", "S1", "S1"),
            ("2020-07-30 00:05:00", "S1", "S1"), ("2020-07-30 00:20:00", "S1", "S1")]
    keep, _ = X.reading_mask(_points(rows), "c", entries)
    assert keep.tolist() == [True, False, False, True]
    assert X.run_exclusions("c", entries) == {"2020-07-29 night": "wet; windy"}


@pytest.mark.parametrize("line, msg", [
    ('{slug: c, kind: readings, sensor: S, date: "2020-07-29", window: midday, from_local: 15:05, reason: r, '
     'evidence: e}', "quoted local clock time"),
    ('{slug: c, kind: readings, sensor: S, date: "2020-07-29", window: midday, untill_local: "15:05", reason: r, '
     'evidence: e}', "unknown key"),
    ('{slug: c, kind: readings, sensor: S, date: "2020-07-29", window: afternoon, reason: r, evidence: e}',
     "window"),
    ('{slug: c, kind: readings, date: "2020-07-29", window: midday, reason: r, evidence: e}', "sensor"),
    ('{slug: c, kind: run, date: "2020-07-29", window: midday, from_local: "15:05", reason: r, evidence: e}',
     "readings entries only"),
    ('{slug: c, kind: drop, date: "2020-07-29", window: midday, reason: r, evidence: e}', "kind"),
    ('{slug: c, kind: run, date: "29/07/2020", window: midday, reason: r, evidence: e}', "YYYY-MM-DD"),
    ('{slug: c, kind: run, date: "2020-07-29", window: midday, reason: r}', "missing"),
    ('{slug: c, kind: readings, sensor: S, date: "2020-07-29", window: midday, from_local: "15:30", '
     'until_local: "15:05", reason: r, evidence: e}', "not before"),
])
def test_a_malformed_entry_fails_loudly(tmp_path, line, msg):
    reg = tmp_path / "reg.yml"
    reg.write_text(f"- {line}\n")
    with pytest.raises(ValueError, match=msg):
        X.load_registry(reg)


def test_an_unquoted_date_is_read_as_its_iso_string(tmp_path):
    reg = tmp_path / "reg.yml"
    reg.write_text("- {slug: c, kind: run, date: 2020-07-29, window: midday, reason: r, evidence: e}\n")
    assert X.run_exclusions("c", X.load_registry(reg)) == {"2020-07-29 midday": "r"}
    keep, recs = X.reading_mask(_points([("2020-07-29 15:00:00", "S", "S")]), "c", X.load_registry(reg))
    assert keep.tolist() == [True] and recs == []
    empty = _points([("2020-07-29 15:00:00", "S", "S")]).iloc[:0]
    assert np.asarray(X.reading_mask(empty, "asheville_nc")[0]).size == 0
