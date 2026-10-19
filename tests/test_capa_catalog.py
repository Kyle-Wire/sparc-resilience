"""CAPA Heat Watch catalog (LST pilot L2): Providence's traverse node and campaign date, and the field that
tells traverse-point availability apart from raster availability."""

from __future__ import annotations

from datetime import date

import pytest

from sparc.data.collect import capa_catalog as C


def test_providence_points_at_the_wu9v7_traverses_and_the_2020_campaign():
    name, e = C.lookup_city("Providence")
    assert name == "Providence, RI"
    assert e["osf_node"] == "wu9v7" and e["folder_hint"] is None and e["year"] == 2020
    assert C.traverse_node(e) == ("wu9v7", None)
    d, src = C.campaign_date(e)
    assert date.fromisoformat(d) == date(2020, 7, 29) and src == "traverse timestamps"
    assert C.availability(e) == {"traverse_points": True, "rasters": None}
    assert "canopy_identification" in e["availability_source"]


def test_providence_matches_the_core_config_campaign_date():
    yaml = pytest.importorskip("yaml")
    from pathlib import Path

    cfg = Path(__file__).resolve().parents[1] / "configs" / "core_providence.yml"
    if not cfg.exists():
        pytest.skip("configs/core_providence.yml not present")
    text = cfg.read_text(encoding="utf-8")
    assert "2020-07-29" in text and "2020-07-29" == C.campaign_date(C.CAPA_CATALOG["Providence, RI"])[0]
    assert yaml.safe_load(text)                                  # still a valid config


def test_availability_is_unknown_unless_stated():
    for name, e in C.CAPA_CATALOG.items():
        av = C.availability(e)
        assert set(av) == {"traverse_points", "rasters"}
        assert all(v in (None, True, False) for v in av.values()), name
        if "availability" not in e:
            assert av == {"traverse_points": None, "rasters": None}, name
    # a multi-city node without a separate traverse node searches the same node and folder
    assert C.traverse_node(C.CAPA_CATALOG["Boston, MA"]) == ("tdsy7", "Boston")
    assert C.campaign_date(C.CAPA_CATALOG["Boston, MA"]) == (None, None)


def test_traverse_node_can_differ_from_the_raster_node():
    e = {"osf_node": "aaaaa", "folder_hint": "City", "traverse_node": "bbbbb", "traverse_folder_hint": "Trav"}
    assert C.traverse_node(e) == ("bbbbb", "Trav")
    assert C.availability({"availability": {"traverse_points": False, "rasters": True}}) == \
        {"traverse_points": False, "rasters": True}


def test_no_other_city_lost_its_entry():
    assert len(C.CAPA_CATALOG) >= 60
    assert all("osf_node" in e and "utm_epsg" in e and "state" in e for e in C.CAPA_CATALOG.values())
