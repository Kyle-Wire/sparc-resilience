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


def test_download_capa_files_keeps_the_folder_layout_and_resumes(tmp_path, monkeypatch):
    import json

    from sparc.data.collect import capa

    listing = [{"name": "af_trav.shp", "id": "1", "path": "/Providence/traverses/af_trav.shp", "size": 3},
               {"name": "af_trav.dbf", "id": "2", "path": "/Providence/traverses/af_trav.dbf", "size": 3},
               {"name": "af_t_f.tif", "id": "3", "path": "/Providence/rasters/af_t_f.tif", "size": 3},
               {"name": "../../evil.csv", "id": "4", "path": "/../../evil.csv", "size": 3}]
    got = []
    monkeypatch.setattr(capa, "_scan_storage", lambda node, top_level_filter=None: listing)
    monkeypatch.setattr(capa, "_http_get", lambda url: got.append(url) or b"abc")
    rec = capa.download_capa_files("wu9v7", tmp_path / "raw")
    assert (tmp_path / "raw" / "Providence" / "traverses" / "af_trav.shp").read_bytes() == b"abc"
    assert (tmp_path / "raw" / "Providence" / "traverses" / "af_trav.dbf").exists()
    assert not (tmp_path / "raw" / "Providence" / "rasters").exists()             # rasters are not fetched
    assert (tmp_path / "raw" / "evil.csv").exists() and not (tmp_path / "evil.csv").exists()
    assert [s["path"] for s in rec["skipped"]] == ["Providence/rasters/af_t_f.tif"]
    assert json.loads((tmp_path / "raw" / "_download.json").read_text())["complete"] is True
    n = len(got)
    capa.download_capa_files("wu9v7", tmp_path / "raw")
    assert len(got) == n                                                              # finished: no new requests
