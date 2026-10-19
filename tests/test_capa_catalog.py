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
    monkeypatch.setattr(capa, "_fetch_file", lambda url, part, size=None: got.append(url) or part.write_bytes(b"abc"))
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


def _fake_osf(tree):
    """``_http_get`` for an OSF node: ``tree`` maps a folder id ('' = root) to its items (name, kind, id, size)."""
    import json

    from sparc.data.collect import capa

    root_url = capa.OSF_API.format(node_id="tdsy7")
    calls = []

    def get(url):
        calls.append(url)
        if url.startswith(root_url):
            fid = url[len(root_url):].strip("/")
            items = [{"id": i, "attributes": {"name": n, "kind": k, "size": z, "materialized_path":
                                              f"/{n}" if fid == "" else f"/{fid}/{n}"}} for n, k, i, z in tree[fid]]
            return json.dumps({"data": items, "links": {}}).encode()
        return b"abc"
    return get, calls


def test_download_with_a_folder_hint_reads_only_that_folder_and_refuses_a_missing_one(tmp_path, monkeypatch):
    """R12: a hint matching no root folder used to give an empty download recorded as complete."""
    from sparc.data.collect import capa

    tree = {"": [("Boston, MA", "folder", "fb", None), ("Boise, ID", "folder", "fi", None),
                 ("shared.csv", "file", "s1", 3)],
            "fb": [("Boston_am.csv", "file", "b1", 3)], "fi": [("AM_Boise.csv", "file", "i1", 3)]}
    get, calls = _fake_osf(tree)
    monkeypatch.setattr(capa, "_http_get", get)
    monkeypatch.setattr(capa, "_fetch_file", lambda url, part, size=None: part.write_bytes(b"abc"))
    with pytest.raises(LookupError) as exc:
        capa.download_capa_files("tdsy7", tmp_path / "ny", folder_hint="New York")
    assert "Boston, MA" in str(exc.value) and "Boise, ID" in str(exc.value)
    assert not (tmp_path / "ny" / "_download.json").exists()
    rec = capa.download_capa_files("tdsy7", tmp_path / "b", folder_hint="Boston")
    assert [f["path"] for f in rec["files"]] == ["fb/Boston_am.csv"] and rec["matched_folders"] == ["Boston, MA"]
    assert any(s["path"] == "shared.csv" and "root-level" in s["reason"] for s in rec["skipped"])
    # the completion record is keyed on (node, hint): another hint downloads again and drops the old files
    rec2 = capa.download_capa_files("tdsy7", tmp_path / "b", folder_hint="Boise")
    assert [f["path"] for f in rec2["files"]] == ["fi/AM_Boise.csv"]
    assert not (tmp_path / "b" / "fb" / "Boston_am.csv").exists()


def test_tdsy7_cities_without_a_folder_are_marked_unavailable():
    for city in ("New York, NY", "Baltimore, MD", "Miami, FL", "Louisville, KY", "Jackson, MS", "Dallas, TX",
                 "El Paso, TX", "Kansas City, MO", "Anchorage, AK", "Denver, CO"):
        e = C.CAPA_CATALOG[city]
        assert C.availability(e) == {"traverse_points": False, "rasters": False}, city
        assert not any(e["folder_hint"].lower() in f.lower() for f in C.TDSY7_FOLDERS), city
    for city in ("Boston, MA", "Boise, ID", "Los Angeles, CA", "Fort Lauderdale, FL", "Sacramento, CA"):
        e = C.CAPA_CATALOG[city]
        assert any(e["folder_hint"].lower() in f.lower() for f in C.TDSY7_FOLDERS), city
        assert C.availability(e)["traverse_points"] is None


def test_fetch_file_resumes_a_cut_off_transfer_with_a_range_request(tmp_path, monkeypatch):
    """R16: a 38 MB archive cut off mid-transfer (IncompleteRead) used to fail the city and be fetched again
    from the start; it is now retried and resumed where it stopped."""
    import http.client
    import urllib.request

    from sparc.data.collect import capa

    body = bytes(range(256)) * 400
    seen = []

    class Resp:
        def __init__(self, data, status, cut=None):
            self.data, self.status, self.cut, self.pos = data, status, cut, 0

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n=-1):
            if self.cut is not None and self.pos >= self.cut:
                raise http.client.IncompleteRead(b"", len(self.data) - self.pos)
            end = len(self.data) if n < 0 else min(self.pos + n, len(self.data))
            if self.cut is not None:
                end = min(end, self.cut)
            out, self.pos = self.data[self.pos:end], end
            return out

    def fake(req, timeout=None):
        rng = req.headers.get("Range")
        seen.append(rng)
        if rng is None:
            return Resp(body, 200, cut=40_000)                     # the first transfer is cut off
        start = int(rng.split("=")[1].rstrip("-"))
        return Resp(body[start:], 206)

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    monkeypatch.setattr("time.sleep", lambda s: None)                    # no back-off wait in the test
    part = tmp_path / "x.zip.part"
    capa._fetch_file("https://files.osf.io/v1/resources/n/providers/osfstorage/f", part, len(body))
    assert part.read_bytes() == body and seen[0] is None and seen[1].startswith("bytes=")

    def gone(req, timeout=None):
        import urllib.error

        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", gone)
    with pytest.raises(Exception) as exc:
        capa._fetch_file("https://files.osf.io/v1/x", tmp_path / "y.part", 10)
    assert getattr(exc.value, "code", None) == 404 and exc.value.sparc_url == "https://files.osf.io/v1/x"


def test_raster_archives_are_not_downloaded(tmp_path, monkeypatch):
    from sparc.data.collect import capa

    listing = [{"name": n, "id": str(i), "path": "/" + n, "size": 3} for i, n in enumerate(
        ["traverses_chw_brockton_092823.zip", "rasters_chw_brockton_092823.zip",
         "Surface Models (LA)/la_surface.zip", "Area-wide_Boise.zip"])]
    monkeypatch.setattr(capa, "_scan_storage", lambda node, top_level_filter=None: listing)
    monkeypatch.setattr(capa, "_fetch_file", lambda url, part, size=None: part.write_bytes(b"abc"))
    rec = capa.download_capa_files("rk75w", tmp_path)
    assert [f["path"] for f in rec["files"]] == ["traverses_chw_brockton_092823.zip"]
    assert len(rec["skipped"]) == 3 and all("maps, not traverse points" in s["reason"] for s in rec["skipped"])
