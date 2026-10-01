"""``/api/meta`` (api.md §2): palettes against the golden LUT of SPEC §6.3, the seed table of §5.4, stage list
fallback and override, output catalog, job kinds, event schema."""

from __future__ import annotations

import sys
import types

import pytest

from sparc.studio import meta

# SPEC §6.3 - ramps and golden LUT entries (written from the spec, not generated from code)
RAMPS = {
    "seqLight": "#cde2fb #9ec5f4 #6da7ec #3987e5 #256abf #184f95 #0d366b".split(),
    "seqDark": "#1d2f45 #184f95 #256abf #3987e5 #6da7ec #9ec5f4 #cde2fb".split(),
    "divLight": "#0d366b #256abf #6da7ec #f0efec #f19a8f #d6403f #8a1f22".split(),
    "divDark": "#9ec5f4 #3987e5 #1f4f8a #383835 #8f3434 #e66767 #f6b3ab".split(),
}
GOLDEN = {  # index → colour at [0], [64], [128], [191], [255]
    "seqLight": ["#cde2fb", "#85b6f0", "#3987e5", "#1e5caa", "#0d366b"],
    "seqDark": ["#1d2f45", "#1e5caa", "#3a87e5", "#85b6f0", "#cde2fb"],
    "divLight": ["#0d366b", "#4a89d6", "#f0eeeb", "#e57168", "#8a1f22"],
    "divDark": ["#9ec5f4", "#2c6ab6", "#393835", "#b94d4d", "#f6b3ab"],
}
INDICES = (0, 64, 128, 191, 255)

# SPEC §5.4 seed table (checkpoint_save: 2 s per 500 MB → 2.1 s for the 525 MB reference checkpoint;
# unpickle: 1 s per 20 MB → 26.25 s for it)
SEED_TABLE = {
    "base_fit:mgwr": 170, "base_fit:gwrf": 23, "base_fit:gam": 6, "base_fit:physics": 4, "base_fit:ols": 0.1,
    "adv_refit": 4.5, "stacker_fit:mean": 0.2, "stacker_fit:nnls": 0.2, "stacker_fit:residual": 14,
    "baseline_fit:*": 12, "engine_pass": 13, "causal_step:dml": 10, "causal_step:spillover": 15,
    "causal_step:cate": 1, "causal_step:dr": 22, "causal_step:sens": 0.5, "causal_step:audit": 0.5,
    "s0_load": 0.3, "s1_influence": 3.3, "checkpoint_save": 2 * 525 / 500, "pareto": 0.5, "climate_model": 20,
    "remote_object": 5, "replicate:*": 254, "variant:*": 1200, "unpickle": 525 / 20, "mediator_fit": 1,
}
STAGE_IDS = ["S0", "S1", "S2_S3", "baselines", "cv_curve", "S4", "S5", "climate", "S6", "S7", "finish"]


@pytest.mark.parametrize("ramp", sorted(GOLDEN))
def test_palette_luts_match_golden_table(ramp):
    pal = meta.palettes()
    assert pal[ramp] == RAMPS[ramp]
    lut = meta.lut_hex(pal[ramp])
    assert len(lut) == 256
    assert [lut[i] for i in INDICES] == GOLDEN[ramp]


def test_api_palettes_and_unit_costs(client):
    body = client.get("/api/meta").json()
    for ramp, stops in RAMPS.items():
        assert body["palettes"][ramp] == stops
        assert [meta.lut_hex(body["palettes"][ramp])[i] for i in INDICES] == GOLDEN[ramp]
    assert len(body["palettes"]["cat"]) == 4
    assert body["unit_costs"] == pytest.approx(SEED_TABLE)


def test_stage_list_falls_back_without_stage_nodes(monkeypatch):
    stub = types.ModuleType("sparc.core.pipeline")
    monkeypatch.setitem(sys.modules, "sparc.core.pipeline", stub)
    stages = meta.stages()
    assert [s["id"] for s in stages] == STAGE_IDS
    assert stages == meta.STATIC_STAGES
    by = {s["id"]: s for s in stages}
    assert by["S2_S3"]["checkpoint_key"] == "S3" and by["S2_S3"]["manifest_timing_key"] == "S2_S3"
    assert by["climate"]["manifest_timing_key"] is None and by["finish"]["checkpoint_key"] is None


def test_stage_list_falls_back_when_pipeline_fails_to_import(monkeypatch):
    monkeypatch.setitem(sys.modules, "sparc.core.pipeline", None)   # import raises ImportError
    assert [s["id"] for s in meta.stages()] == STAGE_IDS


@pytest.mark.parametrize("nodes", [
    [{"id": "S0", "label": "Core label for S0"}, {"id": "S2_S3", "label": "Fit"}],
    [("S0", "Core label for S0"), ("S2_S3", "Fit")],
    {"S0": {"id": "S0", "label": "Core label for S0"}, "S2_S3": {"id": "S2_S3", "label": "Fit"}},
])
def test_stage_list_uses_core_stage_nodes(monkeypatch, nodes):
    stub = types.ModuleType("sparc.core.pipeline")
    stub.STAGE_NODES = nodes
    stub.CHECKPOINT_KEY = {"S2_S3": "S3"}
    monkeypatch.setitem(sys.modules, "sparc.core.pipeline", stub)
    stages = meta.stages()
    assert [s["id"] for s in stages] == STAGE_IDS              # core's table, completed and ordered
    by = {s["id"]: s for s in stages}
    assert by["S0"]["label"] == "Core label for S0" and by["S2_S3"]["label"] == "Fit"
    assert by["S2_S3"]["checkpoint_key"] == "S3"
    assert by["S1"]["label"] == meta.STATIC_STAGES[1]["label"]


def test_output_catalog_lazy_and_tolerant(monkeypatch):
    monkeypatch.setitem(sys.modules, "sparc.core.catalog", None)
    assert meta.output_catalog() == []
    stub = types.ModuleType("sparc.core.catalog")
    stub.OUTPUTS = [types.SimpleNamespace(id="predictions", files=("predictions.parquet",), label="Predictions",
                                          group="model", produced_by="stage:S2_S3", view="accuracy",
                                          formats=("csv",), manifest_key=None)]
    monkeypatch.setitem(sys.modules, "sparc.core.catalog", stub)
    cat = meta.output_catalog()
    assert cat == [{"id": "predictions", "label": "Predictions", "group": "model", "files": ["predictions.parquet"],
                    "produced_by": "stage:S2_S3", "view": "accuracy", "formats": ["csv"], "manifest_key": None}]
    tabs = meta.run_tab_outputs(cat)
    assert tabs["accuracy"] == ["predictions"] and set(tabs) == set(meta.RUN_TAB_IDS)


def test_meta_document(client):
    body = client.get("/api/meta").json()
    assert body["schema_version"] == 1 and body["event_schema_version"] == 1
    assert [s["id"] for s in body["stages"]] == STAGE_IDS
    kinds = {k["kind"]: k for k in body["job_kinds"]}
    assert {"test.sleep", "test.events", "test.fail", "test.ignore_sigterm", "test.pool"} <= set(kinds)
    assert kinds["test.sleep"]["lane"] == "heavy" and kinds["test.sleep"]["executor"] == "process"
    assert kinds["test.sleep"]["params_schema"]["properties"]["seconds"]["type"] == "number"
    assert [m["id"] for m in body["modes"]] == ["fast", "coarse", "full"]
    assert "fill_headroom" in body["edit_modes"] and "buffer" in body["selection_kinds"]
    assert any(w["code"] == "qa.coarse" for w in body["warning_codes"])
    assert set(body["run_tab_outputs"]) == set(meta.RUN_TAB_IDS)


def test_test_kinds_hidden_unless_enabled(make_app):
    from fastapi.testclient import TestClient

    from tests.studio.conftest import AUTH

    with TestClient(make_app(test_kinds=False), headers=AUTH) as c:
        kinds = {k["kind"] for k in c.get("/api/meta").json()["job_kinds"]}
        assert not any(k.startswith("test.") for k in kinds)
        r = c.post("/api/jobs", json={"kind": "test.sleep", "params": {}})
        assert r.status_code == 404 and r.json()["error"]["code"] == "unknown_kind"


def test_event_schema(client):
    schema = client.get("/api/meta/event-schema").json()
    text = str(schema)
    for t in ("run.start", "run.plan", "task.end", "tick", "artifact", "checkpoint", "job.status"):
        assert t in text
    assert "$defs" in schema
