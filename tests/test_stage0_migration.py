"""Stage 0 (correlogram) migration tests.

Verifies:
- The struct + table writer paths produce db-resident artifacts.
- ``read_any`` reads them back regardless of storage_kind.
- The on-demand correlogram figure renderer produces a non-empty PNG.
"""

from __future__ import annotations

import pandas as pd
import pytest

from sparc.registry.run_registry import RunRegistry, set_active_registry
from sparc.registry.store import ArtifactStore
from sparc.report.render import render_png, list_figure_kinds


@pytest.fixture()
def store(tmp_path):
    reg = RunRegistry(tmp_path)
    set_active_registry(reg)
    yield ArtifactStore(reg)
    set_active_registry(None)


def _fake_correlogram_payload():
    return {
        "individual_results": {
            "tair_c": {
                "optimal_bandwidth": 250.0,
                "effective_range": 250.0,
                "max_moran_i": 0.42,
                "correlogram_results": {
                    "correlogram_results": {
                        "lag_distances": [50, 100, 200, 400, 800],
                        "morans_i_values": [0.5, 0.4, 0.3, 0.1, -0.1],
                        "z_scores": [5.0, 4.0, 3.0, 1.0, -1.0],
                        "p_values": [0.001, 0.001, 0.01, 0.3, 0.4],
                    }
                },
            },
        },
        "config": {"target_variable": "tair_c"},
    }


def test_dataset_profile_struct_roundtrip(store):
    profile = {"size_tier": "medium", "n_samples": 1234}
    store.write_struct("0", "dataset_profile", profile, producer="test")
    assert store.has("0", "dataset_profile")
    assert store.read_any("0", "dataset_profile") == profile


def test_correlogram_results_struct_roundtrip(store):
    payload = _fake_correlogram_payload()
    store.write_struct("0", "correlogram_results", payload, producer="test")
    out = store.read_any("0", "correlogram_results")
    assert out["individual_results"]["tair_c"]["optimal_bandwidth"] == 250.0


def test_correlogram_summary_table_roundtrip(store):
    df = pd.DataFrame(
        {
            "Variable": ["tair_c", "albedo"],
            "Optimal_Bandwidth": [250.0, 180.0],
            "Best_Kernel": ["gaussian", "exponential"],
        }
    )
    store.write_table("0", "correlogram_summary", df, producer="test")
    out = store.read_any("0", "correlogram_summary")
    assert list(out["Variable"]) == ["tair_c", "albedo"]


def test_read_any_legacy_path_fallback(tmp_path):
    """When an artifact is registered on disk (legacy), read_any deserializes it."""
    reg = RunRegistry(tmp_path)
    set_active_registry(reg)
    try:
        # Simulate a legacy on-disk JSON artifact.
        stage_dir = tmp_path / "Stage_0_Correlogram"
        stage_dir.mkdir(parents=True, exist_ok=True)
        legacy_path = stage_dir / "dataset_profile.json"
        legacy_path.write_text('{"size_tier": "small"}')
        reg.register_artifact(
            stage="0",
            artifact_id="dataset_profile",
            path=str(legacy_path.relative_to(tmp_path)),
            format="json",
            storage_kind="legacy_path",
            name="dataset_profile.json",
        )
        store = ArtifactStore(reg)
        assert store.has("0", "dataset_profile")
        out = store.read_any("0", "dataset_profile")
        assert out == {"size_tier": "small"}
    finally:
        set_active_registry(None)


def test_correlogram_figure_renderer_registered():
    kinds = list_figure_kinds()
    assert "correlogram" in kinds


def test_render_correlogram_png(store):
    store.write_struct("0", "correlogram_results", _fake_correlogram_payload())
    png = render_png("correlogram", "0", "correlogram_results", variable="tair_c")
    assert isinstance(png, (bytes, bytearray))
    # PNG magic header
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png) > 200


def test_render_correlogram_default_variable(store):
    store.write_struct("0", "correlogram_results", _fake_correlogram_payload())
    png = render_png("correlogram", "0", "correlogram_results")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
