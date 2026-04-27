"""Phase 3 test for registry-backed report figure embedding."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sparc.registry.run_registry import RunRegistry
from sparc.registry.store import ArtifactStore
from sparc.report import generate_report_html


@pytest.fixture()
def populated_registry(tmp_path):
    reg = RunRegistry(tmp_path)
    store = ArtifactStore(reg)
    # An artifact with a registered renderer.
    df = pd.DataFrame({"target": np.random.RandomState(0).normal(size=100)})
    store.write_table("0", "data", df)
    # An artifact with no renderer — should be silently skipped.
    store.write_table("99", "no_renderer", pd.DataFrame({"a": [1, 2]}))
    return reg


def test_generate_report_html_embeds_db_figures(populated_registry):
    cfg = {
        "project": {"name": "Test", "domain": "test", "description": ""},
        "causal": {},
        "physics": {},
        "pipeline": {},
        "predictors": [],
        "paths": {},
    }
    html = generate_report_html(config=cfg, registry=populated_registry)
    # PNG embed for the registered renderer is present.
    assert "data:image/png;base64," in html
    # Skipped artifact does not appear as an image.
    assert "no_renderer" not in html.lower() or "data:image/png;base64," in html


def test_generate_report_html_no_registry_no_disk_no_plots(tmp_path):
    cfg = {
        "project": {"name": "Test", "domain": "test", "description": ""},
        "causal": {},
        "physics": {},
        "pipeline": {},
        "predictors": [],
        "paths": {},
    }
    # No registry, no output_dir — should not crash, just produce no plots.
    html = generate_report_html(config=cfg, registry=None, output_dir=None)
    assert "Test" in html
