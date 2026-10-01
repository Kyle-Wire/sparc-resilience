"""Literature consistency panel: unit conversion and scenario choice."""

from __future__ import annotations

import pytest

from sparc.core.literature import LITERATURE, literature_panel, sparc_effects


def test_sparc_effects_scale_to_per_tenth_and_celsius():
    m = {"scenarios": [
        {"name": "Canopy +5", "mean_delta": -0.10, "mean_delta_se": 0.02, "mean_realized": {"can": 5.0}},
        {"name": "Canopy +10", "mean_delta": -0.36, "mean_delta_se": 0.09, "mean_realized": {"can": 9.0}},
        {"name": "Package", "mean_delta": -2.0, "mean_realized": {"can": 10.0, "alb": 0.1}},
        {"name": "Albedo +0.2", "mean_delta": -0.9, "mean_realized": {"alb": 0.2}}]}
    e = sparc_effects(m, "can", "alb", "degF")
    assert e["canopy"]["scenario"] == "Canopy +10"                        # closest single-variable dose to +10
    assert e["canopy"]["cooling_C"] == pytest.approx(0.36 * (10 / 9) * 5 / 9)
    assert e["canopy"]["se_C"] == pytest.approx(0.09 * (10 / 9) * 5 / 9)
    assert e["albedo"]["cooling_C"] == pytest.approx(0.9 * 0.5 * 5 / 9)   # package is never used
    p = literature_panel(m, "can", "alb", "degF")
    assert len(p["rows"]) == len(LITERATURE) and all(r["sparc_C"] is not None for r in p["rows"])
    assert all(r["status"] in ("abstract", "verify") and r["citation"] for r in LITERATURE)
