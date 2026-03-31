"""Tests for PipelinePaths construction."""
import pytest


def test_pipeline_paths_from_config(tmp_path):
    """PipelinePaths.from_config builds expected stage directories."""
    from sparc.run.pipeline_paths import PipelinePaths

    cfg = {
        "paths": {"project_root": str(tmp_path)},
        "output": {
            "base_dir": str(tmp_path / "output"),
            "stage_dirs": {
                "stage_1": "S1_Correlogram",
                "stage_2": "S2_CV",
            },
        },
    }
    pp = PipelinePaths.from_config(cfg)
    assert "S1_Correlogram" in str(pp.stage1_dir)
    assert "S2_CV" in str(pp.stage2_dir)
