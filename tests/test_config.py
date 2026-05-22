"""Tests for config loading."""
import pytest
from pathlib import Path


def test_load_config_missing_file():
    """load_config raises FileNotFoundError for a nonexistent path."""
    from sparc.config.config import load_config

    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/project.yml")


def test_load_config_legacy_fallback(monkeypatch):
    """Legacy fallback still returns a dict (with deprecation warning)."""
    import warnings
    from sparc.config.config import load_config

    # Ensure SPARC_PROJECT env var is unset so fallback triggers
    monkeypatch.delenv("SPARC_PROJECT", raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cfg = load_config(None)
    assert isinstance(cfg, dict)
    assert "paths" in cfg


def test_load_config_valid_yaml(tmp_path):
    """load_config succeeds on a minimal valid project.yml."""
    from sparc.config.config import load_config

    proj = tmp_path / "project.yml"
    proj.write_text(
        """
project:
  name: test_project
  domain: uhi
data:
  file_path: dummy.csv
  target_column: temp
  coord_columns: [x, y]
  identifier_column: id
crs:
  input: "EPSG:4326"
  projected: "EPSG:32618"
predictors:
  - pred_a
  - pred_b
output:
  base_dir: output
"""
    )
    cfg = load_config(str(proj))
    assert cfg["variables"]["target"] == "temp"
    assert "pred_a" in cfg["predictors"]["base_model"]


def test_jepa_enable_defaults_to_true_when_omitted(tmp_path):
    """JD-3: When project.yml has no jepa: section, config['jepa']['enable']
    must default to True (schema default changed from false to true in JD-3).
    """
    from sparc.config.config import load_config

    proj = tmp_path / "project.yml"
    proj.write_text(
        """
project:
  name: test_jepa_default
  domain: uhi
data:
  file_path: dummy.csv
  target_column: temp
  coord_columns: [x, y]
  identifier_column: id
crs:
  input: "EPSG:4326"
  projected: "EPSG:32618"
predictors:
  - pred_a
output:
  base_dir: output
"""
    )
    cfg = load_config(str(proj))
    # jepa section must be present and enable must be True
    assert "jepa" in cfg, "config must contain a 'jepa' key"
    assert cfg["jepa"].get("enable") is True, (
        f"jepa.enable must default to True, got {cfg['jepa'].get('enable')!r}"
    )


def test_jepa_enable_false_is_honoured(tmp_path):
    """JD-3: An explicit jepa.enable: false in project.yml must still be
    respected — the default-true only applies when the key is absent.
    """
    from sparc.config.config import load_config

    proj = tmp_path / "project.yml"
    proj.write_text(
        """
project:
  name: test_jepa_false
  domain: uhi
data:
  file_path: dummy.csv
  target_column: temp
  coord_columns: [x, y]
  identifier_column: id
crs:
  input: "EPSG:4326"
  projected: "EPSG:32618"
predictors:
  - pred_a
output:
  base_dir: output
jepa:
  enable: false
"""
    )
    cfg = load_config(str(proj))
    assert cfg["jepa"].get("enable") is False, (
        "An explicit jepa.enable: false must be preserved"
    )
