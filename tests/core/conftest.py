"""Shared fixtures for the core pipeline tests.

Synthetic fixtures are session-scoped (they are deterministic and cheap to
share).  The Providence fixture reads ``brown4.csv`` from the repo root via an
absolute path, so tests work from any working directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BROWN_CSV = REPO_ROOT / "brown4.csv"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running statistical checks (deselect with -m 'not slow')")
    try:
        import torch

        torch.set_num_threads(2)
    except Exception:  # pragma: no cover - torch always installed for core
        pass


@pytest.fixture(scope="session")
def synthetic_city():
    from sparc.core.synthetic import make_synthetic_city

    return make_synthetic_city(seed=0)


@pytest.fixture(scope="session")
def synthetic_core_data(synthetic_city):
    from sparc.core.config import core_config_from_dict
    from sparc.core.data import prepare_frame
    from sparc.core.synthetic import synthetic_city_config

    cfg = core_config_from_dict(synthetic_city_config())
    return cfg, prepare_frame(synthetic_city.frame, cfg)


@pytest.fixture(scope="session")
def brown_csv():
    if not BROWN_CSV.exists():
        pytest.skip("brown4.csv not present")
    return BROWN_CSV


@pytest.fixture(scope="session")
def providence_config_path():
    p = REPO_ROOT / "configs" / "core_providence.yml"
    if not p.exists():
        pytest.skip("configs/core_providence.yml not present")
    return p
