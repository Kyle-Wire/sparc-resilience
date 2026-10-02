"""Fixtures of the projects tests: a synthetic demo project and a fake-network job worker."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

FAKE_WORKER = Path(__file__).resolve().parent / "fake_worker.py"


def create(client, name: str, template: str, **options) -> dict:
    body = {"name": name, "template": template}
    if options:
        body["options"] = options
    r = client.post("/api/projects", json=body)
    assert r.status_code == 201, r.text
    return r.json()


@pytest.fixture
def demo(client) -> dict:
    """The synthetic demo with the committed fixture's n and seed (``Project``)."""
    return create(client, "Demo city", "synthetic_demo", n=40, seed=0)["project"]


@pytest.fixture
def fake_network(ctx):
    """Run every process job through ``fake_worker.py`` (the real worker with the fetchers faked)."""
    ex = ctx.jobs.executors["process"]
    ex.command = lambda job: [sys.executable, str(FAKE_WORKER), str(Path(job["job_dir"]))]
    return ex
