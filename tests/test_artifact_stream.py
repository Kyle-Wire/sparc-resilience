"""Phase 2 tests for the WebSocket artifact-event stream."""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from sparc.registry.run_registry import RunRegistry
from sparc.registry.store import ArtifactStore
from sparc.server import app as app_module


@pytest.fixture()
def attached_registry(tmp_path, monkeypatch):
    """Replace the server's registry with a fresh one bound to tmp_path."""
    reg = RunRegistry(tmp_path)
    reg.add_register_listener(app_module._on_artifact_registered)
    monkeypatch.setattr(app_module.state, "registry", reg, raising=False)
    yield reg
    reg.remove_register_listener(app_module._on_artifact_registered)


def test_register_listener_buffers_event(attached_registry):
    app_module.state.event_buffer.clear()
    store = ArtifactStore(attached_registry)
    store.write_table("0", "tbl", pd.DataFrame({"a": [1]}))

    events = [
        e for e in app_module.state.get_buffered_events()
        if e.get("type") == "artifact_written" and e.get("artifact_id") == "tbl"
    ]
    assert len(events) == 1
    ev = events[0]
    assert ev["stage"] == "0"
    assert ev["kind"] == "table"
    assert ev["format"] == "table"


def test_run_events_endpoint_includes_artifact_event(attached_registry):
    app_module.state.event_buffer.clear()
    store = ArtifactStore(attached_registry)
    store.write_struct("3", "diag", {"k": 1})

    with TestClient(app_module.app) as client:
        resp = client.get("/run/events")
        assert resp.status_code == 200
        payload = resp.json()
        assert any(
            e.get("type") == "artifact_written" and e.get("artifact_id") == "diag"
            for e in payload["events"]
        )


def test_websocket_subscribe_receives_artifact_event(attached_registry):
    """Open /run/stream in subscribe-only mode and assert push delivery."""
    with TestClient(app_module.app) as client:
        with client.websocket_connect("/run/stream") as ws:
            ws.send_json({"subscribe": "artifacts"})
            ack = ws.receive_json()
            assert ack.get("type") == "subscribed"

            store = ArtifactStore(attached_registry)
            # Write from a background-ish path; the listener fans out.
            store.write_struct("3", "live_event", {"v": 42})

            # Drain a few messages until we see ours.
            received = None
            for _ in range(10):
                msg = ws.receive_json()
                if (
                    msg.get("type") == "artifact_written"
                    and msg.get("artifact_id") == "live_event"
                ):
                    received = msg
                    break
            assert received is not None, "did not receive artifact_written event"
            assert received["stage"] == "3"
            assert received["kind"] == "struct"
