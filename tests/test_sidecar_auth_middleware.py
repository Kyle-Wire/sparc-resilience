"""Tests for the per-launch sidecar Bearer-token middleware.

Validates three behaviors:
  1. When ``SPARC_SIDECAR_TOKEN`` is unset (or patched to ``None``)
     the middleware is a no-op — every route returns its normal
     response, preserving the existing test suite + ``python -m sparc
     server`` workflow.
  2. With a token set, requests without (or with a wrong) Bearer
     header are rejected with HTTP 401.
  3. ``/health`` is always public, regardless of token state, so the
     desktop splash can ping it before the renderer has fetched the
     token from the Tauri command.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from sparc.server import app as app_module


def _client() -> TestClient:
    return TestClient(app_module.app)


def test_middleware_noop_when_token_unset(monkeypatch):
    monkeypatch.setattr(app_module, "_SIDECAR_TOKEN", None)
    r = _client().get("/health")
    assert r.status_code == 200


def test_health_always_public(monkeypatch):
    monkeypatch.setattr(app_module, "_SIDECAR_TOKEN", "secret-xyz")
    r = _client().get("/health")
    assert r.status_code == 200


def test_protected_route_rejects_missing_bearer(monkeypatch):
    monkeypatch.setattr(app_module, "_SIDECAR_TOKEN", "secret-xyz")
    # Pick any non-public route. ``/config`` is registered and requires
    # no project state to return a 4xx that ISN'T 401 when authed.
    r = _client().get("/config")
    assert r.status_code == 401
    body = r.json()
    assert "sidecar token" in body.get("detail", "").lower()


def test_protected_route_rejects_wrong_bearer(monkeypatch):
    monkeypatch.setattr(app_module, "_SIDECAR_TOKEN", "secret-xyz")
    r = _client().get("/config", headers={"Authorization": "Bearer wrong-token"})
    assert r.status_code == 401


def test_protected_route_accepts_correct_bearer(monkeypatch):
    monkeypatch.setattr(app_module, "_SIDECAR_TOKEN", "secret-xyz")
    r = _client().get("/config", headers={"Authorization": "Bearer secret-xyz"})
    # Anything that isn't 401 proves the middleware let it through.
    # The route itself may return 200 (no project) or another status —
    # we only care that auth didn't reject it.
    assert r.status_code != 401
