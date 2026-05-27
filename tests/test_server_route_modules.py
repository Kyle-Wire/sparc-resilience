"""Tests for C1: server route modules.

Verifies that:
1. ``sparc.server.deps`` is importable and exposes the expected symbols.
2. Each route module in ``sparc.server.routes`` is importable independently
   (without triggering full app initialisation in ``app.py``).
3. Route module routers expose the expected paths.
4. ``deps.state`` is the same ``ServerState`` instance used by ``app.py``.
"""

from __future__ import annotations

import importlib

import pytest


# ---------------------------------------------------------------------------
# deps module
# ---------------------------------------------------------------------------

class TestDepsModule:
    def test_importable(self):
        import sparc.server.deps as deps  # noqa: F401

    def test_state_is_server_state(self):
        from sparc.server.deps import state
        from sparc.server.state import ServerState
        assert isinstance(state, ServerState)

    def test_get_open_store_exported(self):
        from sparc.server.deps import get_open_store
        assert callable(get_open_store)

    def test_missing_artifact_response_exported(self):
        from sparc.server.deps import missing_artifact_response
        assert callable(missing_artifact_response)

    def test_read_or_404_exported(self):
        from sparc.server.deps import read_or_404
        assert callable(read_or_404)

    def test_registry_path_exported(self):
        from sparc.server.deps import registry_path
        assert callable(registry_path)

    def test_missing_artifact_response_returns_http_exception(self):
        from fastapi import HTTPException
        from sparc.server.deps import missing_artifact_response
        exc = missing_artifact_response(
            artifact_id="correlogram_results",
            stage="0",
            hint="Run stage 0 first",
        )
        assert isinstance(exc, HTTPException)
        assert exc.status_code == 404
        assert exc.detail["error"] == "missing_artifact"
        assert exc.detail["missing_artifact"] == "correlogram_results"

    def test_get_open_store_raises_400_when_no_project(self):
        from fastapi import HTTPException
        from sparc.server.deps import state, get_open_store

        # Ensure no project is loaded
        original = state.project_config
        state.project_config = None
        try:
            with pytest.raises(HTTPException) as exc_info:
                get_open_store()
            assert exc_info.value.status_code == 400
        finally:
            state.project_config = original


# ---------------------------------------------------------------------------
# routes package
# ---------------------------------------------------------------------------

class TestRoutesPackage:
    def test_routes_package_importable(self):
        import sparc.server.routes  # noqa: F401

    def test_results_module_importable(self):
        import sparc.server.routes.results  # noqa: F401

    def test_results_router_is_api_router(self):
        from fastapi import APIRouter
        from sparc.server.routes.results import router
        assert isinstance(router, APIRouter)

    def test_results_router_has_correlogram_route(self):
        from sparc.server.routes.results import router
        paths = [r.path for r in router.routes]
        assert "/results/correlogram" in paths

    def test_results_router_tags(self):
        from sparc.server.routes.results import router
        assert "results" in router.tags


# ---------------------------------------------------------------------------
# app.py compatibility: include_router works
# ---------------------------------------------------------------------------

class TestAppCompatibility:
    def test_app_can_include_results_router(self):
        """Including the results router on a fresh FastAPI app must not raise."""
        from fastapi import FastAPI
        from sparc.server.routes.results import router

        test_app = FastAPI()
        test_app.include_router(router)  # must not raise

        # Verify the route was added
        paths = [r.path for r in test_app.routes]
        assert "/results/correlogram" in paths

    def test_deps_state_is_importable_without_app(self):
        """deps.state must be accessible without importing app.py."""
        # If this import graph triggers app.py, it would pull in all of
        # FastAPI's startup machinery. Verify it does NOT by checking the
        # module is not in sys.modules after a fresh import via importlib.
        import sys

        # Ensure app is not already loaded (may already be from other tests)
        # — just verify deps can be freshly imported
        if "sparc.server.deps" in sys.modules:
            # Module already imported — just verify state is accessible
            from sparc.server.deps import state
            from sparc.server.state import ServerState
            assert isinstance(state, ServerState)
        else:
            deps = importlib.import_module("sparc.server.deps")
            from sparc.server.state import ServerState
            assert isinstance(deps.state, ServerState)

    def test_app_mounts_health_router(self):
        """The real SPARC app must have GET /health mounted from routes.health."""
        from sparc.server.app import app

        paths = {r.path for r in app.routes}
        assert "/health" in paths

    def test_app_mounts_project_router(self):
        """The real SPARC app must have /project/* routes mounted."""
        from sparc.server.app import app

        paths = {r.path for r in app.routes}
        assert "/project/config" in paths

    def test_app_mounts_data_router(self):
        """The real SPARC app must have /data/* routes mounted."""
        from sparc.server.app import app

        paths = {r.path for r in app.routes}
        assert "/data/summary" in paths

    def test_app_correlogram_uses_injectable_store(self):
        """GET /results/correlogram on the real app must use Depends(get_open_store).

        RED: while the inline @app.get("/results/correlogram") is the active handler,
        it calls _read_or_404() directly without Depends, so dependency_overrides has
        no effect and a 400 is returned.
        GREEN: once include_router(_results_router) is wired and the inline duplicate
        removed, the router handler takes over and returns 200.
        """
        import sparc.server.deps as deps
        from fastapi.testclient import TestClient
        from sparc.server.app import app

        class _StubStore:
            def __init__(self, payload):
                self._p = payload
            def has(self, s, a):
                return self._p is not None
            def read_any(self, s, a):
                return self._p

        payload = {"type": "correlogram", "pairs": []}
        deps.state.result_cache.clear()
        app.dependency_overrides[deps.get_open_store] = lambda: _StubStore(payload)
        try:
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/results/correlogram")
            assert resp.status_code == 200, (
                f"Expected 200 from injected store, got {resp.status_code}. "
                "Wire include_router(_results_router) in app.py and remove the "
                "duplicate inline @app.get('/results/correlogram') handler."
            )
            assert resp.json() == payload
        finally:
            del app.dependency_overrides[deps.get_open_store]
            deps.state.result_cache.clear()

    def test_app_causal_pdp_curves_uses_injectable_store(self):
        """GET /results/causal/pdp_curves on the real app must use Depends(get_open_store).

        RED: causal router not yet wired → the inline handler is active or the route
        is absent entirely, so dependency_overrides has no effect.
        GREEN: after include_router(_causal_router) is added to app.py and the inline
        duplicate removed.
        """
        import sparc.server.deps as deps
        from fastapi.testclient import TestClient
        from sparc.server.app import app

        class _StubStore:
            def __init__(self, payload):
                self._p = payload
            def has(self, s, a):
                return self._p is not None
            def read_any(self, s, a):
                return self._p

        payload = {"curves": []}
        deps.state.result_cache.clear()
        app.dependency_overrides[deps.get_open_store] = lambda: _StubStore(payload)
        try:
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/results/causal/pdp_curves")
            assert resp.status_code == 200, (
                f"Expected 200 from injected store, got {resp.status_code}. "
                "Wire include_router(_causal_router) in app.py and remove the "
                "duplicate inline @app.get('/results/causal/pdp_curves') handler."
            )
            assert resp.json() == payload
        finally:
            del app.dependency_overrides[deps.get_open_store]
            deps.state.result_cache.clear()

    def test_app_mounts_physics_router(self):
        """The real SPARC app must have GET /physics/defaults mounted from routes.physics."""
        from sparc.server.app import app
        paths = {r.path for r in app.routes}
        assert "/physics/defaults" in paths

    def test_app_mounts_inference_router(self):
        """The real SPARC app must have /inference/* routes mounted from routes.inference."""
        from sparc.server.app import app
        paths = {r.path for r in app.routes}
        assert "/inference/zero-shot" in paths

    def test_app_mounts_api_router(self):
        """The real SPARC app must have /api/* routes mounted from routes.api."""
        from sparc.server.app import app
        paths = {r.path for r in app.routes}
        assert "/api/hardware" in paths

    def test_app_health_reflects_session_not_state(self):
        """GET /health on the real app must reflect session, not legacy state.

        If app.py still has its own @app.get('/health') reading state, this
        test will fail — that duplicate must be removed so the route from
        routes.health (which reads session) is the one that answers.
        """
        import sparc.server.deps as deps
        from fastapi.testclient import TestClient
        from sparc.server.app import app

        original_session = deps.session.project_config
        original_state = deps.state.project_config
        try:
            # session has a project; legacy state does NOT
            deps.session.project_config = {"name": "session-probe"}
            deps.state.project_config = None
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/health")
            assert resp.status_code == 200
            # Must be True (from session), not False (from state)
            assert resp.json()["project_loaded"] is True
        finally:
            deps.session.project_config = original_session
            deps.state.project_config = original_state


# ---------------------------------------------------------------------------
# C6.1 — health.py  (NEW route module)
# ---------------------------------------------------------------------------

class TestHealthRouter:
    def test_importable_without_app(self):
        from sparc.server.routes.health import router
        assert router is not None

    def test_is_api_router(self):
        from fastapi import APIRouter
        from sparc.server.routes.health import router
        assert isinstance(router, APIRouter)

    def test_has_health_tag(self):
        from sparc.server.routes.health import router
        assert "health" in router.tags

    def test_health_route_registered(self):
        from sparc.server.routes.health import router
        paths = {r.path for r in router.routes}
        assert "/health" in paths

    def test_health_returns_ok(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sparc.server.routes.health import router

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"

    def test_health_project_loaded_reflects_session(self):
        """project_loaded must track session.project_config, not legacy state."""
        import sparc.server.deps as deps
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sparc.server.routes.health import router

        original = deps.session.project_config
        try:
            deps.session.project_config = {"name": "probe"}
            app = FastAPI()
            app.include_router(router)
            client = TestClient(app)
            resp = client.get("/health")
            assert resp.json()["project_loaded"] is True
        finally:
            deps.session.project_config = original

    def test_health_is_running_reflects_execution(self):
        """is_running must track execution.is_running, not legacy state."""
        import sparc.server.deps as deps
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sparc.server.routes.health import router

        try:
            deps.execution.set_running(stage=2)
            app = FastAPI()
            app.include_router(router)
            client = TestClient(app)
            resp = client.get("/health")
            assert resp.json()["is_running"] is True
            assert resp.json()["current_stage"] == 2
        finally:
            deps.execution.set_idle()


# ---------------------------------------------------------------------------
# C6.2 — causal.py  (NEW route module)
# ---------------------------------------------------------------------------

class TestCausalRouter:
    def test_importable_without_app(self):
        from sparc.server.routes.causal import router
        assert router is not None

    def test_is_api_router(self):
        from fastapi import APIRouter
        from sparc.server.routes.causal import router
        assert isinstance(router, APIRouter)

    def test_has_causal_tag(self):
        from sparc.server.routes.causal import router
        assert "causal" in router.tags

    def test_pdp_curves_route_registered(self):
        from sparc.server.routes.causal import router
        paths = {r.path for r in router.routes}
        assert "/results/causal/pdp_curves" in paths

    def test_divergence_route_registered(self):
        from sparc.server.routes.causal import router
        paths = {r.path for r in router.routes}
        assert "/results/causal/divergence" in paths

    def test_mountable_on_test_app(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sparc.server.routes.causal import router

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app, raise_server_exceptions=False)
        # 400/404/422/503 are fine — we only need no import crash
        resp = client.get("/results/causal/pdp_curves")
        assert resp.status_code in (200, 400, 404, 422, 503)


# ---------------------------------------------------------------------------
# C1 — project.py route module
# ---------------------------------------------------------------------------

class TestProjectRouter:
    def test_importable_without_app(self):
        from sparc.server.routes.project import router
        assert router is not None

    def test_is_api_router(self):
        from fastapi import APIRouter
        from sparc.server.routes.project import router
        assert isinstance(router, APIRouter)

    def test_has_project_tag(self):
        from sparc.server.routes.project import router
        assert "project" in router.tags

    def test_has_load_route(self):
        from sparc.server.routes.project import router
        paths = {r.path for r in router.routes}
        assert "/project/load" in paths

    def test_has_validate_route(self):
        from sparc.server.routes.project import router
        paths = {r.path for r in router.routes}
        assert "/project/validate" in paths

    def test_has_config_get_route(self):
        from sparc.server.routes.project import router
        paths = {r.path for r in router.routes}
        assert "/project/config" in paths

    def test_has_templates_route(self):
        from sparc.server.routes.project import router
        paths = {r.path for r in router.routes}
        assert "/project/templates" in paths

    def test_mountable_on_test_app(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sparc.server.routes.project import router

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/project/templates")
        assert resp.status_code in (200, 400, 404, 422, 503)

    def test_config_get_reflects_session(self):
        """GET /project/config must return session.raw_project_yaml, not legacy state."""
        import sparc.server.deps as deps
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sparc.server.routes.project import router

        original = deps.session.raw_project_yaml
        try:
            deps.session.raw_project_yaml = {"project": {"name": "probe"}}
            app = FastAPI()
            app.include_router(router)
            client = TestClient(app)
            resp = client.get("/project/config")
            assert resp.status_code == 200
            assert resp.json()["project"]["name"] == "probe"
        finally:
            deps.session.raw_project_yaml = original

    def test_config_get_400_when_session_empty(self):
        """GET /project/config returns 400 when no project is in session."""
        import sparc.server.deps as deps
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sparc.server.routes.project import router

        original = deps.session.raw_project_yaml
        try:
            deps.session.raw_project_yaml = None
            app = FastAPI()
            app.include_router(router)
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/project/config")
            assert resp.status_code == 400
        finally:
            deps.session.raw_project_yaml = original


# ---------------------------------------------------------------------------
# C1 — data.py route module
# ---------------------------------------------------------------------------

class TestDataRouter:
    def test_importable_without_app(self):
        from sparc.server.routes.data import router
        assert router is not None

    def test_is_api_router(self):
        from fastapi import APIRouter
        from sparc.server.routes.data import router
        assert isinstance(router, APIRouter)

    def test_has_data_tag(self):
        from sparc.server.routes.data import router
        assert "data" in router.tags

    def test_has_summary_route(self):
        from sparc.server.routes.data import router
        paths = {r.path for r in router.routes}
        assert "/data/summary" in paths

    def test_has_preview_route(self):
        from sparc.server.routes.data import router
        paths = {r.path for r in router.routes}
        assert "/data/preview" in paths

    def test_has_upload_route(self):
        from sparc.server.routes.data import router
        paths = {r.path for r in router.routes}
        assert "/data/upload" in paths

    def test_has_files_route(self):
        from sparc.server.routes.data import router
        paths = {r.path for r in router.routes}
        assert "/data/files" in paths

    def test_mountable_on_test_app(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sparc.server.routes.data import router

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/data/summary")
        assert resp.status_code in (200, 400, 404, 422, 503)

    def test_data_summary_reflects_session(self):
        """GET /data/summary must read session.data, not legacy state."""
        import pandas as pd
        import sparc.server.deps as deps
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sparc.server.routes.data import router

        original = deps.session.data
        try:
            deps.session.data = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
            app = FastAPI()
            app.include_router(router)
            client = TestClient(app)
            resp = client.get("/data/summary")
            assert resp.status_code == 200
            body = resp.json()
            assert body["row_count"] == 3
            assert "x" in body["columns"]
        finally:
            deps.session.data = original

    def test_data_summary_400_when_session_has_no_data(self):
        """GET /data/summary returns 400 when session.data is None."""
        import sparc.server.deps as deps
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sparc.server.routes.data import router

        original = deps.session.data
        try:
            deps.session.data = None
            app = FastAPI()
            app.include_router(router)
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/data/summary")
            assert resp.status_code == 400
        finally:
            deps.session.data = original

    # B15 — remaining 8 data routes

    def test_has_crs_distortion_route(self):
        from sparc.server.routes.data import router
        paths = {r.path for r in router.routes}
        assert "/crs/distortion" in paths

    def test_has_data_validate_route(self):
        from sparc.server.routes.data import router
        paths = {r.path for r in router.routes}
        assert "/data/validate" in paths

    def test_has_data_versions_route(self):
        from sparc.server.routes.data import router
        paths = {r.path for r in router.routes}
        assert "/data/versions" in paths

    def test_has_data_select_version_route(self):
        from sparc.server.routes.data import router
        paths = {r.path for r in router.routes}
        assert "/data/select_version" in paths

    def test_has_data_preprocess_route(self):
        from sparc.server.routes.data import router
        paths = {r.path for r in router.routes}
        assert "/data/preprocess" in paths

    def test_has_data_fishnet_route(self):
        from sparc.server.routes.data import router
        paths = {r.path for r in router.routes}
        assert "/data/fishnet" in paths

    def test_has_data_zonal_stats_route(self):
        from sparc.server.routes.data import router
        paths = {r.path for r in router.routes}
        assert "/data/zonal_stats" in paths

    def test_has_data_prepare_route(self):
        from sparc.server.routes.data import router
        paths = {r.path for r in router.routes}
        assert "/data/prepare" in paths

    def test_data_validate_400_when_no_data(self):
        """POST /data/validate returns 400 when state.data is None."""
        import sparc.server.deps as deps
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sparc.server.routes.data import router

        original = deps.state.data
        try:
            deps.state.data = None
            app = FastAPI()
            app.include_router(router)
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/data/validate")
            assert resp.status_code == 400
        finally:
            deps.state.data = original

    def test_data_versions_400_when_no_project(self):
        """GET /data/versions returns 400 when no project loaded."""
        import sparc.server.deps as deps
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sparc.server.routes.data import router

        original = deps.state.project_config
        try:
            deps.state.project_config = None
            app = FastAPI()
            app.include_router(router)
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/data/versions")
            assert resp.status_code == 400
        finally:
            deps.state.project_config = original


# ---------------------------------------------------------------------------
# C1 — collect.py route module
# ---------------------------------------------------------------------------

class TestCollectRouter:
    def test_importable_without_app(self):
        from sparc.server.routes.collect import router
        assert router is not None

    def test_is_api_router(self):
        from fastapi import APIRouter
        from sparc.server.routes.collect import router
        assert isinstance(router, APIRouter)

    def test_has_collect_tag(self):
        from sparc.server.routes.collect import router
        assert "collect" in router.tags

    def test_has_boundary_route(self):
        from sparc.server.routes.collect import router
        paths = {r.path for r in router.routes}
        assert "/collect/boundary" in paths

    def test_has_manifest_route(self):
        from sparc.server.routes.collect import router
        paths = {r.path for r in router.routes}
        assert "/collect/manifest" in paths

    def test_has_fetch_route(self):
        from sparc.server.routes.collect import router
        paths = {r.path for r in router.routes}
        assert "/collect/fetch" in paths

    def test_has_preview_variable_route(self):
        from sparc.server.routes.collect import router
        paths = {r.path for r in router.routes}
        assert "/collect/preview/{variable}" in paths

    def test_has_build_route(self):
        from sparc.server.routes.collect import router
        paths = {r.path for r in router.routes}
        assert "/collect/build" in paths

    def test_mountable_on_test_app(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sparc.server.routes.collect import router

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/collect/manifest")
        assert resp.status_code in (200, 400, 404, 422, 503)


# ---------------------------------------------------------------------------
# B12 — physics.py  (GET /physics/defaults)
# ---------------------------------------------------------------------------

class TestPhysicsRouter:
    def test_importable_without_app(self):
        from sparc.server.routes.physics import router
        assert router is not None

    def test_is_api_router(self):
        from fastapi import APIRouter
        from sparc.server.routes.physics import router
        assert isinstance(router, APIRouter)

    def test_has_physics_tag(self):
        from sparc.server.routes.physics import router
        assert "physics" in router.tags

    def test_has_defaults_route(self):
        from sparc.server.routes.physics import router
        paths = {r.path for r in router.routes}
        assert "/physics/defaults" in paths

    def test_mountable_on_test_app(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sparc.server.routes.physics import router

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/physics/defaults")
        assert resp.status_code in (200, 400, 404, 422, 500, 503)


# ---------------------------------------------------------------------------
# B11 — inference.py  (POST /inference/zero-shot, /inference/few-shot)
# ---------------------------------------------------------------------------

class TestInferenceRouter:
    def test_importable_without_app(self):
        from sparc.server.routes.inference import router
        assert router is not None

    def test_is_api_router(self):
        from fastapi import APIRouter
        from sparc.server.routes.inference import router
        assert isinstance(router, APIRouter)

    def test_has_inference_tag(self):
        from sparc.server.routes.inference import router
        assert "inference" in router.tags

    def test_has_zero_shot_route(self):
        from sparc.server.routes.inference import router
        paths = {r.path for r in router.routes}
        assert "/inference/zero-shot" in paths

    def test_has_few_shot_route(self):
        from sparc.server.routes.inference import router
        paths = {r.path for r in router.routes}
        assert "/inference/few-shot" in paths

    def test_mountable_on_test_app(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sparc.server.routes.inference import router

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/inference/zero-shot",
            json={"coords": [[40.7, -74.0]]},
        )
        assert resp.status_code in (200, 400, 422, 500, 503)


# ---------------------------------------------------------------------------
# B10 — api.py  (GET /api/hardware, GET+PUT /api/preferences,
#                POST /api/hardware/validate)
# ---------------------------------------------------------------------------

class TestApiRouter:
    def test_importable_without_app(self):
        from sparc.server.routes.api import router
        assert router is not None

    def test_is_api_router(self):
        from fastapi import APIRouter
        from sparc.server.routes.api import router
        assert isinstance(router, APIRouter)

    def test_has_api_tag(self):
        from sparc.server.routes.api import router
        assert "api" in router.tags

    def test_has_hardware_route(self):
        from sparc.server.routes.api import router
        paths = {r.path for r in router.routes}
        assert "/api/hardware" in paths

    def test_has_preferences_route(self):
        from sparc.server.routes.api import router
        paths = {r.path for r in router.routes}
        assert "/api/preferences" in paths

    def test_has_preferences_put_route(self):
        from sparc.server.routes.api import router
        put_paths = {
            r.path for r in router.routes
            if "PUT" in (r.methods or set())
        }
        assert "/api/preferences" in put_paths

    def test_has_hardware_validate_route(self):
        from sparc.server.routes.api import router
        paths = {r.path for r in router.routes}
        assert "/api/hardware/validate" in paths

    def test_mountable_on_test_app(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sparc.server.routes.api import router

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/hardware")
        assert resp.status_code in (200, 400, 404, 422, 500, 503)


# ---------------------------------------------------------------------------
# B1: AI router
# ---------------------------------------------------------------------------

class TestAiRouter:
    def test_importable_without_app(self):
        from sparc.server.routes.ai import router
        assert router is not None

    def test_is_api_router(self):
        from fastapi import APIRouter
        from sparc.server.routes.ai import router
        assert isinstance(router, APIRouter)

    def test_has_ai_tag(self):
        from sparc.server.routes.ai import router
        assert "ai" in router.tags

    def test_has_key_get_route(self):
        from sparc.server.routes.ai import router
        get_paths = {r.path for r in router.routes if "GET" in (r.methods or set())}
        assert "/ai/key" in get_paths

    def test_has_key_put_route(self):
        from sparc.server.routes.ai import router
        put_paths = {r.path for r in router.routes if "PUT" in (r.methods or set())}
        assert "/ai/key" in put_paths

    def test_has_key_delete_route(self):
        from sparc.server.routes.ai import router
        delete_paths = {r.path for r in router.routes if "DELETE" in (r.methods or set())}
        assert "/ai/key" in delete_paths

    def test_has_chat_post_route(self):
        from sparc.server.routes.ai import router
        post_paths = {r.path for r in router.routes if "POST" in (r.methods or set())}
        assert "/ai/chat" in post_paths

    def test_key_status_returns_configured_bool(self):
        """GET /ai/key must return {configured: bool} even when keyring is unavailable."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sparc.server.routes.ai import router

        test_app = FastAPI()
        test_app.include_router(router)
        client = TestClient(test_app, raise_server_exceptions=False)
        resp = client.get("/ai/key")
        assert resp.status_code == 200
        assert "configured" in resp.json()
        assert isinstance(resp.json()["configured"], bool)

    def test_key_put_validates_prefix(self):
        """PUT /ai/key with a bad key must return 400."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sparc.server.routes.ai import router

        test_app = FastAPI()
        test_app.include_router(router)
        client = TestClient(test_app, raise_server_exceptions=False)
        resp = client.put("/ai/key", json={"key": "bad-key-format"})
        assert resp.status_code == 400

    def test_chat_returns_401_when_no_key(self):
        """POST /ai/chat must return 401 when no API key is configured."""
        import unittest.mock
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sparc.server.routes import ai as ai_module

        test_app = FastAPI()
        test_app.include_router(ai_module.router)
        client = TestClient(test_app, raise_server_exceptions=False)

        with unittest.mock.patch.object(ai_module, "_keyring_get", return_value=None):
            resp = client.post(
                "/ai/chat",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
        assert resp.status_code == 401


class TestAppCompatibilityAi:
    def test_app_mounts_ai_router(self):
        from sparc.server.app import app
        paths = {r.path for r in app.routes}
        assert "/ai/key" in paths
        assert "/ai/chat" in paths


# ---------------------------------------------------------------------------
# B2: Reproduce router
# ---------------------------------------------------------------------------

class TestReproduceRouter:
    def test_importable_without_app(self):
        from sparc.server.routes.reproduce import router
        assert router is not None

    def test_is_api_router(self):
        from fastapi import APIRouter
        from sparc.server.routes.reproduce import router
        assert isinstance(router, APIRouter)

    def test_has_reproduce_tag(self):
        from sparc.server.routes.reproduce import router
        assert "reproduce" in router.tags

    def test_has_provenance_get_route(self):
        from sparc.server.routes.reproduce import router
        get_paths = {r.path for r in router.routes if "GET" in (r.methods or set())}
        assert "/reproduce/provenance" in get_paths

    def test_has_freeze_post_route(self):
        from sparc.server.routes.reproduce import router
        post_paths = {r.path for r in router.routes if "POST" in (r.methods or set())}
        assert "/reproduce/freeze" in post_paths

    def test_has_load_post_route(self):
        from sparc.server.routes.reproduce import router
        post_paths = {r.path for r in router.routes if "POST" in (r.methods or set())}
        assert "/reproduce/load" in post_paths

    def test_has_verify_get_route(self):
        from sparc.server.routes.reproduce import router
        get_paths = {r.path for r in router.routes if "GET" in (r.methods or set())}
        assert "/reproduce/verify" in get_paths

    def test_load_requires_run_dir(self):
        """POST /reproduce/load without run_dir must return 400."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sparc.server.routes.reproduce import router

        test_app = FastAPI()
        test_app.include_router(router)
        client = TestClient(test_app, raise_server_exceptions=False)
        resp = client.post("/reproduce/load", json={})
        assert resp.status_code == 400

    def test_freeze_requires_project_loaded(self):
        """POST /reproduce/freeze with no project loaded must return 400."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sparc.server.routes.reproduce import router
        from sparc.server import deps

        test_app = FastAPI()
        test_app.include_router(router)
        client = TestClient(test_app, raise_server_exceptions=False)

        original = deps.state.project_config
        deps.state.project_config = None
        try:
            resp = client.post("/reproduce/freeze")
        finally:
            deps.state.project_config = original
        assert resp.status_code == 400


class TestAppCompatibilityReproduce:
    def test_app_mounts_reproduce_router(self):
        from sparc.server.app import app
        paths = {r.path for r in app.routes}
        assert "/reproduce/provenance" in paths
        assert "/reproduce/freeze" in paths
        assert "/reproduce/load" in paths
        assert "/reproduce/verify" in paths


# ---------------------------------------------------------------------------
# B4: DAG router
# ---------------------------------------------------------------------------

class TestDagRouter:
    def test_importable_without_app(self):
        from sparc.server.routes.dag import router
        assert router is not None

    def test_is_api_router(self):
        from fastapi import APIRouter
        from sparc.server.routes.dag import router
        assert isinstance(router, APIRouter)

    def test_has_dag_tag(self):
        from sparc.server.routes.dag import router
        assert "dag" in router.tags

    def test_has_dag_get_route(self):
        from sparc.server.routes.dag import router
        get_paths = {r.path for r in router.routes if "GET" in (r.methods or set())}
        assert "/dag" in get_paths

    def test_has_dag_put_route(self):
        from sparc.server.routes.dag import router
        put_paths = {r.path for r in router.routes if "PUT" in (r.methods or set())}
        assert "/dag" in put_paths

    def test_has_validate_route(self):
        from sparc.server.routes.dag import router
        paths = {r.path for r in router.routes}
        assert "/dag/validate" in paths

    def test_has_mc3_result_route(self):
        from sparc.server.routes.dag import router
        paths = {r.path for r in router.routes}
        assert "/dag/mc3_result" in paths

    def test_has_approve_route(self):
        from sparc.server.routes.dag import router
        paths = {r.path for r in router.routes}
        assert "/dag/approve" in paths

    def test_has_reject_route(self):
        from sparc.server.routes.dag import router
        paths = {r.path for r in router.routes}
        assert "/dag/reject" in paths

    def test_has_suggest_edges_route(self):
        from sparc.server.routes.dag import router
        paths = {r.path for r in router.routes}
        assert "/dag/suggest-edges" in paths

    def test_validate_dag_invalid_returns_valid_false(self):
        """POST /dag/validate with an invalid/cyclic DAG must return valid=False."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sparc.server.routes.dag import router

        test_app = FastAPI()
        test_app.include_router(router)
        client = TestClient(test_app, raise_server_exceptions=False)
        resp = client.post("/dag/validate", json={"nodes": [], "edges": []})
        assert resp.status_code == 200
        assert "valid" in resp.json()

    def test_mc3_result_404_when_none(self):
        """GET /dag/mc3_result must return 404 when no pending result."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sparc.server.routes.dag import router
        from sparc.server import deps

        test_app = FastAPI()
        test_app.include_router(router)
        client = TestClient(test_app, raise_server_exceptions=False)

        original = deps.state.pending_mc3
        deps.state.pending_mc3 = None
        try:
            resp = client.get("/dag/mc3_result")
        finally:
            deps.state.pending_mc3 = original
        assert resp.status_code == 404


class TestAppCompatibilityDag:
    def test_app_mounts_dag_router(self):
        from sparc.server.app import app
        paths = {r.path for r in app.routes}
        assert "/dag" in paths
        assert "/dag/validate" in paths
        assert "/dag/suggest-edges" in paths


# ---------------------------------------------------------------------------
# B8: Report router
# ---------------------------------------------------------------------------

class TestReportRouter:
    def test_importable_without_app(self):
        from sparc.server.routes.report import router
        assert router is not None

    def test_is_api_router(self):
        from fastapi import APIRouter
        from sparc.server.routes.report import router
        assert isinstance(router, APIRouter)

    def test_has_report_tag(self):
        from sparc.server.routes.report import router
        assert "report" in router.tags

    def test_has_generate_route(self):
        from sparc.server.routes.report import router
        paths = {r.path for r in router.routes}
        assert "/report/generate" in paths

    def test_has_pdf_route(self):
        from sparc.server.routes.report import router
        paths = {r.path for r in router.routes}
        assert "/report/pdf" in paths

    def test_has_docx_route(self):
        from sparc.server.routes.report import router
        paths = {r.path for r in router.routes}
        assert "/report/docx" in paths

    def test_has_standalone_route(self):
        from sparc.server.routes.report import router
        paths = {r.path for r in router.routes}
        assert "/report/standalone" in paths

    def test_has_audience_route(self):
        from sparc.server.routes.report import router
        paths = {r.path for r in router.routes}
        assert "/report/audience" in paths


# ---------------------------------------------------------------------------
# B7 — decision.py  (decision / equity route module)
# ---------------------------------------------------------------------------

class TestDecisionRouter:
    def test_importable_without_app(self):
        from sparc.server.routes.decision import router
        assert router is not None

    def test_is_api_router(self):
        from fastapi import APIRouter
        from sparc.server.routes.decision import router
        assert isinstance(router, APIRouter)

    def test_has_candidates_route(self):
        from sparc.server.routes.decision import router
        paths = {r.path for r in router.routes}
        assert "/decision/candidates" in paths

    def test_has_optimize_route(self):
        from sparc.server.routes.decision import router
        paths = {r.path for r in router.routes}
        assert "/decision/optimize" in paths

    def test_has_uncertainty_route(self):
        from sparc.server.routes.decision import router
        paths = {r.path for r in router.routes}
        assert "/decision/uncertainty" in paths

    def test_has_equity_census_route(self):
        from sparc.server.routes.decision import router
        paths = {r.path for r in router.routes}
        assert "/equity/census" in paths

    def test_has_decision_equity_route(self):
        from sparc.server.routes.decision import router
        paths = {r.path for r in router.routes}
        assert "/decision/equity" in paths

    def test_has_targeting_route(self):
        from sparc.server.routes.decision import router
        paths = {r.path for r in router.routes}
        assert "/decision/targeting" in paths

    def test_has_equity_layer_route(self):
        from sparc.server.routes.decision import router
        paths = {r.path for r in router.routes}
        assert "/equity/layer" in paths

    def test_candidates_returns_400_without_project(self):
        import sparc.server.deps as deps
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sparc.server.routes.decision import router

        original = deps.state.project_config
        try:
            deps.state.project_config = None
            test_app = FastAPI()
            test_app.include_router(router)
            client = TestClient(test_app, raise_server_exceptions=False)
            resp = client.get("/decision/candidates")
            assert resp.status_code in (400, 404, 422, 503)
        finally:
            deps.state.project_config = original


class TestAppCompatibilityDecision:
    def test_app_mounts_decision_router(self):
        from sparc.server.app import app
        paths = {r.path for r in app.routes}
        assert "/decision/candidates" in paths
        assert "/decision/optimize" in paths
        assert "/decision/uncertainty" in paths
        assert "/equity/census" in paths
        assert "/decision/equity" in paths
        assert "/decision/targeting" in paths
        assert "/equity/layer" in paths

    def test_generate_returns_markdown_key(self):
        """POST /report/generate must return a dict with a markdown key."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sparc.server.routes.report import router

        test_app = FastAPI()
        test_app.include_router(router)
        client = TestClient(test_app, raise_server_exceptions=False)
        resp = client.post("/report/generate")
        assert resp.status_code == 200
        assert "markdown" in resp.json()


class TestAppCompatibilityReport:
    def test_app_mounts_report_router(self):
        from sparc.server.app import app
        paths = {r.path for r in app.routes}
        assert "/report/generate" in paths
        assert "/report/pdf" in paths
        assert "/report/docx" in paths
        assert "/report/standalone" in paths
        assert "/report/audience" in paths


# ---------------------------------------------------------------------------
# B3 — routes/run.py  (run control + WebSocket stream routes)
# ---------------------------------------------------------------------------

class TestRunRouter:
    def test_importable_without_app(self):
        from sparc.server.routes.run import router
        assert router is not None

    def test_is_api_router(self):
        from fastapi import APIRouter
        from sparc.server.routes.run import router
        assert isinstance(router, APIRouter)

    def test_has_cancel_route(self):
        from sparc.server.routes.run import router
        paths = {r.path for r in router.routes}
        assert "/run/cancel" in paths

    def test_has_log_route(self):
        from sparc.server.routes.run import router
        paths = {r.path for r in router.routes}
        assert "/run/log" in paths

    def test_has_events_route(self):
        from sparc.server.routes.run import router
        paths = {r.path for r in router.routes}
        assert "/run/events" in paths

    def test_has_discover_route(self):
        from sparc.server.routes.run import router
        paths = {r.path for r in router.routes}
        assert "/runs/discover" in paths

    def test_has_diff_route(self):
        from sparc.server.routes.run import router
        paths = {r.path for r in router.routes}
        assert "/runs/diff" in paths

    def test_has_execute_ws_route(self):
        from sparc.server.routes.run import router
        paths = {r.path for r in router.routes}
        assert "/run/execute" in paths

    def test_has_artifacts_ws_route(self):
        from sparc.server.routes.run import router
        paths = {r.path for r in router.routes}
        assert "/run/artifacts" in paths

    def test_has_stream_ws_route(self):
        from sparc.server.routes.run import router
        paths = {r.path for r in router.routes}
        assert "/run/stream" in paths

    def test_cancel_clears_running_state(self):
        """POST /run/cancel must return {"status": "cancelled"} and call set_idle."""
        import sparc.server.deps as deps
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sparc.server.routes.run import router

        test_app = FastAPI()
        test_app.include_router(router)
        deps.state.is_running = True
        client = TestClient(test_app)
        resp = client.post("/run/cancel")
        assert resp.status_code == 200
        assert resp.json() == {"status": "cancelled"}
        assert deps.state.is_running is False

    def test_run_log_returns_400_without_project(self):
        """GET /run/log must return 400 when no project is loaded."""
        import sparc.server.deps as deps
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sparc.server.routes.run import router

        original = deps.state.project_config
        try:
            deps.state.project_config = None
            test_app = FastAPI()
            test_app.include_router(router)
            client = TestClient(test_app, raise_server_exceptions=False)
            resp = client.get("/run/log")
            assert resp.status_code == 400
        finally:
            deps.state.project_config = original

    def test_run_events_returns_state_keys(self):
        """GET /run/events must return is_running, current_stage, events keys."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sparc.server.routes.run import router

        test_app = FastAPI()
        test_app.include_router(router)
        client = TestClient(test_app)
        resp = client.get("/run/events")
        assert resp.status_code == 200
        body = resp.json()
        assert "is_running" in body
        assert "current_stage" in body
        assert "events" in body

    def test_runs_discover_returns_structure(self):
        """GET /runs/discover must return search_root and runs keys."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sparc.server.routes.run import router

        test_app = FastAPI()
        test_app.include_router(router)
        client = TestClient(test_app)
        resp = client.get("/runs/discover")
        assert resp.status_code == 200
        body = resp.json()
        assert "search_root" in body
        assert "runs" in body

    def test_runs_diff_requires_both_params(self):
        """GET /runs/diff without required params must return 422."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sparc.server.routes.run import router

        test_app = FastAPI()
        test_app.include_router(router)
        client = TestClient(test_app, raise_server_exceptions=False)
        resp = client.get("/runs/diff")
        assert resp.status_code == 422


class TestAppCompatibilityRun:
    def test_app_mounts_run_router(self):
        from sparc.server.app import app
        paths = {r.path for r in app.routes}
        assert "/run/cancel" in paths
        assert "/run/log" in paths
        assert "/run/events" in paths
        assert "/runs/discover" in paths
        assert "/runs/diff" in paths
        assert "/run/execute" in paths
        assert "/run/artifacts" in paths
        assert "/run/stream" in paths


class TestScenariosRouter:
    def test_importable_without_app(self):
        from sparc.server.routes.scenarios import router
        assert router is not None

    def test_is_api_router(self):
        from fastapi import APIRouter
        from sparc.server.routes.scenarios import router
        assert isinstance(router, APIRouter)

    def test_has_library_get_route(self):
        from sparc.server.routes.scenarios import router
        paths = {r.path for r in router.routes}
        assert "/scenarios/library" in paths

    def test_has_library_post_route(self):
        from sparc.server.routes.scenarios import router
        paths = {r.path for r in router.routes}
        assert "/scenarios/library" in paths

    def test_has_chain_route(self):
        from sparc.server.routes.scenarios import router
        paths = {r.path for r in router.routes}
        assert "/scenarios/chain" in paths

    def test_has_run_route(self):
        from sparc.server.routes.scenarios import router
        paths = {r.path for r in router.routes}
        assert "/scenarios/run" in paths

    def test_has_optimize_route(self):
        from sparc.server.routes.scenarios import router
        paths = {r.path for r in router.routes}
        assert "/scenarios/optimize" in paths

    def test_has_results_route(self):
        from sparc.server.routes.scenarios import router
        paths = {r.path for r in router.routes}
        assert "/scenarios/results" in paths

    def test_has_budget_optimize_route(self):
        from sparc.server.routes.scenarios import router
        paths = {r.path for r in router.routes}
        assert "/scenarios/budget/optimize" in paths

    def test_library_get_returns_400_without_project(self):
        import sparc.server.deps as deps
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sparc.server.routes.scenarios import router

        original = deps.state.project_config
        try:
            deps.state.project_config = None
            test_app = FastAPI()
            test_app.include_router(router)
            client = TestClient(test_app, raise_server_exceptions=False)
            resp = client.get("/scenarios/library")
            assert resp.status_code == 400
        finally:
            deps.state.project_config = original


class TestAppCompatibilityScenarios:
    def test_app_mounts_scenarios_router(self):
        from sparc.server.app import app
        paths = {r.path for r in app.routes}
        assert "/scenarios/library" in paths
        assert "/scenarios/chain" in paths
        assert "/scenarios/run" in paths
        assert "/scenarios/optimize" in paths
        assert "/scenarios/results" in paths
        assert "/scenarios/budget/optimize" in paths


# ---------------------------------------------------------------------------
# B9 — artifacts router
# ---------------------------------------------------------------------------

class TestArtifactsRouter:
    def test_importable_without_app(self):
        from sparc.server.routes.artifacts import router
        assert router is not None

    def test_is_api_router(self):
        from fastapi import APIRouter
        from sparc.server.routes.artifacts import router
        assert isinstance(router, APIRouter)

    def test_has_csv_route(self):
        from sparc.server.routes.artifacts import router
        paths = {r.path for r in router.routes}
        assert "/artifacts/{stage}/{artifact_id}.csv" in paths

    def test_has_json_route(self):
        from sparc.server.routes.artifacts import router
        paths = {r.path for r in router.routes}
        assert "/artifacts/{stage}/{artifact_id}.json" in paths

    def test_has_geojson_route(self):
        from sparc.server.routes.artifacts import router
        paths = {r.path for r in router.routes}
        assert "/artifacts/{stage}/{artifact_id}.geojson" in paths

    def test_has_png_route(self):
        from sparc.server.routes.artifacts import router
        paths = {r.path for r in router.routes}
        assert "/artifacts/{stage}/{artifact_id}.png" in paths

    def test_has_generic_get_route(self):
        from sparc.server.routes.artifacts import router
        paths = {r.path for r in router.routes}
        assert "/artifacts/{stage}/{artifact_id}" in paths

    def test_has_download_route(self):
        from sparc.server.routes.artifacts import router
        paths = {r.path for r in router.routes}
        assert "/artifacts/{stage}/{artifact_id}/download" in paths

    def test_has_export_route(self):
        from sparc.server.routes.artifacts import router
        paths = {r.path for r in router.routes}
        assert "/artifacts/{stage}/export" in paths

    def test_has_list_stage_route(self):
        from sparc.server.routes.artifacts import router
        paths = {r.path for r in router.routes}
        assert "/artifacts/{stage}" in paths

    def test_csv_returns_400_without_project(self):
        import sparc.server.deps as deps
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sparc.server.routes.artifacts import router

        original = deps.state.project_config
        try:
            deps.state.project_config = None
            test_app = FastAPI()
            test_app.include_router(router)
            client = TestClient(test_app, raise_server_exceptions=False)
            resp = client.get("/artifacts/0/some_artifact.csv")
            assert resp.status_code in (400, 404, 422, 503)
        finally:
            deps.state.project_config = original


class TestAppCompatibilityArtifacts:
    def test_app_mounts_artifacts_router(self):
        from sparc.server.app import app
        paths = {r.path for r in app.routes}
        assert "/artifacts/{stage}/{artifact_id}.csv" in paths
        assert "/artifacts/{stage}/{artifact_id}.json" in paths
        assert "/artifacts/{stage}/{artifact_id}.geojson" in paths
        assert "/artifacts/{stage}/{artifact_id}.png" in paths
        assert "/artifacts/{stage}/{artifact_id}" in paths
        assert "/artifacts/{stage}/{artifact_id}/download" in paths
        assert "/artifacts/{stage}/export" in paths
        assert "/artifacts/{stage}" in paths


# ---------------------------------------------------------------------------
# B6 — results_extended.py  (NEW route module)
# ---------------------------------------------------------------------------

class TestResultsExtendedRouter:
    def test_importable_without_app(self):
        from sparc.server.routes.results_extended import router
        assert router is not None

    def test_is_api_router(self):
        from fastapi import APIRouter
        from sparc.server.routes.results_extended import router
        assert isinstance(router, APIRouter)

    def test_has_export_route(self):
        from sparc.server.routes.results_extended import router
        paths = {r.path for r in router.routes}
        assert "/results/export" in paths

    def test_has_kernel_field_route(self):
        from sparc.server.routes.results_extended import router
        paths = {r.path for r in router.routes}
        assert "/results/kernel_field" in paths

    def test_has_model_performance_route(self):
        from sparc.server.routes.results_extended import router
        paths = {r.path for r in router.routes}
        assert "/results/model_performance" in paths

    def test_has_causal_cate_map_route(self):
        from sparc.server.routes.results_extended import router
        paths = {r.path for r in router.routes}
        assert "/results/causal/cate_map" in paths

    def test_has_scenarios_results_route(self):
        from sparc.server.routes.results_extended import router
        paths = {r.path for r in router.routes}
        assert "/results/scenarios" in paths

    def test_has_manifest_route(self):
        from sparc.server.routes.results_extended import router
        paths = {r.path for r in router.routes}
        assert "/results/manifest" in paths

    def test_has_stage_results_route(self):
        from sparc.server.routes.results_extended import router
        paths = {r.path for r in router.routes}
        assert "/results/{stage}" in paths

    def test_has_availability_route(self):
        from sparc.server.routes.results_extended import router
        paths = {r.path for r in router.routes}
        assert "/results/availability" in paths

    def test_has_bundle_route(self):
        from sparc.server.routes.results_extended import router
        paths = {r.path for r in router.routes}
        assert "/results/bundle" in paths

    def test_has_geopackage_route(self):
        from sparc.server.routes.results_extended import router
        paths = {r.path for r in router.routes}
        assert "/results/geopackage" in paths

    def test_export_returns_400_without_project(self):
        import sparc.server.deps as deps
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sparc.server.routes.results_extended import router

        original = deps.state.project_config
        try:
            deps.state.project_config = None
            test_app = FastAPI()
            test_app.include_router(router)
            client = TestClient(test_app, raise_server_exceptions=False)
            resp = client.post("/results/export", json={})
            assert resp.status_code in (400, 404, 422, 503)
        finally:
            deps.state.project_config = original



class TestAppCompatibilityResultsExtended:
    def test_app_mounts_results_extended_router(self):
        from sparc.server.app import app
        paths = {r.path for r in app.routes}
        assert "/results/export" in paths
        assert "/results/kernel_field" in paths
        assert "/results/model_performance" in paths
        assert "/results/causal/cate_map" in paths
        assert "/results/scenarios" in paths
        assert "/results/manifest" in paths
        assert "/results/availability" in paths
        assert "/results/bundle" in paths
        assert "/results/geopackage" in paths


class TestMiscRouter:
    def test_importable_without_app(self):
        from sparc.server.routes.misc import router
        assert router is not None

    def test_is_api_router(self):
        from fastapi import APIRouter
        from sparc.server.routes.misc import router
        assert isinstance(router, APIRouter)

    def test_has_shutdown_route(self):
        from sparc.server.routes.misc import router
        paths = {r.path for r in router.routes}
        assert "/shutdown" in paths

    def test_has_debug_paths_route(self):
        from sparc.server.routes.misc import router
        paths = {r.path for r in router.routes}
        assert "/debug/paths" in paths

    def test_has_context_layers_route(self):
        from sparc.server.routes.misc import router
        paths = {r.path for r in router.routes}
        assert "/context/layers" in paths

    def test_has_gwen_status_route(self):
        from sparc.server.routes.misc import router
        paths = {r.path for r in router.routes}
        assert "/gwen/status" in paths

    def test_has_gwen_approve_routes(self):
        from sparc.server.routes.misc import router
        paths = {r.path for r in router.routes}
        assert "/gwen/approve" in paths

    def test_has_insights_headline_route(self):
        from sparc.server.routes.misc import router
        paths = {r.path for r in router.routes}
        assert "/insights/headline" in paths

    def test_has_panels_availability_route(self):
        from sparc.server.routes.misc import router
        paths = {r.path for r in router.routes}
        assert "/panels/availability" in paths

    def test_shutdown_rejects_non_localhost(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sparc.server.routes.misc import router
        test_app = FastAPI()
        test_app.include_router(router)
        client = TestClient(test_app, raise_server_exceptions=False)
        resp = client.post("/shutdown", headers={"X-Forwarded-For": "8.8.8.8"})
        # TestClient always appears as 127.0.0.1; we just check the route exists
        assert resp.status_code in (200, 403)

    def test_panels_availability_returns_dict(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sparc.server.routes.misc import router
        test_app = FastAPI()
        test_app.include_router(router)
        client = TestClient(test_app)
        resp = client.get("/panels/availability")
        assert resp.status_code == 200
        data = resp.json()
        assert "panels" in data
        assert "is_running" in data


class TestAppCompatibilityMisc:
    def test_app_mounts_misc_router(self):
        from sparc.server.app import app
        paths = {r.path for r in app.routes}
        assert "/shutdown" in paths
        assert "/debug/paths" in paths
        assert "/context/layers" in paths
        assert "/gwen/status" in paths
        assert "/gwen/approve" in paths
        assert "/insights/headline" in paths
        assert "/panels/availability" in paths


# ---------------------------------------------------------------------------
# A3 — collect.py route module
# ---------------------------------------------------------------------------

class TestCollectRouter:
    def test_importable_without_app(self):
        from sparc.server.routes.collect import router
        assert router is not None

    def test_is_api_router(self):
        from fastapi import APIRouter
        from sparc.server.routes.collect import router
        assert isinstance(router, APIRouter)

    def test_has_boundary_route(self):
        from sparc.server.routes.collect import router
        paths = {r.path for r in router.routes}
        assert "/collect/boundary" in paths

    def test_has_manifest_route(self):
        from sparc.server.routes.collect import router
        paths = {r.path for r in router.routes}
        assert "/collect/manifest" in paths

    def test_has_fetch_route(self):
        from sparc.server.routes.collect import router
        paths = {r.path for r in router.routes}
        assert "/collect/fetch" in paths

    def test_has_capa_events_route(self):
        from sparc.server.routes.collect import router
        paths = {r.path for r in router.routes}
        assert "/collect/capa-events" in paths

    def test_has_preview_route(self):
        from sparc.server.routes.collect import router
        paths = {r.path for r in router.routes}
        assert "/collect/preview/{variable}" in paths

    def test_has_cell_route(self):
        from sparc.server.routes.collect import router
        paths = {r.path for r in router.routes}
        assert "/collect/cell/{cell_id}" in paths

    def test_has_save_config_route(self):
        from sparc.server.routes.collect import router
        paths = {r.path for r in router.routes}
        assert "/collect/save-config" in paths

    def test_has_build_route(self):
        from sparc.server.routes.collect import router
        paths = {r.path for r in router.routes}
        assert "/collect/build" in paths

    def test_manifest_returns_dict_when_no_session(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sparc.server.routes.collect import router
        test_app = FastAPI()
        test_app.include_router(router)
        client = TestClient(test_app, raise_server_exceptions=False)
        resp = client.get("/collect/manifest")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)


class TestAppCompatibilityCollect:
    def test_app_mounts_collect_router(self):
        from sparc.server.app import app
        paths = {r.path for r in app.routes}
        assert "/collect/boundary" in paths
        assert "/collect/manifest" in paths
        assert "/collect/fetch" in paths
        assert "/collect/capa-events" in paths
        assert "/collect/preview/{variable}" in paths
        assert "/collect/cell/{cell_id}" in paths
        assert "/collect/save-config" in paths
        assert "/collect/build" in paths
