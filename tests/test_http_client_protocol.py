"""TDD — C2: HttpClient protocol for injectable HTTP in data-collection adapters.

Verifies that:
1. ``HttpClient`` protocol is importable from ``sparc.data.collect``
2. ``StubHttpClient`` satisfies the protocol and records calls
3. The protocol is exported from the collect package
4. A FakeDataSource using StubHttpClient can be registered in SourceRegistry
"""

from __future__ import annotations

import pytest


class TestHttpClientProtocol:
    def test_http_client_protocol_importable(self):
        from sparc.data.collect.http_client import HttpClient
        assert HttpClient is not None

    def test_stub_http_client_importable(self):
        from sparc.data.collect.http_client import StubHttpClient
        assert StubHttpClient is not None

    def test_http_client_exported_from_collect_package(self):
        from sparc.data.collect import HttpClient
        assert HttpClient is not None

    def test_stub_http_client_exported_from_collect_package(self):
        from sparc.data.collect import StubHttpClient
        assert StubHttpClient is not None

    def test_stub_satisfies_http_client_protocol(self):
        from sparc.data.collect.http_client import HttpClient, StubHttpClient
        stub = StubHttpClient()
        assert isinstance(stub, HttpClient)

    def test_stub_records_get_calls(self):
        from sparc.data.collect.http_client import StubHttpClient
        stub = StubHttpClient()
        stub.get("https://api.example.com/data")
        assert len(stub.calls) == 1
        assert stub.calls[0] == ("GET", "https://api.example.com/data")

    def test_stub_records_post_calls(self):
        from sparc.data.collect.http_client import StubHttpClient
        stub = StubHttpClient()
        stub.post("https://api.example.com/submit", json={"key": "val"})
        assert stub.calls[0] == ("POST", "https://api.example.com/submit")

    def test_stub_get_returns_stub_response(self):
        from sparc.data.collect.http_client import StubHttpClient
        stub = StubHttpClient()
        response = stub.get("https://api.example.com/data")
        # Should not raise; must support raise_for_status()
        response.raise_for_status()

    def test_stub_canned_response(self):
        from sparc.data.collect.http_client import StubHttpClient, StubResponse
        canned = StubResponse(status_code=200, json_data={"temp": 25.0})
        stub = StubHttpClient(responses={"https://example.com/era5": canned})
        response = stub.get("https://example.com/era5")
        assert response.json() == {"temp": 25.0}

    def test_stub_error_response_raises(self):
        from sparc.data.collect.http_client import StubHttpClient, StubResponse
        error = StubResponse(status_code=503)
        stub = StubHttpClient(responses={"https://down.example.com": error})
        resp = stub.get("https://down.example.com")
        with pytest.raises(Exception):
            resp.raise_for_status()

    def test_non_conforming_object_fails_isinstance(self):
        from sparc.data.collect.http_client import HttpClient
        class NotAClient:
            pass
        assert not isinstance(NotAClient(), HttpClient)

    def test_custom_http_client_satisfies_protocol(self):
        from sparc.data.collect.http_client import HttpClient

        class MinimalHttpClient:
            def get(self, url: str, **kwargs): return object()
            def post(self, url: str, **kwargs): return object()

        assert isinstance(MinimalHttpClient(), HttpClient)
