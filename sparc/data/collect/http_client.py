"""HttpClient protocol and stub for injectable HTTP in data-collection adapters.

Defines a structural protocol that matches the ``requests`` session/module API
surface used by the fetch functions in ``sparc.data.collect``. Tests and
offline runs can inject a ``StubHttpClient`` to avoid real network calls.

Usage
-----
::

    from sparc.data.collect.http_client import HttpClient, StubHttpClient, StubResponse

    # In a test:
    canned = StubResponse(status_code=200, json_data={"era5_t2m": [20.1, 21.3]})
    stub = StubHttpClient(responses={"https://api.open-meteo.com/v1/era5": canned})

    # Adapters that accept an http_client parameter will use stub instead of requests:
    adapter = ERA5Adapter(http_client=stub)
    result = adapter.fetch(fishnet, bbox, params)
    assert stub.calls  # network was stubbed
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class HttpClient(Protocol):
    """Structural protocol matching the relevant ``requests`` module surface.

    Any object with ``get`` and ``post`` methods satisfies this protocol,
    including ``requests`` module itself (``requests.get`` / ``requests.post``).
    """

    def get(self, url: str, **kwargs: Any) -> Any: ...

    def post(self, url: str, **kwargs: Any) -> Any: ...


class StubResponse:
    """Canned HTTP response for use with :class:`StubHttpClient`.

    Parameters
    ----------
    status_code:
        HTTP status code. Codes >= 400 cause :meth:`raise_for_status` to raise.
    json_data:
        Mapping returned by :meth:`json`. Defaults to ``{}``.
    content:
        Raw bytes returned via ``.content``. Defaults to ``b""``.
    """

    def __init__(
        self,
        status_code: int = 200,
        json_data: dict | None = None,
        content: bytes = b"",
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else {}
        self.content = content

    def raise_for_status(self) -> None:
        """Raise ``RuntimeError`` for 4xx/5xx status codes."""
        if self.status_code >= 400:
            raise RuntimeError(
                f"StubResponse: HTTP {self.status_code} (simulated error)"
            )

    def json(self) -> dict:
        return self._json_data


class StubHttpClient:
    """In-memory :class:`HttpClient` that records calls and returns canned responses.

    Parameters
    ----------
    responses:
        Optional mapping from URL → :class:`StubResponse`. If the requested URL
        is not found, a default 200 empty response is returned.

    Attributes
    ----------
    calls:
        List of ``(method, url)`` tuples accumulated across all requests.
    """

    def __init__(self, responses: dict[str, StubResponse] | None = None) -> None:
        self._responses = responses or {}
        self.calls: list[tuple[str, str]] = []

    def _get_response(self, url: str) -> StubResponse:
        return self._responses.get(url, StubResponse())

    def get(self, url: str, **kwargs: Any) -> StubResponse:
        self.calls.append(("GET", url))
        return self._get_response(url)

    def post(self, url: str, **kwargs: Any) -> StubResponse:
        self.calls.append(("POST", url))
        return self._get_response(url)
