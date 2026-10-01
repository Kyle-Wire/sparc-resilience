"""Error envelope and exception handlers (api.md §0.3, §16).

Every non-2xx JSON response is ``{"error": {code, message, detail?, action?}}``.
Route code raises :class:`ApiError`; FastAPI's own validation errors become
``422 validation`` with ``detail.errors = [{path, message, code}]``; anything
unexpected becomes ``500 internal`` with a ``detail.trace_id`` that is also
logged server-side.
"""

from __future__ import annotations

import logging
import secrets
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("sparc.studio")

__all__ = ["ApiError", "ERROR_CODES", "error_body", "error_response", "validation_error", "install_error_handlers"]

#: every code of api.md §16 with its HTTP status
ERROR_CODES: dict[str, int] = {
    "bad_host": 400, "unauthorized": 401, "bad_origin": 403,
    "not_found": 404, "output_missing": 404, "unknown_kind": 404, "unknown_view": 404, "unknown_layer": 404,
    "no_emulator": 404, "no_climate_factors": 404, "example_unavailable": 404,
    "conflict": 409, "conflict_revision": 409, "active": 409, "not_cancellable": 409, "not_resumable": 409,
    "not_ready": 409, "no_checkpoint": 409, "engine_memory": 409, "untrusted_pickle": 409, "preflight_failed": 409,
    "grid_mismatch": 409, "string_ids": 409, "has_children": 409, "exists": 409, "in_use": 409,
    "imported_in_place": 409, "superseded": 409, "precondition": 409, "locked": 409,
    "too_large": 413, "too_large_inline": 413, "too_many_features": 413,
    "bad_suffix": 415, "no_conversion": 415,
    "validation": 422, "yaml_error": 422, "needs_crs": 422, "needs_layers": 422, "needs_responses": 422,
    "needs_config": 422, "requirements": 422, "template_unavailable": 422, "mismatch": 422,
    "internal": 500,
}
_DEFAULT_CODE = {400: "validation", 401: "unauthorized", 403: "bad_origin", 404: "not_found", 405: "not_found",
                 409: "conflict", 413: "too_large", 415: "bad_suffix", 422: "validation"}


class ApiError(Exception):
    """An error with a code from api.md §16.  ``status`` defaults to the code's status.

    ``action`` is an :class:`~sparc.studio.schemas.common.Action`-shaped dict
    (or model) the UI renders as a one-click remedy.
    """

    def __init__(self, code: str, message: str, *, status: int | None = None, detail: dict | None = None,
                 action: Any = None, headers: dict[str, str] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = int(status or ERROR_CODES.get(code, 400))
        self.detail = detail
        self.action = action
        self.headers = headers

    def body(self) -> dict:
        return error_body(self.code, self.message, detail=self.detail, action=self.action)

    def response(self) -> JSONResponse:
        return JSONResponse(self.body(), status_code=self.status, headers=self.headers)


def error_body(code: str, message: str, *, detail: dict | None = None, action: Any = None) -> dict:
    err: dict[str, Any] = {"code": code, "message": message}
    if detail is not None:
        err["detail"] = detail
    if action is not None:
        err["action"] = action.model_dump(exclude_none=True) if hasattr(action, "model_dump") else action
    return {"error": err}


def error_response(code: str, message: str, *, status: int | None = None, detail: dict | None = None,
                   action: Any = None, headers: dict[str, str] | None = None) -> JSONResponse:
    return ApiError(code, message, status=status, detail=detail, action=action, headers=headers).response()


def validation_error(errors: list[dict], message: str = "invalid request") -> ApiError:
    """``422 validation`` with ``detail.errors = [{path, message, code}]``."""
    return ApiError("validation", message, detail={"errors": errors})


def _loc_path(loc) -> str:
    parts = [str(p) for p in loc if p not in ("body",)]
    if parts and parts[0] in ("query", "path", "header", "cookie"):
        parts = parts[1:]
    return ".".join(parts)


def pydantic_errors(errors) -> list[dict]:
    """Pydantic ``errors()`` → api.md ``{path, message, code}`` rows."""
    out = []
    for e in errors:
        out.append({"path": _loc_path(e.get("loc", ())), "message": str(e.get("msg", "invalid")),
                    "code": str(e.get("type", "invalid"))})
    return out


async def _api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    return exc.response()


async def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return validation_error(pydantic_errors(exc.errors())).response()


async def _http_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    code = _DEFAULT_CODE.get(exc.status_code, "internal" if exc.status_code >= 500 else "not_found")
    message = exc.detail if isinstance(exc.detail, str) else "request failed"
    if exc.status_code == 404 and message == "Not Found":
        message = f"no such resource: {request.url.path}"
    return error_response(code, message, status=exc.status_code, headers=getattr(exc, "headers", None))


async def _unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    trace_id = secrets.token_hex(6)
    log.exception("unhandled error %s on %s %s", trace_id, request.method, request.url.path)
    return error_response("internal", f"{type(exc).__name__}: {exc}"[:500], status=500, detail={"trace_id": trace_id})


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ApiError, _api_error_handler)
    app.add_exception_handler(RequestValidationError, _validation_handler)
    app.add_exception_handler(StarletteHTTPException, _http_handler)
    app.add_exception_handler(Exception, _unhandled_handler)
