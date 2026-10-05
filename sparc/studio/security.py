"""Local-server security (SPEC §10.8, api.md §0.2): token → cookie, Host and Origin checks, safe paths, uploads.

* **Token.**  A per-launch ``secrets.token_urlsafe(32)`` (``--token T`` fixes
  it).  ``GET /auth?t=<token>`` exchanges it for the HttpOnly, SameSite=Strict
  cookie ``sparc_studio``; scripts and tests may send
  ``Authorization: Bearer <token>`` instead.  The cookie value is an HMAC of
  the token, so it stays valid across restarts that keep ``--token`` and dies
  with a new token.
* **Host allowlist.**  ``127.0.0.1:<port>``, ``localhost:<port>``,
  ``[::1]:<port>``, every ``--public-host`` and the host of every
  ``--dev-origin``; anything else → ``400 bad_host`` (DNS-rebinding guard).
* **Origin check.**  POST/PUT/PATCH/DELETE must carry an ``Origin`` (or
  ``Referer``) whose origin is one Studio serves (or a ``--dev-origin``),
  else ``403 bad_origin``.  A request authenticated by ``Authorization:
  Bearer`` and carrying neither header is accepted: a browser cannot attach
  that header cross-site without a CORS preflight, which Studio never grants.
* **Auth scope.**  ``/api/*`` (except ``/api/health``) and
  ``/openapi.json`` need the cookie or the bearer token → else ``401
  unauthorized``.  ``/auth``, the SPA shell and ``/assets/*`` are public.
* **Paths.**  :func:`safe_path` resolves symlinks and rejects ``..``,
  absolute paths and anything outside the allowed roots.
* **Uploads.**  :func:`stream_upload` writes raw ``PUT`` bodies in 1 MB
  chunks to a temporary file and renames it: no ``python-multipart``, no
  buffering of the body in memory, a suffix allowlist and a size cap.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import os
import re
import secrets
from pathlib import Path
from typing import AsyncIterator, Callable, Iterable, Sequence
from urllib.parse import urlsplit

from starlette.datastructures import Headers
from starlette.requests import cookie_parser
from starlette.responses import JSONResponse

from sparc.core import runio
from sparc.studio.errors import ApiError, error_body

__all__ = [
    "COOKIE_NAME", "UPLOAD_SUFFIXES", "UPLOAD_CHUNK", "UNSAFE_METHODS", "generate_token", "session_value",
    "is_loopback", "allowed_hosts", "allowed_origins", "safe_next", "SecurityMiddleware", "PathGuard", "safe_path",
    "check_upload_suffix", "stream_upload", "upload_cap_bytes",
]

COOKIE_NAME = "sparc_studio"
UPLOAD_SUFFIXES = (".csv", ".parquet", ".json", ".yml", ".yaml", ".tif", ".tiff", ".gpkg", ".geojson")
UPLOAD_CHUNK = 1 << 20
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_PUBLIC_API = frozenset({"/api/health"})


def generate_token() -> str:
    return secrets.token_urlsafe(32)


def session_value(token: str) -> str:
    """The cookie value for ``token`` (HMAC-SHA256, hex)."""
    return hmac.new(token.encode("utf-8"), b"sparc-studio-session-v1", hashlib.sha256).hexdigest()


def is_loopback(host: str) -> bool:
    h = host.strip("[]").lower()
    if h in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def _with_port(host: str, port: int) -> set[str]:
    """``host`` as given, plus ``host:port`` when it names no port."""
    h = host.strip().lower().rstrip("/")
    if "://" in h:
        h = urlsplit(h).netloc
    if not h:
        return set()
    if h.startswith("["):                      # [v6] or [v6]:port
        return {h} if re.search(r"\]:\d+$", h) else {h, f"{h}:{port}"}
    if h.count(":") == 1:                      # name:port
        return {h}
    if h.count(":") > 1:                       # bare IPv6 literal
        return {f"[{h}]", f"[{h}]:{port}"}
    return {h, f"{h}:{port}"}


def allowed_hosts(port: int, *, bind_host: str = "127.0.0.1", public_hosts: Iterable[str] = (),
                  dev_origins: Iterable[str] = ()) -> set[str]:
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}
    if bind_host and bind_host not in ("0.0.0.0", "::"):
        hosts |= _with_port(bind_host, port)
    for h in public_hosts:
        hosts |= _with_port(h, port)
    for o in dev_origins:
        netloc = urlsplit(o.strip()).netloc.lower()
        if netloc:
            hosts.add(netloc)
    return hosts


def allowed_origins(hosts: Iterable[str], dev_origins: Iterable[str] = ()) -> set[str]:
    out = {f"{scheme}://{h}" for h in hosts for scheme in ("http", "https")}
    out |= {o.strip().rstrip("/").lower() for o in dev_origins if o.strip()}
    return out


def _origin_of(url: str) -> str | None:
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if not parts.scheme or not parts.netloc:
        return None
    return f"{parts.scheme}://{parts.netloc}".lower()


def safe_next(next_path: str | None) -> str:
    """A same-origin path for ``/auth?next=`` (leading ``/``, not ``//``, no scheme, no backslash), else ``/``."""
    p = (next_path or "").strip()
    if not p.startswith("/") or p.startswith("//") or "\\" in p or "://" in p or any(ord(c) < 32 for c in p):
        return "/"
    return p


class SecurityMiddleware:
    """Pure ASGI middleware: Host allowlist, auth on ``/api`` + ``/openapi.json``, Origin check on unsafe methods.

    ``config`` is called per request and returns an object with ``token``
    (str), ``hosts`` (set) and ``origins`` (set) - normally the app's
    :class:`~sparc.studio.app.StudioContext`, so the values exist only once
    the app is configured.
    """

    def __init__(self, app, config: Callable[[], object]):
        self.app = app
        self.config = config

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        cfg = self.config()
        headers = Headers(scope=scope)
        host = headers.get("host", "").strip().lower()
        if host not in cfg.hosts:
            await self._reject(scope, receive, send, 400, "bad_host", f"Host {host!r} is not allowed")
            return
        path = scope.get("path", "")
        auth = None
        if (path == "/api" or path.startswith("/api/") or path == "/openapi.json") and path not in _PUBLIC_API:
            auth = self._authenticate(headers, cfg.token)
            if auth is None:
                await self._reject(scope, receive, send, 401, "unauthorized",
                                   "missing or invalid credentials: open the link `sparc studio` printed")
                return
        method = scope.get("method", "GET")
        if scope["type"] == "http" and method in UNSAFE_METHODS:
            origin = headers.get("origin")
            if origin is None and headers.get("referer"):
                origin = _origin_of(headers["referer"])
            if origin is None:
                if auth != "bearer":
                    await self._reject(scope, receive, send, 403, "bad_origin",
                                       f"{method} needs an Origin (or Referer) header")
                    return
            elif origin.strip().rstrip("/").lower() not in cfg.origins:
                await self._reject(scope, receive, send, 403, "bad_origin", f"cross-origin {method} refused")
                return
        await self.app(scope, receive, send)

    @staticmethod
    def _authenticate(headers: Headers, token: str | None) -> str | None:
        if not token:
            return None
        authz = headers.get("authorization", "")
        if authz[:7].lower() == "bearer " and hmac.compare_digest(authz[7:].strip().encode(), token.encode()):
            return "bearer"
        cookie = cookie_parser(headers.get("cookie", "")).get(COOKIE_NAME)
        if cookie and hmac.compare_digest(cookie.encode(), session_value(token).encode()):
            return "cookie"
        return None

    @staticmethod
    async def _reject(scope, receive, send, status: int, code: str, message: str) -> None:
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        await JSONResponse(error_body(code, message), status_code=status)(scope, receive, send)


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

def _path_error(raw: str, why: str) -> ApiError:
    return ApiError("validation", f"unsafe path {raw!r}: {why}",
                    detail={"errors": [{"path": "path", "message": why, "code": "unsafe_path"}]})


def safe_path(raw: str | os.PathLike, base: str | os.PathLike, roots: Sequence[str | os.PathLike] | None = None) -> Path:
    """Resolve the relative path ``raw`` under ``base`` and check it stays inside an allowed root.

    Rejects (``422 validation``, code ``unsafe_path``): empty paths, NUL
    bytes, absolute paths (POSIX, UNC or drive letters), any ``..``
    component, and - after resolving symlinks - anything outside ``roots``
    (default: ``base`` itself).
    """
    s = os.fspath(raw) if raw is not None else ""
    if not s or s.strip() == "":
        raise _path_error(s, "empty path")
    if "\x00" in s:
        raise _path_error(s, "NUL byte")
    norm = s.replace("\\", "/")
    if norm.startswith("/") or re.match(r"^[A-Za-z]:", norm) or os.path.isabs(s):
        raise _path_error(s, "absolute paths are not allowed")
    if any(part == ".." for part in norm.split("/")):
        raise _path_error(s, "'..' is not allowed")
    base_r = Path(base).resolve()
    target = (base_r / norm).resolve()
    allowed = [Path(r).resolve() for r in (roots if roots else [base_r])]
    if not any(target == r or r in target.parents for r in allowed):
        raise _path_error(s, "outside the allowed folders")
    return target


class PathGuard:
    """The roots path parameters may resolve into: the workspace plus folders registered at import.

    Feature items call :meth:`allow` for run, study or config directories
    imported in place (outside the workspace) - at import and when the
    registry rescans.
    """

    def __init__(self, workspace_root: str | os.PathLike):
        self._roots: list[Path] = [Path(workspace_root).resolve()]

    def allow(self, path: str | os.PathLike) -> None:
        p = Path(path).resolve()
        if p not in self._roots:
            self._roots.append(p)

    def roots(self) -> list[Path]:
        return list(self._roots)

    def resolve(self, raw: str | os.PathLike, base: str | os.PathLike) -> Path:
        """:func:`safe_path` of ``raw`` under ``base``; the result must lie in ``base`` and in an allowed root."""
        target = safe_path(raw, base)
        if not any(target == r or r in target.parents for r in self._roots):
            raise _path_error(os.fspath(raw), "outside the allowed folders")
        return target


# ---------------------------------------------------------------------------
# uploads
# ---------------------------------------------------------------------------

def upload_cap_bytes(settings) -> int:
    """The upload size cap from the user settings (``upload_max_gb``, default 2 GB)."""
    return int(float(getattr(settings, "upload_max_gb", 2) or 2) * 1024 ** 3)


def check_upload_suffix(name: str, allowed: Iterable[str] | None = UPLOAD_SUFFIXES) -> str:
    """The lower-case suffix of ``name``; ``415 bad_suffix`` when it is not in ``allowed`` (None: any)."""
    suffix = Path(name).suffix.lower()
    if allowed is not None and suffix not in {s.lower() for s in allowed}:
        raise ApiError("bad_suffix", f"files of type {suffix or '(none)'!r} cannot be uploaded",
                       detail={"suffix": suffix, "allowed": sorted(allowed)})
    return suffix


async def stream_upload(request, dest: str | os.PathLike, *, max_bytes: int,
                        allowed_suffixes: Iterable[str] | None = UPLOAD_SUFFIXES,
                        content_types: Iterable[str] | None = None, chunk_size: int = UPLOAD_CHUNK) -> int:
    """Stream a raw ``PUT`` body into ``dest`` without holding it in memory; returns the byte count.

    ``request`` is a Starlette request (its ``stream()``) or any async
    iterator of ``bytes``.  The body goes to a temporary sibling in
    ``chunk_size`` writes and is renamed onto ``dest`` only when complete,
    so a failed or oversized upload never leaves a partial file.  Errors:
    ``415 bad_suffix`` (suffix or content type), ``413 too_large``
    (``Content-Length`` or the streamed size above ``max_bytes``).
    """
    dest = Path(dest)
    check_upload_suffix(dest.name, allowed_suffixes)
    stream: AsyncIterator[bytes]
    headers = getattr(request, "headers", None)
    if headers is not None:
        if content_types is not None:
            ctype = headers.get("content-type", "").split(";")[0].strip().lower()
            if ctype not in {c.lower() for c in content_types}:
                raise ApiError("bad_suffix", f"content type {ctype or '(none)'!r} is not accepted",
                               detail={"content_type": ctype, "allowed": sorted(content_types)})
        length = headers.get("content-length")
        if length is not None and length.isdigit() and int(length) > max_bytes:
            raise _too_large(int(length), max_bytes)
        stream = request.stream()
    else:
        stream = request
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.upload-{secrets.token_hex(4)}.tmp")
    total = 0
    buf = bytearray()
    try:
        with open(tmp, "wb") as f:
            async for part in stream:
                if not part:
                    continue
                total += len(part)
                if total > max_bytes:
                    raise _too_large(total, max_bytes)
                buf += part
                if len(buf) >= chunk_size:
                    f.write(buf)
                    buf.clear()
            if buf:
                f.write(buf)
                buf.clear()
            f.flush()
            os.fsync(f.fileno())
        runio.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return total


def _too_large(n: int, cap: int) -> ApiError:
    return ApiError("too_large", f"upload exceeds the {cap / 2 ** 30:.3g} GB limit",
                    detail={"bytes": n, "max_bytes": cap})
