"""Security (SPEC §10.8, api.md §0.2): auth, cookie exchange, Host and Origin checks, safe paths, uploads."""

from __future__ import annotations

import asyncio
import os
import tracemalloc

import pytest
from fastapi.testclient import TestClient

from sparc.studio.errors import ApiError
from sparc.studio.security import (
    COOKIE_NAME,
    UPLOAD_SUFFIXES,
    PathGuard,
    allowed_hosts,
    check_upload_suffix,
    safe_next,
    safe_path,
    session_value,
    stream_upload,
    upload_cap_bytes,
)
from tests.studio.conftest import ORIGIN, TOKEN


@pytest.fixture
def bare(app):
    """A client with no credentials (lifespan running)."""
    with TestClient(app) as c:
        yield c


def _code(r):
    return r.json()["error"]["code"]


def test_401_without_auth(bare):
    for path in ("/api/meta", "/api/jobs", "/api/settings", "/openapi.json"):
        r = bare.get(path)
        assert r.status_code == 401, path
        assert _code(r) == "unauthorized"
    r = bare.get("/api/meta", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


def test_health_and_spa_need_no_auth(bare):
    r = bare.get("/api/health")
    assert r.status_code == 200 and r.json()["ok"] is True
    assert bare.get("/").status_code == 200


def test_bearer_auth(bare):
    assert bare.get("/api/meta", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200
    assert bare.get("/openapi.json", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200


def test_cookie_exchange(bare):
    r = bare.get("/auth", params={"t": TOKEN, "next": "/p/p_1/launch?x=1"}, follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "/p/p_1/launch?x=1"
    cookie = r.headers["set-cookie"]
    assert cookie.startswith(f"{COOKIE_NAME}={session_value(TOKEN)}")
    low = cookie.lower()
    assert "httponly" in low and "samesite=strict" in low and "path=/" in low and "secure" not in low
    bare.cookies.set(COOKIE_NAME, session_value(TOKEN))
    assert bare.get("/api/meta").status_code == 200
    bare.cookies.set(COOKIE_NAME, "forged")
    assert bare.get("/api/meta").status_code == 401


def test_wrong_token_and_unsafe_next(bare):
    r = bare.get("/auth", params={"t": "nope"}, follow_redirects=False)
    assert r.status_code == 401 and "text/html" in r.headers["content-type"]
    for bad in ("//evil.example/x", "https://evil.example/", "javascript:alert(1)", "relative", "/\\evil"):
        r = bare.get("/auth", params={"t": TOKEN, "next": bad}, follow_redirects=False)
        assert r.headers["location"] == "/", bad
    assert safe_next("/runs") == "/runs" and safe_next(None) == "/"


def test_bad_host_is_400(bare):
    for host in ("evil.example", "127.0.0.1:9999", "localhost"):
        r = bare.get("/api/health", headers={"Host": host})
        assert r.status_code == 400, host
        assert _code(r) == "bad_host"


def test_allowed_hosts():
    hosts = allowed_hosts(8765, public_hosts=["studio.example", "box:9000", "::1"],
                          dev_origins=["http://localhost:5173"])
    assert {"127.0.0.1:8765", "localhost:8765", "[::1]:8765", "studio.example", "studio.example:8765",
            "box:9000", "localhost:5173"} <= hosts
    assert "localhost:8766" not in hosts


def test_cross_origin_post_is_403(bare):
    bare.cookies.set(COOKIE_NAME, session_value(TOKEN))
    r = bare.post("/api/queue/pause", headers={"Origin": "http://evil.example"})
    assert r.status_code == 403 and _code(r) == "bad_origin"
    r = bare.post("/api/queue/pause")                                   # cookie without Origin/Referer
    assert r.status_code == 403
    r = bare.post("/api/queue/pause", headers={"Referer": "http://evil.example/page"})
    assert r.status_code == 403
    assert bare.post("/api/queue/pause", headers={"Origin": ORIGIN}).status_code == 200
    assert bare.post("/api/queue/resume", headers={"Referer": f"{ORIGIN}/jobs"}).status_code == 200
    bare.cookies.clear()
    bearer = {"Authorization": f"Bearer {TOKEN}"}
    assert bare.post("/api/queue/resume", headers=bearer).status_code == 200              # scripts: no Origin
    r = bare.post("/api/queue/resume", headers={**bearer, "Origin": "http://evil.example"})
    assert r.status_code == 403


def test_dev_origin(make_app):
    app = make_app(dev_origins=["http://localhost:5173"])
    with TestClient(app, base_url="http://localhost:5173") as c:
        c.cookies.set(COOKIE_NAME, session_value(TOKEN))
        r = c.post("/api/queue/pause", headers={"Origin": "http://localhost:5173"})
        assert r.status_code == 200


def test_docs_are_disabled(client):
    for path in ("/docs", "/redoc"):
        r = client.get(path)
        assert "swagger" not in r.text.lower() and "redoc.standalone" not in r.text.lower()


# ---------------------------------------------------------------------------
# safe_path
# ---------------------------------------------------------------------------

def test_safe_path(tmp_path):
    root = tmp_path / "root"
    (root / "a" / "b").mkdir(parents=True)
    (root / "a" / "b" / "f.csv").write_text("x")
    assert safe_path("a/b/f.csv", root) == (root / "a" / "b" / "f.csv").resolve()
    assert safe_path("a/./b", root) == (root / "a" / "b").resolve()
    for bad in ("../x", "a/../../x", "a/..", "..", "/etc/passwd", "C:/Windows", "C:\\Windows", "\\\\server\\s",
                "", "a\x00b"):
        with pytest.raises(ApiError) as exc:
            safe_path(bad, root)
        assert exc.value.status == 422 and exc.value.detail["errors"][0]["code"] == "unsafe_path", bad


def test_safe_path_rejects_symlink_escape(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("s")
    os.symlink(outside, root / "link")
    os.symlink(outside / "secret.txt", root / "file_link")
    for raw in ("link/secret.txt", "file_link", "link"):
        with pytest.raises(ApiError):
            safe_path(raw, root)
    inner = root / "inner"
    inner.mkdir()
    os.symlink(inner, root / "ok_link")
    assert safe_path("ok_link", root) == inner.resolve()


def test_path_guard_roots(tmp_path):
    ws, ext = tmp_path / "ws", tmp_path / "external_run"
    ws.mkdir()
    ext.mkdir()
    (ext / "manifest.json").write_text("{}")
    guard = PathGuard(ws)
    with pytest.raises(ApiError):
        guard.resolve("manifest.json", ext)                  # not under an allowed root yet
    guard.allow(ext)
    assert guard.resolve("manifest.json", ext) == (ext / "manifest.json").resolve()


# ---------------------------------------------------------------------------
# uploads
# ---------------------------------------------------------------------------

class FakeRequest:
    def __init__(self, chunks, headers=None):
        self._chunks = chunks
        self.headers = headers or {}
        self.consumed = 0

    async def stream(self):
        for c in self._chunks:
            self.consumed += 1
            yield c


def test_upload_suffix_allowlist(tmp_path):
    for ok in UPLOAD_SUFFIXES:
        assert check_upload_suffix(f"data{ok.upper()}") == ok
    for bad in ("x.exe", "x.py", "x.pkl", "noext", "x.csv.sh"):
        with pytest.raises(ApiError) as exc:
            check_upload_suffix(bad)
        assert exc.value.status == 415 and exc.value.code == "bad_suffix"
    with pytest.raises(ApiError):
        asyncio.run(stream_upload(FakeRequest([b"x"]), tmp_path / "evil.pkl", max_bytes=10))
    assert not list(tmp_path.iterdir())


def test_upload_size_cap(tmp_path):
    from sparc.studio.settings import Settings, default_settings

    cap = upload_cap_bytes(Settings(**default_settings()))
    assert cap == 2 * 1024 ** 3
    req = FakeRequest([b"x" * 10], headers={"content-length": str(cap + 1)})
    with pytest.raises(ApiError) as exc:
        asyncio.run(stream_upload(req, tmp_path / "big.csv", max_bytes=cap))
    assert exc.value.status == 413 and exc.value.code == "too_large"
    assert req.consumed == 0                                   # refused before reading the body
    # a body that lies about (or omits) its length is cut off while streaming, leaving no partial file
    req = FakeRequest([b"y" * 4096] * 300)
    with pytest.raises(ApiError) as exc:
        asyncio.run(stream_upload(req, tmp_path / "big.csv", max_bytes=1 << 20))
    assert exc.value.status == 413
    assert not list(tmp_path.iterdir())


def test_streamed_upload_memory_bound(tmp_path):
    """64 MB streamed in 64 kB pieces: peak Python memory stays near the 1 MB write buffer."""
    total = 64 * 1024 * 1024
    piece = 64 * 1024

    async def body():
        for i in range(total // piece):
            yield bytes([i % 251]) * piece

    tracemalloc.start()
    try:
        n = asyncio.run(stream_upload(body(), tmp_path / "data.parquet", max_bytes=2 * 1024 ** 3))
        _cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert n == total and (tmp_path / "data.parquet").stat().st_size == total
    assert peak < 6 * 1024 * 1024, f"peak {peak / 2**20:.1f} MB"
    with open(tmp_path / "data.parquet", "rb") as f:
        f.seek(piece * 3)
        assert f.read(1) == bytes([3])


def test_allow_remote_required(tmp_path, capsys):
    from sparc.studio.cli import main

    rc = main(["--workspace", str(tmp_path / "ws"), "--host", "0.0.0.0", "--port", "0", "--no-browser"])
    assert rc == 2
    assert "--allow-remote" in capsys.readouterr().err
    assert not (tmp_path / "ws" / "studio.lock.json").exists()
