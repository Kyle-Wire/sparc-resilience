"""``sparc studio`` (SPEC §10.9).

::

    sparc studio [--workspace DIR] [--host 127.0.0.1] [--port 8765|0] [--no-browser] [--allow-remote]
                 [--public-host H ...] [--token T] [--dev-origin URL] [--reindex] [--stop-jobs-on-exit]
                 [--log-level info]
    sparc studio --dump-openapi PATH

Start-up:

1. resolve the workspace (``--workspace``, ``$SPARC_STUDIO_HOME``, ``~/sparc-studio``);
2. if ``studio.lock.json`` names a live server (pid + ``create_time``) whose
   ``/api/health`` answers for this workspace, print (and open) its URL and
   exit 0; if that process is alive but does not answer within 10 s (still
   starting, or hung), exit 1 rather than run a second scheduler on the
   same workspace;
3. bind the socket here - the preferred port, else an ephemeral one (``--port
   0`` asks for one); the occupant of a busy port is never touched;
4. write the lock (with the real port) and the token, run uvicorn on the bound
   socket and open ``/auth?t=…`` unless ``--no-browser``.

``sparc studio``, ``sparc core studio`` and ``python -m sparc.core studio``
reach :func:`main` with their remaining argv (core-progress delegation).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

from sparc.studio import __version__

__all__ = ["main", "add_studio_arguments", "build_parser", "read_lock", "live_server_url", "wait_for_occupant",
           "bind_socket", "StudioServer", "settings_from_args"]

DEFAULT_PORT = 8765
OCCUPANT_WAIT_S = 10.0          # how long a live lock holder may take to answer /api/health
log = logging.getLogger("sparc.studio")


def add_studio_arguments(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--workspace", metavar="DIR", help="workspace folder (default $SPARC_STUDIO_HOME or ~/sparc-studio)")
    p.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT,
                   help=f"port (default {DEFAULT_PORT}; 0 = any free port; a busy port falls back to a free one)")
    p.add_argument("--no-browser", action="store_true", help="do not open a browser")
    p.add_argument("--allow-remote", action="store_true", help="allow binding to a non-loopback address")
    p.add_argument("--public-host", action="append", default=[], metavar="H",
                   help="extra Host header value to accept (repeatable), e.g. a tunnel's host name")
    p.add_argument("--token", help="fix the access token (dev / e2e; default: a new one per launch)")
    p.add_argument("--dev-origin", action="append", default=[], metavar="URL",
                   help="allow this origin for API calls (the Vite dev server, e.g. http://localhost:5173)")
    p.add_argument("--reindex", action="store_true",
                   help="rebuild the SQLite index from the workspace files before starting")
    p.add_argument("--stop-jobs-on-exit", action="store_true",
                   help="cancel running jobs on exit (default: they keep running and are reattached next time)")
    p.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    p.add_argument("--dump-openapi", metavar="PATH",
                   help="write the OpenAPI document to PATH ('-' = stdout) and exit; no server is started")
    return p


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sparc studio",
                                description="SPARC Studio - the local web app for the SPARC urban-heat pipeline.")
    p.add_argument("--version", action="version", version=f"SPARC Studio {__version__}")
    return add_studio_arguments(p)


# ---------------------------------------------------------------------------
# lock file
# ---------------------------------------------------------------------------

def read_lock(workspace) -> dict | None:
    from sparc.studio.workspace import read_json

    lock = read_json(workspace.lock_path)
    return lock if isinstance(lock, dict) else None


def _pid_matches(lock: dict, *, strict: bool = False) -> bool:
    """The lock's pid is alive (and, when the lock recorded it, has the same ``create_time``).

    ``strict``: a lock without a recorded ``create_time`` never matches (a reused pid cannot be told apart).
    """
    try:
        import psutil

        p = psutil.Process(int(lock["pid"]))
        ctime = lock.get("create_time")
        if ctime is None:
            return not strict
        return abs(p.create_time() - float(ctime)) < 0.01
    except Exception:
        return False


def live_server_url(workspace, timeout: float = 2.0) -> str | None:
    """The base URL of a live server already serving ``workspace`` (lock pid alive and health answering)."""
    lock = read_lock(workspace)
    if not lock or not lock.get("port") or not _pid_matches(lock):
        return None
    # the lock's own URL: loopback, or the bind address (e.g. [::1]) - both are in that server's Host allowlist
    base = str(lock.get("url") or f"http://127.0.0.1:{int(lock['port'])}").rstrip("/")
    try:
        req = urllib.request.Request(base + "/api/health")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    if not body.get("ok") or Path(body.get("workspace", "")).resolve() != workspace.root:
        return None
    return base


def wait_for_occupant(workspace, *, wait_s: float | None = None,
                      poll_s: float = 0.5) -> tuple[str | None, int | None]:
    """``(url, None)`` when a live server answers for ``workspace``; ``(None, pid)`` when the lock names a
    live Studio process that does not answer within ``wait_s`` (default :data:`OCCUPANT_WAIT_S`; it is
    still starting - a long ``--reindex`` - or hung); ``(None, None)`` when nobody holds the workspace."""
    deadline = time.monotonic() + (OCCUPANT_WAIT_S if wait_s is None else wait_s)
    while True:
        url = live_server_url(workspace)
        if url:
            return url, None
        lock = read_lock(workspace)
        if not lock or not _pid_matches(lock, strict=True):
            return None, None
        if time.monotonic() >= deadline:
            return None, int(lock["pid"])
        time.sleep(poll_s)


def _auth_url(base: str, token: str | None) -> str:
    return f"{base}/auth?t={token}" if token else base


# ---------------------------------------------------------------------------
# sockets
# ---------------------------------------------------------------------------

def bind_socket(host: str, port: int) -> socket.socket:
    """A listening socket on ``host:port``; when that port is busy, on a free ephemeral port instead."""
    host = host.strip("[]")                    # "[::1]" → "::1"
    family = socket.AF_INET6 if ":" in host else socket.AF_INET

    def _bind(p: int) -> socket.socket:
        s = socket.socket(family, socket.SOCK_STREAM)
        if os.name != "nt":
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, p))
        except OSError:
            s.close()
            raise
        s.listen(128)
        s.set_inheritable(True)
        return s

    try:
        return _bind(port)
    except OSError:
        if port == 0:
            raise
        print(f"sparc studio: port {port} is busy; using a free port instead", file=sys.stderr)
        return _bind(0)


def _server_class():
    import uvicorn

    class StudioServer(uvicorn.Server):
        """uvicorn.Server that ends SSE streams before draining connections (they never end on their own)."""

        def __init__(self, config, sctx=None):
            super().__init__(config)
            self.sctx = sctx

        async def shutdown(self, sockets=None):
            if self.sctx is not None:
                self.sctx.begin_shutdown()
            await super().shutdown(sockets=sockets)

    return StudioServer


def StudioServer(config, sctx=None):          # noqa: N802 - a factory that keeps uvicorn imports lazy
    return _server_class()(config, sctx)


def uvicorn_config(app, *, log_level: str = "warning"):
    import uvicorn

    return uvicorn.Config(app, log_level=log_level, lifespan="on", timeout_graceful_shutdown=5,
                          access_log=False, proxy_headers=False, server_header=False)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def settings_from_args(args: argparse.Namespace, *, port: int | None = None, token: str | None = None):
    from sparc.studio.settings import StudioSettings
    from sparc.studio.workspace import resolve_workspace

    return StudioSettings(workspace=resolve_workspace(args.workspace), host=args.host,
                          port=args.port if port is None else port, token=token or args.token,
                          allow_remote=args.allow_remote, public_hosts=list(args.public_host),
                          dev_origins=list(args.dev_origin), reindex=args.reindex,
                          stop_jobs_on_exit=args.stop_jobs_on_exit, log_level=args.log_level)


def _dump_openapi(args) -> int:
    import tempfile

    from sparc.studio.app import create_app
    from sparc.studio.settings import StudioSettings

    with tempfile.TemporaryDirectory(prefix="sparc-openapi-") as tmp:
        app = create_app(StudioSettings(workspace=tmp, token="openapi", sampler=False))
        doc = app.openapi()
    text = json.dumps(doc, indent=2, sort_keys=False) + "\n"
    if args.dump_openapi == "-":
        sys.stdout.write(text)
    else:
        out = Path(args.dump_openapi)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else list(argv))
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.dump_openapi:
        return _dump_openapi(args)

    from sparc.studio.security import generate_token, is_loopback
    from sparc.studio.workspace import resolve_workspace, utc_now, write_json_atomic, write_private

    ws = resolve_workspace(args.workspace)
    url, holder = wait_for_occupant(ws)
    if holder is not None:
        # two servers on one workspace would both schedule its jobs; the occupant is never touched
        print(f"sparc studio: another SPARC Studio process (pid {holder}) holds {ws.root} but its "
              f"/api/health does not answer; wait for it to finish starting, or stop it and run again",
              file=sys.stderr)
        return 1
    if url:
        token = None
        try:
            token = ws.token_path.read_text("utf-8").strip()
        except OSError:
            pass
        print(f"SPARC Studio is already running for {ws.root}: {_auth_url(url, token)}")
        if not args.no_browser:
            webbrowser.open(_auth_url(url, token))
        return 0
    if not is_loopback(args.host) and not args.allow_remote:
        print(f"sparc studio: binding to {args.host} exposes the server to the network; "
              f"pass --allow-remote to do so", file=sys.stderr)
        return 2
    ws.ensure()
    try:
        sock = bind_socket(args.host, args.port)
    except OSError as exc:
        print(f"sparc studio: cannot bind {args.host}:{args.port}: {exc}", file=sys.stderr)
        return 1
    port = sock.getsockname()[1]
    token = args.token or generate_token()
    settings = settings_from_args(args, port=port, token=token)
    base = settings.url
    write_private(ws.token_path, token + "\n")
    lock = {"pid": os.getpid(), "port": port, "url": base, "version": __version__, "started_utc": utc_now(),
            "workspace": str(ws.root)}
    try:
        import psutil

        lock["create_time"] = psutil.Process().create_time()
    except Exception:
        pass
    write_json_atomic(ws.lock_path, lock, private=True)

    from sparc.studio.app import create_app

    app = create_app(settings)
    sctx = app.state.studio
    server = StudioServer(uvicorn_config(app, log_level="warning"), sctx)
    sctx.request_shutdown = lambda: setattr(server, "should_exit", True)
    link = _auth_url(base, token)
    print(f"SPARC Studio {__version__} - workspace {ws.root}")
    print(f"Open {link}")
    if args.allow_remote and not is_loopback(args.host):
        print("Remote access is on: anyone with this link can use Studio.")
    sys.stdout.flush()
    if not args.no_browser:
        def _open():
            for _ in range(100):
                if server.started:
                    break
                time.sleep(0.05)
            webbrowser.open(link)

        threading.Thread(target=_open, daemon=True).start()
    code = 0
    try:
        server.run(sockets=[sock])
        if not server.started:
            code = 3
    except KeyboardInterrupt:
        code = 0
    except SystemExit as exc:
        code = int(exc.code or 0) if isinstance(exc.code, int) else 1
    finally:
        try:
            sock.close()
        except OSError:
            pass
        current = read_lock(ws)
        if current and current.get("pid") == os.getpid():
            try:
                ws.lock_path.unlink()
            except OSError:
                pass
    return code


if __name__ == "__main__":
    sys.exit(main())
