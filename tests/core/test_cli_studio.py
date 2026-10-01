"""CLI hooks: Studio delegation before argparse, the install hint, --progress/--job-id, exit codes, 130 on cancel."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

import sparc.__main__ as sparc_main
from sparc.core import cli, progress

ROOT = Path(cli.__file__).resolve().parents[2]
HINT = 'pip install "sparc[studio]"'
ENV_KEYS = (progress.ENV_SINK, progress.ENV_LEVEL, progress.ENV_JOB, progress.ENV_CANCEL)


@pytest.fixture(autouse=True)
def clean_progress():
    saved = {k: os.environ.get(k) for k in ENV_KEYS}
    for k in ENV_KEYS:
        os.environ.pop(k, None)
    progress.reset()
    yield
    progress.reset()
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _env() -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SPARC_")}
    env["PYTHONPATH"] = str(ROOT)
    return env


def _run(args, **kw) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=_env(), capture_output=True, text=True,
                          timeout=120, **kw)


@pytest.fixture
def studio_stub(monkeypatch):
    """A stand-in ``sparc.studio.cli`` (and stand-ins for any server dependency not installed here)."""
    calls: list[list[str]] = []
    stub = types.SimpleNamespace(rc=0)
    pkg = types.ModuleType("sparc.studio")
    pkg.__path__ = []
    mod = types.ModuleType("sparc.studio.cli")

    def main(argv):
        calls.append(list(argv))
        return stub.rc

    mod.main = main
    monkeypatch.setitem(sys.modules, "sparc.studio", pkg)
    monkeypatch.setitem(sys.modules, "sparc.studio.cli", mod)
    for dep in cli.STUDIO_DEPS:
        if dep not in sys.modules and importlib.util.find_spec(dep) is None:
            monkeypatch.setitem(sys.modules, dep, types.ModuleType(dep))
    stub.calls = calls
    return stub


# ---------------------------------------------------------------------------
# delegation
# ---------------------------------------------------------------------------

def test_core_studio_reaches_studio_verbatim(studio_stub):
    assert cli.main(["studio", "--help"]) == 0
    assert cli.main(["studio", "--port", "0", "--no-browser", "--workspace", "/w s"]) == 0
    studio_stub.rc = 3
    assert cli.main(["studio"]) == 3                       # Studio's exit code propagates
    assert studio_stub.calls == [["--help"], ["--port", "0", "--no-browser", "--workspace", "/w s"], []]


@pytest.mark.parametrize("argv, expected", [
    (["studio", "--help"], ["--help"]),
    (["core", "studio", "--help"], ["--help"]),
    (["core", "studio", "--dump-openapi", "o.json"], ["--dump-openapi", "o.json"]),
    (["-V", "debug", "studio", "--reindex"], ["--reindex"]),
    (["--verbosity=summary", "core", "studio", "-h"], ["-h"]),
])
def test_sparc_studio_reaches_studio_verbatim(studio_stub, argv, expected):
    with pytest.raises(SystemExit) as ex:
        sparc_main.main(argv)
    assert ex.value.code == 0 and studio_stub.calls == [expected]
    studio_stub.rc = 4
    with pytest.raises(SystemExit) as ex:
        sparc_main.main(argv)
    assert ex.value.code == 4


def test_studio_subparsers_are_listed_in_help(capsys, monkeypatch):
    monkeypatch.setenv("COLUMNS", "240")             # argparse wraps help text at the terminal width
    with pytest.raises(SystemExit):
        sparc_main.main(["--help"])
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    assert out.count("SPARC Studio") == 2 and HINT in out


def test_hint_when_sparc_studio_is_missing(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "sparc.studio", None)
    monkeypatch.setitem(sys.modules, "sparc.studio.cli", None)
    assert cli.main(["studio", "--help"]) == 2
    assert HINT in capsys.readouterr().err
    for argv in (["studio", "--help"], ["core", "studio", "--help"]):
        with pytest.raises(SystemExit) as ex:
            sparc_main.main(argv)
        assert ex.value.code == 2 and HINT in capsys.readouterr().err


@pytest.mark.parametrize("dep", ["fastapi", "uvicorn"])
def test_hint_when_only_a_server_dependency_is_missing(studio_stub, monkeypatch, capsys, dep):
    monkeypatch.setitem(sys.modules, dep, None)
    assert cli.main(["studio", "--help"]) == 2
    err = capsys.readouterr().err
    assert HINT in err and dep in err
    with pytest.raises(SystemExit) as ex:
        sparc_main.main(["studio", "--help"])
    assert ex.value.code == 2 and HINT in capsys.readouterr().err
    assert studio_stub.calls == []


_BLOCKED = """
import runpy, sys, types
mode, module = sys.argv[1], sys.argv[2]
if mode == "no-studio":
    sys.modules["sparc.studio"] = None
else:                                        # Studio present, FastAPI missing (the core CI)
    pkg = types.ModuleType("sparc.studio"); pkg.__path__ = []
    mod = types.ModuleType("sparc.studio.cli"); mod.main = lambda argv: print("STUDIO", argv) or 0
    sys.modules.update({"sparc.studio": pkg, "sparc.studio.cli": mod, "fastapi": None})
sys.argv = [module] + sys.argv[3:]
runpy.run_module(module, run_name="__main__", alter_sys=True)
"""


@pytest.mark.parametrize("mode", ["no-studio", "no-fastapi"])
@pytest.mark.parametrize("entry", [["sparc.core", "studio"], ["sparc", "studio"], ["sparc", "core", "studio"]])
def test_python_m_studio_help_exits_2_with_hint(mode, entry):
    """``python -m sparc.core studio --help`` (and ``sparc [core] studio``) as real processes."""
    out = _run(["-c", _BLOCKED, mode, *entry, "--help"])
    assert out.returncode == 2, (out.stdout, out.stderr)
    assert HINT in out.stderr and "STUDIO" not in out.stdout


# ---------------------------------------------------------------------------
# --progress / --job-id, exit codes
# ---------------------------------------------------------------------------

def test_run_help_lists_progress_flags():
    out = _run(["-m", "sparc.core", "run", "--help"])
    assert out.returncode == 0 and "--progress PATH" in out.stdout and "--job-id" in out.stdout


def test_every_core_subcommand_takes_progress_flags():
    parser = argparse.ArgumentParser()
    cli.add_core_subparsers(parser)
    subs = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    names = [n for n in subs.choices if n != "studio"]
    assert len(names) >= 17
    for name in names:
        opts = {o for a in subs.choices[name]._actions for o in a.option_strings}
        assert {"--progress", "--job-id"} <= opts, name
        func = subs.choices[name].get_default("func")
        assert getattr(func, "__wrapped__", None) is not None, f"{name}: cmd not wrapped by tracked_command"
    args = parser.parse_args(["synth", "--out", "x.csv", "--progress", "ev.jsonl", "--job-id", "j_1"])
    assert (args.progress, args.job_id) == ("ev.jsonl", "j_1")


def test_exit_codes_propagate_through_sparc_core(monkeypatch):
    @cli.tracked_command
    def fake_synth(args):
        return 7

    monkeypatch.setattr(cli, "cmd_core_synth", fake_synth)
    with pytest.raises(SystemExit) as ex:
        sparc_main.main(["core", "synth", "--out", "x.csv"])
    assert ex.value.code == 7
    assert cli.main(["synth", "--out", "x.csv"]) == 7


def test_reproduce_bad_dir_fails_in_the_shell(tmp_path):
    for entry in (["-m", "sparc", "core"], ["-m", "sparc.core"]):
        out = _run([*entry, "reproduce", str(tmp_path / "missing")])
        assert out.returncode != 0 and "not a run directory" in out.stderr


def test_tracked_command_applies_flags_and_returns_130(tmp_path):
    events = tmp_path / "events.jsonl"
    seen = {}

    @cli.tracked_command
    def cmd_core_probe(args):
        seen["env"] = os.environ.get(progress.ENV_SINK)
        progress.emit("probe")
        with progress.task("work"):
            progress.request_cancel()
            progress.check_cancel()
        return 0

    rc = cmd_core_probe(argparse.Namespace(progress=str(events), job_id="j_cli"))
    assert rc == cli.EXIT_CANCELLED == 130
    assert seen["env"] == str(events) and progress.sink_path() == str(events)
    ev = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
    assert [e["type"] for e in ev] == ["probe", "task.start", "cancel.ack", "task.end"]
    assert all(e["job"] == "j_cli" for e in ev) and ev[-1]["status"] == "cancelled"


# ---------------------------------------------------------------------------
# signals through a cmd_* wrapper, in a real process
# ---------------------------------------------------------------------------

_SIGNAL_CMD = """
import argparse, sys, time
from sparc.core import progress
from sparc.core.cli import tracked_command

@tracked_command
def cmd_core_loop(args):
    with progress.task("loop"):
        print("ready", flush=True)
        while True:
            progress.check_cancel()
            time.sleep(0.005)

@tracked_command
def cmd_core_sleep(args):
    print("ready", flush=True)
    time.sleep(120)
    return 0

cmd = {"loop": cmd_core_loop, "sleep": cmd_core_sleep}[sys.argv[1]]
sys.exit(cmd(argparse.Namespace(progress=sys.argv[2], job_id="j_sig")))
"""


def _start(kind, events):
    p = subprocess.Popen([sys.executable, "-c", _SIGNAL_CMD, kind, str(events)], cwd=ROOT, env=_env(),
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "ready", p.stderr.read()
    return p


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_sigterm_stops_a_check_cancel_loop_with_130(tmp_path):
    events = tmp_path / "events.jsonl"
    p = _start("loop", events)
    try:
        p.send_signal(signal.SIGTERM)
        assert p.wait(timeout=30) == 130
    finally:
        p.kill() if p.poll() is None else None
    assert "cancelled" in p.stderr.read()
    ev = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
    ack = [e for e in ev if e["type"] == "cancel.ack"]
    assert len(ack) == 1 and ack[0]["at_path"] == ["task:loop"] and ack[0]["job"] == "j_sig"
    assert [e["status"] for e in ev if e["type"] == "task.end"] == ["cancelled"]


_SIGNAL_REAL_CLI = """
import runpy, sys, time
from sparc.core import progress
import sparc.core.synthetic as synthetic

def make_synthetic_city(seed=0):          # the library call loops on a safe point instead of finishing
    with progress.task("synth"):
        print("ready", flush=True)
        while True:
            progress.check_cancel()
            time.sleep(0.005)

synthetic.make_synthetic_city = make_synthetic_city
sys.argv = ["sparc.core"] + sys.argv[1:]
runpy.run_module("sparc.core", run_name="__main__", alter_sys=True)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_sigterm_to_python_m_sparc_core_exits_130(tmp_path):
    """The real parser and ``cmd_core_synth``: --progress/--job-id applied, SIGTERM → cancel.ack → exit 130."""
    events = tmp_path / "events.jsonl"
    p = subprocess.Popen([sys.executable, "-c", _SIGNAL_REAL_CLI, "synth", "--out", str(tmp_path / "c.csv"),
                          "--progress", str(events), "--job-id", "j_real"], cwd=ROOT, env=_env(),
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == "ready", p.stderr.read()
        p.send_signal(signal.SIGTERM)
        assert p.wait(timeout=30) == 130
    finally:
        p.kill() if p.poll() is None else None
    ev = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
    assert [e["type"] for e in ev if e["type"] != "heartbeat"] == ["task.start", "cancel.ack", "task.end"]
    assert {e["job"] for e in ev} == {"j_real"} and ev[-1]["status"] == "cancelled"
    assert not (tmp_path / "c.csv").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_second_sigterm_interrupts_a_sleeping_process(tmp_path):
    p = _start("sleep", tmp_path / "events.jsonl")
    try:
        p.send_signal(signal.SIGTERM)
        time.sleep(0.5)
        assert p.poll() is None                     # the first signal only asks: no safe point in a sleep
        t0 = time.monotonic()
        p.send_signal(signal.SIGTERM)
        assert p.wait(timeout=30) == 130
        assert time.monotonic() - t0 < 5
    finally:
        p.kill() if p.poll() is None else None
