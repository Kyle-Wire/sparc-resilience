#!/usr/bin/env python
"""Run the shell examples of the SPARC Studio documentation (SPEC §14.6 step 8).

The guides (``docs/studio/USER_GUIDE.md``, ``DEVELOPING.md``, ``RELEASE_NOTES.md`` and the
README's SPARC Studio section) show commands in fenced ``bash`` blocks. Two checks run on them:

1. **Runnable blocks are executed.** A block whose first line is ``# runnable`` runs as written,
   with ``bash -euo pipefail``, from the repository root, against a fresh temporary workspace:

   * ``SPARC_STUDIO_HOME`` points at ``<tmp>/ws`` (the default workspace of ``sparc studio``),
     and ``TMPDIR`` at ``<tmp>/tmp``;
   * this checkout comes first on ``PYTHONPATH``, so ``sparc`` and ``python -m sparc.studio``
     run the code next to the documents even when another copy is installed;
   * each block is its own process group. When it ends (or times out) the group is stopped,
     and so is any process that still names the temporary folder (a server started with ``&``,
     its job workers, an engine host).

   ``# runnable (needs: node)`` marks a block that needs ``studio-web/node_modules`` (after
   ``npm --prefix studio-web ci``). Blocks that call ``curl``, ``git`` or ``npm`` need those
   programs. A block whose needs are not met is skipped, or fails under ``--strict`` and in CI
   (``CI`` set), so the CI step cannot pass without running everything.
2. **Every other shell command is checked statically**, so a document cannot drift silently:
   ``sparc studio …`` / ``python -m sparc.studio …`` flags must parse with the real argument
   parser, ``python <script>`` and ``python -m pytest <paths>`` must name files that exist,
   ``python -m sparc…`` must name an importable module, and ``npm --prefix studio-web run <x>``
   must name a script of ``studio-web/package.json``.

Usage::

    python docs/studio/doctest.py [--list] [--lint-only] [--only TEXT] [--strict] [--keep] [-v]
                                  [--timeout SECONDS] [FILE ...]

Exit status 0 when every check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "studio"
DEFAULT_FILES = [DOCS / "USER_GUIDE.md", DOCS / "DEVELOPING.md", DOCS / "RELEASE_NOTES.md", ROOT / "README.md"]
SHELL_LANGS = {"bash", "sh", "shell"}
RUNNABLE = re.compile(r"^#\s*runnable\b\s*(?:\((?P<opts>[^)]*)\))?\s*$")
FENCE = re.compile(r"^(?P<indent>\s*)(?P<fence>```+)\s*(?P<info>[\w+-]*)")
HEREDOC = re.compile(r"<<-?\s*(['\"]?)(?P<tag>[A-Za-z_][A-Za-z0-9_]*)\1")
OPERATORS = {"&&", "||", ";", "|", "&", "(", ")", ";;"}
TOOLS = ("curl", "git", "npm")


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------

@dataclass
class Block:
    file: Path
    line: int                         # line of the opening fence (1-based)
    lang: str
    text: str
    runnable: bool = False
    needs: list[str] = field(default_factory=list)

    @property
    def where(self) -> str:
        try:
            rel = self.file.relative_to(ROOT)
        except ValueError:
            rel = self.file
        return f"{rel}:{self.line}"

    @property
    def title(self) -> str:
        for s in self.text.splitlines():
            s = s.strip()
            if s and not s.startswith("#"):
                return s[:70]
        return "(empty)"


def extract(path: Path) -> list[Block]:
    """The fenced shell blocks of one Markdown file (indented fences inside lists included)."""
    blocks: list[Block] = []
    lines = path.read_text("utf-8").splitlines()
    i = 0
    while i < len(lines):
        m = FENCE.match(lines[i])
        if not m:
            i += 1
            continue
        indent, fence, lang = m.group("indent"), m.group("fence"), m.group("info").lower()
        start, body = i + 1, []
        i += 1
        while i < len(lines) and not lines[i].strip().startswith(fence):
            line = lines[i]
            body.append(line[len(indent):] if line.startswith(indent) else line.lstrip())
            i += 1
        i += 1
        if lang not in SHELL_LANGS:
            continue
        block = Block(path, start, lang, "\n".join(body) + "\n")
        first = next((s.strip() for s in body if s.strip()), "")
        rm = RUNNABLE.match(first)
        if rm:
            block.runnable = True
            opts = rm.group("opts") or ""
            nm = re.search(r"needs\s*:\s*([\w ,-]+)", opts)
            if nm:
                block.needs = [n.strip() for n in nm.group(1).split(",") if n.strip()]
        blocks.append(block)
    return blocks


# ---------------------------------------------------------------------------
# static checks of every shell command
# ---------------------------------------------------------------------------

def logical_lines(text: str) -> list[tuple[int, str]]:
    """``(line offset, command)`` of a block: comments dropped, ``\\`` continuations joined, here-document
    bodies skipped."""
    out: list[tuple[int, str]] = []
    pending, first = "", 0
    heredoc: str | None = None
    for no, raw in enumerate(text.splitlines()):
        if heredoc is not None:
            if raw.strip() == heredoc:
                heredoc = None
            continue
        line = raw.rstrip()
        if not pending and line.lstrip().startswith("#"):
            continue
        if not pending:
            first = no
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        line = pending + line
        pending = ""
        if line.strip():
            out.append((first, line.strip()))
            hm = HEREDOC.search(line)
            if hm:
                heredoc = hm.group("tag")
    if pending.strip():
        out.append((first, pending.strip()))
    return out


def simple_commands(line: str) -> list[list[str]]:
    """``a && b | c &`` → ``[[a…], [b…], [c…]]`` (env assignments and ``$(…)`` contents not expanded)."""
    lex = shlex.shlex(line, posix=True, punctuation_chars=";&|()")
    lex.whitespace_split = True
    lex.commenters = "#"
    try:
        tokens = list(lex)
    except ValueError:                      # unbalanced quotes in an exotic line: nothing to check
        return []
    cmds, cur = [], []
    for tok in tokens:
        if tok in OPERATORS:
            if cur:
                cmds.append(cur)
            cur = []
        else:
            cur.append(tok)
    if cur:
        cmds.append(cur)
    out = []
    for cmd in cmds:
        while cmd and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", cmd[0]):
            cmd = cmd[1:]                   # VAR=value prefixes
        while cmd and cmd[0] in ("exec", "time", "timeout", "env", "nohup"):
            cmd = cmd[1:]
            while cmd and (cmd[0].startswith("-") or re.match(r"^\d+[smh]?$", cmd[0])
                           or re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", cmd[0])):
                cmd = cmd[1:]
        cmd = [t for t in cmd if not re.match(r"^\d?[<>]", t)]      # redirections
        if cmd:
            out.append(cmd)
    return out


def _studio_parser():
    sys.path.insert(0, str(ROOT))
    from sparc.studio.cli import build_parser

    return build_parser()


def _parse_studio(args: list[str]) -> str | None:
    """``None`` when ``sparc studio <args>`` parses; else the parser's complaint."""
    argv = ["0" if "$" in a else a for a in args]
    err = io.StringIO()
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            _studio_parser().parse_args(argv)
    except SystemExit as exc:
        if exc.code not in (0, None):
            return err.getvalue().strip().splitlines()[-1] if err.getvalue().strip() else f"exit {exc.code}"
    return None


def _npm_scripts() -> set[str]:
    try:
        return set(json.loads((ROOT / "studio-web" / "package.json").read_text("utf-8")).get("scripts") or {})
    except (OSError, ValueError):
        return set()


def _path_exists(arg: str) -> bool:
    return (ROOT / arg.split("::", 1)[0]).exists()


def lint_command(cmd: list[str]) -> str | None:
    """A problem with one simple command, or ``None``."""
    head, args = cmd[0], cmd[1:]
    if head == "sparc" and args[:1] == ["studio"]:
        return _parse_studio(args[1:])
    if head == "sparc" and args[:2] == ["core", "studio"]:
        return _parse_studio(args[2:])
    if head in ("python", "python3"):
        if args[:1] == ["-m"] and len(args) > 1:
            mod, rest = args[1], args[2:]
            if mod in ("sparc.studio", "sparc.core") and (mod == "sparc.studio" or rest[:1] == ["studio"]):
                return _parse_studio(rest if mod == "sparc.studio" else rest[1:])
            if mod == "pytest":
                bad = [a for a in rest if not a.startswith("-") and ("/" in a or a.endswith(".py"))
                       and "$" not in a and not _path_exists(a)]
                return f"pytest path not found: {', '.join(bad)}" if bad else None
            if mod.startswith("sparc"):
                with contextlib.suppress(ModuleNotFoundError, ValueError):
                    if importlib.util.find_spec(mod) is not None:
                        return None
                return f"no module {mod}"
            return None
        if args and not args[0].startswith("-") and args[0].endswith(".py") and "$" not in args[0]:
            return None if _path_exists(args[0]) else f"script not found: {args[0]}"
        return None
    if head == "npm" and "run" in args:
        i = args.index("run")
        if "--prefix" in args and args[args.index("--prefix") + 1:args.index("--prefix") + 2] == ["studio-web"]:
            script = args[i + 1] if i + 1 < len(args) else ""
            return None if script in _npm_scripts() else f"no npm script {script!r} in studio-web/package.json"
    return None


def lint_block(block: Block) -> list[tuple[int, str]]:
    """``(file line, problem)`` for every command of ``block`` that fails its static check."""
    problems = []
    for offset, line in logical_lines(block.text):
        for cmd in simple_commands(line):
            msg = lint_command(cmd)
            if msg:
                problems.append((block.line + 1 + offset, f"{' '.join(cmd)[:90]}: {msg}"))
    return problems


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------

def missing_needs(block: Block) -> list[str]:
    missing = []
    for need in block.needs:
        if need == "node":
            if not shutil.which("npm") or not (ROOT / "studio-web" / "node_modules").is_dir():
                missing.append("node (npm and studio-web/node_modules)")
        elif not shutil.which(need):
            missing.append(need)
    for tool in TOOLS:
        if re.search(rf"(^|[\s;&|(]){tool}\s", block.text, re.M) and not shutil.which(tool):
            missing.append(tool)
    return sorted(set(missing))


def _stop_group(pgid: int) -> None:
    for sig, wait in ((signal.SIGTERM, 5.0), (signal.SIGKILL, 2.0)):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            return
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            try:
                os.killpg(pgid, 0)
            except (ProcessLookupError, PermissionError):
                return
            time.sleep(0.1)


def _stop_stragglers(marker: str) -> None:
    """Processes that escaped the group (own sessions: job workers, the engine host) but name ``marker``."""
    try:
        import psutil
    except ImportError:
        return
    me = os.getpid()
    victims = []
    for p in psutil.process_iter(["pid", "cmdline", "environ"]):
        if p.info["pid"] == me:
            continue
        try:
            cmd = " ".join(p.info.get("cmdline") or [])
            env = p.info.get("environ") or {}
        except Exception:
            continue
        if marker in cmd or marker in str(env.get("SPARC_STUDIO_HOME", "")):
            victims.append(p)
    for p in victims:
        with contextlib.suppress(Exception):
            p.terminate()
    _gone, alive = psutil.wait_procs(victims, timeout=5)
    for p in alive:
        with contextlib.suppress(Exception):
            p.kill()


def block_env(tmp: Path) -> dict[str, str]:
    env = dict(os.environ)
    for key in ("SPARC_STUDIO_RUNNER", "SPARC_STUDIO_REPLAY_SPEED", "SPARC_STUDIO_TEST_KINDS", "SPARC_PROGRESS",
                "SPARC_JOB_ID", "SPARC_CANCEL_FILE"):
        env.pop(key, None)
    (tmp / "tmp").mkdir(parents=True, exist_ok=True)
    env["SPARC_STUDIO_HOME"] = str(tmp / "ws")
    env["TMPDIR"] = str(tmp / "tmp")
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT)] + [p for p in [env.get("PYTHONPATH")] if p])
    env["PYTHONUNBUFFERED"] = "1"
    env["BROWSER"] = "true"                 # a forgotten --no-browser must not open anything
    for key in ("NO_PROXY", "no_proxy"):    # the examples talk to 127.0.0.1
        env[key] = ",".join([v for v in [env.get(key)] if v] + ["127.0.0.1", "localhost"])
    return env


def run_block(block: Block, timeout: float, keep: bool) -> tuple[bool, float, str, Path]:
    tmp = Path(tempfile.mkdtemp(prefix="sparc-doctest-"))
    env = block_env(tmp)
    t0 = time.monotonic()
    proc = subprocess.Popen(["bash", "-euo", "pipefail", "-c", block.text], cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=timeout)
        ok = proc.returncode == 0
        tail = "" if ok else f"exit status {proc.returncode}"
    except subprocess.TimeoutExpired:
        _stop_group(proc.pid)
        out, _ = proc.communicate()
        ok, tail = False, f"timed out after {timeout:.0f} s"
    finally:
        _stop_group(proc.pid)
        _stop_stragglers(str(tmp))
    text = out.decode("utf-8", "replace")
    if tail:
        text = f"{text}\n[{tail}]"
    if not keep:
        shutil.rmtree(tmp, ignore_errors=True)
    return ok, time.monotonic() - t0, text, tmp


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="*", type=Path, help="Markdown files (default: the Studio guides and README.md)")
    ap.add_argument("--list", action="store_true", help="list the shell blocks and exit")
    ap.add_argument("--lint-only", action="store_true", help="check the commands statically, run nothing")
    ap.add_argument("--only", metavar="TEXT", help="run only runnable blocks whose location or text contains TEXT")
    ap.add_argument("--strict", action="store_true", help="fail (instead of skip) blocks whose needs are not met; "
                                                          "the default when $CI is set")
    ap.add_argument("--keep", action="store_true", help="keep each block's temporary folder")
    ap.add_argument("-v", "--verbose", action="store_true", help="print the output of passing blocks too")
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds per runnable block (default 600)")
    args = ap.parse_args(argv)
    strict = args.strict or bool(os.environ.get("CI"))

    files = [f.resolve() for f in args.files] or DEFAULT_FILES
    blocks: list[Block] = []
    for f in files:
        if not f.is_file():
            print(f"FAIL  {f}: no such file")
            return 1
        blocks += extract(f)

    if args.list:
        for b in blocks:
            tag = "runnable" + (f" (needs: {', '.join(b.needs)})" if b.needs else "") if b.runnable else "checked"
            print(f"{b.where:<34} {tag:<26} {b.title}")
        return 0

    failures = 0
    n_lint = 0
    for b in blocks:
        n_lint += len(logical_lines(b.text))
        for line, problem in lint_block(b):
            failures += 1
            print(f"FAIL  {b.where.rsplit(':', 1)[0]}:{line}  {problem}")
    print(f"checked {n_lint} command lines in {len(blocks)} shell blocks of {len(files)} files")
    if args.lint_only:
        return 1 if failures else 0

    ran = skipped = 0
    for b in blocks:
        if not b.runnable or (args.only and args.only not in b.where and args.only not in b.text):
            continue
        need = missing_needs(b)
        if need:
            if strict:
                failures += 1
                print(f"FAIL  {b.where}  missing {', '.join(need)} ({b.title})")
            else:
                skipped += 1
                print(f"SKIP  {b.where}  missing {', '.join(need)} ({b.title})")
            continue
        ok, secs, out, tmp = run_block(b, args.timeout, args.keep)
        ran += 1
        print(f"{'PASS' if ok else 'FAIL'}  {b.where}  {secs:5.1f} s  {b.title}")
        if not ok:
            failures += 1
            print("      " + "\n      ".join(out.rstrip().splitlines()[-40:]))
        elif args.verbose:
            print("      " + "\n      ".join(out.rstrip().splitlines()))
        if args.keep:
            print(f"      kept {tmp}")
    print(f"ran {ran} runnable blocks, skipped {skipped}, {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
