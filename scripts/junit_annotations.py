"""Turn pytest JUnit XML failures into GitHub Actions ``::error`` annotations.

Usage: ``python scripts/junit_annotations.py junit.xml [--title PREFIX] [--max N]``.
Prints one annotation per failed or errored test (message, then the head of the
traceback) and exits 1 when there was any failure, 0 otherwise, so CI can show
failures from the check run without downloading logs.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET


def _escape(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "").replace("\n", "%0A")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("xml")
    p.add_argument("--title", default="pytest")
    p.add_argument("--max", type=int, default=40)
    a = p.parse_args(argv)
    try:
        root = ET.parse(a.xml).getroot()
    except (OSError, ET.ParseError) as exc:
        print(f"::error title={a.title}::no test report ({exc})")
        return 1
    failed = []
    for case in root.iter("testcase"):
        for tag in ("failure", "error"):
            el = case.find(tag)
            if el is not None:
                name = f"{case.get('classname', '')}::{case.get('name', '')}"
                body = (el.get("message") or "") + "\n" + (el.text or "")
                failed.append((name, tag, body))
    totals = {k: sum(int(s.get(k, 0)) for s in root.iter("testsuite")) for k in ("tests", "failures", "errors",
                                                                                 "skipped")}
    print(f"{a.title}: {totals}")
    for name, tag, body in failed[: a.max]:
        lines = body.strip().splitlines()
        text = "\n".join(lines[:12] + (["…"] if len(lines) > 30 else []) + lines[-18:] if len(lines) > 30 else lines)
        print(f"::error title={a.title} {tag}: {name[:180]}::{_escape(text)[:8000]}")
    if len(failed) > a.max:
        print(f"::error title={a.title}::{len(failed) - a.max} more failures not shown")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
