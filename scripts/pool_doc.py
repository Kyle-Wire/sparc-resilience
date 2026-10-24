"""Refresh the generated tables of docs/results/canopy_pooled.md from the batch driver's status and the pool.

    python scripts/pool_doc.py

Reads ``output/core/pooled/run_status.csv`` and ``city_status.csv`` (``scripts/run_canopy_cities.py``) and
re-pools both readings (``sparc.core.identify.pool``). It rewrites only the blocks between
``<!-- generated:NAME:start -->`` and ``<!-- generated:NAME:end -->``; the prose around them is left alone.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import pandas as pd

from sparc.core.identify.pool import pool_runs

WINDOWS = {"morning": "morning 06–07", "midday": "afternoon 15–16", "evening": "evening 19–20", "night": "night 00–01"}


def _f(v, nd=3, sign=True):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{v:+.{nd}f}".replace("-", "−") if sign else f"{v:.{nd}f}"


def _name(slug: str) -> str:
    return slug.rsplit("_", 1)[0].replace("_", " ").title()


def status_tables(cities: pd.DataFrame, runs: pd.DataFrame) -> str:
    lines = ["| City | Status | Idealised lab, street_100 | All runs, +10 pp within 100 m (95% CI) | Runs pooled |",
             "|---|---|---|---|---|"]
    for c in cities.to_dict("records"):
        ci = (f"{_f(c['all_runs_street_100'])} °F ({_f(c['all_runs_lo'])} to {_f(c['all_runs_hi'])})"
              if c.get("all_runs_street_100") == c.get("all_runs_street_100") and c.get("all_runs_street_100") is not None
              else "—")
        lab = str(c.get("lab_street_100") or "—")
        lines.append(f"| {_name(c['city'])} | {c['status']} | {lab.capitalize() if lab != 'nan' else '—'} | {ci} | "
                     f"{int(c['runs_pooled'])} of {int(c['runs'])} |")
    lines += ["", "| City | Run | Cell visits | Within 100 m (SE) | Real-route verdict, street_100 (36 reps) | Pooled | "
              "street_300 verdict | Pooled (300 m) |", "|---|---|---|---|---|---|---|---|"]
    for r in runs.to_dict("records"):
        lines.append(f"| {_name(r['city'])} | {WINDOWS.get(r['window'], r['window'])} | {int(r['visits']):,} | "
                     f"{_f(r['street_100'])} ({_f(r['street_100_se'], sign=False)}) | {str(r['route_verdict']).capitalize()} | "
                     f"{'yes' if r['pooled'] in (True, 'True') else 'no'} | {str(r.get('route_verdict_300', '—')).capitalize()} | "
                     f"{'yes' if r.get('pooled_300') in (True, 'True') else 'no'} |")
    return "\n".join(lines)


def pool_tables(runs: pd.DataFrame) -> str:
    rows = runs.astype(object).where(runs.notna(), None).to_dict("records")
    out = []
    for est, reach in (("street_100", "100 m"), ("street_300", "300 m")):
        res = pool_runs(rows, est)
        out += [f"**{est}** (+10 pp within {reach}):", "",
                "| Window | Cities pooled | Pooled effect (95% CI) | τ | Prediction interval, new city |",
                "|---|---|---|---|---|"]
        sens = []
        for win in ("morning", "midday", "evening", "night"):
            e = res["windows"].get(win) or {}
            f = e.get("fit")
            if not f:
                out.append(f"| {WINDOWS[win]} | 0 | — | — | — |")
                continue
            names = ", ".join(_name(c) for c in e["cities"])
            pi = f"{_f(f['pi_lo'])} to {_f(f['pi_hi'])}" if "pi_lo" in f else "needs 3 cities"
            out.append(f"| {WINDOWS[win]} | {f['k']} ({names}) | **{_f(f['mu'])} °F** ({_f(f['lo'])} to {_f(f['hi'])}) | "
                       f"{_f(f['tau'], sign=False)} | {pi} |")
            for r in e.get("leave_one_out") or []:
                sens.append(f"| {WINDOWS[win]} | {_name(r['left_out'])} | {r['k']} | {_f(r['mu'])} ({_f(r['lo'])} to {_f(r['hi'])}) |")
        if sens:
            out += ["", f"Each city left out in turn ({est}):", "", "| Window | Left out | Cities | Pooled effect (95% CI) |",
                    "|---|---|---|---|"] + sens
        out.append("")
    return "\n".join(out).rstrip()


def replace_block(text: str, name: str, body: str) -> str:
    pat = re.compile(rf"(<!-- generated:{name}:start -->\n).*?(<!-- generated:{name}:end -->)", re.S)
    if not pat.search(text):
        raise SystemExit(f"no generated:{name} block in the document")
    return pat.sub(lambda m: m.group(1) + body + "\n" + m.group(2), text)


def main() -> int:
    pooled = REPO / "output" / "core" / "pooled"
    runs = pd.read_csv(pooled / "run_status.csv")
    cities = pd.read_csv(pooled / "city_status.csv")
    doc = REPO / "docs" / "results" / "canopy_pooled.md"
    text = doc.read_text(encoding="utf-8")
    text = replace_block(text, "status", status_tables(cities, runs))
    text = replace_block(text, "pool", pool_tables(runs))
    doc.write_text(text, encoding="utf-8")
    print(f"refreshed {doc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
