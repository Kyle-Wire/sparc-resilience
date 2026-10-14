"""Smoke-test a results page built by :mod:`sparc.core.results_page` in headless Chromium (maintainers).

    python scripts/results_page/check_page.py <page.html> <run dir> <config> [--build] [--chromium PATH]

* ``--build`` (or a page path that does not exist yet) first builds the page with
  :func:`sparc.core.results_page.build_results_page`;
* checks the HTML shell (doctype, charset, viewport, ``lang``);
* opens every map theme and layer (each CV fold of the folds layer) and reports JavaScript errors;
* at a 390 px phone width: no sideways scroll, and every chart drawn at its own box's width;
* checks the in-browser emulator of the Design theme against the Python reference
  (:func:`sparc.core.emulator.emulate`) for a patch of each lever.

Needs ``pip install playwright`` and a Chromium binary.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def shell_errors(html: str) -> list[str]:
    """What the page's HTML shell lacks (SPEC §6.9: doctype, charset, viewport, lang)."""
    head = html[:2000].lower()
    errs = []
    if not head.lstrip().startswith("<!doctype html>"):
        errs.append("shell: the page does not start with <!doctype html>")
    if not re.search(r"<html[^>]*\blang=\"[a-z-]+\"", head):
        errs.append("shell: no <html lang=…>")
    if '<meta charset="utf-8">' not in head:
        errs.append("shell: no <meta charset=\"utf-8\">")
    if 'name="viewport"' not in head:
        errs.append("shell: no viewport meta")
    return errs


def main(argv=None) -> int:
    from playwright.sync_api import sync_playwright

    from sparc.core.baselines import load_run
    from sparc.core.config import load_core_config
    from sparc.core.emulator import emulate
    from sparc.core.results_page import build_results_page

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("page")
    ap.add_argument("run_dir")
    ap.add_argument("config")
    ap.add_argument("--build", action="store_true", help="build the page first (sparc.core.results_page)")
    ap.add_argument("--chromium", default=None)
    a = ap.parse_args(argv)
    page = Path(a.page)
    run = Path(a.run_dir)
    cfg = load_core_config(a.config)
    if a.build or not page.exists():
        build_results_page(run, copy.deepcopy(cfg), page)
        print(f"built {page} ({page.stat().st_size / 1e6:.1f} MB)")
    errs = shell_errors(page.read_text(encoding="utf-8"))
    n_layers = 0
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=a.chromium) if a.chromium else p.chromium.launch()
        pg = b.new_page(viewport={"width": 1300, "height": 1000})
        pg.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))
        pg.on("console", lambda m: errs.append(f"console: {m.text}") if m.type == "error" and "ERR_" not in m.text else None)
        pg.goto(page.resolve().as_uri())
        pg.wait_for_timeout(1000)
        for g in pg.eval_on_selector_all("#groups button", "bs => bs.map(b => b.dataset.g)"):
            pg.click(f"#groups button[data-g='{g}']")
            for o in pg.eval_on_selector_all("#layerSel option", "os => os.map(o => o.value)"):
                pg.select_option("#layerSel", o)
                n_layers += 1
                pg.wait_for_timeout(60)
                if o == "__folds":
                    for f in pg.eval_on_selector_all("#foldSel option", "os => os.map(o => o.value)"):
                        pg.select_option("#foldSel", f)
                        pg.wait_for_timeout(40)
        if n_layers == 0:
            errs.append("the map explorer lists no layers")
        # a 390 px phone: no sideways scroll, and every chart drawn at its box's width (1 SVG unit = 1 CSS px)
        ph = b.new_page(viewport={"width": 390, "height": 844})
        ph.on("pageerror", lambda e: errs.append(f"pageerror (390 px): {e}"))
        ph.goto(page.resolve().as_uri())
        ph.wait_for_timeout(600)
        ph.evaluate("document.getElementById('detailsGroup').open = true")
        ph.wait_for_timeout(300)
        phone = ph.evaluate("""() => { const de = document.documentElement; const off = [];
            for (const s of document.querySelectorAll('.chart svg[role="img"]')) { const w = s.getBoundingClientRect().width;
              if (w && Math.abs(w - s.viewBox.baseVal.width) > 1.5) off.push(`${(s.closest('[id]') || {}).id}: ${Math.round(w)} px for ${s.viewBox.baseVal.width}`); }
            return { sw: de.scrollWidth, cw: de.clientWidth, off }; }""")
        if phone["sw"] > phone["cw"]:
            errs.append(f"390 px: the page scrolls sideways ({phone['sw']} px wide)")
        errs.extend(f"390 px: chart not drawn at its width ({o})" for o in phone["off"])
        print(f"390 px: page {phone['sw']} px wide, {len(phone['off'])} charts off their width")
        ph.close()
        if (run / "emulator.npz").exists():
            data, *_ = load_run(run, copy.deepcopy(cfg))
            npz = np.load(run / "emulator.npz")
            meta = json.loads((run / "emulator.json").read_text())
            g = data.grid
            c = int(np.argmin(np.hypot(g.ix - np.median(g.ix), g.iy - np.median(g.iy))))
            sel = np.hypot(g.ix - g.ix[c], g.iy - g.iy[c]) <= 4
            for var, d in meta["levers"].items():
                em = {"own": npz[f"{var}__own"],
                      "channels": [{"sigma_cells": ch["sigma_cells"], "coef": npz[ch["coef"]], "weight": npz[ch["weight"]]}
                                   for ch in d["channels"]],
                      "physics": {"dq": npz[f"{var}__dq"], "kernel": npz["physics_kernel"]} if d["physics"] else None}
                dx = np.where(sel, float(d.get("design_dose", 1.0)), 0.0)
                py = emulate(em, g, dx)
                js = np.asarray(pg.evaluate("""([v, dx]) => { const D = window.SPARC_DESIGN;
                    for (const k of Object.keys(D.plan)) delete D.plan[k]; D.plan[v] = Float32Array.from(dx); D.dirty();
                    return Array.from(D.delta()); }""", [var, dx.tolist()]))
                err = float(np.max(np.abs(js - py)))
                print(f"{var}: patch mean python {py[sel].mean():+.4f}, browser {js[sel].mean():+.4f}; "
                      f"max cell difference {err:.2e} (max |ΔT| {np.max(np.abs(py)):.3f})")
                if err > 0.01 * max(np.max(np.abs(py)), 1e-6) + 1e-4:
                    errs.append(f"emulator mismatch for {var}: {err}")
        b.close()
    print(f"{n_layers} map layers opened")
    print("\n".join(errs) if errs else "page OK: no JavaScript errors")
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
