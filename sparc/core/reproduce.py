"""Re-run a finished run from its manifest and check the numbers agree.

``sparc core reproduce <run dir>`` rebuilds the configuration stored in the
run's ``manifest.json`` (paths resolved against the original config
directory), checks the input data hash, re-runs the requested stages into
``<run dir>_reproduce`` and compares:

* out-of-fold skill (stacker and every base model): |ΔR²| ≤ ``tol_r2``;
* scenario mean effects (when S5 is re-run): relative difference ≤
  ``tol_effect`` (or ≤ 0.01 target units for near-zero effects);
* CV design (block, buffer, fold sizes): identical.

Differences in the code fingerprint or environment are reported, not
failed: they explain a mismatch rather than cause one by themselves.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def _load(run_dir: Path) -> dict:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def compare_manifests(old: dict, new: dict, tol_r2: float = 0.01, tol_effect: float = 0.05) -> dict:
    checks = []

    def add(name, ok, detail, hard=True):
        checks.append({"check": name, "ok": bool(ok), "detail": detail, "hard": hard})

    oc, nc = old.get("cv") or {}, new.get("cv") or {}
    add("cv design", oc.get("block_m") == nc.get("block_m") and oc.get("test_sizes") == nc.get("test_sizes"),
        f"block {oc.get('block_m')} → {nc.get('block_m')}, test sizes {oc.get('test_sizes')} → {nc.get('test_sizes')}")
    om, nm = old.get("metrics") or {}, new.get("metrics") or {}
    for k in om:
        if k in nm and "r2" in om[k] and "r2" in nm[k]:
            d = float(nm[k]["r2"]) - float(om[k]["r2"])
            add(f"R² {k}", abs(d) <= tol_r2, f"{om[k]['r2']:.4f} → {nm[k]['r2']:.4f} (Δ {d:+.4f})")
    osc = {s["name"]: s for s in old.get("scenarios") or []}
    for s in new.get("scenarios") or []:
        o = osc.get(s["name"])
        if o is None:
            continue
        a, b = float(o["mean_delta"]), float(s["mean_delta"])
        rel = abs(b - a) / max(abs(a), 1e-12)
        add(f"scenario {s['name']}", rel <= tol_effect or abs(b - a) <= 0.01, f"{a:+.4f} → {b:+.4f} ({rel:.1%})")
    op, np_ = old.get("provenance") or {}, new.get("provenance") or {}
    add("input data", op.get("input_sha256") in (None, np_.get("input_sha256")),
        "same SHA-256" if op.get("input_sha256") == np_.get("input_sha256") else "input data differ")
    add("core code", op.get("code_sha256") == np_.get("code_sha256"),
        "same" if op.get("code_sha256") == np_.get("code_sha256") else
        f"code changed ({(op.get('git') or {}).get('commit', '?')[:8]} → {(np_.get('git') or {}).get('commit', '?')[:8]})",
        hard=False)
    ov, nv = old.get("versions") or {}, new.get("versions") or {}
    diff = {k: (ov.get(k), nv.get(k)) for k in set(ov) | set(nv) if ov.get(k) != nv.get(k)}
    add("package versions", not diff, "same" if not diff else ", ".join(f"{k} {a}→{b}" for k, (a, b) in diff.items()),
        hard=False)
    return {"pass": all(c["ok"] for c in checks if c["hard"]), "checks": checks}


def reproduce(run_dir: str | Path, stages=("S0", "S1", "S2", "S3"), tol_r2: float = 0.01,
              tol_effect: float = 0.05, config_dir: str | Path | None = None) -> dict:
    from sparc.core.config import core_config_from_dict
    from sparc.core.pipeline import run_core
    from sparc.core.provenance import sha256_file

    run_dir = Path(run_dir)
    old = _load(run_dir)
    prov = old.get("provenance") or {}
    base = config_dir or prov.get("config_dir")
    if base is None:
        raise ValueError("the manifest has no config_dir (older run): pass config_dir / --config-dir")
    cfg = core_config_from_dict(old["config"], base_dir=base)
    if prov.get("input_kind") == "frame":
        raise ValueError("this run was made from an in-memory table; reproduce needs a file-based run")
    sha = sha256_file(cfg.data_path)
    if prov.get("input_sha256") and sha != prov["input_sha256"]:
        log.warning("input data differ from the original run (%s)", cfg.data_path)
    cfg.raw["name"] = f"{run_dir.name}_reproduce"
    cfg.raw["output"]["dir"] = str(run_dir.resolve().parent)
    if "S3" in stages and not {"S4", "S5", "S6", "S7"} & set(stages):
        cfg.raw["cv"].setdefault("distance_curve", {})["enabled"] = False
    res = run_core(cfg, stages=stages, write=True)
    out = compare_manifests(old, res.manifest, tol_r2=tol_r2, tol_effect=tol_effect)
    out["original"] = str(run_dir)
    out["reproduction"] = str(res.run_dir)
    (res.run_dir / "reproduce.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out
