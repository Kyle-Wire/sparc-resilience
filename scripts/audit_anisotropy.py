"""
audit_anisotropy.py -- B2 Anisotropy Audit (PI-JEPA Agent Spec §2)

Standalone, read-only script.  For each training city:
  - Loads ("0", "correlogram_matern_fit") + ("0", "correlogram_anisotropy")
    from the city's artifact store.
  - Emits a CSV + console table: city, variable, eccentricity, theta_deg,
    kappa_mean, r_hat_theta, ess_theta, converged.
  - Prints three verdicts: V-ISO, V-DIR-UNRELIABLE, V-DIR-GOOD.
  - Writes output/diagnostics/anisotropy_audit.csv.

If artifacts are absent (Stage 0 pipeline was not run), reports that clearly
and prints the command needed to generate them.

Usage:
    python scripts/audit_anisotropy.py
    python scripts/audit_anisotropy.py --config configs/multicity_pilot.yml
    python scripts/audit_anisotropy.py --output-dir output/diagnostics
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path

# Force UTF-8 output on Windows -- spec rule: no Unicode math in terminal output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s -- %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("audit_anisotropy")

# Variables the spec asks us to focus on (§2 B2)
FOCUS_VARIABLES = ["lst", "pct_impervious", "pct_canopy", "svf", "ndvi", "albedo"]

# Thresholds
ECCENTRICITY_THRESHOLD = 1.15   # below this = effectively isotropic
# ESS floor for theta direction to be considered trustworthy
# Stage 0 uses 400 fast-mode draws; a reasonable floor is ~50 effective samples
ESS_FLOOR = 50.0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/multicity_pilot.yml")
    p.add_argument("--output-dir", default="output/diagnostics")
    return p.parse_args()


def _load_store(city_output_dir: Path):
    """Return an ArtifactStore for the given city output directory, or None."""
    try:
        from sparc.registry.run_registry import RunRegistry
        from sparc.registry.store import ArtifactStore
        reg = RunRegistry(city_output_dir)
        return ArtifactStore(reg)
    except Exception as exc:
        log.debug("Could not open artifact store at %s: %s", city_output_dir, exc)
        return None


def _safe_read(store, stage: str, artifact_id: str):
    """Read a struct artifact; return None on any error."""
    if store is None:
        return None
    try:
        return store.read_struct(stage, artifact_id)
    except Exception:
        return None


def _extract_variable_rows(
    city_slug: str,
    matern_payload: dict | None,
    aniso_payload: dict | None,
) -> list[dict]:
    """Extract per-variable rows from Stage 0 payloads.

    matern_payload: {"variables": {"lst": {"kappa_mean": ..., "r_hat": ..., ...}, ...}}
    aniso_payload:  {"variables": {"lst": {"kappa_x": {...}, "kappa_y": {...},
                                           "theta_unit": {...}, "eccentricity": {...},
                                           "ess_theta": ..., "converged": ...}, ...}}
    """
    rows = []
    variables_with_data = set()

    # Collect from matern payload
    matern_vars: dict = {}
    if matern_payload and isinstance(matern_payload, dict):
        matern_vars = matern_payload.get("variables", {})

    # Collect from anisotropy payload
    aniso_vars: dict = {}
    if aniso_payload and isinstance(aniso_payload, dict):
        aniso_vars = aniso_payload.get("variables", {})

    all_vars = set(matern_vars) | set(aniso_vars)
    variables_with_data.update(all_vars)

    for var in sorted(all_vars):
        m = matern_vars.get(var, {})
        a = aniso_vars.get(var, {})

        # --- kappa_mean from matern payload ---
        kappa_mean = None
        if isinstance(m, dict):
            # May be nested: {"kappa": {"mean": ...}} or flat {"kappa_mean": ...}
            k = m.get("kappa", m.get("kappa_mean", None))
            if isinstance(k, dict):
                kappa_mean = k.get("mean", None)
            elif isinstance(k, (int, float)):
                kappa_mean = float(k)

        # --- r_hat_theta ---
        r_hat_theta = None
        if isinstance(m, dict):
            rh = m.get("r_hat", m.get("r_hat_theta", None))
            if isinstance(rh, dict):
                r_hat_theta = rh.get("theta", rh.get("mean", None))
            elif isinstance(rh, (int, float)):
                r_hat_theta = float(rh)

        # --- anisotropy fields ---
        ecc_mean = None
        theta_deg = None
        ess_theta = None
        converged = None

        if isinstance(a, dict):
            # eccentricity
            ecc = a.get("eccentricity", {})
            if isinstance(ecc, dict):
                ecc_mean = ecc.get("mean", None)
            elif isinstance(ecc, (int, float)):
                ecc_mean = float(ecc)

            # theta_unit -> convert to degrees
            th = a.get("theta_unit", a.get("theta", None))
            if isinstance(th, dict):
                th_mean = th.get("mean", None)
                if th_mean is not None:
                    theta_deg = math.degrees(float(th_mean)) % 180
            elif isinstance(th, (int, float)):
                theta_deg = math.degrees(float(th)) % 180

            ess_theta = a.get("ess_theta", a.get("ess", None))
            converged = a.get("converged", None)

        rows.append({
            "city":         city_slug,
            "variable":     var,
            "eccentricity": ecc_mean,
            "theta_deg":    theta_deg,
            "kappa_mean":   kappa_mean,
            "r_hat_theta":  r_hat_theta,
            "ess_theta":    ess_theta,
            "converged":    converged,
        })

    return rows


def _classify_row(row: dict) -> str:
    """Classify one (city, variable) row.

    Returns: 'V-ISO', 'V-DIR-UNRELIABLE', 'V-DIR-GOOD', or 'MISSING'
    """
    ecc = row.get("eccentricity")
    ess = row.get("ess_theta")
    conv = row.get("converged")

    if ecc is None:
        return "MISSING"

    ecc = float(ecc)
    if ecc < ECCENTRICITY_THRESHOLD:
        return "V-ISO"

    # eccentricity >= threshold -- check if direction is trustworthy
    ess_ok = (ess is not None and float(ess) >= ESS_FLOOR)
    conv_ok = bool(conv) if conv is not None else False

    if ess_ok and conv_ok:
        return "V-DIR-GOOD"
    else:
        return "V-DIR-UNRELIABLE"


def main() -> None:
    args = _parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        log.error("Config not found: %s", config_path)
        sys.exit(1)

    with open(config_path, encoding="utf-8") as fh:
        pilot_cfg = yaml.safe_load(fh)

    mc          = pilot_cfg["multicity"]
    output_root = Path(mc["output_dir"])
    diag_dir    = Path(args.output_dir)
    diag_dir.mkdir(parents=True, exist_ok=True)

    all_cities  = mc["pilot_cities"]
    train_cfgs  = [c for c in all_cities if not c.get("holdout", False)]

    log.info("Auditing anisotropy for %d training cities", len(train_cfgs))
    log.info("Focus variables: %s", ", ".join(FOCUS_VARIABLES))
    log.info("Eccentricity threshold for V-ISO: %.2f", ECCENTRICITY_THRESHOLD)
    log.info("ESS floor for V-DIR-GOOD: %.0f", ESS_FLOOR)

    all_rows:         list[dict] = []
    cities_missing:   list[str]  = []
    cities_with_data: list[str]  = []

    for city_cfg in train_cfgs:
        slug    = city_cfg["city_slug"]
        out_dir = output_root / slug / "output"

        store   = _load_store(out_dir)
        matern  = _safe_read(store, "0", "correlogram_matern_fit")
        aniso   = _safe_read(store, "0", "correlogram_anisotropy")

        if matern is None and aniso is None:
            log.warning("[%s] Stage 0 anisotropy artifacts not found -- city skipped", slug)
            cities_missing.append(slug)
            continue

        cities_with_data.append(slug)
        rows = _extract_variable_rows(slug, matern, aniso)
        if rows:
            all_rows.extend(rows)
            log.info("[%s] %d variable(s) found", slug, len(rows))
        else:
            log.warning("[%s] payloads present but no variables extracted", slug)
            cities_missing.append(slug)

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("  B2 Anisotropy Audit Results")
    print("=" * 70)

    if not all_rows:
        print()
        print("  NO ANISOTROPY DATA FOUND for any training city.")
        print()
        print("  The Stage 0 Matern fit and anisotropy artifacts have not been")
        print("  generated yet.  Run the following to produce them:")
        print()
        print("      python scripts/train_multicity_jepa.py --skip-collect --stages 0")
        print()
        print("  Then re-run this audit.")
        print()
        print("  Cities checked:")
        for slug in [c["city_slug"] for c in train_cfgs]:
            status = "MISSING" if slug in cities_missing else "DATA"
            print(f"    {slug:25s}  {status}")
        _write_empty_csv(diag_dir)
        sys.exit(0)

    # Print table
    header = f"{'city':20s} {'variable':20s} {'eccentricity':12s} {'theta_deg':9s} {'kappa_mean':10s} {'r_hat':7s} {'ess':7s} {'conv':5s} {'class':17s}"
    divider = "-" * len(header)
    print()
    print(header)
    print(divider)

    focus_rows   = [r for r in all_rows if r["variable"] in FOCUS_VARIABLES]
    nonfocus_rows = [r for r in all_rows if r["variable"] not in FOCUS_VARIABLES]

    def _fmt(v, fmt=".3f"):
        return f"{v:{fmt}}" if v is not None else "   N/A"

    for row in sorted(all_rows, key=lambda r: (r["city"], r["variable"])):
        cls = _classify_row(row)
        marker = " <-- focus" if row["variable"] in FOCUS_VARIABLES else ""
        print(
            f"{row['city']:20s} {row['variable']:20s} "
            f"{_fmt(row['eccentricity']):12s} {_fmt(row['theta_deg'], '.1f'):9s} "
            f"{_fmt(row['kappa_mean']):10s} {_fmt(row['r_hat_theta']):7s} "
            f"{_fmt(row['ess_theta'], '.0f'):7s} {str(row['converged']):5s} "
            f"{cls:17s}{marker}"
        )

    print(divider)

    # -----------------------------------------------------------------------
    # Verdicts
    # -----------------------------------------------------------------------
    if focus_rows:
        target_rows = focus_rows
        scope = f"focus variables ({', '.join(FOCUS_VARIABLES)})"
    else:
        target_rows = all_rows
        scope = "all variables (focus variables not found)"

    classified = [_classify_row(r) for r in target_rows]
    n = len(classified)

    n_iso      = classified.count("V-ISO")
    n_unrel    = classified.count("V-DIR-UNRELIABLE")
    n_good     = classified.count("V-DIR-GOOD")
    n_missing  = classified.count("MISSING")

    frac_iso   = n_iso   / n if n else 0
    frac_unrel = n_unrel / n if n else 0
    frac_good  = n_good  / n if n else 0

    print()
    print(f"  Scope: {scope}  (N={n})")
    print()
    print(f"  V-ISO            (ecc < {ECCENTRICITY_THRESHOLD}):         {frac_iso:.1%}  ({n_iso}/{n})")
    print(f"  V-DIR-UNRELIABLE (ecc >= {ECCENTRICITY_THRESHOLD}, low ESS/conv): {frac_unrel:.1%}  ({n_unrel}/{n})")
    print(f"  V-DIR-GOOD       (ecc >= {ECCENTRICITY_THRESHOLD}, ESS+conv OK):  {frac_good:.1%}  ({n_good}/{n})")
    if n_missing:
        print(f"  MISSING (no payload):              {n_missing/n:.1%}  ({n_missing}/{n})")
    print()

    # Branch recommendation per spec §2 B2
    print("  Branch recommendation:")
    if frac_iso >= 0.5:
        verdict = "V-ISO-DOMINANT"
        print("  --> V-ISO dominates. Implement I2 only (range-scaling).")
        print("      Skip I1 angle component and I3.")
        print("      eccentricity~1 means I1 degenerates to isotropic disk anyway.")
    elif frac_unrel >= 0.5:
        verdict = "V-DIR-UNRELIABLE-DOMINANT"
        print("  --> V-DIR-UNRELIABLE dominates. Implement I1 using eccentricity")
        print("      magnitude only (axis ratio). Set theta=0 or ESS-weighted prior.")
        print("      Skip theta-dependent rotation or apply with ESS-weighting.")
        print("      Skip I3 or run with near-zero weight.")
    elif frac_good >= 0.3:
        verdict = "V-DIR-GOOD"
        print("  --> V-DIR-GOOD covers key thermal variables.")
        print("      Full PI-JEPA (I1 with rotation + I3) is justified.")
        print("      Proceed with full anisotropic mask and alignment loss.")
    else:
        verdict = "INDETERMINATE"
        print("  --> Indeterminate: mixed signals or insufficient data.")
        print("      Recommend running Stage 0 for more cities before deciding.")

    print()
    print(f"  Cities with data:   {len(cities_with_data)}")
    print(f"  Cities missing data:{len(cities_missing)}")
    if cities_missing:
        print(f"  Missing: {', '.join(cities_missing)}")
    print("=" * 70)
    print()

    # -----------------------------------------------------------------------
    # Write CSV
    # -----------------------------------------------------------------------
    csv_path = diag_dir / "anisotropy_audit.csv"
    fieldnames = ["city", "variable", "eccentricity", "theta_deg", "kappa_mean",
                  "r_hat_theta", "ess_theta", "converged", "classification"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            row_out = dict(row)
            row_out["classification"] = _classify_row(row)
            writer.writerow(row_out)
    log.info("Anisotropy audit CSV written --> %s", csv_path)

    # -----------------------------------------------------------------------
    # Write JSON summary for runlog consumption
    # -----------------------------------------------------------------------
    summary = {
        "verdict": verdict,
        "n_pairs": n,
        "frac_iso": round(frac_iso, 4),
        "frac_dir_unreliable": round(frac_unrel, 4),
        "frac_dir_good": round(frac_good, 4),
        "n_cities_with_data": len(cities_with_data),
        "n_cities_missing": len(cities_missing),
        "cities_missing": cities_missing,
        "eccentricity_threshold": ECCENTRICITY_THRESHOLD,
        "ess_floor": ESS_FLOOR,
    }
    summary_path = diag_dir / "anisotropy_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("Audit summary written --> %s", summary_path)


def _write_empty_csv(diag_dir: Path) -> None:
    """Write an empty CSV so downstream scripts don't crash on missing file."""
    csv_path = diag_dir / "anisotropy_audit.csv"
    fieldnames = ["city", "variable", "eccentricity", "theta_deg", "kappa_mean",
                  "r_hat_theta", "ess_theta", "converged", "classification"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    summary_path = diag_dir / "anisotropy_audit_summary.json"
    summary_path.write_text(json.dumps({
        "verdict": "DATA_MISSING",
        "note": "Run: python scripts/train_multicity_jepa.py --skip-collect --stages 0",
    }, indent=2))


if __name__ == "__main__":
    main()
