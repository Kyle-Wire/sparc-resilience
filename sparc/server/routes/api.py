"""sparc.server.routes.api — /api/* endpoints.

Migrated from app.py.  Importable independently of the full FastAPI app.

Currently migrated:
  - GET  /api/hardware
  - GET  /api/preferences
  - PUT  /api/preferences
  - POST /api/hardware/validate
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from sparc.server.deps import state

router = APIRouter(tags=["api"])


@router.get("/api/hardware")
async def hardware_profile():
    """Return the auto-detected hardware tier and resulting pipeline knobs."""
    try:
        from sparc.config.hardware_profile import (
            compute_safe_caps,
            detect_profile,
            detect_raw_profile,
            reset_profile_cache,
        )
        from sparc.config.user_preferences import load_preferences

        reset_profile_cache()

        raw = detect_raw_profile()
        effective = detect_profile()
        prefs = load_preferences()
        global_perf = (
            prefs.get("performance", {})
            if isinstance(prefs.get("performance"), dict)
            else {}
        )
        project_perf = {}
        if state.project_config is not None:
            cfg_perf = state.project_config.get("performance", {})
            if isinstance(cfg_perf, dict):
                project_perf = cfg_perf

        return {
            "detected": raw.as_dict(),
            "safe_caps": compute_safe_caps(raw),
            "global_prefs": global_perf,
            "project_overrides": project_perf,
            "effective": effective.as_dict(),
        }
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.get("/api/preferences")
async def get_user_preferences():
    """Return the global user preferences (machine-wide defaults)."""
    try:
        from sparc.config.user_preferences import get_prefs_path, load_preferences
        return {"path": str(get_prefs_path()), "preferences": load_preferences()}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.put("/api/preferences")
async def update_user_preferences(body: dict[str, Any]):
    """Replace or merge the global user preferences."""
    try:
        from sparc.config.hardware_profile import detect_profile, reset_profile_cache
        from sparc.config.user_preferences import (
            load_preferences,
            save_preferences,
            update_performance_preferences,
        )

        if "performance" in body and isinstance(body["performance"], dict):
            update_performance_preferences(body["performance"])
        else:
            current = load_preferences()
            current.update(body)
            save_preferences(current)
            reset_profile_cache()

        return {
            "preferences": load_preferences(),
            "effective": detect_profile().as_dict(),
        }
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.post("/api/hardware/validate")
async def validate_performance_settings(body: dict[str, Any]):
    """Validate proposed performance settings against the safe caps.

    Returns ``{ ok, warnings: [...], errors: [...] }`` where warnings are
    non-blocking and errors are structurally invalid.
    """
    try:
        from sparc.config.hardware_profile import (
            VALID_PRESETS,
            VALID_TIERS,
            compute_safe_caps,
            detect_raw_profile,
        )

        raw = detect_raw_profile()
        caps = compute_safe_caps(raw)
        warnings_out: list[dict[str, Any]] = []
        errors_out: list[dict[str, Any]] = []

        preset = body.get("preset")
        if preset is not None and preset not in VALID_PRESETS:
            errors_out.append({"field": "preset", "message": f"unknown preset '{preset}'"})

        tier = body.get("hardware_tier_override")
        if tier is not None and tier not in VALID_TIERS:
            errors_out.append({"field": "hardware_tier_override", "message": f"unknown tier '{tier}'"})

        for field, cap in (
            ("max_workers", caps["max_workers"]),
            ("outer_jobs", caps["outer_jobs"]),
            ("inner_jobs", caps["inner_jobs"]),
            ("batch_size", caps["batch_size"]),
            ("nuts_thin", caps["nuts_thin"]),
        ):
            val = body.get(field)
            if val is None:
                continue
            try:
                n = int(val)
            except (TypeError, ValueError):
                errors_out.append({"field": field, "message": "must be an integer"})
                continue
            if n < 1:
                errors_out.append({"field": field, "message": "must be >= 1"})
            elif n > cap:
                warnings_out.append({
                    "field": field,
                    "value": n,
                    "cap": cap,
                    "message": f"{field}={n} exceeds the safe cap of {cap} for this hardware",
                })

        mem = body.get("memory_limit_gb")
        if mem is not None:
            try:
                m = float(mem)
                if m < 1.0:
                    errors_out.append({"field": "memory_limit_gb", "message": "must be >= 1.0 GB"})
                elif m > caps["memory_limit_gb"]:
                    warnings_out.append({
                        "field": "memory_limit_gb",
                        "value": m,
                        "cap": caps["memory_limit_gb"],
                        "message": (
                            f"memory_limit_gb={m:.1f} exceeds the safe cap of "
                            f"{caps['memory_limit_gb']:.1f} GB; the OS may kill the pipeline"
                        ),
                    })
            except (TypeError, ValueError):
                errors_out.append({"field": "memory_limit_gb", "message": "must be numeric"})

        return {
            "ok": not errors_out,
            "warnings": warnings_out,
            "errors": errors_out,
            "safe_caps": caps,
        }
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
