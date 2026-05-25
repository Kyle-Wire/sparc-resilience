"""
manifest.py — Variable availability tracking and manifest JSON output.

Tracks the fetch status and spatial coverage of every variable in the
data collection pipeline.  Acts as the gate that prevents the assembler
from running until all required variables are confirmed available.

No network calls are made here — this is pure bookkeeping.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------

class VariableStatus(str, Enum):
    PENDING   = "pending"     # not yet fetched
    FETCHING  = "fetching"    # fetch in progress
    COMPLETE  = "complete"    # fetched, coverage acceptable
    WARNING   = "warning"     # fetched, but coverage < 100 % (non-blocking)
    ERROR     = "error"       # fetch failed or coverage = 0 % (blocking if required)
    SKIPPED   = "skipped"     # not applicable for this study area (e.g. HOLC outside US)


# ---------------------------------------------------------------------------
# Per-variable entry
# ---------------------------------------------------------------------------

@dataclass
class ManifestEntry:
    """Metadata for a single variable in the collection manifest.

    Attributes
    ----------
    name : str
        Variable column name as it will appear in the output GeoParquet.
    description : str
        Human-readable description shown in the desktop manifest table.
    source_name : str
        Name of the API or file source (e.g. "MRLC WCS", "Local file").
    source_tier : int
        Fallback tier used (1 = primary, 2 = first fallback, etc.).
    required : bool
        If True and status is ERROR, the assembler is blocked.
    status : VariableStatus
        Current fetch status.
    coverage_pct : float
        Fraction of grid cells with a non-null value, 0.0–1.0.
    fetch_timestamp : str | None
        ISO 8601 timestamp when the fetch completed.
    notes : str
        Any warnings or error messages from the fetch.
    """

    name: str
    description: str
    source_name: str = "pending"
    source_tier: int = 1
    required: bool = True
    status: VariableStatus = VariableStatus.PENDING
    coverage_pct: float = 0.0
    fetch_timestamp: Optional[str] = None
    notes: str = ""

    def mark_complete(self, coverage_pct: float, source_name: str, tier: int = 1) -> None:
        """Record a successful fetch."""
        self.coverage_pct = coverage_pct
        self.source_name = source_name
        self.source_tier = tier
        self.fetch_timestamp = _now_iso()
        if coverage_pct == 0.0:
            self.status = VariableStatus.ERROR
            self.notes = "Fetch succeeded but no grid cells have data."
        elif coverage_pct < 1.0:
            self.status = VariableStatus.WARNING
            self.notes = f"{(1 - coverage_pct) * 100:.1f}% of cells are missing values."
        else:
            self.status = VariableStatus.COMPLETE
            self.notes = ""

    def mark_error(self, message: str) -> None:
        self.status = VariableStatus.ERROR
        self.fetch_timestamp = _now_iso()
        self.notes = message

    def mark_skipped(self, reason: str = "") -> None:
        self.status = VariableStatus.SKIPPED
        self.required = False
        self.notes = reason or "Not applicable for this study area."

    def mark_fetching(self) -> None:
        self.status = VariableStatus.FETCHING

    @property
    def is_blocking(self) -> bool:
        return self.required and self.status == VariableStatus.ERROR


# ---------------------------------------------------------------------------
# Manifest container
# ---------------------------------------------------------------------------

@dataclass
class VariableManifest:
    """Collection of ManifestEntry records for a single dataset build run.

    Usage
    -----
    manifest = VariableManifest.for_uhi()
    manifest.update("lst", coverage_pct=0.98, source_name="USGS STAC")
    if manifest.can_build:
        assembler.run(manifest)
    manifest.save(output_dir / "data_manifest.json")
    """

    entries: Dict[str, ManifestEntry] = field(default_factory=dict)
    generated_at: str = field(default_factory=_now_iso)
    boundary_description: str = ""
    temporal_description: str = ""

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def for_uhi(cls) -> "VariableManifest":
        """Return a manifest pre-populated with the full UHI variable set."""
        import copy
        return cls.empty([copy.copy(e) for e in _UHI_VARIABLE_DEFINITIONS])

    @classmethod
    def for_domain(cls, domain: str) -> "VariableManifest":
        """Return a manifest pre-populated with variable definitions for *domain*.

        Parameters
        ----------
        domain : str
            Domain identifier, e.g. ``"uhi"``, ``"wildfire"``, ``"coastal"``.

        Raises
        ------
        KeyError
            If no variable definitions are registered for *domain*.
        """
        if domain not in _DOMAIN_VARIABLE_REGISTRY:
            raise KeyError(
                f"No variable definitions registered for domain '{domain}'. "
                f"Known domains: {sorted(_DOMAIN_VARIABLE_REGISTRY)}"
            )
        import copy
        return cls.empty([copy.copy(e) for e in _DOMAIN_VARIABLE_REGISTRY[domain]])

    @classmethod
    def empty(cls, variables: List[ManifestEntry]) -> "VariableManifest":
        """Return a manifest from an explicit variable list."""
        return cls(entries={e.name: e for e in variables})

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def update(
        self,
        name: str,
        *,
        coverage_pct: float,
        source_name: str,
        tier: int = 1,
    ) -> None:
        """Mark a variable complete with coverage statistics."""
        if name not in self.entries:
            raise KeyError(f"Variable '{name}' not in manifest")
        self.entries[name].mark_complete(coverage_pct, source_name, tier)

    def error(self, name: str, message: str) -> None:
        if name not in self.entries:
            raise KeyError(f"Variable '{name}' not in manifest")
        self.entries[name].mark_error(message)

    def skip(self, name: str, reason: str = "") -> None:
        if name not in self.entries:
            raise KeyError(f"Variable '{name}' not in manifest")
        self.entries[name].mark_skipped(reason)

    def fetching(self, name: str) -> None:
        if name not in self.entries:
            raise KeyError(f"Variable '{name}' not in manifest")
        self.entries[name].mark_fetching()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def can_build(self) -> bool:
        """True when no required variable is in ERROR or FETCHING status."""
        return not any(e.is_blocking or e.status == VariableStatus.FETCHING
                       for e in self.entries.values())

    @property
    def blocking_variables(self) -> List[str]:
        return [name for name, e in self.entries.items() if e.is_blocking]

    @property
    def complete_variables(self) -> List[str]:
        return [name for name, e in self.entries.items()
                if e.status in (VariableStatus.COMPLETE, VariableStatus.WARNING, VariableStatus.SKIPPED)]

    def summary(self) -> dict:
        total = len(self.entries)
        complete = len(self.complete_variables)
        blocking = len(self.blocking_variables)
        return {
            "total": total,
            "complete": complete,
            "blocking_errors": blocking,
            "can_build": self.can_build,
        }

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_api_dict(self) -> dict:
        """Return a dict shaped for the frontend CollectManifest TypeScript interface.

        Field mapping vs. to_dict():
          status        → UPPERCASE (matches TS union literal type)
          source_tier   → tier
          notes         → error_message (only populated on error)
        Top-level keys: variables, can_build, blocking_variables,
                        boundary_description, temporal_description
        """
        return {
            "can_build": self.can_build,
            "blocking_variables": self.blocking_variables,
            "boundary_description": self.boundary_description or None,
            "temporal_description": self.temporal_description or None,
            "variables": [
                {
                    "name": e.name,
                    "required": e.required,
                    "status": e.status.value.upper(),
                    "coverage_pct": round(e.coverage_pct, 4) if e.coverage_pct else None,
                    "source_name": e.source_name if e.source_name != "pending" else None,
                    "tier": e.source_tier,
                    "error_message": e.notes if e.status == VariableStatus.ERROR else None,
                }
                for e in self.entries.values()
            ],
        }

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "boundary": self.boundary_description,
            "temporal": self.temporal_description,
            "summary": self.summary(),
            "variables": [
                {
                    "name": e.name,
                    "description": e.description,
                    "source_name": e.source_name,
                    "source_tier": e.source_tier,
                    "required": e.required,
                    "status": e.status.value,
                    "coverage_pct": round(e.coverage_pct, 4),
                    "fetch_timestamp": e.fetch_timestamp,
                    "notes": e.notes,
                }
                for e in self.entries.values()
            ],
        }

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> "VariableManifest":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        entries = {}
        for v in data.get("variables", []):
            e = ManifestEntry(
                name=v["name"],
                description=v.get("description", ""),
                source_name=v.get("source_name", "pending"),
                source_tier=v.get("source_tier", 1),
                required=v.get("required", True),
                status=VariableStatus(v.get("status", "pending")),
                coverage_pct=v.get("coverage_pct", 0.0),
                fetch_timestamp=v.get("fetch_timestamp"),
                notes=v.get("notes", ""),
            )
            entries[e.name] = e
        m = cls(entries=entries)
        m.generated_at = data.get("generated_at", _now_iso())
        m.boundary_description = data.get("boundary", "")
        m.temporal_description = data.get("temporal", "")
        return m


# ---------------------------------------------------------------------------
# UHI variable definitions
# ---------------------------------------------------------------------------


_UHI_VARIABLE_DEFINITIONS: list[ManifestEntry] = [
    # --- Target ---
    ManifestEntry("aat_residual",        "Air temp residual (CAPA − ERA5 background)",                   required=True),
    # --- NOAA CAPA ---
    ManifestEntry("aat_morning",         "CAPA air temperature — morning (~6 AM)",                       required=True),
    ManifestEntry("aat_midday",          "CAPA air temperature — midday (~12–2 PM)",                     required=True),
    ManifestEntry("aat_night",           "CAPA air temperature — night (~10 PM)",                        required=True),
    ManifestEntry("diurnal_aat",         "Diurnal air temp range (midday − night)",                      required=True),
    # --- ERA5 ---
    ManifestEntry("era5_t2m",            "ERA5 background 2 m temperature (Open-Meteo)",                 required=True),
    # --- Landsat spectral ---
    ManifestEntry("lst",                 "Land Surface Temperature (Landsat thermal band B10)",          required=True),
    ManifestEntry("ndvi",                "NDVI — vegetation density; primary cooling predictor",         required=True),
    ManifestEntry("ndbi",                "NDBI — built-up surface fraction",                             required=True),
    ManifestEntry("mndwi",               "MNDWI — urban water bodies (better than NDWI in cities)",     required=True),
    ManifestEntry("albedo",              "Broadband surface albedo — solar energy reflection",           required=True),
    ManifestEntry("ndwi",                "NDWI — open water extent",                                    required=False),
    ManifestEntry("savi",                "SAVI — vegetation on sparse/dry soils",                       required=False),
    ManifestEntry("evi",                 "EVI — enhanced vegetation (dense canopy)",                    required=False),
    ManifestEntry("ndbai",               "NDBaI — bare land / low-albedo surfaces",                     required=False),
    ManifestEntry("nbr",                 "NBR — dry vegetation / post-fire scars",                      required=False),
    ManifestEntry("ndmi",                "NDMI — canopy moisture content",                              required=False),
    ManifestEntry("ui",                  "UI — urban impervious (alternative to NDBI)",                 required=False),
    ManifestEntry("bsi",                 "BSI — bare soil differentiation",                             required=False),
    # --- NLCD ---
    ManifestEntry("pct_impervious",      "NLCD impervious surface fraction (0–100)",                    required=True),
    ManifestEntry("pct_canopy",          "NLCD tree canopy cover fraction (0–100)",                     required=True),
    ManifestEntry("land_cover",          "NLCD land cover class (numeric)",                             required=True),
    # --- Buildings ---
    ManifestEntry("bldg_height_mean",    "Mean building height per cell (m) — fallback chain",          required=True),
    ManifestEntry("bldg_coverage",       "Building footprint coverage fraction per cell",               required=True),
    ManifestEntry("svf",                 "Sky View Factor — H:W proxy (0=canyon, 1=open sky)",         required=True),
    # --- Equity ---
    ManifestEntry("holc_grade",          "HOLC redlining grade (A=4, B=3, C=2, D=1)",                  required=False),
    ManifestEntry("cdc_svi",             "CDC Social Vulnerability Index (0–1)",                        required=False),
    ManifestEntry("ejscreen_score",      "EPA EJScreen composite score",                               required=False),
]

# ---------------------------------------------------------------------------
# Domain variable registry
# ---------------------------------------------------------------------------

#: Maps domain identifier → ordered list of ManifestEntry definitions.
#: Add a new domain by inserting a key here; no other code changes are needed.
#:
#: Example — adding a wildfire domain::
#:
#:     _DOMAIN_VARIABLE_REGISTRY["wildfire"] = [
#:         ManifestEntry("dnbr", "Differenced Normalized Burn Ratio", required=True),
#:         ...
#:     ]
_DOMAIN_VARIABLE_REGISTRY: Dict[str, List[ManifestEntry]] = {
    "uhi": _UHI_VARIABLE_DEFINITIONS,
}
