"""Master GeoPackage builder.

Called at the end of a pipeline run, this module walks all the
stage-specific GeoPackages the pipeline produces and merges them into a
single ``master.gpkg`` at the run root.  Each source file becomes a
named layer, so GIS tools can browse the entire run from one file.

It also registers the master file with the :class:`RunRegistry`, so
the desktop app and any consumer can surface it as the canonical
spatial artifact.

The function is intentionally best-effort: missing layers are logged,
never fatal.  If ``fiona`` / ``geopandas`` aren't importable the
function is a no-op.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, List, Optional, Tuple

__all__ = ["MASTER_GPKG_NAME", "iter_run_gpkgs", "build_master_gpkg"]

MASTER_GPKG_NAME = "master.gpkg"

# Known GPKGs the pipeline writes.  Each entry is
# ``(layer_name, path_attr_or_None, relative_path)``:
#   * If ``path_attr`` is set, look up that attribute on ``paths`` and
#     append ``relative_path`` to it.
#   * Otherwise ``relative_path`` is resolved against
#     ``paths.output_dir``.
# This keeps us robust to the configurable stage-directory names.
_STATIC_SOURCES: Tuple[Tuple[str, Optional[str], str], ...] = (
    ("fishnet", None, "fishnet.gpkg"),
    ("fishnet_with_stats", None, "fishnet_with_stats.gpkg"),
    ("stage2_cv_predictions", "stage2_dir", "spatial_cv_predictions.gpkg"),
    ("stage2_ensemble_predictions", "stage2_dir", "final_ensemble_predictions.gpkg"),
    ("stage2_gwr_coefficients", "stage2_dir", "gwr_coefficient_map.gpkg"),
    ("stage2_gwr_modifiers", "stage2_dir", "gwr_modifier_maps.gpkg"),
    ("stage2_ggpgam_uncertainty", "stage2_dir", "ggpgam_uncertainty_maps.gpkg"),
    ("stage3_cate_maps", "stage3_dir", "spatial_cate_maps.gpkg"),
    ("stage4_scenarios", "stage4_dir", "scenario_results.gpkg"),
    ("stage4_scenarios_dag", "stage4_dir", "scenario_results_dag.gpkg"),
    ("stage4_scenarios_hybrid", "stage4_dir", "scenario_results_hybrid.gpkg"),
    ("stage4_scenarios_reprediction", "stage4_dir", "scenario_results_reprediction.gpkg"),
)


def _sanitize_layer_name(name: str) -> str:
    """GeoPackage layer names can't contain certain characters."""
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_")
    return cleaned[:60] or "layer"


def iter_run_gpkgs(paths: Any) -> List[Tuple[str, Path]]:
    """Return ``(layer_name, path)`` tuples for every GPKG in this run.

    Includes both statically-known sources and a glob sweep of
    ``stage4/*.gpkg`` (Bayesian mode writes one file per intervention)
    minus the master file itself.
    """
    output_dir = Path(paths.output_dir)
    found: List[Tuple[str, Path]] = []

    for layer_name, path_attr, rel in _STATIC_SOURCES:
        base = Path(getattr(paths, path_attr)) if path_attr else output_dir
        candidate = base / rel
        if candidate.exists():
            found.append((layer_name, candidate))

    # Sweep stage-4 for any additional scenario GPKGs we didn't list.
    stage4_dir = Path(getattr(paths, "stage4_dir", output_dir / "stage4"))
    if stage4_dir.exists():
        known = {p.name for _, p in found}
        for extra in sorted(stage4_dir.glob("*.gpkg")):
            if extra.name == MASTER_GPKG_NAME or extra.name in known:
                continue
            found.append((_sanitize_layer_name(f"stage4_{extra.stem}"), extra))

    # Sweep stage-3 for any additional spatial CATE files not in the static list.
    stage3_dir = Path(getattr(paths, "stage3_dir", output_dir / "stage3"))
    if stage3_dir.exists():
        known = {p.name for _, p in found}
        for extra in sorted(stage3_dir.glob("*.gpkg")):
            if extra.name == MASTER_GPKG_NAME or extra.name in known:
                continue
            found.append((_sanitize_layer_name(f"stage3_{extra.stem}"), extra))

    return found


def build_master_gpkg(
    paths: Any,
    *,
    registry: Any = None,
    config: Optional[dict] = None,
) -> Optional[Path]:
    """Merge all stage GPKGs into a single master file.

    Returns the path to the master file, or ``None`` if no sources
    were found.  Failures writing individual layers are logged but do
    not abort the merge.
    """
    try:
        import geopandas as gpd  # noqa: F401
    except ImportError:
        print("  [master gpkg] geopandas not installed — skipping")
        return None

    # When disk writes are disabled the canonical store is artifacts.db; the
    # consolidated master.gpkg is a disk-only convenience artefact, so we
    # skip it here.  Users can rebuild it on demand via the export API.
    try:
        from sparc.run.disk_policy import disk_writes_enabled
        if not disk_writes_enabled():
            print("  [master gpkg] disk writes disabled — skipping master GPKG build")
            return None
    except Exception:  # noqa: BLE001
        pass

    sources = iter_run_gpkgs(paths)

    # Also pull geometry-bearing tables from artifacts.db.
    store_layers: List[Tuple[str, Any]] = []  # (layer_name, GeoDataFrame)
    if registry is not None:
        try:
            from sparc.registry.store import ArtifactStore
            _store = ArtifactStore(registry)
            _STORE_LAYER_MAP = {
                ("2", "spatial_cv_predictions"): "stage2_cv_predictions",
                ("2", "ensemble_predictions_geo"): "stage2_ensemble_predictions",
            }
            manifest_stages = getattr(registry.manifest, "stages", {}) or {}
            for stage_key, stage_manifest in manifest_stages.items():
                # stage_manifest is a StageManifest pydantic model; pull its
                # artifacts dict (id -> ArtifactEntry). Fall back to a raw
                # dict in case a caller passes a plain mapping.
                if isinstance(stage_manifest, dict):
                    artifact_dict = stage_manifest
                else:
                    artifact_dict = getattr(stage_manifest, "artifacts", None) or {}
                if not artifact_dict:
                    continue
                for art_id, entry in artifact_dict.items():
                    sk = getattr(entry, "storage_kind", None)
                    if sk != "table":
                        continue
                    md = getattr(entry, "metadata", {}) or {}
                    if "geometry_col" not in md:
                        continue
                    layer_name = _STORE_LAYER_MAP.get((str(stage_key), art_id),
                                                       _sanitize_layer_name(f"stage{stage_key}_{art_id}"))
                    try:
                        gdf = _store.read_any(stage_key, art_id)
                        if gdf is None or len(gdf) == 0:
                            continue
                        store_layers.append((layer_name, gdf))
                    except Exception as _e:
                        print(f"  [master gpkg] skip store layer {stage_key}/{art_id}: {_e}")
        except Exception as _e:
            print(f"  [master gpkg] could not enumerate store layers: {_e}")

    if not sources and not store_layers:
        print("  [master gpkg] no source GPKGs found")
        return None

    output_dir = Path(paths.output_dir)
    master_path = output_dir / MASTER_GPKG_NAME

    # Remove any prior master so we start fresh.
    if master_path.exists():
        try:
            master_path.unlink()
        except OSError as exc:
            print(f"  [master gpkg] could not remove stale file: {exc}")

    import geopandas as gpd

    written: List[str] = []
    # First, write store-resident geometry layers.
    for layer_name, gdf in store_layers:
        target_name = layer_name
        base = target_name
        i = 2
        while target_name in written:
            target_name = f"{base}_{i}"
            i += 1
        try:
            gdf.to_file(master_path, driver="GPKG", layer=target_name)
            written.append(target_name)
        except Exception as exc:
            import traceback
            print(f"  [master gpkg] failed to write store layer {target_name}: {exc}")
            traceback.print_exc()

    for layer_name, src in sources:
        try:
            # A single source file may itself contain multiple layers
            # (e.g. spatial_cate_maps.gpkg has one layer per treatment).
            import fiona
            layers = fiona.listlayers(str(src))
        except Exception:
            layers = [None]  # let geopandas pick the default

        for inner in layers:
            try:
                gdf = gpd.read_file(src, layer=inner) if inner else gpd.read_file(src)
            except Exception as exc:
                import traceback
                print(f"  [master gpkg] skip {src.name}:{inner} ({exc})")
                traceback.print_exc()
                continue
            if gdf is None or len(gdf) == 0:
                continue
            target_name = layer_name if inner in (None, layer_name) else _sanitize_layer_name(f"{layer_name}__{inner}")
            # Disambiguate collisions.
            base = target_name
            i = 2
            while target_name in written:
                target_name = f"{base}_{i}"
                i += 1
            try:
                gdf.to_file(master_path, driver="GPKG", layer=target_name)
                written.append(target_name)
            except Exception as exc:
                import traceback
                print(f"  [master gpkg] failed to write {target_name}: {exc}")
                traceback.print_exc()

    if not written:
        print("  [master gpkg] no layers written")
        return None

    print(f"  [master gpkg] {len(written)} layer(s): {', '.join(written)}")

    # Register with the run registry so the UI / server can surface it.
    if registry is not None:
        try:
            registry.register_artifact(
                stage="final",
                artifact_id="master_gpkg",
                path=master_path,
                format="gpkg",
                producer="build_master_gpkg",
                consumers=["desktop:export", "gis:tools"],
                metadata={
                    "layers": written,
                    "n_layers": len(written),
                    "sources": [str(p.name) for _, p in sources],
                },
            )
        except Exception as exc:
            print(f"  [master gpkg] registry registration failed: {exc}")

    return master_path
