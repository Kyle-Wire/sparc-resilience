"""Resolve per-variable GWR bandwidths from available sources.

Priority: stage0 correlogram results > manual_parameters config > None.

This extracts the three-path logic that previously lived inside
EnhancedSpatialCV._compute_variable_bandwidths() into a pure, testable
function with no I/O side-effects.
"""

from __future__ import annotations


def resolve_bandwidth(
    config: dict,
    stage0_result: dict | None = None,
    target_var: str | None = None,
) -> dict[str, float] | None:
    """Return per-variable GWR bandwidths or None if none are available.

    Parameters
    ----------
    config : dict
        Raw project config dict.  Checked for
        ``config["manual_parameters"]["bandwidths"]``.
    stage0_result : dict or None
        Parsed correlogram analysis JSON.  Expected shape::

            {"individual_results": {var: {"optimal_bandwidth": float}, ...}}

    target_var : str or None
        Target variable name to exclude from the returned dict.

    Returns
    -------
    dict[str, float] or None
        Mapping of predictor name → bandwidth, or None when no valid
        bandwidths are available from any source.
    """
    # Priority 1 — stage0 correlogram results
    if stage0_result is not None:
        individual = stage0_result.get("individual_results", {})
        bandwidths: dict[str, float] = {}
        for var, result in individual.items():
            if var == target_var:
                continue
            bw = result.get("optimal_bandwidth")
            if bw is not None and float(bw) > 0:
                bandwidths[var] = float(bw)
        if bandwidths:
            return bandwidths

    # Priority 2 — manual_parameters in project config
    manual = config.get("manual_parameters", {}).get("bandwidths", None)
    if manual:
        processed: dict[str, float] = {}
        for var, bw in manual.items():
            try:
                processed[var] = float(bw)
            except (ValueError, TypeError):
                continue
        if processed:
            return processed

    return None
