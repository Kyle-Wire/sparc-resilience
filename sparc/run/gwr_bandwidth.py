"""Resolve per-variable GWR bandwidths from available sources.

Priority: stage0 correlogram results > manual_parameters config > None.

This extracts the three-path logic that previously lived inside
EnhancedSpatialCV._compute_variable_bandwidths() into a pure, testable
function with no I/O side-effects.
"""

from __future__ import annotations

# Values of :attr:`ResolvedBandwidths.source`.
SOURCE_STAGE0 = "stage0"          # correlogram (predictor's own range) — auto
SOURCE_AUTO_WIRED = "auto_wired"  # manual_parameters written by Stage 0 — auto
SOURCE_MANUAL = "manual"          # manual_parameters.bandwidths set by the user


class ResolvedBandwidths(dict):
    """``{predictor: bandwidth}`` that remembers where the values came from.

    A plain ``dict`` subclass, so existing callers (equality checks, JSON
    serialisation, ``.items()``) are unaffected.  ``source`` tells
    ``GWRModel`` whether the bandwidths are an explicit user override
    (``"manual"``) — which a ``KernelField`` cross-range must not clobber —
    or auto-derived from each predictor's own autocorrelation range
    (``"stage0"`` / ``"auto_wired"``), which the target↔predictor
    cross-range supersedes.
    """

    def __init__(self, *args, source: str = SOURCE_STAGE0, **kwargs):
        super().__init__(*args, **kwargs)
        self.source = source

    @property
    def is_user_specified(self) -> bool:
        return self.source == SOURCE_MANUAL

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"ResolvedBandwidths({dict.__repr__(self)}, source={self.source!r})"


def bandwidths_are_user_specified(bandwidths) -> bool:
    """True when *bandwidths* must be treated as an explicit user override.

    Untagged mappings (plain dicts from direct ``GWRModel`` construction)
    are treated as explicit, preserving the historical behaviour.
    """
    if not bandwidths:
        return False
    source = getattr(bandwidths, "source", None)
    if source is None:
        return True
    return source == SOURCE_MANUAL


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
    ResolvedBandwidths (a ``dict[str, float]``) or None
        Mapping of predictor name → bandwidth, or None when no valid
        bandwidths are available from any source.  ``result.source`` is
        ``"stage0"`` (correlogram), ``"manual"`` (user-specified
        ``manual_parameters.bandwidths``) or ``"auto_wired"`` (bandwidths
        Stage 0 wrote into ``manual_parameters`` with
        ``manual_parameters.source == "correlogram_auto"``).
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
            return ResolvedBandwidths(bandwidths, source=SOURCE_STAGE0)

    # Priority 2 — manual_parameters in project config
    manual_section = config.get("manual_parameters", {}) or {}
    manual = manual_section.get("bandwidths", None)
    if manual:
        processed: dict[str, float] = {}
        for var, bw in manual.items():
            try:
                processed[var] = float(bw)
            except (ValueError, TypeError):
                continue
        if processed:
            source = (
                SOURCE_AUTO_WIRED
                if manual_section.get("source") == "correlogram_auto"
                else SOURCE_MANUAL
            )
            return ResolvedBandwidths(processed, source=source)

    return None
