"""LST → air temperature pilot: does satellite land surface temperature carry air-temperature information
beyond land cover, in cities a model never saw?

The target is the **vehicle traverse air temperature** of CAPA Heat Watch campaigns (the measurements),
never CAPA's interpolated area-wide rasters: those are a land-cover model of the traverses (Shandas et al.
2019; ``docs/results/canopy_identification.md``), so a land-cover model would score well on them by
construction.  Surface and air heat islands are known to differ: over 342 European cities in the 2019
heatwave the surface UHI averaged 1.45 °C against 0.26 °C in the air (Venter, Chakraborty & Lee 2021), the
surface and air heat islands are different quantities (Voogt & Oke 2003), and random forests on satellite
predictors explained little of the daily maximum air temperature measured by citizen stations in Oslo
(RMSE 1.85 °C, R² 0.05; Venter et al. 2020).  Forest LST tracks air Tmax while barren and grass surfaces run
10–20 °C hotter (Mildrexler, Zhao & Running 2011).  So the claim is testable and not obvious:

    IF LST carries air-temperature information beyond land cover,
    THEN models that add LST beat a land-cover-only model on cities they never saw.

* :mod:`~sparc.core.lstair.dataset` builds one long table of traverse cells (city × run × 30 m cell) with the
  run's cell-mean air temperature, its anomaly from the run's city background, land-cover predictors with
  focal means, and campaign-matched Landsat LST; one cache per city, so reruns skip finished cities.
* :mod:`~sparc.core.lstair.evaluate` scores the models leave-one-campaign-out: (N) no anomaly, (A) land cover +
  focal features, (B) A + LST, (C) SPARC's physics model with LST entering the heat source as surface–air
  coupling, and turns the unit-paired skill differences (SS = 1 − MSE/MSE_N) into the pre-registered verdict
  (``docs/results/lst_air_pilot.md``), with its sensitivities.
* :mod:`~sparc.core.lstair.landcover_us` adds continuous US land cover (NLCD tree canopy and percent
  impervious, with focal means) to built cities post hoc, for the sensitivity that swaps them in for WorldCover's
  class fractions.
* :mod:`~sparc.core.lstair.calibrate` (extension study S1) calibrates an evaluate run's held-out maps with local
  stations, simulated (k driven cells) and real (NOAA Global Hourly), by simple kriging of their residuals.
* :mod:`~sparc.core.lstair.synthetic` plants worlds where the answer is known (LST informative, redundant,
  or measured at the wrong hour) to check that the harness says what it should before real data arrive.

Like :mod:`sparc.core.identify`, the package sits below ``sparc/core`` and is not part of the resume
fingerprint.  Run it with ``sparc core lstair dataset|add-landcover|evaluate|sensitivity|calibrate|validate``.
"""

from __future__ import annotations

#: Heat Watch run codes by :mod:`sparc.core.identify.traverses` window name (CAPA protocol: 6–7, 15–16 and
#: 19–20 local time; Providence also drove 0–1 the next night).
RUN_CODE = {"morning": "am", "midday": "af", "evening": "pm", "night": "ni"}
RUN_ORDER = ("af", "am", "pm", "ni")


def to_json(obj, indent: int | None = 1) -> str:
    """Strict JSON as the core writes it (:mod:`sparc.core.runio`): NaN/Inf → null, ``allow_nan=False``, numpy and
    pandas values as Python ones, so browsers, ``jq`` and Studio can read every file the pilot writes."""
    import json

    from sparc.core.runio import _jsonable

    return json.dumps(_jsonable(obj), indent=indent, allow_nan=False, default=str)


__all__ = ["RUN_CODE", "RUN_ORDER", "to_json"]
