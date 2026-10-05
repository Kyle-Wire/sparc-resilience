"""Isolating the causal effect of canopy on air temperature.

The Providence target is not a measurement: it is a heat-watch *product*, a
random forest fitted to vehicle-traverse temperatures on land-cover
predictors and used to fill every cell (72% of its values are whole °F).
Where canopy and impervious surface move together, that forest learns canopy
as a proxy for what it cannot see, so the map carries a canopy dependence
even when canopy has no effect at all: the simulation check's null artefact.
No model of the map can remove it, because the map *is* a function of
canopy.

This package asks which designs recover a planted canopy effect, and stay at
zero when none is planted, on the city's real layout:

* :mod:`~sparc.core.identify.campaign` simulates a traverse campaign on top
  of a simulation-check generator: routes along streets, vehicles driving
  them through the afternoon hour (warming drift, a per-vehicle offset,
  sensor noise), and the forest-made map built from those traverses.
* :mod:`~sparc.core.identify.estimators` holds the designs, each returning
  the city-mean change for a uniform +10 pp canopy with a cluster-robust
  standard error:

  - on the **traverse points** (the measurements): footprint regression with
    a smooth spatial basis; **spatial first differences along routes**
    (neighbouring samples of one pass, seconds apart: the drift, the vehicle
    and every smooth confounder cancel, and street trees over pavement vary
    while the road does not); and an **upwind/downwind contrast**, a
    physical signature test (air carries canopy cooling downwind; confounders
    and isotropic interpolators have no direction);
  - on the **map**: the same footprint regression, an overhanging-canopy
    design within paved cells and neighbourhood fixed effects, and first
    differences between neighbouring cells.
* :mod:`~sparc.core.identify.validate` runs every design on every generator
  (null, additive, own_only, physics, coarse_scale, confounded) and reports
  bias, the null artefact, false-positive rate and interval coverage.
* :mod:`~sparc.core.identify.windshift` is the kilometre-scale design: the
  same streets compared across runs under different winds (cell fixed
  effects, run-specific smooth controls, a rotation test); :mod:`kmlab`
  measures its false alarms and power.
* :mod:`~sparc.core.identify.traverses` reads a real campaign as downloaded
  (CAPA Heat Watch, e.g. OSF ``wu9v7`` for Providence), splits it into runs,
  and applies the validated designs: block-scale effects per run, a floor on
  city-wide cooling, and the kilometre test.

Like :mod:`sparc.core.heat` this package sits below ``sparc/core`` and is
not part of the resume fingerprint.  Run it with ``python -m
sparc.core.identify``.
"""

from __future__ import annotations

DOSE = 10.0                     # pp of canopy: the uniform edit every estimate is expressed for

__all__ = ["DOSE"]
