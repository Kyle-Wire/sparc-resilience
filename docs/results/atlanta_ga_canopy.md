# Canopy effect from the traverses: Atlanta, GA (2021-09-04)

The fifth city, and the first of owner decision 9's list (cities whose traverses carry timestamps). It was
run end to end by `scripts/run_canopy_cities.py atlanta_ga` (82 min on 4 cores) under P2 scoring. Run by the
`sparc-steward` agent on 2026-10-24. Outputs (not committed) are in `output/core/atlanta_ga/identify/`.

**In short.**
- **The evening run measures canopy cooling.** +10 pp canopy within 100 m of a street point: **−0.185 °F
  (95% CI −0.306 to −0.063)**. Within 300 m: −0.268 °F (−0.404 to −0.132). This is the strongest block-scale
  reading of any city so far.
- **The afternoon and morning runs are nulls:** −0.003 °F (−0.127 to +0.121) and −0.033 °F (−0.105 to
  +0.039).
- **The real-route lab rates street_100 Trustworthy in the afternoon and evening**, so both runs enter the
  pool. The morning is Direction only (it overstates `additive` and `confounded`).
- **The idealised lab on Atlanta's layout rates street_100 Not trustworthy:** 17% false positives with no
  effect, and it overstates `additive` by about 20%. This is the same leafy-layout pattern as Burlington.
  Pool entry rests on the real routes.

## The city as built

| Item | Value |
|---|---|
| OSF project | `r74h2`: `traverses_chw_atlanta_112921.zip`. The 2021 layout has `datetime`, `temp_f`, `hi_f`, `rh`, `file`, `kmph`, `trav_id`. No reader change was needed. |
| Campaign | 2021-09-04, from the traverse clock: 06–07, 15–16, 19–20 EDT. 74,880 readings, 13 sensors per run, 100% on the grid. |
| Grid | 30 m UTM (EPSG:32616): 564,547 fishnet cells, of which 326,827 lie on CAPA's afternoon map |
| Predictors | WorldCover N33W087, Copernicus DEM N33 W085, Sentinel-2 September 2021 (4 scenes, 16SGC). Open canopy averages 64%, impervious 29%. |
| Forcing | ERA5, 15–16 EDT: SW↓ 801 W/m², clear-sky index 1.00. Wind from KFTY (`72219503888`, 11 km): afternoon from 40° at 1.3 m/s, evening from 168° at 1.8 m/s. The morning was calm at KFTY, so ERA5's wind is used (136° at 1.3 m/s). |

## Labs

| Lab | street_100 | street_300 |
|---|---|---|
| Idealised, 12 replicates | Not trustworthy (null −0.028 °F, 17% false positives; `additive` −0.45 vs −0.38 °F, coverage 58%) | Not trustworthy |
| Real routes, morning (36) | Direction only (misses `additive`, `confounded`) | Trustworthy |
| Real routes, afternoon (36) | **Trustworthy** (floor valid) | Conservative |
| Real routes, evening (36) | **Trustworthy** (floor valid) | Not trustworthy (17% false positives) |
| Kilometre lab (winds 136°, 40°, 168°) | Underpowered: no false alarms, no detection | |

## Real traverses

| Run | Cell visits / passes | Within 100 m (95% CI) | Within 300 m (95% CI) |
|---|---|---|---|
| Morning 06–07 | 10,067 / 85 | −0.033 °F (−0.105 to +0.039) | −0.037 °F (−0.114 to +0.040) |
| Afternoon 15–16 | 8,879 / 142 | −0.003 °F (−0.127 to +0.121) | +0.006 °F (−0.125 to +0.137) |
| **Evening 19–20** | 9,345 / 122 | **−0.185 °F (−0.306 to −0.063)** | **−0.268 °F (−0.404 to −0.132)** |
| All runs pooled | 28,291 | −0.074 °F (−0.144 to −0.005) | −0.102 °F (−0.181 to −0.024) |

- **City-wide floor:** none. The idealised lab validates neither floor on this layout.
- **Kilometre test:** −0.02 °F (−0.08 to +0.04), rotation p = 0.61. Not detected; underpowered.

## Place in the pool

Atlanta's afternoon and evening runs enter the street_100 pool. It is the fourth city in the evening window,
the first window with enough cities for leave-one-city-out
(`docs/results/canopy_pooled.md`).

Reproduce: `python scripts/run_canopy_cities.py atlanta_ga`.
