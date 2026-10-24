# Canopy effect from the traverses: Charleston, SC (2021-07-31)

The seventh city, and the first built with the multi-tile mosaic: its study area spans two Copernicus DEM
tiles (owner decision 13). It was run end to end by `scripts/run_canopy_cities.py charleston_sc` (29 min) by
the `sparc-steward` agent on 2026-10-24. Outputs (not committed) are in
`output/core/charleston_sc/identify/`.

**In short.** All three runs pass their real-route labs, so all three enter the pool. No block-scale effect
is detected in any single run, but every reading is negative, and the evening is strongest at −0.050 °F per
+10 pp within 100 m (−0.120 to +0.021). Pooled over the day: −0.026 °F (−0.071 to +0.019).

| Item | Value |
|---|---|
| OSF project | `b4tfy`: `traverses_chw_charleston_111821.zip` (2021 layout). No reader change was needed. |
| Campaign | 2021-07-31, from the traverse clock: 06–07, 15–16, 19–20 EDT. 57,948 readings, 10 sensors per run, 73–78% on the grid. |
| Grid | 30 m UTM: 196,013 cells on CAPA's afternoon map. The DEM is a mosaic of N32 W081 and N32 W080; elevation runs from −11 to 32 m (marsh and harbour). |
| Predictors | Open canopy averages 58%, impervious 27% |
| Station | KJZI (`72060600198`, 11 km). Winds: morning from 222° at 1.9 m/s, afternoon from 179° at 6.1 m/s, evening from 192° at 2.3 m/s. |

| Lab | street_100 | street_300 |
|---|---|---|
| Idealised (12 reps) | Not trustworthy: it recovers all five effect worlds, but flags an effect in 25% of no-effect campaigns | Not trustworthy |
| Real routes, morning (36) | **Trustworthy** | Conservative |
| Real routes, afternoon (36) | **Trustworthy** | Conservative |
| Real routes, evening (36) | **Trustworthy** | Conservative |

| Run | Cell visits / passes | Within 100 m (95% CI) | Within 300 m (95% CI) |
|---|---|---|---|
| Morning | 7,108 / 60 | −0.001 °F (−0.045 to +0.042) | −0.002 °F (−0.046 to +0.042) |
| Afternoon | 6,071 / 65 | −0.033 °F (−0.128 to +0.063) | −0.052 °F (−0.145 to +0.042) |
| Evening | 4,846 / 50 | −0.050 °F (−0.120 to +0.021) | −0.069 °F (−0.150 to +0.011) |
| All runs | 18,025 | −0.026 °F (−0.071 to +0.019) | −0.037 °F (−0.083 to +0.008) |

- **City-wide floor:** none.
- **Kilometre test:** −0.01 °F (−0.08 to +0.06), rotation p = 0.69.

Reproduce: `python scripts/run_canopy_cities.py charleston_sc`.
