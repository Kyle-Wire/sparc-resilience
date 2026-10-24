# Canopy effect from the traverses: Raleigh, NC (2021-07-23)

The sixth city, run end to end by `scripts/run_canopy_cities.py raleigh_nc` under P2 scoring. Its real-route
labs used only the traverse designs (agent decision A6). Run by the `sparc-steward` agent on 2026-10-24.
Outputs (not committed) are in `output/core/raleigh_nc/identify/`.

**In short.** Raleigh is a leafy southern city (open canopy averages 64%). It shows **no block-scale canopy
effect in any run**. Pooled over its three runs, +10 pp canopy within 100 m changes street air by
**+0.003 °F (95% CI −0.047 to +0.053)**. The evening reading is +0.016 °F (−0.070 to +0.102), which does not
fit the evening cooling found in Providence, Seattle, Burlington and Atlanta. Only the afternoon run passes
the real-route lab, so only it enters the pool.

| Item | Value |
|---|---|
| OSF project | `rwjnx`: `traverses_chw_raleigh-durham_110821.zip` (2021 layout). No reader change was needed. |
| Campaign | 2021-07-23, from the traverse clock: 06–07, 15–16, 19–20 EDT. 99,275 readings, 15–17 sensors per run. 55–63% fall in the study area: the campaign covers Raleigh–Durham. |
| Grid | 30 m UTM (EPSG:32617): 553,527 fishnet cells, of which 417,640 lie on CAPA's afternoon map |
| Station | KRDU (`72306013722`, 11 km) |
| Run time | 22 min for the driver: the idealised lab already existed, and the real-route labs took about 4 min each with A6 |

| Lab | street_100 | street_300 |
|---|---|---|
| Idealised (12 reps) | Conservative: overstates `additive` and `confounded` by about 23% | Conservative |
| Real routes, morning (36) | Direction only | Conservative |
| Real routes, afternoon (36) | **Trustworthy** | Conservative |
| Real routes, evening (36) | Partial (misses `additive`) | Conservative |

| Run | Cell visits / passes | Within 100 m (95% CI) | Within 300 m (95% CI) |
|---|---|---|---|
| Morning | 7,920 / 79 | −0.000 °F (−0.058 to +0.057) | −0.001 °F (−0.066 to +0.065) |
| Afternoon | 6,031 / 75 | −0.008 °F (−0.102 to +0.085) | −0.015 °F (−0.117 to +0.088) |
| Evening | 7,760 / 88 | +0.016 °F (−0.070 to +0.102) | +0.026 °F (−0.080 to +0.133) |
| All runs | 21,711 | +0.003 °F (−0.047 to +0.053) | |

- **City-wide floor:** none (the idealised lab validates neither floor).
- **Kilometre test:** −0.004 °F (−0.066 to +0.058), rotation p = 1.0. Underpowered.
- **The evening null matters for interpreting the pooled evening effect.** Raleigh's evening run did not
  pass its lab, so it is not pooled. But its reading is a well-measured null, so the evening cooling may not
  hold everywhere. This is exactly what the pooled model's τ and prediction interval are meant to capture
  once more cities pass.

Reproduce: `python scripts/run_canopy_cities.py raleigh_nc`.
