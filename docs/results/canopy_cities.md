# Canopy cities in the Phase 3 batch: per-city notes

Short notes for each city run by `scripts/run_canopy_cities.py` after the first seven, which have their own
write-ups:
- Providence: `canopy_identification.md`;
- Albuquerque, Burlington, Seattle, Atlanta, Raleigh, Charleston: `<slug>_canopy.md`.

The pooled tables live in `canopy_pooled.md`.

Every run uses the same pipeline:
- setup;
- an idealised lab (12 replicates);
- real-route labs (36 replicates, traverse designs only, P2 scoring);
- a kilometre lab under the run's winds;
- the estimate.

Readings are +10 pp canopy within 100 m (street_100) or 300 m (street_300) of a street point, in °F, with
95% CIs. Outputs (not committed) are in `output/core/<slug>/identify/`.

## Arlington, VA (`3xvmg`, 2021-07-15)

| Item | Value |
|---|---|
| Campaign | 06–07, 15–16, 19–20 EDT; 32,111 readings, 5 sensors per run, 98% on the grid |
| Grid | 74,876 cells; open canopy 55%, impervious 36%; station KDCA |
| Driver time | 16 min |
| Idealised lab | street_100 Direction only (misses `additive`, `confounded`); street_300 Conservative |

| Run | Within 100 m | Real-route verdict | Within 300 m | Real-route verdict |
|---|---|---|---|---|
| Morning | +0.011 (−0.054 to +0.076) | Conservative | −0.023 (−0.104 to +0.059) | Not trustworthy |
| Afternoon | +0.060 (−0.058 to +0.178) | **Trustworthy (pooled)** | −0.001 (−0.157 to +0.155) | Conservative |
| Evening | +0.025 (−0.091 to +0.141) | Direction only | −0.011 (−0.126 to +0.103) | **Trustworthy (pooled, 300 m)** |
| All runs | +0.033 (−0.036 to +0.102) | | −0.012 (−0.095 to +0.072) | |

- **Summary:** no canopy effect is detected at any scale or time.
- **Kilometre test:** −0.05 (−0.15 to +0.06), rotation p = 0.44. Underpowered.

## Richmond, VA (`3xvmg`, 2021-07-15)

| Item | Value |
|---|---|
| Campaign | the same day as Arlington; 86,455 readings, 11–12 sensors per run, 93–95% on the grid |
| Grid | 274,648 cells; open canopy 55%, impervious 29%; station KRIC |
| Driver time | 44 min |
| Idealised lab | street_100 Direction only (misses `additive`, `physics`, `confounded`); street_300 Conservative |

| Run | Within 100 m | Real-route verdict | Within 300 m | Real-route verdict |
|---|---|---|---|---|
| Morning | +0.016 (−0.028 to +0.059) | Direction only | −0.012 (−0.059 to +0.035) | **Trustworthy (pooled, 300 m)** |
| Afternoon | +0.007 (−0.103 to +0.117) | Not trustworthy (14% false positives) | −0.023 (−0.145 to +0.099) | Not trustworthy |
| Evening | −0.019 (−0.117 to +0.079) | **Trustworthy (pooled)** | −0.065 (−0.167 to +0.038) | Conservative |
| All runs | +0.002 (−0.057 to +0.061) | | −0.030 (−0.093 to +0.033) | |

- **Summary:** no canopy effect is detected.
- **Kilometre test:** +0.08 (−0.01 to +0.17), rotation p = 0.39. Underpowered.
