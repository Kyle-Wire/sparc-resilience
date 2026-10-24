# Canopy effect across cities: pool status

Phase 3 of `docs/agent/ROADMAP.md`. The status table comes from `scripts/run_canopy_cities.py`; its full
output is in `output/core/pooled/` (not committed). Updated by the `sparc-steward` agent on 2026-10-24.

**What is pooled.** The pooled effect is **the effect at the streets campaigns drive**: the mean change in
street air at the street points the vehicles sampled, when canopy rises by +10 pp within 100 m of them. It
is not the effect at an average point of the city. The labs score every design against exactly that
estimand since P2 (owner decision 7). Streets that vans drive are not a random sample of a city's paved
cells. In Seattle and Burlington they are leafier, with 31–41% canopy against 8–14% at random paved cells.
Applying the pooled number elsewhere is an extrapolation.

**Pool-entry rule** (owner decisions 2 and 3): a run enters the pool only if street_100 is **Trustworthy**
on that run's own real routes and noise, in a lab with 36 replicates per world. Runs that fail are reported,
not pooled.

## Status (after P2)

| City (campaign) | Idealised lab, street_100 | All runs, +10 pp within 100 m (95% CI) | Runs pooled |
|---|---|---|---|
| Providence, RI (2020-07-29) | Trustworthy | −0.027 °F (−0.073 to +0.019) | 3 of 4 |
| Albuquerque, NM (2021-07-09) | Trustworthy | −0.167 °F (−0.453 to +0.118) | 0 of 2 |
| Burlington, VT (2020-08-10) | Direction only | −0.012 °F (−0.047 to +0.022) | **2 of 3** |
| Seattle, WA (2020-07-27) | Conservative | −0.045 °F (−0.122 to +0.033) | **1 of 3** |
| Scranton, PA (2023) | traverses carry no timestamps; skipped (owner decision 9) | — | — |

| City | Run | Cell visits | Within 100 m (SE) | Real-route verdict (36 reps) | Pooled |
|---|---|---|---|---|---|
| Providence | morning 06–07 | 2,069 | −0.029 (0.033) | Trustworthy | yes |
| Providence | afternoon 15–16 | 2,154 | +0.024 (0.077) | Direction only | no |
| Providence | evening 19–20 | 2,303 | −0.077 (0.019) | Trustworthy | yes |
| Providence | night 00–01 | 2,481 | −0.016 (0.014) | Trustworthy | yes |
| Albuquerque | morning 06–07 | 8,194 | −0.174 (0.184) | Partial | no |
| Albuquerque | afternoon 15–16 | 13,722 | −0.151 (0.189) | Direction only | no |
| Burlington | morning 06–07 | 4,063 | +0.005 (0.016) | Not trustworthy | no |
| Burlington | afternoon 15–16 | 3,383 | −0.017 (0.041) | Trustworthy | yes |
| Burlington | evening 19–20 | 2,879 | −0.044 (0.037) | Trustworthy | yes |
| Seattle | morning 06–07 | 4,787 | −0.072 (0.048) | Partial | no |
| Seattle | afternoon 15–16 | 5,177 | −0.034 (0.061) | Direction only | no |
| Seattle | evening 19–20 | 5,261 | −0.033 (0.053) | Trustworthy | yes |

## Verdicts before and after P2 (owner decision 7)

**Real-route labs** (36 replicates, verdict rules unchanged).

The "after" column is the stored replicates rescored against the truth at the run's own cells (agent
decision A2). A true re-run gives the same numbers to within 9e-16 (checked on Providence's 4 labs and
Burlington's afternoon one).

| City | Run | street_100 before | street_100 after | street_300 before | street_300 after |
|---|---|---|---|---|---|
| Providence | morning | Trustworthy (floor not valid) | Trustworthy (floor valid) | Direction only | Direction only |
| Providence | afternoon | Direction only | Direction only | Conservative | Direction only |
| Providence | evening | Trustworthy | Trustworthy | Trustworthy | Trustworthy |
| Providence | night | Trustworthy (floor valid) | Trustworthy (floor not valid) | Conservative | Conservative |
| Albuquerque | morning | Partial | Partial | Conservative | Not trustworthy |
| Albuquerque | afternoon | Direction only | Direction only | Not trustworthy | Not trustworthy |
| Burlington | morning | Not trustworthy | Not trustworthy | Conservative | Trustworthy |
| Burlington | afternoon | Conservative | **Trustworthy** | Conservative | Trustworthy |
| Burlington | evening | Conservative | **Trustworthy** | Conservative | Trustworthy |
| Seattle | morning | Conservative | **Partial** (misses `additive`) | Conservative | Trustworthy |
| Seattle | afternoon | Conservative | **Direction only** (misses `additive`, `confounded`) | Trustworthy | Trustworthy |
| Seattle | evening | Conservative | **Trustworthy** | Conservative | Trustworthy |

- **Providence** is unchanged at block scale, as expected: its driven streets and its paved cells have the
  same canopy. Floor validity flips in two runs (morning becomes valid, night invalid). The afternoon's 300 m
  reading drops from Conservative to Direction only.
- **Burlington and Seattle** lose the `own_only` miss that P2 was written to remove.
  - **Burlington's** afternoon and evening runs now pass.
  - **Seattle's** evening run passes.
  - **Seattle's** morning and afternoon now miss `additive` (and the afternoon also `confounded`). The
    design overstates them slightly, as it does in Seattle's idealised lab.
- **Burlington's** morning still fails on false positives (14% with no effect).

**Idealised labs.** Owner decision 7 and agent decision A1 call for 36 replicates under P2.

| City | street_100, 12 reps before P2 | 36 reps after P2 | Floors (100 m / 300 m) after |
|---|---|---|---|
| Providence | Trustworthy | **Trustworthy** (null −0.005 °F, 11%) | not valid / valid |
| Burlington | Direction only | **Not trustworthy**: null mean −0.066 °F (above the 0.05 °F limit); misses `confounded` (bias −0.152 °F, coverage 69%) | not valid / not valid |
| Seattle | Conservative | pending (queued after the new cities) | |
| Albuquerque | Trustworthy | pending (queued after the new cities) | |

- **Burlington's simulated campaigns** run on a street grid cut into short pieces by the unpaved
  (forested) cells: 2,190 route cells against 19,210 paved ones. On those campaigns even the no-effect
  world comes out at −0.066 °F.
- **The real routes do not show this.** On the routes the vans actually drove, Burlington's no-effect world
  is clean in all three runs (+0.010, −0.004, +0.013 °F).
- **So the idealised lab is a poor model of Burlington's campaign.** Pool entry uses the real-route labs.
  The idealised failure is reported and kept as a warning on Burlington's numbers.

The 12-replicate labs rescored under P2 (same replicates, new truth):

| City | street_100, 12 reps before P2 | 12 reps after P2 (rescored) |
|---|---|---|
| Providence | Trustworthy | Trustworthy |
| Albuquerque | Trustworthy | Trustworthy |
| Burlington | Direction only | Direction only (misses `additive`, `confounded`) |
| Seattle | Conservative | Conservative (misses `additive`, which it overstates by 14%) |

**Burlington's idealised gap (owner decision 8).** In the idealised lab Burlington still overstates
`additive` (bias −0.133 °F against the route-cell truth of −0.310 °F) and `confounded` (−0.161 °F).

- **Per-cell noise is not the cause:** removing it from `validate.plant` leaves the bias at −0.121 °F.
- **Different canopy knots, (25, 50, 75), close the gap in Burlington.** That knot set was found on
  Burlington, so it was screened on held-out cities before adoption (agent decision A3).
- **Seattle rejects it:** its `own_only` bias rises from +0.022 to +0.052 °F (coverage 33%), and its 300 m
  design falls to Not trustworthy.
- **The knots stay (15, 40). The gap is open.** Knot placement trades error between worlds, city by city.
  A response that does not depend on knots is the next candidate.

## The pooled model (`python -m sparc.core.identify pool`)

`sparc/core/identify/pool.py`:
- REML random effects, with DerSimonian–Laird as the start and the fallback;
- the prediction interval for a new city, μ ± t(k−2)·√(τ² + se(μ)²);
- leave-one-city-out scoring, which is the Phase 3 gate;
- Stouffer's combination for the kilometre test;
- pooling per time window.

On synthetic cities it recovers μ and τ, with leave-one-city-out coverage of 95.6% for 12 cities
(`tests/core/test_pool.py`).

**The first cross-city pooled estimate:**

| Window | Cities pooled | Pooled effect, +10 pp within 100 m (95% CI) | τ | Prediction interval, new city |
|---|---|---|---|---|
| **evening 19–20** | 3 (Providence −0.077, Burlington −0.044, Seattle −0.033) | **−0.066 °F (−0.098 to −0.035)** | 0.000 | −0.27 to +0.14 |
| afternoon | 1 (Burlington) | −0.017 °F (−0.098 to +0.064) | — | needs 3 cities |
| morning | 1 (Providence) | −0.029 °F (−0.093 to +0.035) | — | needs 3 cities |
| night | 1 (Providence) | −0.016 °F (−0.044 to +0.012) | — | needs 3 cities |

- **In the evening**, streets with 10 pp more canopy within 100 m are about 0.07 °F cooler in the three leafy
  northern cities pooled. The three cities agree (τ estimated at 0).
- **The prediction interval for a new city is wide** (−0.27 to +0.14 °F), because three cities leave one
  degree of freedom.
- **Daytime remains a null.** The only city in the afternoon pool (Burlington) is −0.02 °F (−0.10 to +0.06).
- **The Phase 3 gate** (leave-one-city-out coverage) needs at least 4 cities in a window. It is not met.

**Not built yet:**
- the meta-regression on city covariates (canopy and impervious means, dewpoint, wind, Köppen class);
- the pooled city-wide floor.

## Next

- **More cities with timestamps** (owner decision 9):
  - Atlanta (`r74h2`, 2021-09-04, 3 runs × 13 sensors) and Raleigh (`rwjnx`, 2021-07-23, 3 runs × 15–17
    sensors) are set up.
  - Houston (`yqh5u`) is set up but has a 641,785-cell grid, so its labs will take several hours.
- **Each city's labs take** 15 min (Burlington) to 2 h (Albuquerque) on 4 cores.
