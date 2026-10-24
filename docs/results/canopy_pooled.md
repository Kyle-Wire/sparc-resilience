# Canopy effect across cities: pool status

Phase 3 of `docs/agent/ROADMAP.md`. The status table comes from `scripts/run_canopy_cities.py`; its full
output is in `output/core/pooled/` (not committed). Updated by the `sparc-steward` agent on 2026-10-24.

**Pool-entry rule** (owner decision 2, 2026-10-24): a run enters the pool only if the block-scale design
(street_100) is **Trustworthy** on that run's own real routes and noise, in a lab with 36 replicates per
world (owner decision 3). Runs that fail are reported, not pooled.

## Status

| City (campaign) | Status | Idealised lab, street_100 | All runs, +10 pp within 100 m (95% CI) | Runs pooled |
|---|---|---|---|---|
| Providence, RI (2020-07-29) | ok | Trustworthy | −0.027 °F (−0.073 to +0.019) | 3 of 4 |
| Albuquerque, NM (2021-07-09) | ok | Trustworthy | −0.167 °F (−0.453 to +0.118) | 0 of 2 |
| Burlington, VT (2020-08-10) | ok | Direction only | −0.012 °F (−0.047 to +0.022) | 0 of 3 |
| Seattle, WA (2020-07-27) | ok | Conservative | −0.045 °F (−0.122 to +0.033) | 0 of 3 |
| Scranton, PA (2023) | failed at setup: the traverses carry no timestamps (proposal P3) | — | — | — |

| City | Run | Cell visits | Within 100 m (SE) | Real-route verdict (36 reps) | Pooled |
|---|---|---|---|---|---|
| Providence | morning 06–07 | 2,069 | −0.029 (0.033) | Trustworthy | yes |
| Providence | afternoon 15–16 | 2,154 | +0.024 (0.077) | Direction only | no |
| Providence | evening 19–20 | 2,303 | −0.077 (0.019) | Trustworthy | yes |
| Providence | night 00–01 | 2,481 | −0.016 (0.014) | Trustworthy | yes |
| Albuquerque | morning 06–07 | 8,194 | −0.174 (0.184) | Partial | no |
| Albuquerque | afternoon 15–16 | 13,722 | −0.151 (0.189) | Direction only | no |
| Burlington | morning 06–07 | 4,063 | +0.005 (0.016) | Not trustworthy | no |
| Burlington | afternoon 15–16 | 3,383 | −0.017 (0.041) | Conservative | no |
| Burlington | evening 19–20 | 2,879 | −0.044 (0.037) | Conservative | no |
| Seattle | morning 06–07 | 4,787 | −0.072 (0.048) | Conservative | no |
| Seattle | afternoon 15–16 | 5,177 | −0.034 (0.061) | Conservative | no |
| Seattle | evening 19–20 | 5,261 | −0.033 (0.053) | Conservative | no |

Re-running the batch driver over all five cities only refreshes the estimates, because every lab already
exists. The labs themselves took:

| City | Idealised lab | Each real-route lab (36 reps) |
|---|---|---|
| Providence | 3 min | about 3 min |
| Burlington | 4.4 min | about 6 min |
| Albuquerque | 27 min | 21–26 min |
| Seattle | about 12 min | 15–20 min |

## What can be pooled today

- **Only one city.** Providence's morning, evening and night runs pass. That is no basis for a cross-city
  model, so the random-effects meta-analysis (`pool.py`) has nothing to combine yet.
- **Albuquerque fails on precision.** Its SE is about 0.19 °F per run: the city has almost no street canopy
  and noisy air.
- **Burlington and Seattle fail on scoring.** Their real-route verdicts fail only through the `own_only`
  world. In Burlington this is the opposite direction to its idealised lab. The vans drove leafier streets
  (31–41% canopy) than the paved cells the lab's truth uses (8–14%). In Providence the two agree (19–21%
  against 21%), and Providence passes. Both trace to the lab computing its truth at random paved
  cells, not at the streets campaigns drive: proposal P2
  (`docs/agent/proposals/P2_lab_truth_at_sampled_cells.md`), awaiting approval.
  - **If P2 is approved,** the real-route labs of Burlington, Seattle and Providence are re-run under the new scoring, at
    36 replicates and with the same verdict rules, and this table is updated whatever the result.
- **The block-scale readings agree, pooled or not.**
  - The day-averaged intervals exclude block-scale cooling stronger than 0.07 °F per +10 pp within 100 m in
    Providence, 0.05 °F in Burlington and 0.12 °F in Seattle. Albuquerque's interval is too wide to say.
  - In all three leafy cities the 300 m reading is more negative than the 100 m one, and strongest in the
    evening.

## The pooled model (`python -m sparc.core.identify pool`)

`sparc/core/identify/pool.py` is the random-effects model the roadmap asks for:

- **Fit:** REML for the between-city variance τ², with DerSimonian–Laird as the start and the fallback.
- **Prediction interval** for a new city: μ ± t(k−2)·√(τ² + se(μ)²).
- **Leave-one-city-out scoring** (the Phase 3 gate): each city's own estimate must fall inside the prediction
  from the others, widened by its own sampling error.
- **Stouffer's combination** for the kilometre test.
- **Pooling is per time window.**

On synthetic cities (true effect −0.10 °F, τ = 0.05 °F, SEs 0.02–0.08 °F) it recovers μ and τ to within
0.01 °F with 400 cities. Leave-one-city-out coverage is 95.6% with 12 cities and 98% with 6 (nominal 95%;
`tests/core/test_pool.py`).

On today's pool it has only Providence:

| Window | Cities | Pooled effect (95% CI) | Prediction interval, new city |
|---|---|---|---|
| morning | 1 (Providence) | −0.029 (−0.093 to +0.035) | needs 3 cities |
| afternoon | 0 | — | — |
| evening | 1 (Providence) | −0.077 (−0.115 to −0.039) | needs 3 cities |
| night | 1 (Providence) | −0.016 (−0.044 to +0.012) | needs 3 cities |

**Not built yet:**
- the meta-regression on city covariates: canopy and impervious means, dewpoint, wind, Köppen class;
- the pooled city-wide floor.

Both need several pooled cities to mean anything.

## Next

- **The second dimension of Phase 3:** more cities through `run_canopy_cities.py`. Each city costs 15 min
  (small, like Burlington) to 2 h (large, like Albuquerque) of lab time on 4 cores.
- **Single-tile northern cities so far:**
  - **Seattle** (`mz79p`): done, above.
  - **Scranton** (`8cnq4`): blocked. Its 2023 traverses carry no timestamps or sensor ids (P3).
  - **Portland** (`e7js9`): its OSF listing returned HTTP 401 (private).
  - **Other 2022+ campaigns** may share Scranton's layout. That makes P3 matter beyond one city.
