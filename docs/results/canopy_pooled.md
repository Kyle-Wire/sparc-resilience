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

The batch driver over the three cities took 3.6 min, because every lab already existed. The labs themselves
took:

| City | Idealised lab | Each real-route lab (36 reps) |
|---|---|---|
| Providence | 3 min | about 3 min |
| Burlington | 4.4 min | about 6 min |
| Albuquerque | 27 min | 21–26 min |

## What can be pooled today

- **Only one city.** Providence's morning, evening and night runs pass. That is no basis for a cross-city
  model, so the random-effects meta-analysis (`pool.py`) has nothing to combine yet.
- **Albuquerque fails on precision.** Its SE is about 0.19 °F per run: the city has almost no street canopy
  and noisy air.
- **Burlington fails on scoring.** Its real-route verdicts fail only through the `own_only` world, and in
  the opposite direction to its idealised lab. Both trace to the lab computing its truth at random paved
  cells, not at the streets campaigns drive: proposal P2
  (`docs/agent/proposals/P2_lab_truth_at_sampled_cells.md`), awaiting approval.
  - **If P2 is approved,** Burlington's real-route labs and Providence's are re-run under the new scoring, at
    36 replicates and with the same verdict rules, and this table is updated whatever the result.
- **The block-scale readings agree, pooled or not.** Providence's and Burlington's day-averaged intervals
  exclude block-scale cooling stronger than 0.07 and 0.05 °F per +10 pp within 100 m. Albuquerque's interval
  is too wide to say.

## Next

- **The second dimension of Phase 3:** more cities through `run_canopy_cities.py`. Each city costs 15 min
  (small, like Burlington) to 2 h (large, like Albuquerque) of lab time on 4 cores.
- **The two other single-tile northern cities whose OSF listings show traverses:**
  - Seattle `mz79p`: a 2020 "All Data" bundle.
  - Scranton `8cnq4`: 2023 traverses.
- **Portland `e7js9`:** its OSF listing returned HTTP 401 (private).
