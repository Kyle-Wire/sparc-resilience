# Canopy effect from the traverses: Seattle, WA (2020-07-27)

The fourth city: a second northern, leafy city, run end to end by the Phase 3 batch driver
(`scripts/run_canopy_cities.py seattle_wa`). Run by the `sparc-steward` agent on 2026-10-24. Outputs (not
committed) are in `output/core/seattle_wa/identify/`.

**In short.**
- **No block-scale effect is detected.** Pooled over the three runs, +10 pp canopy within 100 m of a street
  point changes its air by −0.05 °F (95% CI −0.12 to +0.03).
- **Wider readings are negative.** Within 300 m: −0.12 °F (−0.23 to −0.02). They set a validated city-wide
  floor of 0.03 °F of cooling.
- **Not pooled.** The lab replayed every real run and rates the block-scale design **Conservative**: in all
  three runs it misses only the `own_only` world, recovering 70–73% of it. Under the owner's rule Seattle's
  numbers are reported, not pooled.
- **The same pattern as Burlington.** The vans drove streets with 31% canopy, while the lab computes its truth
  at paved cells with 7.7%. That supports proposal P2.

## The city as built (unattended, inside the batch driver)

| Item | Value |
|---|---|
| OSF project | `mz79p`: `All Data_CHW_Seattle KingCounty_093020.zip`, the 2020 CAPA layout. No reader change was needed. |
| Campaign | 2020-07-27, from the traverse clock: 06–07, 15–16 and 19–20 PDT. 110,107 readings from 15 sensors per run. 37–41% fall in the study area: the King County campaign covers more than the city's pilot bbox. |
| Grid | 30 m UTM (EPSG:32610), 391,134 fishnet cells, of which 266,400 lie on CAPA's afternoon map |
| Predictors | WorldCover N45W123, Copernicus DEM N47 W123, Sentinel-2 July 2020 (6 scenes, 10TET). Open canopy averages 46%, impervious 49%. |
| Forcing | ERA5, 15–16 PDT: SW↓ 767 W/m², clear-sky index 1.00. Wind from KBFI (`72793524234`, Boeing Field, 2.7 km): from 330° at 1.5 m/s. |
| Time | setup 4 min; idealised lab about 12 min; each real-route lab (36 reps) about 15–20 min; kmlab about 5 min |

Per run on the grid: 4,787, 5,177 and 5,261 cell visits, on 54, 55 and 52 passes by 8 sensors. Canopy at the
visited cells averages 31%.

## The lab

**Idealised campaigns** (12 replicates):
- street_100 is **Conservative**: clean with no effect (−0.011 °F, 0%), but coverage is 75% in `additive`
  (−0.29 vs −0.24 °F).
- Within 300 m and within 1 km are also Conservative.
- Both floors are valid.

**Real routes** (`lab --routes`, 36 replicates per world, every run):

| Run | Within 100 m | Null: mean, false positives, sd / mean SE | Misses | Within 300 m |
|---|---|---|---|---|
| Morning | **Conservative** | −0.001 °F, 6%, 0.075 / 0.078 | `own_only` (70%) | Conservative |
| Afternoon | **Conservative** | −0.010 °F, 3%, 0.074 / 0.076 | `own_only` (73%) | **Trustworthy** |
| Evening | **Conservative** | −0.015 °F, 3%, 0.069 / 0.078 | `own_only` (72%) | Conservative |

- **The design is calibrated.** It is clean with no effect, its standard errors are right (spread ≈ SE), and
  it recovers `additive`, `physics`, `coarse_scale` and `confounded` in every run.
- **The one miss is the world where all cooling sits in the tree's own cell.** The design measures that
  effect at the streets the vans drove, with 31% canopy, where the next +10 pp cools less. It is scored
  against the effect at random paved cells, with 7.7% canopy, where it cools more.
- **This is the scoring mismatch P2 describes, not a fault of the design.** It is the same as in Burlington.
  Providence passes because its two populations agree (21% against 19–21%).

## Real traverses

| Run | Within 100 m (95% CI) | SE | Within 300 m (95% CI) |
|---|---|---|---|
| Morning 06–07 | −0.072 °F (−0.165 to +0.021) | 0.048 | −0.091 °F (−0.241 to +0.059) |
| Afternoon 15–16 | −0.034 °F (−0.153 to +0.086) | 0.061 | −0.102 °F (−0.254 to +0.051) |
| Evening 19–20 | −0.033 °F (−0.137 to +0.070) | 0.053 | **−0.178 °F (−0.317 to −0.040)** |
| **All runs pooled** | **−0.045 °F (−0.122 to +0.033)** | 0.040 | **−0.124 °F (−0.231 to −0.017)** |

- **City-wide floor** (from the 300 m reading, validated in the idealised lab): at least **0.03 °F** of
  cooling per +10 pp.
- **Within 1 km:** −0.23 °F (−0.53 to +0.07), rated Conservative. It is not used.
- **The pattern is like Providence's:**
  - no detectable block-scale effect;
  - a negative 300 m reading that is strongest in the evening (Providence −0.17 °F, Seattle −0.18 °F within
    300 m).
- **Kilometre scale:**
  - **Winds:** morning from 130° at 0.8 m/s, afternoon from 330° at 1.5 m/s, evening from 310° at 2.9 m/s.
    They span 180°, the widest yet.
  - **Estimate:** upwind − downwind −0.22 °F (−0.38 to −0.07), rotation p = 0.11, one-sided p = 0.06.
  - **`kmlab` under these winds:** no false direction flags, but 0% power for real advection, and "finds
    cooling" in 17% of `own_only` campaigns. A one-sided p of 0.06 is therefore no evidence. **Underpowered;
    not detected.**

## Reproduce

```
python scripts/run_canopy_cities.py seattle_wa
```
