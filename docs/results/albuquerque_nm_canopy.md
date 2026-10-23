# Canopy effect from the traverses: Albuquerque, NM (2021-07-09)

The second city of Phase 2 (`docs/agent/ROADMAP.md`). Run by the `sparc-steward` agent on 2026-10-23.
Outputs (not committed) are in `output/core/albuquerque_nm/identify/`.

**In short.** The second CAPA layout reads after two reader fixes. `scripts/setup_core_city.py` builds the
city from open data, and the idealised lab rates the block-scale design **Trustworthy** on Albuquerque's
layout.

On the real traverses no canopy effect is detected:

| Reading (+10 pp canopy) | Estimate | 95% CI |
|---|---|---|
| Within 100 m, both runs pooled | −0.17 °F | −0.45 to +0.12 |
| Within 100 m, afternoon run | −0.15 °F | −0.52 to +0.22 |

The intervals are about six times wider than Providence's. The city has almost no canopy along the routes:
2.4% on average at the visited cells, and only 8% of those cells have any. Its street air is also noisier
than Providence's.

The lab replayed both of Albuquerque's real runs (routes and noise). With 12 replicates per world, the design
rates **Not trustworthy** in both runs. Albuquerque's numbers are therefore reported, not used. This is a
finding about what one campaign in an arid, nearly treeless city can measure, not a failure of the reader.

## The city as built

| Item | Value |
|---|---|
| OSF project | `np4q9` (catalog `osf_node`). `fetch_capa_raw` took `traverses_chw_albuquerque_100421.zip` (2 s). |
| Campaign | 2021-07-09, from the traverse clock: morning 06:00–07:00 and afternoon 15:00–16:00 MDT. There is no evening run. `configs/multicity_pilot.yml` gives an ERA5-matched date of 2021-06-06, which is wrong by a month: the circularity flagged there is real. |
| Readings | 70,412 in total (28,536 morning, 41,876 afternoon), from 11 and 17 sensors. 1,423 are CAPA's −9999 no-data value and are now dropped. |
| Grid | 30 m UTM (EPSG:32613) over the pilot bbox, clipped to the traverses ± 1 km. That is 543,879 fishnet cells, of which 439,342 lie on CAPA's afternoon map (`aat_midday`, the target). 87% of readings land on it. |
| Predictors | WorldCover N33W108, Copernicus DEM N35 W107, Sentinel-2 July 2021 (7 scenes, tile 13SCU). Open canopy averages 5.9% city-wide, impervious 63.9%. |
| Forcing | ERA5 at the traverse centre, 15–16 MDT: SW↓ 810 W/m², clear-sky index 0.94. The nearest WBAN station, KABQ (`72365023050`, 8 km), reports the afternoon wind as variable (2.1 m/s, no direction), so ERA5's wind (from 244° at 0.5 m/s) is used. |
| Planner layers | HRSL: 593,564 residents |
| Setup time | about 4 min end to end (grid 47 s, forcing 47 s, features 100 s, layers 15 s) |

**Reader fixes this layout needed.** Each has a regression test in `tests/core/test_identify.py`.

- **Time column.** The time is in a shapefile field cut to `datetim`. Without the fix the readings were read
  but had no time, so they formed no runs.
- **Sensor ids.** Most logger files start with a bare number (`1023B_090721_…`, `1036_…`) beside `CAPA1021_…`.
  The sensor id is now taken from both forms.
- **No-data values.** `t_f` holds −9999 where the logger had no reading. Before the fix, one such value in a
  pass put the block-scale estimate at −45 °F per +10 pp. `parse_traverse_table` now treats −9999 as missing.

**Pipeline fixes this city needed.** Each has a regression test.

- **Station wind.** A station reporting only variable wind made the forcing wind NaN. The forcing now falls
  back to ERA5, and the estimate's per-run wind reports "variable" instead of passing NaN to the kilometre
  design.
- **TLS for GDAL.** GDAL 3.10 (rasterio 1.4 wheels) needs `GDAL_CURL_CA_BUNDLE`, not `CURL_CA_BUNDLE`, behind
  the sandbox's TLS-inspecting egress.

## The lab on Albuquerque's layout (idealised campaigns, 12 replicates × 6 worlds, 27 min on 4 cores)

| Design | Verdict | Null: mean, false positives |
|---|---|---|
| Street differences, within 100 m | **Trustworthy** (bias ≤ 0.024 °F in all five effect worlds) | −0.014 °F, 0% |
| Street differences, within 300 m | Conservative (a lower bound; understates `physics` and `confounded`) | −0.009 °F, 0% |
| Street differences, within 1 km | Not trustworthy | −0.008 °F, 25% |
| Wind signature, levels regressions, map designs | Not trustworthy | |

The 100 m and 300 m floors are both validated (they hold in ≥ 92% of campaigns in every world). This
reproduces the Providence verdict on a second, very different layout: arid, gridded, low canopy.

## Real traverses

| Run | Cell visits / passes / sensors | Within 100 m (95% CI) | Within 300 m (95% CI) |
|---|---|---|---|
| Morning 06–07 | 8,194 / 79 / 11 | −0.17 °F (−0.54 to +0.19) | −0.29 °F (−0.70 to +0.12) |
| Afternoon 15–16 | 13,722 / 159 / 17 | −0.15 °F (−0.52 to +0.22) | −0.24 °F (−0.63 to +0.14) |
| **Both runs pooled** (`--window all`) | 21,916 / 238 / 28 | **−0.17 °F (−0.45 to +0.12)** | −0.27 °F (−0.57 to +0.03) |
| Idealised lab verdict | | Trustworthy | Conservative |

- **City-wide floor** (from the 300 m reading): at least 0.01 °F pooled. The afternoon run alone sets none.
- **The 1 km reading** (−0.66 °F, −1.11 to −0.21) is Not trustworthy and is not used.
- **Precision.** The block-scale SE is 0.15 °F pooled and 0.19 °F per run, against 0.04–0.05 °F in the
  idealised lab on the same layout. Two things make it worse:
  - **little canopy contrast along the routes:** canopy averages 2.4% at the visited cells, and 92% of them
    have none;
  - **rough street air:** the difference variance between neighbouring visits is 0.14 °F² in the afternoon
    and 0.04 °F² in the morning, against 0.06 and 0.015 °F² in Providence.

### The lab on the real routes

`lab --routes <raw> --window <run>`, with 12 replicates per world (about 20 and 14 minutes):

| Run | Within 100 m | Null: mean, false positives, sd / mean SE | Within 300 m |
|---|---|---|---|
| Afternoon | **Not trustworthy** (null +0.125 °F; misses `coarse_scale` and `confounded`, bias −0.24 °F) | +0.125, 0%, 0.17 / 0.22 | Not trustworthy |
| Morning | **Not trustworthy** (25% false positives; misses `physics`, `coarse_scale`, `confounded`) | −0.010, 25%, 0.35 / 0.23 | Not trustworthy |

- **Size.** With an SE near 0.22 °F per replicate, effects of the size the worlds plant (0.15–0.26 °F) cannot
  be told apart, and the design's means drift by more than the 0.05 °F tolerance.
- **The morning SE is understated** (spread 0.35 against 0.23).
- **The replicates are not independent draws of the noise.** All twelve reuse the run's own along-pass noise,
  only shifted. Twelve replicates are therefore a weaker test than they look, and the morning's 3 of 12 false
  positives may partly reflect that. In Providence, going from 12 to 36 replicates changed several verdicts.
- **Conclusion:** on Albuquerque's campaign as driven, the block-scale design cannot be trusted for size or
  sign. The pooled −0.17 °F is reported, not used.

### Kilometre scale

- **Winds:** morning from 140° at 1.0 m/s (KABQ); afternoon from 244° at 0.5 m/s (ERA5, because KABQ reported
  variable wind).
- **Estimate:** +0.14 °F upwind − downwind (−0.11 to +0.39), rotation p = 0.17, on 6,034 cells seen in both
  runs.
- **`kmlab --winds 140:1.0 244:0.5`:** no false alarms; detects advection in 8% of advected campaigns.
  **Underpowered**: the day was nearly calm.

## What this adds

- **The path is general.**
  - `fetch_capa_raw` → `setup_core_city.py` → `inspect` → `lab` → `estimate` runs on a second CAPA layout
    and a different climate.
  - Five fixes were needed, three in the reader and two in the pipeline. Each is found by a test now.
- **Not every campaign can measure canopy at block scale.**
  - In a city with almost no street canopy, one campaign gives an interval of about ±0.3 °F per +10 pp.
  - Pooling across cities (Phase 3) should weight each city by its SE. Albuquerque's weight will be small.
  - The real-route lab is the check that says so per city. It should run for every city before its numbers
    enter the pool.

Reproduce:

```
pip install rasterio mgrs                       # not in requirements-core.txt
python scripts/setup_core_city.py albuquerque_nm
python -m sparc.core.identify inspect  -p configs/core_albuquerque_nm.yml --traverses output/core/albuquerque_nm/identify/raw
python -m sparc.core.identify lab      -p configs/core_albuquerque_nm.yml --workers 4
python -m sparc.core.identify lab      -p configs/core_albuquerque_nm.yml --workers 4 --routes output/core/albuquerque_nm/identify/raw --window midday   # and morning
python -m sparc.core.identify kmlab    -p configs/core_albuquerque_nm.yml --workers 4 --winds 140:1.0 244:0.5
python -m sparc.core.identify estimate -p configs/core_albuquerque_nm.yml --traverses output/core/albuquerque_nm/identify/raw --window all --wind midday=244:0.5
```
