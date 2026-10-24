# Canopy effect from the traverses: Burlington, VT (2020-08-10)

The third city, chosen as a northern, leafy counterpart to Providence at the owner's request (2026-10-24).
Run by the `sparc-steward` agent. Outputs (not committed) are in `output/core/burlington_vt/identify/`.

**In short.**
- **The streets the vans drove are leafy.** Canopy averages 41% at the visited cells, and two thirds of
  those cells have some, against 2.4% in Albuquerque.
- **No block-scale canopy effect is detected, and this time the null is tight.** Pooled over the three runs,
  +10 pp canopy within 100 m of a street point changes its air by −0.01 °F (95% CI −0.05 to +0.02).
  An effect larger than 0.05 °F per +10 pp is ruled out.
- **The lab's verdicts are not good enough for the pool.**
  - Idealised campaigns on Burlington's layout rate the design **Direction only**: it overstates the effect.
  - Burlington's real routes rate it **Conservative** (afternoon and evening) or **Not trustworthy**
    (morning). It understates the effect.
  - Under the owner's rule, Burlington's numbers are reported, not pooled.
- **The two failures point in opposite directions, and both trace largely to where the lab computes its
  "truth".** That is the subject of proposal P2 (`docs/agent/proposals/P2_lab_truth_at_sampled_cells.md`),
  which awaits approval.

## The city as built (`scripts/setup_core_city.py burlington_vt`: first try, unattended, 1 min 55 s)

| Item | Value |
|---|---|
| OSF project | `tezb9`: `All Data_Heat Watch Burlington_111820.zip`, the same 2020 CAPA layout as Providence. No reader change was needed. The bicycle traverses in the bundle carry only normalised temperatures and are skipped with that reason. |
| Campaign | 2020-08-10, from the traverse clock: morning 06:00–07:00, afternoon 15:00–16:00, evening 19:00–20:00 EDT. The pilot file's ERA5-matched 2020-07-25 is wrong. |
| Readings | 51,910 (18,877 / 16,136 / 16,897) from sensors CAPA1029–1035; 46–59% land on the grid |
| Grid | 30 m UTM (EPSG:32618) over the pilot bbox: 148,256 cells, of which 84,571 lie on CAPA's afternoon map. Only 23% of cells are paved (impervious > 40%). |
| Predictors | WorldCover N42W075, Copernicus DEM N44 W074, Sentinel-2 August 2020 (2 scenes, tile 18TXQ). Open canopy averages 58% city-wide (median 75%), impervious 21%. |
| Forcing | ERA5, 15–16 EDT: SW↓ 615 W/m², clear-sky index 0.89. KBTV (`72617014742`) reports the afternoon wind as variable, so ERA5's wind (from 239° at 1.1 m/s) is used. |
| Planner layers | HRSL: 75,165 residents |

**Per run on the grid**

| Run | Cell visits / passes / sensors | Canopy at visited cells | Along-pass difference variance, about 30 m / 300 m |
|---|---|---|---|
| Morning | 4,063 / 198 / 6 | 40% | 0.004 / 0.058 °F² |
| Afternoon | 3,383 / 204 / 5 | 41% | 0.018 / 0.301 °F² |
| Evening | 2,879 / 162 / 5 | 40% | 0.010 / 0.232 °F² |

Street air is as smooth as Providence's (0.004–0.018 °F² at 30 m) and far smoother than Albuquerque's.

## The lab

### Idealised campaigns on Burlington's layout (12 replicates × 6 worlds, 4.4 min)

| Design | Verdict | Why |
|---|---|---|
| Street differences, within 100 m | **Direction only** | Clean with no effect (−0.043 °F, 8%), but overstates `additive` (−0.44 vs −0.24 °F) and `confounded` (−0.47 vs −0.24), and `own_only` by 24% |
| Within 300 m | Not trustworthy | |
| Within 1 km | Direction only | |
| Floors (100 m, 300 m) | **not valid** (hold in 75% and 58% of campaigns) | |

### On the real routes (`lab --routes`, 36 replicates × 6 worlds per run, about 6 min each)

| Run | Within 100 m | Null: mean, false positives | Misses | Floor (100 m) |
|---|---|---|---|---|
| Morning | **Not trustworthy** | +0.010 °F, 14% | `own_only`: −0.33 vs −0.52 °F (63%) | valid |
| Afternoon | **Conservative** (a lower bound) | −0.004 °F, 3% | `own_only` (64%) | not valid |
| Evening | **Conservative** | +0.013 °F, 6% | `own_only` (68%) | valid |

Within 300 m is Conservative in all three runs. In every run the design recovers `additive`, `physics`,
`coarse_scale` and `confounded`. It misses only `own_only`, where all cooling sits in the tree's own cell.

### Why the two labs disagree

The lab computes each world's truth at 80 *random paved cells*. In Burlington those cells are not the
streets the campaigns drive:

- **Simulated routes** run through the fragmented paved network's denser core. There, canopy is 10.5%
  against 14.2% at random paved cells. With a saturating canopy response the true effect there is larger:
  −0.29 against −0.24 °F in `additive`, −0.57 against −0.52 °F in `own_only`. The design measures the
  larger effect and is scored as overstating. This explains about a third of the gap; the rest is not yet
  explained.
- **Real routes** run along leafy residential streets: canopy is 41% at the visited cells. There the
  `own_only` effect of the next +10 pp is much smaller than at random paved cells, and the design is scored
  as understating.
- **In Providence** the two populations agree (−0.165 °F either way), which is why its verdicts were not
  affected.

Proposal P2 would score the design against the truth at the cells campaigns actually drive. It is written
up for approval and not built. The verdicts above stand until the owner decides.

## Real traverses

| Run | Within 100 m (95% CI) | SE | Within 300 m (95% CI) |
|---|---|---|---|
| Morning 06–07 | +0.005 °F (−0.027 to +0.037) | 0.016 | −0.007 °F (−0.044 to +0.030) |
| Afternoon 15–16 | −0.017 °F (−0.098 to +0.064) | 0.041 | +0.010 °F (−0.078 to +0.098) |
| Evening 19–20 | −0.044 °F (−0.115 to +0.028) | 0.037 | −0.087 °F (−0.181 to +0.008) |
| **All three runs pooled** | **−0.012 °F (−0.047 to +0.022)** | 0.017 | −0.020 °F (−0.067 to +0.026) |

- **Block scale: a tight null.** It is the most precise block-scale reading of the three cities. It
  excludes cooling stronger than 0.05 °F per +10 pp averaged over the day, and stronger than 0.10 °F in the
  afternoon.
- **City-wide floor:** none. On Burlington's layout the idealised lab did not validate either floor.
- **Kilometre scale:**
  - **Winds:** morning from 201° at 1.0 m/s and afternoon from 239° at 1.1 m/s, both ERA5 because KBTV was
    calm or variable; evening from 315° at 5.1 m/s (KBTV).
  - **Estimate:** +0.02 °F upwind − downwind (+0.00 to +0.04), rotation p = 0.28. Not detected: the sign is
    opposite to advected cooling.
  - **`kmlab --winds 201:1.0 239:1.1 315:5.1`:** no false alarms, 8% power. **Underpowered.**

## What this adds

- **Setup generalises.** A third city, in a third CAPA layout year (2020 bundle with a bicycle side
  campaign), built end to end unattended on the first try.
- **Two leafy northern cities agree on the evidence.**

  | Reading (+10 pp canopy) | Providence | Burlington |
  |---|---|---|
  | Within 100 m, all runs pooled | −0.03 °F (−0.07 to +0.02) | −0.01 °F (−0.05 to +0.02) |
  | Afternoon, within 100 m | +0.02 °F (−0.13 to +0.17) | −0.02 °F (−0.10 to +0.06) |

  Both intervals include zero. Providence's evening run (−0.08 °F) is the only measured block-scale cooling
  so far.
- **The lab's scoring needs the change P2 proposes** before a city's real-route verdict can decide pool entry
  fairly. Today a correct design can fail in a leafy city because its truth is computed on the wrong streets.

## After P2 (owner decision 7, 2026-10-24)

The labs now score street designs against the truth at the streets the campaign drives. On Burlington's real
routes (36 replicates; the stored replicates rescored, which equals a re-run):

| Run | street_100 before → after | street_300 before → after |
|---|---|---|
| Morning | Not trustworthy → Not trustworthy (14% false positives with no effect) | Conservative → Trustworthy |
| Afternoon | Conservative → **Trustworthy** | Conservative → Trustworthy |
| Evening | Conservative → **Trustworthy** | Conservative → Trustworthy |

- **Pooled:** the afternoon and evening runs (owner decision 2).
- **The idealised lab is still Direction only:** it overstates `additive` by −0.13 °F against the route-cell
  truth. The knot change that closed this gap in Burlington harmed Seattle and was rejected (agent decision
  A3). The gap stays open. See `docs/results/canopy_pooled.md`.

Reproduce:

```
python scripts/setup_core_city.py burlington_vt
python -m sparc.core.identify lab      -p configs/core_burlington_vt.yml --workers 4
python -m sparc.core.identify lab      -p configs/core_burlington_vt.yml --workers 4 --reps 36 --routes output/core/burlington_vt/identify/raw --window midday   # and morning, evening
python -m sparc.core.identify kmlab    -p configs/core_burlington_vt.yml --workers 4 --winds 201:1.0 239:1.1 315:5.1
python -m sparc.core.identify estimate -p configs/core_burlington_vt.yml --traverses output/core/burlington_vt/identify/raw --window all
```
