# P3: read traverse layers that carry no timestamps or sensor ids (Scranton 2023 layout)

Status: **awaiting owner approval** (listed in `docs/agent/STATUS.md`). Nothing is built.

## The finding

Scranton & Wilkes-Barre (OSF `8cnq4`, `traverses_chw_scranton_092023.zip`) publishes its traverses in a
layout none of the earlier cities used:

- three shapefiles, `am_trav`, `af_trav` and `pm_trav` (58,094 points in all);
- attributes `t_f`, `rh` and `hi_f` only: **no time, no date, no sensor or file column**.

`inspect` reads every point. The run windows come from the file names: am, af, pm. But the readings form no
runs and no passes, because `to_campaign` needs a time for each sample. `setup_core_city.py scranton_pa`
stops at its first step and now says why ("the traverse files carry no timestamps"). It is recorded in the
batch status as "failed at setup".

The rows look like drive order. Consecutive points are 11.8 m apart (median; 90th percentile 35 m), which
fits a 1 Hz logger at about 40 km/h. Only 1.6% of steps jump more than 200 m, which is where a pass would
break, or one vehicle's track end and the next begin.

## The proposal

For a layer with no time column, take the row order as the time order and assume a 1 Hz logger:

- `t_s` is the row index in seconds from the start of the file;
- the run is the file's window (`am`/`af`/`pm`), on the campaign date taken from the zip name (`092023`)
  where possible;
- the vehicle is unknown, so the whole file counts as one vehicle. Passes still break at jumps over 200 m and
  at gaps.

Every such run would be flagged "time from row order" in `inspect` and in the estimate.

**What the street design still controls, and what it loses:**

- **Kept:** within-pass differences still cancel a vehicle's sensor offset and the slow drift. A pass never
  spans two vehicles unless one track ends within 200 m of where the next begins.
- **Weakened:** the Δtime control becomes "Δrow". It is exact only if the logger ran at 1 Hz with no
  dropped rows.
- **Lost:** per-vehicle structure for the kilometre design and vehicle-offset diagnostics.

**Validation before any Scranton number is used.** The real-route lab replays the run as read, so it would
test the design on these pseudo-passes. The 36-replicate pool-entry rule applies unchanged.

## Cost

About 20 lines in `traverses._parse_frame` / `assign_runs`, a regression test with a synthetic layer of this
layout, and a Scranton batch run. Scranton's grid is about 250k cells, so the labs take about 1 h on 4 cores.

## Alternative

Treat timestamp-less layouts as unreadable for identification, and pick other cities. Seattle (`mz79p`, a 2020
bundle in Providence's layout) is being run instead in the meantime.
