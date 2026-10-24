# P2: score the lab's street designs against the streets the campaign drives

Status: **awaiting owner approval** (listed in `docs/agent/STATUS.md`). Nothing below is built. A trial
implementation was written and reverted, uncommitted, on 2026-10-24, once owner decision 5 (proposals wait
for approval) was read.

## The problem

The street designs (street_100, street_300, reach_1000) estimate canopy's effect at the street points a
campaign samples. The lab scores them against a truth computed somewhere else. `validate.truth_for` takes 80
random *paved* cells (impervious > 40%) as "street points". The simulated campaign drives
`campaign.route_segments`, and the real-route lab drives a run's real cells.

Where the two populations match, this makes no difference. Where they differ, the lab scores a design
against an effect it does not measure. Under a saturating (nonlinear) canopy response, the true effect
differs between low-canopy and high-canopy streets.

Evidence: the true effect of +10 pp within 100 m, for the same planted world, at two sets of street points
(diagnostic run 2026-10-24):

| Layout | Paved cells / simulated route cells | Canopy at random paved cells / route cells | `additive` truth, paved / route | `own_only` truth, paved / route |
|---|---|---|---|---|
| Providence (brown4) | 41,559 / 7,370 | 21.2% / 20.8% | −0.165 / −0.165 | −0.275 / −0.271 |
| Burlington (open data) | 19,210 / 2,190 | 14.2% / 10.5% | −0.239 / −0.294 | −0.521 / −0.570 |

- **Providence:** the populations agree, so the documented verdicts are unaffected.
- **Burlington:** the paved network is fragmented, since only 23% of cells are paved. The simulated routes
  run through the denser, less leafy core. The idealised lab rates street_100 **Direction only**: it
  "overstates" `additive` (−0.44 against −0.24) and `confounded`.
  - About a third of that gap is the mismatch in where the truth is computed.
  - The rest is unexplained. A second suspect is the canopy knots (15%, 40%), which were set on Providence.
- **The real-route labs** show the mismatch in the other direction. On Burlington's real routes the design
  "understates" `own_only`: it recovers 63–68% of it in all three runs, at 36 replicates, and recovers the
  other four worlds. Real routes run along leafier residential streets than random paved cells: canopy is 41%
  at the visited cells against 14% at random paved cells. The truth there is smaller.

**Consequence under the pool-entry rule.** No Burlington run can enter the pool under the current scoring,
although its block-scale reading is the tightest of the three cities (−0.012 °F, −0.047 to +0.022). See
`docs/results/burlington_vt_canopy.md`.

## The change

- **`validate.street_population(layout, routes=None)`:** a real run's unique cells when `routes` is given;
  otherwise the cells of the simulated street grid, `route_segments` with a fixed seed.
- **`validate.truth_for(kind, gen, layout, street_pool=...)`:**
  - draws the 80 street cells from that pool;
  - caches per (world, layout, street population);
  - leaves the city-wide sample unchanged.
- **`replicate`** passes `street_population(layout, routes)`.
- **A regression test** checks that the street truth comes only from the pool, that the grid population is
  paved, and that each population gets its own cache entry.

The verdict rules, replicate counts and designs stay unchanged. Only the definition of "truth at street
points" changes, to the points the design actually measures.

## What it would do to recorded results

- **Providence:** verdicts are expected to stay as they are, since the populations agree. The idealised lab
  would be re-run (3 min) to confirm, and the real-route labs (36 reps, about 3 min each) re-run and
  reported.
- **Albuquerque and Burlington:** their labs would be re-run, and every verdict reported before and after.
  The owner's rules stay in force: 36 replicates and fixed verdict rules.
- **Pool-entry decisions** would rest on the new scoring. Approving P2 means accepting that change before
  seeing the re-run verdicts.

## Cost

About 30 lines in `validate.py`, plus a test. Re-running the labs takes about 3 min (Providence), 15 min
(Burlington) and 2 h (Albuquerque) on 4 cores.
