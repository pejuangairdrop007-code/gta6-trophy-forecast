# GTA VI trophy rarity forecast

Entry for make.fun `gta6-trophy-rarity-forecast-01`: forecast every GTA VI
(PS5) trophy's Exophase earn rate (%) at 2026-11-26T00:00:00Z, as five
quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95) per trophy.

Run it with `python3 main.py`; it reads every `*.in` case from
`$MAKEFUN_INPUT_DIR` (default `/input`) and writes one `*.out` forecast
per case to `$MAKEFUN_OUTPUT_DIR` (default `/output`). Python 3.13,
standard library only, no network, deterministic.

## Model

Exophase refreshes a game's page on its own schedule, so a case's history
is a staircase: long runs of identical reads (a frozen page) punctuated by
refreshes. The program first collapses consecutive identical reads into
one snapshot each.

**When there are at least three distinct snapshots** (the final case), the
median is a log-time trend: per trophy, weighted least squares of
`logit(rate)` on `ln(1 + hours since release)`, fitted over the last 36
hours of snapshots and recency-weighted. The slope is shrunk toward the
median slope of the trophy's pace class — story, strand, grind, or other,
read off its name and description — because one trophy's noisy steps say
little on their own. The extrapolated move from the last rate is capped at
3.5 logit units and damped by 0.75, because beyond the fitted window
earn-rate curves saturate and a log-time line overstates the rise.

**When the page has never moved** (the practice cases: one frozen
launch-night page), the median uses a dilution prior: the frozen page is
as of roughly the release, the tracked-player count grows about linearly
in `1 + hours since release`, and newcomers earn each trophy at a
discounted fraction (0.25) of its current rate, so rates fall toward the
next refresh. A trophy essentially every tracked player has (>95%) stays
near 100%.

**Spread.** The quantiles widen with the horizon honestly: the random-walk
term (0.5 logit per √day), the trend fit's own prediction error at the
target, and a 0.6 floor for model error the trend cannot see, combined in
quadrature — floored at 0.9 and capped at 2.8 logit units (1.6 for the
stale-page branch, where the next refresh's size is unknown). The house
baseline's narrow band scored about 1.51 on the practice because the
first refresh took every rate outside it; this spread is built to survive
a refresh it has not seen.

## Local checks

- Practice cases p00/p01: valid output (55 trophies, monotone quantiles),
  byte-identical across runs, ~0.07 s per run.
- A synthetic final-like case (two days of reads, refresh steps every
  4 h, tracked players 638 → ~45k, rates rising on saturating curves):
  pinball loss 0.270 vs the house baseline's 0.324, typical miss 2.0 vs
  6.2 points. On the practice cases the median falls to ~0.39× the last
  rate (the dilution a refresh brings) instead of staying flat like the
  baseline.
