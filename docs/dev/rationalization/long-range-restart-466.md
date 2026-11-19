# Long-range restart with `"BFGS4"` as the default: measured, removed (#466)

Issue [#466](https://github.com/libKriging/libKriging/issues/466) (part of the 2.0 tracking issue
[#464](https://github.com/libKriging/libKriging/issues/464)). The long-range restart (#433,
`docs/dev/CholJitter.md`) restarted a single-start (`"BFGS"`) noise-free LL fit whose solution had
max_j theta_j / maxdX_j > c (`Optim::set_long_range_restart`, `LK_LONG_RANGE_RESTART`, c = 2), at
most m times (`Optim::set_max_long_range_restart`, `LK_MAX_LONG_RANGE_RESTART`, m = 2). It was
added when `"BFGS"` was the default; T3 (#443) made `"BFGS4"` the default and kept it.

## Status

Done. **Decision: removed** (2.0). Both settings, both environment variables and the criterion in
`Kriging::fit_impl` / `RestartPolicy` are gone; `"BFGS2"` replaces a restarted single start.

## Where the restart acted (a correction of the premise of #466)

#466 states that the restart "now only acts when the user explicitly asks for a single start".
That is incomplete: **every nugget fit with a `"BFGS*"` optimizer, the default `"BFGS4"` included**,
starts with a noise-free `"BFGS"` LL sub-fit (the nugget warm start of #383), and the restart acted
there. Checked on a smoke run (5 sets, n 100, all configurations) and on the whole run below:

| fit | restart |
|---|---|
| `"BFGS"`, noise-free, LL | on (the case it was designed for) |
| `"BFGS"`, noise-free, LOO / LMP | off by construction: bit-identical with `LK_LONG_RANGE_RESTART=0` (480 / 480 sets each) |
| `"BFGSk"`, k > 1, noise-free, any objective | off by construction: bit-identical (960 + 480 + 480 / 960 + 480 + 480 sets) |
| any `"BFGS*"`, nugget, LL / LMP | **on, through the warm start**: results differ on 12-38 % of the sets |

## Protocol

T7 base (`bench/fit-protocol`, `docs/dev/rationalization/T7-benchmark-base.md`), profile
`reference`: gauss / matern5_2; noise-free (LL, LOO, LMP) and estimated nugget (LL, LMP); `sin4`,
`hartmann6`, `branin4` (2 inactive inputs of 4), `sinhalf8` (4 inactive of 8); n 100-200 for every
objective, 500 for the LL objectives, 1000 for noise-free LL; 30 data sets per configuration, 3120
data sets, 16 080 fits, no failure. One run on master `6e008b2` (with the restart), all variants
in the same output directory so that the best of each set includes `BFGS10`:

- `BFGS`, `BFGS2`, `BFGS4`, `BFGS10` (restart on, c = 2, m = 2);
- `BFGS` with `LK_LONG_RANGE_RESTART=0` where the restart can act (noise-free LL, nugget LL / LMP);
- `BFGS2`, `BFGS4` with `LK_LONG_RANGE_RESTART=0` on nugget LL / LMP (the warm start).

Machine (`bench/fit-protocol/results/long-range-466-machine.txt`): 4-vCPU Intel Xeon VM @ 2.1 GHz,
OpenBLAS 0.3.26 pthreads, gcc 13.3, Release; the chunks one at a time, idle machine, 3.7 h.
Miss: objective more than 10 below the best variant of the set (LOO: (n / 2) log(e / e_best) > 10);
Wilson 95 % intervals; paired comparisons on the same sets, exact two-sided sign test on the
discordant sets (only A misses : only B misses). Times are summed over the sets (wall: 4 cores,
parallel starts; CPU: all threads). RMSE ratio: test RMSE of B / A, geometric mean.

Reproduction:

```sh
python3 bench/fit-protocol/long_range_466.py bench/fit-protocol/results/long-range-466.csv.gz --by noise,objective,n
python3 bench/fit-protocol/summarize.py bench/fit-protocol/results/long-range-466.csv.gz --by noise,objective,n
# all tables (by n, function, kernel): bench/fit-protocol/results/long-range-466.md
```

## Measurements

### 1. Single start (`"BFGS"`): restart on vs off

| noise | objective | n | sets | miss on % [95 %] | miss off % [95 %] | only on : only off | p | wall on / off | CPU on / off | RMSE off / on |
|---|---|---|---|---|---|---|---|---|---|---|
| none | LL | 100 | 240 | 10 [7, 15] | 19 [14, 24] | 0 : 20 | 2e-06 | 2.09 | 1.94 | 1.071 |
| none | LL | 200 | 240 | 18 [13, 23] | 29 [24, 35] | 0 : 28 | 8e-09 | 2.47 | 2.43 | 1.321 |
| none | LL | 500 | 240 | 14 [10, 19] | 23 [18, 29] | 0 : 22 | 5e-07 | 2.05 | 2.05 | 1.123 |
| none | LL | 1000 | 240 | 13 [9, 18] | 31 [25, 37] | 0 : 43 | 2e-13 | 2.07 | 2.07 | 1.372 |
| none | LOO, LMP | 100, 200 | 960 | (identical) | | 0 : 0 | | 1.00 | 1.00 | 1 |
| nugget | LL | 100 | 240 | 9 [6, 13] | 15 [11, 20] | 0 : 15 | 6e-05 | 1.72 | 1.62 | 1.038 |
| nugget | LL | 200 | 240 | 20 [15, 25] | 22 [17, 27] | 1 : 6 | 0.12 | 1.78 | 1.76 | 0.999 |
| nugget | LL | 500 | 240 | 24 [19, 30] | 29 [24, 35] | 0 : 12 | 5e-04 | 1.37 | 1.37 | 1.044 |
| nugget | LMP | 100 | 240 | 11 [8, 16] | 15 [11, 21] | 1 : 11 | 0.006 | 1.56 | 1.48 | 1.018 |
| nugget | LMP | 200 | 240 | 25 [20, 31] | 29 [23, 35] | 1 : 10 | 0.012 | 1.65 | 1.64 | 1.041 |

By function, noise-free LL (n 100-1000 pooled, 240 sets): `branin4` 1 vs 6 % (x 3.2 the time),
`hartmann6` 8 vs 22 % (x 1.1), `sin4` 11 vs 30 % (x 2.9), `sinhalf8` 35 vs 43 % (x 2.2).

**The gain on a single start is not within noise**: noise-free LL misses 26 % → 14 % (0 : 113
discordant sets, p 2e-34), at x 2.07 the wall and CPU time; nugget LL / LMP 22 → 18 % (3 : 54)
at x 1.4-1.6. With 4 of 8 inputs inactive (`sinhalf8`) it still leaves 35 % misses.

### 2. Default (`"BFGS4"`): restart on vs off (nugget fits, through the warm start)

| noise | objective | n | sets | miss on % | miss off % | only on : only off | p | wall on / off | CPU on / off | RMSE off / on |
|---|---|---|---|---|---|---|---|---|---|---|
| nugget | LL | 100 | 240 | 3 [2, 6] | 4 [2, 7] | 0 : 1 | 1 | 1.31 | 1.27 | 1.002 |
| nugget | LL | 200 | 240 | 10 [7, 14] | 10 [6, 14] | 1 : 0 | 1 | 1.18 | 1.21 | 1.001 |
| nugget | LL | 500 | 240 | 5 [3, 8] | 5 [3, 9] | 0 : 2 | 0.5 | 1.26 | 1.32 | 1.002 |
| nugget | LMP | 100 | 240 | 3 [1, 6] | 3 [2, 6] | 1 : 2 | 1 | 1.31 | 1.26 | 1.006 |
| nugget | LMP | 200 | 240 | 11 [8, 15] | 11 [8, 15] | 0 : 0 | 1 | 1.15 | 1.17 | 1.002 |
| none | any | any | 1920 | (identical) | | 0 : 0 | | 1.00 | 1.00 | 1 |

**On the default path the restart costs x 1.15-1.31 the time of a nugget fit for no measurable
gain** (2 : 5 discordant sets of 1200, p 0.45; RMSE ratio 1.001-1.006). `BFGS2` alike: 2 : 21 on
1200 nugget sets (the off variant misses more, 1-2 points), x 1.03-1.36 the time.

### 3. Equal cost: `"BFGS"` + restart vs `"BFGS2"`

| noise | objective | n | sets | miss BFGS+r % | miss BFGS2 % | only BFGS+r : only BFGS2 | p | wall BFGS+r / BFGS2 | CPU BFGS+r / BFGS2 | RMSE BFGS2 / BFGS+r |
|---|---|---|---|---|---|---|---|---|---|---|
| none | LL | 100 | 240 | 10 [7, 15] | 14 [10, 19] | 13 : 21 | 0.23 | 1.65 | 1.54 | 1.030 |
| none | LL | 200 | 240 | 18 [13, 23] | 19 [14, 24] | 21 : 24 | 0.77 | 1.37 | 1.49 | 0.941 |
| none | LL | 500 | 240 | 14 [10, 19] | 8 [5, 13] | 23 : 9 | 0.02 | 2.01 | 2.25 | 0.928 |
| none | LL | 1000 | 240 | 13 [9, 18] | 10 [6, 14] | 19 : 11 | 0.2 | 1.29 | 1.57 | 0.972 |
| none | LMP | 100-200 | 480 | 17 [14, 21] | 7 [5, 9] | 59 : 10 | 1e-09 | 1.04 | 1.08 | 0.860 |
| none | LOO | 100-200 | 480 | 37 [33, 41] | 24 [21, 28] | 91 : 29 | 1e-08 | 0.83 | 1.12 | 0.759 |
| nugget | LL | 100-500 | 720 | 18 [15, 20] | 11 [9, 13] | 49 : 0 | 4e-15 | 1.01 | 1.05 | 0.966 |
| nugget | LMP | 100-200 | 480 | 18 [15, 22] | 12 [10, 16] | 27 : 0 | 2e-08 | 0.95 | 1.11 | 0.965 |

On noise-free LL, the only case the restart was designed for: **same miss rate** (14 % [12, 16]
against 13 % [11, 15], 76 : 65 discordant sets, p 0.4), RMSE ratio 0.97, and the restarted single
start costs **x 1.29-2.01 the wall time and x 1.49-2.25 the CPU time of `"BFGS2"`** at every n. By
function: `hartmann6` 8 vs 2 % (19 : 5, p 0.007, for BFGS2), `branin4` 1 vs 0 %, `sinhalf8` 35 vs
33 %, `sin4` 11 vs 15 % (14 : 24, p 0.14, the one family where the restart has fewer misses, not
significant, at x 1.5 the wall time). On the nugget fits `"BFGS2"` misses strictly less (76 : 0
discordant sets) at equal wall time (x 0.95-1.01). Everywhere, `"BFGS4"` dominates both (19 % →
6 % against `BFGS`+restart, 447 : 23).

Threshold 1 and 50 instead of 10 (all noise-free LL sets): `BFGS`+restart vs `BFGS2` 20 vs 18 %
(88 : 68) and 8 vs 6 % (49 : 36); default `BFGS4` on vs off (nugget) 3 : 9 and 0 : 1 discordant
sets. Same conclusions.

## Decision

**Remove** (the first criterion of #466): `"BFGS2"` dominates the restarted single start at equal
or lower cost, on every objective and noise model, and the restart brings nothing measurable on
the default path, where it adds 15-31 % to the time of every nugget fit.

- Removed: `Optim::set_long_range_restart` / `get_long_range_restart`,
  `Optim::set_max_long_range_restart` / `get_max_long_range_restart`, `LK_LONG_RANGE_RESTART`,
  `LK_MAX_LONG_RANGE_RESTART` (now ignored), the long-range criterion of `RestartPolicy` and of
  `Kriging::fit_impl` (about 20 lines, the restart loop otherwise unchanged).
- Changed results: an explicit `"BFGS"` noise-free LL fit returns what it returned with
  `LK_LONG_RANGE_RESTART=0` (more misses: 14 → 26 % here) in half the time; every nugget fit
  (default included) takes 13-24 % less time with the same misses (`BFGS4`); 2-6 points more misses
  for an explicit `"BFGS"` nugget fit. Migration: `"BFGS2"` instead of `"BFGS"` to keep (or better) the quality
  of a restarted single start at a lower cost; `"BFGS4"` (default) or more for a reliable optimum.
- Not done: exposing the settings in the bindings (#468 does not need to handle them).

## Limits (adversarial review)

- **The removal makes the explicit single start worse**, measurably (noise-free LL 14 → 26 %
  misses, RMSE geometric mean x 1.07-1.37). The decision rests on the existence of a better
  option at the same cost, not on the restart being useless: a user who keeps `"BFGS"` and does
  not read the CHANGELOG loses quality. The CHANGELOG entry says it.
- **4 cores**: the wall-time comparison favours `"BFGS2"` (its two starts run in parallel). The
  CPU time (all threads) is the closer proxy for one core: `"BFGS2"` is still x 1.5-2.3 cheaper on
  noise-free LL, but the CPU of a 4-core run includes thread overheads that a 1-core run does not
  have. T3 measured on one core: restart x 2.0-2.4 a single start (its `BFGS-nolr` ratios
  0.41-0.54), a 2-start fit about x 2 by construction: **on one core the two cost about the same**,
  and the miss rates are equal: the equal-cost criterion still holds, the margin is gone.
- **Miss reference = best of the variants** (BFGS10 included); the differences of `BFGS`+restart
  against `BFGS2` are paired on the same sets, free of that bias.
- **Sets are not independent** across kernels and n (same seeds); the pooled p-values overstate
  the precision. Per (noise, objective, n) cell the noise-free LL comparison of section 3 is
  significant once (n 500, for `BFGS2`).
- **Not measured**: n > 1000, LOO / LMP at n 500-1000 and nugget fits at n 1000 (the restart is off
  for noise-free LOO / LMP anyway), matern3_2 / exp, other designs and real data, other BLAS (MKL,
  Accelerate), macOS / Windows. The restart could have a larger gain at n > 1000 (its single-start
  miss rate did not fall with n here: 13 % at n 1000), but `"BFGS2"` keeps its cost ratio there.
- **The constants 2 / 2** were not recalibrated: a different c could change section 1, not
  section 2 (the warm start of a nugget fit gained nothing with c = 2, which fires on most fits).
- The R / Octave / Julia binding tests were not run on the machine of the measurements (not
  installed): left to the CI; the C++ suite and the Python binding tests pass.
