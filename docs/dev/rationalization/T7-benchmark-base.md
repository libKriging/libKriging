# T7. Statistical benchmark base — work log

Task specification: `docs/dev/Rationalization.md`, section T7. Tools: `bench/fit-protocol/`
([README](../../../bench/fit-protocol/README.md)).

## Status

Done (driver, run script, summarizer, reference results). Used by T3.

## Protocol

- Driver `bench-fit-protocol` (`bench/bench-fit-protocol.cpp`, built by `bench/CMakeLists.txt`):
  for each configuration kernel x noise x objective x function x n and each data set, one
  `Kriging` fit per optimizer of `--optim` (constant trend, `normalize = false`), one CSV row
  per fit: objective at the fitted parameters, LL, theta, nugget ratio, test RMSE, wall and
  process CPU time (`getrusage`, all threads).
- Settings of a variant come from the environment (`LK_LONG_RANGE_RESTART`, `LK_CHOL_JITTER`,
  `LK_NUM_THREADS_PERCENT`, any `LK_*`), read by the library; their effective values are in
  the `settings` column and the variant label is `optimizer:environment`.
- Grid: gauss, matern5_2; noise-free (`none`) or estimated nugget on y + N(0, (0.05 sd(y))^2)
  (`nugget`); LL, LOO, LMP (LOO with a nugget is not supported by `Kriging`: skipped);
  `sin4` (4 active inputs), `hartmann6` (6 active), `branin4` (2 active, 2 inactive),
  `sinhalf8` (4 active, 4 inactive); n 100, 200, 500, 1000; data sets 0-29 (the data of
  `bench/thread-budget/jitter_n_study.cpp`, test set of 500 points, RMSE against the
  noise-free function over sd(y_test)).
- Miss: on each data set, the best objective over the variants run is the reference; a fit
  misses when its objective is more than 10 below it (LL, LMP), or, for LOO (mean squared
  error e, minimized), when (n / 2) log(e / e_best) > 10 (the same log-density scale); a failed
  fit is a miss. Wilson 95 % interval on the miss rate.
- `run.py` profiles: `smoke` (n 100, 5 sets, the 40 kernel / noise / objective / function
  combinations: 95 s on the machine below), `reference` (committed), `full` (the whole grid,
  about 30 h on 4 cores: the LOO / LMP fits cost 1-6 min per data set at n 1000).

## Reproduction

```sh
# tables below, from the committed results (seconds)
python3 bench/fit-protocol/summarize.py bench/fit-protocol/results/reference.csv.gz --by noise,objective,n
python3 bench/fit-protocol/summarize.py bench/fit-protocol/results/reference.csv.gz --detail   # = results/summary.md
# new results (about 4 h on 4 cores), then the same tables
python3 bench/fit-protocol/run.py --bin <build>/bench/bench-fit-protocol --out <dir> --profile reference \
    --merge <dir>.csv.gz --summary
```

## Measurements

Reference run (`bench/fit-protocol/results/`): profile `reference` = the full grid at n 100
and 200, the LL objectives (noise-free, nugget) at n 500, noise-free LL at n 1000, 30 data sets
each: 3120 data sets, 10320 fits, no failure; `reference.csv.gz` 400 kB. Variants:
`BFGS`, `BFGS4`, `BFGS10` with the defaults (long-range restart c = 2, jitter per kernel,
thread budget 100 %), and `BFGS` with `LK_LONG_RANGE_RESTART=0` on the noise-free LL
configurations (the only ones where the restart acts). Machine (`results/machine.txt`): 4-vCPU
Intel Xeon VM @ 2.1 GHz, 15 GB, Ubuntu 24.04, gcc 13.3, OpenBLAS 0.3.26 pthreads, Release; library
sources of `ef9cf51`.

Kernels and functions pooled (120 data sets per kernel, 240 per row; per kernel and per
function: `results/summary.md`):

| noise | objective | n | variant | sets | miss | miss % [95 % CI] | RMSE | RMSE / best, median / p90 | wall s | CPU s | wall total s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | LL | 100 | BFGS:default | 240 | 24 | 10 [7, 14] | 0.61 | 1 / 1.27 | 0.0445 | 0.172 | 14 |
| none | LL | 100 | BFGS:LK_LONG_RANGE_RESTART=0 | 240 | 45 | 19 [14, 24] | 0.62 | 1 / 1.73 | 0.0223 | 0.0876 | 6.48 |
| none | LL | 100 | BFGS4:default | 240 | 11 | 5 [3, 8] | 0.58 | 1 / 1.01 | 0.0427 | 0.162 | 11 |
| none | LL | 100 | BFGS10:default | 240 | 3 | 1 [0, 4] | 0.544 | 1 / 1 | 0.096 | 0.372 | 25.3 |
| none | LL | 200 | BFGS:default | 240 | 42 | 18 [13, 23] | 0.359 | 1 / 1.47 | 0.587 | 2.33 | 169 |
| none | LL | 200 | BFGS:LK_LONG_RANGE_RESTART=0 | 240 | 70 | 29 [24, 35] | 0.393 | 1 / 3.49 | 0.235 | 0.922 | 68.7 |
| none | LL | 200 | BFGS4:default | 240 | 10 | 4 [2, 7] | 0.192 | 1 / 1 | 0.186 | 0.619 | 48.6 |
| none | LL | 200 | BFGS10:default | 240 | 3 | 1 [0, 4] | 0.188 | 1 / 1 | 0.358 | 1.22 | 91.5 |
| none | LL | 500 | BFGS:default | 240 | 34 | 14 [10, 19] | 0.0306 | 1 / 1.08 | 0.971 | 3.85 | 262 |
| none | LL | 500 | BFGS:LK_LONG_RANGE_RESTART=0 | 240 | 56 | 23 [18, 29] | 0.0419 | 1 / 3.13 | 0.501 | 1.98 | 138 |
| none | LL | 500 | BFGS4:default | 240 | 6 | 2 [1, 5] | 0.0294 | 1 / 1 | 1.07 | 3.03 | 291 |
| none | LL | 500 | BFGS10:default | 240 | 0 | 0 [0, 2] | 0.0294 | 1 / 1 | 2.53 | 8.38 | 594 |
| none | LL | 1000 | BFGS:default | 240 | 33 | 14 [10, 19] | 0.00524 | 1 / 1.04 | 3.53 | 14 | 1.44e+03 |
| none | LL | 1000 | BFGS:LK_LONG_RANGE_RESTART=0 | 240 | 73 | 30 [25, 37] | 0.00583 | 1 / 4.62 | 1.91 | 7.52 | 588 |
| none | LL | 1000 | BFGS4:default | 240 | 7 | 3 [1, 6] | 0.00512 | 1 / 1 | 6.2 | 17.4 | 1.85e+03 |
| none | LL | 1000 | BFGS10:default | 240 | 1 | 0 [0, 2] | 0.00512 | 1 / 1 | 14.2 | 47.6 | 4.3e+03 |
| none | LMP | 100 | BFGS:default | 240 | 30 | 12 [9, 17] | 0.592 | 1 / 1.39 | 0.0437 | 0.172 | 11.7 |
| none | LMP | 100 | BFGS4:default | 240 | 9 | 4 [2, 7] | 0.545 | 1 / 1 | 0.0837 | 0.323 | 22.3 |
| none | LMP | 100 | BFGS10:default | 240 | 0 | 0 [0, 2] | 0.486 | 1 / 1 | 0.184 | 0.647 | 48 |
| none | LMP | 200 | BFGS:default | 240 | 52 | 22 [17, 27] | 0.298 | 1 / 1.97 | 0.33 | 1.29 | 87.5 |
| none | LMP | 200 | BFGS4:default | 240 | 4 | 2 [1, 4] | 0.179 | 1 / 1 | 0.39 | 1.19 | 104 |
| none | LMP | 200 | BFGS10:default | 240 | 1 | 0 [0, 2] | 0.179 | 1 / 1 | 0.869 | 2.96 | 224 |
| none | LOO | 100 | BFGS:default | 240 | 93 | 39 [33, 45] | 0.644 | 1 / 3.06 | 0.0419 | 0.165 | 15.2 |
| none | LOO | 100 | BFGS4:default | 240 | 24 | 10 [7, 14] | 0.545 | 1 / 1.13 | 0.0635 | 0.25 | 23.2 |
| none | LOO | 100 | BFGS10:default | 240 | 4 | 2 [1, 4] | 0.463 | 1 / 1 | 0.182 | 0.624 | 58.1 |
| none | LOO | 200 | BFGS:default | 240 | 80 | 33 [28, 40] | 0.397 | 1 / 4.4 | 0.339 | 1.32 | 131 |
| none | LOO | 200 | BFGS4:default | 240 | 30 | 12 [9, 17] | 0.185 | 1 / 1.04 | 0.282 | 0.838 | 111 |
| none | LOO | 200 | BFGS10:default | 240 | 4 | 2 [1, 4] | 0.154 | 1 / 1 | 0.645 | 2.13 | 211 |
| nugget | LL | 100 | BFGS:default | 240 | 21 | 9 [6, 13] | 0.61 | 1 / 1.3 | 0.0631 | 0.248 | 18.4 |
| nugget | LL | 100 | BFGS4:default | 240 | 8 | 3 [2, 6] | 0.583 | 1 / 1.1 | 0.086 | 0.336 | 23.6 |
| nugget | LL | 100 | BFGS10:default | 240 | 1 | 0 [0, 2] | 0.578 | 1 / 1 | 0.135 | 0.52 | 35.9 |
| nugget | LL | 200 | BFGS:default | 240 | 47 | 20 [15, 25] | 0.395 | 1 / 1.66 | 0.805 | 3.19 | 230 |
| nugget | LL | 200 | BFGS4:default | 240 | 20 | 8 [5, 13] | 0.264 | 1 / 1.08 | 0.676 | 2.54 | 187 |
| nugget | LL | 200 | BFGS10:default | 240 | 6 | 2 [1, 5] | 0.225 | 1 / 1 | 0.881 | 3.25 | 248 |
| nugget | LL | 500 | BFGS:default | 240 | 58 | 24 [19, 30] | 0.0808 | 1 / 1.31 | 1.46 | 5.81 | 429 |
| nugget | LL | 500 | BFGS4:default | 240 | 10 | 4 [2, 7] | 0.0789 | 1 / 1 | 2.02 | 7.02 | 556 |
| nugget | LL | 500 | BFGS10:default | 240 | 5 | 2 [1, 5] | 0.0783 | 1 / 1 | 3.6 | 12.6 | 911 |
| nugget | LMP | 100 | BFGS:default | 240 | 28 | 12 [8, 16] | 0.631 | 1 / 1.31 | 0.0845 | 0.334 | 23.3 |
| nugget | LMP | 100 | BFGS4:default | 240 | 7 | 3 [1, 6] | 0.611 | 1 / 1.03 | 0.135 | 0.532 | 35.3 |
| nugget | LMP | 100 | BFGS10:default | 240 | 1 | 0 [0, 2] | 0.605 | 1 / 1 | 0.23 | 0.81 | 58.6 |
| nugget | LMP | 200 | BFGS:default | 240 | 57 | 24 [19, 30] | 0.399 | 1 / 1.73 | 0.858 | 3.37 | 236 |
| nugget | LMP | 200 | BFGS4:default | 240 | 21 | 9 [6, 13] | 0.246 | 1 / 1.1 | 1.04 | 3.38 | 286 |
| nugget | LMP | 200 | BFGS10:default | 240 | 5 | 2 [1, 5] | 0.242 | 1 / 1 | 1.66 | 5.94 | 429 |

Noise-free LL by function (kernels and n 100-1000 pooled, 240 data sets per row):

| noise | objective | function | variant | sets | miss | miss % [95 % CI] | RMSE | RMSE / best, median / p90 | wall s | CPU s | wall total s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | LL | branin4 | BFGS:default | 240 | 3 | 1 [0, 4] | 0.00015 | 1 / 1 | 0.651 | 2.57 | 385 |
| none | LL | branin4 | BFGS:LK_LONG_RANGE_RESTART=0 | 240 | 14 | 6 [4, 10] | 0.00017 | 1 / 1 | 0.211 | 0.823 | 104 |
| none | LL | branin4 | BFGS4:default | 240 | 1 | 0 [0, 2] | 0.00015 | 1 / 1 | 0.427 | 1.1 | 358 |
| none | LL | branin4 | BFGS10:default | 240 | 1 | 0 [0, 2] | 0.00015 | 1 / 1 | 0.717 | 2.29 | 862 |
| none | LL | hartmann6 | BFGS:default | 240 | 19 | 8 [5, 12] | 0.389 | 1 / 1.02 | 0.698 | 2.77 | 357 |
| none | LL | hartmann6 | BFGS:LK_LONG_RANGE_RESTART=0 | 240 | 53 | 22 [17, 28] | 0.443 | 1 / 1.73 | 0.487 | 1.91 | 308 |
| none | LL | hartmann6 | BFGS4:default | 240 | 1 | 0 [0, 2] | 0.377 | 1 / 1 | 0.501 | 1.62 | 691 |
| none | LL | hartmann6 | BFGS10:default | 240 | 0 | 0 [0, 2] | 0.377 | 1 / 1 | 1.11 | 3.65 | 1.45e+03 |
| none | LL | sin4 | BFGS:default | 240 | 28 | 12 [8, 16] | 0.0227 | 1 / 1.16 | 0.782 | 3.09 | 615 |
| none | LL | sin4 | BFGS:LK_LONG_RANGE_RESTART=0 | 240 | 73 | 30 [25, 37] | 0.0468 | 1 / 5.7 | 0.294 | 1.16 | 157 |
| none | LL | sin4 | BFGS4:default | 240 | 14 | 6 [4, 10] | 0.0191 | 1 / 1 | 0.459 | 1.42 | 594 |
| none | LL | sin4 | BFGS10:default | 240 | 2 | 1 [0, 3] | 0.0189 | 1 / 1 | 1.23 | 3.79 | 1.46e+03 |
| none | LL | sinhalf8 | BFGS:default | 240 | 83 | 35 [29, 41] | 0.318 | 1 / 4.14 | 1.06 | 4.16 | 526 |
| none | LL | sinhalf8 | BFGS:LK_LONG_RANGE_RESTART=0 | 240 | 104 | 43 [37, 50] | 0.535 | 1.02 / 5.61 | 0.45 | 1.79 | 232 |
| none | LL | sinhalf8 | BFGS4:default | 240 | 18 | 8 [5, 12] | 0.142 | 1 / 1.12 | 0.525 | 1.61 | 558 |
| none | LL | sinhalf8 | BFGS10:default | 240 | 4 | 2 [1, 4] | 0.125 | 1 / 1 | 1.44 | 4.77 | 1.23e+03 |

Observations (to be weighed with the limits below):

- **A single start misses often in every mode**, not only the noise-free LL one the long-range
  restart was calibrated on: 9-24 % with a nugget (LL and LMP), 12-22 % LMP, 33-39 % LOO, at
  n 100-500. The long-range restart halves the noise-free LL misses (n 1000: 30 % → 14 %;
  sin4 30 % → 12 %) but leaves 35 % on `sinhalf8` (43 % without), as in `docs/dev/CholJitter.md`.
- **`BFGS4` dominates the restarted single start** on miss rate everywhere (noise-free LL
  2-5 % against 10-18 %), with a lower median wall time at n 200 (0.19 s against 0.59 s)
  and comparable at n 500 (1.07 s against 0.97 s); at n 1000 it costs x 1.8 the wall time
  (6.2 s against 3.5 s) and x 1.2 the CPU time. Its p90 RMSE ratio to the best fit is <= 1.13
  everywhere; the restarted single start reaches 1.04-1.47 (noise-free LL, n >= 200), 3.1-4.6
  without the restart.
- **`BFGS10` is near the best but not always on it**: 0-2 % misses (up to 5 % for matern5_2
  with a nugget, n 200), i.e. a smaller multistart sometimes finds a better optimum; at x 1.3-2.9
  the median wall time and x 1.3-2.8 the median CPU time of `BFGS4`.
- On 4 cores the wall time of `BFGSk` grows less than k (parallel starts) while the CPU time
  grows about as k / 4 x the single start's; on a 1-core budget the ratio would be ~k. Not
  measured here (T3: 1, 4 and all cores).

## Limits of the protocol (critical review)

- **The reference is the best variant run, not the global optimum**: a miss rate is relative to
  the set of variants, and the variant that defines the best of a set cannot miss it. `BFGS10`'s
  0-2 % is therefore a lower bound of its true miss rate, biased in its favour; adding a variant
  (e.g. BFGS20) can only raise the other rates. Compare variants on the same output directory.
- **A fixed threshold of 10 log-density units** is strict at large n (relative to |LL| ~ n) and
  loose at n 100; it does not say whether a miss matters for prediction: read it with the RMSE
  ratio (median and p90) columns. With a nugget the LL can also be higher at a worse RMSE.
- **LOO on the log-density scale** ((n / 2) log(e / e_best)) is a convention chosen here, not a
  likelihood: a 10-unit LOO miss at n 200 is a 10 % increase of the LOO error.
- **30 sets per configuration**: 0 misses gives a 95 % upper bound of 11 %; differences of a
  few points per configuration are not significant; pooled rows (240 sets) assume independent
  data sets, but the same seeds are reused across kernels and n.
- **One machine, one BLAS** (4-vCPU VM, OpenBLAS): times and, through the thread-dependent
  rounding, a few optimizer paths differ on another machine; macOS / Windows / MKL not run.
- **Reduced grid**: LOO / LMP at n 500 and all nugget / LOO / LMP fits at n 1000 are not in the
  reference (cost); `--profile full` runs them. Constant trend and `normalize = false` only;
  noise levels other than 5 % not run.
- Fits are timed one at a time; the wall time includes the parallel starts, not the
  concurrency of several user fits.

## Decisions

- The protocol, the miss definition and the reference run above are the base of T3 (default
  optimizer, long-range restart). Not decided here.
- Results kept as a compressed CSV in the repository (400 kB) rather than a CI artifact, with the
  machine description and the generated tables (`results/summary.md`).
