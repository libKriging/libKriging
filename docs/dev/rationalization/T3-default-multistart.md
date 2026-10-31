# T3. Default multistart vs single start — work log

Task specification: `docs/dev/Rationalization.md`, section T3.

## Status

Done (pending review). Decision: the `Kriging` default optimizer becomes `"BFGS4"` (was
`"BFGS"`); the long-range restart is kept, for an explicit single start (`"BFGS"`) and the
noise-free warm start of the nugget fits. Limits of the sample at the end.

## Protocol

T7 (statistical benchmark base, PR #447) had no usable driver when this was run: the protocol
below is a minimal one, written to be merged into T7 (one CSV row per fit, the columns T7 asks
for: miss, RMSE, wall and CPU time).

- Driver `bench/rationalization/t3_multistart_study.cpp` (built by `build_study.sh`), grid
  `run_t3.sh`, tables `summarize_t3.py` (miss rates, Wilson 95 % intervals, RMSE ratios, times),
  `rmse_t3.py` (paired RMSE), `threads_t3.py` (1 core against 4). Raw results:
  `bench/rationalization/results/t3-A.csv.gz` (15 360 fits) and `t3-B.csv.gz` (1 520 fits); every
  table here is reproduced by `python3 summarize_t3.py results/t3-A.csv.gz` (and the two others).
- Data: the families of `bench/thread-budget/jitter_n_study.cpp`, same data and test sets
  (seed 9000 + set, 500 test points, seed 777): `sin` d 4, `hartmann6` d 6, `branin` d 4 (2
  inactive inputs), `sinhalf` d 8 (4 inactive). Nugget case: `Kriging(kernel, Nugget)` on
  y + N(0, (0.1 sd y)²); its RMSE is against the noise-free function.
- Grid: gauss and matern5_2; noise-free LL / LOO / LMP, nugget LL / LMP (LOO does not support the
  Nugget noise model); n 100, 200, 500, 1000 for LL, 100–500 for LOO / LMP (an LOO single start
  takes ~60 s at n 1000); **30 data sets per configuration** (136 configurations).
- Variants: `BFGS` (long-range restart at its default c = 2), `BFGS-nolr`
  (`Optim::set_long_range_restart(0)`), `BFGS4`, `BFGS10`. `BFGS-nolr` is run wherever the
  restart can act (noise-free LL; nugget, through the noise-free `BFGS` warm start of
  `fit_impl`, see below) and at n 100 for LOO / LMP, where it is bit-identical to `BFGS` (240 / 240
  and 140 / 140 sets): the restart is off for those objectives.
- **Miss**: the optimized objective more than 10 below the best of the 3–4 variants on the same
  data set (LL, LMP). LOO (`leaveOneOut()`, a mean squared error) is put on the same scale as the
  Gaussian log density of the LOO residuals, n/2 log(LOO / LOO_best) > 10. A failed fit would count
  as a miss (none occurred). Thresholds 1 and 50 as sensitivity.
- Pass A: miss / RMSE / single-core CPU time, 4 processes at a time, each on one core
  (`OPENBLAS_NUM_THREADS=1`, `Optim::set_num_threads_percent(25)`). Pass B: wall time, one process
  alone, 1 core (budget 25 %) and the whole machine (100 %), 10 sets of the LL configurations at n
  200, 5 sets at n 1000 (noise-free) and at n 500 (LOO / LMP, gauss, sin / sinhalf). CPU time is
  the process time (all threads) of the fit only, after a 200 ms pause (the OpenBLAS threads of
  the previous `predict` spin ~100 ms).
- Machine: 4-core Intel Xeon @ 2.10 GHz (cloud VM, 1 thread per core), 15 GB, Ubuntu 24.04,
  kernel 6.18, gcc 13.3, OpenBLAS 0.3.26 pthreads (the version of the machine of
  `bench/thread-budget/machine.txt`), Release build, branch base `aad88a7`.

## Measurements

### Miss rate, pooled (pass A, 30 sets per configuration, Wilson 95 %)

| objective | noise | sets | BFGS | BFGS-nolr | BFGS4 | BFGS10 |
|---|---|---|---|---|---|---|
| LL | none | 960 | 14 % [12, 16] | 26 % [23, 28] | 3 % [2, 5] | 1 % [0, 1] |
| LL | nugget | 960 | 12 % [10, 14] | 17 % [14, 19] | 4 % [3, 6] | 2 % [1, 3] |
| LMP | none | 720 | 17 % [15, 20] | (= BFGS) | 2 % [1, 4] | 0 % [0, 1] |
| LMP | nugget | 720 | 13 % [11, 16] | 16 % [14, 19] | 3 % [2, 5] | 1 % [1, 2] |
| LOO | none | 720 | 37 % [33, 40] | (= BFGS) | 12 % [10, 15] | 1 % [1, 2] |
| all | | 4080 | 18 % [17, 19] | | 5 % [4, 6] | 1 % [1, 1] |

Threshold 1 / 50 instead of 10: BFGS 25 % / 10 %, BFGS4 10 % / 2 %, BFGS10 3 % / 0 % (all
configurations); the ordering BFGS10 < BFGS4 < BFGS ≤ BFGS-nolr holds in every (objective, noise)
cell at the three thresholds.

By factor (all objectives, BFGS / BFGS4 / BFGS10): `sinhalf` 38 / 10 / 3 %, `sin` 23 / 6 / 1 %,
`branin` 7 / 3 / 0 %, `hartmann6` 4 / 0 / 0 %; gauss 21 / 5 / 1 %, matern5_2 15 / 5 / 1 %;
n 100 15 / 4 / 1 %, 200 22 / 6 / 2 %, 500 18 / 5 / 1 %, 1000 14 / 4 / 1 %. The single start does
not get better with n; the worst cells (BFGS / BFGS4, 30 sets each): LOO `sinhalf` gauss n 500
83 / 40 %, LOO `branin` gauss n 500 77 / 50 %, LL `sinhalf` gauss n 500 60 / 20 %, LL nugget `sin`
matern5_2 n 500 50 / 23 %. BFGS10 never exceeds 7 % in a cell (2 / 30).

Paired, same data sets (only BFGS misses : only the other misses):

| objective | noise | BFGS vs BFGS-nolr | BFGS vs BFGS4 | BFGS vs BFGS10 |
|---|---|---|---|---|
| LL | none | 0 : 113 | 108 : 9 | 128 : 2 |
| LL | nugget | 4 : 47 | 74 : 0 | 101 : 0 |
| LMP | nugget | 2 : 27 | 69 : 0 | 84 : 0 |
| LMP | none | (identical) | 109 : 2 | 122 : 0 |
| LOO | none | (identical) | 192 : 14 | 256 : 1 |

### Test RMSE (pass A, paired against BFGS10 on the same set)

| objective | noise | BFGS: geo. mean ratio / > 10 % worse | BFGS-nolr | BFGS4 |
|---|---|---|---|---|
| LL | none | 1.156 / 12 % | 1.399 / 23 % | 1.027 / 4 % |
| LL | nugget | 1.063 / 10 % | 1.102 / 14 % | 1.012 / 4 % |
| LMP | none | 1.247 / 17 % | (= BFGS, n 100: 1.152 / 16 %) | 1.029 / 2 % |
| LMP | nugget | 1.065 / 14 % | 1.090 / 16 % | 1.010 / 5 % |
| LOO | none | 1.567 / 33 % | (= BFGS, n 100: 1.427 / 35 %) | 1.053 / 9 % |

The higher optimum is the better predictor: the share of sets where a variant is > 10 % *better*
than BFGS10 is 0–3 % (BFGS4 1–4 %, BFGS-nolr LOO n 100 7 %). No sign of a degenerate optimum
(higher objective, worse prediction) favouring the multistart.

### CPU time on one core (pass A, summed over the sets, relative to BFGS)

| objective | noise | n | BFGS-nolr | BFGS4 | BFGS10 |
|---|---|---|---|---|---|
| LL | none | 100 / 200 / 500 / 1000 | 0.49 / 0.50 / 0.54 / 0.41 | 1.71 / 1.89 / 1.84 / 1.61 | 4.22 / 4.68 / 4.87 / 4.49 |
| LL | nugget | 100 / 200 / 500 / 1000 | 0.66 / 0.70 / 0.67 / 0.63 | 2.05 / 2.07 / 1.98 / 1.93 | 4.07 / 4.20 / 3.92 / 3.95 |
| LMP | none | 100 / 200 / 500 | 0.98 / – / – | 3.70 / 3.66 / 3.72 | 9.26 / 9.06 / 9.86 |
| LMP | nugget | 100 / 200 / 500 | 0.78 / 0.84 / 0.86 | 2.78 / 3.25 / 3.37 | 6.21 / 7.26 / 8.10 |
| LOO | none | 100 / 200 / 500 | 1.04 / – / – | 3.81 / 2.84 / 2.82 | 10.50 / 6.92 / 7.90 |

The long-range restart doubles the single-start cost of a noise-free LL fit (it fires on most
fits: `BFGS` == `BFGS-nolr` on 4–29 of 30 sets only) and costs x 1.2–1.6 on a nugget fit, through
its warm start. Single-core cost of BFGS4: x 1.6–2.1 (LL), x 2.8–3.8 (LOO / LMP) the single start.

### Wall time, 1 core against 4 (pass B, summed over the same sets)

| objective | noise | n | variant | 1 core | 4 cores | speed-up | CPU 4 cores / 1 core |
|---|---|---|---|---|---|---|---|
| LL | none | 200 | BFGS | 21.0 s | 52.9 s | **0.40** | 10.0 |
| LL | none | 200 | BFGS-nolr | 10.6 s | 23.7 s | 0.45 | 8.9 |
| LL | none | 200 | BFGS4 | 39.8 s | 17.7 s | 2.24 | 1.4 |
| LL | none | 200 | BFGS10 | 99.3 s | 36.0 s | 2.76 | 1.3 |
| LL | nugget | 200 | BFGS | 26.3 s | 67.2 s | **0.39** | 10.2 |
| LL | nugget | 200 | BFGS4 | 54.9 s | 53.3 s | 1.03 | 3.7 |
| LL | nugget | 200 | BFGS10 | 109.2 s | 77.8 s | 1.40 | 2.7 |
| LL | none | 1000 (sin gauss) | BFGS | 26.2 s | 13.5 s | 1.94 | 2.0 |
| LL | none | 1000 (sin gauss) | BFGS4 | 80.5 s | 29.1 s | 2.77 | 1.0 |
| LL | none | 1000 (sin gauss) | BFGS10 | 240.7 s | 69.0 s | 3.49 | 1.0 |

On the whole machine (4 cores), relative to BFGS: n 200 LL BFGS4 **0.34**, BFGS10 0.68; nugget
0.79 / 1.16; n 1000 LL (8 configurations, 5 sets) 1.17 / 2.60, per configuration 0.51 (sin
matern5_2, where the restarted single start is slow) to 3.4 (hartmann6 matern5_2); n 500 LMP 2.61
/ 6.48, LOO 1.92 / 5.57 (gauss, sin / sinhalf).

- **A single start runs 2.5 x slower on 4 cores than on 1 at n 200** (CPU x 10): the per-fit
  OpenMP / BLAS parallelism costs more than it saves at that size (`kOmpMinRows` is 200 on
  Linux). This is T5's subject; it makes `BFGS4` *faster* than `BFGS` on a 4-core machine at
  n 200, which should not be counted as a merit of the multistart.
- The nugget `BFGS4` does not scale (x 1.03): its warm start is a noise-free single start fitted
  before the pool, on all the cores (the slow path above), then the 4 starts.
- Pass A against pass B on one core (sin gauss, noise-free LL, same sets: 10 at n 200, 5 at
  n 1000): summed wall time ratio A / B 0.94–1.01 per variant, objectives equal to 1e-3 or
  better: 4 concurrent single-core processes did not inflate the times, pass A's CPU times stand
  for one core.

## Decisions

1. **Default optimizer of `Kriging`: `"BFGS4"`** (C++ `Kriging` constructor and `fit`, Python
   `Kriging` / `KrigingRegressor`, R `Kriging()` / `fit.Kriging`, Octave / MATLAB, Julia; docs,
   skill references, CHANGELOG). Justification: misses 18 % → 5 % (all cells; 14 → 3 % on LL),
   RMSE geometric mean x 1.06–1.57 → x 1.01–1.05 against BFGS10, at x 1.6–3.8 the single-core time
   and x 0.3–2.6 the 4-core wall time.
   - Why 4 and not 10: BFGS10 brings the misses to 1 % (it is the best variant everywhere), at
     x 2.2–2.8 the CPU time of BFGS4 on every budget (x 4–10 a single start on one core). A default
     must not make the 1–2-core cases (CI, batch jobs with one core per fit, R sessions under
     a quota) 4–10 x slower; BFGS10 stays the recommendation for difficult fits (skill, docs).
   - Why a fixed k and not the core count: the result would depend on the machine. With a fixed k
     the starts are the same everywhere (only the summation order of the parallel gradients
     differs, T5).
   - Not changed: `WarpKriging`, `MLPKriging` (joint BFGS; their own studies), `NestedKriging`
     (its sub-models: x nb_groups the cost, not measured). The Vecchia / Nystrom fits of `Kriging`
     follow the new default (not measured: they target large n, pass `"BFGS"` to keep a single
     start).
2. **The long-range restart is kept** (c = 2, single start only):
   - it removes misses without adding any on noise-free LL (paired 0 : 113) and helps the nugget
     fits through their warm start (4 : 47 LL, 2 : 27 LMP); `"BFGS"` stays a documented choice,
     and the nugget warm start is a `"BFGS"` fit whatever the user's `optim`;
   - its cost (x 2 a noise-free single start, fires on most fits) is the price of a single start
     closer to the multistart; a user who chooses `"BFGS"` for speed can set
     `LK_LONG_RANGE_RESTART=0`.
   - Against dropping it: no measurement shows it harmful (4 + 2 sets of 4080 where only the
     restarted fit misses, all nugget); dropping it would make the nugget default (`BFGS4`, warm
     start) worse in a way not measured here (`BFGS4` without the restart was not run).
   - Not justified by this study: the value c = 2 (only 2 against 0 here; 3 in
     `docs/dev/CholJitter.md` on 6–20 sets) and `max_long_range_restart = 2`.

## Limits of the sample (adversarial review of the conclusions)

- **Reference = best of the variants**, BFGS10 included: a miss is relative to the best found,
  not to the global optimum. A common miss of all variants is invisible (the BFGS10 rate is a lower
  bound), and BFGS10 is favoured by construction (10 of the 15–16 starts of a set). BFGS4 against
  BFGS is free of that bias (each is a minority of the starts) and gives the same conclusion.
- **Threshold**: 10 LL units is the convention of `docs/dev/CholJitter.md`; on LOO the conversion
  n/2 log(MSE ratio) is a choice (10 units = a 22 % LOO MSE excess at n 100, 2 % at n 1000). The
  ordering holds at 1 and 50.
- **4 cores only** ("all cores" = 4 here): the 4-core wall times favour `BFGS4` (one start per core);
  on 10–20 cores BFGS10 would cost one round. Not measured: more cores, macOS / Windows, MKL /
  Accelerate (the thread control of T5), hyper-threading.
- **Families**: 4 synthetic functions, uniform random designs, d 4–8; no real data set, no
  clustered design, no d > 8, no n > 1000 (n 2000 was 2 sets in `docs/dev/CholJitter.md`). LOO /
  LMP stop at n 500. The 136 configurations are not independent (same seeds across kernels /
  objectives): the pooled intervals treat 4080 sets as independent, which overstates the precision
  of the pooled rates (not of the per-configuration ones).
- **Kernels**: gauss and matern5_2 only (matern3_2 and exp not run).
- **Timing pass**: 5–10 sets per configuration, one run each, on a shared cloud VM (no CPU pinning,
  no repetition): the wall-time ratios are indicative to ~±20 %; the CPU-time ratios of pass A use
  all 30 sets.
- The 4-core slowdown of a single start at n 200 (above) is not a property of the multistart and is
  likely to change with T1 / T5; the decision does not rely on it (it rests on the single-core and
  n 1000 numbers).
- The study runs on the T3 branch base (`aad88a7`): T1 (gradients), T2 (restart policy) and T5
  (threads) may change the costs, not the miss ordering (the starts and the objective are
  unchanged by them).

## Follow-ups

- T5: single start slower on 4 cores than on 1 at n 200 (x 2.5, CPU x 10), nugget fit (x 2.6); the
  nugget warm start is serial in front of the pool.
- T7: merge this protocol (driver columns, Wilson intervals, LOO scale) into the benchmark base.
- T2: the restart policy keeps the long-range criterion for a single start.
