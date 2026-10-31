# T8. MLPKriging calibration cost — work log

Task specification: `docs/dev/Rationalization.md`, section T8.

## Status

Done: `calibration_folds` defaults to 3 (was 5), factor no longer clamped to >= 1. Same
optimiser, budget and starting point per fold as before (no warm start, no reduced budget).

## Setup

- Machine: 4-core Intel Xeon @ 2.10 GHz (1 thread per core), Linux, GCC 13, Release, OpenBLAS
  0.3.26 + reference LAPACK.
- "1 core": `taskset -c 0`, `LK_NUM_THREADS_PERCENT=25` (budget 1). "4 cores": `taskset -c 0-3`,
  `LK_NUM_THREADS_PERCENT=0` (budget 4).
- CPU time: `getrusage(RUSAGE_SELF)` user + sys around `fit()`; wall: `steady_clock`.
- Model: `MLPKriging({8}, 2, "tanh", "matern5_2")` (`{16, 8}` in 10D), constant trend,
  `optim = "BFGS"` (the default single start), plus `"BFGS4"` on `branin_proj5`.
- 15 noise-free test functions on [0,1]^d, LHS design, 1000 uniform test points:
  `branin_proj5` (d 5, n 60: the `MLPKrigingCalibrationTest` function), `branin2` (2, 30),
  `hartmann3` (3, 40), `hartmann6` (6, 80), `ishigami` (3, 60), `borehole` (8, 80),
  `friedman` (5, 60), `rosenbrock4` (log, 4, 60), `michalewicz2` (2, 40), `goldstein_log` (2, 40),
  `otlcircuit` (6, 60), `piston` (7, 70), `forrester` (1, 15), `sin_inter2` (2, 100),
  `branin_proj10` (10, 120). Designs 0–2 for all variants (timing on 1 and 4 cores), designs 3–8
  for the main variants: 135 cases.
- Variants were run through temporary environment hooks in `calibrate_variance()`
  (`T8/t8-hooks.patch`, not committed in the library): `LK_T8_START=fit` (warm start from the
  fitted parameters), `LK_T8_ITER=n` (per-call fold budget), `LK_T8_DUMP` (held-out standardised
  residuals, to evaluate several estimators of the factor on the same folds).
- Reproduction: apply `T8/t8-hooks.patch`, build, compile `T8/bench_t8.cpp` against the library
  (flags of `bench/bench-mlpkriging`), then `T8/run.sh c4 0-3 0`, `T8/run.sh c1 0 25`,
  `T8/run2.sh`, `python3 T8/report.py`, `python3 T8/report2.py`.

Coverage = share of test points with |y − m| <= 1.96 s; NLPD = test mean negative log predictive
density. ΔNLPD is the paired per-case difference to the 5-fold baseline, CI by bootstrap.

## Measurements

### Cost (sum over the 45 single-start fits, designs 0–2; ratio to the fit without calibration)

| | 1 core wall / CPU | 4 cores wall / CPU |
|---|---|---|
| K = 0 (no calibration) | 8.5 s / 8.4 s | 7.8 s / 30.7 s |
| K = 5, start point, full budget (**before**) | x3.98 / x3.97 | x2.44 / x1.98 |
| K = 3, start point, full budget (**after**) | x2.36 / x2.36 (re-measured on the final build: x2.21 / x2.21) | x1.71 / x1.38 (final build: x1.81 / x1.46) |
| K = 5, warm start from the fit | x3.93 / x3.92 | x2.36 / x1.86 |
| K = 3, warm start from the fit | x2.34 / x2.34 | x1.71 / x1.37 |
| K = 3, fold budget 10 iterations per call | x1.76 / x1.75 | x1.43 / x1.21 |
| K = 5, fold budget 10 iterations per call | x2.84 / x2.84 | x1.81 / x1.52 |
| K = 10 | x8.74 / x8.74 | x3.90 / x3.34 |

Per case (1 core, CPU): K = 5 median x4.08 (10–90 %: x3.12–x5.61); K = 3 median x2.39
(x1.97–x3.50). `BFGS4` on `branin_proj5` (3 designs): K = 5 x1.74 CPU on 1 core, K = 3 x1.18.

Final build, all 45 + 3 fits: 1 core 8.6 s → 19.1 s CPU with K = 3 (before: 8.4 s → 33.4 s
with K = 5); 4 cores 7.4 s → 13.5 s wall (before: 7.8 s → 19.0 s).

### Quality (135 single-start cases)

| Variant / estimator of c | mean cov | median cov | cases cov < 0.8 | < 0.9 | mean abs(cov − 0.95) | ΔNLPD vs K5 (mean, 95 % CI) |
|---|---|---|---|---|---|---|
| uncalibrated | 0.611 | 0.608 | 102 | 114 | 0.347 | +64.8 [+44.7, +89.7] |
| K = 5, mean z² clamped >= 1 (before) | 0.957 | 0.975 | 3 | 18 | 0.042 | 0 |
| K = 5, q95(abs z) / 1.96, squared | 0.951 | 0.967 | 2 | 20 | 0.040 | −0.06 [−0.23, +0.11] |
| **K = 3, mean z² (after)** | 0.965 | 0.986 | 3 | 14 | 0.044 | +0.08 [−0.30, +0.45] |
| K = 3, fold budget 10 | 0.960 | 0.980 | 2 | 19 | 0.044 | +0.05 [−0.29, +0.38] |
| K = 5, fold budget 10 | 0.947 | 0.969 | 9 | 19 | 0.048 | +0.31 [+0.03, +0.65] |

Designs 0–2 only (48 cases incl. `BFGS4`):

| Variant / estimator | mean cov | cases < 0.8 | median ΔNLPD | mean ΔNLPD |
|---|---|---|---|---|
| K = 5, warm start from the fit, mean | 0.919 | 3 | +0.04 | +0.70 |
| K = 3, warm start from the fit, mean | 0.944 | 1 | −0.06 | +0.18 |
| K = 5, median(z²) / 0.4549 | 0.830 | 20 | +2.10 | +4.97 |
| K = 5, 10 %-trimmed mean of z² (chi²₁-corrected) | 0.894 | 7 | +0.50 | +1.31 |
| K = 3, median / trimmed | 0.861 / 0.916 | 12 / 5 | +0.70 / +0.01 | +3.78 / +1.39 |
| K = 10, mean | 0.942 | 3 | −0.05 | +0.10 |

Mean z² over the held-out points: K = 5 always >= 1.30 (median 60), K = 3 always >= 1.60
(median 93) over 135 fits; below 1 only with the reduced fold budget (1 / 135 with K = 3,
3 / 135 with K = 5).

Mean coverage per function (9 designs; uncalibrated, K = 5, K = 3): borehole 0.51 / 0.94 / 0.96,
branin2 0.77 / 0.99 / 0.99, branin_proj10 0.48 / 0.97 / 0.98, branin_proj5 0.63 / 0.95 / 0.97,
forrester 0.92 / 0.95 / 1.00, friedman 0.67 / 0.96 / 0.99, goldstein_log 0.45 / 0.97 / 0.99,
hartmann3 0.52 / 0.92 / 0.96, hartmann6 0.55 / 0.97 / 0.96, ishigami 0.67 / 0.96 / 0.95,
michalewicz2 0.75 / 0.95 / 0.95, otlcircuit 0.40 / 0.96 / 0.91, piston 0.34 / 0.94 / 0.95,
rosenbrock4 0.55 / 0.95 / 0.93, sin_inter2 0.95 / 0.99 / 0.99.

`MLPKrigingCalibrationTest` (n = 60, seed 407; Kriging: NLPD 3.26, coverage 0.82; uncalibrated
NLPD 17.1 / 94.7, coverage 0.575 / 0.271):

| optim | before (K = 5): var_scale, NLPD, coverage | after (K = 3) |
|---|---|---|
| `BFGS` | 14.8, 5.31, 0.944 | 190, 5.71, 0.994 |
| `BFGS4` | 4396, 4.31, 1.000 | 1154, 3.71, 0.989 |

## Decisions

1. **Calibration kept on by default, K = 3.** Opt-in rejected: without it 102 / 135 cases cover
   < 80 % (mean 0.61). K = 3 has the same quality as K = 5 within the measurement
   uncertainty (ΔNLPD CI [−0.30, +0.45], coverage 0.965 vs 0.957, slightly more conservative,
   as expected from folds trained on 2/3 of the data) at 0.55–0.6 times the CPU time of the
   calibrated fit. `calibration_folds = "5"` restores the former value; models saved before keep
   their own number of folds (saved in the JSON).
2. **No warm start from the fitted parameters.** It does not reduce the cost (x3.93 vs x3.98)
   and under-covers (K = 5: 0.919), as the fit already knows the held-out points. The existing
   choice (start point of the winning start) is confirmed.
3. **No reduced fold budget.** K = 3 with 10 iterations per call saves another 25 % CPU with an
   equivalent quality here, but the folds would then estimate the error of a different, less
   converged procedure than the fit, and K = 5 with the same reduction is significantly worse
   (ΔNLPD CI [+0.03, +0.65], 9 cases < 0.8). Not adopted; `max_iter_bfgs` still applies to the
   folds as to the fit.
4. **Clamp >= 1 removed.** It was never active with the fit's budget (0 / 270 fits for K = 3 and
   K = 5), so the change has no measured effect; its justification ("the folds are slightly
   optimistic") is contradicted by the measured over-coverage of K-fold (training on fewer
   points makes the folds pessimistic). The mean z² is the maximum-likelihood scale of the
   held-out residuals, a value below 1 is as much an estimate as one above.
5. **Mean of z² kept (not a robust estimate).** The median and trimmed means under-cover
   (0.83–0.92): the overconfidence of the uncalibrated model is concentrated in the tail of the
   held-out residuals, which a robust estimate discards. The quantile estimate
   (q95(|z|) / 1.96)² is equivalent to the mean (ΔNLPD CI [−0.23, +0.11]), not better: not
   adopted.

## Other findings (out of scope)

- The "~6x" of `Rationalization.md` §1.5 is not reproduced: K = 5 cost x4.0 the fit (median
  per case x4.1, 90 % quantile x5.6) on 1 core, and the pool does not hide it on 4 cores (wall
  x2.4).
- On 4 cores the **uncalibrated** single-start fit uses 3.6x the CPU time of the 1-core run
  (30.7 s vs 8.4 s; sys time ~30 % of the 4-core runs vs 1 % on 1 core) for an 8 % lower wall time: BLAS / OpenMP threads on
  n <= 120 matrices. It costs more than the calibration itself (relevant to T5).
- 15 / 48 fits (hartmann6, borehole, ...) give different results on 1 and 4 cores (fit and
  calibration alike): thread-count dependence of the fit (§1.3, T1 / T5).
