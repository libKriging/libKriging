# Fixed Cholesky jitter

A fixed jitter on the diagonal of the correlation matrix of noise-free fits, on by default
(PR #415). Summary in [parallel.md](parallel.md) 8.6; this page keeps the decisions, the key
measurements and the jitter of every class and method. The study drivers are in
`bench/thread-budget/` (their CSV output is not kept in the repository: rerun them).

## Why

A single-start fit of a smooth kernel (gauss, matern5_2) on noise-free data depended on the
rounding (BLAS thread count, BLAS build, CPU): 7–75 % of the data sets ended at a different
optimum with 1, 10 or 20 BLAS threads, up to hundreds of log-likelihood units apart (matern3_2 and
exp: bit-identical). The optimizer enters the near-singular region of R, where `safe_chol_lower`
adds a stepwise nugget (`num_nugget` x 10^k when the Cholesky fails or rcond < 1e-18): the
objective is discontinuous there and the L-BFGS-B paths diverge. The thread budget only changed
which outcome a user got (default BLAS count 20 → 10). A fixed jitter makes the objective
continuous (`fit_repro_study.cpp`, table in parallel.md 8.6).

## Values per kernel

`LinearAlgebra::kernel_chol_jitter`: gauss 1e-9, matern5_2 1e-10, others 0 (they never reach the
near-singular region). `set_chol_jitter(x)` / `LK_CHOL_JITTER`: x < 0 per kernel (default), 0 off,
x > 0 that value. Noise-free fits only (nugget and noise models keep their own diagonal).

gauss, 240 data sets (Branin 50, sin n 50–500 d 2–8, Hartmann6 60 / 300), with the starting points
and restart tolerance below (`jitter_scale_study.cpp`):

| J (gauss) | theta differs across budgets 10 / 100 / 200 % | median test RMSE, on / off |
|---|---|---|
| 0 | 18 % | 1 |
| 1e-8 | 2.1 % | 1.0–1.24, Branin x246 |
| **1e-9** (chosen) | 1.7 % | 1.0–1.06, Branin x3.8 |
| 1e-10 | 6.2 % | 1.00, Branin x3.8 |

The remaining divergences are iteration-limit paths on flat objectives (sin d 4); Branin (very
smooth) needs a conditioning that any jitter forbids.

## Search versus committed model

The jitter `J` is in the objective only (optimization, `logLikelihoodFun`, LOO, LMP, and the
approximate methods). Measured on 60 data sets per row (`jitter_split_study.cpp`), against a model
rebuilt at the same theta with jitter 0 (`on>0`), 1e-10 (`on>10`), or a last BFGS without jitter
(`on~0`): `on>0` keeps the reproducibility of the search (0 % divergence) and recovers about half
of its accuracy cost (gauss Branin 30: RMSE 0.059 → 0.037); `on~0` is as accurate as no jitter but
diverges again (up to 93 %).

- **Kriging: committed model exact** (`model_chol_jitter()` = 0; `num_nugget` only if the Cholesky
  fails). With 1e-10 the python-sdist check (n 10, gauss, R near singular) missed y by 1e-5; the
  error at the design points is J x |R^-1 (y - F beta)|.
- **WarpKriging / MLPKriging: committed model exact as Kriging's** (#428; it kept min(J, `num_nugget`)
  = 1e-10 before, #415). An exact model brings the training rows together in Phi where warps
  collapse (Kumaraswamy, saturated tanh MLP), so the predicted variance there is rounding, often
  negative. `predict` therefore floors the variance (below) instead of clamping it to 0.
- **Predicted variance floor** (`KrigingImpl::predict_impl`, all classes): the variance
  `1 - r'R^-1 r + ...` (units of sigma2) is floored at 4 eps instead of 0 (NaN too), so the stdev
  is never 0 (no NaN NLPD) and the stdev derivative is defined, 0, where the variance is at the
  floor (it was NaN, 0/0). At the design points of an exact Kriging model the stdev is then
  ~sigma x 3e-8 instead of 0 or ~1e-8 at random.
- `logLikelihood()` returns the optimized objective; both jitters are saved in the JSON (absent →
  former behaviour), so a later `set_chol_jitter` does not change a fitted model.
- `update_simulate` adds no jitter on the new observations, so that an update point on the
  simulation design is reproduced exactly.

## Starting points and restarts

With the jitter the long-range tail of the LL becomes a smooth plateau that traps random starts
(gauss sin n 200 d 2: 6 / 20 BFGS10 fits at theta ≈ 7–10, RMSE 0.6–0.9 instead of ~1e-4); the
erratic tail used to send them back through the restart criteria. Hence (#418):

- Kriging draws its starts **log-uniform** in the theta bounds (uniform for a non-positive lower
  bound): that case 0.63 → 2.0e-4.
- The restart criterion "no improvement" uses the optimizer's tolerance (`factr * epsmch`) instead
  of an exact comparison of two equal optima.
- WarpKriging draws from `Random` as Kriging (reset to its seed at each fit); log-uniform for
  per-variable warps (LL never worse on 20 sets x 4 configurations), uniform for the joint MLP
  (log-uniform lost 124 LL units, median, on a projected Branin d 5: `warp_start_study.cpp`).

A single start still misses the optimum often, with or without jitter (gauss 40 % vs 44 %,
matern5_2 10 % vs 12 % of 800 data sets: `jitter_multistart_study.cpp`); the jitter changes which
data sets miss. `unstableLLTest` checks the failover with the jitter off and, with it on, that
BFGS10 finds the maximum (theta 0.1506) and LL(BFGS10) >= LL(BFGS).

## Adam phase

Multi-seed study of the WarpKriging warps (gauss, n 400, d 4, reference BLAS; 3 functions — the
bench's sum sin(2 pi x), sum sin(2 pi x^2), a Gaussian bump plus a trend — x 5 data seeds, master
fcabe4c5 against this branch). The CI bench (one data set) showed the branch losing LL on mlp,
neural_mono and categorical; the study separates three causes:

- **Objective, not optimum.** `logLikelihood()` is the objective at the search jitter (1e-9),
  master's at num_nugget: above LL ≈ 1000 that alone costs 30–130 units. Both versions' models
  re-evaluated at 1e-9 (save, load, `logLikelihoodFun`): neural_mono f1 +22 to +44 for the
  branch (5 / 5), ordinal f1 +2 to +6, categorical f1 even.
- **First inner BFGS at a corner.** With the search jitter forced to 1e-10 the branch reproduces
  master exactly: the paths differ by the jitter only. On categorical / ordinal f2, seed 2, the
  gradient at the start theta = 1 points out of the box in every direction; L-BFGS-B steps to the
  upper corner (a KKT point, 1 iteration) and the warm-started runs of the later Adam steps never
  leave it (LL −329, sigma2 1e9). `AdamBFGS` now restarts that first run with the Kriging
  criteria (abnormal stop, <= 2 iterations, at the lower bound), theta contracted toward its
  lower bound as the joint BFGS does: −329 → 831, −287 → 588; the 73 other cases bit-identical.
- **Plateau of the Adam loop.** With a fixed step Adam oscillates around a ridge and the patience
  (20 steps) stops it there: mlp f1 1094 at 60 steps against 1365 at 138 at 1e-10. At the first
  plateau `AdamBFGS` now goes back to the best point with half the step, twice (`lr_decays`,
  `kAdamLrDecays = 2`), before stopping.

Against master (LL as returned, so the branch pays the jitter of the first point):

| warp | branch before | + restart | + restart + 2 decays | patience 60 instead | time, decays / restart |
|---|---|---|---|---|---|
| mlp | median −22, 7 better / 8 worse | same | **median +167, 13 / 2** | +73, 10 / 5 | x2.1 |
| categorical | −16, 1 / 10, one at −1117 | −12, 2 / 9 | 0, 4 / 7 | 0, 6 / 5 | x1.25 |
| ordinal | 0, 4 / 7, one at −863 | 0, 6 / 5 | 0, 6 / 5 | same | x1.3 |
| kumaraswamy | +0.4, 7 / 0 | same | same | same | x1.37 |
| neural_mono | −29, 3 / 8 | same | same (no plateau stop) | same | x1.0 |

Total fit time of the 75 fits: master 64 min, branch with restart 58 min, with the decays 81 min.
Decided 2026-10-24: decays for every warp optimised by Adam (the cost is bounded by
`max_iter_adam`); kumaraswamy and ordinal pay +30 % for no gain. Left: mlp f1, seed 3 (1136
against master's 1365), a local optimum of that path.

## Jitter growing with n (#426)

Measured 2026-10-25 (`jitter_n_study.cpp`, `run_jitter_n.sh`, `summarize_jitter_n.py`; gauss,
single-start BFGS, Branin, sin d 2 / 4 / 8, Hartmann6, n 50–2000: 20 sets up to n = 200, 10 at
500, 6 at 1000, 2 at 2000, 386 in all). Rules anchored at J(100) = 1e-9: `lin` 1e-9 n / 100
(J / lambda_max constant, lambda_max <= n), `sqrt` 1e-9 sqrt(n / 100), `quad` 1e-9 (n / 100)^2
(Cholesky backward error n eps cond, cond <= n / J), against the constants 0, 1e-10, 1e-9, 1e-8.
4-core machine: budgets 100 and 200 % gave the same BLAS thread count, so the divergence is
1 thread against 4. "Material" divergence: the LL differs by > 1 or the test RMSE by > 10 %
across the budgets; the other theta divergences (> 1e-3) end at equivalent points of a flat
objective (sin d 4, n 200: |dLL| < 0.2, same RMSE to 3 digits).

| rule | material divergence, n 50 / 100 / 200 / 500 / 1000 / 2000, all | fits > 3 x the best RMSE of the set, all |
|---|---|---|
| 0 | 0 / 6 / 5 / 2 / 3.3 / 17, **3.6 %** | 6.7 % |
| 1e-10 | 0 / 3 / 0 / 2 / 3.3 / 33, 1.8 % | 7.5 % |
| **1e-9** (current) | 0 / 1 / 0 / 2 / 3.3 / 33, **1.3 %** | 7.3 % |
| 1e-8 | 0 / 0 / 1 / 0 / 3.3 / 0, 0.5 % | 13.5 % |
| lin | 0 / 1 / 1 / 0 / 3.3 / 0, 0.8 % | 8.8 % |
| sqrt | 0 / 1 / 0 / 2 / 0 / 0, 0.5 % | 7.0 % |
| quad | 0 / 1 / 1 / 2 / 0 / 0, 0.8 % | 7.8 % |

Median per-set RMSE ratio to 1e-9 (pct 100), where the rules differ:

| data | n | lin | sqrt | quad | 1e-8 |
|---|---|---|---|---|---|
| Branin | 200 / 500 | 1.08 / 1.04 | 1.02 / 1.01 | 1.17 / 1.08 | 1.50 / 1.06 |
| sin d 2 | 500 / 1000 | 1.06 / 1.02 | 1.03 / 1.01 | 1.14 / 1.04 | 1.09 / 1.02 |
| sin d 4 | 500 | 1.12 | 1.01 | 1.24 | 1.18 |
| Hartmann6 | 2000 (2 sets) | 2.05 | 1.84 | 2.56 | 2.06 |

- **No rule J(n) is better than the constant.** The material divergence of 1e-9 is 1.3 % (5 sets
  of 386) against 0.5–0.8 % for the rules: 2 to 3 sets, within the sampling noise; at n = 2000 one
  set is 17 %. A growing J costs accuracy where n is large (Hartmann6 n 2000 x 2, sin d 4 n 500
  x 1.1–1.2), the opposite of the issue's constraint; Branin's cost does not grow with n (ratios
  <= 1.08 from n = 500).
- **The premise does not hold at the measured n.** lambda_max of R at the fitted theta grows as
  n^0.7 (Branin: 10^1.58 at n 50, 10^2.47 at n 1000), so cond(R + J I) <= lambda_max / J stays
  <= 1e12 with J = 1e-9 (sin d 4 n 1000: lambda_max 10^2.88), far from 1 / eps; R itself is numerically singular (lambda_min <=
  1e-12 lambda_max) at every n from 100 on for the smooth cases, and the jitter fixes the
  conditioning alone, whatever n.
- **The remaining failure at large n is a long-range local optimum, not the rounding.** sin d 4 n
  1000 / 2000 and Hartmann6 n 1000 / 2000: single-start fits end with a long-range theta (up to the
  upper bound 10; not a plateau, see the traces below),
  1000–7500 LL units below the optimum, RMSE x 1.7–270, at every J and not monotonically in J
  (which J escapes depends on the path); a larger J makes the long-range optima more attractive (Hartmann6
  n 2000: both sets on it from J = 1e-8, one of two at 4.5e-9, none at 1e-9 and below). sin d 4 n 2000 with 1e-9: 1 thread
  found LL 2209 (RMSE 0.017), 4 threads a long-range optimum (−2800, RMSE 0.52): the divergence counted at
  n = 2000 is that, and 1e-8 / lin remove it by putting both budgets on the same path, the long-range optimum for that set.
- **Multi-start removes both.** BFGS10, J 0 / 1e-9 / 1e-8, sin d 4 and Hartmann6, n 500 (10 sets)
  and 1000 (6): no divergence at all (theta), no long-range optimum, J without effect (median
  RMSE sin d 4 n 1000: 0.00471 / 0.00483 / 0.00494); single-start 1e-9 against BFGS10: sin d 4
  0.176 → 0.080 (n 500), 0.057 → 0.0048 (n 1000); time x 2.8–8.3.

### Plateau or local optimum: long-range restart

The traces of the single-start fits (every objective evaluation printed by a temporary probe, not
kept, sin d 4 n 500 / 1000, Hartmann6 n 1000, J 1e-9, 22 fits) show no plateau:
the failed fits (LL 400–2150 below BFGS10) converge normally (gradient falling geometrically,
`REL_REDUCTION_OF_F`) to sharp local optima with a long-range component (theta_max / maxdX 3.6–10;
one fit stuck at the upper corner after the first iteration, already restarted by
`num_iters <= 2`, then back at the bound). A plateau criterion does not separate them: the span
in log theta of the evaluations with a gradient < 1 % (5 %) of the first one is 0–0.05 for 11 of
the 12 failed fits and up to 3.6 (6.2) for the good ones; pairs of successive evaluations with
|d log theta| > 0.3 and |df| < 10 occur in both. What separates them is the range:
max_j theta_j / maxdX_j >= 3.6 for 11 of the 12 failed fits, <= 1.3 for every good one.

Restart criterion, implemented (`Optim::set_long_range_restart(c)` / `LK_LONG_RANGE_RESTART`,
default 2, 0 off; `Optim::set_max_long_range_restart` / `LK_MAX_LONG_RANGE_RESTART`, default 2):
one more condition of the existing restart test of a single-start (`BFGS`) noise-free LL fit,
max_j theta_j / maxdX_j > c, at most 2 times per start, same restart point (contracted toward the
lower bound), best kept (`tests/LongRangeRestartTest.cpp`). J 1e-9, budgets 10 / 100 / 200 %;
miss: LL > 10 below the best of the variants of the set:

| data | n (sets) | BFGS miss / RMSE / time | c = 2 | c = 3 | BFGS10 |
|---|---|---|---|---|---|
| sin d 4 | 200 (20) | 20 % / 0.546 / 0.33 s | 0 % / 0.521 / 0.46 s | 5 % / 0.521 / 0.40 s | |
| sin d 4 | 500 (10) | 50 % / 0.176 / 0.76 s | 20 % / 0.0797 / 1.5 s | 20 % / 0.0797 / 1.3 s | 0 % / 0.0797 / 4.1 s |
| sin d 4 | 1000 (6) | 67 % / 0.0566 / 2.3 s | 0 % / 0.00483 / 4.2 s | 0 % / 0.00483 / 4.0 s | 0 % / 0.00483 / 19 s |
| Hartmann6 | 200 (20) | 5 % / 0.488 / 0.24 s | 0 % / 0.476 / 0.37 s | | |
| Hartmann6 | 500 (10) | 10 % / 0.312 / 1.7 s | 0 % / 0.312 / 1.8 s | | 0 % / 0.317 / 4.7 s |
| Hartmann6 | 1000 (6) | 33 % / 0.198 / 5.8 s | 0 % / 0.187 / 7.7 s | 0 % / 0.187 / 6.7 s | 0 % / 0.187 / 19 s |
| Branin | 100 (20) / 500 (10) | 10 / 0 %, same RMSE | 0 / 0 %, x 1.1 time | | |
| sin d 2 / d 8 | 500 (10) | 0 % | 0 %, same RMSE, x 0.85 / 1.4 time | | |

No material divergence across budgets with the criterion (0 of 122 sets, BFGS 4). A maximum of
4 restarts instead of 2: identical results. The 2 misses left on sin d 4 n 500 are a short-range local
optimum (theta_min 0.07, LL −736 against −341) and a near tie (14 LL units): out of the
criterion's reach.

Inactive inputs (their theta is genuinely long range, so the criterion fires on nearly every fit):
`sinhalf` (sin on the first half of the inputs), Branin and Hartmann6 with 2 more inputs than
they use; c = 2 against 0, J 1e-9:

| data | d (active) | n (sets) | BFGS miss / RMSE / time | c = 2 | time c = 2 / BFGS | BFGS10 |
|---|---|---|---|---|---|---|
| sinhalf | 4 (2) | 200 (20) | 15 % / 0.00175 / 0.24 s | 5 % / 0.00175 / 0.69 s | 2.5 | 0 % / 0.00175 / 0.35 s |
| sinhalf | 4 (2) | 500 (10) | 10 % / 8.0e-5 / 0.40 s | 0 % / 6.4e-5 / 1.1 s | 3.2 | 0 % / 6.4e-5 / 1.9 s |
| sinhalf | 4 (2) | 1000 (6) | 0 % / 1.0e-5 / 1.5 s | 0 % / 1.0e-5 / 3.1 s | 2.5 | 0 % / 1.0e-5 / 11 s |
| sinhalf | 8 (4) | 200 (20) | 60 % / 0.727 / 0.22 s | 55 % / 0.685 / 0.64 s | 2.7 | 0 % / 0.522 / 0.40 s |
| sinhalf | 8 (4) | 500 (10) | 70 % / 0.401 / 0.76 s | 50 % / 0.380 / 1.5 s | 1.8 | 0 % / 0.0772 / 3.1 s |
| sinhalf | 8 (4) | 1000 (6) | 83 % / 0.617 / 3.3 s | 83 % / 0.617 / 3.4 s | 1.1 | 0 % / 0.122 / 19 s |
| Branin | 4 (2) | 200 (20) | 30 % / 4.7e-4 / 0.21 s | 10 % / 3.4e-4 / 0.44 s | 2.3 | 0 % / 3.1e-4 / 0.27 s |
| Branin | 4 (2) | 500 / 1000 | 0 % | 0 %, same RMSE | 3.1 / 2.8 | 0 % |
| Hartmann6 | 8 (6) | 200 (20) | 10 % / 0.489 / 0.26 s | 0 % / 0.485 / 0.63 s | 2.7 | 0 % / 0.485 / 0.41 s |
| Hartmann6 | 8 (6) | 500 (10) | 20 % / 0.333 / 0.97 s | 10 % / 0.318 / 1.7 s | 2.1 | 0 % / 0.318 / 3.2 s |
| Hartmann6 | 8 (6) | 1000 (6) | 0 % / 0.187 / 2.4 s | 0 % / 0.187 / 5.7 s | 2.6 | 0 % / 0.187 / 16 s |

- LL never lower (best kept), misses fewer or equal everywhere, no material divergence (BFGS: 4
  of 144 sets, c = 2: 1).
- The cost of an inactive input is the two restarts: x 1.8–3.2 the single-start time. At n = 200
  BFGS10 (its starts in parallel) is faster than the restarted single start and finds the
  optimum every time; from n = 500 on BFGS10 costs x 3–8.
- Not enough with 4 inactive inputs of 8 (sinhalf d 8): 50–83 % misses against 0 % for BFGS10.
  The restart point contracts every theta toward the lower bound, the inactive ones too, and the
  restarted runs end in the same basins; restarting from the solution with only the long-range
  active components contracted is not measured (an inactive and an active long-range component
  cannot be told apart from theta alone).

Not measured: matern5_2; nugget / noise models (the criterion is off there); LOO / LMP objectives
(off); the R / Octave / Julia binding tests (not runnable on the machine of the measurements; the
C++ suite, 220 tests, and the Python tests pass with and without the criterion).

Decision: keep the constant per kernel. The robustness at large n belongs to the starts (#418):
the long-range restart above (c = 2, on by default for a single start) gives BFGS10's optimum on
120 of 122 sets at 1.1–2 x the single-start time (BFGS10: 3–8 x), x 2–3 with inactive inputs; a
multistart remains the only reliable choice with many inactive inputs. Not measured: matern5_2,
n > 2000, more than 2 sets at n 2000, a machine with more cores (budgets 100 / 200 % distinct).

## Open items

- Calibration: gauss predictive sd is overconfident (sd / RMSE 0.25–0.43) with or without jitter;
  matern5_2 ≈ 1 (#425).
- A jitter growing with n (#426): measured, no gain over the constant (above); the large-n failures
  are long-range local optima of the single start, not plateaus; the long-range restart (default
  c = 2) fixes most of them; many inactive inputs still need a multistart (#418).
- A fixed floor for the `simulate` conditional covariance instead of the retry (inconsistency 2,
  #427).
- A floor on the predicted variance, then an exact committed WarpKriging model (#428).
- Single-start robustness at large n (#418); bindings consistency checks (#419).
- Done: getters `chol_jitter()` / `model_chol_jitter()` of `Kriging`, `WarpKriging`,
  `MLPKriging` in the four bindings, kept by `save` / `load` (#424).

# Jitter of each class and method

State after the WarpKriging alignment (2026-10-22).

Rule: the search adds `J`; the committed model is exact for Kriging, at most `N` for WarpKriging;
an approximate method keeps the diagonal of its objective.

## The three mechanisms

| name | value | where it is set | nature |
|---|---|---|---|
| search jitter `J` | `kernel_chol_jitter(kernel)`: gauss 1e-9, matern5_2 1e-10, others 0; `set_chol_jitter(x)` / `LK_CHOL_JITTER` | `Kriging::m_chol_jitter`, `WarpKriging::m_chol_jitter`, saved in the JSON | fixed, the same at every theta |
| `num_nugget` `N` | 1e-10 (`set_num_nugget`) | `LinearAlgebra::num_nugget` | fixed constant |
| retry | `N`·10^k, k = 0..10, added by `safe_chol_lower` when the Cholesky fails or rcond < 1e-18 | `LinearAlgebra::safe_chol_lower_retry` | stepwise: depends on whether a Cholesky fails, so on rounding (the discontinuity the search jitter removes) |

Everything below adds the retry on top when the Cholesky still fails, except where `arma::chol` is
called directly (noted "no retry").

## Kriging, noise-free (`noise = none`)

| method | factorization | diagonal |
|---|---|---|
| fit, `LL` / `LOO` / `LMP` objective (exact method) | `populate_Model` | 1 + `J` |
| `logLikelihood()`, `logLikelihoodFun`, `leaveOneOut()`, `leaveOneOutFun/Vec`, `logMargPost(Fun)` | `make_Model` (objective) | 1 + `J` |
| committed model `m_T` (after BFGS, `optim = "none"`, warm restart) | `commit_model_chol_jitter` / `make_committed_Model` | 1 (exact) |
| `predict` (mean, stdev, cov, deriv) | `m_T` | 1 (exact): interpolates the data |
| `simulate` | `m_T`, then `safe_chol_lower` of the conditional covariance at `X_n` | 1 (exact); conditional covariance: 0 + retry |
| `update(refit = false)` | `update_cholCov` extends `m_T` | 1 (exact) |
| `update(refit = true)` (warm restart) | search on 1 + `J`, then commit | 1 + `J`, then 1 |
| `update_simulate` | `R_uu` exact, eigen-decomposition clamped at 0 | 1 (exact), consistent with `update` |
| `save` / `load` | `chol_jitter` (`J`) written when > 0; `m_T` saved as is | absent → 0 |

## Kriging, `method = "vecchia(m)"` / `"nystrom(k)"` (noise-free)

| method | diagonal |
|---|---|
| Vecchia LL (fit, `logLikelihood`), `predictVecchia`, `simulateVecchia` | neighbour blocks 1 + `J` + `N` |
| Vecchia `LOO` / `LMP` objectives (`build_vecchia_precision`) | neighbour blocks 1 + `J` + `N` |
| `update` (Vecchia, Nystrom) | the model's objective again (warm restart if `refit`), then the light commit (`commit_approx`): as the rows above |
| Nystrom LL, `predictNystrom`, `simulate` (Nystrom) | landmark block + `N` (no retry); residual `D0 + J`, `D0` floored at `N` |
| Nystrom `LOO` / `LMP` objectives (`build_nystrom_precision`) | landmark block + `N` (no retry); residual `D0 + J`, `D0` floored at `N` |
| `simulate` (Nystrom) conditional covariance | eigenvalues floored at max(`N`, sqrt(eps)·max eigenvalue), then `safe_chol_lower` |

The approximate methods keep the search diagonal in their committed state (it comes from the objective):
they do not interpolate exactly, which is in their nature anyway.

## Kriging with a noise model (`nugget`, `noise = vector`)

`J` = 0. The diagonal is that of the model: `alpha`·R + (1 - `alpha`)·I (nugget) or R + diag(noise / sigma2),
for the objective, the committed model, `predict`, `simulate`, `update` and `update_simulate` alike.

## WarpKriging, MLPKriging

`MLPKriging` is a facade over `WarpKriging(mlp_joint)`: same rules. Two values, as Kriging:
`m_chol_jitter` (search: `J` if > 0, else `N`) and `m_model_chol_jitter` (committed model:
min(`m_chol_jitter`, `N`)); `m_cache_jitter` is the one of the current cache (the search's while
optimizing). Observation noise adds to either.

| method | diagonal |
|---|---|
| fit / `update(refit = true)` objective, every start (the parallel clones too), `logLikelihoodFun` | 1 + `m_chol_jitter` |
| `logLikelihood()` | the optimized objective (saved as `objective_ll`); after `update(refit = false)`, the committed model's own |
| committed model (`commit_model_chol_jitter`), `predict`, `simulate`, `update(refit = false)` | 1 + min(`m_chol_jitter`, `N`): gauss 1e-10 |
| `update_simulate` | `R_uu` exact (1, or 1 + noise / sigma2) |
| `simulate` conditional covariance | 0 + retry (`simulate_impl`, as Kriging) |
| `load` of a file without `model_chol_jitter` | `m_chol_jitter` (one jitter for both, as saved) |

Not exact, unlike Kriging: an exact factorization lost the predictive variance where warps bring
training rows together in the feature space (Kumaraswamy, saturated tanh MLP: stdev 0, NaN stdev
derivative and NLPD in WarpKrigingPerWarpTest / WarpKrigingCollapseTest). Interpolation error on the
python-sdist data: warping `"none"` 1.0e-5 (5.6e-5 with `J` = 1e-9), Kumaraswamy 1.1e-6.

## NestedKriging

| step | diagonal |
|---|---|
| fit of the common prior | `Kriging` (or `WarpKriging` when warped) on the data: rules above |
| submodels (`optim = "none"`) | their committed model: Kriging 1 (exact), WarpKriging 1 + min(`J`, `N`) |
| `precompute_nk` group factors | the submodel's committed diagonal (`safe_chol_lower`) |
| `predict`, PoE family (PoE, gPoE, BCM, rBCM) | each submodel's `predict` |
| `predict`, NK aggregation | group factors as above; aggregation matrix `K_M` + `N` (`solve`, `pinv` fallback, no retry) |

No `simulate` / `update`.

## Inconsistencies left

1. **WarpKriging / MLPKriging** commit with `N` instead of exactly (see above; a deliberate
   difference, decided 2026-10-23), and their `update_simulate` is exact while `update` keeps `N` (a 1e-10 relative
   difference).
2. **`simulate` conditional covariance** (Kriging and WarpKriging, exact method) still relies on the
   stepwise retry: at points close to the design (or close to each other) the covariance is singular
   and the added nugget, `N`·10^k, could depend on which Cholesky fails, so on rounding. Measured
   (2026-10-22, `bench/thread-budget/sim_retry_probe.cpp`, 5 configurations x 10 data sets, budgets
   10 / 100 / 200 %, 50 simulations at 200 random points, optionally + design points / a 1e-7
   cluster): the retry fires in nearly every gauss case (the conditional covariance of 200 smooth
   points is numerically singular) and in matern5_2 only with design points or clusters, always
   at the first level (`N`), never at a different level across budgets; simulations differ by at
   most 4e-11 relative across budgets. In practice the retry acts as a fixed `N` here. A fixed floor
   (as `simulateNystrom`'s eigenvalue floor) would remove the residual risk: rcond values of 2e-18,
   next to the 1e-18 threshold of the rcond check, were seen (gauss n = 10). Not changed.
3. **NestedKriging NK aggregation** adds `N` to `K_M` regardless of the kernel (no `J`, no retry, `pinv`
   fallback): fine for stability, noted for completeness.
