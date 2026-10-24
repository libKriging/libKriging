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
- **WarpKriging / MLPKriging: committed model with min(J, `num_nugget`)** (1e-10 for gauss, its
  former diagonal). An exact model lost the predictive variance where warps bring training rows
  together (Kumaraswamy, saturated tanh MLP: stdev 0, NaN derivative / NLPD in
  WarpKrigingPerWarpTest, WarpKrigingCollapseTest). Kept as a deliberate difference (decided
  2026-10-23); aligning would first need a floor on the predicted variance in `predict` (a
  separate change, which would also cover Kriging at the design points).
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

## Open items

- Calibration: gauss predictive sd is overconfident (sd / RMSE 0.25–0.43) with or without jitter;
  matern5_2 ≈ 1 (#425).
- A jitter growing with n (conditioning grows with n), not measured (#426).
- A fixed floor for the `simulate` conditional covariance instead of the retry (inconsistency 2,
  #427).
- A floor on the predicted variance, then an exact committed WarpKriging model (#428).
- Single-start robustness on the plateau (#418); bindings consistency checks (#419); getters
  `chol_jitter()` / `model_chol_jitter()` not exposed in the bindings (#424).

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
